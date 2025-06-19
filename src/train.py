import os
import time
import math
import argparse
import tempfile
import subprocess
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import torchvision.transforms.functional as F_t
import torchvision.ops as ops_nms

from tqdm import tqdm

# ----------------------------------------
# 1) Model Definition (ConvNeXtASPPFPNDetector)
# ----------------------------------------

import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# 1.1) Custom DWConv modules for each stage
class DWConv3x3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
    def forward(self, x):
        return self.dw(x)

class DWConv5x5(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
    def forward(self, x):
        return self.dw(x)

class DWConv3x3_7x7_Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dw7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.attn = nn.Conv2d(dim * 2, 2, kernel_size=1, bias=True)

    def forward(self, x):
        f3 = self.dw3(x)
        f7 = self.dw7(x)
        cat = torch.cat([f3, f7], dim=1)
        w = self.attn(cat)
        w = torch.softmax(w, dim=1)
        w3 = w[:, 0:1].expand(-1, x.size(1), -1, -1)
        w7 = w[:, 1:2].expand(-1, x.size(1), -1, -1)
        out = f3 * w3 + f7 * w7
        return out

# 1.2) Replace DWConv by duck‐typing (ConvNeXtBlock import 없이)
def modify_convnext_backbone(backbone):
    """
    backbone: convnext.features (nn.Sequential of 8 modules)
    Stage1 index = 1, Stage2 index = 3, Stage3 index = 5, Stage4 index = 7
    """
    # Stage1: 3x3
    stage1 = backbone[1]
    for block in stage1:
        if hasattr(block, "dwconv"):
            dim = block.dwconv.in_channels
            block.dwconv = DWConv3x3(dim)

    # Stage2: 5x5
    stage2 = backbone[3]
    for block in stage2:
        if hasattr(block, "dwconv"):
            dim = block.dwconv.in_channels
            block.dwconv = DWConv5x5(dim)

    # Stage3: 3x3 + 7x7_Attn
    stage3 = backbone[5]
    for block in stage3:
        if hasattr(block, "dwconv"):
            dim = block.dwconv.in_channels
            block.dwconv = DWConv3x3_7x7_Attn(dim)
    # Stage4 (index 7)는 원래 7×7 DWConv 유지

# 1.3) ASPP module (dilations = 1, 6, 12 + GroupNorm → ReLU)
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, norm_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.project = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)      # [B, out_channels, H, W]
        x2 = self.conv2(x)      # dilation=1
        x3 = self.conv3(x)      # dilation=6
        x4 = self.conv4(x)      # dilation=12
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)  # [B, out_channels*4, H, W]
        out = self.project(x_cat)  # [B, out_channels, H, W]
        out = self.gn(out)
        return self.relu(out)

# 1.4) Simple FPN (with GroupNorm → ReLU)
class SimpleFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, norm_groups=32):
        super().__init__()
        # 1×1 lateral convs + GN → ReLU
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels),
                nn.ReLU(inplace=True)
            )
            for in_c in in_channels_list
        ])
        # 3×3 smoothing convs + GN → ReLU
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(len(in_channels_list) - 1)
        ])

    def forward(self, c3, c4, p5_aspp, c6):
        # c3: [B, 96,  H/4,  W/4]
        # c4: [B,192,  H/8,  W/8]
        # p5_aspp: [B,256, H/16, W/16]
        # c6: [B,768, H/32, W/32]

        # P6: Stage4 출력 768→256
        p6 = self.lateral_convs[3](c6)

        # P5: ASPP(256→256)
        p5 = self.lateral_convs[2](p5_aspp)

        # P4: Stage2(192→256) + up(P5)
        p4 = self.lateral_convs[1](c4) \
            + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.smooth_convs[0](p4)

        # P3: Stage1(96→256) + up(P4)
        p3 = self.lateral_convs[0](c3) \
            + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth_convs[1](p3)

        return [p3, p4, p5, p6]

# 1.5) AFHead (Anchor-Free One-Stage Detection Head with GroupNorm)
class AFHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=10, norm_groups=32):
        super().__init__()
        # stem: Conv → GN → ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=norm_groups, num_channels=in_channels),
            nn.ReLU(inplace=True),
        )
        # Classification branch: Conv → GN → ReLU → Conv
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=norm_groups, num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)  # num_classes 채널
        )
        # Regression branch: Conv → GN → ReLU → Conv
        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=norm_groups, num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, kernel_size=1)  # 4개 채널: l,t,r,b
        )
        # Center-ness branch: Conv → GN → ReLU → Conv
        self.center_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=norm_groups, num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)  # 1개 채널: center-ness
        )

    def forward(self, x):
        f = self.stem(x)                # [B, in_channels, H, W]
        cls = self.cls_branch(f)        # [B, num_classes, H, W]
        reg = self.reg_branch(f)        # [B, 4,        H, W]
        ctr = self.center_branch(f)     # [B, 1,        H, W]
        return cls, reg, ctr

# 1.6) ConvNeXt + ASPP + FPN + AFHead 통합 모델
class ConvNeXtASPPFPNDetector(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 1) Pretrained ConvNeXt-Tiny backbone
        convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.backbone = convnext.features  # nn.Sequential of length 8

        # 2) DWConv 교체
        modify_convnext_backbone(self.backbone)

        # 3) ASPP (Stage3 출력 384→256)
        # ConvNeXt-Tiny Stage3 출력 채널은 384이므로, in_channels=384
        self.aspp = ASPP(in_channels=384, out_channels=256, norm_groups=32)

        # 4) FPN: 입력 채널 리스트 = [96, 192, 256, 768]
        self.fpn = SimpleFPN(in_channels_list=[96, 192, 256, 768], out_channels=256, norm_groups=32)

        # 5) AFHead: P3, P4, P5, P6 각각에 Head 연결
        self.heads = nn.ModuleList([
            AFHead(in_channels=256, num_classes=num_classes, norm_groups=32)
            for _ in range(4)
        ])

    def forward(self, x):
        # Backbone feature extraction
        x0 = self.backbone[0](x)    # PatchEmbed → [B, 96,  H/4,  W/4]
        c3 = self.backbone[1](x0)   # Stage1    → [B, 96,  H/4,  W/4]

        x1 = self.backbone[2](c3)   # Down1     → [B,192,  H/8,  W/8]
        c4 = self.backbone[3](x1)   # Stage2    → [B,192,  H/8,  W/8]

        x2 = self.backbone[4](c4)   # Down2     → [B,384,  H/16, W/16]
        s3 = self.backbone[5](x2)   # Stage3    → [B,384,  H/16, W/16]

        # ASPP on Stage3 output → P5
        p5 = self.aspp(s3)          # [B,256, H/16, W/16]

        x3 = self.backbone[6](s3)   # Down3     → [B,768,  H/32, W/32]
        c6 = self.backbone[7](x3)   # Stage4    → [B,768,  H/32, W/32]

        # FPN: combine c3, c4, p5, c6 → [P3, P4, P5, P6]
        features = self.fpn(c3, c4, p5, c6)

        # AFHead: 각 레벨별로 cls, reg, ctr 예측
        cls_preds, reg_preds, ctr_preds = [], [], []
        for idx, feat in enumerate(features):
            cls_i, reg_i, ctr_i = self.heads[idx](feat)
            cls_preds.append(cls_i)
            reg_preds.append(reg_i)
            ctr_preds.append(ctr_i)

        return cls_preds, reg_preds, ctr_preds

# ----------------------------------------
# 2) Dataset and Augmentation (YOLO-style labels)
# ----------------------------------------

class BDDYOLODataset(data.Dataset):
    def __init__(self, root, split, transforms=None):
        """
        YOLO 스타일의 txt 라벨을 사용하는 BDD100K 데이터셋 로더.
        root: BDD100K 루트 디렉터리 (images/, labels/ 폴더 포함)
        split: 'train', 'val' 또는 'test'
        transforms: Albumentations Compose 객체
        """
        super().__init__()
        self.img_dir   = os.path.join(root, "images", split)
        self.label_dir = os.path.join(root, "labels", split)
        self.img_paths = sorted([
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        image_np = np.array(img)  # [H, W, 3], np.uint8

        # 대응하는 라벨(.txt)
        base_name  = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base_name + ".txt")

        boxes = []
        labels = []
        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    bw       = float(parts[3]) * w
                    bh       = float(parts[4]) * h
                    x_min = x_center - bw / 2.0
                    y_min = y_center - bh / 2.0
                    x_max = x_center + bw / 2.0
                    y_max = y_center + bh / 2.0
                    # Clamp coordinates
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)
        boxes  = torch.as_tensor(boxes, dtype=torch.float32)  # [N, 4]
        labels = torch.as_tensor(labels, dtype=torch.int64)  # [N]

        # --- 추가: 넓이가 0이거나 잘못된 박스 필터링 ---
        if boxes.numel() > 0:
            valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid_mask]
            labels = labels[valid_mask]

        # *** 변경: boxes가 비어 있어도 Normalize + ToTensor 적용 ***
        if self.transforms is not None:
            # Albumentations는 bboxes가 비어 있어도 image만 변환해 줍니다.
            transformed = self.transforms(
                image=image_np,
                bboxes=boxes.tolist(),
                class_labels=labels.tolist()
            )
            image  = transformed["image"]
            boxes  = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.as_tensor(transformed["class_labels"], dtype=torch.int64)
        else:
            image = F_t.to_tensor(img)  # [3, H, W] (값 [0,1])

        target = {
            "boxes":    boxes,            # [N, 4] (x_min, y_min, x_max, y_max)
            "labels":   labels,           # [N]
            "image_id": torch.tensor([idx])
        }
        return image, target

def get_train_transforms(epoch, base_size=512):
    # 에폭 구간 상관없이 동일한 최소 변환만 적용
    return A.Compose([
        A.Resize(height=base_size, width=int(base_size * 16/9)),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc',
                               label_fields=['class_labels']))


def get_val_transforms(base_size=512):
    return A.Compose([
        A.Resize(height=base_size, width=int(base_size * 16/9)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def collate_fn(batch):
    return tuple(zip(*batch))

# ----------------------------------------
# 3) Loss Functions (with ignore for background)
# ----------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification, with an optional ignore_index for background.
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: [C, H, W] raw logits
        targets: [H, W] with values in {0,1,...,C-1} or ignore_index for background
        """
        C, H, W = inputs.shape
        N = H * W
        inputs = inputs.permute(1, 2, 0).contiguous().view(N, C)  # [N, C]
        targets = targets.view(N)  # [N]

        # Create a mask for valid positions (where target != ignore_index)
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device)

        inputs_valid = inputs[valid_mask]         # [M, C]
        targets_valid = targets[valid_mask]       # [M]

        # One-hot encode targets
        t = torch.zeros_like(inputs_valid)
        t.scatter_(1, targets_valid.unsqueeze(1), 1.0)  # [M, C]

        p = torch.softmax(inputs_valid, dim=1)   # [M, C]
        pt = (p * t).sum(dim=1)                  # [M]

        eps = 1e-8
        loss = -self.alpha * ((1 - pt) ** self.gamma) * torch.log(pt + eps)  # [M]

        return loss.mean()

# ----------------------------------------
# 4) Target Assignment (AFHead 출력 → GT 매핑)
# ----------------------------------------

def assign_targets_to_afhead(gt_boxes, gt_labels, feature_shapes, strides, num_classes):
    """
    AFHead target assignment (하나의 이미지 기준).

    Args:
      gt_boxes: Tensor[N_gt, 4], GT 박스 (x_min, y_min, x_max, y_max) in original pixel coords
      gt_labels: Tensor[N_gt], GT 클래스 IDs (0~num_classes-1)
      feature_shapes: List of (H_i, W_i) for each FPN level
      strides:         List of strides for each level (예: [4,8,16,32])
      num_classes:     Number of object classes (background index = num_classes)
    Returns:
      cls_targets:   List of Tensor[H_i, W_i] with class indices or background=num_classes
      reg_targets:   List of Tensor[4, H_i, W_i] with (l, t, r, b) distances
      ctr_targets:   List of Tensor[1, H_i, W_i] with center-ness
      valid_mask:    List of Tensor[1, H_i, W_i], 1 if pixel is assigned to an object, else 0
    """
    cls_targets = []
    reg_targets = []
    ctr_targets = []
    valid_mask  = []

    if gt_boxes.numel() == 0:
        # GT 박스가 없으면, 모든 위치를 background 처리
        for (H, W) in feature_shapes:
            cls_targets.append(torch.full((H, W), num_classes, dtype=torch.int64, device=gt_boxes.device))
            reg_targets.append(torch.zeros((4, H, W), dtype=torch.float32, device=gt_boxes.device))
            ctr_targets.append(torch.zeros((1, H, W), dtype=torch.float32, device=gt_boxes.device))
            valid_mask.append(torch.zeros((1, H, W), dtype=torch.uint8, device=gt_boxes.device))
        return cls_targets, reg_targets, ctr_targets, valid_mask

    N_gt = gt_boxes.shape[0]
    # 각 GT 박스의 면적
    areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])).view(-1, 1, 1)  # [N_gt,1,1]

    for (H, W), stride in zip(feature_shapes, strides):
        # 1) 각 feature map 위치(i,j)의 원본 이미지 좌표 계산
        ys = (torch.arange(0, H, device=gt_boxes.device).float() + 0.5) * stride  # [H]
        xs = (torch.arange(0, W, device=gt_boxes.device).float() + 0.5) * stride  # [W]
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")                    # [H, W], [H, W]
        xg = x_grid.unsqueeze(0)  # [1, H, W]
        yg = y_grid.unsqueeze(0)  # [1, H, W]

        # 2) 모든 GT 박스와 비교하여 위치가 박스 내부에 있는지 확인
        x_min = gt_boxes[:, 0].view(-1, 1, 1)  # [N_gt, 1, 1]
        y_min = gt_boxes[:, 1].view(-1, 1, 1)
        x_max = gt_boxes[:, 2].view(-1, 1, 1)
        y_max = gt_boxes[:, 3].view(-1, 1, 1)

        inside_x = (xg >= x_min) & (xg <= x_max)  # [N_gt, H, W]
        inside_y = (yg >= y_min) & (yg <= y_max)  # [N_gt, H, W]
        inside_box = inside_x & inside_y         # [N_gt, H, W]

        # 3) 위치별로 “가장 작은 면적의 GT 박스”에 할당
        inside_any = inside_box.any(dim=0)  # [H, W]
        if not inside_any.any():
            cls_targets.append(torch.full((H, W), num_classes, dtype=torch.int64, device=gt_boxes.device))
            reg_targets.append(torch.zeros((4, H, W), dtype=torch.float32, device=gt_boxes.device))
            ctr_targets.append(torch.zeros((1, H, W), dtype=torch.float32, device=gt_boxes.device))
            valid_mask.append(torch.zeros((1, H, W), dtype=torch.uint8, device=gt_boxes.device))
            continue

        very_large = torch.full_like(areas, 1e6)  # [N_gt,1,1]
        areas_for = torch.where(inside_box, areas, very_large)  # [N_gt, H, W]
        min_idx = torch.argmin(areas_for, dim=0)  # [H, W], 값 범위 0~N_gt-1

        assigned_boxes = gt_boxes[min_idx.view(-1)].view(H, W, 4)   # [H, W, 4]
        assigned_labels = gt_labels[min_idx.view(-1)].view(H, W)   # [H, W]

        # --- 수정된 분류 타겟 (background 처리) ---
        cls_t = torch.full((H, W), num_classes, dtype=torch.int64, device=gt_boxes.device)
        cls_t[inside_any] = assigned_labels[inside_any]  # inside_any True인 위치만 실제 클래스 할당

        # 4) regression target (l, t, r, b) 계산
        l = xg[0] - assigned_boxes[:, :, 0]  # [H, W]
        t = yg[0] - assigned_boxes[:, :, 1]
        r = assigned_boxes[:, :, 2] - xg[0]
        b = assigned_boxes[:, :, 3] - yg[0]
        reg_t = torch.stack([l, t, r, b], dim=0)  # [4, H, W]

        # 5) center-ness target 계산 (FCOS 방식 예시)
        lt_min = torch.min(l, r)
        lt_max = torch.max(l, r)
        tb_min = torch.min(t, b)
        tb_max = torch.max(t, b)
        eps = 1e-8
        ctr = torch.sqrt((lt_min / (lt_max + eps)) * (tb_min / (tb_max + eps)))  # [H, W]
        ctr_t = ctr.unsqueeze(0)  # [1, H, W]

        # 6) valid mask: inside_any이 True인 위치만 object, 나머지는 background
        mask = inside_any.unsqueeze(0).to(torch.uint8)  # [1, H, W]

        cls_targets.append(cls_t)
        reg_targets.append(reg_t)
        ctr_targets.append(ctr_t)
        valid_mask.append(mask)

    return cls_targets, reg_targets, ctr_targets, valid_mask


# ----------------------------------------
# 5) Utility Functions for Freeze/Unfreeze
# ----------------------------------------

def set_backbone_requires_grad(model, epoch):
    """
    epoch 구간에 따라 backbone의 특정 stage만 unfreeze 하고 나머지는 freeze 합니다.
    - epoch 1~10: 전체 freeze
    - epoch 11~15: Stage4 unfreeze
    - epoch 16~20: Stage3 + Stage4 unfreeze
    - epoch >20: 전체 unfreeze
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

    if epoch <= 10:
        return
    elif epoch <= 15:
        for name, module in model.backbone.named_children():
            if name == "7":  # Stage4
                for param in module.parameters():
                    param.requires_grad = True
    elif epoch <= 20:
        for name, module in model.backbone.named_children():
            if name in ["5", "7"]:
                for param in module.parameters():
                    param.requires_grad = True
    else:
        for param in model.backbone.parameters():
            param.requires_grad = True

# ----------------------------------------
# 6) Decoding Predictions & NMS Helpers
# ----------------------------------------

def decode_predictions(cls_preds, reg_preds, ctr_preds, strides, score_threshold=0.05, nms_iou_threshold=0.5, top_k=1000):
    """
    AFHead 출력물을 (cls_logits, reg, ctr) → 최종 박스, 클래스, 점수 리스트로 디코딩
    Args:
      cls_preds: List of Tensor, 각 레벨마다 [B, num_classes, H_i, W_i]
      reg_preds: List of Tensor, 각 레벨마다 [B, 4,         H_i, W_i]
      ctr_preds: List of Tensor, 각 레벨마다 [B, 1,         H_i, W_i]
      strides:   List of int, 각 레벨의 stride (예: [4,8,16,32])
      score_threshold: float, 예측 점수 필터 임계값
      nms_iou_threshold: float, NMS IoU 임계값
      top_k: int, NMS 전에 각 레벨에서 상위 top_k만 유지
    Returns:
      final_boxes: List[Tensor] of length B, 각 이미지마다 [N_det, 4] (x_min, y_min, x_max, y_max)
      final_scores: List[Tensor] of length B, 각 이미지마다 [N_det]
      final_labels: List[Tensor] of 길이 B, 각 이미지마다 [N_det] (0~num_classes-1)
    """
    B = int(cls_preds[0].shape[0])
    num_levels = len(cls_preds)
    final_boxes = [[] for _ in range(B)]
    final_scores = [[] for _ in range(B)]
    final_labels = [[] for _ in range(B)]

    for lvl in range(num_levels):
        cls_l = cls_preds[lvl]  # [B, C, H, W]
        reg_l = reg_preds[lvl]  # [B, 4, H, W]
        ctr_l = ctr_preds[lvl]  # [B, 1, H, W]
        stride = strides[lvl]
        _, C, H, W = cls_l.shape

        # (1) 위치 그리드 계산
        ys = (torch.arange(0, H, device=cls_l.device).float() + 0.5) * stride  # [H]
        xs = (torch.arange(0, W, device=cls_l.device).float() + 0.5) * stride  # [W]
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        xg = x_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        yg = y_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        for b in range(B):
            # (2) 예측값 가져오기
            cls_logits = cls_l[b]      # [C, H, W]
            reg_vals = reg_l[b]        # [4, H, W]
            ctr_vals = ctr_l[b]        # [1, H, W]

            # (3) 확률값 계산: softmax 후 class별 confidence
            cls_prob = torch.softmax(cls_logits.view(C, -1).transpose(0, 1), dim=1)  # [H*W, C]
            cls_prob = cls_prob.view(H, W, C).permute(2, 0, 1)                       # [C, H, W]

            # (4) center-ness score: sigmoid
            ctr_prob = torch.sigmoid(ctr_vals).squeeze(0)  # [H, W]

            # (5) 각 위치마다 “최고 클래스 값”과 “클래스 인덱스” 추출
            cls_conf, cls_label = torch.max(cls_prob, dim=0)  # [H, W], [H, W]

            # (6) 최종 score = cls_conf * ctr_prob
            final_conf_map = cls_conf * ctr_prob  # [H, W]

            # (7) score_threshold로 필터링
            keep_mask = final_conf_map > score_threshold  # [H, W]
            if keep_mask.sum() == 0:
                continue

            # (8) 이 위치들의 좌표, 클래스, score, reg 값 추출
            ys_keep = yg.squeeze(0).squeeze(0)[keep_mask]  # [M]
            xs_keep = xg.squeeze(0).squeeze(0)[keep_mask]  # [M]
            scores_keep = final_conf_map[keep_mask]        # [M]
            labels_keep = cls_label[keep_mask]             # [M]

            reg_keep = reg_vals[:, keep_mask]  # [4, M]
            # l, t, r, b 순서
            l = reg_keep[0]
            t = reg_keep[1]
            r = reg_keep[2]
            b_val = reg_keep[3]

            # (9) 박스 좌표 계산
            x_min = xs_keep - l
            y_min = ys_keep - t
            x_max = xs_keep + r
            y_max = ys_keep + b_val

            boxes_lvl = torch.stack([x_min, y_min, x_max, y_max], dim=1)  # [M, 4]

            # (10) NMS 적용 (각 레벨별 top_k 먼저 추려낸 후)
            if boxes_lvl.numel() == 0:
                continue
            # top_k 필터링
            if scores_keep.numel() > top_k:
                topk_vals, topk_idxs = torch.topk(scores_keep, top_k)
                boxes_lvl = boxes_lvl[topk_idxs]
                scores_keep = scores_keep[topk_idxs]
                labels_keep = labels_keep[topk_idxs]

            keep_nms = ops_nms.nms(boxes_lvl, scores_keep, nms_iou_threshold)  # indices
            boxes_nms = boxes_lvl[keep_nms]
            scores_nms = scores_keep[keep_nms]
            labels_nms = labels_keep[keep_nms]

            final_boxes[b].append(boxes_nms)
            final_scores[b].append(scores_nms)
            final_labels[b].append(labels_nms)

    # 각 이미지마다 리스트가 여러 레벨로 나뉘어 있으므로 concat
    final_boxes_cat = []
    final_scores_cat = []
    final_labels_cat = []
    for b in range(B):
        if len(final_boxes[b]) == 0:
            final_boxes_cat.append(torch.zeros((0, 4), device=cls_preds[0].device))
            final_scores_cat.append(torch.zeros((0,), device=cls_preds[0].device))
            final_labels_cat.append(torch.zeros((0,), dtype=torch.int64, device=cls_preds[0].device))
        else:
            final_boxes_cat.append(torch.cat(final_boxes[b], dim=0))   # [N_det, 4]
            final_scores_cat.append(torch.cat(final_scores[b], dim=0)) # [N_det]
            final_labels_cat.append(torch.cat(final_labels[b], dim=0)) # [N_det]
    return final_boxes_cat, final_scores_cat, final_labels_cat

# ----------------------------------------
# 7) Evaluation
# ----------------------------------------

def evaluate(model, val_loader, device):
    """
    Validation: torchmetrics의 MeanAveragePrecision()을 사용하여 mAP@0.5 계산
    Args:
      model: 학습 중인 모델 인스턴스
      val_loader: 검증 데이터 로더
      device: 'cuda:0' 또는 'cpu'
    Returns:
      {"mAP": float} 형태의 딕셔너리 (mAP@0.50 값)
    """
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(device)

    with torch.no_grad():
        for images, targets in val_loader:
            # 1) 데이터를 GPU(device)로 이동
            imgs = torch.stack([img.to(device) for img in images])  # [B, 3, H, W]
            # 2) 모델 추론
            cls_preds, reg_preds, ctr_preds = model(imgs)

            # 3) 예측 결과를 decode하여 (boxes, scores, labels) 형태로 변환
            #    - decode_predictions() 함수는 이미 train.py에 정의되어 있다고 가정합니다.
            final_boxes, final_scores, final_labels = decode_predictions(
                cls_preds, 
                reg_preds, 
                ctr_preds,
                strides=[4, 8, 16, 32],       # train.py에서 정의한 strides 값과 일치시킵니다.
                score_threshold=0.05,         # 필요 시 threshold 조절
                nms_iou_threshold=0.5         # 필요 시 NMS IoU threshold 조절
            )
            # final_boxes: List[Tensor], each tensor shape = [N_i, 4]
            # final_scores: List[Tensor], each tensor shape = [N_i]
            # final_labels: List[Tensor], each tensor shape = [N_i]

            # 4) GT 박스/라벨을 device로 이동
            gt_boxes = [t["boxes"].to(device) for t in targets]
            gt_labels = [t["labels"].to(device) for t in targets]

            # 5) torchmetrics에 맞게 포맷 맞추기
            preds_for_metric = []
            targets_for_metric = []
            B = imgs.shape[0]
            for b_idx in range(B):
                preds_for_metric.append({
                    "boxes": final_boxes[b_idx].to(device),
                    "scores": final_scores[b_idx].to(device),
                    "labels": final_labels[b_idx].to(device),
                })
                targets_for_metric.append({
                    "boxes": gt_boxes[b_idx],
                    "labels": gt_labels[b_idx],
                })

            # 6) metric 업데이트
            metric.update(preds_for_metric, targets_for_metric)

    # 7) 전체 배치(데이터셋)에 대한 mAP 계산
    results = metric.compute()  
    # torchmetrics의 반환값: dict with keys like "map", "map_50", "map_75", ...
    map50 = results["map_50"].item()  # mAP@0.50 값

    return {"mAP": map50}

# ----------------------------------------
# 8) Training and Validation Loops
# ----------------------------------------

def train_one_epoch(model, optimizer, scaler, data_loader, device, epoch, accumulation_steps, args, scheduler=None):
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    loop = tqdm(data_loader, desc=f"Epoch[{epoch}]", leave=False)
    for step, (images, targets) in enumerate(loop):
        B = len(images)
        imgs = torch.stack([img.to(device) for img in images])  # [B, 3, H_img, W_img]
        _, _, H_img, W_img = imgs.shape

        # **feature_shapes를 동적으로 계산**
        feature_shapes = [
            (H_img // 4,  W_img // 4),
            (H_img // 8,  W_img // 8),
            (H_img // 16, W_img // 16),
            (H_img // 32, W_img // 32),
        ]
        strides = [4, 8, 16, 32]

        gt_boxes_batch  = [t["boxes"].to(device)  for t in targets]
        gt_labels_batch = [t["labels"].to(device) for t in targets]

        with autocast():
            cls_preds, reg_preds, ctr_preds = model(imgs)
            total_loss = torch.tensor(0.0, device=device)

            # 배치 내 이미지마다 Loss 계산
            for i in range(B):
                gt_boxes  = gt_boxes_batch[i]
                gt_labels = gt_labels_batch[i]

                # AFHead Target Assignment
                cls_targets, reg_targets, ctr_targets, valid_mask = assign_targets_to_afhead(
                    gt_boxes,
                    gt_labels,
                    feature_shapes,
                    strides,
                    num_classes=model.heads[0].cls_branch[-1].out_channels
                )

                # 각 레벨마다 Prediction vs GT 비교
                for lvl in range(4):
                    pred_cls = cls_preds[lvl][i]  # [C, H_i, W_i]
                    pred_reg = reg_preds[lvl][i]  # [4, H_i, W_i]
                    pred_ctr = ctr_preds[lvl][i]  # [1, H_i, W_i]

                    cls_t = cls_targets[lvl].to(device)      # [H_i, W_i]
                    reg_t = reg_targets[lvl].to(device)      # [4, H_i, W_i]
                    ctr_t = ctr_targets[lvl].to(device)      # [1, H_i, W_i]
                    mask  = valid_mask[lvl].to(device)       # [1, H_i, W_i]

                    # 1) Classification Loss (FocalLoss, background은 ignore_index)
                    fl = FocalLoss(ignore_index=model.heads[0].cls_branch[-1].out_channels)
                    loss_cls = fl(pred_cls, cls_t)

                    # 2) Regression L1 Loss & Center-ness Loss (positive 위치만)
                    mask_flat = mask.view(-1).bool()  # [H_i*W_i]
                    if mask_flat.sum() > 0:
                        pred_reg_flat = pred_reg.view(4, -1).transpose(0, 1)[mask_flat]  # [M, 4]
                        reg_t_flat     = reg_t.view(4, -1).transpose(0, 1)[mask_flat]     # [M, 4]
                        loss_reg = F.l1_loss(pred_reg_flat, reg_t_flat, reduction="mean")

                        pred_ctr_flat = pred_ctr.view(-1)[mask_flat]  # [M]
                        ctr_t_flat    = ctr_t.view(-1)[mask_flat]    # [M]
                        loss_ctr = F.binary_cross_entropy_with_logits(pred_ctr_flat, ctr_t_flat)
                    else:
                        loss_reg = torch.tensor(0.0, device=device)
                        loss_ctr = torch.tensor(0.0, device=device)

                    total_loss += loss_cls + loss_reg + loss_ctr

            # 배치 평균으로 나누기
            total_loss = total_loss / B

        scaler.scale(total_loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        running_loss += total_loss.item() * B
        total_samples += B

    epoch_loss = running_loss / total_samples
    return epoch_loss

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = args.num_classes

    # 6.1) Model Initialization
    model = ConvNeXtASPPFPNDetector(num_classes=num_classes)
    model.to(device)

    # 6.2) Dataset & DataLoader
    print("Loading datasets...")
    train_dataset = BDDYOLODataset(
        root=args.data_path,
        split="train",
        transforms=None  # epoch loop에서 설정
    )
    val_dataset = BDDYOLODataset(
        root=args.data_path,
        split="val",
        transforms=get_val_transforms(base_size=args.base_size)
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 6.3) Optimizer & Learning Rate Scheduler
    def get_param_groups(model, epoch):
        head_params = []
        aspp_fpn_params = []
        backbone_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "heads" in name:
                head_params.append(param)
            elif "aspp" in name or "fpn" in name:
                aspp_fpn_params.append(param)
            else:
                backbone_params.append(param)

        if epoch <= 10:
            lr_backbone = 0.0
            lr_head = 1e-3
            lr_aspp_fpn = 1e-3
        elif epoch <= 20:
            lr_backbone = 1e-5
            lr_head = 1e-4
            lr_aspp_fpn = 1e-4
        elif epoch <= 30:
            lr_backbone = 5e-6
            lr_head = 5e-5
            lr_aspp_fpn = 5e-5
        else:
            lr_backbone = 1e-6
            lr_head = 1e-5
            lr_aspp_fpn = 1e-5

        return [
            {"params": head_params,      "lr": lr_head,      "weight_decay": args.weight_decay},
            {"params": aspp_fpn_params,  "lr": lr_aspp_fpn,  "weight_decay": args.weight_decay},
            {"params": backbone_params,  "lr": lr_backbone,  "weight_decay": args.weight_decay},
        ]

    optimizer = optim.AdamW(get_param_groups(model, epoch=1))
    total_steps = math.ceil(len(train_loader.dataset) / args.batch_size / args.accumulation_steps) * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    # 6.4) Mixed Precision 설정
    scaler = GradScaler()

    # 6.5) Early Stopping & Checkpoint
    best_mAP = 0.0
    epochs_no_improve = 0
    os.makedirs(args.output_dir, exist_ok=True)

    # 6.6) Training Loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        # 1) Freeze/Unfreeze backbone
        set_backbone_requires_grad(model, epoch)
        # 2) Update optimizer param groups (lr 조정)
        new_groups = get_param_groups(model, epoch)
        for param_group, new_group in zip(optimizer.param_groups, new_groups):
             param_group["lr"] = new_group["lr"]
             param_group["weight_decay"] = new_group["weight_decay"]

        # 3) Update train transforms for this epoch
        train_dataset.transforms = get_train_transforms(epoch=None)

        # 4) Train for one epoch
        train_loss = train_one_epoch(
            model, optimizer, scaler, train_loader, device,
            epoch, args.accumulation_steps, args, scheduler=scheduler
        )
        print(f"Epoch [{epoch}] Training Loss: {train_loss:.4f}")

        # 5) Validation mAP 계산
        metrics = evaluate(model, val_loader, device)
        val_mAP = metrics["mAP"]
        print(f"Epoch [{epoch}] Validation mAP@0.5: {val_mAP:.4f}")

        # 6) Check for improvement
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            epochs_no_improve = 0
            ckpt_path = os.path.join(
                args.output_dir,
                f"best_epoch_{epoch:02d}_mAP_{val_mAP:.3f}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mAP": best_mAP,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
        else:
            epochs_no_improve += 1

        # 7) Early Stopping
        if epochs_no_improve >= args.early_stop_patience:
            print(f"No improvement for {args.early_stop_patience} epochs. Early stopping.")
            break

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch}] took {elapsed:.1f}s\n")

    print("Training complete.")
    print(f"Best val mAP@0.5: {best_mAP:.4f}")

# ----------------------------------------
# 9) Argument Parser
# ----------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvNeXt+ASPP+FPN+AFHead on BDD100K (YOLO labels)")
    parser.add_argument("--data-path", type=str,
                        default=r"C:\Users\user\Desktop\graduate\bdd100k",
                        help="BDD100K dataset root directory (contains images/ and labels/ folders)")
    parser.add_argument("--output-dir", type=str,
                        default=r"C:\Users\user\Desktop\graduate\checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for training (e.g., 'cuda:0')")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs to run")
    parser.add_argument("--batch-size", type=int, default=14, help="Batch size per iteration")
    parser.add_argument("--accumulation-steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--base-size", type=int, default=512, help="Base input size (shorter side)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of object classes")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Number of epochs with no improvement before stopping")
    args = parser.parse_args()
    main(args)
