# dyolo/train.py
import os
import cv2
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
from contextlib import contextmanager
from pathlib import Path

# Ensure legacy torch.load behaviour (weights_only=False) for PyTorch 2.6+
os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")

from torch.serialization import add_safe_globals
from torch.nn import Sequential, Conv2d, BatchNorm2d
from ultralytics.nn.tasks import DetectionModel as U_DetectionModel
from ultralytics.nn.modules.conv import Conv
add_safe_globals([U_DetectionModel, Conv, Sequential, Conv2d, BatchNorm2d])

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator

from dyolo_core import DYOLOCore
from losses import ChannelWiseDistillationLoss

# COCO class names used to align pretrained classifier weights with the 5-class subset.
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

def _save_core_when_best(trainer):
    """Called on Ultralytics 'on_model_save'. Save D-YOLO core when best.pt updated."""
    wdir = Path(trainer.save_dir) / "weights"
    best_pt = wdir / "best.pt"
    if not best_pt.exists():
        return
    # save only when best.pt is freshly updated
    mtime = best_pt.stat().st_mtime
    last = getattr(trainer, "_core_saved_mtime", None)
    if last is None or mtime > last:
        torch.save(trainer.core.state_dict(), wdir / "best_core.pth")
        setattr(trainer, "_core_saved_mtime", mtime)
        print(f"[CALLBACK] Saved D-YOLO core ??{wdir/'best_core.pth'} (epoch={trainer.epoch})")

def letterbox_np(img, new_shape, color=(114, 114, 114), scaleup=True, return_meta=False):
    """Resize with aspect ratio, Ultralytics-style padding."""
    h0, w0 = img.shape[:2]
    H, W = new_shape
    r = min(H / h0, W / w0)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = W - new_unpad[0]
    dh = H - new_unpad[1]
    dw /= 2
    dh /= 2
    left, right = int(np.floor(dw)), int(np.ceil(dw))
    top, bottom = int(np.floor(dh)), int(np.ceil(dh))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    if return_meta:
        return img, r, (left, top)
    return img


class YOLOv8BackboneAdapter(nn.Module):
    def __init__(self, det_model):
        super().__init__()
        self.det_model = det_model
        self.layers = det_model.model
        self.detect = self.layers[-1]
        self.from_idx = sorted(list(getattr(self.detect, "f", [15, 18, 21])))
        self._feats = {}

    def _make_hook(self, idx):
        def _hook(module, inp, out):
            self._feats[idx] = out
        return _hook

    @contextmanager
    def _stub_detect(self):
        det = self.layers[-1]
        orig_forward = det.forward
        def passthrough(x, *args, **kwargs):
            return x
        det.forward = passthrough
        try:
            yield
        finally:
            det.forward = orig_forward

    def forward(self, x):
        self._feats.clear()
        hooks = []
        for idx in self.from_idx:
            h = self.layers[idx].register_forward_hook(self._make_hook(idx))
            hooks.append(h)
        try:
            with self._stub_detect():
                _ = self.det_model(x)
        finally:
            for h in hooks:
                h.remove()

        outs = [self._feats[idx] for idx in self.from_idx]
        if len(outs) != 3:
            raise RuntimeError(f"Expected 3 maps, got {len(outs)} from {self.from_idx}.")
        return outs[0], outs[1], outs[2]


class DYoloDetectWrapper(nn.Module):
    """
    Wrap Detect so FA/AFM runs before Detect, and forward attribute writes to Detect.
    """
    def __init__(self, core: DYOLOCore, detect_module: nn.Module):
        super().__init__()
        object.__setattr__(self, "_core_ref", core)  # avoid submodule registration
        self.detect = detect_module
        self.f = getattr(detect_module, "f", None)
        self._core_dev = None

    @property
    def core(self) -> DYOLOCore:
        return object.__getattribute__(self, "_core_ref")

    def __getattr__(self, name):
        if name == "core":
            return object.__getattribute__(self, "_core_ref")
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.detect, name)

    def __setattr__(self, name, value):
        # forward writes like nc/no/stride/names to inner Detect
        if name in {"_modules", "_parameters", "_buffers"} or name in {"_core_ref", "detect", "f", "_core_dev"}:
            return super().__setattr__(name, value)
        if "detect" in self.__dict__ and hasattr(self.detect, name):
            return setattr(self.detect, name, value)
        return super().__setattr__(name, value)

    def _ensure_core_device(self, ref):
        dev = ref.device
        if self._core_dev != dev:
            mods = [
                self.core.fa3,
                self.core.fa4,
                self.core.fa5,
                self.core.afm3,
                self.core.afm4,
                self.core.afm5,
            ]
            extra_attrs = ["cfe", "cfe_proj3", "cfe_proj4", "cfe_proj5"]
            for attr in extra_attrs:
                if hasattr(self.core, attr):
                    mods.append(getattr(self.core, attr))
            for m in mods:
                m.to(dev)
            self._core_dev = dev

    def forward(self, x, *args, **kwargs):
        if os.environ.get("DY_BYPASS_CORE", "0") == "1":
            return self.detect(x, *args, **kwargs)
        if self.training:
            self.core.train()
        else:
            self.core.eval()
        if isinstance(x, (list, tuple)) and len(x) == 3:
            H3, H4, H5 = x
            self._ensure_core_device(H3)
            D3 = self.core.fa3(H3)
            D4 = self.core.fa4(H4)
            D5 = self.core.fa5(H5)
            F3 = self.core.afm3(H3, D3)
            F4 = self.core.afm4(H4, D4)
            F5 = self.core.afm5(H5, D5)        
            out = self.detect([F3, F4, F5], *args, **kwargs)
            if isinstance(out, (list, tuple)) and len(out) == 3 and not getattr(self, "_debug_shape_printed", False):
                shapes = [tuple(o.shape) for o in out]
                print(f"[DYoloDetectWrapper] detect outputs shapes={shapes}, no={self.detect.no}")
                object.__setattr__(self, "_debug_shape_printed", True)
            return out
        return self.detect(x, *args, **kwargs)


class DYoloTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        m = super().get_model(cfg, weights, verbose)
        self.det_model = m

        # Student backbone adapter
        self.student_bb = YOLOv8BackboneAdapter(self.det_model)

        model_device = next(self.det_model.model.parameters()).device
        if hasattr(self, "target_imgsz"):
            img_h, img_w = self.target_imgsz  # e.g., (448, 640)
        else:
            imgsz = self.args.imgsz
            if isinstance(imgsz, (tuple, list)):
                img_h, img_w = (imgsz[0], imgsz[1]) if len(imgsz) >= 2 else (imgsz[0], imgsz[0])
            else:
                img_h = img_w = imgsz

        dummy = torch.zeros(1, 3, img_h, img_w, device=model_device)
        with torch.no_grad():
            H3, H4, H5 = self.student_bb(dummy)
        chs = (H3.shape[1], H4.shape[1], H5.shape[1])

        # D-YOLO Core
        self.core = DYOLOCore(student_backbone=self.student_bb, chs=chs, use_cfe=True)

        # Wrap Detect
        detect_module = self.det_model.model[-1]
        if not isinstance(detect_module, DYoloDetectWrapper):
            wrapped_detect = DYoloDetectWrapper(self.core, detect_module)
            self.det_model.model[-1] = wrapped_detect
        else:
            wrapped_detect = detect_module
        self.base_detect = wrapped_detect.detect  # raw Detect


# ======================================================================
        # [신규] 80-Class COCO 모델을 가중치 소스로 임시 로드
        # ======================================================================
        model_device = next(self.det_model.model.parameters()).device
        print("[HEAD FIX] Loading temporary 80-class COCO model to source head weights...")
        try:
            # cfg가 파일 경로일 수 있으므로, cfg가 None이 아니면 사용하고, 아니면 model.yaml 사용
            cfg_path = cfg if cfg is not None else self.det_model.yaml_file
            coco_model = U_DetectionModel(cfg_path, nc=80).to(model_device)
            coco_model.load(weights, verbose=False) # 원본 .pt 가중치 로드
            coco_detect = coco_model.model[-1] # 80-class Detect 헤드
            print("[HEAD FIX] COCO model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL: Failed to load 80-class COCO weights source model. {e}")
            # coco_model 로드 실패 시, 헤드 가중치 복사 불가.
            coco_detect = None 
        # ======================================================================
        # --------- sync Detect head to 5-class layout ----------

# --------- sync Detect head to 5-class layout ----------
        active = getattr(self.args, "classes", None)

        if active and coco_detect is not None: # [수정] coco_detect가 있을 때만 실행
            active_nc = len(active)
            d = self.base_detect # 5-class target detect module
            head_device = next(self.det_model.parameters()).device
            head_dtype = next(self.det_model.parameters()).dtype
            reg_max = getattr(d, "reg_max", 16)
            reg_out = reg_max * 4
            
            d.nc = active_nc
            d.no = active_nc + reg_out

            target_names = ["person", "bicycle", "car", "motorcycle", "bus"]
            coco_name_to_idx = {name: idx for idx, name in enumerate(COCO80_NAMES)}
            selected_coco_ids = [coco_name_to_idx.get(name, 0) for name in target_names]

# ======================================================================
            # [신규] cv2 (Box Regression) 가중치 수동 복사
            # ======================================================================
            if hasattr(d, "cv2") and isinstance(d.cv2, nn.ModuleList) and \
               hasattr(coco_detect, "cv2") and isinstance(coco_detect.cv2, nn.ModuleList) and \
               len(d.cv2) == len(coco_detect.cv2):
                
                print("[HEAD FIX] Manually copying cv2 (Box) weights...")
                try:
                    with torch.no_grad():
                        for i in range(len(d.cv2)):
                            # cv2는 채널 수가 동일하므로 state_dict()로 전체 복사
                            d.cv2[i].load_state_dict(coco_detect.cv2[i].state_dict())
                    print("[HEAD FIX] cv2 weights copied successfully.")
                except Exception as e:
                    print(f"[HEAD FIX WARNING] Failed to copy cv2 weights: {e}")
            else:
                print("[HEAD FIX WARNING] cv2 (Box) head mismatch or not found. Skipping weight copy.")
            # ======================================================================
# ======================================================================
            # [신규 추가] dfl.conv (Distribution Focal Loss) 가중치 수동 복사
            # (이 블록을 cv2와 cv3 로직 사이에 추가하세요. L340 근처)
            # ======================================================================
            if hasattr(d, "dfl") and isinstance(getattr(d, "dfl", None), nn.Module) and \
               hasattr(coco_detect, "dfl") and isinstance(getattr(coco_detect, "dfl", None), nn.Module):
                
                print("[HEAD FIX] Manually copying dfl.conv weights...")
                try:
                    # d.dfl.conv와 coco_detect.dfl.conv는 state_dict() 구조가 동일함
                    with torch.no_grad():
                        # DFL 모듈 내부의 'conv' 레이어 가중치를 복사
                        d.dfl.conv.load_state_dict(coco_detect.dfl.conv.state_dict())
                    print("[HEAD FIX] dfl.conv weights copied successfully.")
                except Exception as e:
                    print(f"[HEAD FIX WARNING] Failed to copy dfl.conv weights: {e}")
            else:
                print("[HEAD FIX WARNING] dfl.conv module not found. Skipping weight copy.")
            # ======================================================================
# ======================================================================
            # [수정] cv3 (Class) 가중치 복사 로직 (아래 코드로 전체 교체)
            # ======================================================================
            if hasattr(d, "cv3") and isinstance(d.cv3, nn.ModuleList):
                print("[HEAD FIX] Rebuilding cv3 (Class) modules to match COCO structure...")
                try:
                    for i in range(len(d.cv3)): # i = 0, 1, 2 (각 레벨)
                        coco_seq = coco_detect.cv3[i] # 80-class Sequential, e.g., Conv(64,80), Conv(80,80), Conv(80,80)
                        target_seq = d.cv3[i]       # 5-class Sequential, e.g., Conv(64,64), Conv(64,64), Conv(64,5)
                        
                        if not (isinstance(coco_seq, nn.Sequential) and isinstance(target_seq, nn.Sequential) and len(coco_seq) == 3 and len(target_seq) == 3):
                            raise ValueError(f"Sequential structure mismatch at index {i}")

                        # 1. 새 Conv0 생성 (e.g., Conv(64, 80))
                        #    COCO 모델(coco_seq[0])의 가중치를 복사합니다.
                        c_in = target_seq[0].conv.in_channels   # 5-class 모델의 입력 채널 (e.g., 64)
                        c_mid = coco_seq[0].conv.out_channels # 80-class 모델의 중간 채널 (e.g., 80)
                        
                        # Ultralytics의 Conv 블록 타입(Conv)을 사용하여 재생성
                        new_conv0 = type(target_seq[0])( 
                            c_in, 
                            c_mid, 
                            k=target_seq[0].conv.kernel_size[0],
                            s=target_seq[0].conv.stride[0]
                        ).to(head_device, head_dtype)
                        new_conv0.load_state_dict(coco_seq[0].state_dict()) # 가중치 복사

                        # 2. 새 Conv1 생성 (e.g., Conv(80, 80))
                        #    COCO 모델(coco_seq[1])의 가중치를 복사합니다.
                        new_conv1 = type(target_seq[1])(
                            c_mid,
                            c_mid,
                            k=target_seq[1].conv.kernel_size[0],
                            s=target_seq[1].conv.stride[0]
                        ).to(head_device, head_dtype)
                        new_conv1.load_state_dict(coco_seq[1].state_dict()) # 가중치 복사

                        # 3. 새 Conv2 (최종 1x1 Conv) 생성 (e.g., Conv(80, 5))
                        coco_conv_final = coco_seq[-1] # 원본: nn.Conv2d(80, 80, 1)
                        
                        new_conv_final = nn.Conv2d(
                            in_channels=c_mid,      # 80
                            out_channels=active_nc, # 5
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=(coco_conv_final.bias is not None)
                        ).to(head_device, head_dtype)

                        # 80-class -> 5-class 가중치 선별 복사
                        with torch.no_grad():
                            new_conv_final.weight.zero_()
                            if new_conv_final.bias is not None:
                                new_conv_final.bias.zero_()
                            
                            for new_cls_idx, coco_cls_idx in enumerate(selected_coco_ids):
                                if 0 <= coco_cls_idx < coco_conv_final.weight.shape[0]: # 80
                                    # coco_conv_final.weight[idx]의 shape는 [in_channels, 1, 1] (e.g., [80, 1, 1])
                                    # new_conv_final.weight[idx]의 shape도 [in_channels, 1, 1] (e.g., [80, 1, 1])
                                    # in_channels(c_mid)가 80으로 동일하므로 복사 가능
                                    new_conv_final.weight[new_cls_idx] = coco_conv_final.weight[coco_cls_idx]
                                    if new_conv_final.bias is not None and coco_conv_final.bias is not None:
                                        new_conv_final.bias[new_cls_idx] = coco_conv_final.bias[coco_cls_idx]

                        # 4. d.cv3[i]의 Sequential 블록 전체를 새 블록으로 교체
                        d.cv3[i] = nn.Sequential(new_conv0, new_conv1, new_conv_final).to(head_device, head_dtype)
                        
                except Exception as e:
                    print(f"[HEAD FIX CRITICAL] Failed to rebuild cv3 modules. Weights may be random. Error: {e}")
                    import traceback
                    traceback.print_exc()

                print("[HEAD FIX] cv3 modules successfully rebuilt and weights re-mapped.")
            # ======================================================================
            # [수정 완료]
            # ======================================================================
            # 모델 이름 업데이트
            if hasattr(self.det_model, "names"):
                self.det_model.names = target_names
            if hasattr(self, "model") and isinstance(getattr(self, "model"), nn.Module):
                self.model.names = target_names
            
            print(f"[SYNC@build] Detect nc={d.nc}, no={d.no}, names={target_names}")
        elif coco_detect is None:
            print("[HEAD FIX ERROR] COCO source model failed to load. Head weights are random.")
        # update adapter references
        self.student_bb.layers = self.det_model.model
        self.student_bb.detect = self.det_model.model[-1]
        self.student_bb.from_idx = sorted(list(getattr(self.student_bb.detect, "f", self.student_bb.from_idx)))

        # CWD
        self.cwd = ChannelWiseDistillationLoss(scale_weights=(0.7, 0.2, 0.1), temperature=1.0)

        # custom hypers
        self.lambda_cwd = getattr(self, "lambda_cwd", 1.0)
        self.lambda_cwd_start = 0.0
        self.freeze_fa_epoch = getattr(self, "freeze_fa_epoch", 30)
        self.gradient_balance_gamma = getattr(self, "gradient_balance_gamma", 0.05)
        # --- warmup after head surgery (recalibrate detect caches/stride) ---
        self.det_model.eval()
        with torch.no_grad():
            hh, ww = (self.target_imgsz if hasattr(self, "target_imgsz") else (img_h, img_w))
            warm = torch.zeros(1, 3, hh, ww, device=model_device)
            _ = self.det_model(warm)  # wrapper�??�과?�며 Detect ?��? 캐시/stride ?�설??
        self.det_model.train()
# --- [디버깅 코드 수정] ---
        print("\n--- [DEBUG ?? Head Sync Check ---")
        d = self.base_detect
        print(f"[DEBUG ?? detect.nc = {getattr(d,'nc',None)}")
        print(f"[DEBUG ?? detect.no = {getattr(d,'no',None)}")
        
        # [수정] d.m 대신 d.cv2와 d.cv3의 채널을 직접 확인합니다.
        cv2_ch = [seq[-1].out_channels for seq in getattr(d, 'cv2', []) if isinstance(seq, nn.Sequential)]
        cv3_ch = [seq[-1].out_channels for seq in getattr(d, 'cv3', []) if isinstance(seq, nn.Sequential)]
        print(f"[DEBUG ?? detect.cv2 (Box) out channels = {cv2_ch}")
        print(f"[DEBUG ?? detect.cv3 (Cls) out channels = {cv3_ch}")
        
        print(f"[DEBUG ?? model.names = {getattr(self.det_model, 'names', None)}")
        print("--- [DEBUG ?? End Check ---\n")
# --- [디버깅 코드 수정 완료] ---
        return m

    # ---------- dataset/build: force 5-class head sync ----------
    def build_dataset(self, img_path, mode="train", batch=None):
        dset = super().build_dataset(img_path, mode, batch)

        # determine active classes: if args.classes specified, use them (e.g., [0,1,2,3,5])
        active = getattr(self.args, "classes", None)
        print(f"\n[DEBUG ?? Building dataset. Mode: {mode}")
        print(f"[DEBUG ?? Active classes from args: {active}")
        if active is not None and len(active) > 0:
            active_nc = len(active)
            id_map = {int(orig): idx for idx, orig in enumerate(active)}
# ======== [ 핵심 수정 ] ========
            # get_model과 동일한, 하드코딩된 타겟 이름을 사용합니다.
            target_names = ["person", "bicycle", "car", "motorcycle", "bus"]
            
            if active_nc == 5:
                dset.names = target_names
                print(f"[DEBUG ?? Dataset names FORCED to: {dset.names}\n")
            else:
                # (예외 처리) 5개 클래스가 아닌 경우에만 기존 로직 사용
                names_src = getattr(dset, "names", None)
                if isinstance(names_src, dict):
                    active_names = [names_src[i] for i in active]
                elif isinstance(names_src, (list, tuple)):
                    active_names = [names_src[i] for i in active]
                else:
                    active_names = [str(i) for i in range(active_nc)]
                dset.names = list(active_names)
                print(f"[DEBUG ?? Dataset names (Fallback): {dset.names}\n")
            # ======== [ 수정 완료 ] ========
            # remap class ids to contiguous range 0..active_nc-1
            if hasattr(dset, "labels"):
                for label in dset.labels:
                    cls_arr = label.get("cls", None)
                    if cls_arr is None or len(cls_arr) == 0:
                        continue
                    cls_np = np.asarray(cls_arr)
                    cls_flat = cls_np.reshape(-1)
                    for idx, val in enumerate(cls_flat):
                        mapped = id_map.get(int(val))
                        if mapped is None:
                            mapped = 0  # fallback but should not happen
                        cls_flat[idx] = mapped
                    label["cls"] = cls_flat.reshape(cls_np.shape)
        else:
            # fallback to full dataset
            active_nc = int(self.data["nc"])
            nm = self.data.get("names", None)
            if isinstance(nm, dict):
                active_names = [nm[i] for i in range(active_nc)]
            elif isinstance(nm, (list, tuple)):
                active_names = list(nm)
            else:
                active_names = [str(i) for i in range(active_nc)]
        return dset

    # ---------- validator with custom img size & normalization ----------
    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator = DYoloValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
        validator.target_imgsz = getattr(self, "target_imgsz", (448, 640))
        return validator

    # ---------- optimizer ----------
    def build_optimizer(self, model, **kwargs):
        name = str(kwargs.get("name", self.args.optimizer)).lower()
        lr = float(kwargs.get("lr", self.args.lr0))
        wd = float(kwargs.get("weight_decay", getattr(self.args, "weight_decay", 5e-4)))
        mom = float(kwargs.get("momentum", 0.9))

        seen, unique_params = set(), []
        for p in itertools.chain(model.parameters(), self.core.parameters()):
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            if p.requires_grad:
                unique_params.append(p)

        if name == "sgd":
            return torch.optim.SGD(unique_params, lr=lr, momentum=mom, weight_decay=wd)
        elif name in ("adamw", "adam"):
            return torch.optim.AdamW(unique_params, lr=lr, weight_decay=wd)
        else:
            return torch.optim.SGD(unique_params, lr=lr, momentum=mom, weight_decay=wd)

    # ---------- FA freeze schedule ----------
    def set_fa_trainable(self, trainable: bool):
        for m in [self.core.fa3, self.core.fa4, self.core.fa5]:
            for p in m.parameters():
                p.requires_grad = trainable

    # ---------- run detect ----------
    def _yolo_detect(self, F3, F4, F5):
        return self.base_detect([F3, F4, F5])

# [최종 수정] dyolo/train.py의 train_step 함수 전체를 이 코드로 교체하세요.

    # ---------- train step ----------
    def train_step(self, batch):
        device = self.device
        imgs = batch['img'].to(device, non_blocking=True)
        imgs_clear = batch.get('img_clear', imgs).to(device, non_blocking=True)

        # freeze at epoch
        if self.epoch == self.freeze_fa_epoch:
            self.set_fa_trainable(False)
            current_lr = self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') else self.args.lr0
            self.optimizer = self.build_optimizer(self.model, name=self.args.optimizer, lr=current_lr)

        # [수정 1] with torch.amp.autocast 블록을 if문 밖으로 꺼내어 들여쓰기를 수정합니다.
        with torch.amp.autocast('cuda', enabled=False):
            imgs_fp32 = imgs.float()
            imgs_clear_fp32 = imgs_clear.float()

            # student pathway
            (H3, H4, H5), (D3, D4, D5), (F3, F4, F5) = self.core.forward_student(imgs_fp32)
            
            # [수정 2] nb와 batch_progress를 여기서 한 번만 정의합니다.
            nb = max(getattr(self, 'nb', 1), 1)
            batch_progress = (getattr(self, 'batch_i', 0) + 1) / nb

            # [수정 5] 블렌딩된 피처로 preds를 계산합니다.
            preds = self._yolo_detect(F3, F4, F5)
            det_loss, loss_items = self.criterion(preds, batch)

            # [수정 6] D-YOLO 논문 전략에 따라 30 에포크 기준으로 손실 계산을 분리합니다.
            if self.epoch < self.freeze_fa_epoch:
                # --- Epoch 0-29: FA 모듈 훈련 ---
                
                # 1. CWD Loss 계산
                C3, C4, C5 = self.core.forward_teacher(imgs_clear_fp32)
                cwd_loss = self.cwd([C3, C4, C5], [D3, D4, D5]) # D 피처 사용

                # 2. CWD 웜업
                warm_epochs = max(getattr(self, 'lambda_cwd_warmup_epochs', 3), 1)
                progress = min(1.0, (self.epoch + batch_progress) / warm_epochs)
                beta_cwd = float(progress) * self.lambda_cwd 
                self.current_lambda_cwd = beta_cwd 

                # 3. Grad Penalty 계산
                def _grad_norm(loss_term, tensors):
                    grads = torch.autograd.grad(loss_term, tensors, retain_graph=True, allow_unused=True, create_graph=True)
                    grads = [g for g in grads if g is not None]
                    if not grads:
                        return torch.zeros((), device=loss_term.device, dtype=loss_term.dtype, requires_grad=True)
                    norm_sq = sum(g.pow(2).sum() for g in grads)
                    eps = loss_term.new_tensor(1e-12)
                    return torch.sqrt(norm_sq + eps)

                # [수정] 그래디언트 계산 기준을 D가 아닌 F로 변경합니다 (det_loss가 F에 의존)
                #det_grad_norm = _grad_norm(det_loss, [F3, F4, F5]) 
                cwd_grad_norm = _grad_norm(cwd_loss, [D3, D4, D5]) # cwd_loss는 D에 의존
                
                # [수정] Grad Penalty 수식 수정 (D-YOLO 논문 Eq. 10)
                # D에 대한 det_loss 그래디언트와 cwd_loss 그래디언트의 밸런스를 맞춰야 합니다.
                # det_loss는 F를 통해 D에 의존합니다. (det_loss -> F -> D)
                det_grad_norm_on_D = _grad_norm(det_loss, [D3, D4, D5])
                
                grad_penalty = self.gradient_balance_gamma * torch.abs(det_grad_norm_on_D - cwd_grad_norm)

# 4. 최종 손실 (Stage 1)
                total_loss = det_loss + (beta_cwd * cwd_loss) + grad_penalty
            
            else:
                # --- Epoch 30+: FA 모듈 동결 ---
                # 오직 det_loss로만 훈련합니다.
                total_loss = det_loss
                self.current_lambda_cwd = 0.0 # 로깅을 위해 0으로 설정

        return total_loss, loss_items

    # ---------- teacher clear image injection ----------
    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if "img_clear" in batch:
            return batch

        clear_train = getattr(self, "clear_train_path", None)
        clear_val = getattr(self, "clear_val_path", None)

        im_files = batch.get("im_file") or batch.get("paths")
        if not im_files:
            raise RuntimeError("Ultralytics batch has no file path list (im_file/paths).")

        imgs = batch["img"]
        device = imgs.device
        dtype = imgs.dtype
        target_h, target_w = imgs.shape[2], imgs.shape[3]

        clear_imgs = []
        for f in im_files:
            base = os.path.basename(f)
            is_train = ("\\images\\train\\" in f) or ("/images/train/" in f)
            clear_root = clear_train if is_train else clear_val
            if not clear_root:
                split = "train" if is_train else "val"
                raise FileNotFoundError(f"CLEAR path for split '{split}' is not set.")
            clear_path = os.path.join(clear_root, base)
            if not os.path.exists(clear_path):
                raise FileNotFoundError(f"Paired clear image not found: {clear_path}")
            clear_img = cv2.imread(clear_path, cv2.IMREAD_COLOR)
            if clear_img is None:
                raise RuntimeError(f"Failed to read clear image: {clear_path}")
            clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
            clear_lb, _, _ = letterbox_np(clear_img, (target_h, target_w), return_meta=True)
            clear_tensor = torch.from_numpy(clear_lb).permute(2, 0, 1).float() / 255.0
            clear_imgs.append(clear_tensor.to(device=device, dtype=dtype))

        batch["img_clear"] = torch.stack(clear_imgs, dim=0)
        return batch


class DYoloValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self._val_printed = False
        self.target_imgsz = (448, 640)

    def _peek_pred_stats(self, preds):
        # reg_out 추정
        reg_out = 64
        try:
            det = getattr(self.model, "model", [None])[-1]
            if det is not None and hasattr(det, "reg_max"):
                reg_out = int(det.reg_max) * 4
        except Exception:
            pass

        if isinstance(preds, (list, tuple)):
            for i, p in enumerate(preds):
                if hasattr(p, "shape") and p.ndim == 4:
                    B, C, H, W = p.shape
                    if C > reg_out:
                        cls = p[:, reg_out:, :, :]
                        cls_sig = cls.sigmoid()
                        print(f"[VAL DEBUG] lvl{i} cls max={cls_sig.max().item():.4f}, "
                              f"mean={cls_sig.mean().item():.4f}, shape={tuple(cls.shape)}")

    def postprocess(self, preds):
        # ?�시 로짓 ?�태 ?�인 (1?�만)
        if not self._val_printed:
            try:
                self._peek_pred_stats(preds)
            except Exception as e:
                print(f"[VAL DEBUG] peek error: {e}")

        outs = super().postprocess(preds)

        if not self._val_printed:
            # 박스 개수 집계 (?�전 처리)
            def count_boxes(p):
                if hasattr(p, "shape"):
                    return int(p.shape[0])
                if hasattr(p, "boxes"):
                    try: return int(len(p.boxes))
                    except: pass
                if isinstance(p, dict):
                    if "det" in p and hasattr(p["det"], "shape"): return int(p["det"].shape[0])
                    if "boxes" in p:
                        b = p["boxes"]
                        if hasattr(b, "shape"): return int(b.shape[0])
                        if hasattr(b, "__len__"): return int(len(b))
                if isinstance(p, (list, tuple)):
                    return sum(count_boxes(x) for x in p)
                return 0

            seq = outs if isinstance(outs, (list, tuple)) else [outs]
            counts = [count_boxes(p) for p in seq]
            print(f"[VAL DEBUG] items={len(seq)} non_empty={sum(c>0 for c in counts)} counts={counts}")
            self._val_printed = True

        return outs


def main():
    # --- settings ---
    MODEL_CONFIG = 'yolov8n.yaml'
    
    # [수정 1] 전처리된 448x640 데이터셋 경로로 변경
    DATA_CONFIG = r'C:\Users\user\Desktop\dyolo\datasets\VOC-Foggy-448\voc-foggy-448.yaml'
    CLEAR_TRAIN_PATH = r'C:\Users\user\Desktop\dyolo\datasets\VOC-Clear-448\images\train'
    CLEAR_VAL_PATH   = r'C:\Users\user\Desktop\dyolo\datasets\VOC-Clear-448\images\val'

    EPOCHS = 100
    BATCH_SIZE = 16
    
    # [수정 2] imgsz는 가장 긴 변인 640 (정수)로 설정
    IMG_SIZE = [448, 640]
    
    OPTIMIZER = 'SGD'
    LEARNING_RATE = 0.01
    COSINE_LR = True
    MOSAIC = 0.0

    # overrides
    yolo_overrides = {
        "model": "yolov8n.pt",
        "data": DATA_CONFIG,     # [수정 1] 반영
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMG_SIZE,       # [수정 2] 반영 (640 정수)
        
        # [확인] rect=True는 직사각형 학습에 필수
        "rect": True,
        "amp": True,
        "conf": 0.001,
        "optimizer": OPTIMIZER,
        "lr0": LEARNING_RATE,
        "cos_lr": COSINE_LR,
        "mosaic": MOSAIC,
        "cache": "disk",
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "flipud": 0.0,
        "degrees": 0.0,
        "scale": 0.5,
        "shear": 0.0,
        "translate": 0.1,
        "workers": 8,
        "classes": [0, 1, 2, 3, 4],
    }

    trainer = DYoloTrainer(overrides=yolo_overrides)

    # custom attrs (not in args)
    trainer.lambda_cwd = 1.0
    trainer.freeze_fa_epoch = 30
    trainer.clear_train_path = CLEAR_TRAIN_PATH
    trainer.clear_val_path = CLEAR_VAL_PATH
    
    # [수정 3] 모델 빌드 기준 해상도를 (H, W) 튜플로 하드코딩
    #          (IMG_SIZE, IMG_SIZE)가 아닌 (448, 640)으로 수정
    trainer.target_imgsz = (448, 640)
    
    trainer.add_callback("on_model_save", _save_core_when_best)
    trainer.train()


if __name__ == "__main__":
    main()
