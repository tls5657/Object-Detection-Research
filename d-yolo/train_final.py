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
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
add_safe_globals([DetectionModel, Conv, Sequential, Conv2d, BatchNorm2d])

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
            tau = float(getattr(self.core, "tau", 1.0))
            if tau < 1.0:
                F3 = H3 + tau * (F3 - H3)
                F4 = H4 + tau * (F4 - H4)
                F5 = H5 + tau * (F5 - H5)
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

        # --- FIX: 채널 ?�로�??�상???�드 고정 ---
        model_device = next(self.det_model.model.parameters()).device
        if hasattr(self, "target_imgsz"):
            img_h, img_w = self.target_imgsz  # e.g., (448, 640)
        else:
            # ?�시 target_imgsz�????�팅?�을 ?�만 기존 로직 ?�용(백업)
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
        self.core.tau = 0.0

        # Wrap Detect
        detect_module = self.det_model.model[-1]
        if not isinstance(detect_module, DYoloDetectWrapper):
            wrapped_detect = DYoloDetectWrapper(self.core, detect_module)
            self.det_model.model[-1] = wrapped_detect
        else:
            wrapped_detect = detect_module
        self.base_detect = wrapped_detect.detect  # raw Detect

        # --------- sync Detect head to 5-class layout ----------

        active = getattr(self.args, "classes", None)  # expected [0,1,2,3,4]

        if active:

            active_nc = len(active)

            d = self.base_detect

            head_device = next(self.det_model.parameters()).device

            head_dtype = next(self.det_model.parameters()).dtype

            reg_max = getattr(d, "reg_max", 16)

            reg_out = reg_max * 4

            d.nc = active_nc

            d.no = active_nc + reg_out



            target_names = ["person", "bicycle", "car", "motorcycle", "bus"]

            coco_name_to_idx = {name: idx for idx, name in enumerate(COCO80_NAMES)}

            selected_coco_ids = [coco_name_to_idx.get(name, 0) for name in target_names]



            old_heads = list(getattr(d, "m", []))

            new_heads = []

            for old_conv in old_heads:

                new_conv = nn.Conv2d(old_conv.in_channels, d.no, 1, bias=True)

                with torch.no_grad():

                    new_conv.weight.zero_()

                    new_conv.bias.zero_()

                    src_total = old_conv.weight.shape[0]

                    reg_copy = min(reg_out, src_total)

                    if reg_copy > 0:

                        new_conv.weight[:reg_copy] = old_conv.weight[:reg_copy]

                        new_conv.bias[:reg_copy] = old_conv.bias[:reg_copy]

                    for new_cls_idx, coco_cls_idx in enumerate(selected_coco_ids):

                        src_idx = reg_out + coco_cls_idx

                        dst_idx = reg_out + new_cls_idx

                        if 0 <= src_idx < src_total and dst_idx < new_conv.weight.shape[0]:

                            new_conv.weight[dst_idx] = old_conv.weight[src_idx]

                            new_conv.bias[dst_idx] = old_conv.bias[src_idx]

                new_conv.to(device=head_device, dtype=head_dtype)

                new_heads.append(new_conv)

            if new_heads:

                d.m = nn.ModuleList(new_heads)



            if hasattr(d, "cv3"):

                for seq in d.cv3:

                    if not (isinstance(seq, nn.Sequential) and len(seq) > 0):

                        continue

                    last = seq[-1]

                    if not isinstance(last, nn.Conv2d):

                        continue

                    if last.out_channels == active_nc:

                        continue

                    new_cls_conv = nn.Conv2d(last.in_channels, active_nc, kernel_size=1, bias=last.bias is not None)

                    with torch.no_grad():

                        new_cls_conv.weight.zero_()

                        if new_cls_conv.bias is not None:

                            new_cls_conv.bias.zero_()

                        src_total = last.weight.shape[0]

                        for new_cls_idx, coco_cls_idx in enumerate(selected_coco_ids):

                            if 0 <= coco_cls_idx < src_total:

                                new_cls_conv.weight[new_cls_idx] = last.weight[coco_cls_idx]

                                if new_cls_conv.bias is not None and last.bias is not None:

                                    new_cls_conv.bias[new_cls_idx] = last.bias[coco_cls_idx]

                    new_cls_conv.to(device=head_device, dtype=head_dtype)

                    seq[-1] = new_cls_conv

            if hasattr(d, "initialize_biases"):

                d.initialize_biases()



            if hasattr(self.det_model, "names"):

                self.det_model.names = target_names

            if hasattr(self, "model") and isinstance(getattr(self, "model"), nn.Module):

                self.model.names = target_names

            print(f"[SYNC@build] Detect nc={d.nc}, no={d.no}, names={target_names}")
        # update adapter references
        self.student_bb.layers = self.det_model.model
        self.student_bb.detect = self.det_model.model[-1]
        self.student_bb.from_idx = sorted(list(getattr(self.student_bb.detect, "f", self.student_bb.from_idx)))

        # CWD
        self.cwd = ChannelWiseDistillationLoss(scale_weights=(0.7, 0.2, 0.1), temperature=1.0)

        # custom hypers
        self.lambda_cwd = getattr(self, "lambda_cwd", 1.0)
        self.lambda_cwd_start = 0.0
        self.gradient_balance_gamma = getattr(self, "gradient_balance_gamma", 0.05)
        self.freeze_fa_epoch = getattr(self, "freeze_fa_epoch", 30)
        # --- warmup after head surgery (recalibrate detect caches/stride) ---
        self.det_model.eval()
        with torch.no_grad():
            hh, ww = (self.target_imgsz if hasattr(self, "target_imgsz") else (img_h, img_w))
            warm = torch.zeros(1, 3, hh, ww, device=model_device)
            _ = self.det_model(warm)  # wrapper�??�과?�며 Detect ?��? 캐시/stride ?�설??
        self.det_model.train()
# --- [?�버�?코드 ?? ---
        print("\n--- [DEBUG ?? Head Sync Check ---")
        d = self.base_detect
        print(f"[DEBUG ?? detect.nc = {getattr(d,'nc',None)}")
        print(f"[DEBUG ?? detect.no = {getattr(d,'no',None)}")
        print(f"[DEBUG ?? head conv out channels = {[m.out_channels for m in getattr(d,'m',[])]}")
        print(f"[DEBUG ?? model.names = {getattr(self.det_model, 'names', None)}")
        print("--- [DEBUG ?? End Check ---\n")
        # --- [?�버�?코드 ???? ---
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
            names_src = getattr(dset, "names", None)
            if isinstance(names_src, dict):
                active_names = [names_src[i] for i in active]
            elif isinstance(names_src, (list, tuple)):
                active_names = [names_src[i] for i in active]
            else:
                active_names = [str(i) for i in range(active_nc)]
                print(f"[DEBUG ?? Dataset names BEFORE remap: {getattr(dset, 'names', 'N/A')}")
            dset.names = list(active_names)
            print(f"[DEBUG ?? Dataset names AFTER remap: {dset.names}\n")
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

        # # sync Detect head to active_nc
        # d = self.base_detect  # raw Detect
        # if getattr(d, "nc", None) != active_nc:
        #     d.nc = active_nc
        #     d.no = active_nc + getattr(d, "reg_max", 0) * 4
        #     if hasattr(d, "m"):
        #         d.m = nn.ModuleList(
        #             [nn.Conv2d(conv.in_channels, d.no, kernel_size=1, bias=True) for conv in d.m]
        #         )
        #         if hasattr(d, "initialize_biases"):
        #             d.initialize_biases()
        # update names for metrics

        # # one-time print (sanity)
        # if not getattr(self, "_printed_nc", False):
        #     self._printed_nc = True

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

    # ---------- train step ----------
    def train_step(self, batch):
        device = self.device
        imgs = batch["img"].to(device, non_blocking=True)
        imgs_clear = batch.get("img_clear", imgs).to(device, non_blocking=True)

        # freeze at epoch
        if self.epoch == self.freeze_fa_epoch:
            self.set_fa_trainable(False)
            current_lr = self.optimizer.param_groups[0]["lr"] if hasattr(self, "optimizer") else self.args.lr0
            self.optimizer = self.build_optimizer(self.model, name=self.args.optimizer, lr=current_lr)

        # student pathway
        (H3, H4, H5), (D3, D4, D5), (F3, F4, F5) = self.core.forward_student(imgs)
        preds = self._yolo_detect(F3, F4, F5)

        det_loss, loss_items = self.criterion(preds, batch)

        # teacher + CWD
        C3, C4, C5 = self.core.forward_teacher(imgs_clear)
        cwd_loss = self.cwd([C3, C4, C5], [D3, D4, D5])

        nb = max(getattr(self, "nb", 1), 1)
        batch_progress = (getattr(self, "batch_i", 0) + 1) / nb
        W = max(getattr(self, "core_warmup_epochs", 3), 1)
        tau = min(1.0, (self.epoch + batch_progress) / W)
        self.core.tau = float(tau)
        # 변�?(3 epoch ?�업, tau?� 같�? ?�이?�어)
        Wl = max(getattr(self, "lambda_cwd_warmup_epochs", 3), 1)
        warm = min(1.0, (self.epoch + batch_progress) / Wl)
        lambda_dynamic = warm * self.lambda_cwd

        def _grad_norm(loss_term, tensors):
            grads = torch.autograd.grad(loss_term, tensors, retain_graph=True, allow_unused=True, create_graph=True)
            grads = [g for g in grads if g is not None]
            if not grads:
                return torch.zeros((), device=loss_term.device, dtype=loss_term.dtype, requires_grad=True)
            norm_sq = sum(g.pow(2).sum() for g in grads)
            eps = loss_term.new_tensor(1e-12)
            return torch.sqrt(norm_sq + eps)

        det_grad_norm = _grad_norm(det_loss, [D3, D4, D5])
        cwd_grad_norm = _grad_norm(cwd_loss, [D3, D4, D5])
        grad_penalty = self.gradient_balance_gamma * torch.abs(det_grad_norm - cwd_grad_norm)

        self.current_lambda_cwd = float(lambda_dynamic)
        total_loss = det_loss + lambda_dynamic * cwd_loss + grad_penalty
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
        self.target_imgsz = (640, 640)

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
    DATA_CONFIG = r'C:\Users\user\Desktop\dyolo\datasets\VOC-Foggy-448\voc-foggy-448.yaml'
    CLEAR_TRAIN_PATH = r'C:\Users\user\Desktop\dyolo\datasets\VOC-Clear-448\images\train'
    CLEAR_VAL_PATH   = r'C:\Users\user\Desktop\dyolo\datasets\VOC-Clear-448\images\val'

    EPOCHS = 100
    BATCH_SIZE = 16
    
    # [?�정 1] ?�상?��? 640 ?�수?�서 [448, 640] 리스?�로 변�?
    IMG_SIZE = 640
    
    OPTIMIZER = 'SGD'
    LEARNING_RATE = 0.01
    COSINE_LR = True
    MOSAIC = 0.0

    # overrides
    yolo_overrides = {
        "model": "yolov8n.pt",
        "data": DATA_CONFIG,
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMG_SIZE,
        
        # [?�정 2] 직사각형 ?�련 �?검�?모드 ?�성??
        "rect": True,           
        "conf": 0.001,
        "optimizer": OPTIMIZER,
        "lr0": LEARNING_RATE,
        "cos_lr": COSINE_LR,
        "mosaic": MOSAIC,
        "cache": "disk",
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "fliplr": 0.0, "flipud": 0.0,
        "degrees": 0.0, "scale": 0.0, "shear": 0.0, "translate": 0.0,
        "workers": 8,
        "classes": [0, 1, 2, 3, 4],
    }

    trainer = DYoloTrainer(overrides=yolo_overrides)

    # custom attrs (not in args)
    trainer.lambda_cwd = 1.0
    trainer.freeze_fa_epoch = 30
    trainer.clear_train_path = CLEAR_TRAIN_PATH
    trainer.clear_val_path = CLEAR_VAL_PATH
    trainer.target_imgsz = (IMG_SIZE, IMG_SIZE)
    trainer.add_callback("on_model_save", _save_core_when_best)
    trainer.train()


if __name__ == "__main__":
    main()
