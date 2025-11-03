import torch
from pathlib import Path

from ultralytics import YOLO

from train_final import (
    DYoloDetectWrapper,
    YOLOv8BackboneAdapter as _YOLOv8BackboneAdapter,
)
from dyolo_core import DYOLOCore

# ensure pickle compatibility
YOLOv8BackboneAdapter = _YOLOv8BackboneAdapter

# paths
MODEL_PATH = Path(r"C:\Users\user\Desktop\dyolo\runs\detect\train\weights\best.pt")
CORE_PATH = MODEL_PATH.parent / "best_core.pth"
DATA_YAML_PATH = Path(r"C:\Users\user\Desktop\dyolo\datasets\rtts\data.yaml")

TARGET_IMGSZ = 640
BATCH_SIZE = 16


def _ensure_dyolo_wrapper(model, device, imgsz):
    """Wrap Ultralytics detect head so FA/AFM are active."""
    layers = getattr(getattr(model, "model", None), "model", None)
    if layers is None or not len(layers):
        raise RuntimeError("Unexpected Ultralytics model structure.")

    if isinstance(layers[-1], DYoloDetectWrapper):
        return layers[-1]

    student_bb = YOLOv8BackboneAdapter(model)
    if isinstance(imgsz, (list, tuple)):
        img_h = int(imgsz[0])
        img_w = int(imgsz[1] if len(imgsz) > 1 else imgsz[0])
    else:
        img_h = img_w = int(imgsz)

    dummy = torch.zeros(1, 3, img_h, img_w, device=device)
    with torch.no_grad():
        H3, H4, H5 = student_bb(dummy)
    chs = (H3.shape[1], H4.shape[1], H5.shape[1])

    core = DYOLOCore(student_backbone=student_bb, chs=chs, use_cfe=False)
    core.tau = 1.0

    base_detect = layers[-1]
    wrapper = DYoloDetectWrapper(core, base_detect)
    layers[-1] = wrapper

    student_bb.layers = layers
    student_bb.detect = wrapper
    student_bb.from_idx = sorted(list(getattr(wrapper, "f", student_bb.from_idx)))
    return wrapper


def _load_core_weights(model, device):
    """Load FA/AFM parameters saved alongside best.pt."""
    layers = getattr(getattr(model, "model", None), "model", None)
    if layers is None or not isinstance(layers[-1], DYoloDetectWrapper):
        print("[INFO] DYoloDetectWrapper not found; skipping core restore.")
        return

    wrapper: DYoloDetectWrapper = layers[-1]
    if CORE_PATH.exists():
        state = torch.load(CORE_PATH, map_location=device)
        missing, unexpected = wrapper.core.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading core: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading core: {unexpected}")
        print(f"[INFO] Restored D-YOLO core weights from {CORE_PATH}")
    else:
        print(f"[WARN] {CORE_PATH} not found. Using core state from best.pt only.")

    core_modules = [
        wrapper.core.fa3,
        wrapper.core.fa4,
        wrapper.core.fa5,
        wrapper.core.afm3,
        wrapper.core.afm4,
        wrapper.core.afm5,
    ]
    for attr in ("cfe", "cfe_proj3", "cfe_proj4", "cfe_proj5"):
        module = getattr(wrapper.core, attr, None)
        if module is not None:
            core_modules.append(module)

    for module in core_modules:
        module.to(device=device, dtype=torch.float32)

    wrapper._core_dev = device


def main():
    model = YOLO(str(MODEL_PATH))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")

    _ensure_dyolo_wrapper(model, device, TARGET_IMGSZ)
    model.to(device)
    _load_core_weights(model, device)

    results = model.val(
        data=str(DATA_YAML_PATH),
        imgsz=TARGET_IMGSZ,
        batch=BATCH_SIZE,
        conf=0.001,
        iou=0.7,
        rect=True,
        workers=0,
        device=device,
    )

    metrics = results.box.get_metrics()
    print("\n===== RTTS Evaluation =====")
    if hasattr(model.model, "names"):
        print("model.names:", model.model.names)
    print(f"mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
    print(f"mAP50:    {metrics['metrics/mAP50(B)']:.4f}")
    print("==========================\n")


if __name__ == "__main__":
    main()
