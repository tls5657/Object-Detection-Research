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

    base_detect = layers[-1]
    wrapper = DYoloDetectWrapper(core, base_detect)
    layers[-1] = wrapper

    student_bb.layers = layers
    student_bb.detect = wrapper
    student_bb.from_idx = sorted(list(getattr(wrapper, "f", student_bb.from_idx)))
    return wrapper


# [수정] test.py의 _load_core_weights 함수를 아래 코드로 전체 교체

def _load_core_weights(model, device):
    """Load FA/AFM parameters saved alongside best.pt."""
    layers = getattr(getattr(model, "model", None), "model", None)
    if layers is None or not isinstance(layers[-1], DYoloDetectWrapper):
        print("[INFO] DYoloDetectWrapper not found; skipping core restore.")
        return

    wrapper: DYoloDetectWrapper = layers[-1]
    if CORE_PATH.exists():
        state = torch.load(CORE_PATH, map_location=device)
        
        # [핵심 수정] 
        # best_core.pth에서 'backbone_s.'로 시작하는 모든 키를 필터링(제거)합니다.
        # (backbone 가중치는 best.pt에 이미 올바르게 저장되어 있으므로)
        # 오직 fa3, fa4, fa5, afm3, afm4, afm5 등 D-YOLO 핵심 모듈만 로드합니다.
        core_only_state = {k: v for k, v in state.items() if not k.startswith("backbone_s.")}

        # 필터링된 state_dict를 로드합니다.
        missing, unexpected = wrapper.core.load_state_dict(core_only_state, strict=False)
        
        # 'unexpected' 키에 'backbone_s' 관련 키들이 뜨는 것은 정상이므로,
        # 'missing' 키만 확인하여 FA/AFM이 잘 로드되었는지 확인합니다.
        if missing:
            print(f"[WARN] Missing keys when loading core: {missing}")
        
        # unexpected 키는 대부분 'backbone_s...'이므로 무시해도 됩니다.
        # if unexpected:
        #    print(f"[WARN] Unexpected keys when loading core: {unexpected}")
            
        print(f"[INFO] Restored D-YOLO core weights (FA/AFM) from {CORE_PATH}")
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
    # [수정] MODEL_PATH를 'train58' 등 최신 경로로 업데이트하세요.
    MODEL_PATH = Path(r"C:\Users\user\Desktop\dyolo\runs\detect\train58\weights\best.pt")
    CORE_PATH = MODEL_PATH.parent / "best_core.pth"
    DATA_YAML_PATH = Path(r"C:\Users\user\Desktop\dyolo\datasets\rtts\data.yaml")

    TARGET_IMGSZ = 640  # [수정] 448, 640 튜플이 아닌 640 정수로 설정
    BATCH_SIZE = 16

    model = YOLO(str(MODEL_PATH))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")

    # [수정] imgsz 인자를 640 정수로 전달
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
        workers=8,              # [권장] 0 -> 8 (학습과 동일하게)
        device=device,
        split='test',           # [필수] 'val'이 아닌 'test' 세트 사용
        amp=False               # [권장] FA/AFM의 FP32 연산을 보장하기 위해 AMP 끄기
    )

    metrics = results.box.get_metrics()
    print("\n===== RTTS **Test Set** Evaluation =====") # 제목 변경
    if hasattr(model.model, "names"):
        print("model.names:", model.model.names)
    print(f"mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
    print(f"mAP50:     {metrics['metrics/mAP50(B)']:.4f}")
    print("=========================================\n")


if __name__ == "__main__":
    main()