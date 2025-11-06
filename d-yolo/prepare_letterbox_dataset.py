import math
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


def letterbox_np(image: np.ndarray,
                 new_shape: Tuple[int, int],
                 color: Tuple[int, int, int] = (114, 114, 114),
                 scaleup: bool = True) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad image to new_shape while keeping aspect ratio."""
    h0, w0 = image.shape[:2]
    new_h, new_w = new_shape
    r = min(new_h / h0, new_w / w0)
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    if (w0, h0) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw = new_w - new_unpad[0]
    dh = new_h - new_unpad[1]
    dw /= 2
    dh /= 2

    left, right = int(math.floor(dw)), int(math.ceil(dw))
    top, bottom = int(math.floor(dh)), int(math.ceil(dh))

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image, r, (float(left), float(top))


def load_yolo_labels(label_path: Path) -> np.ndarray:
    """Load YOLO-format labels as numpy array with shape (N,5)."""
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    raw = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
    if raw.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return raw.reshape(-1, 5)


def save_yolo_labels(label_path: Path, labels: np.ndarray) -> None:
    """Save YOLO-format labels. Creates empty file if no labels."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if labels.size == 0:
        label_path.write_text("")
    else:
        np.savetxt(
            label_path,
            labels,
            fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"],
        )


def process_split(src_img_dir: Path,
                  src_label_dir: Path,
                  dst_img_dir: Path,
                  dst_label_dir: Path,
                  target_shape: Tuple[int, int],
                  color: Tuple[int, int, int]) -> None:
    """Convert all images/labels within a split."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths: Iterable[Path] = sorted(
        p for p in src_img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for img_path in image_paths:
        rel_name = img_path.stem
        dst_img_path = dst_img_dir / f"{rel_name}.jpg"
        label_src = src_label_dir / f"{rel_name}.txt"
        label_dst = dst_label_dir / f"{rel_name}.txt"

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        h0, w0 = image.shape[:2]
        labels = load_yolo_labels(label_src)

        boxes = None
        if labels.size != 0:
            xywh = labels[:, 1:5].copy()
            xywh[:, 0] *= w0
            xywh[:, 1] *= h0
            xywh[:, 2] *= w0
            xywh[:, 3] *= h0

            x, y, w, h = np.split(xywh, 4, axis=1)
            boxes = np.concatenate([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w0)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h0)

        dst_image, gain, (pad_x, pad_y) = letterbox_np(
            image, target_shape, color=color, scaleup=True
        )

        if boxes is not None and boxes.size != 0:
            boxes[:, 0::2] = boxes[:, 0::2] * gain + pad_x
            boxes[:, 1::2] = boxes[:, 1::2] * gain + pad_y
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, target_shape[1])
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, target_shape[0])

            w_new = boxes[:, 2] - boxes[:, 0]
            h_new = boxes[:, 3] - boxes[:, 1]
            x_new = boxes[:, 0] + w_new / 2
            y_new = boxes[:, 1] + h_new / 2

            labels[:, 1] = x_new / target_shape[1]
            labels[:, 2] = y_new / target_shape[0]
            labels[:, 3] = w_new / target_shape[1]
            labels[:, 4] = h_new / target_shape[0]
            labels[:, 1:] = np.clip(labels[:, 1:], 0.0, 1.0)

        cv2.imwrite(str(dst_img_path), dst_image)
        save_yolo_labels(label_dst, labels)


def main():
    TARGET_SHAPE = (448, 640)
    COLOR = (114, 114, 114)
    BASE = Path(r"C:/Users/user/Desktop/dyolo/datasets")
    dataset_configs = [
        ("VOC-Foggy", "VOC-Foggy-448", "voc-foggy.yaml", "voc-foggy-448.yaml"),
        ("VOC_Clear", "VOC-Clear-448", None, None),
    ]

    for src_name, dst_name, yaml_in, yaml_out in dataset_configs:
        src_root = BASE / src_name
        dst_root = BASE / dst_name

        print(f"[INFO] Processing {src_root} -> {dst_root}")
        for split in ("train", "val"):
            src_img_dir = src_root / "images" / split
            src_label_dir = src_root / "labels" / split
            dst_img_dir = dst_root / "images" / split
            dst_label_dir = dst_root / "labels" / split

            if not src_img_dir.exists():
                print(f"[WARN] Image directory not found: {src_img_dir}, skipping split '{split}'.")
                continue

            process_split(
                src_img_dir,
                src_label_dir,
                dst_img_dir,
                dst_label_dir,
                target_shape=TARGET_SHAPE,
                color=COLOR,
            )

        if yaml_in and yaml_out:
            src_yaml = src_root / yaml_in
            dst_yaml = dst_root / yaml_out
            if src_yaml.exists():
                text_in = src_yaml.read_text(encoding='utf-8')
                text_out = text_in.replace(str(src_root), str(dst_root))
                dst_yaml.parent.mkdir(parents=True, exist_ok=True)
                dst_yaml.write_text(text_out, encoding='utf-8')
                print(f"[INFO] Wrote {dst_yaml}")


if __name__ == "__main__":
    main()
