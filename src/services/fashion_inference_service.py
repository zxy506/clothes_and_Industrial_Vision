"""Fashion-MNIST image classification using the formal 30-epoch checkpoints."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from PIL import Image, ImageOps

from src.models.cnn import SmallCNN
from src.services.inference_service import ROOT


CLASSES = (
    ("0", "T-shirt/top", "T恤/上衣"), ("1", "Trouser", "裤子"),
    ("2", "Pullover", "套衫"), ("3", "Dress", "连衣裙"),
    ("4", "Coat", "外套"), ("5", "Sandal", "凉鞋"),
    ("6", "Shirt", "衬衫"), ("7", "Sneaker", "运动鞋"),
    ("8", "Bag", "包"), ("9", "Ankle boot", "短靴"),
)


def checkpoint_path(mode: str, seed: int):
    if mode not in {"A", "B", "C25", "C"} or seed not in {0, 1, 2}:
        raise ValueError("未知 Fashion 模型设置")
    return ROOT / "checkpoints" / "fashion" / f"cnn_{mode}_seed{seed}_30epochs.pt"


def load_classifier(mode: str, seed: int, device: str | None = None):
    path = checkpoint_path(mode, seed)
    if not path.is_file():
        raise FileNotFoundError(f"衣物模型文件缺失：{path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("model"), payload.get("mode"), payload.get("seed"), payload.get("epochs")) != (
            "SmallCNN", mode, seed, 30):
        raise ValueError(f"衣物模型元数据与选择不匹配：{path}")
    model = SmallCNN().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, device, hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def preprocess(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("L").resize((28, 28), Image.Resampling.BILINEAR)
    array = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).unsqueeze(0).float().div_(255)


def predict(model, device: str, images: list[Image.Image]):
    if not images:
        return []
    batch = torch.stack([preprocess(image) for image in images]).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch), dim=1).cpu().numpy()
    results = []
    for row in probabilities:
        index = int(row.argmax())
        results.append({"class_code": CLASSES[index][0], "class_en": CLASSES[index][1],
                        "class_zh": CLASSES[index][2], "confidence": float(row[index]),
                        "probabilities": {CLASSES[i][0]: float(row[i]) for i in range(10)}})
    return results
