"""Offline NEU image-level classification with saved project checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.models.cnn import NEUCNN


ROOT = Path(__file__).resolve().parents[2]
CLASSES = (
    ("Cr", "Crazing", "龟裂"), ("In", "Inclusion", "夹杂"),
    ("Pa", "Patches", "斑块"), ("PS", "Pitted Surface", "麻点"),
    ("RS", "Rolled-in Scale", "轧入氧化皮"), ("Sc", "Scratches", "划痕"),
)


def checkpoint_path(mode: str, seed: int) -> Path:
    if mode not in {"A", "B", "C"} or seed not in {0, 1, 2}:
        raise ValueError("未知模型设置")
    return ROOT / "checkpoints" / "neu" / f"cnn_{mode}_seed{seed}.pt"


def load_classifier(mode: str, seed: int, device: str | None = None):
    path = checkpoint_path(mode, seed)
    if not path.is_file():
        raise FileNotFoundError(f"模型文件缺失：{path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("model") != "NEUCNN" or payload.get("mode") != mode or payload.get("seed") != seed:
        raise ValueError(f"模型元数据与选择不匹配：{path}")
    model = NEUCNN().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return model, device, version


def preprocess(image: Image.Image) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("L")
    image = image.resize((128, 128), Image.Resampling.BILINEAR)
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
        results.append({"class_code": CLASSES[index][0],
                        "class_en": CLASSES[index][1],
                        "class_zh": CLASSES[index][2],
                        "confidence": float(row[index]),
                        "probabilities": {CLASSES[i][0]: float(row[i]) for i in range(6)}})
    return results
