"""Offline labeled NEU-CVAE sample generation and local PNG export."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import torch
from PIL import Image

from src.models.cvae_conv import NEUCVAE
from src.services.inference_service import CLASSES, ROOT


def load_generator(device: str | None = None):
    path = ROOT / "checkpoints" / "neu" / "cvae_seed0.pt"
    if not path.is_file():
        raise FileNotFoundError(f"生成器文件缺失：{path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("model") != "NEUCVAE" or payload.get("classes") != 6:
        raise ValueError("生成器元数据不匹配")
    model = NEUCVAE(latent_dim=payload["latent_dim"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, device, hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def generate(model, device: str, class_code: str, count: int, seed: int):
    codes = [row[0] for row in CLASSES]
    if class_code not in codes or not 1 <= count <= 48 or seed < 0:
        raise ValueError("生成参数无效")
    labels = torch.full((count,), codes.index(class_code), dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        tensor = model.sample(labels, rng).cpu().clamp(0, 1)
    return [Image.fromarray((image.squeeze().numpy() * 255).round().astype(np.uint8))
            for image in tensor]


def save_generated(images: list[Image.Image], class_code: str, seed: int, version: str):
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    folder = ROOT / "generated" / "app" / f"{stamp}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    names = []
    for index, image in enumerate(images):
        name = f"{class_code}_{index + 1:03d}.png"
        image.save(folder / name)
        names.append(name)
    metadata = {"class_code": class_code, "seed": seed, "count": len(images),
                "model_version": version, "files": names}
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return folder, metadata
