"""Load only the requested fixed NEU-CLS split at 128×128 grayscale."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_neu(*, include_test=False, root="data/raw/neu_cls", manifest="data/splits/neu.json"):
    record = json.loads(Path(manifest).read_text(encoding="utf-8"))
    classes = record["classes"]
    names = {name.lower(): index for index, name in enumerate(classes)}

    def split(paths):
        images, labels = [], []
        for relative in paths:
            path = Path(root) / relative
            prefix = path.stem.split("_")[0].lower()
            if prefix not in names:
                raise ValueError(f"unknown NEU class: {path}")
            with Image.open(path) as image:
                pixels = np.array(image.convert("L").resize((128, 128), Image.Resampling.BILINEAR),
                                  dtype=np.uint8, copy=True)
            images.append(torch.from_numpy(pixels).unsqueeze(0))
            labels.append(names[prefix])
        return torch.stack(images).float().div_(255), torch.tensor(labels, dtype=torch.long)

    train = split(record["small_train"])
    validation = split(record["splits"]["validation"])
    test = split(record["splits"]["test"]) if include_test else None
    return train, validation, test
