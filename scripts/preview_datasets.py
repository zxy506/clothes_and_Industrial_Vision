"""Save reproducible training-only sample grids for data inspection."""

import json
import os
from pathlib import Path

Path("logs/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import FashionMNIST


def save_grid(images, labels, output, title):
    fig, axes = plt.subplots(1, len(images), figsize=(1.8 * len(images), 2.4))
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


fashion = json.loads(Path("data/splits/fashion.json").read_text(encoding="utf-8"))
dataset = FashionMNIST("data/raw/fashion_mnist", train=True, download=False)
images = []
for class_id in range(10):
    index = next(i for i in fashion["indices"]["train"] if int(dataset.targets[i]) == class_id)
    images.append(dataset.data[index].numpy())
save_grid(images, [str(i) for i in range(10)], Path("results/figures/fashion_train_samples.png"),
          "Fashion-MNIST training samples")

neu = json.loads(Path("data/splits/neu.json").read_text(encoding="utf-8"))
images = []
for label in neu["classes"]:
    relative = next(p for p in neu["small_train"] if Path(p).stem.lower().startswith(label.lower() + "_"))
    with Image.open(Path("data/raw/neu_cls") / relative) as image:
        images.append(image.convert("L").copy())
save_grid(images, neu["classes"], Path("results/figures/neu_train_samples.png"),
          "NEU-CLS small training samples")
print("Saved training-only sample grids to results/figures/")
