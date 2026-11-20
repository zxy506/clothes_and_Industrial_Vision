"""Preview training-only NEU affine/flip settings before classifier runs."""

import os
from pathlib import Path

Path("logs/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torchvision.transforms import RandomAffine

from src.datasets.neu_loader import load_neu


(images, labels), _, _ = load_neu()
torch.manual_seed(0)
config = yaml.safe_load(Path("configs/neu.yaml").read_text(encoding="utf-8"))
augment = RandomAffine(config["augmentation"]["rotation_degrees"],
                       fill=config["augmentation"]["affine_fill"])
fig, axes = plt.subplots(6, 2, figsize=(5, 12))
names = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
for class_id, name in enumerate(names):
    index = int(torch.nonzero(labels == class_id)[0])
    image = images[index]
    transformed = augment(image)
    if torch.rand(()) < 0.5:
        transformed = torch.flip(transformed, dims=(-1,))
    for col, data in enumerate((image, transformed)):
        axes[class_id, col].imshow(data.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[class_id, col].axis("off")
    axes[class_id, 0].set_ylabel(name, rotation=0, labelpad=18)
axes[0, 0].set_title("Real small train")
axes[0, 1].set_title("Traditional augmentation")
fig.tight_layout()
fig.savefig("results/figures/neu_augmentation_preview_revised.png", dpi=130)
plt.close(fig)
