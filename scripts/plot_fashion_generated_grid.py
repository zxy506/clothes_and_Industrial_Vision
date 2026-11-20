"""Display eight fixed, labeled CVAE samples per Fashion class."""

import os
from pathlib import Path

Path("logs/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


pool = torch.load("generated/fashion/pool_seed0.pt", map_location="cpu", weights_only=True)
fig, axes = plt.subplots(10, 8, figsize=(10, 12))
for class_id in range(10):
    for sample_id in range(8):
        index = class_id * 500 + sample_id
        assert int(pool["labels"][index]) == class_id
        ax = axes[class_id, sample_id]
        ax.imshow(pool["images"][index].squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    axes[class_id, 0].set_ylabel(str(class_id), rotation=0, labelpad=15)
fig.suptitle("Fashion CVAE: eight generated samples per condition")
fig.tight_layout()
fig.savefig("results/figures/fashion_cvae_all_classes.png", dpi=140)
plt.close(fig)
