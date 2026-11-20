"""Save six-class NEU generation grids and training-only nearest neighbors."""

import json
import os
from pathlib import Path

Path("logs/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.datasets.neu_loader import load_neu
from src.training.train_neu_cnn import generated_pool


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images, labels = generated_pool(0, device)
(real, real_labels), _, _ = load_neu()
names = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
fig, axes = plt.subplots(6, 7, figsize=(12, 11))
rows = []
for class_id, name in enumerate(names):
    generated = images[class_id * 90]
    distances = torch.cdist(generated.flatten(1), real.flatten(1)).squeeze(0)
    nearest = int(distances.argmin())
    real_id = int(torch.nonzero(real_labels == class_id)[0])
    selections = [real[real_id]] + [images[class_id * 90 + j] for j in range(5)] + [real[nearest]]
    for column, data in enumerate(selections):
        axes[class_id, column].imshow(data.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[class_id, column].axis("off")
    axes[class_id, 0].set_ylabel(name, rotation=0, labelpad=18)
    rows.append({"condition": name, "generated_index": class_id * 90,
                 "nearest_train_index": nearest,
                 "nearest_train_class": names[int(real_labels[nearest])],
                 "nearest_l2": float(distances[nearest])})
for col, title in enumerate(["Real train"] + [f"Generated {i}" for i in range(5)] + ["Nearest train"]):
    axes[0, col].set_title(title)
fig.tight_layout()
fig.savefig("results/figures/neu_cvae_diagnostic.png", dpi=130)
plt.close(fig)
Path("results/raw/neu_cvae_nearest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(rows)
