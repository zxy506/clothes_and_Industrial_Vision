"""Show condition labels, real training examples and nearest training neighbors."""

import json
import os
from pathlib import Path

Path("logs/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.datasets.fashion_dataset import load_fashion


pool = torch.load("generated/fashion/pool_seed0.pt", map_location="cpu", weights_only=True)
(x_train, y_train), _, _ = load_fashion()
fig, axes = plt.subplots(10, 3, figsize=(5.5, 18))
rows = []
for class_id in range(10):
    generated_id = class_id * 500
    image = pool["images"][generated_id]
    same_class = torch.nonzero(y_train == class_id).flatten()
    real_id = int(same_class[0])
    distances = torch.cdist(image.flatten(1), x_train.flatten(1)).squeeze(0)
    nearest_id = int(distances.argmin())
    for column, data in enumerate((x_train[real_id], image, x_train[nearest_id])):
        axes[class_id, column].imshow(data.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[class_id, column].axis("off")
    axes[class_id, 0].set_ylabel(f"class {class_id}", rotation=0, labelpad=27)
    rows.append({"condition_label": class_id, "generated_pool_index": generated_id,
                 "real_train_index": real_id, "nearest_train_index": nearest_id,
                 "nearest_train_label": int(y_train[nearest_id]),
                 "nearest_l2": float(distances[nearest_id])})
for column, title in enumerate(("Real train", "CVAE generated", "Nearest train")):
    axes[0, column].set_title(title)
fig.tight_layout()
output = Path("results/figures/fashion_cvae_diagnostic.png")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=130)
plt.close(fig)
Path("results/raw/fashion_cvae_nearest.json").write_text(
    json.dumps({"generator_sha256": pool["generator_sha256"], "rows": rows}, indent=2),
    encoding="utf-8")
print(f"Saved {output} and nearest-neighbor records")
