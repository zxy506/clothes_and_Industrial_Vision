"""Create figures and failure records directly from frozen test outputs."""

import json
import os
from pathlib import Path

Path("logs/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import FashionMNIST


root = Path("results/figures")
root.mkdir(parents=True, exist_ok=True)
summary = json.loads(Path("results/summary/fashion_final.json").read_text(encoding="utf-8"))["summary"]
modes = ["A", "B", "C25", "C"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
    ax.bar(modes, [summary[m][f"clean_{metric}_mean"] for m in modes],
           yerr=[summary[m][f"clean_{metric}_std"] for m in modes], capsize=4)
    ax.set_ylim(0.75, 0.9)
    ax.set_title(title)
    ax.set_ylabel("Official test score")
fig.tight_layout()
fig.savefig(root / "fashion_final_abc.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
for metric in ("accuracy", "macro_f1"):
    ax.errorbar([0, 25, 50],
                [summary[m][f"clean_{metric}_mean"] for m in ("A", "C25", "C")],
                yerr=[summary[m][f"clean_{metric}_std"] for m in ("A", "C25", "C")],
                marker="o", capsize=4, label=metric)
ax.set(xlabel="Generated proportion (%)", ylabel="Official test score",
       title="Fashion CVAE proportion ablation")
ax.set_xticks([0, 25, 50])
ax.legend()
fig.tight_layout()
fig.savefig(root / "fashion_final_ablation.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, mode in zip(axes, ("A", "B", "C")):
    matrix = sum(np.asarray(json.loads(Path(f"results/raw/fashion_final/{mode}_seed{s}.json")
                                       .read_text(encoding="utf-8"))["clean"]["confusion_matrix"])
                 for s in (0, 1, 2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(title=mode, xlabel="Predicted", ylabel="True",
           xticks=range(10), yticks=range(10))
fig.colorbar(image, ax=axes, shrink=0.7)
fig.savefig(root / "fashion_final_confusions.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
for mode in ("A", "B", "C25", "C"):
    ax.plot(range(10), summary[mode]["clean_recall_mean"], marker="o", label=mode)
ax.set(xlabel="Fashion-MNIST class index", ylabel="Recall (three-seed mean)",
       title="Fashion class recall", xticks=range(10), ylim=(0, 1))
ax.legend()
fig.tight_layout()
fig.savefig(root / "fashion_final_recall.png", dpi=150)
plt.close(fig)

test = FashionMNIST("data/raw/fashion_mnist", train=False, download=False)
a = np.load("results/raw/fashion_final/A_seed0_predictions.npz")
c = np.load("results/raw/fashion_final/C_seed0_predictions.npz")
labels = a["labels"]
cases = []
fig, axes = plt.subplots(1, 3, figsize=(8, 3))
for ax, class_id in zip(axes, (2, 4, 5)):
    candidates = np.flatnonzero((labels == class_id) &
                                (a["clean_predictions"] == class_id) &
                                (c["clean_predictions"] != class_id))
    if len(candidates) == 0:
        raise RuntimeError(f"no comparative failure for class {class_id}")
    index = int(candidates[0])
    ax.imshow(test.data[index], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"true {class_id}; A {class_id}; C {int(c['clean_predictions'][index])}")
    ax.axis("off")
    cases.append({"official_test_index": index, "true_label": class_id,
                  "A_seed0_prediction": class_id,
                  "C_seed0_prediction": int(c["clean_predictions"][index]),
                  "selection_rule": "first test index where A is correct and C is wrong"})
fig.tight_layout()
fig.savefig(root / "fashion_final_failures.png", dpi=150)
plt.close(fig)
path = Path("results/raw/fashion_final_failures.json")
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
print(cases)
