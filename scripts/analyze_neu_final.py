"""Plot NEU A/B/C test results and save representative C errors."""

import json
import os
from pathlib import Path

Path("logs/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


summary = json.loads(Path("results/summary/neu_final.json").read_text(encoding="utf-8"))["summary"]
classes = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
root = Path("results/figures")
root.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
    ax.bar(["A", "B", "C"], [summary[m][f"{metric}_mean"] for m in "ABC"],
           yerr=[summary[m][f"{metric}_std"] for m in "ABC"], capsize=4)
    ax.set(ylim=(0, 1), ylabel="NEU official test score", title=title)
fig.tight_layout()
fig.savefig(root / "neu_final_abc.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
for mode in "ABC":
    ax.plot(classes, summary[mode]["recall_mean"], marker="o", label=mode)
ax.set(ylim=(0, 1), xlabel="Defect class", ylabel="Recall (three-seed mean)",
       title="NEU class recall")
ax.legend()
fig.tight_layout()
fig.savefig(root / "neu_final_recall.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, mode in zip(axes, "ABC"):
    matrix = sum(np.asarray(json.loads(Path(f"results/raw/neu_final/{mode}_seed{s}.json")
                                       .read_text(encoding="utf-8"))["confusion_matrix"])
                 for s in (0, 1, 2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set(title=mode, xlabel="Predicted", ylabel="True",
           xticks=range(6), yticks=range(6), xticklabels=classes, yticklabels=classes)
fig.colorbar(im, ax=axes, shrink=0.7)
fig.savefig(root / "neu_final_confusions.png", dpi=150, bbox_inches="tight")
plt.close(fig)

a = np.load("results/raw/neu_final/A_seed0_predictions.npz")
c = np.load("results/raw/neu_final/C_seed0_predictions.npz")
labels = a["labels"]
manifest = json.loads(Path("data/splits/neu.json").read_text(encoding="utf-8"))
cases = []
for class_id in range(6):
    candidates = np.flatnonzero((labels == class_id) &
                                (a["predictions"] == class_id) &
                                (c["predictions"] != class_id))
    if len(candidates):
        index = int(candidates[0])
        cases.append({"test_index": index, "image": manifest["splits"]["test"][index],
                      "true": classes[class_id], "A_seed0": classes[class_id],
                      "C_seed0": classes[int(c["predictions"][index])],
                      "selection_rule": "first test index of class with A correct and C wrong"})
if len(cases) < 3:
    raise RuntimeError("fewer than three comparative NEU failures")
cases = cases[:3]
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, row in zip(axes, cases):
    with Image.open(Path("data/raw/neu_cls") / row["image"]) as image:
        ax.imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"true {row['true']}; A {row['A_seed0']}; C {row['C_seed0']}")
    ax.axis("off")
fig.tight_layout()
fig.savefig(root / "neu_final_failures.png", dpi=150)
plt.close(fig)
path = Path("results/raw/neu_final_failures.json")
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
print(cases)
