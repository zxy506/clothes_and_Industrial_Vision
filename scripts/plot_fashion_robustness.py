"""Plot fixed-noise robustness from three-seed test summaries."""

import json
import os
from pathlib import Path

Path("logs/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


d = json.loads(Path("results/summary/fashion_final.json").read_text(encoding="utf-8"))["summary"]
modes = ["A", "B", "C25", "C"]
x = np.arange(len(modes))
fig, ax = plt.subplots(figsize=(7, 4))
for offset, condition in ((-0.18, "clean"), (0.18, "noisy")):
    ax.bar(x + offset, [d[m][f"{condition}_accuracy_mean"] for m in modes], width=0.34,
           yerr=[d[m][f"{condition}_accuracy_std"] for m in modes], capsize=3,
           label=condition)
ax.set(xticks=x, xticklabels=modes, ylabel="Accuracy", ylim=(0, 1),
       title="Fashion: clean vs fixed Gaussian noise (sigma=0.15)")
ax.legend()
fig.tight_layout()
fig.savefig("results/figures/fashion_final_robustness.png", dpi=150)
plt.close(fig)
