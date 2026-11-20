"""Plot Fashion preliminary learning curves from actual JSONL records."""

import json
import os
from pathlib import Path

Path("logs/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


output = Path("results/figures")
output.mkdir(parents=True, exist_ok=True)
rows = [r for r in records("logs/fashion_cvae_seed0.jsonl") if r["kind"] == "epoch"]
fig, ax = plt.subplots(figsize=(7, 4))
for key in ("loss", "reconstruction", "kl"):
    ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], label=key)
ax.set(xlabel="Epoch", ylabel="Loss per image", title="Fashion CVAE training")
ax.legend()
fig.tight_layout()
fig.savefig(output / "fashion_cvae_training.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
summary = {}
for mode in "ABC":
    run = records(f"logs/fashion_cnn_{mode}_seed0_10epochs.jsonl")
    rows = [r for r in run if r["kind"] == "epoch"]
    ax.plot([r["epoch"] for r in rows], [r["validation_accuracy"] for r in rows], label=mode)
    best = max(rows, key=lambda r: r["validation_accuracy"])
    summary[mode] = {"best_validation_accuracy": best["validation_accuracy"],
                     "best_epoch": best["epoch"], "final_epoch": rows[-1]["epoch"],
                     "updates_per_epoch": run[0]["updates_per_epoch"], "seed": run[0]["seed"],
                     "status": "preliminary; not final test result"}
ax.set(xlabel="Epoch", ylabel="Validation accuracy", title="Fashion preliminary A/B/C")
ax.legend()
fig.tight_layout()
fig.savefig(output / "fashion_abc_preliminary.png", dpi=150)
plt.close(fig)
Path("results/summary/fashion_stage2_preliminary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(summary)
