"""Freeze the Fashion protocol, then run all required classifier conditions."""

import hashlib
import json
from pathlib import Path

import yaml

from src.training.train_cnn import run


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


config = yaml.safe_load(Path("configs/fashion.yaml").read_text(encoding="utf-8"))
epochs = config["training"]["epochs"]
seeds = config["training"]["seeds"]
if epochs != 30 or seeds != [0, 1, 2]:
    raise ValueError("formal Fashion protocol requires 30 epochs and seeds 0,1,2")
protocol = {
    "status": "frozen_before_test_evaluation",
    "split_sha256": digest("data/splits/fashion.json"),
    "config_sha256": digest("configs/fashion.yaml"),
    "cnn_source_sha256": digest("src/models/cnn.py"),
    "train_source_sha256": digest("src/training/train_cnn.py"),
    "cvae_sha256": digest("checkpoints/fashion/cvae_seed0.pt"),
    "epochs": epochs, "seeds": seeds, "modes": ["A", "B", "C25", "C"],
    "updates_per_epoch": 40, "batch_size": 128,
    "noise_sigma": config["robustness"]["gaussian_sigma"],
    "selection": "best validation accuracy, first epoch on tie",
    "test_policy": "evaluate only after all runs complete; no test-based tuning",
}
path = Path("results/raw/fashion_formal_protocol.json")
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    if json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise FileExistsError("formal protocol has changed; refusing to overwrite")
else:
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
for seed in seeds:
    for mode in protocol["modes"]:
        log = Path(f"logs/fashion_cnn_{mode}_seed{seed}_{epochs}epochs.jsonl")
        checkpoint = Path(f"checkpoints/fashion/cnn_{mode}_seed{seed}_{epochs}epochs.pt")
        if log.exists() or checkpoint.exists():
            if not (log.exists() and checkpoint.exists() and '"kind": "summary"' in log.read_text(encoding="utf-8")):
                raise FileExistsError(f"partial run requires inspection: {mode} seed {seed}")
            print(f"Complete run already exists: {mode} seed {seed}", flush=True)
            continue
        print(f"Starting {mode} seed {seed}", flush=True)
        run(mode, epochs, seed)
print("All formal Fashion classifier runs complete", flush=True)
