"""Freeze NEU settings, then train A/B/C with three classifier seeds."""

import hashlib
import json
from pathlib import Path

import yaml

from src.training.train_neu_cnn import run


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


config = yaml.safe_load(Path("configs/neu.yaml").read_text(encoding="utf-8"))
if config["training"]["seeds"] != [0, 1, 2] or config["training"]["epochs"] != 40:
    raise ValueError("unexpected NEU formal protocol")
protocol = {
    "status": "frozen_before_test_evaluation",
    "config_sha256": digest("configs/neu.yaml"),
    "split_sha256": digest("data/splits/neu.json"),
    "cnn_source_sha256": digest("src/models/cnn.py"),
    "cvae_source_sha256": digest("src/models/cvae_conv.py"),
    "train_source_sha256": digest("src/training/train_neu_cnn.py"),
    "cvae_checkpoint_sha256": digest("checkpoints/neu/cvae_seed0.pt"),
    "small_train_per_class": 90, "validation_per_class": 60,
    "test_per_class": 60, "epochs": 40, "updates_per_epoch": 17,
    "batch_size": 32, "modes": ["A", "B", "C"], "seeds": [0, 1, 2],
    "selection": "best validation accuracy, first epoch on tie",
    "test_policy": "evaluate only after nine runs complete; no test-based tuning",
}
path = Path("results/raw/neu_formal_protocol.json")
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    if json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise FileExistsError("NEU protocol changed; refusing to overwrite")
else:
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
for seed in protocol["seeds"]:
    for mode in protocol["modes"]:
        log = Path(f"logs/neu_cnn_{mode}_seed{seed}.jsonl")
        checkpoint = Path(f"checkpoints/neu/cnn_{mode}_seed{seed}.pt")
        if log.exists() or checkpoint.exists():
            if not (log.exists() and checkpoint.exists() and '"kind": "summary"' in log.read_text(encoding="utf-8")):
                raise FileExistsError(f"partial run requires inspection: {mode} seed {seed}")
            print(f"Complete run already exists: {mode} seed {seed}", flush=True)
            continue
        print(f"Starting {mode} seed {seed}", flush=True)
        run(mode, seed)
print("All formal NEU classifier runs complete", flush=True)
