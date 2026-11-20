"""Summarize measured training/generation costs and benchmark single-image inference."""

import json
import statistics
import time
from pathlib import Path

import torch

from src.datasets.fashion_dataset import load_fashion
from src.models.cnn import SmallCNN
from src.models.cvae_conv import FashionCVAE


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
(x_train, _), _, _ = load_fashion()
image = x_train[:1].to(device)
training = {}
for mode in ("A", "B", "C25", "C"):
    rows = []
    for seed in (0, 1, 2):
        path = Path(f"logs/fashion_cnn_{mode}_seed{seed}_30epochs.jsonl")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        row = next(r for r in records if r["kind"] == "summary")
        rows.append(row)
    training[mode] = {
        "seconds_by_seed": [r["training_seconds"] for r in rows],
        "seconds_mean": statistics.mean(r["training_seconds"] for r in rows),
        "gpu_peak_allocated_bytes_by_seed": [r["gpu_peak_allocated_bytes"] for r in rows],
    }

single = {}
for mode in ("A", "B", "C25", "C"):
    ckpt = torch.load(f"checkpoints/fashion/cnn_{mode}_seed0_30epochs.pt",
                      map_location="cpu", weights_only=True)
    model = SmallCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.inference_mode():
        for _ in range(20):
            model(image)
        sync(device)
        start = time.perf_counter()
        for _ in range(200):
            model(image)
        sync(device)
    single[mode] = (time.perf_counter() - start) / 200

cvae_records = [json.loads(line) for line in Path("logs/fashion_cvae_seed0.jsonl")
                .read_text(encoding="utf-8").splitlines()]
cvae_train_seconds = cvae_records[-1]["elapsed_seconds"]
generator_ckpt = torch.load("checkpoints/fashion/cvae_seed0.pt", map_location="cpu", weights_only=True)
generator_model = FashionCVAE(latent_dim=generator_ckpt["latent_dim"]).to(device)
generator_model.load_state_dict(generator_ckpt["state_dict"])
generator_model.eval()
condition = torch.tensor([0], device=device)
with torch.inference_mode():
    for _ in range(20):
        generator_model.sample(condition)
    sync(device)
    start = time.perf_counter()
    for _ in range(200):
        generator_model.sample(condition)
    sync(device)
cvae_single_seconds = (time.perf_counter() - start) / 200
pool_records = {str(seed): json.loads(Path(f"generated/fashion/pool_seed{seed}.json")
                                    .read_text(encoding="utf-8")) for seed in (0, 1, 2)}
report = {
    "device": str(device), "timing_scope": "model loops; excludes Python startup and data loading",
    "cnn_training": training,
    "cnn_single_image_seconds_seed0": single,
    "cvae_training_seconds_seed0": cvae_train_seconds,
    "cvae_single_image_seconds_seed0": cvae_single_seconds,
    "cvae_generation_pool": pool_records,
    "generation_single_image_estimate_note": "pool time / 5000 is batch throughput, not standalone latency",
}
path = Path("results/summary/fashion_costs.json")
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
