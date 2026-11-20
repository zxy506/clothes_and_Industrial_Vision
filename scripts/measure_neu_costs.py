"""Summarize NEU training/generation cost and single-image model latency."""

import json
import statistics
import time
from pathlib import Path

import torch

from src.datasets.neu_loader import load_neu
from src.models.cnn import NEUCNN
from src.models.cvae_conv import NEUCVAE


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
(images, _), _, _ = load_neu()
image = images[:1].to(device)
training = {}
latencies = {}
for mode in "ABC":
    rows = []
    for seed in (0, 1, 2):
        log = Path(f"logs/neu_cnn_{mode}_seed{seed}.jsonl")
        entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        rows.append(next(r for r in entries if r["kind"] == "summary"))
    training[mode] = {
        "seconds_by_seed": [r["training_seconds"] for r in rows],
        "seconds_mean": statistics.mean(r["training_seconds"] for r in rows),
        "gpu_peak_allocated_bytes_by_seed": [r["gpu_peak_allocated_bytes"] for r in rows],
    }
    checkpoint = torch.load(f"checkpoints/neu/cnn_{mode}_seed0.pt",
                            map_location="cpu", weights_only=True)
    model = NEUCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        for _ in range(20):
            model(image)
        sync(device)
        start = time.perf_counter()
        for _ in range(200):
            model(image)
        sync(device)
    latencies[mode] = (time.perf_counter() - start) / 200

cvae_log = [json.loads(line) for line in Path("logs/neu_cvae_seed0.jsonl")
            .read_text(encoding="utf-8").splitlines()]
checkpoint = torch.load("checkpoints/neu/cvae_seed0.pt", map_location="cpu", weights_only=True)
model = NEUCVAE(latent_dim=checkpoint["latent_dim"]).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
condition = torch.tensor([0], device=device)
with torch.inference_mode():
    for _ in range(20):
        model.sample(condition)
    sync(device)
    start = time.perf_counter()
    for _ in range(200):
        model.sample(condition)
    sync(device)
generation_single_seconds = (time.perf_counter() - start) / 200
pool_records = {str(seed): json.loads(Path(f"generated/neu/pool_seed{seed}.json")
                                    .read_text(encoding="utf-8")) for seed in (0, 1, 2)}
report = {
    "device": str(device), "timing_scope": "model loops; excludes process startup and data loading",
    "cnn_training": training, "cnn_single_image_seconds_seed0": latencies,
    "cvae_training_seconds": cvae_log[-1]["training_seconds"],
    "cvae_gpu_peak_allocated_bytes": cvae_log[-1]["gpu_peak_allocated_bytes"],
    "cvae_single_image_seconds": generation_single_seconds,
    "cvae_generation_pools": pool_records,
}
path = Path("results/summary/neu_costs.json")
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
