"""Fixed-update Fashion CNN runs for real, traditional, and CVAE mixtures."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torchvision.transforms import RandomAffine

from src.datasets.fashion_dataset import load_fashion
from src.models.cnn import SmallCNN
from src.models.cvae_conv import FashionCVAE
from src.utils.seed import seed_everything


@torch.no_grad()
def evaluate(model, x, y, device, batch_size=256):
    model.eval()
    total_loss = correct = 0
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device)
        yb = y[start:start + batch_size].to(device)
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += int((logits.argmax(1) == yb).sum().item())
    return {"loss": total_loss / len(x), "accuracy": correct / len(x)}


def generated_pool(checkpoint_path, seed, device):
    path = Path(f"generated/fashion/pool_seed{seed}.pt")
    checkpoint_bytes = checkpoint_path.read_bytes()
    version = hashlib.sha256(checkpoint_bytes).hexdigest()
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["generator_sha256"] != version:
            raise ValueError("generated pool uses a different CVAE checkpoint")
        return payload["images"], payload["labels"]
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = FashionCVAE(latent_dim=ckpt["latent_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    labels = torch.arange(10).repeat_interleave(500)
    rng = torch.Generator(device=device).manual_seed(seed)
    t0 = time.perf_counter()
    images = torch.cat([model.sample(batch.to(device), rng).cpu() for batch in labels.split(128)])
    elapsed = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels,
                "generator_sha256": version, "sample_seed": seed,
                "generation_seconds": elapsed}, path)
    Path(f"generated/fashion/pool_seed{seed}.json").write_text(
        json.dumps({"images": len(images), "labels_per_class": 500,
                    "generator_sha256": version, "sample_seed": seed,
                    "generation_seconds": elapsed}, indent=2), encoding="utf-8")
    return images, labels


def run(mode: str, epochs: int, seed: int):
    config = yaml.safe_load(Path("configs/fashion.yaml").read_text(encoding="utf-8"))
    seed_everything(seed)
    sampling_rng = torch.Generator().manual_seed(seed + 10000)
    pool_rng = torch.Generator().manual_seed(seed + 20000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (x_train, y_train), (x_val, y_val), _ = load_fashion()  # test remains unopened by training
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    batch_size = config["training"]["batch_size"]
    updates_per_epoch = math.ceil(len(x_train) / batch_size)
    augmentation = RandomAffine(config["augmentation"]["rotation_degrees"],
                                translate=(config["augmentation"]["translation_fraction"],) * 2,
                                scale=tuple(config["augmentation"]["scale_range"]))
    pool_images = pool_labels = None
    if mode in {"C", "C25"}:
        pool_images, pool_labels = generated_pool(Path("checkpoints/fashion/cvae_seed0.pt"), seed, device)
    log_path = Path(f"logs/fashion_cnn_{mode}_seed{seed}_{epochs}epochs.jsonl")
    checkpoint = Path(f"checkpoints/fashion/cnn_{mode}_seed{seed}_{epochs}epochs.pt")
    if log_path.exists() or checkpoint.exists():
        raise FileExistsError("existing CNN run; refusing to overwrite")
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_accuracy = -1.0
    synthetic_fraction = {"A": 0.0, "B": 0.5, "C": 0.5, "C25": 0.25}[mode]
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"kind": "config", "mode": mode, "seed": seed,
                              "epochs": epochs, "batch_size": batch_size,
                              "updates_per_epoch": updates_per_epoch,
                              "learning_rate": config["training"]["learning_rate"],
                              "train_count": len(x_train), "validation_count": len(x_val),
                              "synthetic_fraction": synthetic_fraction,
                              "sampling_seed": seed + 10000,
                              "device": str(device), "test_used": False}) + "\n")
        for epoch in range(epochs):
            model.train()
            loss_sum = 0.0
            for _ in range(updates_per_epoch):
                if mode == "A":
                    real_ids = torch.randint(len(x_train), (batch_size,), generator=sampling_rng)
                    xb, yb = x_train[real_ids], y_train[real_ids]
                else:
                    synthetic_count = int(batch_size * synthetic_fraction)
                    real_ids = torch.randint(len(x_train), (batch_size - synthetic_count,), generator=sampling_rng)
                    source_ids = torch.randint(len(x_train), (synthetic_count,), generator=sampling_rng)
                    if mode == "B":
                        augmented = augmentation(x_train[source_ids])
                        labels = y_train[source_ids]
                    else:
                        labels = y_train[source_ids]
                        # Match the source class distribution for B and C.
                        pool_ids = labels * 500 + torch.randint(500, (len(labels),), generator=pool_rng)
                        augmented = pool_images[pool_ids]
                        assert torch.equal(pool_labels[pool_ids], labels)
                    xb = torch.cat((x_train[real_ids], augmented))
                    yb = torch.cat((y_train[real_ids], labels))
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            val = evaluate(model, x_val, y_val, device)
            row = {"kind": "epoch", "epoch": epoch + 1,
                   "train_loss": loss_sum / updates_per_epoch,
                   "validation_loss": val["loss"], "validation_accuracy": val["accuracy"],
                   "elapsed_seconds": round(time.perf_counter() - start, 2)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
            if val["accuracy"] > best_accuracy:
                best_accuracy = val["accuracy"]
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "model": "SmallCNN",
                            "mode": mode, "seed": seed, "epochs": epochs,
                            "best_epoch": epoch + 1, "validation_accuracy": best_accuracy,
                            "split": "data/splits/fashion.json"}, checkpoint)
        log.write(json.dumps({"kind": "summary", "best_validation_accuracy": best_accuracy,
                              "training_seconds": round(time.perf_counter() - start, 3),
                              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                              if device.type == "cuda" else None}) + "\n")
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["A", "B", "C", "C25"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.mode, args.epochs, args.seed)
