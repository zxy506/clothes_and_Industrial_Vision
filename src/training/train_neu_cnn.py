"""NEU small-sample A/B/C runs with identical classifier budgets."""

import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torchvision.transforms import RandomAffine

from src.datasets.neu_loader import load_neu
from src.models.cnn import NEUCNN
from src.models.cvae_conv import NEUCVAE
from src.training.train_cnn import evaluate
from src.utils.seed import seed_everything


def generated_pool(seed, device):
    checkpoint = Path("checkpoints/neu/cvae_seed0.pt")
    version = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path = Path(f"generated/neu/pool_seed{seed}.pt")
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["generator_sha256"] != version:
            raise ValueError("NEU generated pool has different CVAE version")
        return payload["images"], payload["labels"]
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = NEUCVAE(latent_dim=ckpt["latent_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    labels = torch.arange(6).repeat_interleave(90)
    rng = torch.Generator(device=device).manual_seed(seed)
    start = time.perf_counter()
    images = torch.cat([model.sample(batch.to(device), rng).cpu() for batch in labels.split(32)])
    elapsed = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels, "seed": seed,
                "generator_sha256": version, "generation_seconds": elapsed}, path)
    path.with_suffix(".json").write_text(json.dumps({"images": len(images),
        "labels_per_class": 90, "seed": seed, "generator_sha256": version,
        "generation_seconds": elapsed}, indent=2), encoding="utf-8")
    return images, labels


def run(mode, seed):
    if mode not in {"A", "B", "C"}:
        raise ValueError(mode)
    config = yaml.safe_load(Path("configs/neu.yaml").read_text(encoding="utf-8"))
    seed_everything(seed)
    sampling_rng = torch.Generator().manual_seed(seed + 30000)
    pool_rng = torch.Generator().manual_seed(seed + 40000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (x_train, y_train), (x_val, y_val), _ = load_neu()
    model = NEUCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    updates = math.ceil(len(x_train) / batch_size)
    augmentation = RandomAffine(config["augmentation"]["rotation_degrees"],
                                fill=config["augmentation"]["affine_fill"])
    pool_images = pool_labels = None
    if mode == "C":
        pool_images, pool_labels = generated_pool(seed, device)
    log_path = Path(f"logs/neu_cnn_{mode}_seed{seed}.jsonl")
    checkpoint = Path(f"checkpoints/neu/cnn_{mode}_seed{seed}.pt")
    if log_path.exists() or checkpoint.exists():
        raise FileExistsError(f"existing NEU run: {mode} seed {seed}")
    best = -1.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"kind": "config", "mode": mode, "seed": seed,
                              "train_count": len(x_train), "validation_count": len(x_val),
                              "epochs": epochs, "batch_size": batch_size,
                              "updates_per_epoch": updates, "learning_rate": config["training"]["learning_rate"],
                              "synthetic_fraction": 0 if mode == "A" else 0.5,
                              "sampling_seed": seed + 30000,
                              "split": "data/splits/neu.json:small_train", "test_used": False,
                              "device": str(device)}) + "\n")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for _ in range(updates):
                if mode == "A":
                    ids = torch.randint(len(x_train), (batch_size,), generator=sampling_rng)
                    x, y = x_train[ids], y_train[ids]
                else:
                    real_ids = torch.randint(len(x_train), (batch_size // 2,), generator=sampling_rng)
                    source_ids = torch.randint(len(x_train), (batch_size // 2,), generator=sampling_rng)
                    labels = y_train[source_ids]
                    if mode == "B":
                        augmented = augmentation(x_train[source_ids])
                        if torch.rand(()) < config["augmentation"]["horizontal_flip_probability"]:
                            augmented = torch.flip(augmented, dims=(-1,))
                    else:
                        pool_ids = labels * 90 + torch.randint(90, (len(labels),), generator=pool_rng)
                        augmented = pool_images[pool_ids]
                        assert torch.equal(pool_labels[pool_ids], labels)
                    x = torch.cat((x_train[real_ids], augmented))
                    y = torch.cat((y_train[real_ids], labels))
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            val = evaluate(model, x_val, y_val, device, batch_size=64)
            row = {"kind": "epoch", "epoch": epoch + 1,
                   "train_loss": train_loss / updates,
                   "validation_loss": val["loss"], "validation_accuracy": val["accuracy"],
                   "elapsed_seconds": round(time.perf_counter() - start, 3)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(mode, seed, row, flush=True)
            if val["accuracy"] > best:
                best = val["accuracy"]
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "model": "NEUCNN",
                            "mode": mode, "seed": seed, "best_epoch": epoch + 1,
                            "validation_accuracy": best,
                            "split": "data/splits/neu.json:small_train"}, checkpoint)
        log.write(json.dumps({"kind": "summary", "best_validation_accuracy": best,
                              "training_seconds": round(time.perf_counter() - start, 3),
                              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                              if device.type == "cuda" else None}) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["A", "B", "C"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.mode, args.seed)
