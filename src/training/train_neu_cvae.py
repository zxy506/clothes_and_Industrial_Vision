"""Train NEU CVAE on the fixed 90-image-per-class small training split."""

import json
import time
from pathlib import Path

import torch
import yaml

from src.datasets.neu_loader import load_neu
from src.models.cvae_conv import NEUCVAE, cvae_loss
from src.utils.seed import seed_everything


def main():
    config = yaml.safe_load(Path("configs/neu.yaml").read_text(encoding="utf-8"))
    seed = 0
    seed_everything(seed)
    (images, labels), _, _ = load_neu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NEUCVAE(latent_dim=config["cvae"]["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    beta = config["cvae"]["beta"]
    log_path = Path("logs/neu_cvae_seed0.jsonl")
    checkpoint = Path("checkpoints/neu/cvae_seed0.pt")
    if log_path.exists() or checkpoint.exists():
        raise FileExistsError("NEU CVAE run exists; refusing to overwrite")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"kind": "config", "seed": seed, "epochs": epochs,
                              "train_count": len(images), "batch_size": batch_size,
                              "latent_dim": model.latent_dim, "beta": beta,
                              "reduction": "pixel/latent sum per image, batch mean",
                              "split": "data/splits/neu.json", "device": str(device),
                              "test_used": False}) + "\n")
        for epoch in range(epochs):
            model.train()
            totals = torch.zeros(3)
            order = torch.randperm(len(images))
            for ids in order.split(batch_size):
                x, y = images[ids].to(device), labels[ids].to(device)
                output, mu, logvar = model(x, y)
                loss, rec, kl = cvae_loss(output, x, mu, logvar, beta)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                totals += torch.tensor([loss.item(), rec.item(), kl.item()]) * len(ids)
            row = {"kind": "epoch", "epoch": epoch + 1,
                   "loss": float(totals[0] / len(images)),
                   "reconstruction": float(totals[1] / len(images)),
                   "kl": float(totals[2] / len(images)),
                   "elapsed_seconds": round(time.perf_counter() - start, 3)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(row, flush=True)
        log.write(json.dumps({"kind": "summary",
                              "training_seconds": round(time.perf_counter() - start, 3),
                              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                              if device.type == "cuda" else None}) + "\n")
    torch.save({"state_dict": model.state_dict(), "model": "NEUCVAE",
                "latent_dim": model.latent_dim, "classes": 6,
                "seed": seed, "epochs": epochs,
                "train_split": "data/splits/neu.json:small_train"}, checkpoint)
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
