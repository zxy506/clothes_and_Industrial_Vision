"""Train Fashion CVAE using only the fixed 5,000-image training split."""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from src.datasets.fashion_dataset import load_fashion
from src.models.cvae_conv import FashionCVAE, cvae_loss
from src.utils.seed import seed_everything


def run(epochs: int, seed: int):
    config = yaml.safe_load(Path("configs/fashion.yaml").read_text(encoding="utf-8"))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (x_train, y_train), _, _ = load_fashion()
    model = FashionCVAE(latent_dim=config["cvae"]["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    batch_size = config["training"]["batch_size"]
    beta = config["cvae"]["beta"]
    log_path = Path(f"logs/fashion_cvae_seed{seed}.jsonl")
    checkpoint = Path(f"checkpoints/fashion/cvae_seed{seed}.pt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() or checkpoint.exists():
        raise FileExistsError("existing CVAE run; refusing to overwrite")
    start = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"kind": "config", "seed": seed, "epochs": epochs,
                              "batch_size": batch_size, "beta": beta,
                              "latent_dim": model.latent_dim,
                              "reduction": "sum per image, mean batch",
                              "train_count": len(x_train), "device": str(device)}) + "\n")
        for epoch in range(epochs):
            model.train()
            totals = torch.zeros(3)
            order = torch.randperm(len(x_train))
            for ids in order.split(batch_size):
                x, y = x_train[ids].to(device), y_train[ids].to(device)
                reconstruction, mu, logvar = model(x, y)
                loss, rec, kl = cvae_loss(reconstruction, x, mu, logvar, beta)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                totals += torch.tensor([loss.item(), rec.item(), kl.item()]) * len(ids)
            row = {"kind": "epoch", "epoch": epoch + 1,
                   "loss": float(totals[0] / len(x_train)),
                   "reconstruction": float(totals[1] / len(x_train)),
                   "kl": float(totals[2] / len(x_train)),
                   "elapsed_seconds": round(time.perf_counter() - start, 2)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
    torch.save({"state_dict": model.state_dict(), "model": "FashionCVAE",
                "seed": seed, "epochs": epochs, "latent_dim": model.latent_dim,
                "train_split": "data/splits/fashion.json"}, checkpoint)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.epochs, args.seed)
