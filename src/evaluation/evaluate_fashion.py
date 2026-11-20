"""One-time final Fashion-MNIST test evaluation for the frozen formal protocol."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from src.datasets.fashion_dataset import load_fashion
from src.models.cnn import SmallCNN


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict(model, images, device, batch_size=256):
    output = []
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in images.split(batch_size):
            output.append(model(batch.to(device)).argmax(1).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(output).numpy(), time.perf_counter() - start


def metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "recall_by_class": recall_score(labels, predictions, labels=list(range(10)),
                                         average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(10))).tolist(),
    }


def main():
    protocol_path = Path("results/raw/fashion_formal_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["split_sha256"] != sha256("data/splits/fashion.json"):
        raise ValueError("split changed after protocol freeze")
    if protocol["config_sha256"] != sha256("configs/fashion.yaml"):
        raise ValueError("config changed after protocol freeze")
    if protocol["train_source_sha256"] != sha256("src/training/train_cnn.py"):
        raise ValueError("training source changed after protocol freeze")
    for seed in protocol["seeds"]:
        for mode in protocol["modes"]:
            path = Path(f"logs/fashion_cnn_{mode}_seed{seed}_{protocol['epochs']}epochs.jsonl")
            ckpt = Path(f"checkpoints/fashion/cnn_{mode}_seed{seed}_{protocol['epochs']}epochs.pt")
            if not path.exists() or not ckpt.exists() or '"kind": "summary"' not in path.read_text(encoding="utf-8"):
                raise RuntimeError(f"formal run incomplete: {mode} seed {seed}")
    # Open the official test set only after every classifier run is complete.
    _, _, test = load_fashion(include_test=True)
    images, labels_tensor = test
    labels = labels_tensor.numpy()
    rng = torch.Generator().manual_seed(20261119)
    noisy = (images + protocol["noise_sigma"] * torch.randn(images.shape, generator=rng)).clamp(0, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("results/raw/fashion_final")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in protocol["seeds"]:
        for mode in protocol["modes"]:
            ckpt_path = Path(f"checkpoints/fashion/cnn_{mode}_seed{seed}_{protocol['epochs']}epochs.pt")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model = SmallCNN().to(device)
            model.load_state_dict(ckpt["state_dict"])
            clean_pred, clean_seconds = predict(model, images, device)
            noisy_pred, noisy_seconds = predict(model, noisy, device)
            record = {
                "mode": mode, "seed": seed, "checkpoint_sha256": sha256(ckpt_path),
                "selected_epoch": ckpt["best_epoch"],
                "validation_accuracy": ckpt["validation_accuracy"],
                "clean": metrics(labels, clean_pred), "noisy": metrics(labels, noisy_pred),
                "noise_sigma": protocol["noise_sigma"],
                "clean_batch_inference_seconds": clean_seconds,
                "noisy_batch_inference_seconds": noisy_seconds,
                "test_count": len(labels),
            }
            result_path = output_dir / f"{mode}_seed{seed}.json"
            pred_path = output_dir / f"{mode}_seed{seed}_predictions.npz"
            if result_path.exists() or pred_path.exists():
                raise FileExistsError(f"test result already exists: {mode} seed {seed}")
            result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            np.savez_compressed(pred_path, labels=labels, clean_predictions=clean_pred,
                                noisy_predictions=noisy_pred)
            results.append(record)
            print(mode, seed, "accuracy", record["clean"]["accuracy"],
                  "macro_f1", record["clean"]["macro_f1"], flush=True)
    summary = {}
    for mode in protocol["modes"]:
        group = [r for r in results if r["mode"] == mode]
        summary[mode] = {}
        for condition in ("clean", "noisy"):
            for metric in ("accuracy", "macro_f1"):
                values = [r[condition][metric] for r in group]
                summary[mode][f"{condition}_{metric}_mean"] = float(np.mean(values))
                summary[mode][f"{condition}_{metric}_std"] = float(np.std(values, ddof=1))
            recalls = np.asarray([r[condition]["recall_by_class"] for r in group])
            summary[mode][f"{condition}_recall_mean"] = recalls.mean(0).tolist()
            summary[mode][f"{condition}_recall_std"] = recalls.std(0, ddof=1).tolist()
    summary_path = Path("results/summary/fashion_final.json")
    if summary_path.exists():
        raise FileExistsError(f"summary already exists: {summary_path}")
    summary_path.write_text(json.dumps({"protocol": str(protocol_path), "summary": summary}, indent=2),
                            encoding="utf-8")
    print("Saved final Fashion test summary", flush=True)


if __name__ == "__main__":
    main()
