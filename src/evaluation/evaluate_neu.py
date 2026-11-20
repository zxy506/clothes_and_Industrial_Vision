"""Final independent NEU-CLS test evaluation after all nine runs complete."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from src.datasets.neu_loader import load_neu
from src.models.cnn import NEUCNN


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict(model, images, device):
    predictions = []
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in images.split(64):
            predictions.append(model(batch.to(device)).argmax(1).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(predictions).numpy(), time.perf_counter() - start


def main():
    protocol = json.loads(Path("results/raw/neu_formal_protocol.json").read_text(encoding="utf-8"))
    for key, path in (("config_sha256", "configs/neu.yaml"),
                      ("split_sha256", "data/splits/neu.json"),
                      ("train_source_sha256", "src/training/train_neu_cnn.py"),
                      ("cvae_checkpoint_sha256", "checkpoints/neu/cvae_seed0.pt")):
        if protocol[key] != digest(path):
            raise ValueError(f"NEU protocol mismatch: {path}")
    for seed in protocol["seeds"]:
        for mode in protocol["modes"]:
            log = Path(f"logs/neu_cnn_{mode}_seed{seed}.jsonl")
            ckpt = Path(f"checkpoints/neu/cnn_{mode}_seed{seed}.pt")
            if not log.exists() or not ckpt.exists() or '"kind": "summary"' not in log.read_text(encoding="utf-8"):
                raise RuntimeError(f"incomplete NEU run: {mode} seed {seed}")
    # Official independent test split is accessed only after all nine runs.
    _, _, (images, labels_tensor) = load_neu(include_test=True)
    labels = labels_tensor.numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("results/raw/neu_final")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in protocol["seeds"]:
        for mode in protocol["modes"]:
            ckpt_path = Path(f"checkpoints/neu/cnn_{mode}_seed{seed}.pt")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model = NEUCNN().to(device)
            model.load_state_dict(ckpt["state_dict"])
            pred, seconds = predict(model, images, device)
            record = {
                "mode": mode, "seed": seed, "classes": ["Cr", "In", "Pa", "PS", "RS", "Sc"],
                "checkpoint_sha256": digest(ckpt_path), "selected_epoch": ckpt["best_epoch"],
                "validation_accuracy": ckpt["validation_accuracy"],
                "accuracy": float(accuracy_score(labels, pred)),
                "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
                "recall_by_class": recall_score(labels, pred, labels=list(range(6)),
                                                average=None, zero_division=0).tolist(),
                "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(6))).tolist(),
                "test_count": len(labels), "batch_inference_seconds": seconds,
            }
            out = output_dir / f"{mode}_seed{seed}.json"
            pred_out = output_dir / f"{mode}_seed{seed}_predictions.npz"
            if out.exists() or pred_out.exists():
                raise FileExistsError(f"NEU test result already exists: {mode} seed {seed}")
            out.write_text(json.dumps(record, indent=2), encoding="utf-8")
            np.savez_compressed(pred_out, labels=labels, predictions=pred)
            records.append(record)
            print(mode, seed, "accuracy", record["accuracy"],
                  "macro_f1", record["macro_f1"], flush=True)
    summary = {}
    for mode in protocol["modes"]:
        group = [r for r in records if r["mode"] == mode]
        summary[mode] = {}
        for metric in ("accuracy", "macro_f1"):
            values = [r[metric] for r in group]
            summary[mode][f"{metric}_mean"] = float(np.mean(values))
            summary[mode][f"{metric}_std"] = float(np.std(values, ddof=1))
        recalls = np.asarray([r["recall_by_class"] for r in group])
        summary[mode]["recall_mean"] = recalls.mean(0).tolist()
        summary[mode]["recall_std"] = recalls.std(0, ddof=1).tolist()
    path = Path("results/summary/neu_final.json")
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"protocol": "results/raw/neu_formal_protocol.json",
                                "summary": summary}, indent=2), encoding="utf-8")
    print("Saved NEU final summary", flush=True)


if __name__ == "__main__":
    main()
