"""Read-only verification of frozen experiments; writes a separate audit report.

This does not retrain models or overwrite any formal test artifact.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from src.datasets.fashion_dataset import load_fashion
from src.datasets.neu_loader import load_neu
from src.evaluation.evaluate_fashion import predict as predict_fashion
from src.evaluation.evaluate_neu import predict as predict_neu
from src.models.cnn import NEUCNN, SmallCNN


ROOT = Path(__file__).resolve().parents[1]


def digest(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def close(a, b, message: str) -> None:
    check(np.allclose(a, b, rtol=0, atol=1e-10), message)


def verify_split() -> dict:
    fashion = read_json("data/splits/fashion.json")
    ft, fv = map(set, (fashion["indices"][key] for key in ("train", "validation")))
    check(len(ft) == len(fv) == 5000 and not ft & fv, "Fashion split overlap/count")
    check(fashion["source_test_size"] == 10000, "Fashion test count")
    neu = read_json("data/splits/neu.json")
    groups = {key: set(value) for key, value in neu["splits"].items()}
    check([len(groups[key]) for key in ("train", "validation", "test")] == [1080, 360, 360],
          "NEU split counts")
    check(all(not groups[a] & groups[b] for a, b in
              (("train", "validation"), ("train", "test"), ("validation", "test"))),
          "NEU split overlap")
    small = set(neu["small_train"])
    check(len(small) == 540 and small <= groups["train"], "NEU small_train invalid")
    for key, paths in (*neu["splits"].items(), ("small_train", neu["small_train"])):
        counts = Counter(Path(path).stem.split("_")[0] for path in paths)
        expected = 90 if key == "small_train" else {"train": 180, "validation": 60, "test": 60}[key]
        check(set(counts) == set(neu["classes"]) and all(n == expected for n in counts.values()),
              f"NEU class balance: {key}")
        check(all((ROOT / "data/raw/neu_cls" / path).is_file() for path in paths),
              f"NEU missing image: {key}")
    return {"fashion_train": len(ft), "fashion_validation": len(fv),
            "fashion_test": fashion["source_test_size"],
            "neu_train": len(groups["train"]), "neu_validation": len(groups["validation"]),
            "neu_test": len(groups["test"]), "neu_small_train": len(small)}


def verify_protocol() -> dict:
    mappings = {
        "fashion": ("results/raw/fashion_formal_protocol.json", {
            "split_sha256": "data/splits/fashion.json",
            "config_sha256": "configs/fashion.yaml",
            "cnn_source_sha256": "src/models/cnn.py",
            "train_source_sha256": "src/training/train_cnn.py",
            "cvae_sha256": "checkpoints/fashion/cvae_seed0.pt",
        }),
        "neu": ("results/raw/neu_formal_protocol.json", {
            "split_sha256": "data/splits/neu.json",
            "config_sha256": "configs/neu.yaml",
            "cnn_source_sha256": "src/models/cnn.py",
            "cvae_source_sha256": "src/models/cvae_conv.py",
            "train_source_sha256": "src/training/train_neu_cnn.py",
            "cvae_checkpoint_sha256": "checkpoints/neu/cvae_seed0.pt",
        }),
    }
    status = {}
    for dataset, (file, mapping) in mappings.items():
        protocol = read_json(file)
        status[dataset] = {key: digest(path) == protocol[key] for key, path in mapping.items()}
    # The shared CNN source was extended with NEUCNN after the Fashion freeze.
    # An exact-byte snapshot of its Fashion-only version is archived separately.
    fashion_protocol = read_json("results/raw/fashion_formal_protocol.json")
    status["fashion"]["frozen_cnn_snapshot_sha256"] = (
        digest("reports/frozen_sources/fashion_cnn.py") == fashion_protocol["cnn_source_sha256"])
    check(all(value for key, value in status["fashion"].items() if key != "cnn_source_sha256"),
          "Fashion protocol asset changed")
    check(all(status["neu"].values()), "NEU protocol asset changed")
    return status


def verify_results(dataset: str, modes: tuple[str, ...], count: int, classes: int) -> dict:
    summary = read_json(f"results/summary/{dataset}_final.json")["summary"]
    raw_dir = ROOT / "results" / "raw" / f"{dataset}_final"
    checked = 0
    for mode in modes:
        records = []
        for seed in (0, 1, 2):
            stem = f"{mode}_seed{seed}"
            record = json.loads((raw_dir / f"{stem}.json").read_text(encoding="utf-8"))
            with np.load(raw_dir / f"{stem}_predictions.npz") as saved:
                y = saved["labels"]
                check(len(y) == count, f"{dataset} sample count: {stem}")
                conditions = (("clean", saved["clean_predictions"]),
                              ("noisy", saved["noisy_predictions"])) if dataset == "fashion" else (
                                  (None, saved["predictions"]),)
                for condition, pred in conditions:
                    metrics = record[condition] if condition else record
                    close(accuracy_score(y, pred), metrics["accuracy"], f"accuracy: {stem}")
                    close(f1_score(y, pred, labels=list(range(classes)), average="macro",
                                   zero_division=0), metrics["macro_f1"], f"Macro-F1: {stem}")
                    close(recall_score(y, pred, labels=list(range(classes)), average=None,
                                       zero_division=0), metrics["recall_by_class"], f"recall: {stem}")
                    check(confusion_matrix(y, pred, labels=list(range(classes))).tolist() ==
                          metrics["confusion_matrix"], f"confusion: {stem}")
            ckpt = (f"checkpoints/fashion/cnn_{stem}_30epochs.pt" if dataset == "fashion"
                    else f"checkpoints/neu/cnn_{stem}.pt")
            check(digest(ckpt) == record["checkpoint_sha256"], f"checkpoint hash: {stem}")
            log = (ROOT / "logs" / (f"fashion_cnn_{stem}_30epochs.jsonl" if dataset == "fashion"
                                    else f"neu_cnn_{stem}.jsonl"))
            check(log.is_file() and '"kind": "summary"' in log.read_text(encoding="utf-8"),
                  f"incomplete training log: {stem}")
            records.append(record)
            checked += 1
        fields = ("clean_accuracy", "clean_macro_f1", "noisy_accuracy", "noisy_macro_f1") if dataset == "fashion" else ("accuracy", "macro_f1")
        for field in fields:
            if dataset == "fashion":
                condition, metric = field.split("_", 1)
                values = [r[condition][metric] for r in records]
            else:
                values = [r[field] for r in records]
            close(np.mean(values), summary[mode][f"{field}_mean"], f"summary mean: {mode}/{field}")
            close(np.std(values, ddof=1), summary[mode][f"{field}_std"],
                  f"summary std: {mode}/{field}")
    return {"runs_checked": checked, "test_images_per_run": count, "modes": list(modes)}


def rerun_inference() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, (fashion_x, fashion_y) = load_fashion(include_test=True)
    rng = torch.Generator().manual_seed(20261119)
    fashion_noisy = (fashion_x + 0.15 * torch.randn(fashion_x.shape, generator=rng)).clamp(0, 1)
    fashion_payload = torch.load(ROOT / "checkpoints/fashion/cnn_B_seed0_30epochs.pt",
                                 map_location="cpu", weights_only=True)
    fashion_model = SmallCNN().to(device)
    fashion_model.load_state_dict(fashion_payload["state_dict"])
    fashion_clean_pred, _ = predict_fashion(fashion_model, fashion_x, device)
    fashion_noisy_pred, _ = predict_fashion(fashion_model, fashion_noisy, device)
    with np.load(ROOT / "results/raw/fashion_final/B_seed0_predictions.npz") as saved:
        check(np.array_equal(saved["labels"], fashion_y.numpy()), "Fashion labels changed")
        check(np.array_equal(saved["clean_predictions"], fashion_clean_pred),
              "Fashion clean inference changed")
        check(np.array_equal(saved["noisy_predictions"], fashion_noisy_pred),
              "Fashion noisy inference changed")

    _, _, (neu_x, neu_y) = load_neu(include_test=True)
    neu_payload = torch.load(ROOT / "checkpoints/neu/cnn_B_seed0.pt",
                             map_location="cpu", weights_only=True)
    neu_model = NEUCNN().to(device)
    neu_model.load_state_dict(neu_payload["state_dict"])
    neu_pred, _ = predict_neu(neu_model, neu_x, device)
    with np.load(ROOT / "results/raw/neu_final/B_seed0_predictions.npz") as saved:
        check(np.array_equal(saved["labels"], neu_y.numpy()), "NEU labels changed")
        check(np.array_equal(saved["predictions"], neu_pred), "NEU inference changed")
    return {"device": str(device), "fashion_B_seed0_clean_exact_match": True,
            "fashion_B_seed0_noisy_exact_match": True, "neu_B_seed0_exact_match": True}


def main() -> None:
    report = {"split": verify_split(), "protocol_hash_matches": verify_protocol(),
              "fashion": verify_results("fashion", ("A", "B", "C25", "C"), 10000, 10),
              "neu": verify_results("neu", ("A", "B", "C"), 360, 6),
              "rerun_inference": rerun_inference()}
    path = ROOT / "reports/final_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
