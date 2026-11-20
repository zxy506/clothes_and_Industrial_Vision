"""Build immutable, stratified index manifests from original dataset labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stratified_indices(labels, *, classes, counts, seed):
    """Return disjoint per-split indices, with a separate deterministic RNG per class."""
    labels = np.asarray(labels)
    result = {name: [] for name in counts}
    requested = sum(counts.values())
    for class_id in range(classes):
        available = np.flatnonzero(labels == class_id)
        if len(available) < requested:
            raise ValueError(f"class {class_id}: expected {requested}, found {len(available)}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, class_id]))
        chosen = rng.permutation(available)[:requested]
        offset = 0
        for name, size in counts.items():
            result[name].extend(int(i) for i in chosen[offset:offset + size])
            offset += size
    all_indices = [i for split in result.values() for i in split]
    if len(all_indices) != len(set(all_indices)):
        raise AssertionError("split overlap detected")
    return result


def build_fashion(root: Path, output: Path, seed: int = 20261119):
    from torchvision.datasets import FashionMNIST

    train = FashionMNIST(root=root, train=True, download=True)
    test = FashionMNIST(root=root, train=False, download=True)
    if len(train) != 60000 or len(test) != 10000:
        raise ValueError("unexpected Fashion-MNIST size")
    indices = stratified_indices(train.targets, classes=10,
                                 counts={"train": 500, "validation": 500}, seed=seed)
    labels = np.asarray(train.targets)
    manifest = {
        "dataset": "Fashion-MNIST", "source": "torchvision.datasets.FashionMNIST",
        "split_seed": seed, "source_train_size": len(train),
        "source_test_size": len(test), "test": "official_test_full",
        "indices": indices,
        "class_counts": {name: {str(c): int(np.sum(labels[ids] == c)) for c in range(10)}
                         for name, ids in indices.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        if old != manifest:
            raise FileExistsError(f"refusing to overwrite different split: {output}")
        return manifest
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/fashion_mnist"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/fashion.json"))
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    result = build_fashion(args.root, args.output, args.seed)
    print(f"Fashion split: {len(result['indices']['train'])} train, "
          f"{len(result['indices']['validation'])} validation, 10000 official test")
