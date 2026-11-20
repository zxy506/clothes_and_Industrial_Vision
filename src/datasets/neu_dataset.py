"""Validate the official NEU-CLS image-only archive and build fixed manifests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image

from .split_builder import stratified_indices


CLASSES = ("Cr", "In", "Pa", "PS", "RS", "Sc")
LABELS = {
    "Cr": ["Crazing", "龟裂"], "In": ["Inclusion", "夹杂"],
    "Pa": ["Patches", "斑块"], "PS": ["Pitted Surface", "麻点"],
    "RS": ["Rolled-in Scale", "轧入氧化皮"], "Sc": ["Scratches", "划痕"],
}


def identify_class(path: Path):
    for part in (path.parent.name, path.stem):
        match = re.match(r"^(Cr|In|Pa|PS|RS|Sc)(?:[_\-\s\d]|$)", part, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return None


def build_neu(root: Path, output: Path, seed: int = 20261119):
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"})
    by_class = {c.lower(): [] for c in CLASSES}
    invalid = []
    for path in files:
        label = identify_class(path.relative_to(root))
        if label not in by_class:
            invalid.append(str(path.relative_to(root)))
            continue
        with Image.open(path) as image:
            if image.size != (200, 200) or image.mode not in {"L", "P"}:
                invalid.append(f"{path.relative_to(root)}: size={image.size}, mode={image.mode}")
                continue
            image.verify()
        by_class[label].append(path.relative_to(root).as_posix())
    counts = {c: len(by_class[c.lower()]) for c in CLASSES}
    if invalid or any(n != 300 for n in counts.values()):
        raise ValueError(f"NEU-CLS validation failed: counts={counts}, invalid={invalid[:20]}")
    paths = [path for c in CLASSES for path in by_class[c.lower()]]
    labels = [i for i in range(6) for _ in range(300)]
    split = stratified_indices(labels, classes=6,
                               counts={"train": 180, "validation": 60, "test": 60}, seed=seed)
    manifest = {
        "dataset": "NEU-CLS", "source": "Northeastern University official NEU-CLS archive",
        "split_seed": seed, "classes": list(CLASSES), "labels": LABELS,
        "class_counts": counts,
        "splits": {name: [paths[i] for i in indices] for name, indices in split.items()},
    }
    # Keep exactly 90 of the 180 training images per class. The train split is
    # already class grouped and randomly ordered by stratified_indices.
    manifest["small_train"] = [paths[i] for class_id in range(6)
                               for i in split["train"][class_id * 180:class_id * 180 + 90]]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != manifest:
            raise FileExistsError(f"refusing to overwrite different split: {output}")
        return manifest
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/neu_cls"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/neu.json"))
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    result = build_neu(args.root, args.output, args.seed)
    print({name: len(paths) for name, paths in result["splits"].items()},
          "small_train", len(result["small_train"]))
