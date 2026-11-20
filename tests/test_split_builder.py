import numpy as np
import pytest

from src.datasets.split_builder import stratified_indices


def test_stratified_indices_are_balanced_disjoint_and_reproducible():
    labels = np.repeat(np.arange(3), 12)
    first = stratified_indices(labels, classes=3, counts={"train": 4, "validation": 3}, seed=42)
    second = stratified_indices(labels, classes=3, counts={"train": 4, "validation": 3}, seed=42)
    assert first == second
    assert len(set(first["train"]) & set(first["validation"])) == 0
    assert np.bincount(labels[first["train"]]).tolist() == [4, 4, 4]
    assert np.bincount(labels[first["validation"]]).tolist() == [3, 3, 3]


def test_rejects_insufficient_samples():
    with pytest.raises(ValueError):
        stratified_indices([0, 0, 1], classes=2, counts={"train": 2, "validation": 1}, seed=42)
