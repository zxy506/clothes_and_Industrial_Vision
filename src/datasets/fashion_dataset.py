"""Fashion-MNIST loader bound to the saved, fixed index manifest."""

import json
from pathlib import Path

import torch
from torchvision.datasets import FashionMNIST


def load_fashion(root="data/raw/fashion_mnist", manifest="data/splits/fashion.json", *, include_test=False):
    record = json.loads(Path(manifest).read_text(encoding="utf-8"))
    train = FashionMNIST(root, train=True, download=False)
    def subset(indices):
        ids = torch.as_tensor(indices, dtype=torch.long)
        return train.data[ids].unsqueeze(1).float().div_(255), train.targets[ids].long()
    x_train, y_train = subset(record["indices"]["train"])
    x_val, y_val = subset(record["indices"]["validation"])
    test_data = None
    if include_test:
        test = FashionMNIST(root, train=False, download=False)
        test_data = (test.data.unsqueeze(1).float().div_(255), test.targets.long())
    return (x_train, y_train), (x_val, y_val), test_data
