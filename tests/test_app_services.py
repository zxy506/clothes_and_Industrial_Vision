import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from src.services import inference_service
from src.services import fashion_inference_service
from src.services.generation_service import generate, load_generator
from src.services.history_service import query_history, save_predictions, to_csv_bytes
from ui.pages.recognition import _read_image


def test_real_batch_inference_and_history(tmp_path):
    model, device, version = inference_service.load_classifier("A", 0)
    manifest = json.loads((inference_service.ROOT / "data/splits/neu.json").read_text(encoding="utf-8"))
    paths = manifest["small_train"][:2]
    images = []
    for relative in paths:
        with Image.open(inference_service.ROOT / "data/raw/neu_cls" / relative) as image:
            images.append(image.copy())
    rows = inference_service.predict(model, device, images)
    assert len(rows) == 2
    assert all(len(row["probabilities"]) == 6 for row in rows)
    assert all(abs(sum(row["probabilities"].values()) - 1) < 1e-5 for row in rows)
    db = tmp_path / "history.db"
    saved = save_predictions([{"image_name": Path(path).name, **row}
                              for path, row in zip(paths, rows)], mode="A", seed=0,
                             version=version, path=db)
    assert saved == 2
    history = query_history(path=db)
    assert len(history) == 2 and Path(paths[0]).name.encode() in to_csv_bytes(history)
    db.unlink()  # verifies the SQLite connection was closed on Windows


def test_real_generator_seed_and_bad_upload():
    model, device, _ = load_generator()
    first = generate(model, device, "Cr", 2, 123)
    second = generate(model, device, "Cr", 2, 123)
    assert len(first) == 2 and first[0].size == (128, 128)
    assert first[0].tobytes() == second[0].tobytes()
    with pytest.raises(ValueError):
        generate(model, device, "Cr", 49, 0)
    with pytest.raises((UnidentifiedImageError, OSError)):
        _read_image(SimpleNamespace(getvalue=lambda: b"not-an-image"))


def test_missing_classifier_checkpoint_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(inference_service, "checkpoint_path", lambda mode, seed: tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="模型文件缺失"):
        inference_service.load_classifier("A", 0)


def test_real_fashion_inference_matches_formal_test_and_history(tmp_path):
    model, device, version = fashion_inference_service.load_classifier("B", 0)
    samples = sorted((inference_service.ROOT / "demo_samples/fashion_test").glob("*.png"))[:3]
    images = []
    for path in samples:
        with Image.open(path) as image:
            images.append(image.copy())
    predictions = fashion_inference_service.predict(model, device, images)
    with np.load(inference_service.ROOT / "results/raw/fashion_final/B_seed0_predictions.npz") as saved:
        expected = [str(saved["clean_predictions"][int(path.name.split("_")[0])]) for path in samples]
    assert [row["class_code"] for row in predictions] == expected
    assert all(len(row["probabilities"]) == 10 and
               abs(sum(row["probabilities"].values()) - 1) < 1e-5 for row in predictions)
    db = tmp_path / "fashion_history.db"
    rows = [{"image_name": path.name, **result} for path, result in zip(samples, predictions)]
    save_predictions(rows, mode="B", seed=0, version=version, path=db,
                     model_family="FashionCNN")
    assert len(query_history(path=db, model_family="FashionCNN")) == 3
    assert query_history(path=db, model_family="NEUCNN").empty
    db.unlink()
