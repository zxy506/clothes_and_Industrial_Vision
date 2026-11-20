import json
from pathlib import Path

import torch

from src.models.cnn import NEUCNN
from src.models.cvae_conv import NEUCVAE, cvae_loss


def test_neu_splits_and_model_shapes():
    manifest = json.loads(Path("data/splits/neu.json").read_text(encoding="utf-8"))
    train = set(manifest["small_train"])
    validation = set(manifest["splits"]["validation"])
    test = set(manifest["splits"]["test"])
    assert (len(train), len(validation), len(test)) == (540, 360, 360)
    assert not (train & validation or train & test or validation & test)
    x = torch.rand(2, 1, 128, 128)
    y = torch.tensor([0, 5])
    assert NEUCNN()(x).shape == (2, 6)
    model = NEUCVAE()
    reconstruction, mu, logvar = model(x, y)
    assert reconstruction.shape == x.shape and mu.shape == logvar.shape == (2, 64)
    loss, rec, kl = cvae_loss(reconstruction, x, mu, logvar, 1.0)
    assert torch.isfinite(loss) and rec > 0 and kl >= 0
