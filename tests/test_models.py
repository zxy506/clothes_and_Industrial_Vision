import torch

from src.models.cnn import SmallCNN
from src.models.cvae_conv import FashionCVAE, cvae_loss


def test_fashion_models_forward_and_backward():
    x = torch.rand(4, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 3])
    assert SmallCNN()(x).shape == (4, 10)
    model = FashionCVAE()
    reconstructed, mu, logvar = model(x, y)
    assert reconstructed.shape == x.shape
    assert mu.shape == logvar.shape == (4, 32)
    assert reconstructed.min() >= 0 and reconstructed.max() <= 1
    loss, rec, kl = cvae_loss(reconstructed, x, mu, logvar, 1.0)
    assert torch.isfinite(loss) and rec > 0 and kl >= 0
    loss.backward()
    assert model.encoder[0].weight.grad is not None
