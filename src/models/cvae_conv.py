"""Conditional convolutional VAE for 28×28 Fashion-MNIST images."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionCVAE(nn.Module):
    def __init__(self, classes: int = 10, latent_dim: int = 32):
        super().__init__()
        self.classes = classes
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + classes, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), nn.Flatten(),
        )
        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.decode_fc = nn.Linear(latent_dim + classes, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x, y):
        condition = F.one_hot(y, self.classes).to(dtype=x.dtype)
        condition = condition[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        h = self.encoder(torch.cat((x, condition), dim=1))
        return self.mu(h), self.logvar(h)

    def decode(self, z, y):
        condition = F.one_hot(y, self.classes).to(dtype=z.dtype)
        h = self.decode_fc(torch.cat((z, condition), dim=1)).reshape(-1, 64, 7, 7)
        return self.decoder(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z, y), mu, logvar

    @torch.no_grad()
    def sample(self, y, generator=None):
        z = torch.randn((len(y), self.latent_dim), device=y.device, generator=generator)
        return self.decode(z, y)


def cvae_loss(reconstruction, x, mu, logvar, beta: float):
    # Sum pixels and latent dimensions per image; average across the batch.
    rec = F.binary_cross_entropy(reconstruction, x, reduction="none").flatten(1).sum(1).mean()
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(1).mean()
    return rec + beta * kl, rec, kl


class NEUCVAE(nn.Module):
    """Independent six-condition 128×128 convolutional VAE."""

    def __init__(self, classes: int = 6, latent_dim: int = 64):
        super().__init__()
        self.classes = classes
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + classes, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), nn.Flatten(),
        )
        self.mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.logvar = nn.Linear(128 * 8 * 8, latent_dim)
        self.decode_fc = nn.Linear(latent_dim + classes, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x, y):
        condition = F.one_hot(y, self.classes).to(dtype=x.dtype)
        condition = condition[:, :, None, None].expand(-1, -1, 128, 128)
        h = self.encoder(torch.cat((x, condition), dim=1))
        return self.mu(h), self.logvar(h)

    def decode(self, z, y):
        condition = F.one_hot(y, self.classes).to(dtype=z.dtype)
        h = self.decode_fc(torch.cat((z, condition), dim=1)).reshape(-1, 128, 8, 8)
        return self.decoder(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z, y), mu, logvar

    @torch.no_grad()
    def sample(self, y, generator=None):
        z = torch.randn((len(y), self.latent_dim), device=y.device, generator=generator)
        return self.decode(z, y)
