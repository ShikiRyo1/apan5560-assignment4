import torch
from torch import nn
from torch.nn import functional as F


class VAELoss(nn.Module):
    def __init__(self, reconstruction_weight=500.0, kl_weight=1.0):
        super().__init__()
        if reconstruction_weight <= 0 or kl_weight < 0:
            raise ValueError("Loss weights must be positive / nonnegative")
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight

    def components(self, recon_x, x, mu, logvar):
        batch_size = x.size(0)
        bce = F.binary_cross_entropy(recon_x, x, reduction="sum") / batch_size
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        total = self.reconstruction_weight * bce + self.kl_weight * kld
        return total, bce, kld

    def forward(self, recon_x, x, mu, logvar):
        total, _, _ = self.components(recon_x, x, mu, logvar)
        return total
