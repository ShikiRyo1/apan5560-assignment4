"""Assignment 3's exact MNIST layers; a standard GAN, not the Week 6 WGAN."""
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        if z_dim != 100:
            raise ValueError("Assignment 3 fixes the latent dimension at 100.")
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(64, 1, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError("Noise must have shape (batch, 100).")
        x = self.fc(z).reshape(z.shape[0], 128, 7, 7)
        return self.tanh(self.deconv2(self.relu(self.bn1(self.deconv1(x)))))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def logits(self, x):
        """Pre-sigmoid scores for numerically stable BCEWithLogitsLoss."""
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 28, 28):
            raise ValueError("Images must have shape (batch, 1, 28, 28).")
        x = self.act1(self.conv1(x))
        x = self.act2(self.bn2(self.conv2(x)))
        return self.fc(self.flatten(x))

    def forward(self, x):
        """The assignment's requested real/fake probability, shape (batch, 1)."""
        return torch.sigmoid(self.logits(x))


class GAN(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        self.model_name = "GAN"
        self.training_history = []
        self.discriminator_updates = 0
        self.generator_updates = 0

    def forward(self, z):
        return self.generator(z)


def initialize_weights(module):
    """Conventional DCGAN initialization; preserves the specified layer list."""
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
