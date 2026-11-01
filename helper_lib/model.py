import torch
from torch import nn
from torch.nn import functional as F


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        features = self.flatten(x)
        return self.fc_mu(features), self.fc_logvar(features)


class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.convtrans1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)
        self.convtrans2 = nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1)
        self.convtrans3 = nn.ConvTranspose2d(32, 1, 3, 2, 1, output_padding=1)

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        x = F.relu(self.convtrans1(x))
        x = F.relu(self.convtrans2(x))
        return torch.sigmoid(self.convtrans3(x))


class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.model_name = "VAE"
        self.training_history = []

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def reconstruct(self, x):
        mu, _ = self.encoder(x)
        return self.decoder(mu)


def get_model(model_name, num_classes=10, latent_dim=None):
    latent_dim = (100 if model_name == "GAN" else 2) if latent_dim is None else latent_dim
    if model_name == "GAN":
        from .gan_model import GAN
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        return GAN(latent_dim)
    if model_name == "VAE":
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        return VAE(latent_dim)
    if model_name == "FCNN":
        layers = [
            nn.Flatten(), nn.Linear(3 * 32 * 32, 200), nn.ReLU(),
            nn.Linear(200, 150), nn.ReLU(), nn.Linear(150, num_classes),
        ]
    elif model_name == "CNN":
        layers = [
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        ]
    elif model_name == "EnhancedCNN":
        layers = []
        channels = [3, 16, 32, 64, 128]
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(2),
            ])
        layers.extend([
            nn.Flatten(), nn.Linear(128 * 2 * 2, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, num_classes),
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = nn.Sequential(*layers)
    model.model_name = model_name
    model.num_classes = num_classes
    model.training_history = []
    return model
