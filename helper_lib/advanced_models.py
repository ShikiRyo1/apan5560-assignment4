"""CIFAR-10 EBM and epsilon-predicting U-Net; adapted from Module 8.

Images are NCHW RGB tensors in [-1, 1]. No class labels are used.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class EnergyModel(nn.Module):
    """One unconstrained scalar energy per 32x32 RGB image."""
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(width, width*2, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(width*2, width*4, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(width*4, width*4, 3, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(width*4*2*2, width*2), nn.SiLU(),
            nn.Linear(width*2, 1))

    def forward(self, images):
        return self.net(images).flatten()


def langevin(model, initial, steps=60, step_size=10.0, noise_std=0.005, generator=None):
    """Finite, clipped noisy gradient sampler, as in the classroom example.

    It is an approximate short-run sampler, not an exact equilibrium draw.
    autograd.grad only requests input derivatives and never fills parameter .grad.
    The caller may be under no_grad, but must not be under inference_mode.
    """
    images = initial.detach().clone()
    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn(images.shape, device=images.device, generator=generator)
            images = (images + noise_std * noise).clamp(-1, 1)
        with torch.enable_grad():
            images.requires_grad_(True)
            gradient, = torch.autograd.grad(model(images).sum(), images)
        with torch.no_grad():
            images = (images - step_size * gradient.clamp(-0.03, 0.03)).clamp(-1, 1)
    return images.detach()


class TimeEmbedding(nn.Module):
    """Module 8 Part 3 convention: sin block, cos block, denominator half-1."""
    def __init__(self, dimension=64):
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError('dimension must be even and at least 4')
        half = dimension // 2
        self.register_buffer('frequencies', torch.exp(-math.log(10000) * torch.arange(half) / (half-1)))

    def forward(self, timesteps):
        angles = timesteps.float()[:, None] * self.frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.time = nn.Linear(time_dim, outgoing)
        self.norm2 = nn.GroupNorm(8, outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x, time):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(time))[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class UNet(nn.Module):
    """Three downsamplings: 32 -> 16 -> 8 -> 4, with matching skip connections."""
    def __init__(self, width=32, time_dim=64):
        super().__init__()
        self.time = nn.Sequential(TimeEmbedding(time_dim), nn.Linear(time_dim, time_dim*2),
                                  nn.SiLU(), nn.Linear(time_dim*2, time_dim))
        self.input = nn.Conv2d(3, width, 3, padding=1)
        channels = [width, width*2, width*4]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = width
        for outgoing in channels:
            self.down_blocks.append(ResidualBlock(current, outgoing, time_dim))
            self.downsamples.append(nn.Conv2d(outgoing, outgoing, 3, 2, 1))
            current = outgoing
        self.middle = ResidualBlock(current, current, time_dim)
        self.up_blocks = nn.ModuleList()
        for outgoing in reversed(channels):
            self.up_blocks.append(ResidualBlock(current + outgoing, outgoing, time_dim))
            current = outgoing
        self.output = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 3, 3, padding=1))

    def forward(self, images, timesteps):
        time = self.time(timesteps)
        h = self.input(images)
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, time)
            skips.append(h)
            h = down(h)
        h = self.middle(h, time)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
            h = block(torch.cat((h, skip), dim=1), time)
        return self.output(h)


class DiffusionSchedule(nn.Module):
    def __init__(self, steps=1000):
        super().__init__()
        self.steps = steps
        betas = torch.linspace(1e-4, 0.02, steps)
        self.register_buffer('alpha_bars', (1-betas).cumprod(0))

    def add_noise(self, clean, times, noise):
        alpha = self.alpha_bars[times][:, None, None, None]
        return alpha.sqrt()*clean + (1-alpha).sqrt()*noise


@torch.no_grad()
def ddim_sample(model, schedule, initial, steps=50):
    """Deterministic eta=0 DDIM, including the final clean-image transition."""
    if not 2 <= steps <= schedule.steps:
        raise ValueError('DDIM steps must be between 2 and the training schedule length')
    times = torch.linspace(schedule.steps-1, 0, steps, device=initial.device).long()
    images = initial.clone()
    for index, time in enumerate(times):
        t = time.expand(images.shape[0])
        predicted_noise = model(images, t)
        alpha = schedule.alpha_bars[time]
        clean = ((images - (1-alpha).sqrt()*predicted_noise) / alpha.sqrt()).clamp(-1, 1)
        # Recompute epsilon after clipping, consistently with the clipped x0.
        predicted_noise = (images - alpha.sqrt()*clean) / (1-alpha).sqrt()
        previous = schedule.alpha_bars[times[index+1]] if index+1 < len(times) else images.new_tensor(1.)
        images = previous.sqrt()*clean + (1-previous).sqrt()*predicted_noise
    return images.clamp(-1, 1)
