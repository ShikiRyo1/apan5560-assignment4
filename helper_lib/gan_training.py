"""One D step and one G step per batch; independent Adam states."""
import time
import torch
from torch import nn


def make_optimizers(model, learning_rate=0.0002):
    return {name: torch.optim.Adam(getattr(model, name).parameters(),
                                  lr=learning_rate, betas=(0.5, 0.999))
            for name in ("generator", "discriminator")}


def discriminator_step(model, real, optimizer):
    model.generator.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    z = torch.randn(real.shape[0], model.latent_dim, device=real.device)
    with torch.no_grad():
        fake = model.generator(z)
    criterion = nn.BCEWithLogitsLoss()
    real_logits = model.discriminator.logits(real)
    fake_logits = model.discriminator.logits(fake.detach())
    loss = (criterion(real_logits, torch.ones_like(real_logits)) +
            criterion(fake_logits, torch.zeros_like(fake_logits))) / 2
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite discriminator loss.")
    loss.backward()
    optimizer.step()
    model.discriminator_updates += 1
    return float(loss.detach()), float(real_logits.sigmoid().mean().detach()), float(fake_logits.sigmoid().mean().detach())


def generator_step(model, batch_size, device, optimizer):
    model.discriminator.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    # D parameters are frozen but its differentiable operations remain in G's graph.
    # Its training-mode BatchNorm running buffers continue to update.
    for parameter in model.discriminator.parameters():
        parameter.requires_grad_(False)
    try:
        z = torch.randn(batch_size, model.latent_dim, device=device)
        logits = model.discriminator.logits(model.generator(z))
        loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite generator loss.")
        loss.backward()
        optimizer.step()
    finally:
        for parameter in model.discriminator.parameters():
            parameter.requires_grad_(True)
    model.generator_updates += 1
    return float(loss.detach())


def train_gan(model, data_loader, criterion=None, optimizer=None, device="cpu",
              epochs=1, on_epoch=None):
    """Classroom helper signature retained; GAN uses explicit two-network BCE."""
    if criterion is not None:
        raise ValueError("GAN losses are explicit; leave criterion=None.")
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    model.to(device)
    optimizer = optimizer or make_optimizers(model)
    for _ in range(epochs):
        model.train()
        start = time.perf_counter()
        totals = [0.0] * 4
        count = batches = 0
        for images, _labels in data_loader:
            real = images.to(device, non_blocking=True)
            n = real.shape[0]
            d_loss, d_real, d_fake = discriminator_step(model, real, optimizer["discriminator"])
            g_loss = generator_step(model, n, device, optimizer["generator"])
            for index, value in enumerate((d_loss, g_loss, d_real, d_fake)):
                totals[index] += value * n
            count += n
            batches += 1
        if not count:
            raise ValueError("Training loader is empty.")
        row = {"epoch": len(model.training_history) + 1,
               "discriminator_loss": totals[0] / count,
               "generator_loss": totals[1] / count,
               "d_real_probability": totals[2] / count,
               "d_fake_probability": totals[3] / count,
               "samples": count, "batches": batches,
               "discriminator_updates": model.discriminator_updates,
               "generator_updates": model.generator_updates,
               "elapsed_seconds": time.perf_counter() - start}
        model.training_history.append(row)
        print(row, flush=True)
        if on_epoch:
            on_epoch(model, optimizer, row)
    return model
