import torch
from torch.nn import functional as F
from .gan_training import train_gan


def train_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    model.to(device)
    history = getattr(model, "training_history", [])
    for _ in range(epochs):
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            n = labels.size(0)
            loss_sum += loss.item() * n
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += n
        if count == 0:
            raise ValueError("Training loader is empty")
        row = {"epoch": len(history) + 1, "loss": loss_sum / count,
               "accuracy": correct / count, "samples": count}
        history.append(row)
        print(f"epoch={row['epoch']} train_loss={row['loss']:.4f} "
              f"train_accuracy={row['accuracy']:.2%} samples={count}")
    model.training_history = history
    return model


def train_vae_model(model, data_loader, criterion, optimizer,
                    device="cpu", epochs=10):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    model.to(device)
    history = getattr(model, "training_history", [])
    for _ in range(epochs):
        model.train()
        totals = [0.0, 0.0, 0.0]
        count = 0
        for images, _labels in data_loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(images)
            loss = criterion(recon, images, mu, logvar)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite VAE training loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                n = images.size(0)
                bce = F.binary_cross_entropy(recon, images, reduction="sum") / n
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n
            n = images.size(0)
            for i, value in enumerate((loss, bce, kld)):
                totals[i] += value.item() * n
            count += n
        if count == 0:
            raise ValueError("Training loader is empty")
        row = {"epoch": len(history) + 1, "loss": totals[0] / count,
               "bce_per_image": totals[1] / count,
               "kl_per_image": totals[2] / count, "samples": count}
        history.append(row)
        print(f"epoch={row['epoch']} loss={row['loss']:.4f} "
              f"BCE/image={row['bce_per_image']:.4f} "
              f"KL/image={row['kl_per_image']:.4f} samples={count}", flush=True)
    model.training_history = history
    return model
