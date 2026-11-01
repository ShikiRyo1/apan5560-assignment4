"""Portable checkpoint with both networks, both optimizers and RNG states."""
from pathlib import Path
import torch
from .gan_model import GAN


def save_gan_model(model, path, optimizer=None, metadata=None, loader_generator=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 4, "model_name": "GAN", "variant": "MNIST-BCE-GAN",
                "latent_dim": model.latent_dim, "model_state": model.state_dict(),
                "optimizer_state": None if optimizer is None else {
                    name: opt.state_dict() for name, opt in optimizer.items()},
                "history": model.training_history,
                "discriminator_updates": model.discriminator_updates,
                "generator_updates": model.generator_updates, "metadata": metadata or {},
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "loader_rng_state": None if loader_generator is None else loader_generator.get_state()}, path)
    return path


def load_gan_model(path, device="cpu", restore_rng=False, loader_generator=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 4 or checkpoint.get("variant") != "MNIST-BCE-GAN":
        raise ValueError("Expected the Assignment 3 MNIST GAN checkpoint.")
    model = GAN(checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.training_history = checkpoint["history"]
    model.discriminator_updates = checkpoint["discriminator_updates"]
    model.generator_updates = checkpoint["generator_updates"]
    model.eval()
    if restore_rng:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        if loader_generator is not None and checkpoint["loader_rng_state"] is not None:
            loader_generator.set_state(checkpoint["loader_rng_state"].cpu())
    return model, checkpoint


def restore_optimizer_states(optimizers, checkpoint):
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint["optimizer_state"][name])
