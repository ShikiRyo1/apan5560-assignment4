"""Classroom helpers, loaded only when requested.

The CPU inference container does not need torchvision or matplotlib.
"""
from importlib import import_module

_EXPORTS = {
    "get_data_loader": "data_loader", "get_model": "model",
    "train_model": "trainer", "train_vae_model": "trainer",
    "train_gan": "gan_training", "evaluate_model": "evaluator",
    "choose_device": "utils", "save_model": "utils", "load_model": "utils",
    "save_vae_model": "utils", "load_vae_model": "utils",
    "VAELoss": "losses", "generate_samples": "generator",
    "get_gan_data_loader": "gan_data",
    "save_gan_model": "gan_checkpoint", "load_gan_model": "gan_checkpoint",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
