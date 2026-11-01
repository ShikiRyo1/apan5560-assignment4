from pathlib import Path
import torch
from .model import get_model


def choose_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or --device auto")
    return torch.device(requested)


def save_model(model, path, optimizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1, "model_name": model.model_name,
        "num_classes": model.num_classes,
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "history": model.training_history,
        "preprocessing": "RGB 32x32; ToTensor; float32 in [0,1]",
        "classes": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
    }
    torch.save(checkpoint, path)
    return path


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    model = get_model(checkpoint["model_name"], checkpoint["num_classes"])
    model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.training_history = checkpoint["history"]
    model.eval()
    return model, checkpoint


def save_vae_model(model, path, optimizer=None, criterion=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 2, "model_name": "VAE", "latent_dim": model.latent_dim,
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "history": model.training_history,
        "dataset": "FashionMNIST", "preprocessing": "Pad(2); ToTensor; [1,32,32] in [0,1]",
        "reconstruction_weight": criterion.reconstruction_weight if criterion is not None else 500.0,
        "kl_weight": criterion.kl_weight if criterion is not None else 1.0,
    }
    torch.save(checkpoint, path)
    return path


def load_vae_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["format_version"] != 2 or checkpoint["model_name"] != "VAE":
        raise ValueError("Expected a VAE checkpoint")
    model = get_model("VAE", latent_dim=checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.training_history = checkpoint["history"]
    model.eval()
    return model, checkpoint
