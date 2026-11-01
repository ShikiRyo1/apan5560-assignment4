"""Offline CPU inference from the bundled, trained MNIST generator."""
import base64
import hashlib
from io import BytesIO
import math
from pathlib import Path
import secrets

import numpy as np
from PIL import Image
import torch

from helper_lib.gan_model import Generator

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mnist_gan_generator.pt"


def png_bytes(pixels):
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


class GANService:
    def __init__(self, model_path=MODEL_PATH):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint.get("latent_dim") != 100:
            raise ValueError("Expected Assignment 3 generator checkpoint.")
        self.model = Generator()
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.metadata = {
            "model": "Assignment3-MNIST-GAN", "version": "3.0.0", "device": "cpu",
            "latent_dim": 100, "output_shape_per_image": [1, 28, 28],
            "dataset": "MNIST", "training_split_size": checkpoint["metadata"]["training_samples"],
            "training_epochs": checkpoint["metadata"]["completed_epochs"],
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "normalization": "Generator Tanh [-1,1]; returned PNG uint8 round((x+1)*127.5)",
            "limitations": "Unconditional generation; no requested digit label; ambiguous or malformed digits can occur.",
        }

    def generate(self, count=16, seed=None):
        seed = secrets.randbelow(2**31) if seed is None else seed
        # Per-request RNG does not change the process-wide training/random state.
        rng = torch.Generator(device="cpu").manual_seed(seed)
        with torch.inference_mode():
            images = self.model(torch.randn(count, 100, generator=rng))
        pixels = images.add(1).mul(127.5).clamp(0, 255).round().byte().squeeze(1).numpy()
        columns = min(8, math.ceil(math.sqrt(count)))
        rows = math.ceil(count / columns)
        grid = np.zeros((rows * 28, columns * 28), dtype=np.uint8)
        individual = []
        for index, pixel in enumerate(pixels):
            row, col = divmod(index, columns)
            grid[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = pixel
            individual.append(base64.b64encode(png_bytes(pixel)).decode("ascii"))
        return {"model": self.metadata["model"], "seed": seed, "count": count,
                "image_shape": [1, 28, 28], "mime_type": "image/png",
                "images_base64": individual,
                "grid_png_base64": base64.b64encode(png_bytes(grid)).decode("ascii"),
                "grid_shape": list(grid.shape)}
