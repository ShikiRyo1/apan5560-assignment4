from pathlib import Path
import math
import matplotlib.pyplot as plt
import torch


def plot_image_grid(images, path=None, title="Generated samples", show=True):
    n = len(images)
    if n < 1:
        raise ValueError("At least one image is required")
    cols = min(5, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < n:
            pixels = images[i].detach().cpu()
            if pixels.shape[0] == 3:
                ax.imshow(pixels.permute(1, 2, 0).clamp(0, 1).numpy())
            else:
                ax.imshow(pixels.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Sample {i + 1}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def generate_samples(model, device, num_samples=10, path=None, show=True, seed=42):
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    model.to(device)
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim, generator=generator).to(device)
        if model.model_name == 'GAN':
            samples = ((model.generator(z).cpu() + 1) / 2).clamp(0, 1)
            title = 'MNIST GAN: generated samples (checkpoint training scope applies)'
        else:
            samples = model.decoder(z).cpu()
            title = 'VAE: samples from N(0, I)'
    model.train(was_training)
    plot_image_grid(samples, path, title, show)
    return samples
