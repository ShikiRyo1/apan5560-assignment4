"""MNIST training split only. No labels or test images enter GAN training."""
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST


def get_gan_data_loader(data_root="data", batch_size=128, seed=42, download=False):
    dataset = MNIST(data_root, train=True, download=download)
    # MNIST fits comfortably in RAM; preconvert once instead of PIL every epoch.
    images = dataset.data.unsqueeze(1).float().div_(127.5).sub_(1)
    tensor_dataset = TensorDataset(images, dataset.targets)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True,
                      num_workers=0, drop_last=False,
                      generator=torch.Generator().manual_seed(seed),
                      pin_memory=torch.cuda.is_available())
