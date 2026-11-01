from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True, download=False,
                    seed=42, dataset="CIFAR10"):
    data_dir = Path(data_dir).resolve()
    if dataset == "FashionMNIST":
        dataset_type = datasets.FashionMNIST
        transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
    elif dataset == "CIFAR10":
        dataset_type = datasets.CIFAR10
        transform = transforms.ToTensor()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    dataset = dataset_type(
        root=str(data_dir), train=train, download=download,
        transform=transform,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=0, generator=generator, drop_last=False,
    )
