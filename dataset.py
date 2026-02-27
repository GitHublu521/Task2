"""CIFAR-10 and CIFAR-10-C data loading utilities."""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Standard transforms
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

transform_basic = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

transform_augmented = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# ---------------------------------------------------------------------------
# CIFAR-10 loaders
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size: int = 128, subset_size: int = None,
                        num_workers: int = 2, augment: bool = False,
                        data_dir: str = "data"):
    """Return train and test DataLoaders for CIFAR-10.

    Args:
        batch_size: Batch size for both loaders.
        subset_size: If set, use only this many samples from training set.
        num_workers: Number of data loading workers.
        augment: Whether to apply data augmentation to training set.
        data_dir: Directory to download/load CIFAR-10 data.

    Returns:
        (train_loader, test_loader)
    """
    train_transform = transform_augmented if augment else transform_basic

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_basic
    )

    if subset_size is not None and subset_size < len(train_dataset):
        # Deterministic subset: first N samples
        indices = list(range(subset_size))
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# CIFAR-10-C loader
# ---------------------------------------------------------------------------

def load_cifar10c(corruption: str, severity: int,
                  cifar10c_path: str = "data/CIFAR-10-C",
                  batch_size: int = 128, num_workers: int = 2):
    """Load a specific corruption and severity from CIFAR-10-C .npy files.

    Each .npy file has shape (50000, 32, 32, 3) with 5 severities of 10000
    images each. Severity s uses indices [(s-1)*10000 : s*10000].

    Args:
        corruption: Name of the corruption (e.g., "gaussian_noise").
        severity: Severity level (1-5).
        cifar10c_path: Path to the CIFAR-10-C directory with .npy files.
        batch_size: Batch size.
        num_workers: Number of workers.

    Returns:
        DataLoader for the specified corruption/severity.
    """
    images_path = os.path.join(cifar10c_path, f"{corruption}.npy")
    labels_path = os.path.join(cifar10c_path, "labels.npy")

    if not os.path.exists(images_path):
        raise FileNotFoundError(
            f"CIFAR-10-C data not found at {images_path}.\n"
            f"Download from: https://zenodo.org/record/2535967\n"
            f"Extract to: {cifar10c_path}/"
        )

    images = np.load(images_path)
    labels = np.load(labels_path)

    # Slice by severity (each severity has 10000 images)
    start = (severity - 1) * 10000
    end = severity * 10000
    images = images[start:end]
    labels = labels[start:end]

    # Convert to torch tensors: (N, H, W, C) -> (N, C, H, W), normalize
    images = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    images = (images - mean) / std

    labels = torch.from_numpy(labels).long()

    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
