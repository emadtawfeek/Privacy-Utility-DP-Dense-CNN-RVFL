"""Torchvision image dataset loading with stratified validation splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class ImageDatasetBundle:
    dataset: str
    task_type: str
    num_classes: int
    input_shape: tuple[int, int, int]
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    dataset_summary: dict[str, Any]


def _targets(dataset: Dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Torchvision dataset does not expose targets.")
    if isinstance(targets, torch.Tensor):
        return targets.cpu().numpy().astype(np.int64)
    return np.asarray(targets, dtype=np.int64)


def _limit_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    if max_samples is None or len(indices) <= max_samples:
        return indices
    _, limited = train_test_split(
        indices,
        test_size=max_samples,
        random_state=seed,
        stratify=labels[indices],
    )
    return np.asarray(limited, dtype=np.int64)


def load_image_dataset(
    dataset: str,
    data_dir: str | Path,
    seed: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    validation_fraction: float = 0.15,
) -> ImageDatasetBundle:
    """Load MNIST or CIFAR-10 and create a stratified validation split."""
    root = Path(data_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if dataset == "mnist":
        transform = transforms.ToTensor()
        train_full = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_full = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        input_shape = (1, 28, 28)
    elif dataset == "cifar10":
        transform = transforms.ToTensor()
        train_full = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_full = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        input_shape = (3, 32, 32)
    else:
        raise ValueError(f"Unsupported image dataset: {dataset}")

    labels = _targets(train_full)
    train_indices = np.arange(len(labels), dtype=np.int64)
    train_indices = _limit_indices(train_indices, labels, max_train_samples, seed)
    validation_note = "stratified validation split from official training set"
    if validation_fraction <= 0:
        train_idx = train_indices
        val_size = min(1000, len(train_indices))
        val_idx = _limit_indices(train_indices, labels, val_size, seed + 1)
        validation_note = (
            "diagnostic validation subset overlaps training set; multiclass "
            "threshold selection is not used"
        )
    else:
        train_idx, val_idx = train_test_split(
            train_indices,
            test_size=validation_fraction,
            random_state=seed,
            stratify=labels[train_indices],
        )

    test_labels = _targets(test_full)
    test_indices = np.arange(len(test_labels), dtype=np.int64)
    test_indices = _limit_indices(test_indices, test_labels, max_test_samples, seed)

    train_dataset = Subset(train_full, train_idx.tolist())
    validation_dataset = Subset(train_full, val_idx.tolist())
    test_dataset = Subset(test_full, test_indices.tolist())
    train_distribution = np.bincount(labels[train_idx], minlength=10).astype(int)
    val_distribution = np.bincount(labels[val_idx], minlength=10).astype(int)
    test_distribution = np.bincount(test_labels[test_indices], minlength=10).astype(int)
    summary = {
        "dataset": dataset,
        "task_type": "multiclass",
        "num_classes": 10,
        "train_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "test_size": len(test_dataset),
        "input_shape": str(input_shape),
        "feature_count": int(np.prod(input_shape)),
        "class_distribution": {
            "train": train_distribution.tolist(),
            "validation": val_distribution.tolist(),
            "test": test_distribution.tolist(),
        },
        "validation_note": validation_note,
    }
    return ImageDatasetBundle(
        dataset=dataset,
        task_type="multiclass",
        num_classes=10,
        input_shape=input_shape,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        dataset_summary=summary,
    )
