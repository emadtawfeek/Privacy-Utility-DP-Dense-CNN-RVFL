"""Shared configuration and reproducibility helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ["mnist", "cifar10", "cardio", "heart"]
DEFAULT_MODELS = ["dense", "cnn", "rvfl"]
ABADI2016_DATASETS = ["mnist", "cifar10"]
ABADI2016_MODELS = ["abadi"]
DEFAULT_SEEDS = [42, 43, 44]
EPSILONS = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
ABADI2016_EPSILONS = [0.5, 2.0, 8.0]
CLIPPING_NORMS = [0.5, 1.0, 2.0, 4.0]
MAX_GRAD_NORM = 1.0
PRIMARY_CLIPPING_NORM = 1.0
PRIMARY_EPSILON = 4.0
QUICK_TEST_EPSILONS = [0.5, 2.0]
QUICK_MAX_TRAIN_SAMPLES = 1000
QUICK_MAX_TEST_SAMPLES = 300

DATASET_SETTINGS = {
    "cardio": {
        "target": "cardio",
        "task_type": "binary",
        "num_classes": 2,
        "batch_size": 256,
        "epochs": 30,
        "delta": 1e-5,
    },
    "heart": {
        "target": "target",
        "task_type": "binary",
        "num_classes": 2,
        "batch_size": 64,
        "epochs": 30,
        "delta": None,
    },
    "mnist": {
        "task_type": "multiclass",
        "num_classes": 10,
        "batch_size": 256,
        "epochs": 10,
        "delta": 1e-5,
    },
    "cifar10": {
        "task_type": "multiclass",
        "num_classes": 10,
        "batch_size": 256,
        "epochs": 10,
        "delta": 1e-5,
    },
}

ABADI2016_SETTINGS = {
    "mnist": {
        "model_key": "abadi",
        "epsilons": [0.5, 2.0, 8.0],
        "batch_size": 600,
        "epochs": 100,
        "quick_epochs": 2,
        "max_grad_norm": 4.0,
        "lr": 0.1,
        "optimizer": "sgd",
        "lr_schedule": {
            "name": "linear_then_constant",
            "initial_lr": 0.1,
            "final_lr": 0.052,
            "decay_epochs": 10,
        },
        "pca_components": 60,
        "pca_noise_by_epsilon": {0.5: 16.0, 2.0: 7.0, 8.0: 4.0},
        "reported_accuracy_by_epsilon": {0.5: 0.90, 2.0: 0.95, 8.0: 0.97},
        "notes": (
            "Abadi et al. MNIST profile: 60-dimensional noisy PCA projection, "
            "one 1000-unit hidden layer, lot size 600, clipping threshold 4, "
            "learning rate 0.1 linearly decayed to 0.052 over 10 epochs."
        ),
    },
    "cifar10": {
        "model_key": "abadi",
        "epsilons": [2.0, 4.0, 8.0],
        "batch_size_by_epsilon": {2.0: 2000, 4.0: 4000, 8.0: 4000},
        "epochs": 250,
        "quick_epochs": 2,
        "max_grad_norm": 3.0,
        "lr": 0.001,
        "optimizer": "sgd",
        "lr_schedule": None,
        "reported_accuracy_by_epsilon": {2.0: 0.67, 4.0: 0.70, 8.0: 0.73},
        "notes": (
            "Abadi et al. CIFAR-10 profile: center crop to 24x24, tutorial-style "
            "CNN with two 64-channel convolutional layers and two 384-unit fully "
            "connected layers, clipping 3, large lots. The original paper used "
            "CIFAR-100 public pretraining for convolutional layers; this project "
            "records that pretraining is not implemented unless added explicitly."
        ),
    },
}

RESULT_COLUMNS = [
    "dataset",
    "seed",
    "model",
    "model_key",
    "architecture_family",
    "architecture_variant",
    "privacy",
    "task_type",
    "num_classes",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "training_time_seconds",
    "inference_time_seconds",
    "epsilon_target",
    "epsilon_achieved",
    "target_epsilon",
    "achieved_epsilon",
    "delta",
    "clipping_norm",
    "max_grad_norm",
    "noise_multiplier",
    "secure_rng_used",
    "accountant",
    "rdp_alpha_mode",
    "alpha_count",
    "n_train",
    "n_validation",
    "n_test",
    "input_shape",
    "feature_count",
    "trainable_parameters",
    "total_parameters",
    "threshold",
    "epochs_completed",
    "private_setup_method",
    "n_random_features",
    "activation",
    "selected_hyperparameters",
    "paper_profile",
    "reference_source",
    "configuration_match",
    "abadi_reference_accuracy",
    "best_validation_auroc",
    "status",
    "run_status",
    "error_message",
]


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create and return the standard output directories."""
    root = Path(output_dir).resolve()
    paths = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def configure_runtime(torch_threads: int) -> None:
    """Configure CPU thread pools before model training starts."""
    thread_count = max(1, int(torch_threads))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(thread_count)

    import torch

    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(max(1, thread_count // 2))
    except RuntimeError:
        # PyTorch permits setting inter-op threads only once per process.
        pass


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible CPU execution."""
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def parse_seeds(value: str | None) -> list[int]:
    """Parse a comma-separated seed list."""
    if value is None:
        return DEFAULT_SEEDS.copy()
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")
    return seeds


def get_rdp_alphas(mode: str) -> list[float | int] | None:
    """Return the configured RDP orders, or None for Opacus defaults."""
    if mode == "wide":
        return (
            [1 + x / 10 for x in range(1, 100)]
            + list(range(12, 256))
            + [256, 512, 1024, 2048, 4096, 8192, 16384]
        )
    return None


def format_number_token(value: float | None) -> str:
    """Create a filesystem-safe token for an optional numeric value."""
    if value is None:
        return "na"
    return str(value).replace(".", "p")


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    """Return unique strings without changing their original order."""
    return list(dict.fromkeys(values))
