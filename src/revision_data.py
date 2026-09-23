"""Fixed feature maps and fixed benchmark partitions for reviewer experiments.

These are public-benchmark preparations. The privacy claim is conditional on
the prepared cohort, partition membership, and public sizes. This module does
not claim DP for raw-file deduplication, cohort selection, or dataset audit logs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from data_loader import DatasetBundle, load_csv_auto, validate_dataset, _tensor_dataset
from preprocessing import clean_dataset


PREPROCESSING_VERSION = "fixed_bounds_v1"
# Prespecified numerical engineering scales, not clinical diagnostic cutoffs.
NUMERIC_BOUNDS = {
    "cardio": {"age_years": (0, 120), "height": (100, 250), "weight": (20, 300),
               "ap_hi": (40, 300), "ap_lo": (20, 200), "BMI": (5, 100)},
    "heart": {"age": (0, 120), "trestbps": (40, 300), "chol": (0, 700),
              "thalach": (30, 250), "oldpeak": (0, 10)},
}
CATEGORIES = {
    "cardio": {"gender": [1, 2], "cholesterol": [1, 2, 3], "gluc": [1, 2, 3],
               "smoke": [0, 1], "alco": [0, 1], "active": [0, 1]},
    "heart": {"cp": [0, 1, 2, 3, 4], "restecg": [0, 1, 2], "slope": [0, 1, 2, 3],
              "ca": [0, 1, 2, 3, 4], "thal": [0, 1, 2, 3, 6, 7],
              "sex": [0, 1], "fbs": [0, 1], "exang": [0, 1]},
}


def fixed_tabular_features(frame, dataset):
    """Recordwise transform: no fit, learned category list, mean, or median."""
    columns, names = [], []
    for name, (low, high) in NUMERIC_BOUNDS[dataset].items():
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        values = np.where(missing, (low + high) / 2, values)
        columns.extend([2 * (np.clip(values, low, high) - low) / (high - low) - 1,
                        missing.astype(float)])
        names.extend([name, name + "_missing"])
    for name, levels in CATEGORIES[dataset].items():
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        columns.extend([(values == level).astype(float) for level in levels])
        columns.append((~np.isin(values, levels)).astype(float))
        names.extend([f"{name}_{level}" for level in levels] + [f"{name}_unknown"])
    return np.stack(columns, axis=1).astype(np.float32), names


def _digest_indices(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def load_revision_bundle(dataset, data_dir, split_seed=2026, max_train=None, max_test=None):
    """Public data partitioning independent of model seed, no label stratification.

    Private validation is not accessed by the revision runner. Test scores are
    explicitly treated as public benchmark evaluation, outside training DP.
    """
    data_dir = Path(data_dir)
    rng = np.random.default_rng(split_seed)
    if dataset in {"cardio", "heart"}:
        source = data_dir / ("cardio_train.csv" if dataset == "cardio" else "heart.csv")
        raw = load_csv_auto(source)
        validate_dataset(raw, dataset)
        clean, cleaning = clean_dataset(raw, dataset)
        features, names = fixed_tabular_features(clean, dataset)
        target = "cardio" if dataset == "cardio" else "target"
        labels = clean[target].to_numpy(dtype=np.float32)
        full = _tensor_dataset(features, labels, "binary")
        permutation = rng.permutation(len(full))
        first, second = int(.70 * len(full)), int(.85 * len(full))
        train_idx, val_idx, test_idx = np.split(permutation, [first, second])
        if max_train is not None:
            train_idx = train_idx[:max_train]
        if max_test is not None:
            test_idx = test_idx[:max_test]
        train = Subset(full, train_idx.tolist())
        validation = Subset(full, val_idx.tolist())
        test = Subset(full, test_idx.tolist())
        shape, task_type, classes = (features.shape[1],), "binary", 2
        train_targets = labels[train_idx].astype(np.int64)
        test_targets = labels[test_idx].astype(np.int64)
        prep = {"version": PREPROCESSING_VERSION, "numeric_bounds": NUMERIC_BOUNDS[dataset],
                "categories": CATEGORIES[dataset], "feature_names": names,
                "fit_on_training_records": False, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "public_cohort_cleaning": cleaning}
    else:
        from torchvision import datasets, transforms
        dataset_cls = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}[dataset]
        full = dataset_cls(root=data_dir, train=True, download=False, transform=transforms.ToTensor())
        test_full = dataset_cls(root=data_dir, train=False, download=False, transform=transforms.ToTensor())
        permutation = rng.permutation(len(full))
        first = int(.85 * len(full))
        train_idx, val_idx = permutation[:first], permutation[first:]
        test_idx = rng.permutation(len(test_full))
        if max_train is not None:
            train_idx = train_idx[:max_train]
        if max_test is not None:
            test_idx = test_idx[:max_test]
        train, validation = Subset(full, train_idx.tolist()), Subset(full, val_idx.tolist())
        test = Subset(test_full, test_idx.tolist())
        shape = (1, 28, 28) if dataset == "mnist" else (3, 32, 32)
        task_type, classes = "multiclass", 10
        train_targets = np.asarray(full.targets, dtype=np.int64)[train_idx]
        test_targets = np.asarray(test_full.targets, dtype=np.int64)[test_idx]
        def content_digest(ds):
            digest = hashlib.sha256()
            digest.update(np.asarray(ds.data).tobytes())
            digest.update(np.asarray(ds.targets, dtype="<i8").tobytes())
            return digest.hexdigest()
        prep = {"version": "torchvision_totensor_fixed_0_1", "fit_on_training_records": False,
                "train_content_sha256": content_digest(full), "test_content_sha256": content_digest(test_full)}
    train_counts = np.bincount(train_targets, minlength=classes)
    test_counts = np.bincount(test_targets, minlength=classes)
    majority_class = int(np.argmax(train_counts))
    summary = {
        "dataset": dataset, "split_seed": split_seed, "split_method": "public_fixed_unstratified_permutation",
        "train_size": len(train), "validation_size": len(validation), "test_size": len(test),
        "train_indices_sha256": _digest_indices(train_idx), "test_indices_sha256": _digest_indices(test_idx),
        "validation_indices_sha256": _digest_indices(val_idx), "preprocessing": prep,
        "validation_use": "none", "evaluation_scope": "public_benchmark_test_records",
        "privacy_scope": "single_training_run_conditional_on_fixed_public_cohort_partition_and_sizes",
        "train_class_counts": train_counts.tolist(), "test_class_counts": test_counts.tolist(),
        "majority_class_train": majority_class,
        "majority_baseline_test_accuracy": float(test_counts[majority_class] / len(test_targets)),
        "class_count_scope": "public_benchmark_only_not_a_private_data_release",
    }
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Empty training or test partition.")
    return DatasetBundle(dataset, task_type, classes, shape, train, validation, test,
                         int(np.prod(shape)), None, summary)
