"""Dataset loading and audit reporting."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset

from preprocessing import PreparedData, prepare_dataset


EXPECTED_COLUMNS = {
    "cardio": [
        "age",
        "gender",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "cardio",
    ],
    "heart": [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ],
}

TARGET_COLUMNS = {
    "cardio": "cardio",
    "heart": "target",
}


@dataclass
class DatasetBundle:
    dataset: str
    task_type: str
    num_classes: int
    input_shape: tuple[int, ...]
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    feature_count: int
    prepared_data: PreparedData | None
    dataset_summary: dict[str, Any]


def derive_target(frame: pd.DataFrame, dataset: str) -> pd.Series:
    """Return the numeric binary target for a supported dataset."""
    return pd.to_numeric(frame[TARGET_COLUMNS[dataset]], errors="coerce")


def _sniff_separator(path: Path) -> str | None:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return None


def load_csv_auto(path: str | Path) -> pd.DataFrame:
    """Load a CSV using delimiter detection and standard missing-value markers."""
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}\n"
            "Place the requested CSV in the data directory or pass its path explicitly."
        )

    separator = _sniff_separator(csv_path)
    read_kwargs = {
        "na_values": ["?", "NA", "N/A", "null", "NULL"],
        "keep_default_na": True,
    }
    if separator:
        frame = pd.read_csv(csv_path, sep=separator, **read_kwargs)
    else:
        frame = pd.read_csv(csv_path, sep=None, engine="python", **read_kwargs)

    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.shape[1] == 1:
        raise ValueError(
            f"Could not detect a valid CSV separator for {csv_path}. "
            "Expected a comma-, semicolon-, tab-, or pipe-separated file."
        )
    return frame


def validate_dataset(frame: pd.DataFrame, dataset: str) -> None:
    """Check that the columns required by the selected public dataset exist."""
    if dataset not in EXPECTED_COLUMNS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    missing = [column for column in EXPECTED_COLUMNS[dataset] if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{dataset} dataset is missing required columns: {missing}. "
            f"Found columns: {frame.columns.tolist()}"
        )


def audit_dataset(
    frame: pd.DataFrame,
    dataset: str,
    source_path: str | Path,
) -> dict[str, object]:
    """Print and return a machine-readable dataset audit."""
    target = TARGET_COLUMNS[dataset]
    target_values = derive_target(frame, dataset)
    target_distribution = target_values.value_counts(dropna=False).sort_index()
    missing_by_column = frame.isna().sum()
    duplicates = int(frame.duplicated().sum())
    features = [column for column in frame.columns if column != target]

    print(f"\nDataset: {dataset}")
    print(f"Shape: {frame.shape}")
    print("Target distribution:")
    print(target_distribution.to_string())
    print("Missing values:")
    print(missing_by_column.to_string())
    print(f"Duplicate rows: {duplicates}")
    print(f"Feature names: {features}")

    return {
        "dataset": dataset,
        "source_path": str(Path(source_path).expanduser().resolve()),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "target_column": target,
        "target_distribution": json.dumps(
            {str(key): int(value) for key, value in target_distribution.items()}
        ),
        "missing_values_total": int(missing_by_column.sum()),
        "missing_values_by_column": json.dumps(
            {str(key): int(value) for key, value in missing_by_column.items() if value}
        ),
        "duplicate_rows": duplicates,
        "feature_count": len(features),
        "feature_names": json.dumps(features),
    }


def load_and_audit(
    dataset: str,
    path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = load_csv_auto(path)
    validate_dataset(frame, dataset)
    return frame, audit_dataset(frame, dataset, path)


def save_dataset_summary(audits: list[dict[str, object]], output_path: str | Path) -> None:
    pd.DataFrame(audits).to_csv(output_path, index=False)


def _limit_arrays(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(y) <= max_samples:
        return X, y
    _, selected = train_test_split(
        np.arange(len(y)),
        test_size=max_samples,
        random_state=seed,
        stratify=y.astype(np.int64),
    )
    selected = np.asarray(selected, dtype=np.int64)
    return X[selected], y[selected]


def _tensor_dataset(X: np.ndarray, y: np.ndarray, task_type: str) -> TensorDataset:
    targets = (
        torch.from_numpy(y.astype(np.float32))
        if task_type == "binary"
        else torch.from_numpy(y.astype(np.int64))
    )
    return TensorDataset(torch.from_numpy(X.astype(np.float32)), targets)


def _class_distribution(y: np.ndarray, num_classes: int) -> list[int]:
    return np.bincount(y.astype(np.int64), minlength=num_classes).astype(int).tolist()


def load_tabular_bundle(
    dataset: str,
    path: str | Path,
    seed: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> DatasetBundle:
    """Load, clean, split, preprocess, and wrap a tabular dataset."""
    frame = load_csv_auto(path)
    validate_dataset(frame, dataset)
    prepared = prepare_dataset(frame, dataset, seed)
    task_type = "binary"
    num_classes = 2

    X_train, y_train = _limit_arrays(
        prepared.X_train,
        prepared.y_train,
        max_train_samples,
        seed,
    )
    X_validation = prepared.X_validation
    y_validation = prepared.y_validation
    X_test, y_test = _limit_arrays(
        prepared.X_test,
        prepared.y_test,
        max_test_samples,
        seed,
    )
    feature_count = int(X_train.shape[1])
    input_shape = (feature_count,)
    summary = {
        "dataset": dataset,
        "task_type": task_type,
        "num_classes": num_classes,
        "train_size": len(y_train),
        "validation_size": len(y_validation),
        "test_size": len(y_test),
        "input_shape": str(input_shape),
        "feature_count": feature_count,
        "class_distribution": {
            "train": _class_distribution(y_train, num_classes),
            "validation": _class_distribution(y_validation, num_classes),
            "test": _class_distribution(y_test, num_classes),
        },
        **prepared.cleaning_summary,
    }
    return DatasetBundle(
        dataset=dataset,
        task_type=task_type,
        num_classes=num_classes,
        input_shape=input_shape,
        train_dataset=_tensor_dataset(X_train, y_train, task_type),
        validation_dataset=_tensor_dataset(X_validation, y_validation, task_type),
        test_dataset=_tensor_dataset(X_test, y_test, task_type),
        feature_count=feature_count,
        prepared_data=prepared,
        dataset_summary=summary,
    )


def load_dataset_bundle(
    dataset: str,
    *,
    seed: int,
    cardio_path: str | Path,
    heart_path: str | Path,
    data_dir: str | Path,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    image_validation_fraction: float = 0.15,
) -> DatasetBundle:
    """Load any supported dataset as torch datasets plus metadata."""
    if dataset in {"cardio", "heart"}:
        path = cardio_path if dataset == "cardio" else heart_path
        return load_tabular_bundle(
            dataset,
            path,
            seed,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
        )
    if dataset in {"mnist", "cifar10"}:
        from datasets_image import load_image_dataset

        image_bundle = load_image_dataset(
            dataset,
            data_dir,
            seed,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            validation_fraction=image_validation_fraction,
        )
        return DatasetBundle(
            dataset=image_bundle.dataset,
            task_type=image_bundle.task_type,
            num_classes=image_bundle.num_classes,
            input_shape=image_bundle.input_shape,
            train_dataset=image_bundle.train_dataset,
            validation_dataset=image_bundle.validation_dataset,
            test_dataset=image_bundle.test_dataset,
            feature_count=int(np.prod(image_bundle.input_shape)),
            prepared_data=None,
            dataset_summary=image_bundle.dataset_summary,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")
