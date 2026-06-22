"""Leakage-safe cleaning, splitting, and feature preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CARDIO_NUMERIC = ["age_years", "height", "weight", "ap_hi", "ap_lo", "BMI"]
CARDIO_CATEGORICAL = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
HEART_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
HEART_CATEGORICAL = ["cp", "restecg", "slope", "ca", "thal"]
HEART_BINARY = ["sex", "fbs", "exang"]
MIN_CARDIO_CLEANED_ROWS = 10_000


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    cleaning_summary: dict[str, Any]


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_cardio(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = frame.copy()
    rows_before = len(data)
    if "id" in data.columns:
        data = data.drop(columns=["id"])

    numeric_source = ["age", "height", "weight", "ap_hi", "ap_lo", "cardio"]
    for column in numeric_source:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["age_years"] = data["age"] / 365.25
    height_metres = data["height"] / 100.0
    data["BMI"] = data["weight"] / height_metres.pow(2)
    data = data.drop(columns=["age"])

    plausible = (
        data["height"].between(120, 220)
        & data["weight"].between(30, 200)
        & data["ap_hi"].between(80, 250)
        & data["ap_lo"].between(40, 150)
        & (data["ap_hi"] >= data["ap_lo"])
    )
    data = data.loc[plausible].copy()
    data = data.dropna(subset=["cardio"])
    data["cardio"] = data["cardio"].astype(np.int64)

    selected = CARDIO_NUMERIC + CARDIO_CATEGORICAL + ["cardio"]
    data = data[selected]
    if len(data) < MIN_CARDIO_CLEANED_ROWS:
        raise ValueError(
            "Cardio dataset validation failed: "
            f"only {len(data):,} rows remain after cleaning; at least "
            f"{MIN_CARDIO_CLEANED_ROWS:,} are required. "
            "The wrong file was probably loaded instead of the large Kaggle "
            "cardiovascular dataset."
        )
    summary = {
        "rows_before_cleaning": rows_before,
        "rows_after_cleaning": len(data),
        "rows_removed": rows_before - len(data),
        "duplicates_removed": 0,
    }
    return data, summary


def clean_heart(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows_before = len(frame)
    duplicates = int(frame.duplicated().sum())
    data = frame.drop_duplicates().copy()

    selected = HEART_NUMERIC + HEART_CATEGORICAL + HEART_BINARY + ["target"]
    data = data[selected]
    for column in selected:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["target"])
    data["target"] = (data["target"] > 0).astype(np.int64)

    summary = {
        "rows_before_cleaning": rows_before,
        "rows_after_cleaning": len(data),
        "rows_removed": rows_before - len(data),
        "duplicates_removed": duplicates,
    }
    return data, summary


def clean_dataset(frame: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if dataset == "cardio":
        return clean_cardio(frame)
    if dataset == "heart":
        return clean_heart(frame)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _split_indices(target: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(target))
    train_idx, remainder_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=target,
    )
    validation_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=0.50,
        random_state=seed,
        stratify=target.iloc[remainder_idx],
    )
    return train_idx, validation_idx, test_idx


def _build_preprocessor(dataset: str) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    binary_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    if dataset == "cardio":
        return ColumnTransformer(
            [
                ("numeric", numeric_pipeline, CARDIO_NUMERIC),
                ("categorical", categorical_pipeline, CARDIO_CATEGORICAL),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    if dataset == "heart":
        return ColumnTransformer(
            [
                ("numeric", numeric_pipeline, HEART_NUMERIC),
                ("categorical", categorical_pipeline, HEART_CATEGORICAL),
                ("binary", binary_pipeline, HEART_BINARY),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def prepare_dataset(frame: pd.DataFrame, dataset: str, seed: int) -> PreparedData:
    """Clean, split, fit preprocessing on train only, and transform all partitions."""
    clean, cleaning_summary = clean_dataset(frame, dataset)
    target_columns = {
        "cardio": "cardio",
        "heart": "target",
    }
    target_column = target_columns[dataset]
    target = clean[target_column]
    if target.nunique() != 2:
        raise ValueError(
            f"{dataset} target must contain two classes after cleaning; found "
            f"{sorted(target.unique().tolist())}."
        )

    train_idx, validation_idx, test_idx = _split_indices(target, seed)
    features = clean.drop(columns=[target_column])
    preprocessor = _build_preprocessor(dataset)
    X_train = preprocessor.fit_transform(features.iloc[train_idx])
    X_validation = preprocessor.transform(features.iloc[validation_idx])
    X_test = preprocessor.transform(features.iloc[test_idx])

    arrays = [X_train, X_validation, X_test]
    arrays = [np.asarray(array, dtype=np.float32) for array in arrays]
    y = target.to_numpy(dtype=np.float32)
    feature_names = preprocessor.get_feature_names_out().tolist()

    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Preprocessing produced non-finite feature values.")

    return PreparedData(
        X_train=arrays[0],
        X_validation=arrays[1],
        X_test=arrays[2],
        y_train=y[train_idx],
        y_validation=y[validation_idx],
        y_test=y[test_idx],
        feature_names=feature_names,
        cleaning_summary=cleaning_summary,
    )
