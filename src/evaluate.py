"""Evaluation utilities for binary and multiclass classification."""

from __future__ import annotations

import time
import warnings

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader


METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]


def select_f1_threshold(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    default: float = 0.5,
) -> float:
    """Select a binary threshold using validation F1 only."""
    y_validation = np.asarray(y_validation, dtype=np.int64)
    probabilities = np.asarray(validation_probabilities, dtype=np.float64)
    finite = np.isfinite(probabilities)
    if finite.sum() == 0 or np.unique(y_validation[finite]).size < 2:
        return float(default)

    precision, recall, thresholds = precision_recall_curve(
        y_validation[finite],
        probabilities[finite],
    )
    if thresholds.size == 0:
        return float(default)

    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_f1 = np.nanmax(f1_values)
    best_indices = np.flatnonzero(np.isclose(f1_values, best_f1, rtol=0.0, atol=1e-12))
    if best_indices.size == 0:
        return float(default)
    candidate_thresholds = thresholds[best_indices]
    selected = candidate_thresholds[
        np.argmin(np.abs(candidate_thresholds - float(default)))
    ]
    return float(np.clip(selected, 0.0, 1.0))


def logits_to_probabilities(
    logits: np.ndarray,
    task_type: str,
) -> np.ndarray:
    """Convert model logits to probabilities for metrics."""
    logits = np.asarray(logits, dtype=np.float64)
    if task_type == "binary":
        logits = np.clip(logits.reshape(-1), -500, 500)
        return 1.0 / (1.0 + np.exp(-logits))
    shifted = logits - np.nanmax(logits, axis=1, keepdims=True)
    exponentiated = np.exp(np.clip(shifted, -500, 500))
    denominator = exponentiated.sum(axis=1, keepdims=True)
    return np.divide(
        exponentiated,
        denominator,
        out=np.full_like(exponentiated, 1.0 / exponentiated.shape[1]),
        where=denominator > 0,
    )


def predict_logits(
    model: torch.nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Collect logits and labels from a loader and measure inference time."""
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for features, targets in loader:
            outputs = model(features)
            logits.append(outputs.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())
    elapsed = time.perf_counter() - start
    return (
        np.concatenate([np.atleast_1d(part) for part in logits]),
        np.concatenate([np.atleast_1d(part) for part in labels]),
        elapsed,
    )


def evaluate_from_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    task_type: str,
    num_classes: int,
    threshold: float = 0.5,
) -> tuple[dict[str, float], str]:
    """Evaluate binary or multiclass probabilities."""
    y_true = np.asarray(y_true, dtype=np.int64)
    notes: list[str] = []
    if task_type == "binary":
        probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        predictions = (probabilities >= threshold).astype(np.int64)
        metrics = {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                metrics["auroc"] = float(roc_auc_score(y_true, probabilities))
            except Exception as error:
                metrics["auroc"] = float("nan")
                notes.append(f"AUROC failed: {type(error).__name__}: {error}")
            try:
                metrics["auprc"] = float(average_precision_score(y_true, probabilities))
            except Exception as error:
                metrics["auprc"] = float("nan")
                notes.append(f"AUPRC failed: {type(error).__name__}: {error}")
        return metrics, " | ".join(notes)

    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.argmax(probabilities, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
    }
    classes = np.arange(num_classes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            metrics["auroc"] = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except Exception as error:
            metrics["auroc"] = float("nan")
            notes.append(f"macro AUROC failed: {type(error).__name__}: {error}")
        try:
            y_binary = label_binarize(y_true, classes=classes)
            metrics["auprc"] = float(
                average_precision_score(y_binary, probabilities, average="macro")
            )
        except Exception as error:
            metrics["auprc"] = float("nan")
            notes.append(f"macro AUPRC failed: {type(error).__name__}: {error}")
    return metrics, " | ".join(notes)
