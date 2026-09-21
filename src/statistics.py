"""Aggregation, paper tables, and paired tests for architecture comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from config import PRIMARY_EPSILON
from evaluate import METRIC_NAMES


PRIVATE_MODELS = ["DP-Dense", "DP-CNN", "DP-RVFL"]
NON_PRIVATE_MODELS = ["Dense", "CNN", "RVFL"]
TABLE_METRICS = [*METRIC_NAMES, "training_time_seconds", "inference_time_seconds"]
ABADI2016_REFERENCE = [
    {
        "dataset": "mnist",
        "epsilon_target": 0.5,
        "abadi_reference_accuracy": 0.90,
        "reference_setup": (
            "60-dimensional PCA, one 1000-unit hidden layer, lot size 600, "
            "clipping threshold 4"
        ),
    },
    {
        "dataset": "mnist",
        "epsilon_target": 2.0,
        "abadi_reference_accuracy": 0.95,
        "reference_setup": (
            "60-dimensional PCA, one 1000-unit hidden layer, lot size 600, "
            "clipping threshold 4"
        ),
    },
    {
        "dataset": "mnist",
        "epsilon_target": 8.0,
        "abadi_reference_accuracy": 0.97,
        "reference_setup": (
            "60-dimensional PCA, one 1000-unit hidden layer, lot size 600, "
            "clipping threshold 4"
        ),
    },
    {
        "dataset": "cifar10",
        "epsilon_target": 2.0,
        "abadi_reference_accuracy": 0.67,
        "reference_setup": (
            "pretrained convolutional layers, large lot size, sigma 6, clipping 3"
        ),
    },
    {
        "dataset": "cifar10",
        "epsilon_target": 4.0,
        "abadi_reference_accuracy": 0.70,
        "reference_setup": (
            "pretrained convolutional layers, large lot size, sigma 6, clipping 3"
        ),
    },
    {
        "dataset": "cifar10",
        "epsilon_target": 8.0,
        "abadi_reference_accuracy": 0.73,
        "reference_setup": (
            "pretrained convolutional layers, large lot size, sigma 6, clipping 3"
        ),
    },
]


def _success(results: pd.DataFrame) -> pd.DataFrame:
    data = results.loc[results["run_status"] == "success"].copy()
    numeric = [
        *METRIC_NAMES,
        "training_time_seconds",
        "inference_time_seconds",
        "epsilon_target",
        "epsilon_achieved",
        "achieved_epsilon",
        "delta",
        "max_grad_norm",
        "noise_multiplier",
        "trainable_parameters",
        "total_parameters",
    ]
    for column in numeric:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _mean_sd(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return "NA"
    if len(clean) == 1:
        return f"{clean.mean():.4f} +/- NA"
    return f"{clean.mean():.4f} +/- {clean.std(ddof=1):.4f}"


def _aggregate(
    data: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if data.empty:
        return pd.DataFrame()
    for group_values, group in data.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_columns, group_values))
        row["successful_runs"] = int(len(group))
        for metric in metrics:
            row[f"{metric}_mean"] = pd.to_numeric(group[metric], errors="coerce").mean()
            row[f"{metric}_mean_sd"] = _mean_sd(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def architecture_summary(results: pd.DataFrame) -> pd.DataFrame:
    data = _success(results)
    columns = [
        "dataset",
        "model",
        "architecture_family",
        "architecture_variant",
        "privacy",
        "trainable_parameters",
        "total_parameters",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)
    return (
        data.sort_values(["dataset", "model", "privacy"])
        .drop_duplicates(columns)
        .reindex(columns=columns)
    )


def non_private_comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    data = _success(results)
    data = data.loc[data["privacy"] == "non_private"]
    return _aggregate(
        data,
        ["dataset", "model", "architecture_family", "architecture_variant"],
        TABLE_METRICS,
    )


def private_architecture_comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    data = _success(results)
    data = data.loc[data["privacy"] == "differential_private"]
    return _aggregate(
        data,
        [
            "dataset",
            "epsilon_target",
            "model",
            "architecture_family",
            "architecture_variant",
        ],
        [*METRIC_NAMES, "training_time_seconds", "epsilon_achieved"],
    )


def privacy_utility_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    data = _success(results)
    data = data.loc[data["privacy"] == "differential_private"]
    columns = [
        "dataset",
        "model",
        "architecture_family",
        "architecture_variant",
        "epsilon_target",
        "epsilon_achieved",
        "delta",
        "max_grad_norm",
        *METRIC_NAMES,
        "training_time_seconds",
        "noise_multiplier",
        "seed",
        "status",
    ]
    return data.reindex(columns=columns)


def private_vs_nonprivate_gap_table(results: pd.DataFrame) -> pd.DataFrame:
    data = _success(results)
    non_private = data.loc[data["privacy"] == "non_private"]
    private = data.loc[data["privacy"] == "differential_private"]
    rows: list[dict[str, Any]] = []
    for _, private_row in private.iterrows():
        reference = non_private.loc[
            (non_private["dataset"] == private_row["dataset"])
            & (non_private["seed"] == private_row["seed"])
            & (non_private["architecture_family"] == private_row["architecture_family"])
        ]
        if reference.empty:
            continue
        reference_row = reference.iloc[0]
        for metric in METRIC_NAMES:
            rows.append(
                {
                    "dataset": private_row["dataset"],
                    "architecture": private_row["architecture_family"],
                    "epsilon_target": private_row["epsilon_target"],
                    "seed": private_row["seed"],
                    "metric": metric,
                    "non_private_score": reference_row[metric],
                    "private_score": private_row[metric],
                    "utility_drop": reference_row[metric] - private_row[metric],
                }
            )
    return pd.DataFrame(rows)


def statistical_tests(results: pd.DataFrame) -> pd.DataFrame:
    data = _success(results)
    rows: list[dict[str, Any]] = []

    def add_test(
        dataset: str,
        epsilon: float | None,
        reference: pd.DataFrame,
        comparator: pd.DataFrame,
        reference_model: str,
        comparator_model: str,
        metric: str,
    ) -> None:
        paired = reference[["seed", metric]].merge(
            comparator[["seed", metric]],
            on="seed",
            suffixes=("_reference", "_comparator"),
        ).dropna()
        row = {
            "dataset": dataset,
            "comparison": f"{reference_model} vs {comparator_model}",
            "reference_model": reference_model,
            "comparator_model": comparator_model,
            "epsilon_target": epsilon,
            "metric": metric,
            "n_pairs": int(len(paired)),
            "mean_reference": (
                float(paired[f"{metric}_reference"].mean()) if len(paired) else np.nan
            ),
            "mean_comparator": (
                float(paired[f"{metric}_comparator"].mean()) if len(paired) else np.nan
            ),
            "t_statistic": np.nan,
            "p_value": np.nan,
            "notes": "",
        }
        if len(paired) < 2:
            row["notes"] = "NA: at least two successful paired runs are required."
        else:
            test = stats.ttest_rel(
                paired[f"{metric}_reference"],
                paired[f"{metric}_comparator"],
                nan_policy="omit",
            )
            row["t_statistic"] = float(test.statistic)
            row["p_value"] = float(test.pvalue)
            row["notes"] = "Two-sided paired t-test across matched seeds."
        rows.append(row)

    for dataset in sorted(data["dataset"].dropna().unique()):
        dataset_data = data.loc[data["dataset"] == dataset]
        for epsilon in sorted(dataset_data["epsilon_target"].dropna().unique()):
            private = dataset_data.loc[
                (dataset_data["privacy"] == "differential_private")
                & np.isclose(dataset_data["epsilon_target"], epsilon)
            ]
            pairs = [
                ("DP-Dense", "DP-CNN"),
                ("DP-Dense", "DP-RVFL"),
                ("DP-CNN", "DP-RVFL"),
            ]
            for reference_model, comparator_model in pairs:
                reference = private.loc[private["model"] == reference_model]
                comparator = private.loc[private["model"] == comparator_model]
                for metric in METRIC_NAMES:
                    add_test(
                        dataset,
                        float(epsilon),
                        reference,
                        comparator,
                        reference_model,
                        comparator_model,
                        metric,
                    )

            non_private = dataset_data.loc[dataset_data["privacy"] == "non_private"]
            for architecture in ["Dense", "CNN", "RVFL"]:
                reference = non_private.loc[
                    non_private["architecture_family"] == architecture
                ]
                comparator = private.loc[
                    private["architecture_family"] == architecture
                ]
                for metric in METRIC_NAMES:
                    add_test(
                        dataset,
                        float(epsilon),
                        reference,
                        comparator,
                        architecture,
                        f"DP-{architecture}",
                        metric,
                    )
    return pd.DataFrame(rows)


def abadi2016_reference_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Compare available image DP rows with headline Abadi et al. results.

    This is a sanity/reference table, not a reproduction claim. The project uses
    different architectures, epochs, optimizers, and preprocessing choices.
    """
    data = _success(results)
    data = data.loc[
        (data["privacy"] == "differential_private")
        & data["dataset"].isin({"mnist", "cifar10"})
    ]
    rows: list[dict[str, Any]] = []
    for reference in ABADI2016_REFERENCE:
        subset = data.loc[
            (data["dataset"] == reference["dataset"])
            & np.isclose(data["epsilon_target"], reference["epsilon_target"])
        ]
        row = dict(reference)
        row.update(
            {
                "current_best_model": "NA",
                "current_best_accuracy_mean": np.nan,
                "current_best_accuracy_sd": np.nan,
                "accuracy_difference_current_minus_abadi": np.nan,
                "successful_current_runs": 0,
                "comparison_note": (
                    "Reference only: not an exact reproduction of Abadi et al. "
                    "because this project uses different model families and "
                    "training settings."
                ),
            }
        )
        if not subset.empty:
            grouped = subset.groupby("model")["accuracy"]
            means = grouped.mean().dropna()
            if not means.empty:
                best_model = str(means.idxmax())
                best_values = pd.to_numeric(
                    subset.loc[subset["model"] == best_model, "accuracy"],
                    errors="coerce",
                ).dropna()
                row.update(
                    {
                        "current_best_model": best_model,
                        "current_best_accuracy_mean": float(best_values.mean()),
                        "current_best_accuracy_sd": (
                            float(best_values.std(ddof=1))
                            if len(best_values) > 1
                            else np.nan
                        ),
                        "accuracy_difference_current_minus_abadi": float(
                            best_values.mean() - reference["abadi_reference_accuracy"]
                        ),
                        "successful_current_runs": int(len(best_values)),
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def generate_all_tables(results: pd.DataFrame, tables_dir: str | Path) -> None:
    output = Path(tables_dir)
    output.mkdir(parents=True, exist_ok=True)
    architecture_summary(results).to_csv(output / "architecture_summary.csv", index=False)
    non_private_comparison_table(results).to_csv(
        output / "non_private_comparison_table.csv",
        index=False,
    )
    private_architecture_comparison_table(results).to_csv(
        output / "private_architecture_comparison_table.csv",
        index=False,
    )
    privacy_utility_summary_table(results).to_csv(
        output / "privacy_utility_summary_table.csv",
        index=False,
    )
    private_vs_nonprivate_gap_table(results).to_csv(
        output / "private_vs_nonprivate_gap_table.csv",
        index=False,
    )
    abadi2016_reference_comparison(results).to_csv(
        output / "abadi2016_reference_comparison.csv",
        index=False,
    )
    statistical_tests(results).to_csv(output / "statistical_tests.csv", index=False)
