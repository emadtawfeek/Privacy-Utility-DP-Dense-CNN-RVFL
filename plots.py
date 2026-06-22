"""Matplotlib-only paper figures for the architecture DP comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PRIMARY_EPSILON


PRIVATE_MODELS = [
    "DP-Dense",
    "DP-CNN",
    "DP-RVFL",
    "DP-Abadi-MNIST-MLP",
    "DP-Abadi-CIFAR-CNN",
]
NON_PRIVATE_MODELS = ["Dense", "CNN", "RVFL", "Abadi-MNIST-MLP", "Abadi-CIFAR-CNN"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1": "F1-score",
    "auroc": "AUROC",
    "auprc": "AUPRC",
}


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _success(results: pd.DataFrame) -> pd.DataFrame:
    data = results.loc[results["run_status"] == "success"].copy()
    for column in [
        "epsilon_target",
        "epsilon_achieved",
        "accuracy",
        "f1",
        "auroc",
        "auprc",
        "training_time_seconds",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _reference_private_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    """Use epsilon=4 when available; otherwise use the largest successful epsilon."""
    private = data.loc[data["model"].isin(PRIVATE_MODELS)].copy()
    if private.empty:
        return private, None
    primary = private.loc[
        np.isclose(private["epsilon_target"], PRIMARY_EPSILON, equal_nan=False)
    ]
    if not primary.empty:
        return primary, PRIMARY_EPSILON
    epsilons = private["epsilon_target"].dropna()
    if epsilons.empty:
        return private, None
    epsilon = float(epsilons.max())
    return private.loc[
        np.isclose(private["epsilon_target"], epsilon, equal_nan=False)
    ], epsilon


def _mean_error(grouped: pd.core.groupby.DataFrameGroupBy, metric: str):
    return grouped[metric].mean(), grouped[metric].std(ddof=1).fillna(0.0)


def _plot_metric_vs_epsilon(
    dataset: str,
    data: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    private = data.loc[data["model"].isin(PRIVATE_MODELS)]
    if private.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ticks: list[float] = []
    for model in PRIVATE_MODELS:
        subset = private.loc[private["model"] == model]
        if subset.empty:
            continue
        grouped = subset.groupby("epsilon_target", sort=True)
        means, errors = _mean_error(grouped, metric)
        means = means.dropna()
        if means.empty:
            continue
        errors = errors.reindex(means.index).fillna(0.0)
        ticks.extend(means.index.tolist())
        ax.errorbar(
            means.index,
            means.values,
            yerr=errors.values,
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=model,
        )
    for model in NON_PRIVATE_MODELS:
        baseline = data.loc[data["model"] == model, metric].mean()
        if np.isfinite(baseline):
            ax.axhline(baseline, linestyle="--", linewidth=1.0, label=model)
    ax.set_xlabel("Target epsilon")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"{dataset.upper()}: {METRIC_LABELS[metric]} vs epsilon")
    ax.set_xticks(sorted(set(ticks)))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    _save(fig, output_dir, f"{dataset}_{metric}_vs_epsilon")


def _plot_privacy_utility(dataset: str, data: pd.DataFrame, output_dir: Path) -> None:
    private = data.loc[data["model"].isin(PRIVATE_MODELS)].dropna(
        subset=["epsilon_achieved"]
    )
    if private.empty:
        return
    metric = "accuracy" if data["task_type"].iloc[0] == "multiclass" else "auroc"
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for model in PRIVATE_MODELS:
        subset = private.loc[private["model"] == model].dropna(subset=[metric])
        if subset.empty:
            continue
        grouped = subset.groupby("epsilon_target", sort=True)
        epsilon = grouped["epsilon_achieved"].mean()
        utility = grouped[metric].mean()
        error = grouped[metric].std(ddof=1).fillna(0.0)
        ax.errorbar(
            epsilon.values,
            utility.values,
            yerr=error.values,
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=model,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Achieved epsilon (log scale)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"{dataset.upper()}: privacy-utility trade-off")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, output_dir, f"{dataset}_privacy_utility_tradeoff")


def _plot_training_time(dataset: str, data: pd.DataFrame, output_dir: Path) -> None:
    rows = data.copy()
    primary_private, selected_epsilon = _reference_private_rows(rows)
    non_private = rows.loc[rows["privacy"] == "non_private"]
    combined = pd.concat([non_private, primary_private], ignore_index=True)
    if combined.empty:
        return
    order = [*NON_PRIVATE_MODELS, *PRIVATE_MODELS]
    means = combined.groupby("model")["training_time_seconds"].mean().reindex(order).dropna()
    errors = (
        combined.groupby("model")["training_time_seconds"]
        .std(ddof=1)
        .reindex(means.index)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(means))
    ax.bar(x, means.values, yerr=errors.values, capsize=3)
    ax.set_ylabel("Training time (seconds)")
    if selected_epsilon is None:
        ax.set_title(f"{dataset.upper()}: training time comparison")
    else:
        ax.set_title(
            f"{dataset.upper()}: training time comparison (private epsilon={selected_epsilon:g})"
        )
    ax.set_xticks(x, means.index, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, output_dir, f"{dataset}_training_time_comparison")


def _plot_non_private(dataset: str, data: pd.DataFrame, output_dir: Path) -> None:
    subset = data.loc[data["model"].isin(NON_PRIVATE_MODELS)]
    if subset.empty:
        return
    metrics = ["accuracy", "f1", "auroc", "auprc"]
    means = subset.groupby("model")[metrics].mean().reindex(NON_PRIVATE_MODELS)
    errors = subset.groupby("model")[metrics].std(ddof=1).reindex(NON_PRIVATE_MODELS).fillna(0)
    means = means.dropna(how="all")
    if means.empty:
        return
    x = np.arange(len(means.index))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for index, metric in enumerate(metrics):
        ax.bar(
            x + (index - 1.5) * width,
            means[metric],
            width,
            yerr=errors.reindex(means.index)[metric],
            capsize=2,
            label=METRIC_LABELS[metric],
        )
    ax.set_ylabel("Score")
    ax.set_title(f"{dataset.upper()}: non-private architecture comparison")
    ax.set_xticks(x, means.index)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncols=2)
    _save(fig, output_dir, f"{dataset}_non_private_baseline_comparison")


def _plot_all_datasets_accuracy(data: pd.DataFrame, output_dir: Path) -> None:
    private = data.loc[data["model"].isin(PRIVATE_MODELS)]
    if private.empty:
        return
    datasets = sorted(private["dataset"].dropna().unique())
    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(7.5, max(4.0, 3.2 * len(datasets))),
        squeeze=False,
        sharex=True,
    )
    for axis, dataset in zip(axes.ravel(), datasets):
        subset = private.loc[private["dataset"] == dataset]
        for model in PRIVATE_MODELS:
            model_data = subset.loc[subset["model"] == model]
            if model_data.empty:
                continue
            grouped = model_data.groupby("epsilon_target", sort=True)["accuracy"]
            axis.plot(grouped.mean().index, grouped.mean().values, marker="o", label=model)
        axis.set_title(dataset.upper())
        axis.set_ylabel("Accuracy")
        axis.grid(True, axis="y", alpha=0.25)
    axes.ravel()[-1].set_xlabel("Target epsilon")
    axes.ravel()[0].legend(ncols=3, fontsize=8)
    _save(fig, output_dir, "all_datasets_accuracy_vs_epsilon")


def _plot_architecture_rank(data: pd.DataFrame, output_dir: Path) -> None:
    private, selected_epsilon = _reference_private_rows(data)
    if private.empty:
        return
    rows = []
    for dataset, group in private.groupby("dataset"):
        metric = "accuracy" if group["task_type"].iloc[0] == "multiclass" else "auroc"
        means = group.groupby("model")[metric].mean().dropna()
        if means.empty:
            continue
        rows.append({"dataset": dataset, "model": means.idxmax(), "score": means.max()})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = np.arange(len(frame))
    ax.bar(x, frame["score"])
    ax.set_xticks(x, [f"{row.dataset}\n{row.model}" for row in frame.itertuples()], rotation=0)
    ax.set_ylabel("Best private utility")
    if selected_epsilon is None:
        ax.set_title("Best private architecture by dataset")
    else:
        ax.set_title(f"Best private architecture by dataset (epsilon={selected_epsilon:g})")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, output_dir, "architecture_rank_by_dataset")


def _plot_gap(data: pd.DataFrame, output_dir: Path) -> None:
    non_private = data.loc[data["privacy"] == "non_private"]
    private, selected_epsilon = _reference_private_rows(data)
    rows = []
    for _, private_row in private.iterrows():
        reference = non_private.loc[
            (non_private["dataset"] == private_row["dataset"])
            & (non_private["architecture_family"] == private_row["architecture_family"])
            & (non_private["seed"] == private_row["seed"])
        ]
        if reference.empty:
            continue
        metric = "accuracy" if private_row["task_type"] == "multiclass" else "auroc"
        rows.append(
            {
                "label": f"{private_row['dataset']} {private_row['architecture_family']}",
                "drop": reference.iloc[0][metric] - private_row[metric],
            }
        )
    if not rows:
        return
    frame = pd.DataFrame(rows).groupby("label")["drop"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(frame))
    ax.bar(x, frame.values)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, frame.index, rotation=25, ha="right")
    ax.set_ylabel("Utility drop")
    if selected_epsilon is None:
        ax.set_title("Private vs non-private utility gap")
    else:
        ax.set_title(f"Private vs non-private utility gap (epsilon={selected_epsilon:g})")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, output_dir, "private_vs_nonprivate_gap")


def generate_all_figures(results: pd.DataFrame, figures_dir: str | Path) -> None:
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _success(results)
    if data.empty:
        return
    for dataset in sorted(data["dataset"].dropna().unique()):
        subset = data.loc[data["dataset"] == dataset]
        for metric in METRIC_LABELS:
            _plot_metric_vs_epsilon(dataset, subset, metric, output_dir)
        _plot_privacy_utility(dataset, subset, output_dir)
        _plot_training_time(dataset, subset, output_dir)
        _plot_non_private(dataset, subset, output_dir)
    _plot_all_datasets_accuracy(data, output_dir)
    _plot_architecture_rank(data, output_dir)
    _plot_gap(data, output_dir)
