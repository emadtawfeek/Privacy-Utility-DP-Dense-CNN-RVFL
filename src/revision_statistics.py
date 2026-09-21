"""Summaries and paired contrasts for the prespecified reviewer experiment grid."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["accuracy", "precision", "recall", "f1", "auroc", "auprc", "training_time_seconds"]


def _interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(np.mean(values)) if n else np.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    half = float(stats.t.ppf(.975, n-1) * sd / np.sqrt(n)) if n > 1 else np.nan
    return n, mean, sd, mean-half, mean+half


def _signflip(differences):
    """Exact two-sided paired randomization test for <=16 seed pairs."""
    n = len(differences)
    if n < 2:
        return np.nan
    if n > 16:
        return float(stats.ttest_1samp(differences, 0).pvalue)
    observed = abs(np.mean(differences))
    extreme = sum(abs(np.mean(np.asarray(signs)*differences)) >= observed-1e-14
                  for signs in itertools.product([-1, 1], repeat=n))
    return extreme / 2**n


def write_analysis(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output / "all_results.csv", index=False)
    data = frame.loc[frame["status"] == "success"].copy()
    if data.empty:
        return
    keys = ["dataset", "model", "privacy", "epsilon_target"]
    summary = []
    for values, group in data.groupby(keys, dropna=False):
        for metric in METRICS:
            n, mean, sd, low, high = _interval(group[metric])
            summary.append(dict(zip(keys, values)) | {
                "metric": metric, "n_runs": n, "mean": mean, "sd": sd,
                "ci95_low": low, "ci95_high": high,
                "uncertainty_scope": "across_training_seeds_on_one_fixed_test_partition",
            })
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    privacy_cols = ["dataset", "seed", "model", "epsilon_target", "epsilon_achieved", "delta",
                    "privacy_type", "adjacency", "native_adjacency", "sampling", "sample_rate",
                    "steps_planned", "steps_completed", "empty_batches", "max_grad_norm",
                    "public_reference_size", "expected_batch_size",
                    "noise_multiplier", "noise_scale_laplace", "calibration_tolerance",
                    "calibration_epsilon", "accountant", "secure_rng_used", "privacy_scope"]
    data.loc[data.privacy == "private"].reindex(columns=privacy_cols).to_csv(output / "privacy_protocol.csv", index=False)
    hyper = []
    import json
    for row in data.to_dict("records"):
        hyper.append({k: row.get(k) for k in ["dataset", "model", "privacy", "epsilon_target", "seed",
                                             "trainable_parameters", "frozen_parameters", "total_parameters",
                                             "fitted_coefficients"]} | json.loads(row["hyperparameters"]))
    pd.DataFrame(hyper).to_csv(output / "hyperparameters.csv", index=False)
    contrasts = []
    private = data.loc[data.privacy == "private"]
    for (dataset, epsilon), group in private.groupby(["dataset", "epsilon_target"]):
        ref = group.loc[group.model_key == "rvfl"]
        for comparator, comp in group.loc[group.model_key != "rvfl"].groupby("model"):
            if ref.empty:
                continue
            joined = ref.merge(comp, on="seed", suffixes=("_ref", "_comp"), validate="one_to_one")
            for metric in ["accuracy", "f1", "auroc", "training_time_seconds"]:
                difference = (joined[metric+"_ref"] - joined[metric+"_comp"]).to_numpy(dtype=float)
                difference = difference[np.isfinite(difference)]
                n, mean, sd, low, high = _interval(difference)
                contrasts.append({"dataset": dataset, "epsilon_target": epsilon, "reference": "DP-RVFL",
                    "comparator": comparator, "metric": metric, "n_pairs": n,
                    "mean_difference_reference_minus_comparator": mean, "sd_difference": sd,
                    "ci95_low": low, "ci95_high": high, "p_value": _signflip(difference),
                    "test": "exact_paired_sign_flip" if n <= 16 else "paired_t",
                    "privacy_comparison_note": "DPELM is pure DP; SGD is approximate DP at the recorded delta",
                    "inference_scope": "stochastic_training_variability_conditional_on_fixed_split"})
    contrast_frame = pd.DataFrame(contrasts)
    if not contrast_frame.empty:
        contrast_frame["p_holm"] = np.nan
        valid = contrast_frame.p_value.dropna().sort_values()
        adjusted = np.maximum.accumulate(valid.to_numpy() * np.arange(len(valid), 0, -1))
        contrast_frame.loc[valid.index, "p_holm"] = np.minimum(1, adjusted)
        contrast_frame["multiple_testing_family"] = "all_reported_contrasts_in_this_manifest"
    contrast_frame.to_csv(output / "paired_comparisons.csv", index=False)


def write_figures(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(results)
    data = data.loc[(data.status == "success") & (data.privacy == "private")]
    for dataset, group in data.groupby("dataset"):
        for metric in ["accuracy", "f1", "training_time_seconds"]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for name, model in group.groupby("model"):
                values = model.groupby("epsilon_target")[metric].agg(["mean", "std", "count"]).sort_index()
                if (values["count"] > 1).all():
                    ax.errorbar(values.index, values["mean"], yerr=values["std"], marker="o", capsize=2, label=name)
                else:
                    ax.plot(values.index, values["mean"], marker="o", label=name)
            ax.set(xscale="log", xlabel="Target epsilon", ylabel=metric.replace("_", " "),
                   title=f"{dataset.upper()}  |  mean and SD across training seeds")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7, loc="best")
            fig.text(.01, .01, "DPELM: pure DP. DP-SGD: approximate DP; see privacy_protocol.csv.", fontsize=7)
            fig.tight_layout(rect=(0, .03, 1, 1))
            fig.savefig(output / f"{dataset}_{metric}.png", dpi=180)
            plt.close(fig)
