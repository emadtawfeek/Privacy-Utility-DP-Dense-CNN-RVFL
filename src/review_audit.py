"""Produce a read-only reviewer audit from an existing revision-run directory.

No manuscript result is asserted until the corresponding tasks succeeded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit(run_dir: Path, output_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {task["task_id"]: task for task in manifest["tasks"]}
    rows = []
    for task_id, task in expected.items():
        result_path = run_dir / "runs" / f"{task_id}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            status = result.get("status", "malformed")
            if result.get("manifest_identity") != manifest["identity_sha256"]:
                status = "identity_mismatch"
        else:
            result = {}
            status = "missing"
        rows.append({**task, "status": status, "accuracy": result.get("accuracy"),
                     "f1": result.get("f1"), "auroc": result.get("auroc"),
                     "epsilon_achieved": result.get("epsilon_achieved"),
                     "training_time_seconds": result.get("training_time_seconds"),
                     "setup_time_seconds": result.get("setup_time_seconds"),
                     "inference_time_seconds": result.get("inference_time_seconds"),
                     "forward_milliseconds_per_record": result.get("forward_milliseconds_per_record"),
                     "estimated_forward_macs_per_record": result.get("estimated_forward_macs_per_record"),
                     "parameter_storage_bytes": result.get("parameter_storage_bytes"),
                     "trainable_parameters": result.get("trainable_parameters"),
                     "majority_baseline_test_accuracy": result.get("majority_baseline_test_accuracy"),
                     "error": result.get("error")})
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "coverage_and_metrics.csv", index=False)
    successful = frame.loc[frame.status == "success"].copy()
    if not successful.empty:
        successful["accuracy_minus_majority_baseline"] = successful.accuracy - successful.majority_baseline_test_accuracy
        successful["setup_plus_training_seconds"] = successful.setup_time_seconds.fillna(0) + successful.training_time_seconds
        successful.to_csv(output_dir / "utility_and_cost_per_run.csv", index=False)
        grouped = successful.groupby(["dataset", "model_key", "privacy", "epsilon", "dpelm_width"], dropna=False)
        summary = grouped.agg(n_runs=("accuracy", "size"), accuracy_mean=("accuracy", "mean"),
                              accuracy_sd=("accuracy", "std"),
                              accuracy_minus_majority_mean=("accuracy_minus_majority_baseline", "mean"),
                              training_seconds_mean=("training_time_seconds", "mean"),
                              setup_plus_training_seconds_mean=("setup_plus_training_seconds", "mean"),
                              forward_milliseconds_per_record_mean=("forward_milliseconds_per_record", "mean"),
                              forward_macs_per_record_mean=("estimated_forward_macs_per_record", "mean"),
                              trainable_parameters_mean=("trainable_parameters", "mean")).reset_index()
        summary.to_csv(output_dir / "privacy_mode_separated_summary.csv", index=False)
        # A private table and a non-private figure may describe the same family,
        # but they must never be treated as values from the same experiment mode.
        provenance = successful[["dataset", "model_key", "privacy", "epsilon", "dpelm_width",
                                 "seed", "task_id", "accuracy"]].copy()
        provenance["recommended_label"] = provenance.privacy.map({
            "private": "PRIVATE, target epsilon shown; per-run budget",
            "non_private": "NON-PRIVATE, no epsilon"})
        provenance.to_csv(output_dir / "figure_table_provenance.csv", index=False)
    counts = frame.status.value_counts().to_dict()
    result = {"expected": len(expected), "status_counts": counts,
              "complete": counts.get("success", 0) == len(expected),
              "smoke": bool(manifest.get("configuration", {}).get("smoke", False)),
              "development": bool(manifest.get("configuration", {}).get("development", False)),
              "manifest_identity_sha256": manifest["identity_sha256"],
              "report_scope": "public benchmark, fixed test partition; not clinical external validation"}
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Reviewer experiment audit", "",
             f"Expected tasks: {len(expected)}. Successful: {counts.get('success', 0)}. "
             f"Failed: {counts.get('failed', 0)}. Missing: {counts.get('missing', 0)}.", "",
             "The summary separates private and non-private rows; Figure 2 from the submitted manuscript "
             "must not be numerically compared with private Table 1 without an explicit mode label.",
             "Accuracy uncertainty across seeds is conditional on one fixed split, not external clinical validation.",
             "Training time excludes setup; use setup-plus-training for end-to-end comparisons. "
             "Forward latency is only available when --benchmark-diagnostics was enabled.",
             "Compute time, MAC estimates and parameter count do not establish causality or lower privacy epsilon.", ""]
    if not result["complete"]:
        lines.append("**Incomplete: do not use aggregate values as final manuscript results.**")
    if result["smoke"] or result["development"]:
        lines.append("**Smoke/development protocol: execution check only, not manuscript evidence.**")
    (output_dir / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir or run_dir / "reviewer_audit"
    print(json.dumps(audit(run_dir, output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
