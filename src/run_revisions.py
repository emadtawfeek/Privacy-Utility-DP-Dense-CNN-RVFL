"""Reviewer revision experiments with explicit assumptions and immutable manifests.

Run from the project root: py -3.12 src/run_revisions.py --help
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import pandas as pd
import torch

from config import DATASET_SETTINGS, PROJECT_ROOT, configure_runtime, set_random_seed
from dp_accounting import validate_secure_rng_available
from dpelm import DPELM
from evaluate import evaluate_from_probabilities, logits_to_probabilities, predict_logits
from models import build_model, count_parameters
from revision_data import load_revision_bundle
from revision_statistics import write_analysis, write_figures
from revision_training import train_private_model
from train_nonprivate import make_loader, train_nonprivate_model

MODEL_KEYS = ["dense", "cnn", "rvfl", "elm", "frozen_dense", "logistic", "logistic_raw", "dpelm"]
PROTOCOL_VERSION = "reviewer_revision_v1"


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def _list(value, cast=str):
    return list(dict.fromkeys(cast(item.strip()) for item in value.split(",") if item.strip()))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default="mnist,cifar10,cardio,heart")
    p.add_argument("--models", default=",".join(MODEL_KEYS))
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--epsilons", default="0.1,0.5,1,2,4,8")
    p.add_argument("--privacy", choices=["both", "private", "non_private"], default="both")
    p.add_argument("--dpelm-widths", default="1,3,5,10")
    p.add_argument("--random-features", type=int, default=200)
    p.add_argument("--max-input-dim", type=int, default=1024)
    p.add_argument("--activation", choices=["tanh", "relu", "sigmoid", "sin", "radbas"], default="tanh")
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--epochs-image", type=int, default=10)
    p.add_argument("--epochs-tabular", type=int, default=30)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--accountant", choices=["rdp", "prv"], default="rdp")
    p.add_argument("--rdp-alpha-mode", choices=["wide", "default"], default="wide")
    p.add_argument("--epsilon-tolerance", type=float, default=1e-4)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs_reviewer_revision")
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-test-samples", type=int)
    p.add_argument("--development", action="store_true", help="Seeded nonsecure noise; never label as production evidence")
    p.add_argument("--smoke", action="store_true", help="One epoch, small data, one seed, epsilon 2 unless explicitly overridden")
    p.add_argument("--dry-run", action="store_true", help="Validate data and export a complete experiment manifest without training")
    p.add_argument("--resume", action="store_true", help="Reuse successful tasks only if configuration, code and data fingerprints match")
    return p


def validate(args):
    for name, cast in [("datasets", str), ("models", str), ("seeds", int), ("epsilons", float), ("dpelm_widths", int)]:
        setattr(args, name, _list(getattr(args, name), cast))
        if not getattr(args, name):
            raise ValueError(f"{name} cannot be empty")
    if set(args.datasets) - set(DATASET_SETTINGS) or set(args.models) - set(MODEL_KEYS):
        raise ValueError("Unknown dataset or model key.")
    if any(not math.isfinite(e) or e <= args.epsilon_tolerance for e in args.epsilons):
        raise ValueError("Epsilons must be finite and exceed the positive calibration tolerance.")
    if not (0 < args.delta < 1 and args.epsilon_tolerance > 0):
        raise ValueError("Invalid delta or calibration tolerance.")
    for name in ["epochs_image", "epochs_tabular", "random_features", "max_input_dim", "torch_threads"]:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ["batch_size", "max_train_samples", "max_test_samples"]:
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if min(args.dpelm_widths) < 1 or not (math.isfinite(args.lr) and args.lr > 0 and
            math.isfinite(args.max_grad_norm) and args.max_grad_norm > 0 and
            math.isfinite(args.weight_decay) and args.weight_decay >= 0 and
            math.isfinite(args.momentum) and 0 <= args.momentum < 1):
        raise ValueError("Invalid training settings.")
    if args.smoke:
        args.epochs_image = args.epochs_tabular = 1
        args.max_train_samples = args.max_train_samples or 128
        args.max_test_samples = args.max_test_samples or 64
        args.seeds = args.seeds[:1]
        if args.epsilons == [.1, .5, 1., 2., 4., 8.]:
            args.epsilons = [2.]
    return args


def tasks_for(args):
    modes = ["non_private", "private"] if args.privacy == "both" else [args.privacy]
    tasks = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for key in args.models:
                for width in args.dpelm_widths if key == "dpelm" else [None]:
                    for mode in modes:
                        for epsilon in args.epsilons if mode == "private" else [None]:
                            task = {"dataset": dataset, "seed": seed, "model_key": key,
                                    "dpelm_width": width, "privacy": mode, "epsilon": epsilon}
                            digest = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:16]
                            task["task_id"] = f"{dataset}_{key}_{digest}"
                            tasks.append(task)
    return tasks


def describe_model(model, task, bundle, args):
    key = task["model_key"]
    epochs = args.epochs_image if bundle.task_type == "multiclass" else args.epochs_tabular
    batch_size = args.batch_size or DATASET_SETTINGS[bundle.dataset]["batch_size"]
    hp = {
        "optimizer": args.optimizer, "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "batch_size_requested": batch_size, "batch_size_effective_base": min(batch_size, len(bundle.train_dataset)),
        "epochs": epochs, "early_stopping": "disabled", "checkpoint": "final_fixed_epoch",
        "validation_access": "none", "hyperparameter_selection": "prespecified_config_no_search",
        "threshold": .5 if bundle.task_type == "binary" else "argmax",
        "loss": "unweighted_BCEWithLogitsLoss" if bundle.task_type == "binary" else "CrossEntropyLoss",
        "dropout": .2 if key == "dense" else 0.0,
        "initialization": "PyTorch layer defaults; frozen RVFL weights U(-1,1), bias U(0,1)",
        "sgd_momentum": args.momentum if args.optimizer == "sgd" else 0.0,
        "adam_betas": "0.9,0.999" if args.optimizer == "adam" else "not_applicable",
        "learning_rate_schedule": "constant", "max_input_dim": args.max_input_dim,
        "hidden_width": getattr(model, "n_random_features", None),
        "activation": getattr(model, "activation_name", "ReLU" if key in {"dense", "cnn"} else "none"),
        "input_dim_original": getattr(model, "original_input_dim", bundle.feature_count),
        "input_dim_after_projection": getattr(model, "reduced_input_dim", bundle.feature_count),
        "architecture": str(model), "split_seed": args.split_seed,
        "max_grad_norm": args.max_grad_norm,
    }
    if key == "dpelm":
        hp.update(optimizer="closed_form_solve", learning_rate=None, weight_decay=None,
                  batch_size_effective_base=min(1024, len(bundle.train_dataset)), batch_size_requested=1024,
                  epochs=0, loss="one_hot_least_squares", initialization="independent_N(0,1)_weights_and_bias",
                  hidden_width=task["dpelm_width"], activation="sigmoid", max_input_dim=None,
                  threshold="binary_softmax_class1_ge_0.5_otherwise_argmax", max_grad_norm=None,
                  sgd_momentum=None, adam_betas=None, learning_rate_schedule=None,
                  probability_mapping="softmax_scores_for_rank_metrics_only_not_calibrated_probabilities",
                  regularization="none", numerical_solver="solve_with_pinv_only_if_singular")
    return hp


def run_task(task, bundle, args):
    set_random_seed(task["seed"])
    private = task["privacy"] == "private"
    key = task["model_key"]
    row = task | {"epsilon_target": task["epsilon"], "status": "failed", "error": "",
                  "protocol_version": PROTOCOL_VERSION, "development": args.development, "smoke": args.smoke,
                  "n_train": len(bundle.train_dataset), "n_validation": len(bundle.validation_dataset),
                  "n_test": len(bundle.test_dataset), "max_grad_norm": args.max_grad_norm if private else None,
                  "adjacency": "add_remove_one" if private else "not_applicable",
                  "privacy_type": "approximate_dp" if private else "non_private",
                  "privacy_scope": bundle.dataset_summary["privacy_scope"],
                  "validation_use": "none", "test_scope": "public_benchmark_only",
                  "delta": args.delta if private else None}
    if key == "dpelm":
        model = DPELM(bundle.input_shape, bundle.num_classes, task["dpelm_width"], task["seed"])
        name = ("DPELM" if private else "ELMClosedForm") + f"-L{task['dpelm_width']}"
        learned = task["dpelm_width"] * bundle.num_classes
        trainable, total = 0, sum(b.numel() for b in model.buffers())
        frozen = total - learned
    else:
        model, spec = build_model(model_key=key, dataset=bundle.dataset, input_shape=bundle.input_shape,
            task_type=bundle.task_type, num_classes=bundle.num_classes, seed=task["seed"],
            privacy=task["privacy"], n_random_features=args.random_features,
            rvfl_activation=args.activation, max_input_dim=args.max_input_dim)
        name = spec.model_name
        trainable, total = count_parameters(model)
        frozen, learned = total-trainable, trainable
    hp = describe_model(model, task, bundle, args)
    row.update(model=name, trainable_parameters=trainable, total_parameters=total,
               frozen_parameters=frozen, fitted_coefficients=learned,
               parameter_count_definition="DPELM total includes buffers; fitted coefficients are solved, not gradient-trained",
               hyperparameters=json.dumps(hp, sort_keys=True))
    if key == "dpelm":
        row.update(model.fit(bundle.train_dataset, epsilon=task["epsilon"], seed=task["seed"], secure=not args.development))
    elif private:
        result = train_private_model(model=model, train_dataset=bundle.train_dataset, val_loader=None, test_loader=None,
            task_type=bundle.task_type, epsilon=task["epsilon"], delta=args.delta,
            max_grad_norm=args.max_grad_norm, epochs=hp["epochs"], lr=args.lr, seed=task["seed"],
            batch_size=hp["batch_size_requested"], num_workers=0, secure_mode=not args.development,
            rdp_alpha_mode=args.rdp_alpha_mode, optimizer_name=args.optimizer,
            sgd_momentum=args.momentum, weight_decay=args.weight_decay, accountant_name=args.accountant,
            epsilon_tolerance=args.epsilon_tolerance, public_reference_size=len(bundle.train_dataset))
        row.update(asdict(result))
    else:
        row.update(asdict(train_nonprivate_model(model=model, train_dataset=bundle.train_dataset,
            task_type=bundle.task_type, batch_size=hp["batch_size_requested"], epochs=hp["epochs"],
            lr=args.lr, seed=task["seed"], num_workers=0, optimizer_name=args.optimizer,
            weight_decay=args.weight_decay, sgd_momentum=args.momentum)))
    test_loader = make_loader(bundle.test_dataset, min(1024, hp["batch_size_requested"]), False, task["seed"], 0)
    logits, targets, inference = predict_logits(model, test_loader)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Non-finite test predictions.")
    if key == "dpelm":
        probabilities = logits_to_probabilities(logits, "multiclass")
        if bundle.task_type == "binary":
            probabilities = probabilities[:, 1]
    else:
        probabilities = logits_to_probabilities(logits, bundle.task_type)
    metrics, notes = evaluate_from_probabilities(targets, probabilities, task_type=bundle.task_type,
                                                num_classes=bundle.num_classes, threshold=.5)
    row.update(metrics, inference_time_seconds=inference, status="success", metric_notes=notes)
    row["total_training_seconds_including_setup"] = row["training_time_seconds"] + row.get("setup_time_seconds", 0.)
    return row


def main(argv=None):
    args = validate(parser().parse_args(argv))
    configure_runtime(args.torch_threads)
    if not args.development and args.privacy != "non_private" and any(k != "dpelm" for k in args.models):
        validate_secure_rng_available()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output directory is not empty. Choose a new directory or --resume with the identical protocol.")
    bundles = {dataset: load_revision_bundle(dataset, args.data_dir, args.split_seed,
               args.max_train_samples, args.max_test_samples) for dataset in args.datasets}
    tasks = tasks_for(args)
    config = {key: value for key, value in vars(args).items() if key not in {"resume", "dry_run", "output_dir"}}
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    datasets = {key: value.dataset_summary for key, value in bundles.items()}
    environment = {name: importlib.metadata.version(name) for name in
                   ["torch", "torchvision", "opacus", "numpy", "pandas", "scipy", "scikit-learn"]}
    manifest_identity = {"protocol": PROTOCOL_VERSION, "configuration": json_safe(config),
                         "source_sha256": code_hashes, "datasets": datasets, "environment": environment,
                         "python": sys.version, "platform": platform.platform()}
    identity = hashlib.sha256(json.dumps(manifest_identity, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text())['identity_sha256'] != identity:
            raise ValueError("Resume refused: configuration, code, data, or partition fingerprint changed.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs").mkdir(exist_ok=True)
    manifest = manifest_identity | {"identity_sha256": identity, "tasks": tasks, "task_count": len(tasks),
        "created_utc": datetime.now(timezone.utc).isoformat(), "environment": environment,
        "python": sys.version, "platform": platform.platform(),
        "publication_scope": "public_benchmark_evaluation; per-model training budgets; not a jointly private sweep",
        "DPELM_note": "paper Algorithm 1 research implementation; pure-DP theorem is ideal arithmetic, not finite-precision certification",
        "selection_note": "No private validation, early stopping, threshold tuning, or automatic best-model selection",
        "composition_note": "Multiple releases on overlapping protected records require composition; never advertise each-run epsilon for the full sweep"}
    if not args.resume:
        save_json(manifest_path, manifest)
    print(f"{len(tasks)} prespecified tasks. Output: {output}", flush=True)
    print("Privacy scope: one training run conditional on fixed public cohort/partitions/sizes. Test evaluation is public.", flush=True)
    if args.dry_run:
        pd.DataFrame(tasks).to_csv(output / "experiment_plan.csv", index=False)
        print("Dry run complete; no model trained.")
        return 0
    results = []
    for index, task in enumerate(tasks, 1):
        path = output / "runs" / (task["task_id"] + ".json")
        previous = json.loads(path.read_text()) if args.resume and path.exists() else None
        if previous and previous.get("status") == "success" and previous.get("manifest_identity") == identity:
            row = previous
            print(f"[{index}/{len(tasks)}] reuse {task['task_id']}", flush=True)
        else:
            print(f"[{index}/{len(tasks)}] {task['dataset']} {task['model_key']} {task['privacy']} epsilon={task['epsilon']} width={task['dpelm_width']}", flush=True)
            try:
                row = run_task(task, bundles[task["dataset"]], args)
            except Exception as error:
                row = task | {"epsilon_target": task["epsilon"], "status": "failed", "error": f"{type(error).__name__}: {error}"}
                path.with_suffix(".error.log").write_text(traceback.format_exc(), encoding="utf-8")
                print(row["error"], flush=True)
            row["manifest_identity"] = identity
            save_json(path, row)
        results.append(row)
    write_analysis(results, output / "tables")
    write_figures(results, output / "figures")
    failed = sum(row["status"] != "success" for row in results)
    print(f"Finished: {len(results)-failed} successful; {failed} failed. Full-run results are not implied by smoke tests.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
