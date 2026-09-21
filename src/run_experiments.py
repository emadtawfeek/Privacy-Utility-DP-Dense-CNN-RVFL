"""Run the four-dataset privacy-utility architecture comparison."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


def _early_thread_count() -> int:
    default = max(1, min(8, os.cpu_count() or 1))
    for index, argument in enumerate(sys.argv):
        if argument == "--torch_threads" and index + 1 < len(sys.argv):
            try:
                return max(1, int(sys.argv[index + 1]))
            except ValueError:
                return default
        if argument.startswith("--torch_threads="):
            try:
                return max(1, int(argument.split("=", 1)[1]))
            except ValueError:
                return default
    return default


for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = str(_early_thread_count())


import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import (
    ABADI2016_DATASETS,
    ABADI2016_EPSILONS,
    ABADI2016_MODELS,
    ABADI2016_SETTINGS,
    CLIPPING_NORMS,
    DATASET_SETTINGS,
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    EPSILONS,
    MAX_GRAD_NORM,
    PRIMARY_EPSILON,
    PROJECT_ROOT,
    QUICK_MAX_TEST_SAMPLES,
    QUICK_MAX_TRAIN_SAMPLES,
    QUICK_TEST_EPSILONS,
    RESULT_COLUMNS,
    configure_runtime,
    ensure_output_dirs,
    format_number_token,
    get_rdp_alphas,
    parse_seeds,
    set_random_seed,
)
from data_loader import load_dataset_bundle
from evaluate import (
    evaluate_from_probabilities,
    logits_to_probabilities,
    predict_logits,
    select_f1_threshold,
)
from models import build_model, count_parameters
from paper_profiles import abadi_pca_noise_for_epsilon, fit_abadi_mnist_noisy_pca
from train_nonprivate import make_loader, train_nonprivate_model
from train_private import train_private_model


LOGGER = logging.getLogger("architecture_dp_experiments")


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _configure_logging(log_file: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def _parse_list(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default.copy()
    parsed = [item.strip().lower() for item in value.split(",") if item.strip()]
    return parsed or default.copy()


def _parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if value is None:
        return default.copy()
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("List argument must contain at least one number.")
    return parsed


def _batch_size(dataset: str, args: argparse.Namespace) -> int:
    if args.paper_profile == "abadi2016":
        settings = ABADI2016_SETTINGS[dataset]
        if dataset == "cifar10":
            return int(settings["batch_size_by_epsilon"].get(8.0, 4000))
        return int(settings["batch_size"])
    if args.batch_size is not None:
        return args.batch_size
    return int(DATASET_SETTINGS[dataset]["batch_size"])


def _epochs(dataset: str, args: argparse.Namespace) -> int:
    if args.quick_test:
        if args.paper_profile == "abadi2016":
            return int(ABADI2016_SETTINGS[dataset]["quick_epochs"])
        return 1
    if args.paper_profile == "abadi2016":
        return int(ABADI2016_SETTINGS[dataset]["epochs"])
    if dataset in {"mnist", "cifar10"}:
        return args.epochs_image
    return args.epochs_tabular


def _delta(dataset: str, n_train: int) -> float:
    return min(1e-5, 1.0 / (2.0 * max(n_train, 1)))


def _task_filename(task: dict[str, Any]) -> str:
    epsilon = task.get("epsilon")
    clip = task.get("max_grad_norm")
    privacy = task["privacy"]
    profile = task.get("paper_profile", "default")
    profile_token = "" if profile == "default" else f"{profile}_"
    return (
        f"run_{profile_token}{task['dataset']}_seed{task['seed']}_{task['model_key']}_{privacy}_"
        f"eps{format_number_token(epsilon)}_clip{format_number_token(clip)}.csv"
    )


def _base_result(task: dict[str, Any]) -> dict[str, Any]:
    is_private = task["privacy"] == "private"
    return {
        "dataset": task["dataset"],
        "seed": task["seed"],
        "model": "",
        "model_key": task["model_key"],
        "architecture_family": "",
        "architecture_variant": "",
        "privacy": "differential_private" if is_private else "non_private",
        "task_type": "",
        "num_classes": np.nan,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "auroc": np.nan,
        "auprc": np.nan,
        "training_time_seconds": np.nan,
        "inference_time_seconds": np.nan,
        "epsilon_target": task.get("epsilon"),
        "epsilon_achieved": np.nan,
        "target_epsilon": task.get("epsilon"),
        "achieved_epsilon": np.nan,
        "delta": np.nan,
        "clipping_norm": task.get("max_grad_norm") if is_private else np.nan,
        "max_grad_norm": task.get("max_grad_norm") if is_private else np.nan,
        "noise_multiplier": np.nan,
        "secure_rng_used": task.get("secure_mode") if is_private else np.nan,
        "accountant": "",
        "rdp_alpha_mode": task.get("rdp_alpha_mode", "") if is_private else "",
        "alpha_count": task.get("alpha_count", np.nan) if is_private else np.nan,
        "n_train": np.nan,
        "n_validation": np.nan,
        "n_test": np.nan,
        "input_shape": "",
        "feature_count": np.nan,
        "trainable_parameters": np.nan,
        "total_parameters": np.nan,
        "threshold": np.nan,
        "epochs_completed": np.nan,
        "private_setup_method": "",
        "n_random_features": task.get("rvfl_random_features", np.nan),
        "activation": task.get("rvfl_activation", ""),
        "selected_hyperparameters": "",
        "paper_profile": task.get("paper_profile", "default"),
        "reference_source": task.get("reference_source", ""),
        "configuration_match": task.get("configuration_match", ""),
        "abadi_reference_accuracy": task.get("abadi_reference_accuracy", np.nan),
        "best_validation_auroc": np.nan,
        "status": "failed",
        "run_status": "failed",
        "error_message": "",
    }


def _write_result(result: dict[str, Any], temp_path: str | Path) -> None:
    frame = pd.DataFrame([{column: result.get(column, np.nan) for column in RESULT_COLUMNS}])
    frame.to_csv(temp_path, index=False)


def _write_table_with_fallback(frame: pd.DataFrame, path: Path, run_id: str) -> Path:
    try:
        frame.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{run_id}{path.suffix}")
        frame.to_csv(fallback, index=False)
        LOGGER.warning("Could not overwrite locked file %s; wrote %s.", path, fallback)
        return fallback


def _task_has_readable_result(task: dict[str, Any]) -> bool:
    path = Path(task["temp_path"])
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        pd.read_csv(path, nrows=1)
    except Exception:
        return False
    return True


def _log_task_progress(index: int, total: int, task: dict[str, Any], prefix: str = "") -> None:
    LOGGER.info(
        "%s[%d/%d] dataset=%s seed=%s model=%s privacy=%s epsilon=%s",
        prefix,
        index,
        total,
        task["dataset"],
        task["seed"],
        task["model_key"],
        task["privacy"],
        task["epsilon"],
    )


def _run_tasks_sequentially(
    tasks: list[dict[str, Any]],
    *,
    skip_completed: bool,
    prefix: str = "",
) -> None:
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        if skip_completed and _task_has_readable_result(task):
            _log_task_progress(index, total, task, prefix=f"{prefix}skip completed ")
            continue
        _log_task_progress(index, total, task, prefix=prefix)
        run_experiment_task(task)


def _run_tasks_in_parallel(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    batch_size = args.parallel_batch_size or max(args.n_jobs, args.n_jobs * 2)
    batch_size = max(1, int(batch_size))
    completed_before = sum(_task_has_readable_result(task) for task in tasks)
    if completed_before:
        LOGGER.info(
            "Resume/skip mode: %d/%d configurations already have readable result files.",
            completed_before,
            len(tasks),
        )

    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        pending = [task for task in batch if not _task_has_readable_result(task)]
        batch_label = f"{start + 1}-{start + len(batch)}"
        if not pending:
            LOGGER.info("Parallel batch %s skipped; all result files already exist.", batch_label)
            continue
        LOGGER.info(
            "Parallel batch %s: running %d pending configurations with n_jobs=%d.",
            batch_label,
            len(pending),
            args.n_jobs,
        )
        try:
            Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
                delayed(run_experiment_task)(task) for task in pending
            )
        except Exception as error:
            missing = [task for task in pending if not _task_has_readable_result(task)]
            LOGGER.warning(
                "Parallel batch %s failed with %s: %s. Keeping completed result "
                "files and rerunning %d missing configurations sequentially.",
                batch_label,
                type(error).__name__,
                error,
                len(missing),
            )
            _run_tasks_sequentially(
                missing,
                skip_completed=True,
                prefix="recovery ",
            )


def _read_result_files(result_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    skipped: list[Path] = []
    for path in result_files:
        try:
            frames.append(pd.read_csv(path))
        except Exception as error:
            skipped.append(path)
            LOGGER.warning("Skipping unreadable result file %s: %s", path, error)
    if not frames:
        raise RuntimeError("No readable result files were produced.")
    if skipped:
        LOGGER.warning("Skipped %d unreadable result files during merge.", len(skipped))
    return pd.concat(frames, ignore_index=True)


def run_experiment_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run one independent configuration and write one result row."""
    result = _base_result(task)
    temp_path = Path(task["temp_path"])
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        configure_runtime(task["torch_threads"])
        set_random_seed(task["seed"])
        bundle = load_dataset_bundle(
            task["dataset"],
            seed=task["seed"],
            cardio_path=task["cardio_path"],
            heart_path=task["heart_path"],
            data_dir=task["data_dir"],
            max_train_samples=task["max_train_samples"],
            max_test_samples=task["max_test_samples"],
            image_validation_fraction=task["image_validation_fraction"],
        )
        batch_size = task["batch_size"]
        train_loader = make_loader(
            bundle.train_dataset,
            batch_size,
            True,
            task["seed"],
            task["num_workers"],
        )
        val_loader = make_loader(
            bundle.validation_dataset,
            batch_size,
            False,
            task["seed"],
            task["num_workers"],
        )
        test_loader = make_loader(
            bundle.test_dataset,
            batch_size,
            False,
            task["seed"],
            task["num_workers"],
        )
        pca_projection = None
        if task["paper_profile"] == "abadi2016" and task["dataset"] == "mnist":
            pca_projection = fit_abadi_mnist_noisy_pca(
                bundle.train_dataset,
                seed=task["seed"],
                components=task["pca_components"],
                noise_sigma=task["pca_noise_sigma"],
                cache_dir=task["paper_profile_cache_dir"],
            )
        model, spec = build_model(
            model_key=task["model_key"],
            dataset=task["dataset"],
            input_shape=bundle.input_shape,
            task_type=bundle.task_type,
            num_classes=bundle.num_classes,
            seed=task["seed"],
            privacy=task["privacy"],
            n_random_features=task["rvfl_random_features"],
            rvfl_activation=task["rvfl_activation"],
            paper_profile=task["paper_profile"],
            pca_projection=pca_projection,
        )
        trainable_parameters, total_parameters = count_parameters(model)
        result.update(
            {
                "model": spec.model_name,
                "architecture_family": spec.architecture_family,
                "architecture_variant": spec.architecture_variant,
                "task_type": bundle.task_type,
                "num_classes": bundle.num_classes,
                "n_train": len(bundle.train_dataset),
                "n_validation": len(bundle.validation_dataset),
                "n_test": len(bundle.test_dataset),
                "input_shape": str(bundle.input_shape),
                "feature_count": bundle.feature_count,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
            }
        )
        selected_hyperparameters = {
            "epochs": task["epochs"],
            "batch_size": batch_size,
            "lr": task["lr"],
            "optimizer": task["optimizer_name"],
            "sgd_momentum": task["sgd_momentum"],
            "paper_profile": task["paper_profile"],
            "reference_source": task["reference_source"],
            "configuration_match": task["configuration_match"],
        }
        if task["lr_schedule"]:
            selected_hyperparameters["lr_schedule"] = task["lr_schedule"]
        if task["paper_profile"] == "abadi2016":
            selected_hyperparameters.update(
                {
                    "abadi_reference_accuracy": task["abadi_reference_accuracy"],
                    "abadi_reference_note": task["paper_profile_notes"],
                }
            )
            if task["dataset"] == "mnist":
                selected_hyperparameters.update(
                    {
                        "pca_components": task["pca_components"],
                        "pca_noise_sigma": task["pca_noise_sigma"],
                        "dp_pca_implemented": True,
                    }
                )
            if task["dataset"] == "cifar10":
                selected_hyperparameters.update(
                    {
                        "center_crop": "24x24",
                        "public_cifar100_pretraining_implemented": False,
                    }
                )
        if task["model_key"] == "rvfl":
            selected_hyperparameters.update(
                {
                    "n_random_features": task["rvfl_random_features"],
                    "activation": task["rvfl_activation"],
                    "original_input_dim": getattr(model, "original_input_dim", np.nan),
                    "reduced_input_dim": getattr(model, "reduced_input_dim", np.nan),
                }
            )
        result["selected_hyperparameters"] = json.dumps(
            selected_hyperparameters,
            sort_keys=True,
            default=_json_safe,
        )

        if task["privacy"] == "private":
            delta = _delta(task["dataset"], len(bundle.train_dataset))
            result["delta"] = delta
            train_result = train_private_model(
                model=model,
                train_dataset=bundle.train_dataset,
                val_loader=val_loader,
                test_loader=test_loader,
                task_type=bundle.task_type,
                epsilon=task["epsilon"],
                delta=delta,
                max_grad_norm=task["max_grad_norm"],
                epochs=task["epochs"],
                lr=task["lr"],
                seed=task["seed"],
                batch_size=batch_size,
                num_workers=task["num_workers"],
                secure_mode=task["secure_mode"],
                rdp_alpha_mode=task["rdp_alpha_mode"],
                optimizer_name=task["optimizer_name"],
                sgd_momentum=task["sgd_momentum"],
                lr_schedule=task["lr_schedule"],
            )
            result.update(
                {
                    "training_time_seconds": train_result.training_time_seconds,
                    "epochs_completed": train_result.epochs_completed,
                    "epsilon_achieved": train_result.epsilon_achieved,
                    "achieved_epsilon": train_result.epsilon_achieved,
                    "delta": delta,
                    "noise_multiplier": train_result.noise_multiplier,
                    "private_setup_method": train_result.private_setup_method,
                    "secure_rng_used": train_result.secure_rng_used,
                    "accountant": train_result.accountant,
                    "rdp_alpha_mode": train_result.rdp_alpha_mode,
                    "alpha_count": train_result.alpha_count,
                }
            )
        else:
            train_result = train_nonprivate_model(
                model=model,
                train_dataset=bundle.train_dataset,
                task_type=bundle.task_type,
                batch_size=batch_size,
                epochs=task["epochs"],
                lr=task["lr"],
                seed=task["seed"],
                num_workers=task["num_workers"],
                optimizer_name=task["optimizer_name"],
                lr_schedule=task["lr_schedule"],
            )
            result.update(
                {
                    "training_time_seconds": train_result.training_time_seconds,
                    "epochs_completed": train_result.epochs_completed,
                }
            )

        val_logits, val_targets, _ = predict_logits(model, val_loader)
        test_logits, test_targets, inference_time = predict_logits(model, test_loader)
        val_probabilities = logits_to_probabilities(val_logits, bundle.task_type)
        test_probabilities = logits_to_probabilities(test_logits, bundle.task_type)
        threshold = (
            select_f1_threshold(val_targets, val_probabilities)
            if bundle.task_type == "binary"
            else np.nan
        )
        metrics, metric_notes = evaluate_from_probabilities(
            test_targets,
            test_probabilities,
            task_type=bundle.task_type,
            num_classes=bundle.num_classes,
            threshold=0.5 if np.isnan(threshold) else threshold,
        )
        result.update(metrics)
        result.update(
            {
                "inference_time_seconds": inference_time,
                "threshold": threshold,
                "status": "success",
                "run_status": "success",
                "error_message": metric_notes,
            }
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        result["error_message"] = message
        result["status"] = (
            "infeasible"
            if "privacy budget is too low" in message.lower()
            or "infeasible" in message.lower()
            else "failed"
        )
        result["run_status"] = "failed"
        error_log = temp_path.with_suffix(".error.log")
        try:
            error_log.write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            result["error_message"] += " (exception trace could not be written)"
    finally:
        _write_result(result, temp_path)
    return result


def _build_tasks(
    *,
    datasets: list[str],
    models: list[str],
    privacy: str,
    epsilons: list[float],
    seeds: list[int],
    args: argparse.Namespace,
    temp_dir: Path,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    privacy_modes = (
        ["non_private", "private"]
        if privacy == "both"
        else ["non_private" if privacy == "non_private" else "private"]
    )
    for dataset in datasets:
        profile_settings = (
            ABADI2016_SETTINGS[dataset] if args.paper_profile == "abadi2016" else None
        )
        for seed in seeds:
            for model_key in models:
                for privacy_mode in privacy_modes:
                    if privacy_mode == "private":
                        dataset_epsilons = (
                            [float(epsilon) for epsilon in profile_settings["epsilons"]]
                            if profile_settings is not None
                            and args.epsilons is None
                            and not args.quick_test
                            else epsilons
                        )
                        epsilon_clip_pairs = [
                            (
                                epsilon,
                                (
                                    float(profile_settings["max_grad_norm"])
                                    if profile_settings is not None
                                    else args.max_grad_norm
                                ),
                            )
                            for epsilon in dataset_epsilons
                        ]
                        if args.run_clipping_sensitivity:
                            epsilon_clip_pairs.extend(
                                (PRIMARY_EPSILON, clipping_norm)
                                for clipping_norm in CLIPPING_NORMS
                            )
                        epsilon_clip_pairs = list(dict.fromkeys(epsilon_clip_pairs))
                    else:
                        epsilon_clip_pairs = [(None, np.nan)]
                    for epsilon, max_grad_norm in epsilon_clip_pairs:
                        batch_size = _batch_size(dataset, args)
                        lr = args.lr
                        optimizer_name = "adam"
                        sgd_momentum = 0.9
                        lr_schedule = None
                        pca_components = np.nan
                        pca_noise_sigma = np.nan
                        reference_source = ""
                        configuration_match = "project_default"
                        abadi_reference_accuracy = np.nan
                        paper_profile_notes = ""
                        if profile_settings is not None:
                            lr = float(profile_settings["lr"])
                            optimizer_name = str(profile_settings["optimizer"])
                            sgd_momentum = 0.0
                            lr_schedule = profile_settings["lr_schedule"]
                            reference_source = "Abadi et al. 2016, Deep Learning with Differential Privacy"
                            paper_profile_notes = str(profile_settings["notes"])
                            reported = profile_settings["reported_accuracy_by_epsilon"]
                            if epsilon is not None:
                                abadi_reference_accuracy = reported.get(float(epsilon), np.nan)
                            if dataset == "mnist":
                                batch_size = int(profile_settings["batch_size"])
                                pca_components = int(profile_settings["pca_components"])
                                pca_noise_sigma = (
                                    0.0
                                    if privacy_mode == "non_private"
                                    else float(
                                        profile_settings["pca_noise_by_epsilon"].get(
                                            float(epsilon),
                                            abadi_pca_noise_for_epsilon(float(epsilon)),
                                        )
                                    )
                                )
                                configuration_match = (
                                    "abadi2016_mnist_reported_pca_mlp_hyperparameters"
                                )
                            elif dataset == "cifar10":
                                if epsilon is not None:
                                    batch_size = int(
                                        profile_settings["batch_size_by_epsilon"].get(
                                            float(epsilon),
                                            4000,
                                        )
                                    )
                                configuration_match = (
                                    "abadi2016_cifar10_architecture_hyperparameters_"
                                    "without_cifar100_public_pretraining"
                                )
                        task = {
                            "dataset": dataset,
                            "seed": seed,
                            "model_key": model_key,
                            "privacy": privacy_mode,
                            "epsilon": epsilon,
                            "max_grad_norm": max_grad_norm,
                            "epochs": _epochs(dataset, args),
                            "batch_size": batch_size,
                            "lr": lr,
                            "optimizer_name": optimizer_name,
                            "sgd_momentum": sgd_momentum,
                            "lr_schedule": lr_schedule,
                            "num_workers": args.num_workers,
                            "torch_threads": args.torch_threads,
                            "cardio_path": str(_project_path(args.cardio_path)),
                            "heart_path": str(_project_path(args.heart_path)),
                            "data_dir": str(_project_path(args.data_dir)),
                            "max_train_samples": args.max_train_samples,
                            "max_test_samples": args.max_test_samples,
                            "image_validation_fraction": args.image_validation_fraction,
                            "rvfl_random_features": args.rvfl_random_features,
                            "rvfl_activation": args.rvfl_activation,
                            "secure_mode": args.secure_rng,
                            "rdp_alpha_mode": args.rdp_alpha_mode,
                            "alpha_count": args.alpha_count,
                            "paper_profile": args.paper_profile,
                            "paper_profile_cache_dir": args.paper_profile_cache_dir,
                            "paper_profile_notes": paper_profile_notes,
                            "pca_components": pca_components,
                            "pca_noise_sigma": pca_noise_sigma,
                            "reference_source": reference_source,
                            "configuration_match": configuration_match,
                            "abadi_reference_accuracy": abadi_reference_accuracy,
                        }
                        task["temp_path"] = str(temp_dir / _task_filename(task))
                        tasks.append(task)
    return tasks


def _dataset_summaries(
    datasets: list[str],
    seeds: list[int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        bundle = load_dataset_bundle(
            dataset,
            seed=seeds[0],
            cardio_path=_project_path(args.cardio_path),
            heart_path=_project_path(args.heart_path),
            data_dir=_project_path(args.data_dir),
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            image_validation_fraction=args.image_validation_fraction,
        )
        row = dict(bundle.dataset_summary)
        row["seed_for_split"] = seeds[0]
        row["class_distribution"] = json.dumps(row["class_distribution"])
        rows.append(row)
    return pd.DataFrame(rows)


def _parser() -> argparse.ArgumentParser:
    cpu_count = os.cpu_count() or 1
    parser = argparse.ArgumentParser(
        description="Compare Dense, CNN, and RVFL architectures with and without DP-SGD."
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--dataset", choices=[*DEFAULT_DATASETS, "both"], default=None)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--paper-profile", choices=["default", "abadi2016"], default="default")
    parser.add_argument("--paper-profile-cache-dir", default="outputs/cache")
    parser.add_argument(
        "--privacy",
        choices=["both", "non_private", "private"],
        default="both",
    )
    parser.add_argument("--epsilons", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=max(1, cpu_count - 1))
    parser.add_argument("--parallel-batch-size", dest="parallel_batch_size", type=int, default=None)
    parser.add_argument("--torch_threads", type=int, default=max(1, min(8, cpu_count)))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--resume-run-dir", default=None)
    parser.add_argument("--cardio_path", default="data/cardio_train.csv")
    parser.add_argument("--heart_path", default="data/heart.csv")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs_tabular", type=int, default=30)
    parser.add_argument("--epochs_image", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--image-validation-fraction", type=float, default=0.15)
    parser.add_argument("--run_clipping_sensitivity", action="store_true")
    parser.add_argument("--rvfl_random_features", type=int, default=200)
    parser.add_argument("--rvfl_activation", default="tanh")
    parser.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--secure-rng", "--secure-mode", dest="secure_rng", action="store_true")
    parser.add_argument("--fast-dev-mode", action="store_true")
    parser.add_argument("--rdp-alpha-mode", choices=["default", "wide"], default="wide")
    parser.add_argument("--production-dp", action="store_true")
    return parser


def _resolve_datasets(args: argparse.Namespace) -> list[str]:
    if args.dataset:
        if args.dataset == "both":
            return ["cardio", "heart"]
        return [args.dataset]
    return _parse_list(args.datasets, DEFAULT_DATASETS)


def _validate_choices(datasets: list[str], models: list[str]) -> None:
    allowed_datasets = DEFAULT_DATASETS
    allowed_models = DEFAULT_MODELS
    invalid_datasets = sorted(set(datasets) - set(allowed_datasets))
    invalid_models = sorted(set(models) - set(allowed_models))
    if invalid_datasets:
        raise ValueError(f"Unsupported datasets: {invalid_datasets}")
    if invalid_models:
        raise ValueError(f"Unsupported models: {invalid_models}")


def _validate_profile_choices(
    datasets: list[str],
    models: list[str],
    paper_profile: str,
) -> None:
    if paper_profile != "abadi2016":
        _validate_choices(datasets, models)
        return
    invalid_datasets = sorted(set(datasets) - set(ABADI2016_DATASETS))
    invalid_models = sorted(set(models) - set(ABADI2016_MODELS))
    if invalid_datasets:
        raise ValueError(
            "--paper-profile abadi2016 supports only datasets "
            f"{ABADI2016_DATASETS}; got {invalid_datasets}."
        )
    if invalid_models:
        raise ValueError(
            "--paper-profile abadi2016 supports only --models abadi; "
            f"got {invalid_models}."
        )


def main() -> int:
    if "--legacy-protocol" not in sys.argv and not any(arg in sys.argv for arg in ["-h", "--help"]):
        raise SystemExit("Historical protocol disabled by default. Use src/run_revisions.py for reviewer experiments. "
                         "Add --legacy-protocol only for exploratory legacy runs; its preprocessing and validation "
                         "are NOT covered by the revised training privacy claim.")
    if "--legacy-protocol" in sys.argv:
        sys.argv.remove("--legacy-protocol")
    args = _parser().parse_args()
    if args.paper_profile == "abadi2016":
        if args.dataset is None and args.datasets == ",".join(DEFAULT_DATASETS):
            args.datasets = ",".join(ABADI2016_DATASETS)
        if args.models == ",".join(DEFAULT_MODELS):
            args.models = ",".join(ABADI2016_MODELS)
        if args.epsilons is None:
            args.epsilons = None
        if args.batch_size is not None:
            print("Error: --batch_size cannot be combined with --paper-profile abadi2016.")
            return 2
        if args.max_grad_norm != MAX_GRAD_NORM:
            print("Error: --max_grad_norm cannot be combined with --paper-profile abadi2016.")
            return 2
        args.image_validation_fraction = 0.0
    if args.fast_dev_mode and (args.secure_rng or args.production_dp):
        print("Error: --fast-dev-mode cannot be combined with secure/production DP modes.")
        return 2
    if args.production_dp:
        args.secure_rng = True
        args.rdp_alpha_mode = "wide"
    if (
        args.n_jobs < 1
        or args.torch_threads < 1
        or args.num_workers < 0
        or args.image_validation_fraction < 0
        or (args.parallel_batch_size is not None and args.parallel_batch_size < 1)
    ):
        print(
            "Error: n_jobs, torch_threads, and parallel_batch_size must be positive; "
            "num_workers cannot be negative."
        )
        return 2

    try:
        datasets = _resolve_datasets(args)
        models = _parse_list(args.models, DEFAULT_MODELS)
        _validate_profile_choices(datasets, models, args.paper_profile)
        seeds = parse_seeds(args.seeds)
        epsilons = _parse_float_list(
            args.epsilons,
            ABADI2016_EPSILONS if args.paper_profile == "abadi2016" else EPSILONS,
        )
    except ValueError as error:
        print(f"Error: {error}")
        return 2

    if args.quick_test:
        seeds = seeds[:1]
        if args.paper_profile == "abadi2016":
            epsilons = [2.0]
        else:
            epsilons = QUICK_TEST_EPSILONS.copy()
        if args.max_train_samples is None:
            args.max_train_samples = QUICK_MAX_TRAIN_SAMPLES
        if args.max_test_samples is None:
            args.max_test_samples = QUICK_MAX_TEST_SAMPLES
    rdp_alphas = get_rdp_alphas(args.rdp_alpha_mode)
    args.alpha_count = len(rdp_alphas) if rdp_alphas is not None else 0

    configure_runtime(args.torch_threads)
    output_paths = ensure_output_dirs(_project_path(args.output_dir))
    if args.resume_run_dir:
        temp_dir = _project_path(args.resume_run_dir)
        if not temp_dir.exists() or not temp_dir.is_dir():
            print(f"Error: --resume-run-dir does not exist or is not a directory: {temp_dir}")
            return 2
        run_id = temp_dir.name
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_dir = output_paths["logs"] / "runs" / run_id
        temp_dir.mkdir(parents=True, exist_ok=False)
    _configure_logging(output_paths["logs"] / f"experiment_{run_id}.log")
    if args.resume_run_dir:
        LOGGER.info("Resuming from run directory: %s", temp_dir)

    if args.privacy in {"both", "private"}:
        if args.production_dp:
            LOGGER.info(
                "Production DP mode: secure RNG enabled and wide RDP alpha range used."
            )
        elif not args.secure_rng:
            LOGGER.warning(
                "DP development mode: secure RNG disabled. Do not use these results as final privacy claims."
            )
        LOGGER.info(
            "DP settings: secure_rng_used=%s rdp_alpha_mode=%s alpha_count=%d.",
            args.secure_rng,
            args.rdp_alpha_mode,
            args.alpha_count,
        )
        if args.secure_rng:
            try:
                from dp_accounting import validate_secure_rng_available

                validate_secure_rng_available()
            except RuntimeError as error:
                LOGGER.error(str(error))
                return 2

    run_config = {
        **vars(args),
        "datasets_selected": datasets,
        "models_selected": models,
        "selected_seeds": seeds,
        "epsilon_grid": epsilons,
        "run_id": run_id,
        "project_root": PROJECT_ROOT,
        "output_dir_resolved": output_paths["root"],
        "method_note": (
            "This study compares Dense, CNN, and RVFL architectures under identical "
            "non-private and DP-SGD settings; it does not propose DP-SGD itself."
        ),
        "tabular_cnn_note": (
            "CNN is natural for images; the tabular 1D-CNN is exploratory because "
            "tabular features do not have true spatial structure."
        ),
    }
    (output_paths["logs"] / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=_json_safe),
        encoding="utf-8",
    )

    try:
        dataset_summary = _dataset_summaries(datasets, seeds, args)
    except Exception as error:
        LOGGER.error("Dataset loading failed before task construction: %s", error)
        return 1
    _write_table_with_fallback(
        dataset_summary,
        output_paths["tables"] / "dataset_summary.csv",
        run_id,
    )

    tasks = _build_tasks(
        datasets=datasets,
        models=models,
        privacy=args.privacy,
        epsilons=epsilons,
        seeds=seeds,
        args=args,
        temp_dir=temp_dir,
    )
    if args.run_clipping_sensitivity and args.privacy in {"both", "private"}:
        LOGGER.info(
            "Clipping sensitivity requested; main comparison uses max_grad_norm=%s. "
            "Sensitivity rows can be run by overriding --max_grad_norm in separate runs.",
            args.max_grad_norm,
        )
    LOGGER.info(
        "Starting %d configurations (%s execution).",
        len(tasks),
        "parallel" if args.parallel else "sequential",
    )
    if args.parallel:
        _run_tasks_in_parallel(tasks, args)
    else:
        _run_tasks_sequentially(
            tasks,
            skip_completed=bool(args.resume_run_dir),
        )

    result_files = sorted(temp_dir.glob("run_*.csv"))
    if not result_files:
        LOGGER.error("No result files were produced.")
        return 1
    try:
        results = _read_result_files(result_files)
    except RuntimeError as error:
        LOGGER.error(str(error))
        return 1
    results = results.reindex(columns=RESULT_COLUMNS)
    _write_table_with_fallback(
        results,
        output_paths["tables"] / "all_results.csv",
        run_id,
    )

    from plots import generate_all_figures
    from statistics import generate_all_tables

    generate_all_tables(results, output_paths["tables"])
    generate_all_figures(results, output_paths["figures"])

    success_count = int((results["run_status"] == "success").sum())
    failure_count = int((results["run_status"] == "failed").sum())
    LOGGER.info(
        "Finished: %d successful and %d failed configurations. Outputs: %s",
        success_count,
        failure_count,
        output_paths["root"],
    )
    return 0 if success_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
