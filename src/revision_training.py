"""Auditable DP-SGD for one fixed training partition and fixed configuration."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache

import torch
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader

from config import get_rdp_alphas
from dp_accounting import achieved_epsilon, create_privacy_engine
from train_nonprivate import apply_lr_schedule, make_criterion, make_loader


@lru_cache(maxsize=256)
def _calibrate(epsilon, delta, sample_rate, steps, accountant_name, orders, tolerance):
    """Cache only data-independent public accounting configurations."""
    kwargs = {"alphas": list(orders)} if orders else {}
    sigma = float(get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
        sample_rate=sample_rate, steps=steps, accountant=accountant_name,
        epsilon_tolerance=tolerance, **kwargs))
    calibration = create_accountant(mechanism=accountant_name)
    calibration.history = [(sigma, sample_rate, steps)]
    return sigma, float(calibration.get_epsilon(delta=delta, **kwargs))


@dataclass
class PrivateTrainResult:
    training_time_seconds: float
    epochs_completed: int
    epsilon_achieved: float
    noise_multiplier: float
    private_setup_method: str
    accountant: str
    secure_rng_used: bool
    rdp_alpha_mode: str
    alpha_count: int
    sample_rate: float
    expected_batch_size: int
    steps_planned: int
    steps_completed: int
    empty_batches: int
    calibration_epsilon: float
    calibration_tolerance: float
    setup_time_seconds: float
    accountant_history: list
    rdp_orders: list
    public_reference_size: int
    diagnostic_history: list | None = None
    adjacency: str = "add_remove_one"
    sampling: str = "poisson"
    checkpoint_selection: str = "final_fixed_epoch"


def train_private_model(
    *, model, train_dataset, val_loader, test_loader, task_type,
    epsilon, delta, max_grad_norm, epochs, lr, seed, batch_size, num_workers,
    secure_mode=False, rdp_alpha_mode="wide", optimizer_name=None,
    sgd_momentum=0.9, lr_schedule=None, weight_decay=0.0,
    accountant_name="rdp", epsilon_tolerance=1e-4,
    public_reference_size=None,
    benchmark_diagnostics=False,
):
    """Explicit Poisson draws, exact-step calibration, no data-dependent selection.

    Empty Poisson batches still receive a noise-only optimizer step. No
    optimizer/accountant fallback is allowed. The caller must use fixed or
    separately private preprocessing; this trainer cannot certify that part.
    """
    del val_loader, test_loader
    if not (math.isfinite(epsilon) and epsilon > 0 and 0 < delta < 1):
        raise ValueError("Require finite epsilon > 0 and 0 < delta < 1.")
    if epochs < 1 or batch_size < 1 or len(train_dataset) < 1:
        raise ValueError("Training data, batch size, and epochs must be positive.")
    if not (math.isfinite(max_grad_norm) and max_grad_norm > 0 and lr > 0):
        raise ValueError("Clipping norm and learning rate must be positive.")
    if not 0 < epsilon_tolerance < epsilon:
        raise ValueError("Calibration tolerance must lie between zero and epsilon.")
    if accountant_name not in {"rdp", "prv"}:
        raise ValueError("Supported accountants: rdp, prv.")
    optimizer_name = optimizer_name or "adam"
    if optimizer_name not in {"adam", "sgd"}:
        raise ValueError("Supported optimizers: adam, sgd.")
    setup_start = time.perf_counter()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("DP-SGD needs trainable parameters.")
    optimizer = (
        torch.optim.SGD(parameters, lr=lr, momentum=sgd_momentum, weight_decay=weight_decay)
        if optimizer_name == "sgd" else
        torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    )
    # This reference size is a PUBLIC protocol parameter, held fixed across
    # add/remove neighbors. Do not re-estimate it on confidential populations.
    reference_size = len(train_dataset) if public_reference_size is None else public_reference_size
    if not isinstance(reference_size, int) or reference_size < 1:
        raise ValueError("public_reference_size must be a positive public integer.")
    batches_per_epoch = math.ceil(reference_size / batch_size)
    sample_rate = 1.0 / batches_per_epoch
    planned_steps = epochs * batches_per_epoch
    loader = DPDataLoader(train_dataset, sample_rate=sample_rate,
                          generator=torch.Generator().manual_seed(seed), num_workers=num_workers)
    loader.batch_sampler.steps = batches_per_epoch
    orders = get_rdp_alphas(rdp_alpha_mode) if accountant_name == "rdp" else None
    if accountant_name == "rdp" and orders is None:
        orders = list(create_accountant(mechanism="rdp").DEFAULT_ALPHAS)
    sigma, calibrated_epsilon = _calibrate(epsilon, delta, sample_rate, planned_steps,
        accountant_name, tuple(orders or []), epsilon_tolerance)
    if not math.isfinite(calibrated_epsilon) or calibrated_epsilon > epsilon + 1e-10:
        raise RuntimeError("Calibrated noise exceeds the target privacy budget.")
    engine = create_privacy_engine(
        accountant_name=accountant_name, secure_mode=secure_mode, rdp_alphas=None,
    )
    private_model, private_optimizer, private_loader = engine.make_private(
        module=model, optimizer=optimizer, data_loader=loader,
        noise_multiplier=sigma, max_grad_norm=max_grad_norm,
        poisson_sampling=True, clipping="flat", loss_reduction="mean",
    )
    if not math.isclose(private_loader.sample_rate, sample_rate, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Calibrated and implemented sampling rates differ.")
    # Avoid int(1/q) roundoff in Opacus' default sampler step count. Also
    # keep the normalization and accounting parameters fixed across neighbors.
    private_loader.batch_sampler.steps = batches_per_epoch
    private_optimizer.expected_batch_size = max(1, int(reference_size * sample_rate))
    private_optimizer.attach_step_hook(engine.accountant.get_optimizer_hook_fn(sample_rate=sample_rate))
    criterion = make_criterion(task_type, train_dataset)
    setup_seconds = time.perf_counter() - setup_start
    started = time.perf_counter()
    steps = empty_batches = 0
    diagnostic_history = [] if benchmark_diagnostics else None
    try:
        for epoch in range(epochs):
            apply_lr_schedule(private_optimizer, lr, lr_schedule, epoch)
            private_model.train()
            loss_sum = observed_examples = clipped_examples = 0
            norm_sum = 0.0
            for features, targets in private_loader:
                private_optimizer.zero_grad(set_to_none=True)
                if len(targets) == 0:
                    empty_batches += 1
                    for parameter in parameters:
                        parameter.grad = torch.zeros_like(parameter)
                        parameter.grad_sample = torch.empty(
                            (0, *parameter.shape), dtype=parameter.dtype, device=parameter.device
                        )
                else:
                    logits = private_model(features)
                    loss = criterion(logits, targets.float() if task_type == "binary" else targets.long())
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite loss; experiment invalid.")
                    loss.backward()
                    if benchmark_diagnostics:
                        with torch.no_grad():
                            squared = torch.zeros(len(targets), device=features.device)
                            for parameter in parameters:
                                samples = parameter.grad_sample
                                if isinstance(samples, list):
                                    samples = torch.cat(samples, dim=0)
                                squared += samples.reshape(len(targets), -1).square().sum(dim=1)
                            norms = squared.sqrt()
                            clipped_examples += int((norms > max_grad_norm).sum())
                            norm_sum += float(norms.sum())
                            loss_sum += float(loss.detach()) * len(targets)
                            observed_examples += len(targets)
                private_optimizer.step()
                steps += 1
            if benchmark_diagnostics:
                diagnostic_history.append({"epoch": epoch + 1,
                    "training_loss_mean": loss_sum / observed_examples if observed_examples else None,
                    "poisson_record_draws": observed_examples,
                    "preclip_gradient_l2_mean": norm_sum / observed_examples if observed_examples else None,
                    "clipping_fraction": clipped_examples / observed_examples if observed_examples else None,
                    "clipped_examples": clipped_examples})
        elapsed = time.perf_counter() - started
        history = [list(item) for item in engine.accountant.history]
        if steps != planned_steps or sum(item[2] for item in history) != steps:
            raise RuntimeError("Accounted, executed, and planned step counts differ.")
        achieved = achieved_epsilon(engine, delta=delta, rdp_alphas=orders)
        if not math.isfinite(achieved) or achieved > epsilon + 1e-10:
            raise RuntimeError("Achieved epsilon exceeds the target.")
        if not all(torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError("Non-finite trained parameters.")
        return PrivateTrainResult(
            training_time_seconds=elapsed, epochs_completed=epochs,
            epsilon_achieved=achieved, noise_multiplier=sigma,
            private_setup_method=f"explicit_poisson/exact_steps/{optimizer_name}/{accountant_name}",
            accountant=accountant_name, secure_rng_used=secure_mode,
            rdp_alpha_mode=rdp_alpha_mode if orders else "not_applicable",
            alpha_count=len(orders or []), sample_rate=sample_rate,
            expected_batch_size=int(private_optimizer.expected_batch_size),
            steps_planned=planned_steps, steps_completed=steps, empty_batches=empty_batches,
            calibration_epsilon=calibrated_epsilon, calibration_tolerance=epsilon_tolerance,
            setup_time_seconds=setup_seconds, accountant_history=history, rdp_orders=list(orders or []),
            public_reference_size=reference_size,
            diagnostic_history=diagnostic_history,
        )
    finally:
        private_model.to_standard_module()
