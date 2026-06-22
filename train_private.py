"""Reusable Opacus DP-SGD trainer for dense, CNN, and RVFL models."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset

from config import get_rdp_alphas
from dp_accounting import achieved_epsilon, create_privacy_engine, get_noise_multiplier
from train_nonprivate import apply_lr_schedule, make_criterion, make_loader


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


def _make_optimizer(
    model: nn.Module,
    lr: float,
    optimizer_name: str,
    sgd_momentum: float,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=sgd_momentum)
    return torch.optim.Adam(parameters, lr=lr)


def _setup_private(
    *,
    model: nn.Module,
    train_dataset: Dataset,
    task_type: str,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    num_workers: int,
    epsilon: float,
    delta: float,
    max_grad_norm: float,
    accountant_name: str,
    optimizer_name: str,
    sgd_momentum: float,
    secure_mode: bool,
    rdp_alpha_mode: str,
):
    rdp_alphas = get_rdp_alphas(rdp_alpha_mode)
    alpha_count = len(rdp_alphas) if rdp_alphas is not None else 0
    optimizer = _make_optimizer(model, lr, optimizer_name, sgd_momentum)
    loader = make_loader(train_dataset, batch_size, True, seed, num_workers)
    criterion = make_criterion(task_type, train_dataset)
    privacy_engine = create_privacy_engine(
        accountant_name=accountant_name,
        secure_mode=secure_mode,
        rdp_alphas=rdp_alphas,
    )

    if rdp_alphas is None:
        private_model, private_optimizer, private_loader = (
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                target_epsilon=epsilon,
                target_delta=delta,
                epochs=epochs,
                max_grad_norm=max_grad_norm,
            )
        )
        method = f"make_private_with_epsilon/{optimizer_name}/{accountant_name}"
    else:
        sample_rate = 1 / len(loader)
        sigma = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant_name=accountant_name,
            rdp_alphas=rdp_alphas,
        )
        private_model, private_optimizer, private_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=sigma,
            max_grad_norm=max_grad_norm,
        )
        method = (
            f"make_private/calibrated_noise/{optimizer_name}/"
            f"{accountant_name}/{rdp_alpha_mode}_alphas"
        )
    return (
        private_model,
        private_optimizer,
        private_loader,
        criterion,
        privacy_engine,
        method,
        alpha_count,
    )


def train_private_model(
    *,
    model: nn.Module,
    train_dataset: Dataset,
    val_loader,
    test_loader,
    task_type: str,
    epsilon: float,
    delta: float,
    max_grad_norm: float,
    epochs: int,
    lr: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    secure_mode: bool = False,
    rdp_alpha_mode: str = "wide",
    optimizer_name: str | None = None,
    sgd_momentum: float = 0.9,
    lr_schedule: dict | None = None,
) -> PrivateTrainResult:
    """Train one private model with Opacus DP-SGD.

    ``val_loader`` and ``test_loader`` are accepted to keep the public interface
    explicit and reusable; evaluation remains outside the trainer.
    """
    del val_loader, test_loader
    setup_errors: list[str] = []
    setup = None
    selected_accountant = ""
    optimizer_candidates = (optimizer_name,) if optimizer_name else ("adam", "sgd")
    for accountant_name in ("rdp", "prv"):
        for candidate_optimizer in optimizer_candidates:
            try:
                setup = _setup_private(
                    model=model,
                    train_dataset=train_dataset,
                    task_type=task_type,
                    batch_size=batch_size,
                    epochs=epochs,
                    lr=lr,
                    seed=seed,
                    num_workers=num_workers,
                    epsilon=epsilon,
                    delta=delta,
                    max_grad_norm=max_grad_norm,
                    accountant_name=accountant_name,
                    optimizer_name=candidate_optimizer,
                    sgd_momentum=sgd_momentum,
                    secure_mode=secure_mode,
                    rdp_alpha_mode=rdp_alpha_mode,
                )
                selected_accountant = accountant_name
                break
            except Exception as error:
                setup_errors.append(
                    f"{accountant_name}/{candidate_optimizer}: {type(error).__name__}: {error}"
                )
        if setup is not None:
            break
    if setup is None:
        raise RuntimeError(" | ".join(setup_errors))

    (
        private_model,
        private_optimizer,
        private_loader,
        criterion,
        privacy_engine,
        setup_method,
        alpha_count,
    ) = setup
    start = time.perf_counter()
    epochs_completed = 0
    for epoch in range(epochs):
        apply_lr_schedule(private_optimizer, lr, lr_schedule, epoch)
        private_model.train()
        for features, targets in private_loader:
            private_optimizer.zero_grad(set_to_none=True)
            logits = private_model(features)
            if task_type == "binary":
                loss = criterion(logits, targets.float())
            else:
                loss = criterion(logits, targets.long())
            loss.backward()
            private_optimizer.step()
        epochs_completed = epoch + 1

    rdp_alphas = get_rdp_alphas(rdp_alpha_mode)
    achieved = achieved_epsilon(
        privacy_engine,
        delta=delta,
        rdp_alphas=rdp_alphas,
    )
    return PrivateTrainResult(
        training_time_seconds=time.perf_counter() - start,
        epochs_completed=epochs_completed,
        epsilon_achieved=achieved,
        noise_multiplier=float(getattr(private_optimizer, "noise_multiplier", float("nan"))),
        private_setup_method=setup_method,
        accountant=selected_accountant,
        secure_rng_used=secure_mode,
        rdp_alpha_mode=rdp_alpha_mode,
        alpha_count=alpha_count,
    )
