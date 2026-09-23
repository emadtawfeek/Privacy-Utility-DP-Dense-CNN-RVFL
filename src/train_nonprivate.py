"""Reusable non-private training loop for all architecture families."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainResult:
    training_time_seconds: float
    epochs_completed: int
    diagnostic_history: list | None = None


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=max(1, min(batch_size, len(dataset))),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _binary_pos_weight(dataset: Dataset) -> torch.Tensor:
    positives = 0.0
    total = 0.0
    for _, target in dataset:
        positives += float(target)
        total += 1.0
    negatives = total - positives
    return torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32)


def make_criterion(task_type: str, train_dataset: Dataset, *, use_empirical_class_weights: bool = False) -> nn.Module:
    """Unweighted loss by default: no unaccounted access to private class counts."""
    if task_type == "binary":
        return nn.BCEWithLogitsLoss(
            pos_weight=_binary_pos_weight(train_dataset) if use_empirical_class_weights else None
        )
    return nn.CrossEntropyLoss()


def _make_optimizer(
    model: nn.Module,
    lr: float,
    optimizer_name: str,
    weight_decay: float = 0.0,
    sgd_momentum: float = 0.0,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=sgd_momentum)
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def _lr_for_epoch(default_lr: float, lr_schedule: dict | None, epoch_index: int) -> float:
    if not lr_schedule:
        return default_lr
    if lr_schedule.get("name") != "linear_then_constant":
        return default_lr
    initial_lr = float(lr_schedule["initial_lr"])
    final_lr = float(lr_schedule["final_lr"])
    decay_epochs = max(1, int(lr_schedule["decay_epochs"]))
    if epoch_index >= decay_epochs:
        return final_lr
    fraction = epoch_index / decay_epochs
    return initial_lr + (final_lr - initial_lr) * fraction


def apply_lr_schedule(
    optimizer: torch.optim.Optimizer,
    default_lr: float,
    lr_schedule: dict | None,
    epoch_index: int,
) -> float:
    lr = _lr_for_epoch(default_lr, lr_schedule, epoch_index)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_nonprivate_model(
    *,
    model: nn.Module,
    train_dataset: Dataset,
    task_type: str,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    num_workers: int,
    optimizer_name: str = "adam",
    lr_schedule: dict | None = None,
    weight_decay: float = 0.0,
    sgd_momentum: float = 0.0,
    benchmark_diagnostics: bool = False,
) -> TrainResult:
    """Train a model without differential privacy."""
    loader = make_loader(train_dataset, batch_size, True, seed, num_workers)
    criterion = make_criterion(task_type, train_dataset)
    optimizer = _make_optimizer(model, lr, optimizer_name, weight_decay, sgd_momentum)
    start = time.perf_counter()
    epochs_completed = 0
    diagnostic_history = [] if benchmark_diagnostics else None
    for epoch in range(epochs):
        apply_lr_schedule(optimizer, lr, lr_schedule, epoch)
        model.train()
        loss_sum = examples = 0
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            if task_type == "binary":
                loss = criterion(logits, targets.float())
            else:
                loss = criterion(logits, targets.long())
            loss.backward()
            optimizer.step()
            if benchmark_diagnostics:
                loss_sum += float(loss.detach()) * len(targets)
                examples += len(targets)
        if benchmark_diagnostics:
            diagnostic_history.append({"epoch": epoch + 1, "training_loss_mean": loss_sum / examples,
                                       "examples_seen": examples})
        epochs_completed = epoch + 1
    return TrainResult(
        training_time_seconds=time.perf_counter() - start,
        epochs_completed=epochs_completed,
        diagnostic_history=diagnostic_history,
    )
