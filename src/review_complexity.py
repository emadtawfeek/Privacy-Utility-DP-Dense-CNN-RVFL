"""Transparent inference-operation estimates and opt-in CPU latency measurements.

MAC counts include Linear/Conv kernels and explicit DPELM matrix products. They
exclude nonlinearities, pooling, memory traffic and DP-SGD backward/accounting.
"""
from __future__ import annotations

import math
import time

import torch
from torch import nn


def estimate_forward_macs(model, input_shape):
    if hasattr(model, "hidden_nodes") and hasattr(model, "random_weights"):
        width = model.hidden_nodes
        return int(math.prod(input_shape) * width + width * model.num_classes)
    counts = []
    handles = []

    def hook(module, _inputs, output):
        if isinstance(module, nn.Linear):
            counts.append(output.numel() * module.in_features)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            kernel = math.prod(module.kernel_size)
            counts.append(output.numel() * (module.in_channels // module.groups) * kernel)

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            handles.append(module.register_forward_hook(hook))
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros((1, *input_shape), dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return int(sum(counts))


def parameter_storage_bytes(model):
    return int(sum(p.numel() * p.element_size() for p in model.parameters()) +
               sum(b.numel() * b.element_size() for b in model.buffers()))


def benchmark_forward_batch(model, batch, warmups=3, repeats=10):
    """End-to-end forward latency on a resident tensor, excluding data loading."""
    was_training = model.training
    timings = []
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(warmups):
                model(batch)
            for _ in range(repeats):
                start = time.perf_counter()
                model(batch)
                timings.append(time.perf_counter() - start)
    finally:
        model.train(was_training)
    ordered = sorted(timings)
    median = (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2
    return {"forward_batch_size": len(batch), "forward_warmups": warmups,
            "forward_repeats": repeats, "forward_batch_median_seconds": median,
            "forward_milliseconds_per_record": 1000 * median / len(batch),
            "forward_records_per_second": len(batch) / median if median > 0 else None}
