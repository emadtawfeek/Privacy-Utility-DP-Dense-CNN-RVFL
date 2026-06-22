"""Legacy compatibility shim.

The active four-dataset paper pipeline uses ``train_private.train_private_model``
for DP-Dense, DP-CNN, and DP-RVFL. This module remains importable so older
notebooks fail with a clear message instead of a missing-symbol import error.
"""

from __future__ import annotations


def train_dp_mlp(*args, **kwargs):
    raise RuntimeError(
        "train_dp_mlp is deprecated. Use train_private.train_private_model via "
        "src/run_experiments.py with --models dense --privacy private."
    )
