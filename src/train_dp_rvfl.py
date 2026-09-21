"""Legacy compatibility shim.

The active pipeline uses ``train_private.train_private_model`` with the RVFL
model constructed in ``models.build_model``. DP-RVFL is still a custom PyTorch
frozen-random-feature model whose output layer is trained with Opacus DP-SGD.
"""

from __future__ import annotations


def train_dp_rvfl(*args, **kwargs):
    raise RuntimeError(
        "train_dp_rvfl is deprecated. Use train_private.train_private_model via "
        "src/run_experiments.py with --models rvfl --privacy private."
    )
