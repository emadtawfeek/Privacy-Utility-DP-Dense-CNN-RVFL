"""Legacy compatibility shim for the earlier CVD-only baseline study.

The active paper pipeline compares neural architecture families:
Dense, CNN, and RVFL. Non-private training is implemented in
``train_nonprivate.train_nonprivate_model`` and orchestrated by
``run_experiments.py``.
"""

from __future__ import annotations


CLASSICAL_MODELS: list[str] = []


def train_classical_baseline(*args, **kwargs):
    raise RuntimeError(
        "Classical baselines are not part of the current architecture-comparison "
        "pipeline. Use src/run_experiments.py --models dense,cnn,rvfl."
    )


def train_rvfln_baseline_from_prepared(*args, **kwargs):
    raise RuntimeError(
        "The current RVFL model is implemented in models.RVFLNet and trained via "
        "train_nonprivate.train_nonprivate_model."
    )
