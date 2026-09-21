"""Ono et al. MDAI 2024 Algorithm 1, DOI 10.1007/978-3-031-68208-7_14.

Sigmoid random features; independent Laplace noise on all entries of H.T@H and
H.T@Y at scale L*(L+2)/epsilon; solve the noisy linear system. No DP-SGD.
The paper proves pure DP under replace-one adjacency. The same conservative
scale bounds add/remove adjacency too (L^2+L <= L^2+2L).

This is a research numerical implementation, not an audited finite-precision
pure-DP sampler. Its privacy statement is the paper's ideal-arithmetic analysis.
"""
from __future__ import annotations

import math
import secrets
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def laplace_noise(shape, scale, *, secure, rng):
    if not secure:
        return rng.laplace(0.0, scale, size=shape)
    # OS-backed randomness; inverse CDF still has finite precision limitations.
    size = int(np.prod(shape))
    uniforms = np.fromiter(((secrets.randbits(52) + .5) / 2**52 for _ in range(size)),
                           dtype=np.float64, count=size).reshape(shape)
    centered = uniforms - .5
    return -scale * np.sign(centered) * np.log1p(-2 * np.abs(centered))


class DPELM(nn.Module):
    def __init__(self, input_shape, num_classes, hidden_nodes=10, seed=42):
        super().__init__()
        if hidden_nodes < 1:
            raise ValueError("DPELM hidden width must be positive.")
        self.hidden_nodes, self.num_classes = hidden_nodes, num_classes
        self.original_input_dim = math.prod(input_shape)
        generator = torch.Generator().manual_seed(seed)
        # Normal mean/std not specified in the paper; N(0,1) is explicit here.
        self.register_buffer("random_weights", torch.randn(self.original_input_dim, hidden_nodes,
                             generator=generator, dtype=torch.float64))
        self.register_buffer("random_bias", torch.randn(hidden_nodes, generator=generator, dtype=torch.float64))
        self.register_buffer("beta", torch.zeros(hidden_nodes, num_classes, dtype=torch.float64))

    def hidden(self, inputs):
        return torch.sigmoid(torch.flatten(inputs, start_dim=1).double() @ self.random_weights + self.random_bias)

    def forward(self, inputs):
        return self.hidden(inputs) @ self.beta

    def fit(self, dataset, *, epsilon=None, seed=42, secure=True, batch_size=1024):
        if epsilon is not None and (not math.isfinite(epsilon) or epsilon <= 0):
            raise ValueError("DPELM epsilon must be positive and finite.")
        start = time.perf_counter()
        width = self.hidden_nodes
        gram = np.zeros((width, width), dtype=np.float64)
        cross = np.zeros((width, self.num_classes), dtype=np.float64)
        with torch.no_grad():
            for features, targets in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                hidden = self.hidden(features).numpy()
                onehot = np.eye(self.num_classes)[targets.numpy().astype(np.int64)]
                gram += hidden.T @ hidden
                cross += hidden.T @ onehot
        scale = width * (width + 2) / epsilon if epsilon is not None else 0.0
        if epsilon is not None:
            rng = np.random.default_rng(seed + 9173)
            # Do not symmetrize: Algorithm 1 perturbs all matrix entries independently.
            gram += laplace_noise(gram.shape, scale, secure=secure, rng=rng)
            cross += laplace_noise(cross.shape, scale, secure=secure, rng=rng)
        solver = "solve"
        try:
            beta = np.linalg.solve(gram, cross)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram) @ cross
            solver = "pinv_postprocessing_fallback"
        if not np.isfinite(beta).all():
            raise FloatingPointError("DPELM solve produced non-finite coefficients.")
        self.beta.copy_(torch.from_numpy(beta))
        return {"training_time_seconds": time.perf_counter()-start, "epochs_completed": 0,
                "steps_completed": 1, "noise_scale_laplace": scale, "linear_solver": solver,
                "epsilon_achieved": epsilon, "delta": 0.0 if epsilon is not None else None,
                "native_adjacency": "replace_one", "comparison_adjacency": "add_remove_one",
                "privacy_type": "pure_dp" if epsilon is not None else "non_private",
                "accountant": "analytic_L_times_L_plus_2", "sampling": "full_dataset_once",
                "secure_rng_used": bool(secure and epsilon is not None),
                "noise_implementation": "os_random_inverse_cdf_research" if secure else "seeded_numpy_development",
                "finite_precision_certified": False, "coefficient_count": width*self.num_classes,
                "reference_source": "Ono et al. MDAI 2024 Algorithm 1 DOI 10.1007/978-3-031-68208-7_14"}
