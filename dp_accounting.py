"""Opacus accounting helpers used by both private model trainers."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Sequence

import numpy as np
from opacus import PrivacyEngine
from opacus.accountants import create_accountant
from opacus.accountants.prv import PRVAccountant
from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.utils import MAX_SIGMA
from opacus.accountants.utils import get_noise_multiplier as opacus_get_noise_multiplier
from opacus.accountants.analysis.prv import Domain


SECURE_RNG_ERROR = (
    "Secure RNG requires torchcsprng. Install it or run without "
    "--production-dp for development only."
)


class WideAlphaPRVAccountant(PRVAccountant):
    """PRV accountant using wider RDP orders for its internal safe-domain bound."""

    def __init__(self, rdp_alphas: Sequence[float | int]) -> None:
        super().__init__()
        self.rdp_alphas = list(rdp_alphas)

    def _get_domain(self, prvs, num_self_compositions, eps_error, delta_error):
        total_self_compositions = sum(num_self_compositions)
        domain_size = _compute_safe_domain_size_with_alphas(
            prvs=prvs,
            max_self_compositions=num_self_compositions,
            eps_error=eps_error,
            delta_error=delta_error,
            rdp_alphas=self.rdp_alphas,
        )
        mesh_size = eps_error / np.sqrt(
            total_self_compositions * np.log(12 / delta_error) / 2
        )
        return Domain.create_aligned(-domain_size, domain_size, mesh_size)


def _compute_safe_domain_size_with_alphas(
    *,
    prvs,
    max_self_compositions: Sequence[int],
    eps_error: float,
    delta_error: float,
    rdp_alphas: Sequence[float | int],
) -> float:
    """Mirror Opacus PRV domain sizing but pass explicit RDP orders."""
    total_compositions = sum(max_self_compositions)

    rdp_accountant = RDPAccountant()
    for prv, max_self_composition in zip(prvs, max_self_compositions):
        rdp_accountant.history.append(
            (prv.noise_multiplier, prv.sample_rate, max_self_composition)
        )
    domain_size = rdp_accountant.get_epsilon(
        delta=delta_error / 4,
        alphas=list(rdp_alphas),
    )

    for prv, _ in zip(prvs, max_self_compositions):
        rdp_accountant = RDPAccountant()
        rdp_accountant.history = [(prv.noise_multiplier, prv.sample_rate, 1)]
        domain_size = max(
            domain_size,
            rdp_accountant.get_epsilon(
                delta=delta_error / (8 * total_compositions),
                alphas=list(rdp_alphas),
            ),
        )

    return max(domain_size, eps_error) + 3


def validate_secure_rng_available() -> None:
    """Fail early with a project-specific message if secure Opacus RNG cannot run."""
    try:
        import torch  # noqa: F401  # Load PyTorch DLL directories on Windows first.
        import torchcsprng  # noqa: F401
    except ImportError as error:
        raise RuntimeError(SECURE_RNG_ERROR) from error


def create_privacy_engine(
    *,
    accountant_name: str,
    secure_mode: bool,
    rdp_alphas: Sequence[float | int] | None,
) -> PrivacyEngine:
    """Create a PrivacyEngine and attach a wide-alpha PRV accountant if needed."""
    try:
        if secure_mode:
            privacy_engine = PrivacyEngine(
                accountant=accountant_name,
                secure_mode=secure_mode,
            )
        else:
            # The runner logs the project-level DP development-mode warning once.
            # Avoid repeating Opacus' constructor warning for every configuration.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Secure RNG turned off.*",
                    category=UserWarning,
                    module="opacus.privacy_engine",
                )
                privacy_engine = PrivacyEngine(
                    accountant=accountant_name,
                    secure_mode=secure_mode,
                )
    except ImportError as error:
        if secure_mode:
            raise RuntimeError(SECURE_RNG_ERROR) from error
        raise

    if accountant_name == "prv" and rdp_alphas is not None:
        privacy_engine.accountant = WideAlphaPRVAccountant(rdp_alphas)
    return privacy_engine


def _create_calibration_accountant(
    accountant_name: str,
    rdp_alphas: Sequence[float | int] | None,
):
    if accountant_name == "prv" and rdp_alphas is not None:
        return WideAlphaPRVAccountant(rdp_alphas)
    return create_accountant(mechanism=accountant_name)


def _epsilon_for_accountant(
    accountant,
    *,
    delta: float,
    rdp_alphas: Sequence[float | int] | None,
) -> float:
    if accountant.mechanism() == "rdp" and rdp_alphas is not None:
        return float(accountant.get_epsilon(delta=delta, alphas=list(rdp_alphas)))
    return float(accountant.get_epsilon(delta=delta))


def get_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int,
    accountant_name: str,
    rdp_alphas: Sequence[float | int] | None,
) -> float:
    """Compute Opacus noise, using explicit alphas when this version supports them."""
    if rdp_alphas is None:
        parameters = inspect.signature(opacus_get_noise_multiplier).parameters
        kwargs = {
            "target_epsilon": target_epsilon,
            "target_delta": target_delta,
            "sample_rate": sample_rate,
            "epochs": epochs,
        }
        if "accountant" in parameters:
            kwargs["accountant"] = accountant_name
        return float(opacus_get_noise_multiplier(**kwargs))

    steps = int(epochs / sample_rate)
    eps_high = float("inf")
    accountant = _create_calibration_accountant(accountant_name, rdp_alphas)

    sigma_low, sigma_high = 0.0, 10.0
    while eps_high > target_epsilon:
        sigma_high = 2 * sigma_high
        accountant.history = [(sigma_high, sample_rate, steps)]
        eps_high = _epsilon_for_accountant(
            accountant,
            delta=target_delta,
            rdp_alphas=rdp_alphas,
        )
        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while target_epsilon - eps_high > 0.01:
        sigma = (sigma_low + sigma_high) / 2
        accountant.history = [(sigma, sample_rate, steps)]
        eps = _epsilon_for_accountant(
            accountant,
            delta=target_delta,
            rdp_alphas=rdp_alphas,
        )

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return float(sigma_high)


def achieved_epsilon(
    privacy_engine: PrivacyEngine,
    *,
    delta: float,
    rdp_alphas: Sequence[float | int] | None,
) -> float:
    """Return achieved epsilon, passing wide alphas to RDP when supported."""
    accountant = privacy_engine.accountant
    if accountant.mechanism() == "rdp" and rdp_alphas is not None:
        return float(accountant.get_epsilon(delta=delta, alphas=list(rdp_alphas)))
    if hasattr(privacy_engine, "get_epsilon"):
        return float(privacy_engine.get_epsilon(delta))
    return float(accountant.get_epsilon(delta=delta))
