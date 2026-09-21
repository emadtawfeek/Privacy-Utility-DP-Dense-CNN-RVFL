"""Paper-specific reference profiles used for reproduction-style comparisons."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from config import PROJECT_ROOT


def abadi_pca_noise_for_epsilon(epsilon: float) -> float:
    """Return the MNIST PCA noise multiplier reported by Abadi et al."""
    if epsilon <= 0.5:
        return 16.0
    if epsilon <= 2.0:
        return 7.0
    return 4.0


def _cache_path(
    *,
    dataset: str,
    seed: int,
    components: int,
    noise_sigma: float,
    train_size: int,
    cache_dir: str | Path,
) -> Path:
    root = Path(cache_dir).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    return root / (
        f"{dataset}_abadi_pca_seed{seed}_n{train_size}_"
        f"k{components}_sigma{str(noise_sigma).replace('.', 'p')}.pt"
    )


def fit_abadi_mnist_noisy_pca(
    train_dataset: Dataset,
    *,
    seed: int,
    components: int = 60,
    noise_sigma: float = 7.0,
    cache_dir: str | Path = "outputs/cache",
    batch_size: int = 1024,
) -> torch.Tensor:
    """Fit the noisy PCA projection described in Abadi et al. for MNIST.

    The paper normalizes each sampled training vector to unit L2 norm, adds
    Gaussian noise to A^T A, and uses the leading principal directions. This
    helper implements that reference preprocessing for comparability.
    """
    path = _cache_path(
        dataset="mnist",
        seed=seed,
        components=components,
        noise_sigma=noise_sigma,
        train_size=len(train_dataset),
        cache_dir=cache_dir,
    )
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=max(1, min(batch_size, len(train_dataset))),
        shuffle=False,
        num_workers=0,
    )
    covariance = torch.zeros((28 * 28, 28 * 28), dtype=torch.float64)
    for features, _ in loader:
        flat = features.reshape(features.shape[0], -1).to(torch.float64)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        normalized = flat / norms
        covariance += normalized.t().matmul(normalized)

    noise = torch.randn(
        covariance.shape,
        dtype=covariance.dtype,
        generator=generator,
    ) * float(noise_sigma)
    symmetric_noise = (noise + noise.t()) / 2.0
    noisy_covariance = covariance + symmetric_noise
    _, eigenvectors = torch.linalg.eigh(noisy_covariance)
    projection = eigenvectors[:, -components:].t().contiguous().to(torch.float32)
    torch.save(projection, path)
    return projection
