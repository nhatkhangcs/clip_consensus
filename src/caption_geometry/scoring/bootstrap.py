from __future__ import annotations

import torch


def bootstrap_mean_ci(
    values: torch.Tensor,
    samples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    clusters: torch.Tensor | None = None,
) -> dict[str, float]:
    """Percentile bootstrap CI of a mean, in percent.

    With `clusters`, whole clusters (e.g. all queries of one image) are resampled
    together, so correlated outcomes do not shrink the interval.
    """

    numeric = values.float().cpu()

    if clusters is None:
        means = _bootstrap_means(numeric, samples=samples, seed=seed)
    else:
        means = _cluster_bootstrap_means(numeric, clusters.cpu(), samples=samples, seed=seed)

    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q
    return {
        "estimate": numeric.mean().item() * 100.0,
        "low": torch.quantile(means, lower_q).item() * 100.0,
        "high": torch.quantile(means, upper_q).item() * 100.0,
    }


def bootstrap_gain_ci(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    samples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    clusters: torch.Tensor | None = None,
) -> dict[str, float]:
    delta = candidate.float().cpu() - baseline.float().cpu()
    return bootstrap_mean_ci(delta, samples=samples, seed=seed, confidence=confidence, clusters=clusters)


def _bootstrap_means(values: torch.Tensor, samples: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_values = values.numel()
    indices = torch.randint(n_values, (samples, n_values), generator=generator)
    return values[indices].mean(dim=1)


def _cluster_bootstrap_means(
    values: torch.Tensor,
    clusters: torch.Tensor,
    samples: int,
    seed: int,
) -> torch.Tensor:
    if clusters.shape != values.shape:
        raise ValueError("clusters must have the same shape as values")

    _, cluster_index = torch.unique(clusters, return_inverse=True)
    n_clusters = int(cluster_index.max().item()) + 1
    sums = torch.zeros(n_clusters, dtype=values.dtype).index_add_(0, cluster_index, values)
    counts = torch.zeros(n_clusters, dtype=values.dtype).index_add_(0, cluster_index, torch.ones_like(values))

    generator = torch.Generator(device="cpu").manual_seed(seed)
    picks = torch.randint(n_clusters, (samples, n_clusters), generator=generator)
    return sums[picks].sum(dim=1) / counts[picks].sum(dim=1)
