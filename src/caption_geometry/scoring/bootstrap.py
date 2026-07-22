from __future__ import annotations

import torch


def bootstrap_mean_ci(
    values: torch.Tensor,
    samples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float]:
    numeric = values.float().cpu()
    means = _bootstrap_means(numeric, samples=samples, seed=seed)
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
) -> dict[str, float]:
    delta = candidate.float().cpu() - baseline.float().cpu()
    return bootstrap_mean_ci(delta, samples=samples, seed=seed, confidence=confidence)


def _bootstrap_means(values: torch.Tensor, samples: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_values = values.numel()
    indices = torch.randint(n_values, (samples, n_values), generator=generator)
    return values[indices].mean(dim=1)

