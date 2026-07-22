from __future__ import annotations

from collections.abc import Iterable

import torch


def compute_recalls(ranks: Iterable[int] | torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    rank_tensor = torch.as_tensor(list(ranks) if not torch.is_tensor(ranks) else ranks)
    rank_tensor = rank_tensor.float()

    output = {
        f"R@{k}": (rank_tensor <= k).float().mean().item() * 100.0
        for k in ks
    }
    output["MR"] = rank_tensor.mean().item()
    return output


def recall_at_k_t2i(
    scores: torch.Tensor,
    k_captions: int = 5,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Text-to-image recalls for a [N*K, N] score matrix."""

    ranks = t2i_ranks(scores, k_captions)
    return compute_recalls(ranks.cpu(), ks)


def recall_at_k_i2t(
    scores_t2i: torch.Tensor,
    k_captions: int = 5,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Image-to-text recalls using the best-ranked caption among each image's positives."""

    ranks = i2t_ranks(scores_t2i, k_captions)
    return compute_recalls(ranks.cpu(), ks)


def retrieval_metrics(
    scores_t2i: torch.Tensor,
    k_captions: int = 5,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, dict[str, float]]:
    return {
        "t2i": recall_at_k_t2i(scores_t2i, k_captions=k_captions, ks=ks),
        "i2t": recall_at_k_i2t(scores_t2i, k_captions=k_captions, ks=ks),
    }


def t2i_ranks(scores: torch.Tensor, k_captions: int = 5) -> torch.Tensor:
    _validate_score_matrix(scores, k_captions)
    num_queries, n_images = scores.shape
    gt = torch.arange(n_images, device=scores.device).repeat_interleave(k_captions)
    gt_scores = scores[torch.arange(num_queries, device=scores.device), gt]
    return (scores > gt_scores[:, None]).sum(dim=1) + 1


def i2t_ranks(scores_t2i: torch.Tensor, k_captions: int = 5) -> torch.Tensor:
    _validate_score_matrix(scores_t2i, k_captions)
    scores_i2t = scores_t2i.T.contiguous()
    n_images = scores_i2t.shape[0]
    grouped = scores_i2t.reshape(n_images, n_images, k_captions)
    image_idx = torch.arange(n_images, device=scores_t2i.device)
    positive_best = grouped[image_idx, image_idx].max(dim=1).values
    return (scores_i2t > positive_best[:, None]).sum(dim=1) + 1


def r1_outcomes(scores_t2i: torch.Tensor, k_captions: int = 5) -> dict[str, torch.Tensor]:
    return {
        "i2t": i2t_ranks(scores_t2i, k_captions) <= 1,
        "t2i": t2i_ranks(scores_t2i, k_captions) <= 1,
    }


def _validate_score_matrix(scores: torch.Tensor, k_captions: int) -> None:
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [N*K, N], got {tuple(scores.shape)}")

    if scores.shape[0] != scores.shape[1] * k_captions:
        raise ValueError(
            f"score rows must equal columns * k_captions, got {tuple(scores.shape)} and K={k_captions}"
        )
