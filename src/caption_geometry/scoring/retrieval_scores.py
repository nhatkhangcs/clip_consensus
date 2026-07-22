from __future__ import annotations

from collections.abc import Callable

import torch

from caption_geometry.scoring.consensus_residual import (
    compute_full_consensus,
    compute_residual,
    normalize,
)

SUPPORT_BASELINE_VARIANTS = (
    "base",
    "base_avg_support",
    "base_max_support",
    "base_consensus",
    "full",
)

ALL_RETRIEVAL_VARIANTS = (
    "base",
    "base_avg_support",
    "base_max_support",
    "base_consensus",
    "base_residual",
    "consensus_only",
    "residual_only",
    "consensus_residual",
    "full",
)

VARIANT_LABELS = {
    "base": "Base image-text",
    "base_avg_support": "Base + avg support-caption sim",
    "base_max_support": "Base + max support-caption sim",
    "base_consensus": "Base + consensus",
    "base_residual": "Base + residual",
    "consensus_only": "Consensus only",
    "residual_only": "Residual only",
    "consensus_residual": "Consensus + residual",
    "full": "Full consensus-residual",
}


def score_retrieval_variant(
    variant: str,
    images: torch.Tensor,
    query_captions: torch.Tensor,
    support_captions: torch.Tensor | None = None,
    support_source_indices: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.4,
    gamma: float = 1.8,
    leave_one_out: bool = True,
    query_batch_size: int | None = None,
) -> torch.Tensor:
    support = query_captions if support_captions is None else support_captions
    source_indices = _resolve_support_source_indices(
        query_captions=query_captions,
        support_captions=support,
        support_source_indices=support_source_indices,
        leave_one_out=leave_one_out,
    )

    scorers: dict[str, Callable[[], torch.Tensor]] = {
        "base": lambda: score_base(
            images=images,
            query_captions=query_captions,
            alpha=alpha,
            query_batch_size=query_batch_size,
        ),
        "base_avg_support": lambda: score_base_avg_support(
            images=images,
            query_captions=query_captions,
            support_captions=support,
            support_source_indices=source_indices,
            alpha=alpha,
            support_weight=beta,
            leave_one_out=leave_one_out,
            query_batch_size=query_batch_size,
        ),
        "base_max_support": lambda: score_base_max_support(
            images=images,
            query_captions=query_captions,
            support_captions=support,
            support_source_indices=source_indices,
            alpha=alpha,
            support_weight=beta,
            leave_one_out=leave_one_out,
            query_batch_size=query_batch_size,
        ),
        "base_consensus": lambda: score_consensus_residual(
            images=images,
            query_captions=query_captions,
            support_captions=support,
            support_source_indices=source_indices,
            alpha=alpha,
            beta=beta,
            gamma=0.0,
            leave_one_out=leave_one_out,
            query_batch_size=query_batch_size,
        ),
        "base_residual": lambda: score_consensus_residual(
            images=images,
            query_captions=query_captions,
            support_captions=support,
            support_source_indices=source_indices,
            alpha=alpha,
            beta=0.0,
            gamma=gamma,
            leave_one_out=leave_one_out,
            query_batch_size=query_batch_size,
        ),
        "consensus_only": lambda: score_consensus_residual(
            images=images,
            query_captions=query_captions,
            support_captions=support,
            support_source_indices=source_indices,
            alpha=0.0,
            beta=beta,
            gamma=0.0,
            leave_one_out=leave_one_out,
            query_batch_size=query_batch_size,
        ),
        "residual_only": lambda: score_consensus_residual(
            images=images,
            query_captions=query_captions,
            support_captions=support,
            support_source_indices=source_indices,
            alpha=0.0,
            beta=0.0,
            gamma=gamma,
            leave_one_out=leave_one_out,
            query_batch_size=query_batch_size,
        ),
        "consensus_residual": lambda: score_consensus_residual(
            images=images,
            query_captions=query_captions,
            support_captions=support,
            support_source_indices=source_indices,
            alpha=0.0,
            beta=beta,
            gamma=gamma,
            leave_one_out=leave_one_out,
            query_batch_size=query_batch_size,
        ),
        "full": lambda: score_consensus_residual(
            images=images,
            query_captions=query_captions,
            support_captions=support,
            support_source_indices=source_indices,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            leave_one_out=leave_one_out,
            query_batch_size=query_batch_size,
        ),
    }

    if variant not in scorers:
        raise ValueError(f"Unsupported retrieval variant: {variant}")

    return scorers[variant]()


def score_base(
    images: torch.Tensor,
    query_captions: torch.Tensor,
    alpha: float = 1.0,
    query_batch_size: int | None = None,
) -> torch.Tensor:
    _validate_retrieval_inputs(images, query_captions)
    queries = _flatten_queries(query_captions)
    return alpha * _chunked_matmul(queries, images, query_batch_size)


def score_base_avg_support(
    images: torch.Tensor,
    query_captions: torch.Tensor,
    support_captions: torch.Tensor,
    support_source_indices: torch.Tensor | None = None,
    alpha: float = 1.0,
    support_weight: float = 0.4,
    leave_one_out: bool = True,
    query_batch_size: int | None = None,
) -> torch.Tensor:
    _validate_retrieval_inputs(images, query_captions, support_captions)
    queries = _flatten_queries(query_captions)
    support_mean = support_captions.mean(dim=1)
    scores = (
        alpha * _chunked_matmul(queries, images, query_batch_size)
        + support_weight * _chunked_matmul(queries, support_mean, query_batch_size)
    )

    if leave_one_out:
        source_indices = _resolve_support_source_indices(
            query_captions, support_captions, support_source_indices, leave_one_out=True
        )
        _replace_positive_scores(
            scores=scores,
            images=images,
            query_captions=query_captions,
            component=_positive_avg_support_component(
                query_captions=query_captions,
                support_captions=support_captions,
                support_source_indices=source_indices,
            ),
            alpha=alpha,
            component_weight=support_weight,
        )

    return scores


def score_base_max_support(
    images: torch.Tensor,
    query_captions: torch.Tensor,
    support_captions: torch.Tensor,
    support_source_indices: torch.Tensor | None = None,
    alpha: float = 1.0,
    support_weight: float = 0.4,
    leave_one_out: bool = True,
    query_batch_size: int | None = None,
) -> torch.Tensor:
    _validate_retrieval_inputs(images, query_captions, support_captions)
    queries = _flatten_queries(query_captions)
    scores = (
        alpha * _chunked_matmul(queries, images, query_batch_size)
        + support_weight * _chunked_support_max(queries, support_captions, query_batch_size)
    )

    if leave_one_out:
        source_indices = _resolve_support_source_indices(
            query_captions, support_captions, support_source_indices, leave_one_out=True
        )
        _replace_positive_scores(
            scores=scores,
            images=images,
            query_captions=query_captions,
            component=_positive_max_support_component(
                query_captions=query_captions,
                support_captions=support_captions,
                support_source_indices=source_indices,
            ),
            alpha=alpha,
            component_weight=support_weight,
        )

    return scores


def score_consensus_residual(
    images: torch.Tensor,
    query_captions: torch.Tensor,
    support_captions: torch.Tensor,
    support_source_indices: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.4,
    gamma: float = 1.8,
    leave_one_out: bool = True,
    query_batch_size: int | None = None,
) -> torch.Tensor:
    _validate_retrieval_inputs(images, query_captions, support_captions)
    queries = _flatten_queries(query_captions)
    consensus = compute_full_consensus(support_captions)
    residual = compute_residual(images, consensus)
    scores = (
        alpha * _chunked_matmul(queries, images, query_batch_size)
        + beta * _chunked_matmul(queries, consensus, query_batch_size)
        + gamma * _chunked_matmul(queries, residual, query_batch_size)
    )

    if leave_one_out:
        source_indices = _resolve_support_source_indices(
            query_captions, support_captions, support_source_indices, leave_one_out=True
        )
        positive_consensus = _positive_consensus(support_captions, source_indices, query_captions.shape[1])
        positive_images = images[:, None, :].expand_as(positive_consensus)
        positive_residual = compute_residual(
            positive_images.reshape(-1, positive_images.shape[-1]),
            positive_consensus.reshape(-1, positive_consensus.shape[-1]),
        ).reshape_as(positive_consensus)
        positive_queries = query_captions
        component = (
            beta * (positive_consensus * positive_queries).sum(dim=-1)
            + gamma * (positive_residual * positive_queries).sum(dim=-1)
        )
        _replace_positive_scores(
            scores=scores,
            images=images,
            query_captions=query_captions,
            component=component,
            alpha=alpha,
            component_weight=1.0,
        )

    return scores


def default_support_source_indices(
    captions: torch.Tensor,
    support_count: int | None = None,
) -> torch.Tensor:
    n_images, k_captions, _ = captions.shape
    count = k_captions if support_count is None else support_count
    return torch.arange(count, device=captions.device).expand(n_images, count)


def _resolve_support_source_indices(
    query_captions: torch.Tensor,
    support_captions: torch.Tensor,
    support_source_indices: torch.Tensor | None,
    leave_one_out: bool,
) -> torch.Tensor | None:
    if not leave_one_out:
        return None

    if support_source_indices is not None:
        return support_source_indices.to(device=support_captions.device, dtype=torch.long)

    if support_captions.shape[1] == query_captions.shape[1]:
        return default_support_source_indices(support_captions)

    return torch.full(
        support_captions.shape[:2],
        fill_value=-1,
        dtype=torch.long,
        device=support_captions.device,
    )


def _positive_support_mask(
    support_source_indices: torch.Tensor,
    query_count: int,
) -> torch.Tensor:
    query_ids = torch.arange(query_count, device=support_source_indices.device)
    return support_source_indices[:, None, :] != query_ids[None, :, None]


def _positive_avg_support_component(
    query_captions: torch.Tensor,
    support_captions: torch.Tensor,
    support_source_indices: torch.Tensor,
) -> torch.Tensor:
    mask = _positive_support_mask(support_source_indices, query_captions.shape[1])
    sims = torch.einsum("nkd,nsd->nks", query_captions, support_captions)
    masked = sims * mask.to(dtype=sims.dtype)
    counts = mask.sum(dim=-1).clamp_min(1)
    return masked.sum(dim=-1) / counts


def _positive_max_support_component(
    query_captions: torch.Tensor,
    support_captions: torch.Tensor,
    support_source_indices: torch.Tensor,
) -> torch.Tensor:
    mask = _positive_support_mask(support_source_indices, query_captions.shape[1])
    sims = torch.einsum("nkd,nsd->nks", query_captions, support_captions)
    values = sims.masked_fill(~mask, -torch.inf).max(dim=-1).values
    return torch.where(mask.any(dim=-1), values, torch.zeros_like(values))


def _positive_consensus(
    support_captions: torch.Tensor,
    support_source_indices: torch.Tensor,
    query_count: int,
) -> torch.Tensor:
    mask = _positive_support_mask(support_source_indices, query_count)
    weights = mask.to(dtype=support_captions.dtype)
    summed = torch.einsum("nks,nsd->nkd", weights, support_captions)
    counts = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return normalize(summed / counts)


def _replace_positive_scores(
    scores: torch.Tensor,
    images: torch.Tensor,
    query_captions: torch.Tensor,
    component: torch.Tensor,
    alpha: float,
    component_weight: float,
) -> None:
    n_images, k_queries, _ = query_captions.shape
    queries = _flatten_queries(query_captions)
    rows = torch.arange(n_images * k_queries, device=scores.device)
    columns = torch.arange(n_images, device=scores.device).repeat_interleave(k_queries)
    positive_images = images.repeat_interleave(k_queries, dim=0)
    base = alpha * (positive_images * queries).sum(dim=-1)
    scores[rows, columns] = base + component_weight * component.reshape(-1)


def _flatten_queries(query_captions: torch.Tensor) -> torch.Tensor:
    return query_captions.reshape(query_captions.shape[0] * query_captions.shape[1], -1)


def _chunked_matmul(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    query_batch_size: int | None,
) -> torch.Tensor:
    if query_batch_size is None:
        return queries @ candidates.T

    return torch.cat([chunk @ candidates.T for chunk in queries.split(query_batch_size, dim=0)], dim=0)


def _chunked_support_max(
    queries: torch.Tensor,
    support_captions: torch.Tensor,
    query_batch_size: int | None,
) -> torch.Tensor:
    chunks = queries.split(query_batch_size, dim=0) if query_batch_size else (queries,)
    return torch.cat(
        [
            torch.einsum("bd,nsd->bns", chunk, support_captions).max(dim=-1).values
            for chunk in chunks
        ],
        dim=0,
    )


def _validate_retrieval_inputs(
    images: torch.Tensor,
    query_captions: torch.Tensor,
    support_captions: torch.Tensor | None = None,
) -> None:
    if images.ndim != 2:
        raise ValueError(f"images must have shape [N, D], got {tuple(images.shape)}")

    if query_captions.ndim != 3:
        raise ValueError(f"query_captions must have shape [N, K, D], got {tuple(query_captions.shape)}")

    if images.shape[0] != query_captions.shape[0]:
        raise ValueError("images and query_captions must have the same number of records")

    if images.shape[1] != query_captions.shape[2]:
        raise ValueError("image and query embedding dimensions must match")

    if support_captions is None:
        return

    if support_captions.ndim != 3:
        raise ValueError(f"support_captions must have shape [N, S, D], got {tuple(support_captions.shape)}")

    if support_captions.shape[0] != images.shape[0]:
        raise ValueError("support_captions and images must have the same number of records")

    if support_captions.shape[2] != images.shape[1]:
        raise ValueError("support and image embedding dimensions must match")

