from __future__ import annotations

import torch


def normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def compute_full_consensus(caps: torch.Tensor) -> torch.Tensor:
    """Average support captions per image and normalize to [N, D]."""

    _validate_captions(caps)
    return normalize(caps.mean(dim=1))


def leave_one_out_consensus(caps: torch.Tensor) -> torch.Tensor:
    """Build c_i,-m for each image/caption pair as [N, K, D]."""

    _validate_captions(caps)
    _, k_captions, _ = caps.shape
    total = caps.sum(dim=1, keepdim=True)
    denominator = max(k_captions - 1, 1)
    return normalize((total - caps) / denominator)


def compute_residual(images: torch.Tensor, consensus: torch.Tensor) -> torch.Tensor:
    """Normalize the image-private residual after projecting image onto consensus."""

    _validate_pair(images, consensus)
    coeff = (images * consensus).sum(dim=-1, keepdim=True)
    projection = coeff * consensus
    return normalize(images - projection)


def score_matrix_no_loo(
    images: torch.Tensor,
    captions: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.4,
    gamma: float = 1.8,
    query_batch_size: int | None = None,
) -> torch.Tensor:
    """Score all caption queries against all image candidates without LOO correction."""

    return _score_matrix(
        images=images,
        captions=captions,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        leave_one_out=False,
        query_batch_size=query_batch_size,
    )


def score_matrix_with_loo(
    images: torch.Tensor,
    captions: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.4,
    gamma: float = 1.8,
    query_batch_size: int | None = None,
) -> torch.Tensor:
    """Score all caption queries and correct the positive image column with LOO support."""

    return _score_matrix(
        images=images,
        captions=captions,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        leave_one_out=True,
        query_batch_size=query_batch_size,
    )


def _score_matrix(
    images: torch.Tensor,
    captions: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    leave_one_out: bool,
    query_batch_size: int | None,
) -> torch.Tensor:
    _validate_images_and_captions(images, captions)

    n_images, k_captions, dim = captions.shape
    queries = captions.reshape(n_images * k_captions, dim)
    consensus = compute_full_consensus(captions)
    residual = compute_residual(images, consensus)

    scores = _weighted_scores(
        queries=queries,
        images=images,
        consensus=consensus,
        residual=residual,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        query_batch_size=query_batch_size,
    )

    if leave_one_out:
        _apply_loo_positive_correction(
            scores=scores,
            images=images,
            captions=captions,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    return scores


def _weighted_scores(
    queries: torch.Tensor,
    images: torch.Tensor,
    consensus: torch.Tensor,
    residual: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    query_batch_size: int | None,
) -> torch.Tensor:
    if query_batch_size is None:
        return (
            alpha * (queries @ images.T)
            + beta * (queries @ consensus.T)
            + gamma * (queries @ residual.T)
        )

    chunks = [
        (
            alpha * (chunk @ images.T)
            + beta * (chunk @ consensus.T)
            + gamma * (chunk @ residual.T)
        )
        for chunk in queries.split(query_batch_size, dim=0)
    ]
    return torch.cat(chunks, dim=0)


def _apply_loo_positive_correction(
    scores: torch.Tensor,
    images: torch.Tensor,
    captions: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
) -> None:
    n_images, k_captions, dim = captions.shape
    queries = captions.reshape(n_images * k_captions, dim)
    columns = torch.arange(n_images, device=scores.device).repeat_interleave(k_captions)
    rows = torch.arange(n_images * k_captions, device=scores.device)

    loo_consensus = leave_one_out_consensus(captions).reshape(n_images * k_captions, dim)
    positive_images = images.repeat_interleave(k_captions, dim=0)
    loo_residual = compute_residual(positive_images, loo_consensus)

    corrected = (
        alpha * (positive_images * queries).sum(dim=-1)
        + beta * (loo_consensus * queries).sum(dim=-1)
        + gamma * (loo_residual * queries).sum(dim=-1)
    )
    scores[rows, columns] = corrected


def _validate_images_and_captions(images: torch.Tensor, captions: torch.Tensor) -> None:
    _validate_captions(captions)

    if images.ndim != 2:
        raise ValueError(f"images must have shape [N, D], got {tuple(images.shape)}")

    if images.shape[0] != captions.shape[0]:
        raise ValueError("images and captions must have the same number of image records")

    if images.shape[1] != captions.shape[2]:
        raise ValueError("image and caption embedding dimensions must match")


def _validate_captions(captions: torch.Tensor) -> None:
    if captions.ndim != 3:
        raise ValueError(f"captions must have shape [N, K, D], got {tuple(captions.shape)}")


def _validate_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")

