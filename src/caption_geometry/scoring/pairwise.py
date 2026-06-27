from __future__ import annotations

import torch

from caption_geometry.scoring.consensus_residual import compute_full_consensus, compute_residual


def candidate_features(
    images: torch.Tensor,
    support_captions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    consensus = compute_full_consensus(support_captions)
    residual = compute_residual(images, consensus)
    return images, consensus, residual


def score_text_against_images(
    text_embeddings: torch.Tensor,
    images: torch.Tensor,
    consensus: torch.Tensor,
    residual: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.4,
    gamma: float = 1.8,
) -> torch.Tensor:
    return (
        alpha * (text_embeddings @ images.T)
        + beta * (text_embeddings @ consensus.T)
        + gamma * (text_embeddings @ residual.T)
    )


def score_image_text_pairs(
    text_embeddings: torch.Tensor,
    image_indices: torch.Tensor,
    images: torch.Tensor,
    consensus: torch.Tensor,
    residual: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.4,
    gamma: float = 1.8,
) -> torch.Tensor:
    selected_images = images[image_indices]
    selected_consensus = consensus[image_indices]
    selected_residual = residual[image_indices]
    return (
        alpha * (text_embeddings * selected_images).sum(dim=-1)
        + beta * (text_embeddings * selected_consensus).sum(dim=-1)
        + gamma * (text_embeddings * selected_residual).sum(dim=-1)
    )

