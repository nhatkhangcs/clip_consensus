from __future__ import annotations

import torch

from caption_geometry.scoring.consensus_residual import compute_residual, leave_one_out_consensus


def caption_consensus_similarity(captions: torch.Tensor) -> torch.Tensor:
    loo = leave_one_out_consensus(captions)
    return (loo * captions).sum(dim=-1)


def caption_residual_norm(captions: torch.Tensor) -> torch.Tensor:
    loo = leave_one_out_consensus(captions)
    coeff = (captions * loo).sum(dim=-1, keepdim=True)
    residual = captions - coeff * loo
    return residual.norm(dim=-1)


def residual_query_similarity(images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
    loo = leave_one_out_consensus(captions)
    expanded_images = images[:, None, :].expand_as(captions)
    residual = compute_residual(
        expanded_images.reshape(-1, expanded_images.shape[-1]),
        loo.reshape(-1, loo.shape[-1]),
    ).reshape_as(captions)
    return (residual * captions).sum(dim=-1)
