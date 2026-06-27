from __future__ import annotations

import torch

from caption_geometry.scoring.consensus_residual import leave_one_out_consensus


def caption_consensus_similarity(captions: torch.Tensor) -> torch.Tensor:
    loo = leave_one_out_consensus(captions)
    return (loo * captions).sum(dim=-1)


def caption_residual_norm(captions: torch.Tensor) -> torch.Tensor:
    loo = leave_one_out_consensus(captions)
    coeff = (captions * loo).sum(dim=-1, keepdim=True)
    residual = captions - coeff * loo
    return residual.norm(dim=-1)

