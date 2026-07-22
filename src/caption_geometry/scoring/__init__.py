"""Scoring and retrieval metrics."""

from caption_geometry.scoring.consensus_residual import (
    compute_full_consensus,
    compute_residual,
    leave_one_out_consensus,
    normalize,
    score_matrix_no_loo,
    score_matrix_with_loo,
)
from caption_geometry.scoring.metrics import recall_at_k_i2t, recall_at_k_t2i
from caption_geometry.scoring.retrieval_scores import (
    ALL_RETRIEVAL_VARIANTS,
    SUPPORT_BASELINE_VARIANTS,
    VARIANT_LABELS,
    score_retrieval_variant,
)

__all__ = [
    "compute_full_consensus",
    "compute_residual",
    "leave_one_out_consensus",
    "normalize",
    "recall_at_k_i2t",
    "recall_at_k_t2i",
    "score_matrix_no_loo",
    "score_matrix_with_loo",
    "ALL_RETRIEVAL_VARIANTS",
    "SUPPORT_BASELINE_VARIANTS",
    "VARIANT_LABELS",
    "score_retrieval_variant",
]
