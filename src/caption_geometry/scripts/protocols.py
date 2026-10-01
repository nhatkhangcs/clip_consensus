from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from caption_geometry.scoring.metrics import r1_outcomes, retrieval_metrics
from caption_geometry.scoring.retrieval_scores import score_retrieval_variant

PROTOCOLS = ("loo", "fixed")


@dataclass(frozen=True)
class ProtocolSplit:
    """One evaluation pass: queries scored against candidates carrying `support` captions."""

    query_captions: torch.Tensor
    support_captions: torch.Tensor
    support_source_indices: torch.Tensor | None
    leave_one_out: bool


@dataclass(frozen=True)
class ProtocolResult:
    metrics: dict[str, dict[str, float]]
    outcomes: dict[str, torch.Tensor]
    clusters: dict[str, torch.Tensor]


def protocol_splits(
    captions: torch.Tensor,
    protocol: str,
    support_captions: torch.Tensor | None = None,
    support_source_indices: torch.Tensor | None = None,
) -> Iterator[ProtocolSplit]:
    """Yield the query/support splits that make up one protocol.

    loo:   every caption is a query; the positive candidate drops the query from its
           support set (K-1 supports) while negatives keep all K.
    fixed: fold f holds caption f out as the only query; every candidate uses the
           remaining K-1 captions, so the support index never depends on the query.
    """

    if protocol == "loo":
        support = captions if support_captions is None else support_captions
        yield ProtocolSplit(captions, support, support_source_indices, leave_one_out=True)
        return

    if protocol == "fixed":
        k_captions = captions.shape[1]

        if k_captions < 2:
            raise ValueError("fixed protocol needs at least two captions per image")

        for fold in range(k_captions):
            keep = [index for index in range(k_captions) if index != fold]
            yield ProtocolSplit(
                query_captions=captions[:, fold : fold + 1],
                support_captions=captions[:, keep],
                support_source_indices=None,
                leave_one_out=False,
            )
        return

    raise ValueError(f"Unsupported protocol: {protocol}")


def evaluate_protocol(
    variant: str,
    images: torch.Tensor,
    captions: torch.Tensor,
    protocol: str,
    alpha: float,
    beta: float,
    gamma: float,
    query_batch_size: int | None = None,
    splits: list[ProtocolSplit] | None = None,
) -> ProtocolResult:
    """Score one variant under a protocol, averaging metrics over its splits.

    Outcomes are R@1 hits per query, concatenated across splits; clusters give the
    image each outcome belongs to, for image-level bootstrap resampling.
    """

    resolved = splits if splits is not None else list(protocol_splits(captions, protocol))
    metric_rows = []
    outcomes = {"i2t": [], "t2i": []}
    clusters = {"i2t": [], "t2i": []}
    image_ids = torch.arange(images.shape[0], device=images.device)

    for split in resolved:
        k_queries = split.query_captions.shape[1]
        scores = score_retrieval_variant(
            variant=variant,
            images=images,
            query_captions=split.query_captions,
            support_captions=split.support_captions,
            support_source_indices=split.support_source_indices,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            leave_one_out=split.leave_one_out,
            query_batch_size=query_batch_size,
        )
        metric_rows.append(retrieval_metrics(scores, k_captions=k_queries))
        hits = r1_outcomes(scores, k_captions=k_queries)
        outcomes["i2t"].append(hits["i2t"])
        outcomes["t2i"].append(hits["t2i"])
        clusters["i2t"].append(image_ids)
        clusters["t2i"].append(image_ids.repeat_interleave(k_queries))

    return ProtocolResult(
        metrics=average_metrics(metric_rows),
        outcomes={key: torch.cat(value) for key, value in outcomes.items()},
        clusters={key: torch.cat(value) for key, value in clusters.items()},
    )


def average_metrics(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        direction: {
            key: sum(row[direction][key] for row in rows) / len(rows)
            for key in rows[0][direction]
        }
        for direction in rows[0]
    }


def slice_images(
    images: torch.Tensor,
    captions: torch.Tensor,
    image_range: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict to images [start, end), e.g. COCO tune2500 = 0 2500, eval2500 = 2500 5000."""

    if image_range is None:
        return images, captions

    start, end = image_range
    return images[start:end], captions[start:end]
