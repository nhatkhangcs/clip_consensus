from __future__ import annotations

import argparse
from caption_geometry.io import write_json
from caption_geometry.scoring.metrics import retrieval_metrics
from caption_geometry.scoring.retrieval_scores import (
    ALL_RETRIEVAL_VARIANTS,
    VARIANT_LABELS,
    score_retrieval_variant,
)
from caption_geometry.scripts.common import (
    load_embedding_pair,
    resolve_compute_device,
    sample_support_captions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate text-image retrieval from cached embeddings.")
    parser.add_argument("--image-emb")
    parser.add_argument("--caption-emb")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--backbone")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=1.8)
    parser.add_argument("--variant", choices=ALL_RETRIEVAL_VARIANTS, default="full")
    parser.add_argument("--all-ablations", action="store_true")
    parser.add_argument("--leave-one-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-batch-size", type=int)
    parser.add_argument("--support-count", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_compute_device(args.device)
    images, query_captions, metadata = load_embedding_pair(
        image_path=args.image_emb,
        caption_path=args.caption_emb,
        dataset=args.dataset,
        split=args.split,
        backbone=args.backbone,
        device=device,
    )
    support_captions, support_source_indices = sample_support_captions(
        query_captions,
        args.support_count,
        args.seed,
    )

    variant_names = ALL_RETRIEVAL_VARIANTS if args.all_ablations else (args.variant,)
    results = {
        name: evaluate_variant(
            name=name,
            images=images,
            query_captions=query_captions,
            support_captions=support_captions,
            support_source_indices=support_source_indices,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            leave_one_out=args.leave_one_out,
            query_batch_size=args.query_batch_size,
        )
        for name in variant_names
    }
    payload = {
        "weights": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
        "leave_one_out": args.leave_one_out,
        "support_count": support_captions.shape[1],
        "seed": args.seed,
        "metadata": metadata,
        "results": results,
    }

    if args.output:
        write_json(args.output, payload)

    print_json(payload)


def evaluate_variant(
    name: str,
    images,
    query_captions,
    support_captions,
    support_source_indices,
    alpha: float,
    beta: float,
    gamma: float,
    leave_one_out: bool,
    query_batch_size: int | None,
) -> dict[str, object]:
    scores = score_retrieval_variant(
        variant=name,
        images=images,
        query_captions=query_captions,
        support_captions=support_captions,
        support_source_indices=support_source_indices,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        leave_one_out=leave_one_out,
        query_batch_size=query_batch_size,
    )
    return {
        "label": VARIANT_LABELS[name],
        "metrics": retrieval_metrics(scores, k_captions=query_captions.shape[1]),
    }


def print_json(payload: dict[str, object]) -> None:
    import json

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
