from __future__ import annotations

import argparse
import json
from statistics import mean, pstdev

from caption_geometry.io import write_json
from caption_geometry.scoring.metrics import retrieval_metrics
from caption_geometry.scoring.retrieval_scores import VARIANT_LABELS, score_retrieval_variant
from caption_geometry.scripts.common import (
    corrupt_support_captions,
    load_embedding_pair,
    resolve_compute_device,
    sample_support_captions,
)
from caption_geometry.scripts.table_utils import format_metric, markdown_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate robustness to mismatched support captions.")
    parser.add_argument("--image-emb")
    parser.add_argument("--caption-emb")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--backbone")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=1.8)
    parser.add_argument("--mismatch-rates", type=float, nargs="+", default=(0.0, 0.2, 0.4, 0.6))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0,))
    parser.add_argument("--support-seed", type=int, default=0)
    parser.add_argument("--support-count", type=int)
    parser.add_argument("--leave-one-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-batch-size", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
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
        args.support_seed,
    )
    rows = [
        evaluate_rate(
            rate=rate,
            seeds=args.seeds,
            images=images,
            query_captions=query_captions,
            support_captions=support_captions,
            support_source_indices=support_source_indices,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            variant=args.variant,
            leave_one_out=args.leave_one_out,
            query_batch_size=args.query_batch_size,
        )
        for rate in args.mismatch_rates
    ]
    table = mismatch_table(rows)
    payload = {
        "variant": args.variant,
        "variant_label": VARIANT_LABELS.get(args.variant, args.variant),
        "weights": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
        "leave_one_out": args.leave_one_out,
        "support_count": support_captions.shape[1],
        "support_seed": args.support_seed,
        "seeds": list(args.seeds),
        "metadata": metadata,
        "results": rows,
        "markdown_table": table,
    }
    write_json(args.output, payload)
    write_markdown(args.markdown_output, table)
    print(table)


def evaluate_rate(
    rate: float,
    seeds: list[int],
    images,
    query_captions,
    support_captions,
    support_source_indices,
    alpha: float,
    beta: float,
    gamma: float,
    variant: str,
    leave_one_out: bool,
    query_batch_size: int | None,
) -> dict[str, object]:
    seed_rows = [
        evaluate_seed(
            rate=rate,
            seed=seed,
            images=images,
            query_captions=query_captions,
            support_captions=support_captions,
            support_source_indices=support_source_indices,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            variant=variant,
            leave_one_out=leave_one_out,
            query_batch_size=query_batch_size,
        )
        for seed in seeds
    ]
    return {
        "mismatch_rate": rate,
        "mean": summarize(seed_rows),
        "seeds": seed_rows,
    }


def evaluate_seed(
    rate: float,
    seed: int,
    images,
    query_captions,
    support_captions,
    support_source_indices,
    alpha: float,
    beta: float,
    gamma: float,
    variant: str,
    leave_one_out: bool,
    query_batch_size: int | None,
) -> dict[str, object]:
    corrupted_support, corrupted_sources, replaced = corrupt_support_captions(
        support_captions=support_captions,
        support_source_indices=support_source_indices,
        caption_pool=query_captions,
        mismatch_rate=rate,
        seed=seed,
    )
    scores = score_retrieval_variant(
        variant=variant,
        images=images,
        query_captions=query_captions,
        support_captions=corrupted_support,
        support_source_indices=corrupted_sources,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        leave_one_out=leave_one_out,
        query_batch_size=query_batch_size,
    )
    metrics = retrieval_metrics(scores, k_captions=query_captions.shape[1])
    return {
        "seed": seed,
        "replaced_support_captions": replaced,
        "metrics": metrics,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    output = {}

    for direction in ("i2t", "t2i"):
        values = [row["metrics"][direction]["R@1"] for row in rows]
        output[direction] = {
            "R@1": mean(values),
            "R@1_std": pstdev(values) if len(values) > 1 else 0.0,
        }

    return output


def mismatch_table(rows: list[dict[str, object]]) -> str:
    baseline = rows[0]["mean"]
    table_rows = []

    for row in rows:
        summary = row["mean"]
        i2t = summary["i2t"]["R@1"]
        t2i = summary["t2i"]["R@1"]
        table_rows.append(
            [
                f"{row['mismatch_rate'] * 100:.0f}%",
                format_mean_std(i2t, summary["i2t"]["R@1_std"]),
                format_delta(i2t - baseline["i2t"]["R@1"]),
                format_mean_std(t2i, summary["t2i"]["R@1_std"]),
                format_delta(t2i - baseline["t2i"]["R@1"]),
            ]
        )

    return markdown_table(
        ["Mismatch rate", "I2T@1", "Delta I2T@1", "T2I@1", "Delta T2I@1"],
        table_rows,
    )


def format_mean_std(mean_value: float, std_value: float) -> str:
    if std_value == 0.0:
        return format_metric(mean_value)

    return f"{format_metric(mean_value)} +/- {format_metric(std_value)}"


def format_delta(value: float) -> str:
    return f"{value:+.2f}"


def write_markdown(path: str | None, table: str) -> None:
    if not path:
        return

    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
