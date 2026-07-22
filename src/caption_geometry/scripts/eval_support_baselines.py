from __future__ import annotations

import argparse
import json
from pathlib import Path

from caption_geometry.io import write_json
from caption_geometry.scoring.bootstrap import bootstrap_gain_ci, bootstrap_mean_ci
from caption_geometry.scoring.metrics import r1_outcomes, retrieval_metrics
from caption_geometry.scoring.retrieval_scores import (
    SUPPORT_BASELINE_VARIANTS,
    VARIANT_LABELS,
    score_retrieval_variant,
)
from caption_geometry.scripts.common import (
    load_embedding_pair,
    resolve_compute_device,
    sample_support_captions,
)
from caption_geometry.scripts.table_utils import format_ci, format_metric, markdown_table, metric_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate reviewer-facing support-caption baselines.")
    parser.add_argument("--image-emb")
    parser.add_argument("--caption-emb")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--backbone")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda-weight", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=1.8)
    parser.add_argument("--leave-one-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-batch-size", type=int)
    parser.add_argument("--support-count", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
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
        args.seed,
    )
    results = evaluate_support_baselines(
        images=images,
        query_captions=query_captions,
        support_captions=support_captions,
        support_source_indices=support_source_indices,
        alpha=args.alpha,
        support_weight=args.lambda_weight,
        gamma=args.gamma,
        leave_one_out=args.leave_one_out,
        query_batch_size=args.query_batch_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    table = support_baseline_table(results, include_ci=args.bootstrap_samples > 0)
    payload = {
        "weights": {
            "alpha": args.alpha,
            "lambda": args.lambda_weight,
            "beta": args.lambda_weight,
            "gamma": args.gamma,
        },
        "leave_one_out": args.leave_one_out,
        "support_count": support_captions.shape[1],
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "metadata": metadata,
        "results": results,
        "markdown_table": table,
    }
    write_json(args.output, payload)
    write_markdown(args.markdown_output, table)
    print(table)


def evaluate_support_baselines(
    images,
    query_captions,
    support_captions,
    support_source_indices,
    alpha: float,
    support_weight: float,
    gamma: float,
    leave_one_out: bool,
    query_batch_size: int | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, object]]:
    results = {}
    outcomes = {}

    for variant in SUPPORT_BASELINE_VARIANTS:
        scores = score_retrieval_variant(
            variant=variant,
            images=images,
            query_captions=query_captions,
            support_captions=support_captions,
            support_source_indices=support_source_indices,
            alpha=alpha,
            beta=support_weight,
            gamma=gamma,
            leave_one_out=leave_one_out,
            query_batch_size=query_batch_size,
        )
        metrics = retrieval_metrics(scores, k_captions=query_captions.shape[1])
        outcomes[variant] = r1_outcomes(scores, k_captions=query_captions.shape[1])
        results[variant] = {
            "label": VARIANT_LABELS[variant],
            "metrics": metrics,
        }

    if bootstrap_samples > 0:
        attach_bootstrap_intervals(results, outcomes, bootstrap_samples, bootstrap_seed)

    return results


def attach_bootstrap_intervals(
    results: dict[str, dict[str, object]],
    outcomes: dict[str, dict[str, object]],
    samples: int,
    seed: int,
) -> None:
    baseline = outcomes["base"]

    for variant, variant_outcomes in outcomes.items():
        ci = {
            direction: bootstrap_mean_ci(values, samples=samples, seed=seed)
            for direction, values in variant_outcomes.items()
        }
        gain_ci = {
            direction: bootstrap_gain_ci(
                baseline[direction],
                values,
                samples=samples,
                seed=seed,
            )
            for direction, values in variant_outcomes.items()
        }
        results[variant]["r1_ci"] = ci
        results[variant]["r1_gain_vs_base_ci"] = gain_ci


def support_baseline_table(
    results: dict[str, dict[str, object]],
    include_ci: bool,
) -> str:
    headers = ["Method", "I2T@1", "T2I@1"]

    if include_ci:
        headers.extend(["Delta I2T@1 vs Base", "Delta T2I@1 vs Base"])

    rows = []

    for variant in SUPPORT_BASELINE_VARIANTS:
        row = results[variant]
        metrics = row["metrics"]
        i2t = format_metric(metric_value(metrics, "i2t"))
        t2i = format_metric(metric_value(metrics, "t2i"))

        if include_ci:
            i2t = format_ci(row["r1_ci"]["i2t"])
            t2i = format_ci(row["r1_ci"]["t2i"])
            rows.append(
                [
                    row["label"],
                    i2t,
                    t2i,
                    format_ci(row["r1_gain_vs_base_ci"]["i2t"]),
                    format_ci(row["r1_gain_vs_base_ci"]["t2i"]),
                ]
            )
            continue

        rows.append([row["label"], i2t, t2i])

    return markdown_table(headers, rows)


def write_markdown(path: str | None, table: str) -> None:
    if not path:
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
