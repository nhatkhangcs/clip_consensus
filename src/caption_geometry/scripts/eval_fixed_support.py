from __future__ import annotations

import argparse
import json
from pathlib import Path

from caption_geometry.io import write_json
from caption_geometry.scoring.bootstrap import bootstrap_gain_ci, bootstrap_mean_ci
from caption_geometry.scoring.retrieval_scores import VARIANT_LABELS
from caption_geometry.scripts.common import load_embedding_pair, resolve_compute_device
from caption_geometry.scripts.protocols import PROTOCOLS, evaluate_protocol, protocol_splits, slice_images
from caption_geometry.scripts.table_utils import format_ci, format_metric, markdown_table, metric_value
from caption_geometry.scripts.tune_weights import TUNED_WEIGHTS

DEFAULT_VARIANTS = ("base", "base_avg_support", "base_max_support", "base_consensus", "base_residual", "full")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate scoring variants under a support-independent protocol: each fold holds one "
            "caption per image out as the query and every candidate keeps the same K-1 supports."
        )
    )
    parser.add_argument("--image-emb")
    parser.add_argument("--caption-emb")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--backbone")
    parser.add_argument(
        "--image-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Evaluate on images [START, END), e.g. 2500 5000 for COCO eval2500 within test5k.",
    )
    parser.add_argument("--protocol", choices=PROTOCOLS, default="fixed")
    parser.add_argument("--variants", nargs="+", choices=tuple(TUNED_WEIGHTS), default=DEFAULT_VARIANTS)
    parser.add_argument(
        "--weights-json",
        help="tune_weights output; each variant uses its own tuned weights. Overrides --beta/--gamma.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=1.8)
    parser.add_argument(
        "--reference",
        choices=tuple(TUNED_WEIGHTS),
        default="base",
        help="Variant the paired-gain columns are measured against, e.g. base_consensus.",
    )
    parser.add_argument("--query-batch-size", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_compute_device(args.device)
    images, captions, metadata = load_embedding_pair(
        image_path=args.image_emb,
        caption_path=args.caption_emb,
        dataset=args.dataset,
        split=args.split,
        backbone=args.backbone,
        device=device,
    )
    images, captions = slice_images(images, captions, args.image_range)
    weights = resolve_weights(args)
    splits = list(protocol_splits(captions, args.protocol))
    results = {}
    outcomes = {}

    for variant in args.variants:
        result = evaluate_protocol(
            variant=variant,
            images=images,
            captions=captions,
            protocol=args.protocol,
            query_batch_size=args.query_batch_size,
            splits=splits,
            **weights[variant],
        )
        results[variant] = {
            "label": VARIANT_LABELS[variant],
            "weights": weights[variant],
            "metrics": result.metrics,
        }
        outcomes[variant] = result

    if args.bootstrap_samples > 0:
        attach_clustered_intervals(results, outcomes, args.reference, args.bootstrap_samples, args.bootstrap_seed)

    table = results_table(results, include_ci=args.bootstrap_samples > 0, reference=args.reference)
    payload = {
        "protocol": args.protocol,
        "image_range": args.image_range,
        "reference": args.reference,
        "weights_json": args.weights_json,
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed, "cluster": "image"},
        "metadata": metadata,
        "results": results,
        "markdown_table": table,
    }
    write_json(args.output, payload)
    write_markdown(args.markdown_output, table)
    print(table)


def resolve_weights(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Per-variant weights: tuned values if given, else CLI weights masked to what the variant uses."""

    tuned = {}

    if args.weights_json:
        payload = json.loads(Path(args.weights_json).read_text(encoding="utf-8"))
        tuned = payload["best_by_variant"]

        if payload.get("protocol") != args.protocol:
            print(f"warning: weights were tuned under protocol={payload.get('protocol')!r}, evaluating {args.protocol!r}")

    weights = {}

    for variant in args.variants:
        if variant in tuned:
            row = tuned[variant]
            weights[variant] = {"alpha": row["alpha"], "beta": row["beta"], "gamma": row["gamma"]}
            continue

        used = TUNED_WEIGHTS[variant]
        weights[variant] = {
            "alpha": args.alpha,
            "beta": args.beta if "beta" in used else 0.0,
            "gamma": args.gamma if "gamma" in used else 0.0,
        }

    return weights


def attach_clustered_intervals(results, outcomes, reference: str, samples: int, seed: int) -> None:
    baseline = outcomes.get(reference)

    for variant, result in outcomes.items():
        results[variant]["r1_ci"] = {
            direction: bootstrap_mean_ci(
                values, samples=samples, seed=seed, clusters=result.clusters[direction]
            )
            for direction, values in result.outcomes.items()
        }

        if baseline is None:
            continue

        results[variant]["r1_gain_vs_reference_ci"] = {
            direction: bootstrap_gain_ci(
                baseline.outcomes[direction],
                values,
                samples=samples,
                seed=seed,
                clusters=result.clusters[direction],
            )
            for direction, values in result.outcomes.items()
        }


def results_table(results: dict[str, dict[str, object]], include_ci: bool, reference: str = "base") -> str:
    has_gain = include_ci and all("r1_gain_vs_reference_ci" in row for row in results.values())
    headers = ["Method", "beta", "gamma", "I2T@1", "T2I@1"]

    if has_gain:
        name = VARIANT_LABELS[reference]
        headers.extend([f"Delta I2T@1 vs {name}", f"Delta T2I@1 vs {name}"])

    rows = []

    for row in results.values():
        weights = row["weights"]
        line = [row["label"], f"{weights['beta']:g}", f"{weights['gamma']:g}"]

        if include_ci:
            line.extend([format_ci(row["r1_ci"]["i2t"]), format_ci(row["r1_ci"]["t2i"])])
        else:
            line.extend(
                [format_metric(metric_value(row["metrics"], "i2t")), format_metric(metric_value(row["metrics"], "t2i"))]
            )

        if has_gain:
            gain = row["r1_gain_vs_reference_ci"]
            line.extend([format_ci(gain["i2t"]), format_ci(gain["t2i"])])

        rows.append(line)

    return markdown_table(headers, rows)


def write_markdown(path: str | None, table: str) -> None:
    if not path:
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
