from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, pstdev

import torch

from caption_geometry.io import load_embedding_bundle, write_json
from caption_geometry.scoring.consensus_residual import normalize
from caption_geometry.scoring.retrieval_scores import VARIANT_LABELS
from caption_geometry.scripts.common import (
    corrupt_support_captions,
    load_embedding_pair,
    resolve_compute_device,
    sample_support_captions,
)
from caption_geometry.scripts.protocols import (
    PROTOCOLS,
    ProtocolSplit,
    average_metrics,
    evaluate_protocol,
    protocol_splits,
    slice_images,
)
from caption_geometry.scripts.table_utils import format_metric, markdown_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate robustness to mismatched support captions.")
    parser.add_argument("--image-emb")
    parser.add_argument("--caption-emb")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--backbone")
    parser.add_argument("--image-range", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument("--protocol", choices=PROTOCOLS, default="fixed")
    parser.add_argument(
        "--replacement-pool-emb",
        help=(
            "Caption embeddings to draw mismatched supports from (e.g. COCO train). Default: "
            "fixed protocol uses the other images' supports, which never contain a query; "
            "loo uses the evaluated captions themselves, so a query can land in a negative's support."
        ),
    )
    parser.add_argument("--variant", default="full")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=1.8)
    parser.add_argument("--mismatch-rates", type=float, nargs="+", default=(0.0, 0.2, 0.4, 0.6))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0,))
    parser.add_argument("--support-seed", type=int, default=0)
    parser.add_argument("--support-count", type=int, help="loo protocol only.")
    parser.add_argument("--query-batch-size", type=int)
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
    external_pool = load_pool(args.replacement_pool_emb, device)

    if args.protocol == "fixed" and args.support_count is not None:
        raise ValueError("--support-count applies to the loo protocol only")

    if args.protocol == "loo" and external_pool is None:
        print("warning: loo protocol without --replacement-pool-emb draws replacements from the query set")

    clean_splits = base_splits(captions, args)
    rows = [
        evaluate_rate(rate, clean_splits, images, external_pool, args)
        for rate in args.mismatch_rates
    ]
    table = mismatch_table(rows)
    payload = {
        "protocol": args.protocol,
        "image_range": args.image_range,
        "replacement_pool": args.replacement_pool_emb or ("other_supports" if args.protocol == "fixed" else "queries"),
        "variant": args.variant,
        "variant_label": VARIANT_LABELS.get(args.variant, args.variant),
        "weights": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
        "support_count": clean_splits[0].support_captions.shape[1],
        "support_seed": args.support_seed,
        "seeds": list(args.seeds),
        "metadata": metadata,
        "results": rows,
        "markdown_table": table,
    }
    write_json(args.output, payload)
    write_markdown(args.markdown_output, table)
    print(table)


def load_pool(path: str | None, device):
    if not path:
        return None

    return normalize(load_embedding_bundle(path, map_location=device).embeddings.float())


def base_splits(captions, args: argparse.Namespace) -> list[ProtocolSplit]:
    if args.protocol == "fixed":
        return list(protocol_splits(captions, "fixed"))

    support, sources = sample_support_captions(captions, args.support_count, args.support_seed)
    return list(protocol_splits(captions, "loo", support, sources))


def corrupt_split(split: ProtocolSplit, rate: float, seed: int, external_pool) -> tuple[ProtocolSplit, int]:
    if external_pool is not None:
        pool, exclude_same_image = external_pool, False
    elif split.leave_one_out:
        pool, exclude_same_image = split.query_captions, True
    else:
        pool, exclude_same_image = split.support_captions, True

    sources = split.support_source_indices

    if sources is None:
        sources = split.support_captions.new_zeros(split.support_captions.shape[:2], dtype=torch.long)

    corrupted, corrupted_sources, replaced = corrupt_support_captions(
        support_captions=split.support_captions,
        support_source_indices=sources,
        caption_pool=pool,
        mismatch_rate=rate,
        seed=seed,
        exclude_same_image=exclude_same_image,
    )
    resolved_sources = corrupted_sources if split.leave_one_out else None
    return (
        ProtocolSplit(split.query_captions, corrupted, resolved_sources, split.leave_one_out),
        replaced,
    )


def evaluate_rate(rate: float, clean_splits, images, external_pool, args) -> dict[str, object]:
    seed_rows = [evaluate_seed(rate, seed, clean_splits, images, external_pool, args) for seed in args.seeds]
    return {
        "mismatch_rate": rate,
        "mean": summarize(seed_rows),
        "seeds": seed_rows,
    }


def evaluate_seed(rate: float, seed: int, clean_splits, images, external_pool, args) -> dict[str, object]:
    metric_rows = []
    replaced_total = 0

    # Fold 0 keeps `seed` (matches earlier loo runs); later folds get independent draws.
    for fold, split in enumerate(clean_splits):
        corrupted, replaced = corrupt_split(split, rate, seed + 1000 * fold, external_pool)
        replaced_total += replaced
        result = evaluate_protocol(
            variant=args.variant,
            images=images,
            captions=split.query_captions,
            protocol=args.protocol,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            query_batch_size=args.query_batch_size,
            splits=[corrupted],
        )
        metric_rows.append(result.metrics)

    return {
        "seed": seed,
        "replaced_support_captions": replaced_total,
        "metrics": average_metrics(metric_rows),
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

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
