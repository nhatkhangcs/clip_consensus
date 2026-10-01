from __future__ import annotations

import argparse
import json
from itertools import product

from tqdm import tqdm

from caption_geometry.io import write_json
from caption_geometry.scripts.common import load_embedding_pair, resolve_compute_device
from caption_geometry.scripts.protocols import PROTOCOLS, evaluate_protocol, protocol_splits, slice_images

DEFAULT_BETAS = (
    0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0,
)
DEFAULT_GAMMAS = (
    0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.6, 4.2, 5.0, 6.0, 8.0,
)
DEFAULT_VARIANTS = ("base_avg_support", "base_max_support", "base_consensus", "base_residual", "full")

# Which weights each variant actually uses; the rest stay at zero.
TUNED_WEIGHTS = {
    "base": (),
    "base_avg_support": ("beta",),
    "base_max_support": ("beta",),
    "base_consensus": ("beta",),
    "base_residual": ("gamma",),
    "full": ("beta", "gamma"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid-search weights for every scoring variant with one shared objective."
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
        help="Tune on images [START, END), e.g. 0 2500 for COCO tune2500 within test5k.",
    )
    parser.add_argument("--protocol", choices=PROTOCOLS, default="fixed")
    parser.add_argument("--variants", nargs="+", choices=tuple(TUNED_WEIGHTS), default=DEFAULT_VARIANTS)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    parser.add_argument("--query-batch-size", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
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
    splits = list(protocol_splits(captions, args.protocol))
    best_by_variant = {}
    grids = {}

    for variant in args.variants:
        rows = [
            evaluate_weights(variant, images, captions, args, splits, weights)
            for weights in tqdm(weight_grid(variant, args), desc=f"Tuning {variant}")
        ]
        best = dict(max(rows, key=lambda row: row["objective"]))
        best["on_grid_edge"] = grid_edges(variant, best, args)
        best_by_variant[variant] = best
        grids[variant] = rows

    payload = {
        "metric": "i2t_R@1_plus_t2i_R@1",
        "protocol": args.protocol,
        "image_range": args.image_range,
        "best_by_variant": best_by_variant,
        "best": best_by_variant.get("full"),
        "grids": grids,
        "metadata": metadata,
    }
    write_json(args.output, payload)
    print_tuning_result(payload)


def weight_grid(variant: str, args: argparse.Namespace) -> list[dict[str, float]]:
    tuned = TUNED_WEIGHTS[variant]
    betas = args.betas if "beta" in tuned else (0.0,)
    gammas = args.gammas if "gamma" in tuned else (0.0,)
    return [{"alpha": args.alpha, "beta": beta, "gamma": gamma} for beta, gamma in product(betas, gammas)]


def evaluate_weights(variant, images, captions, args, splits, weights) -> dict[str, object]:
    result = evaluate_protocol(
        variant=variant,
        images=images,
        captions=captions,
        protocol=args.protocol,
        query_batch_size=args.query_batch_size,
        splits=splits,
        **weights,
    )
    objective = result.metrics["i2t"]["R@1"] + result.metrics["t2i"]["R@1"]
    return {**weights, "objective": objective, "metrics": result.metrics}


def grid_edges(variant: str, best: dict[str, object], args: argparse.Namespace) -> list[str]:
    """Name the weights whose optimum sits on the largest grid value (grid may be too narrow)."""

    limits = {"beta": max(args.betas), "gamma": max(args.gammas)}
    return [name for name in TUNED_WEIGHTS[variant] if best[name] >= limits[name] > 0]


def print_tuning_result(payload: dict[str, object]) -> None:
    for variant, best in payload["best_by_variant"].items():
        edge = f"  [ON GRID EDGE: {', '.join(best['on_grid_edge'])}]" if best["on_grid_edge"] else ""
        print(
            f"{variant:18s} beta={best['beta']:<6} gamma={best['gamma']:<6} "
            f"I2T@1={best['metrics']['i2t']['R@1']:.2f} T2I@1={best['metrics']['t2i']['R@1']:.2f}{edge}"
        )

    print(json.dumps({"protocol": payload["protocol"], "image_range": payload["image_range"]}))


if __name__ == "__main__":
    main()
