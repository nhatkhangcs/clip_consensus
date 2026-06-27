from __future__ import annotations

import argparse
from itertools import product

from tqdm import tqdm

from caption_geometry.io import write_json
from caption_geometry.scoring.consensus_residual import score_matrix_no_loo, score_matrix_with_loo
from caption_geometry.scoring.metrics import retrieval_metrics
from caption_geometry.scripts.common import (
    load_embedding_pair,
    resolve_compute_device,
    sample_caption_subset,
)

DEFAULT_BETAS = (0.0, 0.1, 0.2, 0.4, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0)
DEFAULT_GAMMAS = (0.0, 0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.6, 4.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid-search consensus/residual weights.")
    parser.add_argument("--image-emb")
    parser.add_argument("--caption-emb")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--backbone")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    parser.add_argument("--leave-one-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-batch-size", type=int)
    parser.add_argument("--support-count", type=int)
    parser.add_argument("--seed", type=int, default=0)
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
    captions = sample_caption_subset(captions, args.support_count, args.seed)
    scorer = score_matrix_with_loo if args.leave_one_out else score_matrix_no_loo
    rows = []

    for beta, gamma in tqdm(list(product(args.betas, args.gammas)), desc="Tuning weights"):
        scores = scorer(
            images=images,
            captions=captions,
            alpha=args.alpha,
            beta=beta,
            gamma=gamma,
            query_batch_size=args.query_batch_size,
        )
        metrics = retrieval_metrics(scores, k_captions=captions.shape[1])
        objective = metrics["i2t"]["R@1"] + metrics["t2i"]["R@1"]
        rows.append(
            {
                "alpha": args.alpha,
                "beta": beta,
                "gamma": gamma,
                "objective": objective,
                "metrics": metrics,
            }
        )

    best = max(rows, key=lambda row: row["objective"])
    payload = {
        "metric": "i2t_R@1_plus_t2i_R@1",
        "leave_one_out": args.leave_one_out,
        "support_count": captions.shape[1],
        "seed": args.seed,
        "best": best,
        "grid": rows,
        "metadata": metadata,
    }
    write_json(args.output, payload)
    print_tuning_result(payload)


def print_tuning_result(payload: dict[str, object]) -> None:
    import json

    print(json.dumps(payload["best"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
