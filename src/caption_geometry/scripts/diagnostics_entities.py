from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

from caption_geometry.io import load_embedding_bundle, write_json
from caption_geometry.scoring.consensus_residual import normalize
from caption_geometry.scoring.diagnostics import (
    caption_consensus_similarity,
    caption_residual_norm,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correlate caption geometry with preprocessed entity-ratio annotations."
    )
    parser.add_argument("--caption-emb", required=True)
    parser.add_argument("--entity-ratios-jsonl", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    captions = normalize(load_embedding_bundle(args.caption_emb).embeddings.float())
    rho = caption_consensus_similarity(captions)
    eta = caption_residual_norm(captions)
    rows = read_entity_rows(args.entity_ratios_jsonl)

    shared_pairs = []
    specific_pairs = []

    for row in rows:
        image_index = int(row["image_index"])
        caption_index = int(row["caption_index"])
        shared_pairs.append(
            (
                float(rho[image_index, caption_index].item()),
                float(row["shared_entity_ratio"]),
            )
        )
        specific_pairs.append(
            (
                float(eta[image_index, caption_index].item()),
                float(row["caption_specific_entity_ratio"]),
            )
        )

    payload = {
        "rho_shared_entity_ratio": correlations(shared_pairs),
        "eta_caption_specific_entity_ratio": correlations(specific_pairs),
        "total": len(rows),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def read_entity_rows(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def correlations(pairs: list[tuple[float, float]]) -> dict[str, float]:
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    pearson = pearsonr(left, right)
    spearman = spearmanr(left, right)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


if __name__ == "__main__":
    main()

