from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from caption_geometry.io import load_embedding_bundle, write_json
from caption_geometry.scoring.consensus_residual import normalize
from caption_geometry.scoring.diagnostics import (
    caption_consensus_similarity,
    caption_residual_norm,
    residual_query_similarity,
)
from caption_geometry.scripts.table_utils import format_metric, markdown_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correlate caption geometry with preprocessed entity-ratio annotations."
    )
    parser.add_argument("--caption-emb", required=True)
    parser.add_argument("--image-emb")
    parser.add_argument("--entity-ratios-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    captions = normalize(load_embedding_bundle(args.caption_emb).embeddings.float())
    quantities = {
        "consensus_similarity": caption_consensus_similarity(captions),
        "caption_residual_norm": caption_residual_norm(captions),
    }

    if args.image_emb:
        images = normalize(load_embedding_bundle(args.image_emb).embeddings.float())
        quantities["residual_query_similarity"] = residual_query_similarity(images, captions)

    rows = read_entity_rows(args.entity_ratios_jsonl)
    results = {
        name: quantity_correlations(values, rows)
        for name, values in quantities.items()
    }
    table = diagnostics_table(results)
    payload = {
        "quantities": results,
        "total": len(rows),
        "markdown_table": table,
    }
    write_json(args.output, payload)
    write_markdown(args.markdown_output, table)
    print(table)
    print(json.dumps(payload, indent=2, sort_keys=True))


def read_entity_rows(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def correlations(pairs: list[tuple[float, float]]) -> dict[str, float]:
    left = torch.tensor([pair[0] for pair in pairs], dtype=torch.float64)
    right = torch.tensor([pair[1] for pair in pairs], dtype=torch.float64)
    return {
        "pearson_r": pearson_correlation(left, right),
        "spearman_r": pearson_correlation(average_ranks(left), average_ranks(right)),
    }


def quantity_correlations(values, rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    shared_pairs = []
    specific_pairs = []

    for row in rows:
        image_index = int(row["image_index"])
        caption_index = int(row["caption_index"])
        value = float(values[image_index, caption_index].item())
        shared_pairs.append((value, float(row["shared_entity_ratio"])))
        specific_pairs.append((value, float(row["caption_specific_entity_ratio"])))

    return {
        "shared_entities": correlations(shared_pairs),
        "caption_specific_entities": correlations(specific_pairs),
    }


def diagnostics_table(results: dict[str, dict[str, dict[str, float]]]) -> str:
    labels = {
        "consensus_similarity": "consensus similarity",
        "caption_residual_norm": "caption residual norm",
        "residual_query_similarity": "residual-query similarity",
    }
    rows = [
        [
            labels.get(name, name),
            format_metric(result["shared_entities"]["pearson_r"]),
            format_metric(result["caption_specific_entities"]["pearson_r"]),
        ]
        for name, result in results.items()
    ]
    return markdown_table(
        [
            "Quantity",
            "Correlation with shared entities",
            "Correlation with caption-specific entities",
        ],
        rows,
    )


def write_markdown(path: str | None, table: str) -> None:
    if not path:
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(table + "\n", encoding="utf-8")


def pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    denominator = centered_left.norm() * centered_right.norm()

    if denominator.item() == 0.0:
        return float("nan")

    return float((centered_left @ centered_right / denominator).item())


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    sorted_values, order = torch.sort(values)
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ranks = torch.empty_like(values)
    start = 0

    for count in counts.tolist():
        end = start + count
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end

    return ranks


if __name__ == "__main__":
    main()
