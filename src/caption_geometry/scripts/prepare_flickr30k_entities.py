from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from caption_geometry.io import load_embedding_bundle

ENTITY_PATTERN = re.compile(r"/EN#([^/\]\s]+)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Flickr30K Entities shared/specific ratios from sentence annotations."
    )
    parser.add_argument("--caption-emb", required=True)
    parser.add_argument("--sentences-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = load_embedding_bundle(args.caption_emb).metadata.get("records") or []
    sentence_dir = Path(args.sentences_dir)
    rows = [
        row
        for image_index, record in enumerate(records)
        for row in image_rows(image_index, record, sentence_dir)
    ]
    write_jsonl(args.output, rows)


def image_rows(image_index: int, record: dict[str, object], sentence_dir: Path) -> list[dict[str, object]]:
    image_id = str(record.get("image_id") or Path(str(record["image_path"])).stem)
    sentence_path = resolve_sentence_path(record, sentence_dir)
    lines = sentence_lines(sentence_path)
    entity_sets = [extract_entities(line) for line in lines]
    rows = []

    for caption_index, entities in enumerate(entity_sets):
        others = set().union(
            *[other for idx, other in enumerate(entity_sets) if idx != caption_index]
        )
        shared = entities & others
        specific = entities - others
        denominator = max(len(entities), 1)
        rows.append(
            {
                "image_index": image_index,
                "caption_index": caption_index,
                "image_id": image_id,
                "entity_count": len(entities),
                "shared_entity_ratio": len(shared) / denominator,
                "caption_specific_entity_ratio": len(specific) / denominator,
            }
        )

    return rows


def resolve_sentence_path(record: dict[str, object], sentence_dir: Path) -> Path:
    candidates = [
        str(record.get("image_id") or ""),
        Path(str(record.get("image_path") or "")).stem,
    ]

    for candidate in candidates:
        path = sentence_dir / f"{candidate}.txt"

        if candidate and path.exists():
            return path

    tried = ", ".join(str(sentence_dir / f"{candidate}.txt") for candidate in candidates if candidate)
    raise FileNotFoundError(f"Could not find Flickr30K Entities sentence file. Tried: {tried}")


def sentence_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip()]


def extract_entities(sentence: str) -> set[str]:
    return set(ENTITY_PATTERN.findall(sentence))


def write_jsonl(path: str | Path, rows: list[dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
