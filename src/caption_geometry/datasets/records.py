from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ImageRecord:
    """Unified image item with all support/query captions."""

    image_id: str
    image_path: Path
    captions: tuple[str, ...]
    split: str

    def to_json(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "image_path": str(self.image_path),
            "captions": list(self.captions),
            "split": self.split,
        }


def load_records(
    dataset: str,
    split: str,
    data_root: str | Path = "data",
    caption_count: int | None = 5,
) -> list[ImageRecord]:
    """Load records from a supported dataset into the common format."""

    name = dataset.lower()

    if name == "coco":
        from caption_geometry.datasets.coco import load_coco_karpathy

        return load_coco_karpathy(data_root, split, caption_count)

    if name in {"flickr30k", "flickr"}:
        from caption_geometry.datasets.flickr30k import load_flickr30k_karpathy

        return load_flickr30k_karpathy(data_root, split, caption_count)

    raise ValueError(f"Unsupported dataset: {dataset}")


def ensure_caption_count(records: Sequence[ImageRecord], caption_count: int) -> None:
    missing = [
        record.image_id
        for record in records
        if len(record.captions) < caption_count
    ]

    if missing:
        examples = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} records have fewer than {caption_count} captions: {examples}"
        )


def trim_captions(
    records: Iterable[ImageRecord],
    caption_count: int | None,
) -> list[ImageRecord]:
    if caption_count is None:
        return list(records)

    materialized = list(records)
    ensure_caption_count(materialized, caption_count)
    return [
        ImageRecord(
            image_id=record.image_id,
            image_path=record.image_path,
            captions=tuple(record.captions[:caption_count]),
            split=record.split,
        )
        for record in materialized
    ]


def select_split(records: Sequence[ImageRecord], split: str) -> list[ImageRecord]:
    """Apply canonical retrieval splits over Karpathy split labels."""

    normalized = split.lower()

    if normalized == "test5k":
        return [record for record in records if record.split == "test"]

    if normalized == "tune2500":
        return [record for record in records if record.split == "test"][:2500]

    if normalized == "eval2500":
        return [record for record in records if record.split == "test"][2500:5000]

    return [record for record in records if record.split == normalized]

