from __future__ import annotations

import json
from pathlib import Path

from caption_geometry.datasets.records import ImageRecord, select_split, trim_captions


def load_flickr30k_karpathy(
    data_root: str | Path = "data",
    split: str = "test1k",
    caption_count: int | None = 5,
) -> list[ImageRecord]:
    """Load Flickr30K records from a Karpathy-style split file."""

    root = Path(data_root)
    split_path = root / "karpathy" / "dataset_flickr32k.json"

    with split_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = [
        _record_from_karpathy_image(root / "flickr30k" / "images", item)
        for item in payload["images"]
    ]
    selected = select_split(records, "test" if split.lower() == "test1k" else split)
    return trim_captions(selected, caption_count)


def _record_from_karpathy_image(image_root: Path, item: dict[str, object]) -> ImageRecord:
    filename = str(item["filename"])
    captions = tuple(_sentence_text(sentence) for sentence in item["sentences"])
    image_id = str(item.get("imgid") or Path(filename).stem)

    return ImageRecord(
        image_id=image_id,
        image_path=image_root / filename,
        captions=captions,
        split=str(item["split"]).lower(),
    )


def _sentence_text(sentence: dict[str, object]) -> str:
    raw = sentence.get("raw")

    if raw:
        return str(raw)

    tokens = sentence.get("tokens") or []
    return " ".join(str(token) for token in tokens)

