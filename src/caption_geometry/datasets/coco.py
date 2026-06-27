from __future__ import annotations

import json
from pathlib import Path

from caption_geometry.datasets.records import ImageRecord, select_split, trim_captions


def load_coco_karpathy(
    data_root: str | Path = "data",
    split: str = "test5k",
    caption_count: int | None = 5,
) -> list[ImageRecord]:
    """Load COCO records from the Karpathy split JSON."""

    root = Path(data_root)
    split_path = root / "karpathy" / "dataset_coco.json"

    with split_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = [_record_from_karpathy_image(root / "coco", item) for item in payload["images"]]
    selected = select_split(records, split)
    return trim_captions(selected, caption_count)


def _record_from_karpathy_image(coco_root: Path, item: dict[str, object]) -> ImageRecord:
    filepath = str(item.get("filepath") or "")
    filename = str(item["filename"])
    image_path = coco_root / filepath / filename
    captions = tuple(_sentence_text(sentence) for sentence in item["sentences"])
    image_id = str(item.get("cocoid") or item.get("imgid") or Path(filename).stem)

    return ImageRecord(
        image_id=image_id,
        image_path=image_path,
        captions=captions,
        split=str(item["split"]).lower(),
    )


def _sentence_text(sentence: dict[str, object]) -> str:
    raw = sentence.get("raw")

    if raw:
        return str(raw)

    tokens = sentence.get("tokens") or []
    return " ".join(str(token) for token in tokens)

