from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from caption_geometry.datasets.records import ImageRecord


@dataclass(frozen=True)
class EmbeddingBundle:
    embeddings: torch.Tensor
    metadata: dict[str, Any]


def save_embedding_bundle(
    path: str | Path,
    embeddings: torch.Tensor,
    records: Sequence[ImageRecord],
    metadata: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "embeddings": embeddings.detach().cpu(),
        "records": [record.to_json() for record in records],
        "metadata": metadata,
    }
    torch.save(payload, destination)


def load_embedding_bundle(path: str | Path, map_location: str | torch.device = "cpu") -> EmbeddingBundle:
    payload = torch.load(Path(path), map_location=map_location)

    if torch.is_tensor(payload):
        return EmbeddingBundle(embeddings=payload, metadata={})

    embeddings = payload["embeddings"]
    metadata = dict(payload.get("metadata") or {})
    metadata["records"] = payload.get("records") or []
    return EmbeddingBundle(embeddings=embeddings, metadata=metadata)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def default_embedding_path(
    dataset: str,
    split: str,
    backbone: str,
    kind: str,
    output_dir: str | Path = "outputs/embeddings",
) -> Path:
    slug = backbone.replace("/", "-")
    return Path(output_dir) / f"{dataset}_{split}_{slug}_{kind}.pt"

