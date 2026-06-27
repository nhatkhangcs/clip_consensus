from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypeVar

import torch
from PIL import Image

from caption_geometry.datasets.records import ImageRecord
from caption_geometry.io import default_embedding_path, load_embedding_bundle
from caption_geometry.scoring.consensus_residual import normalize

T = TypeVar("T")


def batched(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_images(records: Sequence[ImageRecord]) -> list[Image.Image]:
    images = []

    for record in records:
        with Image.open(record.image_path) as image:
            images.append(image.convert("RGB"))

    return images


def resolve_embedding_path(
    explicit_path: str | None,
    dataset: str | None,
    split: str | None,
    backbone: str | None,
    kind: str,
) -> Path:
    if explicit_path:
        return Path(explicit_path)

    missing = [
        name
        for name, value in {
            "dataset": dataset,
            "split": split,
            "backbone": backbone,
        }.items()
        if value is None
    ]

    if missing:
        joined = ", ".join(missing)
        flag = "--image-emb" if kind == "images" else "--caption-emb"
        raise ValueError(f"Provide {flag} or set these fields: {joined}")

    return default_embedding_path(dataset=dataset, split=split, backbone=backbone, kind=kind)


def load_embedding_pair(
    image_path: str | None,
    caption_path: str | None,
    dataset: str | None,
    split: str | None,
    backbone: str | None,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    resolved_image = resolve_embedding_path(image_path, dataset, split, backbone, "images")
    resolved_caption = resolve_embedding_path(caption_path, dataset, split, backbone, "captions")

    image_bundle = load_embedding_bundle(resolved_image, map_location=device)
    caption_bundle = load_embedding_bundle(resolved_caption, map_location=device)
    images = normalize(image_bundle.embeddings.float())
    captions = normalize(caption_bundle.embeddings.float())

    metadata = {
        "image_embedding_path": str(resolved_image),
        "caption_embedding_path": str(resolved_caption),
        "image_metadata": image_bundle.metadata,
        "caption_metadata": caption_bundle.metadata,
    }
    return images, captions, metadata


def resolve_compute_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def sample_caption_subset(
    captions: torch.Tensor,
    support_count: int | None,
    seed: int,
) -> torch.Tensor:
    if support_count is None:
        return captions

    n_images, k_captions, dim = captions.shape

    if not 1 <= support_count <= k_captions:
        raise ValueError(f"--support-count must be between 1 and {k_captions}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.stack(
        [torch.randperm(k_captions, generator=generator)[:support_count] for _ in range(n_images)]
    ).to(captions.device)
    gather_index = indices[:, :, None].expand(n_images, support_count, dim)
    return captions.gather(dim=1, index=gather_index)
