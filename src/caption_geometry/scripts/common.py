from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypeVar

import torch
from PIL import Image

from caption_geometry.datasets.records import ImageRecord
from caption_geometry.io import default_embedding_path, load_embedding_bundle
from caption_geometry.scoring.consensus_residual import normalize
from caption_geometry.scoring.retrieval_scores import default_support_source_indices

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
    support_captions, _ = sample_support_captions(captions, support_count, seed)
    return support_captions


def sample_support_captions(
    captions: torch.Tensor,
    support_count: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if support_count is None:
        return captions, default_support_source_indices(captions)

    n_images, k_captions, dim = captions.shape

    if not 1 <= support_count <= k_captions:
        raise ValueError(f"--support-count must be between 1 and {k_captions}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.stack(
        [torch.randperm(k_captions, generator=generator)[:support_count] for _ in range(n_images)]
    ).to(captions.device)
    gather_index = indices[:, :, None].expand(n_images, support_count, dim)
    return captions.gather(dim=1, index=gather_index), indices


def corrupt_support_captions(
    support_captions: torch.Tensor,
    support_source_indices: torch.Tensor,
    caption_pool: torch.Tensor,
    mismatch_rate: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not 0.0 <= mismatch_rate <= 1.0:
        raise ValueError("--mismatch-rate must be between 0 and 1")

    n_images, support_count, _ = support_captions.shape
    pool_images, pool_captions, _ = caption_pool.shape
    total_slots = n_images * support_count
    replace_count = round(total_slots * mismatch_rate)

    if replace_count == 0:
        return support_captions, support_source_indices, 0

    if pool_images < 2:
        raise ValueError("At least two images are required to create mismatched support captions")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    flat_slots = torch.randperm(total_slots, generator=generator)[:replace_count]
    target_images = flat_slots // support_count
    target_slots = flat_slots % support_count
    replacement_images = torch.randint(pool_images - 1, (replace_count,), generator=generator)
    replacement_images += (replacement_images >= target_images).long()
    replacement_captions = torch.randint(pool_captions, (replace_count,), generator=generator)

    device = support_captions.device
    target_images = target_images.to(device)
    target_slots = target_slots.to(device)
    replacement_images = replacement_images.to(device)
    replacement_captions = replacement_captions.to(device)

    corrupted = support_captions.clone()
    corrupted_sources = support_source_indices.clone().to(device=device, dtype=torch.long)
    corrupted[target_images, target_slots] = caption_pool[replacement_images, replacement_captions]
    corrupted_sources[target_images, target_slots] = -1
    return corrupted, corrupted_sources, replace_count
