from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch

from caption_geometry.scoring.consensus_residual import normalize

BackendName = Literal["transformers", "open_clip"]


@dataclass(frozen=True)
class EncoderPreset:
    backend: BackendName
    model_id: str | None = None
    model_name: str | None = None
    pretrained: str | None = None


BACKBONE_PRESETS: dict[str, EncoderPreset] = {
    "siglip-base-patch16-224": EncoderPreset(
        backend="transformers",
        model_id="google/siglip-base-patch16-224",
    ),
    "openclip-vit-b-32": EncoderPreset(
        backend="open_clip",
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    ),
}


class DualEncoder(ABC):
    """Frozen dual-encoder interface returning normalized embeddings."""

    @abstractmethod
    def encode_images(self, pil_images: Sequence[Any]) -> torch.Tensor:
        """Encode a batch of PIL images as [B, D] float32 normalized tensors."""

    @abstractmethod
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode a batch of text strings as [B, D] float32 normalized tensors."""


class TransformersDualEncoder(DualEncoder):
    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "auto",
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = resolve_device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    @torch.inference_mode()
    def encode_images(self, pil_images: Sequence[Any]) -> torch.Tensor:
        inputs = self.processor(images=list(pil_images), return_tensors="pt")
        inputs = _move_to_device(inputs, self.device)
        features = self.model.get_image_features(**inputs)
        return normalize(features.float()).cpu()

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(
            text=list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = _move_to_device(inputs, self.device)
        features = self.model.get_text_features(**inputs)
        return normalize(features.float()).cpu()


class OpenClipDualEncoder(DualEncoder):
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str | torch.device = "auto",
    ) -> None:
        import open_clip

        self.device = resolve_device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.inference_mode()
    def encode_images(self, pil_images: Sequence[Any]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in pil_images]).to(self.device)
        features = self.model.encode_image(batch)
        return normalize(features.float()).cpu()

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(self.device)
        features = self.model.encode_text(tokens)
        return normalize(features.float()).cpu()


def build_encoder(
    backbone: str,
    backend: BackendName | None = None,
    model_id: str | None = None,
    model_name: str | None = None,
    pretrained: str | None = None,
    device: str | torch.device = "auto",
) -> DualEncoder:
    preset = BACKBONE_PRESETS.get(backbone)
    resolved_backend = backend or _preset_value(preset, "backend")

    if resolved_backend == "transformers":
        resolved_model_id = model_id or _preset_value(preset, "model_id")
        return TransformersDualEncoder(model_id=resolved_model_id, device=device)

    if resolved_backend == "open_clip":
        resolved_model_name = model_name or _preset_value(preset, "model_name")
        resolved_pretrained = pretrained or _preset_value(preset, "pretrained")
        return OpenClipDualEncoder(
            model_name=resolved_model_name,
            pretrained=resolved_pretrained,
            device=device,
        )

    raise ValueError(f"Unsupported encoder backend: {resolved_backend}")


def resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def _move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _preset_value(preset: EncoderPreset | None, field_name: str) -> str:
    if preset is None:
        raise ValueError(
            "Unknown backbone. Provide --backend with --model-id or --model-name/--pretrained."
        )

    value = getattr(preset, field_name)

    if value is None:
        raise ValueError(f"Backbone preset does not define {field_name}.")

    return value

