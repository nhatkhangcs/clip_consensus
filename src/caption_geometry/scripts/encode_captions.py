from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from caption_geometry.datasets import load_records
from caption_geometry.io import default_embedding_path, save_embedding_bundle
from caption_geometry.models.encoders import build_encoder
from caption_geometry.scripts.common import batched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode captions with a frozen dual encoder.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--backend", choices=("transformers", "open_clip"))
    parser.add_argument("--model-id")
    parser.add_argument("--model-name")
    parser.add_argument("--pretrained")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--caption-count", type=int, default=5)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = load_records(
        dataset=args.dataset,
        split=args.split,
        data_root=args.data_root,
        caption_count=args.caption_count,
    )
    encoder = build_encoder(
        backbone=args.backbone,
        backend=args.backend,
        model_id=args.model_id,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )

    flat_captions = [
        caption
        for record in records
        for caption in record.captions
    ]
    chunks = [
        encoder.encode_texts(batch)
        for batch in tqdm(list(batched(flat_captions, args.batch_size)), desc="Encoding captions")
    ]
    embeddings = torch.cat(chunks, dim=0).reshape(len(records), args.caption_count, -1)
    output = Path(args.output) if args.output else default_embedding_path(
        args.dataset,
        args.split,
        args.backbone,
        "captions",
    )
    save_embedding_bundle(
        path=output,
        embeddings=embeddings,
        records=records,
        metadata={
            "dataset": args.dataset,
            "split": args.split,
            "backbone": args.backbone,
            "kind": "captions",
            "caption_count": args.caption_count,
        },
    )


if __name__ == "__main__":
    main()

