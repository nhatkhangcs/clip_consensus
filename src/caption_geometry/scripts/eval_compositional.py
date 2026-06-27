from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from caption_geometry.io import write_json
from caption_geometry.models.encoders import build_encoder
from caption_geometry.scoring.consensus_residual import normalize
from caption_geometry.scoring.pairwise import candidate_features, score_image_text_pairs
from caption_geometry.scripts.common import batched, load_embedding_pair, resolve_compute_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate compositional caption-choice items from JSONL."
    )
    parser.add_argument("--items-jsonl", required=True)
    parser.add_argument("--image-emb")
    parser.add_argument("--caption-emb")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--backend", choices=("transformers", "open_clip"))
    parser.add_argument("--model-id")
    parser.add_argument("--model-name")
    parser.add_argument("--pretrained")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=1.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_compute_device(args.device)
    images, support_captions, metadata = load_embedding_pair(
        image_path=args.image_emb,
        caption_path=args.caption_emb,
        dataset=args.dataset,
        split=args.split,
        backbone=args.backbone,
        device=device,
    )
    encoder = build_encoder(
        backbone=args.backbone,
        backend=args.backend,
        model_id=args.model_id,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )
    items = read_items(args.items_jsonl)
    flat_texts = [caption for item in items for caption in item["captions"]]
    text_embeddings = encode_texts(encoder, flat_texts, args.batch_size).to(device)
    text_embeddings = normalize(text_embeddings.float())
    features = candidate_features(images, support_captions)

    offset = 0
    correct = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    for item in items:
        captions = item["captions"]
        count = len(captions)
        candidate_embeddings = text_embeddings[offset : offset + count]
        image_indices = torch.full((count,), int(item["image_index"]), device=device)
        scores = score_image_text_pairs(
            text_embeddings=candidate_embeddings,
            image_indices=image_indices,
            images=features[0],
            consensus=features[1],
            residual=features[2],
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )
        prediction = int(scores.argmax().item())
        gold = int(item["gold_index"])
        is_correct = prediction == gold
        category = str(item.get("category") or "overall")

        correct += int(is_correct)
        by_category[category]["correct"] += int(is_correct)
        by_category[category]["total"] += 1
        offset += count

    payload = {
        "accuracy": correct / len(items) * 100.0,
        "total": len(items),
        "by_category": {
            category: {
                "accuracy": row["correct"] / row["total"] * 100.0,
                "total": row["total"],
            }
            for category, row in sorted(by_category.items())
        },
        "weights": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
        "metadata": metadata,
    }

    if args.output:
        write_json(args.output, payload)

    print(json.dumps(payload, indent=2, sort_keys=True))


def read_items(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_texts(encoder, texts: list[str], batch_size: int) -> torch.Tensor:
    chunks = [
        encoder.encode_texts(batch)
        for batch in tqdm(list(batched(texts, batch_size)), desc="Encoding candidates")
    ]
    return torch.cat(chunks, dim=0)


if __name__ == "__main__":
    main()

