# Caption Geometry

Training-free consensus-residual scoring for frozen CLIP-style image-text retrieval.

The method keeps the image-text encoder frozen, builds a support-caption consensus for each candidate image, decomposes the image embedding into a consensus-aligned component and an image-private residual, then combines:

```text
score = alpha * base_similarity
      + beta  * query_to_caption_consensus
      + gamma * query_to_image_residual
```

## Install

```bash
pip install -e ".[encoders,datasets,dev]"
```

## Data Layout

COCO Karpathy retrieval expects:

```text
data/
  coco/
    train2014/
    val2014/
    annotations/
  karpathy/
    dataset_coco.json
```

Flickr30K retrieval expects:

```text
data/
  flickr30k/
    images/
  karpathy/
    dataset_flickr32k.json
```

## First Milestone

Encode COCO test5k with SigLIP:

```bash
python -m caption_geometry.scripts.encode_images \
  --dataset coco \
  --split test5k \
  --backbone siglip-base-patch16-224 \
  --output outputs/embeddings/coco_test5k_siglip-base-patch16-224_images.pt

python -m caption_geometry.scripts.encode_captions \
  --dataset coco \
  --split test5k \
  --backbone siglip-base-patch16-224 \
  --output outputs/embeddings/coco_test5k_siglip-base-patch16-224_captions.pt
```

Evaluate base and full scoring:

```bash
python -m caption_geometry.scripts.eval_retrieval \
  --image-emb outputs/embeddings/coco_test5k_siglip-base-patch16-224_images.pt \
  --caption-emb outputs/embeddings/coco_test5k_siglip-base-patch16-224_captions.pt \
  --alpha 1.0 \
  --beta 0.4 \
  --gamma 1.8 \
  --leave-one-out \
  --output outputs/results/coco_test5k_siglip_full.json
```

Run all retrieval ablations:

```bash
python -m caption_geometry.scripts.eval_retrieval \
  --image-emb outputs/embeddings/coco_test5k_siglip-base-patch16-224_images.pt \
  --caption-emb outputs/embeddings/coco_test5k_siglip-base-patch16-224_captions.pt \
  --all-ablations \
  --leave-one-out \
  --output outputs/results/coco_test5k_siglip_ablations.json
```

Sweep a sampled caption count:

```bash
python -m caption_geometry.scripts.eval_retrieval \
  --image-emb outputs/embeddings/coco_test5k_siglip-base-patch16-224_images.pt \
  --caption-emb outputs/embeddings/coco_test5k_siglip-base-patch16-224_captions.pt \
  --support-count 3 \
  --seed 0
```

Tune beta/gamma on a held-out split:

```bash
python -m caption_geometry.scripts.tune_weights \
  --image-emb outputs/embeddings/coco_tune2500_siglip-base-patch16-224_images.pt \
  --caption-emb outputs/embeddings/coco_tune2500_siglip-base-patch16-224_captions.pt \
  --output outputs/results/tune_siglip_coco.json
```

## Generic Later-Stage Inputs

Compositional evaluation accepts JSONL items shaped like:

```json
{"image_index": 0, "captions": ["positive caption", "hard negative caption"], "gold_index": 0, "category": "replace_object"}
```

Entity diagnostics accepts preprocessed JSONL rows shaped like:

```json
{"image_index": 0, "caption_index": 3, "shared_entity_ratio": 0.6, "caption_specific_entity_ratio": 0.2}
```

## Validation

```bash
conda run -n vlm env PYTHONPATH=src python -m unittest discover -s tests
```
