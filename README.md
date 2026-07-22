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

## Reviewer-Facing Experiments

Support-caption baselines:

```bash
conda run -n vlm env PYTHONPATH=src python -m caption_geometry.scripts.eval_support_baselines \
  --image-emb outputs/embeddings/coco_test5k_siglip-base-patch16-224_images.pt \
  --caption-emb outputs/embeddings/coco_test5k_siglip-base-patch16-224_captions.pt \
  --lambda-weight 0.4 \
  --gamma 1.8 \
  --leave-one-out \
  --query-batch-size 1024 \
  --bootstrap-samples 1000 \
  --output outputs/results/coco_test5k_siglip_support_baselines.json \
  --markdown-output outputs/results/coco_test5k_siglip_support_baselines.md
```

Support-caption mismatch robustness:

```bash
conda run -n vlm env PYTHONPATH=src python -m caption_geometry.scripts.eval_mismatch_robustness \
  --image-emb outputs/embeddings/coco_test5k_siglip-base-patch16-224_images.pt \
  --caption-emb outputs/embeddings/coco_test5k_siglip-base-patch16-224_captions.pt \
  --variant full \
  --alpha 1.0 \
  --beta 0.4 \
  --gamma 1.8 \
  --mismatch-rates 0 0.2 0.4 0.6 \
  --seeds 0 1 2 3 4 \
  --leave-one-out \
  --query-batch-size 1024 \
  --output outputs/results/coco_test5k_siglip_mismatch.json \
  --markdown-output outputs/results/coco_test5k_siglip_mismatch.md
```

Flickr30K Entities diagnostic:

```bash
conda run -n vlm env PYTHONPATH=src python -m caption_geometry.scripts.prepare_flickr30k_entities \
  --caption-emb outputs/embeddings/flickr30k_test1k_siglip-base-patch16-224_captions.pt \
  --sentences-dir data/flickr30k_entities/repo/Sentences \
  --output outputs/results/flickr30k_entities_ratios.jsonl

conda run -n vlm env PYTHONPATH=src python -m caption_geometry.scripts.diagnostics_entities \
  --image-emb outputs/embeddings/flickr30k_test1k_siglip-base-patch16-224_images.pt \
  --caption-emb outputs/embeddings/flickr30k_test1k_siglip-base-patch16-224_captions.pt \
  --entity-ratios-jsonl outputs/results/flickr30k_entities_ratios.jsonl \
  --output outputs/results/flickr30k_entities_diagnostics.json \
  --markdown-output outputs/results/flickr30k_entities_diagnostics.md
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
