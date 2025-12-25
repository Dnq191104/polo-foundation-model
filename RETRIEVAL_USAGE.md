# Step 6: Baseline Retrieval System - Usage Guide

This guide explains how to use the baseline retrieval system for fashion product search.

## Overview

The baseline system uses:
- **Model**: OpenCLIP ViT-B/32 (CPU-friendly)
- **Approach**: Two-stage retrieval with image+text fusion
- **Fusion**: Weighted sum (default: 0.7 image, 0.3 text)
- **Stage 1**: Image-only candidate generation (top-200)
- **Stage 2**: Multimodal re-ranking (final top-10)

## Quick Start

### 1. Build Catalog Embeddings (Offline)

First, encode your catalog items (train split):

```bash
python scripts/build_catalog_embeddings.py \
    --dataset_dir data/processed_v2/hf \
    --split train \
    --output artifacts/retrieval/openclip_vitb32_v0 \
    --batch_size 32
```

**Options:**
- `--model_name`: OpenCLIP model (default: `ViT-B-32`)
- `--pretrained`: Weights version (default: `openai`)
- `--device`: Device to use (`cpu`, `cuda`, or auto-detect)
- `--batch_size`: Batch size for image encoding (default: 32)
- `--text_batch_size`: Batch size for text encoding (default: 64)

**Output:**
- `catalog_img.npy`: Image embeddings (normalized)
- `catalog_txt.npy`: Text embeddings (normalized)
- `catalog_meta.parquet`: Metadata (item_ID, category2, attributes)
- `manifest.json`: Configuration and checksums

**Expected time:** ~5-15 minutes for 40k items on CPU

### 2. Evaluate Retrieval Performance

Run evaluation on validation queries:

```bash
python scripts/eval_retrieval.py \
    --catalog_dir artifacts/retrieval/openclip_vitb32_v0 \
    --query_dataset data/processed_v2/hf \
    --query_split validation \
    --output metrics_report.json \
    --output_dir artifacts/retrieval/openclip_vitb32_v0/eval \
    --top_k 10 \
    --exclude_self
```

**Key Options:**
- `--top_k`: Number of results to evaluate (default: 10)
- `--weight_image`: Image weight (default: 0.7)
- `--candidate_n`: Stage 1 candidates (default: 200)
- `--exclude_self`: Exclude query item from results (important for validation!)
- `--save_per_query`: Save detailed per-query results to parquet
- `--max_queries`: Limit number of queries (for quick tests)

**Output:**
- Console report with key metrics
- `metrics_report.json`: Full metrics JSON
- `artifacts/.../eval/metrics.json`: Overall metrics
- `artifacts/.../eval/metrics_by_category.json`: Category breakdown
- `artifacts/.../eval/metrics_by_material.json`: Material breakdown
- `per_query_results.parquet`: Per-query details (if `--save_per_query`)

**Metrics Computed:**
- `category_match@10`: % of top-10 with same category
- `material_match@10` (known only): % material match (skips "unknown")
- `pattern_match@10`: % pattern match
- `neckline_match@10`, `sleeve_match@10`: Attribute matches
- Sliced by category2 and material

### 3. Build Qualitative Gallery

Generate HTML gallery for visual inspection:

```bash
python scripts/build_gallery.py \
    --catalog_dir artifacts/retrieval/openclip_vitb32_v0 \
    --query_dataset data/processed_v2/hf \
    --query_split validation \
    --output gallery.html \
    --n_samples 100 \
    --stratify_by category2
```

**Options:**
- `--n_samples`: Number of query samples (default: 100)
- `--stratify_by`: Column to stratify sampling (default: `category2`)
- `--top_k`: Results per query (default: 10)
- `--seed`: Random seed for sampling (default: 42)

**Output:**
- `gallery.html`: Self-contained HTML file with embedded images
- Open in any web browser for review
- Green highlight: Category match
- Blue highlight: Material match

**Features:**
- Organized by category
- Shows query image + attributes + text
- Shows top-10 with scores (fused, image, text)
- Visual indicators for matches

## Understanding the Metrics

### Category Match@10
Percentage of top-10 results that share the same `category2` as the query.

**Target:** ≥60% for major categories (tees, blouses, dresses)

### Material Match@10 (Known Only)
Percentage of top-10 results with matching `attr_material_primary`.

**Important:** Computed only on queries where material is not "unknown" (avoids contaminating metrics with unpredictable cases).

**Target:** ≥40% (material is harder than category)

### Gallery Review
Qualitative check on 50-100 sampled queries.

**Success criteria:**
- Top-10 "looks reasonable" for ≥70% of queries
- Material and design are similar to query
- No systematic category confusions (e.g., dresses → skirts)

## Typical Workflow

```bash
# 1. Build catalog (once)
python scripts/build_catalog_embeddings.py \
    --dataset_dir data/processed_v2/hf \
    --split train \
    --output artifacts/retrieval/openclip_vitb32_v0

# 2. Run evaluation
python scripts/eval_retrieval.py \
    --catalog_dir artifacts/retrieval/openclip_vitb32_v0 \
    --query_dataset data/processed_v2/hf \
    --query_split validation \
    --output_dir artifacts/retrieval/openclip_vitb32_v0/eval \
    --exclude_self \
    --save_per_query

# 3. Build gallery for review
python scripts/build_gallery.py \
    --catalog_dir artifacts/retrieval/openclip_vitb32_v0 \
    --query_dataset data/processed_v2/hf \
    --query_split validation \
    --output artifacts/retrieval/openclip_vitb32_v0/gallery.html \
    --n_samples 100

# 4. Open gallery in browser
# (double-click gallery.html or open with browser)
```

## Troubleshooting

### Out of Memory (CPU)
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--text_batch_size`

### Slow Encoding
- Expected: ~0.5-1s per item on CPU
- Consider using GPU if available (`--device cuda`)
- Embeddings are cached, only need to build once

### Low Metrics
- Check gallery for failure modes
- Try adjusting `--weight_image` (e.g., 0.8 for more image weight)
- Verify attribute extraction quality (Step 5)

### Gallery Too Large
- Reduce `--n_samples` (try 50)
- File size ~5-20MB for 100 queries with images

## Next Steps

After establishing baseline:

1. **Document baseline metrics** for comparison
2. **Identify failure modes** from gallery
3. **Proceed to Step 7**: Train domain-specific embeddings
4. **Compare** fine-tuned model vs baseline

## Files Created

```
artifacts/retrieval/openclip_vitb32_v0/
├── catalog_img.npy              # Image embeddings
├── catalog_txt.npy              # Text embeddings
├── catalog_meta.parquet         # Metadata
├── manifest.json                # Configuration
├── eval/
│   ├── metrics.json            # Overall metrics
│   ├── metrics_by_category.json
│   ├── metrics_by_material.json
│   └── per_query_results.parquet
└── gallery.html                 # Visual gallery
```

## Integration with Python Code

```python
from src.retrieval.embedder import OpenCLIPEmbedder
from src.retrieval.engine import RetrievalEngine

# Initialize
embedder = OpenCLIPEmbedder(model_name="ViT-B-32", pretrained="openai")
engine = RetrievalEngine("artifacts/retrieval/openclip_vitb32_v0")

# Encode query
query_img = Image.open("query.jpg")
query_text = "cotton graphic tee"
img_vec = embedder.encode_image(query_img)
txt_vec = embedder.encode_text(query_text)

# Search
results = engine.search(
    img_vec,
    txt_vec,
    top_k=10,
    weight_image=0.7
)

# Results contain:
for result in results:
    print(f"Rank {result['rank']}: {result['metadata']['item_ID']}")
    print(f"  Score: {result['score']:.3f}")
    print(f"  Category: {result['metadata']['category2']}")
```

## Performance Targets (Baseline Checkpoint)

✓ **Speed**: <500ms per query on CPU (exact search over 40k)  
✓ **Category@10**: ≥60% for major categories  
✓ **Material@10** (known): ≥40%  
✓ **Gallery**: ≥70% "looks reasonable"  

These targets establish a **reference point** for Step 7 (fine-tuning).

