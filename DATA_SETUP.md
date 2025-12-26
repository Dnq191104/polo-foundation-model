# Data and Artifacts Setup Guide

This repository contains only the **codebase**. All large data files, model weights, and artifacts are excluded from version control and must be recreated using the provided scripts.

## 🚨 Important Notes

- **Repository Size**: Only ~15MB (code only)
- **Large Files Excluded**: All `*.npy`, `*.pt`, `*.parquet` files are in `.gitignore`
- **Reproducibility**: All results can be reproduced using the scripts below

## 📊 Expected File Sizes (For Reference)

| Component | Size | Description |
|-----------|------|-------------|
| Raw dataset | ~2GB | Original fashion images + metadata |
| Processed dataset | ~1GB | HuggingFace dataset format |
| Catalog embeddings | ~1.8GB | `catalog_img.npy` + `catalog_txt.npy` |
| Training pairs | ~500MB | `pairs.parquet` |
| Model checkpoint | ~500MB | Single `.pt` file |
| **Total artifacts** | **~4GB** | All generated files combined |

## 🛠️ Step-by-Step Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download and Process Data

```bash
# Process the raw fashion dataset
python data/process_data.py

# This creates:
# - data/processed_v2/hf/ (HuggingFace dataset)
# - Attribute extraction and validation
```

### Step 3: Build Baseline Catalog (Step 6)

```bash
# Create embeddings for the full training set
python scripts/build_catalog_embeddings.py \
    --dataset data/processed_v2/hf \
    --output artifacts/retrieval/openclip_vitb32_v0 \
    --model ViT-B-32 \
    --pretrained openai

# This creates:
# - artifacts/retrieval/openclip_vitb32_v0/catalog_img.npy (~900MB)
# - artifacts/retrieval/openclip_vitb32_v0/catalog_txt.npy (~900MB)
# - artifacts/retrieval/openclip_vitb32_v0/catalog_meta.parquet (~50MB)
```

### Step 4: Evaluate Baseline Performance

```bash
# Run full evaluation on validation set
python scripts/eval_retrieval.py \
    --catalog_dir artifacts/retrieval/openclip_vitb32_v0 \
    --query_dataset data/processed_v2/hf \
    --query_split validation \
    --output_dir artifacts/retrieval/openclip_vitb32_v0/eval \
    --exclude_self \
    --save_per_query

# This creates evaluation reports and metrics
```

### Step 5: Prepare Step 7 Training Data

```bash
# 5a. Build product groups for strong positives
python scripts/step7_build_groups.py \
    --dataset data/processed_v2/hf \
    --split train \
    --output artifacts/step7/groups.json \
    --min_confidence 0.6

# 5b. Mine hard negatives from baseline
python scripts/step7_mine_hard_negatives.py \
    --baseline_catalog artifacts/retrieval/openclip_vitb32_v0 \
    --dataset data/processed_v2/hf \
    --output artifacts/step7/hard_negatives.json

# 5c. Build training pair dataset
python scripts/step7_build_pair_dataset.py \
    --dataset data/processed_v2/hf \
    --groups artifacts/step7/groups.json \
    --hard_negatives artifacts/step7/hard_negatives.json \
    --output artifacts/step7/pairs.parquet
```

### Step 6: Train Protocol Model (Step 7)

```bash
# Train the attribute-aware retrieval model
python scripts/step7_train_protocol.py \
    --pairs artifacts/step7/pairs.parquet \
    --dataset data/processed_v2/hf \
    --output artifacts/step7/runs/run_001 \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-4 \
    --seed 42

# This creates model checkpoints in artifacts/step7/runs/run_001/checkpoints/
```

### Step 7: Full Evaluation (Step 7)

```bash
# Evaluate the fine-tuned model
python scripts/step7_eval_full.py \
    --checkpoint artifacts/step7/runs/run_001/checkpoints/epoch_10.pt \
    --dataset data/processed_v2/hf \
    --baseline_metrics artifacts/retrieval/openclip_vitb32_v0/eval/metrics.json \
    --output artifacts/step7/runs/run_001/eval_full

# This creates the final evaluation report and scoreboard
```

## 🎯 Success Criteria (Step 7 Scoreboard)

The protocol model is considered successful if it achieves:

| Metric | Target | Current Baseline | Expected Improvement |
|--------|--------|------------------|---------------------|
| **Neckline@10** | +5 to +10 pts | 18.3% | → 23.3% to 28.3% |
| **Cardigans@10** | +10 pts | ~45% | → 55%+ |
| **Shorts@10** | +10 pts | ~38% | → 48%+ |
| **Denim@10** | +10 pts | ~42% | → 52%+ |
| **Leather@10** | +10 pts | ~38% | → 48%+ |

## 📈 Monitoring Training

```bash
# Check training progress
tail -f artifacts/step7/runs/run_001/train_log.jsonl

# Validate checkpoint during training
python scripts/step7_validate_checkpoint.py \
    --checkpoint artifacts/step7/runs/run_001/checkpoints/epoch_5.pt \
    --baseline_catalog artifacts/retrieval/openclip_vitb32_v0
```

## 🐛 Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce batch size in training scripts
2. **Slow Training**: Training on CPU - expect 30-60 min per epoch
3. **Unicode Errors**: Fixed in current codebase (Windows-compatible)
4. **Checkpoint Loading**: Ensure using linear projections (not MLP)

### Data Download:

If you need the original dataset, download from:
- [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- Or use the provided sample data for testing

## 📝 File Organization

```
artifacts/
├── retrieval/openclip_vitb32_v0/     # Step 6 baseline
│   ├── catalog_*.npy                # Embeddings (1.8GB)
│   ├── eval/                        # Baseline metrics
│   └── gallery.html                 # Visual results
└── step7/
    ├── groups.json                  # Product groups
    ├── pairs.parquet                # Training pairs (500MB)
    └── runs/run_001/
        ├── checkpoints/             # Model weights (500MB each)
        └── eval_full/               # Final evaluation

data/
├── processed_v2/hf/                 # HuggingFace dataset (1GB)
└── [other processed data]
```

## ⏱️ Time Estimates

| Step | Time | Description |
|------|------|-------------|
| Data processing | 10-15 min | Initial setup |
| Catalog building | 20-30 min | Create embeddings |
| Step 7 training | 5-8 hours | Full 10 epochs |
| Evaluation | 10-15 min | Per model checkpoint |

**Total time to reproduce all results: ~6-9 hours**

---

## 🎉 You're Done!

After completing all steps, you'll have:
- ✅ Working fashion retrieval system
- ✅ Step 7 fine-tuned model with improved attribute awareness
- ✅ Complete evaluation reports and scoreboards
- ✅ Visual galleries showing retrieval results
- ✅ Reproducible research codebase

See `STEP7_IMPLEMENTATION_SUMMARY.md` for detailed technical documentation.


