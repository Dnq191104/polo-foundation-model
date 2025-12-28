# Gold Dev Eval Usage Guide

The Gold Dev Eval is a trusted, reusable evaluation set designed for rapid iteration during model development. It uses weighted stratified sampling to ensure good coverage of weak categories (shorts, rompers, cardigans) and supports manual verification for label quality.

## Overview

The Gold Dev Eval provides:
- **Weighted sampling**: 3x boost for weak categories to ensure adequate evaluation signal
- **Manual verification**: Tools to check and correct labels for maximum trust
- **Comprehensive metrics**: Category Recall@K, per-category breakdowns, and weak-category averages
- **Rapid iteration**: Fast evaluation for development workflow

## Quick Start

```bash
# 1. Create gold eval set (300 queries, 3x weight on weak categories)
python scripts/create_gold_dev_eval.py \
  --dataset data/processed_v2/hf \
  --output artifacts/step7/gold_dev_eval_ids.json \
  --n_queries 300

# 2. Export for manual verification
python scripts/verify_gold_dev_eval.py \
  --export \
  --gold_eval artifacts/step7/gold_dev_eval_ids.json \
  --dataset data/processed_v2/hf \
  --output_dir artifacts/step7/verification

# 3. [Manually verify labels in the exported CSV/HTML]

# 4. Import verified labels
python scripts/verify_gold_dev_eval.py \
  --import_csv artifacts/step7/verification/gold_eval_verification.csv \
  --gold_eval artifacts/step7/gold_dev_eval_ids.json \
  --overrides_output artifacts/step7/gold_dev_eval_overrides.json

# 5. Evaluate model on gold set
python scripts/eval_gold_dev.py \
  --checkpoint artifacts/step7/runs/run_001/checkpoints/best.pt \
  --gold_eval artifacts/step7/gold_dev_eval_ids.json \
  --overrides artifacts/step7/gold_dev_eval_overrides.json \
  --dataset data/processed_v2/hf \
  --output artifacts/step7/gold_dev_eval_metrics.json
```

## Detailed Workflow

### 1. Creating the Gold Eval Set

Use `create_gold_dev_eval.py` to generate a weighted stratified sample:

```bash
python scripts/create_gold_dev_eval.py \
  --dataset data/processed_v2/hf \
  --split validation \
  --output artifacts/step7/gold_dev_eval_ids.json \
  --n_queries 300 \
  --weak_boost 3.0 \
  --seed 42
```

**Parameters:**
- `--n_queries`: Target number of queries (200-500 recommended)
- `--weak_boost`: Weight multiplier for weak categories (default: 3.0)
- `--seed`: Random seed for reproducibility

**Weak Categories:** shorts, rompers, cardigans (excluding 'graphic')

The script outputs:
- `gold_dev_eval_ids.json`: Query indices with metadata
- Console summary of category distribution and weak category representation

### 2. Manual Verification

Export queries for manual review using `verify_gold_dev_eval.py`:

```bash
python scripts/verify_gold_dev_eval.py \
  --export \
  --gold_eval artifacts/step7/gold_dev_eval_ids.json \
  --dataset data/processed_v2/hf \
  --output_dir artifacts/step7/verification
```

This creates:
- `gold_eval_verification.csv`: Spreadsheet for editing labels
- `gold_eval_verification.html`: Visual gallery for review
- `VERIFICATION_INSTRUCTIONS.txt`: Detailed verification guidelines

**Verification Focus Areas:**
- Category accuracy (especially weak categories)
- Material correctness
- Pattern detection
- Neckline/sleeve attributes for upper body garments

**CSV Format:**
- `current_*` columns: Original labels from dataset
- `verified_*` columns: Corrected labels (fill these in)
- `verification_notes`: Any observations or issues

### 3. Importing Verified Labels

After manual verification, import the corrected labels:

```bash
python scripts/verify_gold_dev_eval.py \
  --import_csv artifacts/step7/verification/gold_eval_verification.csv \
  --gold_eval artifacts/step7/gold_dev_eval_ids.json \
  --output artifacts/step7/gold_dev_eval_verified.json
```

This creates an updated gold eval set with verified labels applied.

### 4. Running Gold Eval Metrics

Evaluate your model using `eval_gold_dev.py`:

```bash
python scripts/eval_gold_dev.py \
  --checkpoint artifacts/step7/runs/run_001/checkpoints/best.pt \
  --gold_eval artifacts/step7/gold_dev_eval_verified.json \
  --dataset data/processed_v2/hf \
  --output artifacts/step7/gold_dev_eval_metrics.json \
  --report artifacts/step7/gold_dev_eval_report.txt
```

**Metrics Computed:**
- **Overall Recall@K**: Category accuracy at K=1, 5, 10
- **Per-category metrics**: Recall@K for each category
- **Weak-category average**: Mean Recall@K over weak categories
- **Verification stats**: Number of verified queries, corrections applied

**Sample Output:**
```
OVERALL CATEGORY RECALL:
----------------------------------------
Category Recall@1: 0.156
Category Recall@5: 0.423
Category Recall@10: 0.589

WEAK CATEGORIES (shorts/rompers/cardigans):
----------------------------------------
shorts: Recall@1=0.089, @5=0.267, @10=0.378
rompers: Recall@1=0.122, @5=0.356, @10=0.489
cardigans: Recall@1=0.145, @5=0.401, @10=0.556

WEAK CATEGORY AVERAGES:
Weak Avg Recall@1: 0.119
Weak Avg Recall@5: 0.341
Weak Avg Recall@10: 0.474
```

### 5. Integration with Existing Workflow

The gold eval set is compatible with the existing dev evaluation pipeline:

```bash
# Use gold eval with existing step7_eval_dev.py
python scripts/step7_eval_dev.py \
  --checkpoint artifacts/step7/runs/run_001/checkpoints/best.pt \
  --dataset data/processed_v2/hf \
  --dev_ids artifacts/step7/gold_dev_eval_verified.json \
  --output artifacts/step7/dev_eval_results.json
```

## Files and Directory Structure

```
artifacts/step7/
├── gold_dev_eval_ids.json              # Immutable list of query IDs
├── gold_dev_eval_overrides.json        # ID -> corrections mapping
├── gold_dev_eval_metrics.json          # Latest evaluation metrics
├── gold_dev_eval_report.txt            # Human-readable report
└── verification/                       # Verification workspace
    ├── gold_eval_verification.csv      # CSV for manual editing
    ├── gold_eval_verification.html     # HTML gallery
    └── VERIFICATION_INSTRUCTIONS.txt   # Verification guidelines
```

## Best Practices

### Evaluation Strategy
- **Use verified labels**: Always prefer verified gold eval over unverified
- **Monitor weak categories**: Focus on weak-category average as key metric
- **Rapid iteration**: Use for development, full eval for final validation

### Verification Guidelines
- **Spend time on weak categories**: Shorts, rompers, cardigans need extra attention
- **Check attribute consistency**: Ensure material/pattern/attributes match visual content
- **Document issues**: Use verification_notes for systematic problems
- **Batch verification**: Verify in focused sessions rather than sporadically

### Maintenance
- **Re-verify periodically**: Labels may need updates as understanding improves
- **Version control**: Keep track of verification versions
- **Share corrections**: Apply learnings to broader dataset

## Troubleshooting

### Common Issues

**Low weak category performance:**
- Check if weak categories are properly represented in eval set
- Verify labels are correct for these categories
- Consider increasing weak_boost factor

**Inconsistent metrics:**
- Ensure using same gold eval set across experiments
- Check if verified labels are being applied correctly
- Verify dataset split consistency

**Verification workflow issues:**
- Make sure CSV is saved with correct encoding
- Check that query_idx column is preserved when editing
- Ensure verified columns contain actual corrections

### Getting Help

- Check VERIFICATION_INSTRUCTIONS.txt for detailed guidelines
- Examine existing verified examples for consistency
- Review metrics report for unusual patterns

## Integration with Step 7 Scoreboard

The gold eval metrics can be used alongside the Step 7 scoreboard:

```bash
# Run full Step 7 evaluation with gold eval metrics
python scripts/step7_eval_full.py \
  --checkpoint artifacts/step7/runs/run_001/checkpoints/best.pt \
  --dataset data/processed_v2/hf \
  --baseline_metrics artifacts/retrieval/openclip_vitb32_v0/eval/metrics.json \
  --output artifacts/step7/runs/run_001/eval_full

# Compare with gold eval results
python scripts/eval_gold_dev.py \
  --checkpoint artifacts/step7/runs/run_001/checkpoints/best.pt \
  --gold_eval artifacts/step7/gold_dev_eval_verified.json \
  --dataset data/processed_v2/hf
```

The gold eval provides fast, trusted signal for iteration, while full eval provides comprehensive validation.
