#!/usr/bin/env python
"""
Step 7 Grounding Audit

QA tool to verify that extracted attributes refer to the target garment (category2).
Samples items stratified by category and generates a CSV for manual review.

Usage:
    python scripts/step7_grounding_audit.py \
        --dataset data/processed_v2/hf \
        --split train \
        --output step7_grounding_audit.csv \
        --n_samples 200
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
from datasets import load_from_disk

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.attribute_validator import AttributeValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 7 grounding audit for attribute extraction"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to HuggingFace dataset with extracted attributes"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to sample from (default: train)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="step7_grounding_audit.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of samples to audit (default: 200)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--stratify_by",
        type=str,
        default="category2",
        help="Column to stratify by (default: category2)"
    )
    
    return parser.parse_args()


def load_dataset_split(dataset_dir: str, split: str):
    """Load dataset split."""
    print(f"Loading dataset from: {dataset_dir}")
    ds = load_from_disk(dataset_dir)
    
    if isinstance(ds, dict) or hasattr(ds, 'keys'):
        if split not in ds:
            available = list(ds.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")
        ds = ds[split]
    
    print(f"Loaded split '{split}': {len(ds)} items")
    return ds


def export_grounding_audit_csv(
    dataset,
    output_path: str,
    n_samples: int = 200,
    seed: int = 42,
    stratify_by: str = "category2"
):
    """
    Export grounding audit CSV with focus on primary garment matching.
    
    This CSV is tailored for reviewing whether extracted attributes
    refer to the target garment (category2).
    """
    validator = AttributeValidator()
    
    # Stratified sample
    sample = validator.sample_for_spotcheck(
        dataset,
        n=n_samples,
        seed=seed,
        stratify_by=stratify_by
    )
    
    # Build rows with grounding-specific questions
    rows = []
    for i in range(len(sample)):
        item = sample[i]
        
        row = {
            'index': i,
            'item_ID': item.get('item_ID', ''),
            'category2': item.get('category2', ''),
            'text': item.get('text', ''),
            
            # Extracted attributes
            'material_primary': item.get('attr_material_primary', 'unknown'),
            'pattern_primary': item.get('attr_pattern_primary', 'unknown'),
            'neckline_primary': item.get('attr_neckline_primary', 'unknown'),
            'sleeve_primary': item.get('attr_sleeve_primary', 'unknown'),
            
            # Grounding review questions
            'material_refers_to_category2': '',  # yes / no / n/a
            'pattern_refers_to_category2': '',
            'neckline_refers_to_category2': '',
            'sleeve_refers_to_category2': '',
            
            # Overall assessment
            'grounding_quality': '',  # good / partial / poor
            'contamination_source': '',  # e.g., "jacket" if dress text mentions jacket
            'notes': '',
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"\nExported {len(rows)} samples to: {output_path}")
    return output_path


def print_coverage_summary(dataset):
    """Print coverage summary by category."""
    validator = AttributeValidator()
    
    print("\n" + "=" * 60)
    print("COVERAGE SUMMARY")
    print("=" * 60)
    
    coverage = validator.compute_coverage_by_category(dataset, 'category2')
    
    # Sort by count
    if 'category2' in dataset.column_names:
        cat_counts = Counter(dataset['category2'])
        sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
        
        print(f"\n{'Category':<20} {'Count':>8} {'Material':>10} {'Pattern':>10} {'Neckline':>10} {'Sleeve':>10}")
        print("-" * 80)
        
        for category, count in sorted_cats[:15]:
            if category in coverage:
                stats = coverage[category]
                mat_cov = stats.get('material', {}).get('coverage_pct', 0)
                pat_cov = stats.get('pattern', {}).get('coverage_pct', 0)
                neck_cov = stats.get('neckline', {}).get('coverage_pct', 0)
                sleeve_cov = stats.get('sleeve', {}).get('coverage_pct', 0)
                
                print(
                    f"{category:<20} {count:>8} "
                    f"{mat_cov:>9.1f}% {pat_cov:>9.1f}% "
                    f"{neck_cov:>9.1f}% {sleeve_cov:>9.1f}%"
                )


def main():
    args = parse_args()
    
    print("=" * 60)
    print("STEP 7 GROUNDING AUDIT")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Load dataset
    ds = load_dataset_split(args.dataset, args.split)
    
    # Print coverage summary
    print_coverage_summary(ds)
    
    # Export audit CSV
    export_grounding_audit_csv(
        ds,
        args.output,
        n_samples=args.n_samples,
        seed=args.seed,
        stratify_by=args.stratify_by
    )
    
    print("\n" + "=" * 60)
    print("INSTRUCTIONS FOR REVIEWERS")
    print("=" * 60)
    print("\n1. Open the CSV in a spreadsheet editor")
    print("\n2. For each row, read the 'text' and check if extracted attributes")
    print("   refer to the garment specified in 'category2'")
    print("\n3. For each attribute (*_refers_to_category2 columns), mark:")
    print("   - 'yes' if the attribute clearly refers to the category2 garment")
    print("   - 'no' if it refers to a different garment mentioned in text")
    print("   - 'n/a' if attribute is 'unknown' or not applicable")
    print("\n4. Fill 'grounding_quality':")
    print("   - 'good': all non-unknown attributes refer to category2")
    print("   - 'partial': some attributes refer correctly, others don't")
    print("   - 'poor': most attributes refer to wrong garment")
    print("\n5. If contamination found, note the source garment in 'contamination_source'")
    print("   (e.g., if extracting from 'jacket' when category2 is 'dress')")
    print("\n6. Add any notes for schema improvements in 'notes' column")
    print("\n7. Return completed CSV for analysis")
    print("\n" + "=" * 60)
    print("\nAfter review, common fixes if grounding is poor:")
    print("  - Add category2 synonyms to 'garment_keywords' in attribute_schema.yaml")
    print("  - Add noise phrases to filter out secondary garments")
    print("  - Adjust sentence priority rules in AttributeExtractor")


if __name__ == "__main__":
    main()

