#!/usr/bin/env python
"""
Attribute Spot-Check Utility

Script for sampling dataset items and exporting them for human review
of attribute extraction quality.

Usage:
    python scripts/spotcheck_attributes.py --dataset data/processed/hf --output spotcheck.csv --n 100
    python scripts/spotcheck_attributes.py --analyze spotcheck_completed.csv
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_from_disk, Dataset

from src.datasets.attribute_validator import AttributeValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spot-check attribute extraction quality"
    )
    
    # Mode selection
    parser.add_argument(
        "--analyze",
        type=str,
        default=None,
        help="Path to completed spot-check CSV to analyze (instead of sampling)"
    )
    
    # Sampling options
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/hf",
        help="Path to HuggingFace dataset with extracted attributes"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to sample from"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="spotcheck_sample.csv",
        help="Output CSV path for spot-check sample"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of samples to draw"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--stratify",
        type=str,
        default="category2",
        help="Column to stratify samples by (or 'none')"
    )
    
    # Report options
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate coverage report before sampling"
    )
    parser.add_argument(
        "--by-category",
        action="store_true",
        help="Include category breakdown in report"
    )
    
    # Display options  
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of samples to preview in terminal"
    )
    
    return parser.parse_args()


def load_dataset_with_attrs(dataset_path: str, split: str) -> Dataset:
    """Load dataset and verify it has attribute columns."""
    print(f"Loading dataset from: {dataset_path}")
    
    ds = load_from_disk(dataset_path)
    
    if isinstance(ds, dict) or hasattr(ds, 'keys'):
        if split not in ds:
            available = list(ds.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")
        ds = ds[split]
    
    print(f"Loaded {len(ds)} samples")
    
    # Check for attribute columns
    attr_cols = [c for c in ds.column_names if c.startswith('attr_')]
    if not attr_cols:
        print("\nWARNING: No attribute columns found!")
        print("Run with --extract_attributes flag in process_data.py first.")
        print(f"Available columns: {ds.column_names}")
    else:
        print(f"Found attribute columns: {attr_cols}")
    
    return ds


def generate_report(ds: Dataset, by_category: bool = False) -> None:
    """Generate and print coverage report."""
    validator = AttributeValidator()
    report = validator.generate_coverage_report(ds, by_category=by_category)
    print(report)


def export_sample(
    ds: Dataset,
    output_path: str,
    n: int,
    seed: int,
    stratify: str
) -> None:
    """Export sample for spot-checking."""
    validator = AttributeValidator()
    
    stratify_by = stratify if stratify != 'none' else None
    
    validator.export_spotcheck_csv(
        ds,
        output_path,
        n=n,
        seed=seed
    )
    
    print(f"\nExported {n} samples to: {output_path}")
    print("\nInstructions for reviewers:")
    print("-" * 40)
    print("1. Open the CSV in a spreadsheet editor")
    print("2. For each row, review the 'text' and extracted attributes")
    print("3. Fill in 'review_status': correct / partial / wrong")
    print("4. Fill in 'error_type' for wrong/partial items:")
    print("   - missed_tag: attribute present in text but not extracted")
    print("   - wrong_tag: extracted tag doesn't match text")
    print("   - conflict_rule: multiple tags where one should win")
    print("5. Add any notes in the 'notes' column")
    print("6. Save and run --analyze on the completed file")


def preview_samples(ds: Dataset, n: int = 5) -> None:
    """Preview samples in terminal."""
    validator = AttributeValidator()
    sample = validator.sample_for_spotcheck(ds, n=n)
    
    print(f"\n{'=' * 60}")
    print(f"PREVIEW OF {n} SAMPLES")
    print('=' * 60)
    
    for i in range(len(sample)):
        item = sample[i]
        formatted = validator.format_spotcheck_item(item)
        print(formatted)
        print()


def analyze_results(csv_path: str) -> None:
    """Analyze completed spot-check results."""
    validator = AttributeValidator()
    
    print(f"Analyzing spot-check results from: {csv_path}")
    print("=" * 60)
    
    results = validator.analyze_spotcheck_results(csv_path)
    
    print(f"\nTotal reviewed: {results['total_reviewed']}")
    
    print("\nReview Status Distribution:")
    print("-" * 40)
    for status, count in results.get('status_counts', {}).items():
        pct = count / results['total_reviewed'] * 100 if results['total_reviewed'] > 0 else 0
        print(f"  {status}: {count} ({pct:.1f}%)")
    
    if 'usable_rate' in results:
        print(f"\nUsable Rate (correct + partial): {results['usable_rate']:.1f}%")
        
        # Success threshold check
        if results['usable_rate'] >= 80:
            print("✓ PASSED: Usable rate meets 80% threshold")
        else:
            print("✗ NEEDS IMPROVEMENT: Usable rate below 80%")
    
    if results.get('error_types'):
        print("\nError Types:")
        print("-" * 40)
        for error_type, count in results['error_types'].items():
            if error_type and str(error_type) != 'nan':
                print(f"  {error_type}: {count}")
    
    print("\nBy Attribute:")
    print("-" * 40)
    for attr_name, stats in results.get('by_attribute', {}).items():
        unknown_pct = stats.get('unknown_pct', 0)
        print(f"  {attr_name}: {100 - unknown_pct:.1f}% coverage")
    
    print("\nRecommendations:")
    print("-" * 40)
    
    error_types = results.get('error_types', {})
    missed = error_types.get('missed_tag', 0)
    wrong = error_types.get('wrong_tag', 0)
    conflict = error_types.get('conflict_rule', 0)
    
    if missed > wrong:
        print("- Consider expanding trigger phrases in attribute_schema.yaml")
    if wrong > missed:
        print("- Consider adding exclusion rules or narrowing trigger phrases")
    if conflict > 0:
        print("- Review conflict resolution rules for priority settings")


def main():
    args = parse_args()
    
    # Analyze mode
    if args.analyze:
        analyze_results(args.analyze)
        return
    
    # Load dataset
    ds = load_dataset_with_attrs(args.dataset, args.split)
    
    # Generate report if requested
    if args.report:
        generate_report(ds, by_category=args.by_category)
        print()
    
    # Preview samples
    if args.preview > 0:
        preview_samples(ds, n=args.preview)
    
    # Export sample
    export_sample(
        ds,
        args.output,
        n=args.n,
        seed=args.seed,
        stratify=args.stratify
    )


if __name__ == "__main__":
    main()

