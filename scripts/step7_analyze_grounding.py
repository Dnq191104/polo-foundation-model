#!/usr/bin/env python
"""
Analyze Step 7 Grounding Audit Results

Analyzes completed grounding audit CSV and provides recommendations.

Usage:
    python scripts/step7_analyze_grounding.py \
        --input step7_grounding_audit_completed.csv
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze grounding audit results"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to completed grounding audit CSV"
    )
    
    return parser.parse_args()


def analyze_grounding_results(csv_path: str):
    """Analyze grounding audit CSV and print report."""
    df = pd.read_csv(csv_path)
    
    print("=" * 60)
    print("GROUNDING AUDIT ANALYSIS")
    print("=" * 60)
    print(f"Total samples reviewed: {len(df)}")
    print()
    
    # Overall quality distribution
    if 'grounding_quality' in df.columns:
        quality_counts = df['grounding_quality'].value_counts()
        print("OVERALL GROUNDING QUALITY:")
        print("-" * 40)
        for quality, count in quality_counts.items():
            if quality and str(quality) != 'nan':
                pct = count / len(df) * 100
                print(f"  {quality:<10} {count:>4} ({pct:>5.1f}%)")
        
        # Success rate
        good_count = quality_counts.get('good', 0)
        partial_count = quality_counts.get('partial', 0)
        usable_rate = (good_count + partial_count) / len(df) * 100
        
        print()
        print(f"Usable rate (good + partial): {usable_rate:.1f}%")
        
        if usable_rate >= 80:
            print("✓ PASSED: Grounding quality meets 80% threshold")
        else:
            print("✗ NEEDS IMPROVEMENT: Grounding quality below 80%")
        print()
    
    # Per-attribute grounding
    print("PER-ATTRIBUTE GROUNDING:")
    print("-" * 40)
    
    attr_cols = [
        'material_refers_to_category2',
        'pattern_refers_to_category2',
        'neckline_refers_to_category2',
        'sleeve_refers_to_category2'
    ]
    
    for col in attr_cols:
        if col in df.columns:
            attr_name = col.replace('_refers_to_category2', '')
            yes_count = (df[col] == 'yes').sum()
            no_count = (df[col] == 'no').sum()
            total_applicable = yes_count + no_count  # Excluding n/a
            
            if total_applicable > 0:
                correct_pct = yes_count / total_applicable * 100
                print(f"  {attr_name:<12} {correct_pct:>5.1f}% correct ({yes_count}/{total_applicable})")
    
    print()
    
    # Contamination sources
    if 'contamination_source' in df.columns:
        contamination = df['contamination_source'].dropna()
        contamination = contamination[contamination != '']
        
        if len(contamination) > 0:
            print("CONTAMINATION SOURCES (top 10):")
            print("-" * 40)
            
            sources = Counter(contamination)
            for source, count in sources.most_common(10):
                print(f"  {source:<20} {count:>4}")
            print()
    
    # By category breakdown
    if 'category2' in df.columns and 'grounding_quality' in df.columns:
        print("BY CATEGORY:")
        print("-" * 40)
        
        cat_quality = df.groupby('category2')['grounding_quality'].value_counts()
        cat_totals = df['category2'].value_counts()
        
        for category in cat_totals.index[:10]:
            total = cat_totals[category]
            good = cat_quality.get((category, 'good'), 0)
            partial = cat_quality.get((category, 'partial'), 0)
            usable = (good + partial) / total * 100 if total > 0 else 0
            
            print(f"  {category:<20} {usable:>5.1f}% usable (n={total})")
        print()
    
    # Recommendations
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    # Check if any attribute has low correctness
    low_attrs = []
    for col in attr_cols:
        if col in df.columns:
            attr_name = col.replace('_refers_to_category2', '')
            yes_count = (df[col] == 'yes').sum()
            no_count = (df[col] == 'no').sum()
            total_applicable = yes_count + no_count
            
            if total_applicable > 0:
                correct_pct = yes_count / total_applicable * 100
                if correct_pct < 80:
                    low_attrs.append((attr_name, correct_pct))
    
    if low_attrs:
        print("\n⚠ Attributes with low grounding correctness:")
        for attr_name, correct_pct in low_attrs:
            print(f"  - {attr_name}: {correct_pct:.1f}%")
        print("\nAction: Review and strengthen category2 anchoring for these attributes")
        print("  1. Add/refine garment synonyms in attribute_schema.yaml")
        print("  2. Review sentence priority rules in AttributeExtractor")
        print("  3. Add noise phrases for common contamination sources")
    else:
        print("\n✓ All attributes have good grounding correctness (>80%)")
    
    # Check contamination
    if 'contamination_source' in df.columns:
        contamination = df['contamination_source'].dropna()
        contamination = contamination[contamination != '']
        
        if len(contamination) > 10:
            print(f"\n⚠ Contamination detected in {len(contamination)} samples")
            sources = Counter(contamination)
            top_sources = [s for s, _ in sources.most_common(5)]
            print(f"  Top sources: {', '.join(top_sources)}")
            print("\nAction: Add noise filters for these garment types")
        else:
            print("\n✓ Minimal contamination detected")
    
    print("\n" + "=" * 60)


def main():
    args = parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    analyze_grounding_results(args.input)


if __name__ == "__main__":
    main()

