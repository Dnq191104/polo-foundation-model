#!/usr/bin/env python3
"""
Analyze Poor-Performing Categories

Examines data quality and coverage for categories that performed poorly
in evaluation: shorts, rompers, graphic tees.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_poor_categories(dataset_path: str, split: str = "train"):
    """Analyze poor-performing categories in detail."""

    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)[split]

    # Categories that performed poorly
    poor_categories = ['shorts', 'rompers', 'graphic']

    print(f"\nAnalyzing {len(dataset)} items for poor categories: {poor_categories}")

    # Collect statistics
    category_counts = Counter()
    attribute_coverage = defaultdict(lambda: defaultdict(Counter))
    attribute_quality = defaultdict(lambda: defaultdict(list))

    for item in tqdm(dataset, desc="Analyzing items"):
        category = item.get('category2', '')

        if category in poor_categories:
            category_counts[category] += 1

            # Check attribute coverage and quality
            attributes = {
                'material': item.get('attr_material_primary', 'unknown'),
                'pattern': item.get('attr_pattern_primary', 'unknown'),
                'neckline': item.get('attr_neckline_primary', 'unknown'),
                'sleeve': item.get('attr_sleeve_primary', 'unknown')
            }

            for attr_name, attr_value in attributes.items():
                attribute_coverage[category][attr_name][attr_value] += 1

                # Track quality issues
                if attr_value == 'unknown':
                    attribute_quality[category][f"{attr_name}_missing"].append(item.get('item_ID', 'unknown'))

    # Print results
    print("\n" + "="*60)
    print("POOR CATEGORY ANALYSIS RESULTS")
    print("="*60)

    print("\n1. CATEGORY COUNTS:")
    print("-" * 30)
    for cat, count in sorted(category_counts.items()):
        print("12")

    print("\n2. ATTRIBUTE COVERAGE BY CATEGORY:")
    print("-" * 40)
    for category in poor_categories:
        if category in attribute_coverage:
            print(f"\n{category.upper()} ({category_counts.get(category, 0)} items):")

            for attr_name in ['material', 'pattern', 'neckline', 'sleeve']:
                if attr_name in attribute_coverage[category]:
                    coverage = attribute_coverage[category][attr_name]
                    total = sum(coverage.values())
                    unknown_count = coverage.get('unknown', 0)

                    print(f"  {attr_name}:")
                    print(f"    Coverage: {(total - unknown_count)/total*100:.1f}% ({total - unknown_count}/{total})")

                    # Show top values
                    top_values = coverage.most_common(3)
                    for value, count in top_values:
                        if value != 'unknown':
                            print(f"    - {value}: {count} ({count/total*100:.1f}%)")

    print("\n3. QUALITY CONCERNS:")
    print("-" * 25)
    for category in poor_categories:
        if category in attribute_quality:
            quality_issues = attribute_quality[category]
            total_items = category_counts[category]

            print(f"\n{category.upper()}:")
            for issue_type, items in quality_issues.items():
                issue_count = len(items)
                print(".1f")

    print("\n4. RECOMMENDATIONS:")
    print("-" * 20)

    # Analyze if categories are too rare
    for cat, count in category_counts.items():
        if count < 100:
            print(f"⚠️  {cat}: Only {count} samples - consider data augmentation or exclusion")
        elif count < 500:
            print(f"📊 {cat}: {count} samples - rare but trainable with oversampling")

    # Check for missing attributes
    for category in poor_categories:
        if category in attribute_coverage:
            for attr_name in ['neckline', 'pattern']:
                if attr_name in attribute_coverage[category]:
                    coverage = attribute_coverage[category][attr_name]
                    total = sum(coverage.values())
                    unknown_pct = coverage.get('unknown', 0) / total * 100

                    if unknown_pct > 50:
                        print(f"❌ {category} {attr_name}: {unknown_pct:.1f}% missing - poor data quality")

    print("\n5. TRAINING STRATEGIES:")
    print("-" * 25)
    print("• Increase oversampling weight for rare categories")
    print("• Consider category-specific data augmentation")
    print("• Review annotation quality for missing attributes")
    print("• Progressive fine-tuning: easy categories first")


def main():
    parser = argparse.ArgumentParser(description="Analyze poor-performing categories")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")

    args = parser.parse_args()
    analyze_poor_categories(args.dataset, args.split)


if __name__ == "__main__":
    main()

