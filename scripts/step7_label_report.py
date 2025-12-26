#!/usr/bin/env python
"""
Step 7 Label Coverage Report

Analyzes dataset for label coverage and quality issues.
Checks coverage per attribute, rare/weak classes, and potential training poisons.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_attribute_coverage(dataset, attribute_columns: list, output_path: Path):
    """
    Analyze coverage statistics for each attribute.
    """
    print("Analyzing attribute coverage...")

    total_samples = len(dataset)
    coverage_stats = {}

    for attr_col in attribute_columns:
        if attr_col not in dataset.column_names:
            print(f"Warning: {attr_col} not found in dataset")
            continue

        values = []
        null_count = 0
        unknown_count = 0

        for item in tqdm(dataset, desc=f"Processing {attr_col}"):
            val = item.get(attr_col)
            if val is None or val == "" or str(val).lower() in ['null', 'none']:
                null_count += 1
                values.append(None)
            elif str(val).lower() == 'unknown':
                unknown_count += 1
                values.append('unknown')
            else:
                values.append(val)

        # Calculate coverage
        present_count = total_samples - null_count - unknown_count
        coverage_pct = (present_count / total_samples) * 100
        unknown_pct = (unknown_count / total_samples) * 100
        null_pct = (null_count / total_samples) * 100

        # Value distribution (excluding null/unknown)
        valid_values = [v for v in values if v is not None and v != 'unknown']
        value_counts = Counter(valid_values)
        top_10_values = value_counts.most_common(10)

        coverage_stats[attr_col] = {
            'total_samples': total_samples,
            'present_count': present_count,
            'present_pct': coverage_pct,
            'unknown_count': unknown_count,
            'unknown_pct': unknown_pct,
            'null_count': null_count,
            'null_pct': null_pct,
            'unique_values': len(set(valid_values)),
            'top_10_values': top_10_values
        }

        print(".1f"
              ".1f"
              f"unique: {len(set(valid_values))}")

    # Save detailed stats
    with open(output_path / "attribute_coverage.json", 'w') as f:
        json.dump(coverage_stats, f, indent=2, default=str)

    return coverage_stats


def analyze_weak_classes_and_rare_materials(dataset, output_path: Path):
    """
    Analyze specific weak classes and rare materials we're targeting.
    """
    print("Analyzing weak classes and rare materials...")

    # Define target categories/materials
    weak_classes = ['cardigans', 'shorts']
    rare_materials = ['denim', 'leather']

    stats = {
        'weak_classes': {},
        'rare_materials': {},
        'cross_analysis': {}
    }

    # Count occurrences
    category_counts = Counter()
    material_counts = Counter()

    for item in tqdm(dataset, desc="Counting categories/materials"):
        cat = item.get('category2', 'unknown')
        mat = item.get('attr_material_primary', 'unknown')

        if cat and cat != 'unknown':
            category_counts[cat] += 1
        if mat and mat != 'unknown':
            material_counts[mat] += 1

    total_samples = len(dataset)

    # Analyze weak classes
    for weak_class in weak_classes:
        count = category_counts.get(weak_class, 0)
        pct = (count / total_samples) * 100
        stats['weak_classes'][weak_class] = {
            'count': count,
            'percentage': pct,
            'is_weak': pct < 5.0  # Flag if less than 5%
        }
        print(".1f")

    # Analyze rare materials
    for rare_mat in rare_materials:
        count = material_counts.get(rare_mat, 0)
        pct = (count / total_samples) * 100
        stats['rare_materials'][rare_mat] = {
            'count': count,
            'percentage': pct,
            'is_rare': pct < 2.0  # Flag if less than 2%
        }
        print(".1f")

    # Cross analysis: rare materials in weak classes
    for weak_class in weak_classes:
        class_items = [item for item in dataset if item.get('category2') == weak_class]
        for rare_mat in rare_materials:
            mat_in_class = sum(1 for item in class_items
                             if item.get('attr_material_primary') == rare_mat)
            key = f"{weak_class}_{rare_mat}"
            stats['cross_analysis'][key] = {
                'count': mat_in_class,
                'percentage_of_class': (mat_in_class / len(class_items)) * 100 if class_items else 0,
                'percentage_of_total': (mat_in_class / total_samples) * 100
            }

    # Save stats
    with open(output_path / "weak_rare_analysis.json", 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


def generate_report(coverage_stats: dict, weak_rare_stats: dict, output_path: Path):
    """
    Generate human-readable report.
    """
    report_path = output_path / "label_report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 7 LABEL COVERAGE REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("ATTRIBUTE COVERAGE SUMMARY\n")
        f.write("-" * 30 + "\n")

        for attr, stats in coverage_stats.items():
            attr_name = attr.replace('attr_', '').replace('_primary', '')
            f.write(f"\n{attr_name.upper()}:\n")
            f.write(".1f")
            f.write(f"  - Unknown: {stats['unknown_count']} ({stats['unknown_pct']:.1f}%)\n")
            f.write(f"  - Null/Missing: {stats['null_count']} ({stats['null_pct']:.1f}%)\n")
            f.write(f"  - Unique values: {stats['unique_values']}\n")

            if stats['top_10_values']:
                f.write("  - Top 10 values:\n")
                for val, count in stats['top_10_values'][:5]:  # Show top 5
                    pct = (count / stats['present_count']) * 100
                    f.write(f"    - {val}: {count} ({pct:.1f}%)\n")

        f.write("\n\nTARGETED WEAK CLASSES & RARE MATERIALS\n")
        f.write("-" * 40 + "\n")

        f.write("\nWeak Classes (target: >5% coverage):\n")
        for cls, stats in weak_rare_stats['weak_classes'].items():
            status = "[OK]" if not stats['is_weak'] else "[WEAK]"
            f.write(f"  - {cls}: {stats['count']} ({stats['percentage']:.1f}%) {status}\n")

        f.write("\nRare Materials (target: >2% coverage):\n")
        for mat, stats in weak_rare_stats['rare_materials'].items():
            status = "[OK]" if not stats['is_rare'] else "[RARE]"
            f.write(f"  - {mat}: {stats['count']} ({stats['percentage']:.1f}%) {status}\n")

        f.write("\n\nCROSS ANALYSIS (Rare materials in weak classes):\n")
        f.write("-" * 50 + "\n")
        for key, stats in weak_rare_stats['cross_analysis'].items():
            if stats['count'] > 0:
                f.write(f"  - {key}: {stats['count']} items ")
                f.write(".1f")
                f.write(".1f")

        # Training implications
        f.write("\n\nTRAINING IMPLICATIONS\n")
        f.write("-" * 20 + "\n")

        # Check for potential issues
        issues = []

        for attr, stats in coverage_stats.items():
            if stats['present_pct'] < 50:
                attr_name = attr.replace('attr_', '').replace('_primary', '')
                issues.append(f"Low coverage for {attr_name} ({stats['present_pct']:.1f}%)")

        weak_classes_issues = [cls for cls, stats in weak_rare_stats['weak_classes'].items() if stats['is_weak']]
        if weak_classes_issues:
            issues.append(f"Very weak classes: {', '.join(weak_classes_issues)}")

        rare_mat_issues = [mat for mat, stats in weak_rare_stats['rare_materials'].items() if stats['is_rare']]
        if rare_mat_issues:
            issues.append(f"Very rare materials: {', '.join(rare_mat_issues)}")

        if issues:
            f.write("[WARNING] POTENTIAL ISSUES:\n")
            for issue in issues:
                f.write(f"  - {issue}\n")
            f.write("\nThese may cause training instability or poor performance on rare cases.\n")
        else:
            f.write("[SUCCESS] No major coverage issues detected.\n")

        f.write("\nReport generated for training quality assessment.\n")

    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Step 7 label coverage report"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to analyze"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for reports"
    )

    args = parser.parse_args()

    print("Step 7 Label Coverage Report")
    print("=" * 40)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output}")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset from: {args.dataset}")
    ds = load_from_disk(args.dataset)
    if isinstance(ds, dict):
        if args.split not in ds:
            raise ValueError(f"Split '{args.split}' not found. Available: {list(ds.keys())}")
        dataset = ds[args.split]
    else:
        dataset = ds

    print(f"Loaded {len(dataset)} samples")

    # Define attributes to analyze
    attribute_columns = [
        'category2',
        'attr_material_primary',
        'attr_pattern_primary',
        'attr_neckline_primary',
        'attr_sleeve_primary'
    ]

    # Run analyses
    coverage_stats = analyze_attribute_coverage(dataset, attribute_columns, output_path)
    weak_rare_stats = analyze_weak_classes_and_rare_materials(dataset, output_path)

    # Generate report
    generate_report(coverage_stats, weak_rare_stats, output_path)

    print(f"\nAll reports saved to: {output_path}")


if __name__ == "__main__":
    main()
