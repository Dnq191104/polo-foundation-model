#!/usr/bin/env python
"""
Step 7 Pair Leakage Checks

Checks pairs.parquet for training data quality issues:
- Self-negatives (same item as both anchor and negative)
- Duplicate pairs
- Label conflicts (graphic category vs pattern contradictions)
- Attribute consistency issues
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_self_negatives(pairs_df: pd.DataFrame) -> dict:
    """
    Check if any item appears as both anchor and other in same pair.
    """
    print("Checking for self-negatives...")

    self_negative_pairs = []

    for idx, row in tqdm(pairs_df.iterrows(), desc="Checking pairs"):
        anchor_idx = row['anchor_idx']
        other_idx = row['other_idx']

        if anchor_idx == other_idx:
            self_negative_pairs.append({
                'pair_idx': idx,
                'anchor_idx': anchor_idx,
                'other_idx': other_idx,
                'pair_type': row.get('pair_type', 'unknown'),
                'label': row.get('label', 'unknown')
            })

    return {
        'count': len(self_negative_pairs),
        'percentage': (len(self_negative_pairs) / len(pairs_df)) * 100,
        'examples': self_negative_pairs[:10]  # First 10 examples
    }


def check_duplicate_pairs(pairs_df: pd.DataFrame) -> dict:
    """
    Check for exact duplicate pairs (same anchor + same other + same label).
    """
    print("Checking for duplicate pairs...")

    # Create signature for each pair
    pair_signatures = []
    for idx, row in pairs_df.iterrows():
        signature = (
            row['anchor_idx'],
            row['other_idx'],
            row.get('label', 0),
            row.get('pair_type', 'unknown')
        )
        pair_signatures.append((idx, signature))

    # Find duplicates
    signature_counts = Counter(sig for _, sig in pair_signatures)
    duplicates = {sig: count for sig, count in signature_counts.items() if count > 1}

    duplicate_pairs = []
    for orig_idx, sig in pair_signatures:
        if signature_counts[sig] > 1:
            duplicate_pairs.append({
                'pair_idx': orig_idx,
                'signature': sig,
                'duplicate_count': signature_counts[sig]
            })

    return {
        'unique_signatures': len(signature_counts),
        'duplicate_groups': len(duplicates),
        'total_duplicate_pairs': len(duplicate_pairs),
        'percentage_duplicates': (len(duplicate_pairs) / len(pairs_df)) * 100,
        'examples': list(duplicates.keys())[:5]  # First 5 duplicate signatures
    }


def check_graphic_contradictions(pairs_df: pd.DataFrame, dataset) -> dict:
    """
    Check for 'graphic' category contradictions.

    The policy is: 'graphic' should be treated as pattern-only, not garment category.
    But if pairs are generated with 'graphic' as category signal, this creates conflicts.
    """
    print("Checking graphic category contradictions...")

    # Get all items with category2='graphic'
    graphic_items = set()
    for idx, item in enumerate(dataset):
        if item.get('category2') == 'graphic':
            graphic_items.add(idx)

    if not graphic_items:
        return {'count': 0, 'items_found': 0, 'message': 'No graphic items found in dataset'}

    print(f"Found {len(graphic_items)} items with category2='graphic'")

    # Check how graphic items are used in pairs
    contradictions = []

    for idx, row in tqdm(pairs_df.iterrows(), desc="Checking graphic usage"):
        anchor_idx = row['anchor_idx']

        # If anchor is graphic, check if it's used as category signal
        if anchor_idx in graphic_items:
            pair_type = row.get('pair_type', '')

            # Check for category-based pair types
            if any(keyword in pair_type.lower() for keyword in ['category', 'same_cat', 'diff_cat']):
                contradictions.append({
                    'pair_idx': idx,
                    'anchor_idx': anchor_idx,
                    'pair_type': pair_type,
                    'issue': 'graphic_item_used_as_category_anchor'
                })

        # Check other item - if graphic item is paired with non-graphic anchor
        other_idx = row['other_idx']
        if other_idx in graphic_items and anchor_idx not in graphic_items:
            pair_type = row.get('pair_type', '')
            label = row.get('label', 0)
            # For graphic contradictions, check if this is being treated as category signal
            if any(keyword in pair_type.lower() for keyword in ['category', 'same_cat', 'diff_cat']):
                contradictions.append({
                    'pair_idx': idx,
                    'anchor_idx': anchor_idx,
                    'other_idx': other_idx,
                    'pair_type': pair_type,
                    'label': label,
                    'issue': 'graphic_item_used_as_category_pair'
                })

    return {
        'count': len(contradictions),
        'graphic_items_total': len(graphic_items),
        'percentage_contradictions': (len(contradictions) / len(pairs_df)) * 100,
        'examples': contradictions[:10]
    }


def check_attribute_consistency(pairs_df: pd.DataFrame, dataset) -> dict:
    """
    Check for attribute consistency issues in pairs.

    For attribute-targeted pairs (e.g., "same_material_diff_category"),
    verify the attributes actually differ as expected.
    """
    print("Checking attribute consistency...")

    issues = []

    # Build index to item mapping for fast lookup
    idx_to_item = {}
    for idx, item in enumerate(dataset):
        idx_to_item[idx] = item

    for idx, row in tqdm(pairs_df.iterrows(), desc="Checking attributes"):
        anchor_idx = row['anchor_idx']
        pair_type = row.get('pair_type', '')

        if anchor_idx not in idx_to_item:
            continue

        anchor_item = idx_to_item[anchor_idx]

        # Check different pair types
        if 'same_material' in pair_type:
            # Should have same material
            anchor_mat = anchor_item.get('attr_material_primary', 'unknown')

            # Check other item based on pair type and label
            other_idx = row['other_idx']
            if other_idx in idx_to_item:
                other_item = idx_to_item[other_idx]
                other_mat = other_item.get('attr_material_primary', 'unknown')
                label = row.get('label', 0)

                if 'same_material' in pair_type:
                    # Should have same material if label=1 (positive)
                    if label == 1 and anchor_mat != other_mat and anchor_mat != 'unknown' and other_mat != 'unknown':
                        issues.append({
                            'pair_idx': idx,
                            'type': 'material_mismatch_in_same_material_positive',
                            'anchor_material': anchor_mat,
                            'other_material': other_mat,
                            'anchor_idx': anchor_idx,
                            'other_idx': other_idx,
                            'label': label
                        })
                elif 'diff_material' in pair_type or 'hard_negative' in pair_type:
                    # Should have different material if label=0 (negative)
                    if label == 0 and anchor_mat == other_mat and anchor_mat != 'unknown':
                        issues.append({
                            'pair_idx': idx,
                            'type': 'material_match_in_diff_material_negative',
                            'anchor_material': anchor_mat,
                            'other_material': other_mat,
                            'anchor_idx': anchor_idx,
                            'other_idx': other_idx,
                            'label': label
                        })

    return {
        'count': len(issues),
        'percentage_issues': (len(issues) / len(pairs_df)) * 100,
        'examples': issues[:10]
    }


def generate_leakage_report(results: dict, output_path: Path):
    """
    Generate human-readable leakage report.
    """
    report_path = output_path / "pair_leakage_report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 7 PAIR LEAKAGE CHECKS REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Self-negatives
        self_neg = results['self_negatives']
        f.write("SELF-NEGATIVES (Same item as anchor + negative)\n")
        f.write("-" * 50 + "\n")
        f.write(f"Count: {self_neg['count']}\n")
        f.write(".1f")
        if self_neg['count'] > 0:
            f.write("[WARNING] ISSUE: Training will be confused by self-negatives!\n")
            f.write("Examples:\n")
            for ex in self_neg['examples'][:3]:
                f.write(f"  Pair {ex['pair_idx']}: anchor {ex['anchor_idx']} in negatives\n")
        else:
            f.write("[SUCCESS] No self-negatives found.\n")
        f.write("\n")

        # Duplicates
        dups = results['duplicates']
        f.write("DUPLICATE PAIRS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Unique signatures: {dups['unique_signatures']}\n")
        f.write(f"Duplicate groups: {dups['duplicate_groups']}\n")
        f.write(f"Total duplicate pairs: {dups['total_duplicate_pairs']}\n")
        f.write(".1f")
        if dups['total_duplicate_pairs'] > 0:
            f.write("[WARNING] ISSUE: Duplicate pairs waste compute and may cause overfitting.\n")
        else:
            f.write("[SUCCESS] No duplicate pairs found.\n")
        f.write("\n")

        # Graphic contradictions
        graphic = results['graphic_contradictions']
        f.write("GRAPHIC CATEGORY CONTRADICTIONS\n")
        f.write("-" * 35 + "\n")
        f.write(f"Graphic items in dataset: {graphic.get('graphic_items_total', 0)}\n")
        f.write(f"Contradiction pairs: {graphic['count']}\n")
        f.write(".1f")
        if graphic['count'] > 0:
            f.write("[WARNING] ISSUE: 'graphic' used as category signal conflicts with pattern-only policy!\n")
            f.write("Examples:\n")
            for ex in graphic['examples'][:3]:
                f.write(f"  {ex['issue']}: pair {ex['pair_idx']}, type '{ex['pair_type']}'\n")
        else:
            f.write("[SUCCESS] No graphic contradictions found.\n")
        f.write("\n")

        # Attribute consistency
        attr = results['attribute_consistency']
        f.write("ATTRIBUTE CONSISTENCY ISSUES\n")
        f.write("-" * 30 + "\n")
        f.write(f"Issues found: {attr['count']}\n")
        f.write(".1f")
        if attr['count'] > 0:
            f.write("[WARNING] ISSUE: Pair labels don't match actual attribute values!\n")
            f.write("Examples:\n")
            for ex in attr['examples'][:3]:
                f.write(f"  {ex['type']}: {ex.get('anchor_material', 'unknown')} vs {ex.get('positive_material', ex.get('negative_material', 'unknown'))}\n")
        else:
            f.write("[SUCCESS] No attribute consistency issues found.\n")
        f.write("\n")

        # Overall assessment
        f.write("OVERALL ASSESSMENT\n")
        f.write("-" * 20 + "\n")

        total_issues = (
            self_neg['count'] +
            dups['total_duplicate_pairs'] +
            graphic['count'] +
            attr['count']
        )

        if total_issues == 0:
            f.write("[SUCCESS] EXCELLENT: No leakage issues detected!\n")
            f.write("Pair generation is clean and ready for training.\n")
        elif total_issues < len(results.get('pairs_df', [])) * 0.01:  # < 1%
            f.write("[WARNING] MINOR ISSUES: Some problems detected but likely tolerable.\n")
            f.write("Monitor training for unexpected behavior.\n")
        else:
            f.write("[CRITICAL] MAJOR ISSUES: Significant leakage detected!\n")
            f.write("Fix pair generation before training - these will poison the model.\n")

        f.write(f"\nTotal problematic pairs: {total_issues}\n")

    print(f"Leakage report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check Step 7 pairs for leakage and quality issues"
    )

    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to pairs.parquet file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory (for item metadata)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for reports"
    )

    args = parser.parse_args()

    print("Step 7 Pair Leakage Checks")
    print("=" * 35)
    print(f"Pairs: {args.pairs}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pairs
    print(f"\nLoading pairs from: {args.pairs}")
    pairs_df = pd.read_parquet(args.pairs)
    print(f"Loaded {len(pairs_df)} pairs")

    # Load dataset for metadata
    print(f"Loading dataset from: {args.dataset}")
    from datasets import load_from_disk
    ds = load_from_disk(args.dataset)
    if isinstance(ds, dict):
        if args.split not in ds:
            raise ValueError(f"Split '{args.split}' not found. Available: {list(ds.keys())}")
        dataset = ds[args.split]
    else:
        dataset = ds
    print(f"Loaded {len(dataset)} items for metadata")

    # Run all checks
    results = {
        'self_negatives': check_self_negatives(pairs_df),
        'duplicates': check_duplicate_pairs(pairs_df),
        'graphic_contradictions': check_graphic_contradictions(pairs_df, dataset),
        'attribute_consistency': check_attribute_consistency(pairs_df, dataset)
    }

    # Save detailed results
    with open(output_path / "pair_leakage_results.json", 'w') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for key, val in results.items():
            json_results[key] = val.copy()
            # Remove non-serializable examples
            if 'examples' in json_results[key]:
                json_results[key]['examples'] = str(json_results[key]['examples'])[:500] + "..."
        json.dump(json_results, f, indent=2)

    # Generate report
    generate_leakage_report(results, output_path)

    print(f"\nAll results saved to: {output_path}")

    # Exit with error if major issues found
    total_issues = (
        results['self_negatives']['count'] +
        results['duplicates']['total_duplicate_pairs'] +
        results['graphic_contradictions']['count'] +
        results['attribute_consistency']['count']
    )

    if total_issues > 0:
        print(f"\n[WARNING] Found {total_issues} problematic pairs. Check the report for details.")
        if total_issues > len(pairs_df) * 0.05:  # > 5%
            print("[CRITICAL] Major leakage detected! Fix pair generation before training.")
            sys.exit(1)


if __name__ == "__main__":
    main()
