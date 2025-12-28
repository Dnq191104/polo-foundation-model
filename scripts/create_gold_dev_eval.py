#!/usr/bin/env python
"""
Create Gold Dev Evaluation Set

Generates weighted stratified sample of item IDs for trusted evaluation during training.
Includes extra weighting for weak categories (shorts, rompers, cardigans) to ensure
they are well-represented in the evaluation signal.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_gold_dev_eval_set(
    dataset_path,
    split='validation',
    n_queries=300,
    weak_boost=3.0,
    min_per_weak_category=30,
    seed=42
):
    """
    Create weighted stratified gold dev eval set.

    Args:
        dataset_path: Path to HuggingFace dataset
        split: Dataset split to sample from
        n_queries: Target number of queries (200-500 recommended)
        weak_boost: Weight multiplier for weak categories (default: 3.0)
        seed: Random seed for reproducibility

    Returns:
        Dict with eval_ids and metadata
    """

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)[split]

    print(f"Dataset size: {len(dataset)}")

    # Define weak categories (excluding 'graphic' per user preference)
    weak_categories = ['shorts', 'rompers', 'cardigans']

    # Group by category
    category_items = defaultdict(list)
    category_counts = defaultdict(int)

    for i, item in enumerate(dataset):
        category = item.get('category2', 'unknown')
        category_items[category].append(i)
        category_counts[category] += 1

    print(f"Found {len(category_items)} categories")

    # Floor-based sampling: guarantee minimum samples for weak categories first
    total_items = len(dataset)
    eval_ids = []
    np.random.seed(seed)

    # Step 1: Reserve minimum samples for weak categories
    remaining_quota = n_queries
    reserved_weak = {}

    print(f"Reserving minimum samples for weak categories (min_per_weak_category={min_per_weak_category}):")
    for weak_cat in weak_categories:
        if weak_cat in category_items:
            available_ids = category_items[weak_cat]
            n_reserve = min(min_per_weak_category, len(available_ids))

            # Sample minimum samples for this weak category
            sampled_ids = np.random.choice(available_ids, n_reserve, replace=False)
            eval_ids.extend(sampled_ids.tolist())
            remaining_quota -= n_reserve
            reserved_weak[weak_cat] = n_reserve

            print(f"  {weak_cat}: {n_reserve} samples reserved")

    # Step 2: Weighted stratified sampling for remaining quota
    if remaining_quota > 0:
        print(f"\nRemaining quota: {remaining_quota} samples")
        print("Applying weighted stratified sampling for remaining categories...")

        # Calculate base weights (excluding already reserved weak categories)
        category_weights = {}
        for category, ids in category_items.items():
            if category not in weak_categories:  # Skip weak categories already reserved
                # Base proportional weight
                base_weight = len(ids) / total_items
                category_weights[category] = base_weight

        # Add remaining weak category samples with boosted weights
        for weak_cat in weak_categories:
            if weak_cat in category_items:
                reserved = reserved_weak.get(weak_cat, 0)
                remaining_weak_ids = len(category_items[weak_cat]) - reserved
                if remaining_weak_ids > 0:
                    # Base proportional weight for remaining items
                    base_weight = remaining_weak_ids / total_items
                    # Boost weak categories
                    weight = base_weight * weak_boost
                    category_weights[weak_cat] = weight
                    print(f"  Boosting remaining {weak_cat}: {base_weight:.4f} -> {weight:.4f} (x{weak_boost})")

        # Normalize weights
        if category_weights:
            total_weight = sum(category_weights.values())
            category_weights = {k: v/total_weight for k, v in category_weights.items()}

            # Sample remaining quota
            for category, ids in category_items.items():
                if category in category_weights:
                    weight = category_weights[category]
                    # Adjust for already reserved samples in weak categories
                    already_reserved = reserved_weak.get(category, 0)
                    available_ids = [idx for idx in ids if idx not in eval_ids]

                    if available_ids:
                        n_category_samples = max(0, int(remaining_quota * weight))

                        # Don't exceed available items or remaining quota
                        n_category_samples = min(n_category_samples, len(available_ids), remaining_quota)

                        if n_category_samples > 0:
                            sampled_ids = np.random.choice(available_ids, n_category_samples, replace=False)
                            eval_ids.extend(sampled_ids.tolist())
                            remaining_quota -= n_category_samples

                            print(".1f")

    # Final verification
    if len(eval_ids) != n_queries:
        print(f"Warning: Final size {len(eval_ids)} differs from target {n_queries}")
        if len(eval_ids) > n_queries:
            # Trim if overshot
            eval_ids = eval_ids[:n_queries]
        # If undershot, we keep what we have (better than failing)

    print(f"\nFinal gold eval set: {len(eval_ids)} queries")

    # Verify stratification
    category_counts_eval = defaultdict(int)
    weak_category_counts_eval = defaultdict(int)

    for idx in eval_ids:
        category = dataset[idx].get('category2', 'unknown')
        category_counts_eval[category] += 1
        if category in weak_categories:
            weak_category_counts_eval[category] += 1

    print("Category distribution in gold eval set:")
    for cat, count in sorted(category_counts_eval.items()):
        pct = count / len(eval_ids) * 100
        marker = " [WEAK]" if cat in weak_categories else ""
        print(f"  {cat}: {count} ({pct:.1f}%){marker}")

    # Calculate weak category representation
    total_weak_in_dataset = sum(category_counts[cat] for cat in weak_categories if cat in category_counts)
    total_weak_in_eval = sum(weak_category_counts_eval.values())

    weak_pct_dataset = total_weak_in_dataset / total_items * 100
    weak_pct_eval = total_weak_in_eval / len(eval_ids) * 100

    print(f"\nWeak category summary:")
    print(f"  Dataset: {total_weak_in_dataset}/{total_items} ({weak_pct_dataset:.1f}%)")
    print(f"  Gold eval: {total_weak_in_eval}/{len(eval_ids)} ({weak_pct_eval:.1f}%)")
    print(".1f")

    # Prepare metadata
    metadata = {
        'n_queries': len(eval_ids),
        'weak_boost': weak_boost,
        'min_per_weak_category': min_per_weak_category,
        'weak_categories': weak_categories,
        'reserved_weak_samples': reserved_weak,
        'seed': seed,
        'category_distribution': dict(sorted(category_counts_eval.items())),
        'weak_category_distribution': dict(sorted(weak_category_counts_eval.items())),
        'dataset_split': split,
        'created_at': str(np.datetime64('now')),
        'version': '1.1',
        'sampling_method': 'floor_based'
    }

    return {
        'eval_ids': eval_ids,
        'metadata': metadata
    }


def main():
    parser = argparse.ArgumentParser(description="Create gold dev evaluation set")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--output", type=str, default="artifacts/step7/gold_dev_eval_ids.json", help="Output JSON file")
    parser.add_argument("--n_queries", type=int, default=300, help="Number of queries (200-500 recommended)")
    parser.add_argument("--weak_boost", type=float, default=3.0, help="Weight multiplier for weak categories")
    parser.add_argument("--min_per_weak_category", type=int, default=30, help="Minimum samples per weak category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    result = create_gold_dev_eval_set(
        args.dataset,
        args.split,
        args.n_queries,
        args.weak_boost,
        args.min_per_weak_category,
        args.seed
    )

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nGold dev eval IDs saved to: {output_path}")
    print(f"Ready for manual verification with verify_gold_dev_eval.py")


if __name__ == "__main__":
    main()
