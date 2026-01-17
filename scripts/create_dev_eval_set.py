#!/usr/bin/env python
"""
Create Dev Evaluation Set

Generates stratified sample of item IDs for fast evaluation during training.
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


def create_dev_eval_set(dataset_path, split='validation', n_queries=500, seed=42):
    """Create stratified dev eval set"""

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)[split]

    print(f"Dataset size: {len(dataset)}")

    # Group by category
    category_items = defaultdict(list)
    for i, item in enumerate(dataset):
        category = item.get('category2', 'unknown')
        category_items[category].append(i)

    print(f"Found {len(category_items)} categories")

    # Calculate stratified sample sizes
    total_items = len(dataset)
    dev_ids = []

    np.random.seed(seed)

    for category, ids in category_items.items():
        # Proportional sampling, minimum 5 per category
        category_proportion = len(ids) / total_items
        n_category_samples = max(5, int(n_queries * category_proportion))

        # Don't exceed available items
        n_category_samples = min(n_category_samples, len(ids))

        # Sample without replacement
        sampled_ids = np.random.choice(ids, n_category_samples, replace=False)
        dev_ids.extend(sampled_ids.tolist())

        print(".1f")

    # Trim to exact size if needed
    if len(dev_ids) > n_queries:
        dev_ids = dev_ids[:n_queries]

    print(f"\nFinal dev set: {len(dev_ids)} queries")

    # Verify stratification
    category_counts = defaultdict(int)
    for idx in dev_ids:
        category = dataset[idx].get('category2', 'unknown')
        category_counts[category] += 1

    print("Category distribution in dev set:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    return dev_ids


def main():
    parser = argparse.ArgumentParser(description="Create dev evaluation set")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--n_queries", type=int, default=500, help="Number of queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    dev_ids = create_dev_eval_set(
        args.dataset, args.split, args.n_queries, args.seed
    )

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(dev_ids, f, indent=2)

    print(f"\nDev eval IDs saved to: {output_path}")


if __name__ == "__main__":
    main()






