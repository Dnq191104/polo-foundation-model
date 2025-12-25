#!/usr/bin/env python
"""
Build and Audit Product Groups

Derives product group keys and audits their quality.

Usage:
    python scripts/step7_build_groups.py \
        --dataset data/processed_v2/hf \
        --split train \
        --output artifacts/step7/groups.json \
        --min_confidence 0.6
"""

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_from_disk

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.group_key import GroupKeyDeriver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and audit product groups"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to HuggingFace dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path for group index"
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.6,
        help="Minimum confidence for grouping (default: 0.6)"
    )
    parser.add_argument(
        "--audit_samples",
        type=int,
        default=50,
        help="Number of groups to audit (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
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


def print_stats(stats: dict):
    """Print group statistics."""
    print("\n" + "=" * 60)
    print("GROUP STATISTICS")
    print("=" * 60)
    print(f"Total items: {stats['total_items']}")
    print(f"Total groups: {stats['total_groups']}")
    print(f"Multi-item groups: {stats['multi_item_groups']}")
    print(f"Singleton groups: {stats['singleton_groups']}")
    print(f"Average group size: {stats['avg_group_size']:.2f}")
    print(f"Max group size: {stats['max_group_size']}")
    print()
    
    print("SIZE DISTRIBUTION:")
    print("-" * 40)
    size_dist = stats['size_distribution']
    for size in sorted(size_dist.keys()):
        count = size_dist[size]
        print(f"  Size {size}: {count} groups")


def audit_groups(dataset, group_index: dict, metadata: list, n_samples: int = 50, seed: int = 42):
    """Audit random groups for quality."""
    random.seed(seed)
    
    # Get multi-item groups
    multi_groups = {k: v for k, v in group_index.items() if len(v) > 1}
    
    if not multi_groups:
        print("\nWARNING: No multi-item groups found to audit")
        return
    
    # Sample groups
    sample_keys = random.sample(list(multi_groups.keys()), min(n_samples, len(multi_groups)))
    
    print("\n" + "=" * 60)
    print(f"AUDIT SAMPLE ({len(sample_keys)} groups)")
    print("=" * 60)
    print("\nReview these groups to verify items belong together:")
    print()
    
    for i, group_key in enumerate(sample_keys[:10]):  # Print first 10
        indices = multi_groups[group_key]
        print(f"\nGroup {i+1}: {group_key} (size={len(indices)})")
        print("-" * 40)
        
        for idx in indices:
            item = dataset[int(idx)]
            meta = metadata[idx]
            
            item_id = item.get('item_ID', '')
            category = item.get('category2', '')
            text_preview = item.get('text', '')[:60]
            
            print(f"  [{idx}] {item_id} | {category}")
            print(f"      Method: {meta['method']} (conf={meta['confidence']:.2f})")
            print(f"      Text: {text_preview}...")
    
    if len(sample_keys) > 10:
        print(f"\n... ({len(sample_keys) - 10} more groups sampled)")
    
    print("\n" + "=" * 60)
    print("AUDIT INSTRUCTIONS")
    print("=" * 60)
    print("\n1. Review the sampled groups above")
    print("2. Verify that items within each group are:")
    print("   - Same product in different views/poses")
    print("   - OR same product in different colors/sizes")
    print("   - NOT completely different products")
    print("\n3. If groups look good:")
    print("   -> Use for strong positive pairs in training")
    print("\n4. If many groups are wrong:")
    print("   -> Adjust min_confidence threshold")
    print("   -> Or disable group-based strong positives")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("BUILD PRODUCT GROUPS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Load dataset
    ds = load_dataset_split(args.dataset, args.split)
    
    # Build groups
    print("\nDeriving group keys...")
    deriver = GroupKeyDeriver()
    result = deriver.build_group_index(ds, min_confidence=args.min_confidence)
    
    # Print statistics
    print_stats(result['stats'])
    
    # Audit sample
    if args.audit_samples > 0:
        audit_groups(
            ds,
            result['index'],
            result['metadata'],
            n_samples=args.audit_samples,
            seed=args.seed
        )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare for JSON serialization (convert int keys to strings if needed)
    serializable_result = {
        'index': {k: [int(idx) for idx in v] for k, v in result['index'].items()},
        'stats': result['stats'],
        'config': {
            'dataset': args.dataset,
            'split': args.split,
            'min_confidence': args.min_confidence,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_result, f, indent=2)
    
    print(f"\nSUCCESS: Groups saved to: {args.output}")
    print(f"  Total groups: {result['stats']['total_groups']}")
    print(f"  Multi-item groups: {result['stats']['multi_item_groups']}")
    
    # Recommendation
    multi_pct = result['stats']['multi_item_groups'] / result['stats']['total_groups'] * 100 if result['stats']['total_groups'] > 0 else 0
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if multi_pct > 10:
        print(f"\nGOOD: {multi_pct:.1f}% of groups have multiple items")
        print("  -> Use group-based strong positives in training")
        print(f"  -> Potential strong positive pairs: ~{result['stats']['multi_item_groups']} groups")
    else:
        print(f"\nWARNING: Only {multi_pct:.1f}% of groups have multiple items")
        print("  -> Limited benefit from group-based strong positives")
        print("  -> Consider falling back to attribute-based positives only")


if __name__ == "__main__":
    main()

