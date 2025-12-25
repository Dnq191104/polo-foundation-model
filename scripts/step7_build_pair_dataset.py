#!/usr/bin/env python
"""
Build Step 7 Pair Dataset

Integrates all pair generation strategies:
- Strong positives (group-based or attribute-based)
- Medium positives (category + material)
- Weak positives (category + design tags)
- Hard negatives (mined + attribute-targeted)
- Easy negatives (different category)

Usage:
    python scripts/step7_build_pair_dataset.py \
        --dataset data/processed_v2/hf \
        --split train \
        --groups artifacts/step7/groups.json \
        --hard_negatives artifacts/step7/hard_negatives.parquet \
        --output artifacts/step7/pairs.parquet \
        --n_pairs 50000
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.pair_generator import AttributePairGenerator
from src.datasets.graphic_policy import GraphicPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Step 7 pair dataset"
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
        "--groups",
        type=str,
        default=None,
        help="Path to groups JSON (optional, for strong positives)"
    )
    parser.add_argument(
        "--hard_negatives",
        type=str,
        default=None,
        help="Path to mined hard negatives parquet (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file for pairs"
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=50000,
        help="Target number of pairs (default: 50000)"
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


def load_groups(groups_path: str):
    """Load pre-computed groups."""
    if not groups_path or not Path(groups_path).exists():
        return None
    
    print(f"Loading groups from: {groups_path}")
    with open(groups_path, 'r') as f:
        data = json.load(f)
    
    return data.get('index', {})


def load_hard_negatives(hard_neg_path: str):
    """Load pre-mined hard negatives."""
    if not hard_neg_path or not Path(hard_neg_path).exists():
        return None
    
    print(f"Loading hard negatives from: {hard_neg_path}")
    df = pd.read_parquet(hard_neg_path)
    
    # Build index: anchor_idx -> list of negative info
    hard_neg_index = defaultdict(list)
    for _, row in df.iterrows():
        hard_neg_index[int(row['anchor_idx'])].append({
            'neg_idx': int(row['neg_idx']),
            'mismatch_types': row['mismatch_types'],
        })
    
    return dict(hard_neg_index)


def generate_group_positives(groups_index: dict, n_target: int, seed: int):
    """Generate positives from groups."""
    import random
    rng = random.Random(seed)
    
    pairs = []
    
    for group_key, indices in groups_index.items():
        if len(indices) < 2:
            continue
        
        # Generate all pairs within group
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i + 1:]:
                pairs.append({
                    'anchor_idx': idx1,
                    'other_idx': idx2,
                    'label': 1,  # Positive
                    'pair_type': 'strong_positive_group',
                    'metadata': f'group={group_key}'
                })
    
    rng.shuffle(pairs)
    return pairs[:n_target]


def main():
    args = parse_args()
    
    print("=" * 60)
    print("BUILD STEP 7 PAIR DATASET")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Target pairs: {args.n_pairs}")
    print(f"Groups: {args.groups or 'None'}")
    print(f"Hard negatives: {args.hard_negatives or 'None'}")
    print("=" * 60)
    
    # Load dataset
    ds = load_dataset_split(args.dataset, args.split)
    
    # Load optional inputs
    groups_index = load_groups(args.groups)
    hard_neg_index = load_hard_negatives(args.hard_negatives)
    
    # Initialize pair generator with graphic policy
    print("\nInitializing pair generator...")
    graphic_policy = GraphicPolicy(remap_to_pattern=True)
    pair_gen = AttributePairGenerator(
        unknown_value='unknown',
        seed=args.seed,
        graphic_policy=graphic_policy
    )
    
    # Target distribution (allocate among pair types)
    n_target = args.n_pairs
    
    # If we have groups, use them for strong positives
    if groups_index:
        n_strong_group = int(n_target * 0.25)  # 25% from groups
        n_strong_attr = int(n_target * 0.10)   # 10% from attributes
    else:
        n_strong_group = 0
        n_strong_attr = int(n_target * 0.35)   # 35% from attributes only
    
    n_medium = int(n_target * 0.20)  # 20%
    n_weak = int(n_target * 0.10)    # 10%
    
    # Negatives
    if hard_neg_index:
        n_hard_mined = int(n_target * 0.25)  # 25% from mined
        n_hard_attr = int(n_target * 0.05)   # 5% from attribute-targeted
    else:
        n_hard_mined = 0
        n_hard_attr = int(n_target * 0.30)   # 30% from attribute-targeted
    
    n_easy = int(n_target * 0.05)  # 5%
    
    all_pairs = []
    
    # 1. Group-based strong positives
    if groups_index and n_strong_group > 0:
        print(f"\nGenerating {n_strong_group} group-based strong positives...")
        group_pairs = generate_group_positives(groups_index, n_strong_group, args.seed)
        all_pairs.extend(group_pairs)
        print(f"  Generated: {len(group_pairs)}")
    
    # 2. Attribute-based strong positives
    if n_strong_attr > 0:
        print(f"\nGenerating {n_strong_attr} attribute-based strong positives...")
        strong_pos = pair_gen.generate_strong_positives(ds, n_pairs=n_strong_attr)
        for idx1, idx2 in strong_pos:
            all_pairs.append({
                'anchor_idx': idx1,
                'other_idx': idx2,
                'label': 1,
                'pair_type': 'strong_positive_attr',
                'metadata': 'same_category_material'
            })
        print(f"  Generated: {len(strong_pos)}")
    
    # 3. Medium positives
    print(f"\nGenerating {n_medium} medium positives...")
    medium_pos = pair_gen.generate_medium_positives(ds, n_pairs=n_medium)
    for idx1, idx2 in medium_pos:
        all_pairs.append({
            'anchor_idx': idx1,
            'other_idx': idx2,
            'label': 1,
            'pair_type': 'medium_positive',
            'metadata': 'same_category_design'
        })
    print(f"  Generated: {len(medium_pos)}")
    
    # 4. Weak positives
    print(f"\nGenerating {n_weak} weak positives...")
    # Note: This uses the same method as medium but without material constraint
    # You may want to add a separate method for purely design-based matching
    weak_pos = pair_gen.generate_medium_positives(ds, n_pairs=n_weak, exclude_strong=True)
    for idx1, idx2 in weak_pos:
        all_pairs.append({
            'anchor_idx': idx1,
            'other_idx': idx2,
            'label': 1,
            'pair_type': 'weak_positive',
            'metadata': 'same_category_shared_design'
        })
    print(f"  Generated: {len(weak_pos)}")
    
    # 5. Mined hard negatives
    if hard_neg_index and n_hard_mined > 0:
        print(f"\nGenerating {n_hard_mined} mined hard negatives...")
        
        # Sample from mined negatives
        import random
        rng = random.Random(args.seed)
        
        all_mined = []
        for anchor_idx, neg_list in hard_neg_index.items():
            for neg_info in neg_list:
                all_mined.append({
                    'anchor_idx': anchor_idx,
                    'other_idx': neg_info['neg_idx'],
                    'label': 0,
                    'pair_type': f"hard_negative_mined",
                    'metadata': f"mismatch={neg_info['mismatch_types']}"
                })
        
        rng.shuffle(all_mined)
        sampled_mined = all_mined[:n_hard_mined]
        all_pairs.extend(sampled_mined)
        print(f"  Generated: {len(sampled_mined)}")
    
    # 6. Attribute-targeted hard negatives
    print(f"\nGenerating {n_hard_attr} attribute-targeted hard negatives...")
    hard_neg_pairs = pair_gen.generate_hard_negatives(ds, n_pairs=n_hard_attr)
    for idx1, idx2 in hard_neg_pairs:
        all_pairs.append({
            'anchor_idx': idx1,
            'other_idx': idx2,
            'label': 0,
            'pair_type': 'hard_negative_material',
            'metadata': 'same_category_diff_material'
        })
    print(f"  Generated: {len(hard_neg_pairs)}")
    
    # 7. Easy negatives
    print(f"\nGenerating {n_easy} easy negatives...")
    easy_neg_pairs = pair_gen.generate_easy_negatives(ds, n_pairs=n_easy)
    for idx1, idx2 in easy_neg_pairs:
        all_pairs.append({
            'anchor_idx': idx1,
            'other_idx': idx2,
            'label': 0,
            'pair_type': 'easy_negative',
            'metadata': 'different_category'
        })
    print(f"  Generated: {len(easy_neg_pairs)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_pairs)
    
    # Add anchor metadata for convenience
    print("\nAdding anchor metadata...")
    anchor_categories = []
    anchor_materials = []
    anchor_necklines = []
    
    for idx in tqdm(df['anchor_idx'], desc="Loading anchor metadata"):
        item = ds[int(idx)]
        anchor_categories.append(item.get('category2', ''))
        anchor_materials.append(item.get('attr_material_primary', 'unknown'))
        anchor_necklines.append(item.get('attr_neckline_primary', 'unknown'))
    
    df['anchor_category'] = anchor_categories
    df['anchor_material'] = anchor_materials
    df['anchor_neckline'] = anchor_necklines
    
    # Statistics
    print("\n" + "=" * 60)
    print("PAIR DATASET STATISTICS")
    print("=" * 60)
    print(f"Total pairs: {len(df)}")
    print()
    
    print("BY TYPE:")
    print("-" * 40)
    type_counts = df['pair_type'].value_counts()
    for ptype, count in type_counts.items():
        pct = count / len(df) * 100
        print(f"  {ptype:30s} {count:>6} ({pct:>5.1f}%)")
    
    print()
    print("BY LABEL:")
    print("-" * 40)
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        label_name = "Positive" if label == 1 else "Negative"
        pct = count / len(df) * 100
        print(f"  {label_name:12s} {count:>6} ({pct:>5.1f}%)")
    
    print()
    print("TOP ANCHOR CATEGORIES:")
    print("-" * 40)
    cat_counts = df['anchor_category'].value_counts()
    for cat, count in cat_counts.head(10).items():
        print(f"  {cat:20s} {count:>6}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    
    print(f"\n✓ Pair dataset saved to: {args.output}")
    print(f"  Total pairs: {len(df)}")
    print(f"  Positives: {label_counts.get(1, 0)}")
    print(f"  Negatives: {label_counts.get(0, 0)}")
    
    # Save stats
    stats = {
        'total_pairs': len(df),
        'pair_type_distribution': type_counts.to_dict(),
        'label_distribution': label_counts.to_dict(),
        'category_distribution': cat_counts.to_dict(),
    }
    
    stats_path = output_path.parent / f"{output_path.stem}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()

