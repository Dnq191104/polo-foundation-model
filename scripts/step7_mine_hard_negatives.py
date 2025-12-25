#!/usr/bin/env python
"""
Mine Hard Negatives Using Step 6 Baseline

Uses baseline retrieval to find "confuser" items that are visually similar
but differ on target attributes (material, neckline, pattern).

Usage:
    python scripts/step7_mine_hard_negatives.py \
        --catalog_dir artifacts/retrieval/openclip_vitb32_v0 \
        --dataset data/processed_v2/hf \
        --split train \
        --output artifacts/step7/hard_negatives.parquet \
        --top_k 50 \
        --max_anchors 10000
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.embedder import OpenCLIPEmbedder
from src.retrieval.engine import RetrievalEngine
from src.datasets.graphic_policy import GraphicPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine hard negatives using Step 6 baseline"
    )
    
    parser.add_argument(
        "--catalog_dir",
        type=str,
        required=True,
        help="Path to Step 6 catalog embeddings"
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
        help="Output parquet file for hard negatives"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of top candidates to check (default: 50)"
    )
    parser.add_argument(
        "--max_anchors",
        type=int,
        default=10000,
        help="Max number of anchors to process (default: 10000)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model name"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Pretrained weights"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--weight_image",
        type=float,
        default=0.7,
        help="Image weight in fusion"
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


def mine_hard_negatives_for_anchor(
    anchor_item: dict,
    anchor_idx: int,
    retrieval_results: list,
    dataset,
    graphic_policy: GraphicPolicy
) -> list:
    """
    Mine hard negatives from retrieval results for a single anchor.
    
    Hard negatives are candidates that:
    - Match category2 (excluding graphic)
    - Differ on a target attribute (material, neckline, pattern)
    
    Args:
        anchor_item: Anchor dataset item
        anchor_idx: Anchor index
        retrieval_results: List of retrieval results
        dataset: Full dataset
        graphic_policy: Graphic policy instance
        
    Returns:
        List of hard negative dicts
    """
    anchor_cat = anchor_item.get('category2', '')
    anchor_mat = anchor_item.get('attr_material_primary', 'unknown')
    anchor_neck = anchor_item.get('attr_neckline_primary', 'unknown')
    anchor_pat = anchor_item.get('attr_pattern_primary', 'unknown')
    
    # Skip if anchor is graphic
    if graphic_policy.should_exclude_for_category_matching(anchor_cat):
        return []
    
    hard_negs = []
    
    for result in retrieval_results:
        result_idx = result['idx']
        
        # Skip self
        if result_idx == anchor_idx:
            continue
        
        result_item = dataset[int(result_idx)]
        result_cat = result_item.get('category2', '')
        result_mat = result_item.get('attr_material_primary', 'unknown')
        result_neck = result_item.get('attr_neckline_primary', 'unknown')
        result_pat = result_item.get('attr_pattern_primary', 'unknown')
        
        # Skip if result is graphic
        if graphic_policy.should_exclude_for_category_matching(result_cat):
            continue
        
        # Must match category
        if result_cat != anchor_cat:
            continue
        
        # Check for attribute mismatches
        mismatch_types = []
        
        # Material mismatch
        if (anchor_mat != 'unknown' and 
            result_mat != 'unknown' and 
            anchor_mat != result_mat):
            mismatch_types.append('material')
        
        # Neckline mismatch (important for Step 7!)
        if (anchor_neck != 'unknown' and 
            result_neck != 'unknown' and 
            anchor_neck != result_neck):
            mismatch_types.append('neckline')
        
        # Pattern mismatch
        if (anchor_pat != 'unknown' and 
            result_pat != 'unknown' and 
            anchor_pat != result_pat):
            mismatch_types.append('pattern')
        
        if mismatch_types:
            hard_negs.append({
                'anchor_idx': anchor_idx,
                'neg_idx': result_idx,
                'mismatch_types': ','.join(mismatch_types),
                'retrieval_rank': result['rank'],
                'similarity': result['score'],
                'anchor_category': anchor_cat,
                'neg_category': result_cat,
                'anchor_material': anchor_mat,
                'neg_material': result_mat,
                'anchor_neckline': anchor_neck,
                'neg_neckline': result_neck,
                'anchor_pattern': anchor_pat,
                'neg_pattern': result_pat,
            })
    
    return hard_negs


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MINE HARD NEGATIVES")
    print("=" * 60)
    print(f"Catalog: {args.catalog_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Top-k: {args.top_k}")
    print(f"Max anchors: {args.max_anchors}")
    print("=" * 60)
    
    # Load dataset
    ds = load_dataset_split(args.dataset, args.split)
    
    # Load retrieval engine
    print("\nLoading retrieval engine...")
    engine = RetrievalEngine(args.catalog_dir, exclude_self=False)
    
    # Load embedder for queries
    print("Loading query embedder...")
    embedder = OpenCLIPEmbedder(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device
    )
    
    # Initialize graphic policy
    graphic_policy = GraphicPolicy(remap_to_pattern=True)
    
    # Determine anchors to process
    n_anchors = min(args.max_anchors, len(ds))
    anchor_indices = list(range(n_anchors))
    
    print(f"\nProcessing {n_anchors} anchors...")
    
    # Mine hard negatives
    all_hard_negs = []
    
    for anchor_idx in tqdm(anchor_indices, desc="Mining hard negatives"):
        anchor_item = ds[anchor_idx]
        
        # Encode query
        img = anchor_item['image']
        text = anchor_item.get('text', '')
        
        img_vec = embedder.encode_image(img, normalize=True)
        txt_vec = embedder.encode_text(text, normalize=True) if text else None
        
        # Run retrieval
        results = engine.search(
            img_vec,
            txt_vec,
            query_item_id=anchor_item.get('item_ID'),
            top_k=args.top_k,
            candidate_n=200,
            weight_image=args.weight_image,
            return_metadata=True
        )
        
        # Mine hard negatives from results
        hard_negs = mine_hard_negatives_for_anchor(
            anchor_item,
            anchor_idx,
            results,
            ds,
            graphic_policy
        )
        
        all_hard_negs.extend(hard_negs)
    
    # Convert to dataframe
    if not all_hard_negs:
        print("\n⚠ No hard negatives found!")
        return
    
    df = pd.DataFrame(all_hard_negs)
    
    # Statistics
    print("\n" + "=" * 60)
    print("MINING STATISTICS")
    print("=" * 60)
    print(f"Total anchors processed: {n_anchors}")
    print(f"Total hard negatives mined: {len(df)}")
    print(f"Avg hard negatives per anchor: {len(df) / n_anchors:.1f}")
    print()
    
    # Mismatch type distribution
    print("MISMATCH TYPE DISTRIBUTION:")
    print("-" * 40)
    mismatch_counts = defaultdict(int)
    for types_str in df['mismatch_types']:
        for mtype in types_str.split(','):
            mismatch_counts[mtype] += 1
    
    for mtype, count in sorted(mismatch_counts.items(), key=lambda x: -x[1]):
        pct = count / len(df) * 100
        print(f"  {mtype:12s} {count:>6} ({pct:>5.1f}%)")
    
    print()
    
    # Category distribution
    print("TOP CATEGORIES:")
    print("-" * 40)
    cat_counts = df['anchor_category'].value_counts()
    for cat, count in cat_counts.head(10).items():
        print(f"  {cat:20s} {count:>6}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    
    print(f"\n✓ Hard negatives saved to: {args.output}")
    print(f"  Total pairs: {len(df)}")
    print(f"  Neckline mismatches: {mismatch_counts.get('neckline', 0)}")
    print(f"  Material mismatches: {mismatch_counts.get('material', 0)}")
    print(f"  Pattern mismatches: {mismatch_counts.get('pattern', 0)}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if mismatch_counts.get('neckline', 0) > 100:
        print("\n✓ Good coverage of neckline hard negatives")
        print("  → Will help improve neckline@10 metric")
    else:
        print("\n⚠ Limited neckline hard negatives")
        print("  → Consider mining from more anchors")
    
    if len(df) / n_anchors < 1.0:
        print("\n⚠ Low hard negative rate")
        print("  → Consider:")
        print("    - Increasing top_k")
        print("    - Mining from more diverse categories")


if __name__ == "__main__":
    main()

