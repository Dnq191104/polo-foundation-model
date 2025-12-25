#!/usr/bin/env python
"""
Validate Step 7 Checkpoint

Quick validation during training using mini catalog and fixed queries.

Usage:
    python scripts/step7_validate_checkpoint.py \
        --checkpoint artifacts/step7/runs/run_001/checkpoints/epoch_5.pt \
        --catalog_dir artifacts/retrieval/openclip_vitb32_v0 \
        --query_dataset data/processed_v2/hf \
        --output artifacts/step7/runs/run_001/validation/epoch_5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.protocol_model import ProtocolModel
from src.utils.scoreboard import Step7Scoreboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Step 7 checkpoint"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to validate"
    )
    parser.add_argument(
        "--catalog_dir",
        type=str,
        required=True,
        help="Path to Step 6 baseline catalog (for comparison)"
    )
    parser.add_argument(
        "--query_dataset",
        type=str,
        required=True,
        help="Path to query dataset"
    )
    parser.add_argument(
        "--query_split",
        type=str,
        default="validation",
        help="Query split (default: validation)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for validation results"
    )
    parser.add_argument(
        "--n_catalog",
        type=int,
        default=5000,
        help="Number of catalog items to encode (default: 5000)"
    )
    parser.add_argument(
        "--n_queries",
        type=int,
        default=500,
        help="Number of queries to evaluate (default: 500)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use"
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


def build_mini_catalog(model: ProtocolModel, dataset, n_items: int):
    """Build mini catalog embeddings."""
    print(f"\nBuilding mini catalog ({n_items} items)...")
    
    img_embeddings = []
    txt_embeddings = []
    metadata = []
    
    for i in tqdm(range(min(n_items, len(dataset))), desc="Encoding catalog"):
        item = dataset[i]
        
        # Encode
        img_emb = model.encode_image_numpy(item['image'])
        txt_emb = model.encode_text_numpy(item.get('text', ''))
        
        img_embeddings.append(img_emb)
        txt_embeddings.append(txt_emb)
        
        metadata.append({
            'item_ID': item.get('item_ID', ''),
            'category2': item.get('category2', ''),
            'attr_material_primary': item.get('attr_material_primary', 'unknown'),
            'attr_neckline_primary': item.get('attr_neckline_primary', 'unknown'),
        })
    
    img_embeddings = np.array(img_embeddings)
    txt_embeddings = np.array(txt_embeddings)
    
    print(f"Mini catalog built: {img_embeddings.shape}")
    
    return {
        'img_embeddings': img_embeddings,
        'txt_embeddings': txt_embeddings,
        'metadata': metadata
    }


def evaluate_mini_catalog(
    model: ProtocolModel,
    catalog: dict,
    query_dataset,
    n_queries: int,
    weight_image: float = 0.7,
    top_k: int = 10
):
    """Evaluate on mini catalog."""
    print(f"\nEvaluating on {n_queries} queries...")
    
    n_queries = min(n_queries, len(query_dataset))
    
    # Simple metrics
    category_matches = []
    neckline_matches = []
    material_matches = []
    
    for i in tqdm(range(n_queries), desc="Running queries"):
        query_item = query_dataset[i]
        
        # Encode query
        query_img_emb = model.encode_image_numpy(query_item['image'])
        query_txt_emb = model.encode_text_numpy(query_item.get('text', ''))
        
        # Compute similarities
        img_sims = catalog['img_embeddings'] @ query_img_emb
        txt_sims = catalog['txt_embeddings'] @ query_txt_emb
        
        fused_sims = weight_image * img_sims + (1 - weight_image) * txt_sims
        
        # Get top-k
        top_indices = np.argsort(fused_sims)[::-1][:top_k]
        
        # Check matches
        query_cat = query_item.get('category2', '')
        query_neck = query_item.get('attr_neckline_primary', 'unknown')
        query_mat = query_item.get('attr_material_primary', 'unknown')
        
        cat_match = False
        neck_match = False
        mat_match = False
        
        for idx in top_indices:
            result_meta = catalog['metadata'][idx]
            
            if result_meta['category2'] == query_cat:
                cat_match = True
            
            if query_neck != 'unknown' and result_meta['attr_neckline_primary'] == query_neck:
                neck_match = True
            
            if query_mat != 'unknown' and result_meta['attr_material_primary'] == query_mat:
                mat_match = True
        
        category_matches.append(cat_match)
        
        if query_neck != 'unknown':
            neckline_matches.append(neck_match)
        
        if query_mat != 'unknown':
            material_matches.append(mat_match)
    
    # Compute metrics
    metrics = {
        f'category_match@{top_k}': np.mean(category_matches) if category_matches else 0,
        f'neckline_match@{top_k}': np.mean(neckline_matches) if neckline_matches else 0,
        f'material_match@{top_k}_known_only': np.mean(material_matches) if material_matches else 0,
        'n_queries': n_queries,
        'n_neckline_known': len(neckline_matches),
        'n_material_known': len(material_matches),
    }
    
    return metrics


def main():
    args = parse_args()
    
    print("=" * 60)
    print("VALIDATE STEP 7 CHECKPOINT")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Mini catalog size: {args.n_catalog}")
    print(f"Num queries: {args.n_queries}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading checkpoint...")
    model = ProtocolModel.load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    
    # Load dataset
    catalog_dataset = load_dataset_split(args.query_dataset, "train")
    query_dataset = load_dataset_split(args.query_dataset, args.query_split)
    
    # Build mini catalog
    catalog = build_mini_catalog(model, catalog_dataset, args.n_catalog)
    
    # Evaluate
    metrics = evaluate_mini_catalog(
        model,
        catalog,
        query_dataset,
        args.n_queries
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    for metric_name, value in metrics.items():
        if metric_name.startswith('n_'):
            print(f"{metric_name}: {value}")
        else:
            print(f"{metric_name}: {value * 100:.1f}%")
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Quick comparison to baseline (if baseline metrics available)
    baseline_metrics_path = Path(args.catalog_dir) / "eval" / "metrics.json"
    
    if baseline_metrics_path.exists():
        print("\n" + "=" * 60)
        print("COMPARISON TO BASELINE")
        print("=" * 60)
        
        try:
            scoreboard = Step7Scoreboard(str(baseline_metrics_path))
            
            # Wrap metrics in expected format
            wrapped_metrics = {
                'metrics': {
                    'overall': metrics
                }
            }
            
            # Save wrapped format
            temp_checkpoint_metrics = output_dir / "metrics_wrapped.json"
            with open(temp_checkpoint_metrics, 'w') as f:
                json.dump(wrapped_metrics, f, indent=2)
            
            # Compute deltas (will show "N/A" for missing slices, but overall will work)
            deltas = scoreboard.compute_deltas(str(temp_checkpoint_metrics))
            
            print(f"\nNeckline delta: {deltas['overall']['neckline']['delta']:+.1f}%")
            print(f"Material delta: {deltas['overall']['material_known']['delta']:+.1f}%")
            print(f"Category delta: {deltas['overall']['category']['delta']:+.1f}%")
            
        except Exception as e:
            print(f"Could not compute baseline comparison: {e}")


if __name__ == "__main__":
    main()

