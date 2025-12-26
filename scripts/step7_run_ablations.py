#!/usr/bin/env python
"""
Step 7 Ablations Runner

Runs quick ablation experiments to compare training strategies:
- Contrastive only
- Contrastive + category only
- Contrastive + all attributes

Each runs for 1-2 epochs with mini-evaluation.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import torch
from datasets import load_from_disk

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.protocol_model import ProtocolModel
from src.training.sampler import BalancedPairSampler
from src.training.losses import create_combined_loss
from scripts.step7_train_protocol import load_pairs, load_dataset_split


def create_ablation_config(
    ablation_name: str,
    base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create configuration for a specific ablation.
    """
    config = base_config.copy()

    if ablation_name == "contrastive_only":
        config.update({
            'w_category': 0.0,
            'w_material': 0.0,
            'w_neckline': 0.0,
            'w_pattern': 0.0,
            'w_sleeve': 0.0,
            'description': 'Contrastive loss only'
        })
    elif ablation_name == "contrastive_plus_category":
        config.update({
            'w_category': 0.5,
            'w_material': 0.0,
            'w_neckline': 0.0,
            'w_pattern': 0.0,
            'w_sleeve': 0.0,
            'description': 'Contrastive + category only'
        })
    elif ablation_name == "contrastive_plus_all":
        config.update({
            'w_category': 0.2,
            'w_material': 0.2,
            'w_neckline': 0.1,
            'w_pattern': 0.1,
            'w_sleeve': 0.1,
            'description': 'Contrastive + all attributes'
        })
    elif ablation_name == "contrastive_optimized":
        config.update({
            'w_category': 0.5,    # Increased from 0.2
            'w_material': 0.1,    # Decreased from 0.2
            'w_neckline': 0.2,    # Increased from 0.1
            'w_pattern': 0.1,     # Keep same
            'w_sleeve': 0.1,      # Keep same
            'description': 'Optimized weights: more category, less material'
        })
    else:
        raise ValueError(f"Unknown ablation: {ablation_name}")

    return config


def run_mini_eval(
    model: ProtocolModel,
    dataset,
    n_queries: int = 100
) -> Dict[str, float]:
    """
    Run a very quick evaluation on a small subset.
    """
    model.eval()

    # Sample random queries
    indices = torch.randperm(len(dataset))[:n_queries].tolist()
    queries = [dataset[idx] for idx in indices]

    # Get embeddings for all items (cache for speed)
    if not hasattr(run_mini_eval, '_embeddings'):
        print("Computing embeddings for mini-eval...")
        all_embeddings = []
        for i in range(0, len(dataset), 32):  # Batch process
            batch_items = [dataset[j] for j in range(i, min(i+32, len(dataset)))]
            batch_images = [item['image'] for item in batch_items]
            batch_tensors = torch.stack([model.preprocess(img) for img in batch_images]).to(model.device)
            with torch.no_grad():
                embeddings = model.forward_image(batch_tensors, return_attributes=False)['embedding']
            all_embeddings.append(embeddings.cpu())
        run_mini_eval._embeddings = torch.cat(all_embeddings, dim=0)
        run_mini_eval._embeddings_norm = run_mini_eval._embeddings / run_mini_eval._embeddings.norm(dim=1, keepdim=True)

    embeddings = run_mini_eval._embeddings_norm

    # Evaluate queries
    total_correct = 0
    category_correct = 0
    material_correct = 0

    for query_idx, query_item in zip(indices, queries):
        query_category = query_item.get('category2', '')

        # Get query embedding
        query_img = query_item['image']
        query_tensor = model.preprocess(query_img).unsqueeze(0).to(model.device)
        with torch.no_grad():
            query_emb = model.forward_image(query_tensor, return_attributes=False)['embedding']
        query_emb = query_emb / query_emb.norm(dim=1, keepdim=True)
        query_emb = query_emb.cpu()

        # Compute similarities
        similarities = (query_emb @ embeddings.T).squeeze()

        # Get top-10 (excluding self)
        similarities = similarities.clone()
        similarities[query_idx] = -float('inf')  # Exclude self
        top10_indices = torch.topk(similarities, 10).indices.tolist()

        # Check if correct category in top-10
        top10_items = [dataset[idx] for idx in top10_indices]

        # Category match (only evaluation metric for mini-eval)
        if any(item.get('category2') == query_category for item in top10_items):
            category_correct += 1
            total_correct += 1

    return {
        'total_acc': total_correct / n_queries,
        'category_acc': category_correct / n_queries,
        'material_acc': 0.0  # Not evaluated in mini-eval
    }


def run_ablation(
    ablation_name: str,
    config: Dict[str, Any],
    pair_df: pd.DataFrame,
    dataset,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run a single ablation experiment.
    """
    print(f"\n{'='*50}")
    print(f"Running ablation: {ablation_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*50}")

    ablation_dir = output_dir / ablation_name
    ablation_dir.mkdir(exist_ok=True)

    # Create model
    model = ProtocolModel(
        model_name="ViT-B-32",
        pretrained="openai",
        projection_hidden=None,
        use_attribute_heads=True,
        device=config['device']
    )

    # Create loss
    criterion = create_combined_loss(
        contrastive_weight=1.0,
        attribute_weight=1.0,
        anchor_weight=config.get('w_anchor', 0.0),
        w_category=config['w_category'],
        w_material=config['w_material'],
        w_neckline=config['w_neckline'],
        w_pattern=config['w_pattern'],
        w_sleeve=config['w_sleeve']
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    # Create sampler
    sampler = BalancedPairSampler(
        pair_df=pair_df,
        batch_size=config['batch_size'],
        seed=config['seed']
    )

    # Run 1-2 epochs
    results = []
    n_batches_per_epoch = min(50, len(pair_df) // config['batch_size'])  # Quick test

    for epoch in range(1, config['epochs'] + 1):
        print(f"Epoch {epoch}/{config['epochs']}")

        model.train()
        epoch_losses = []

        for batch_idx in range(n_batches_per_epoch):
            batch_df = sampler.sample_batch()

            # Simplified training (just positive pairs for contrastive learning)
            # Filter for positive pairs only (label=1)
            positive_pairs = batch_df[batch_df['label'] == 1].head(8)

            if len(positive_pairs) == 0:
                continue  # Skip if no positive pairs in this batch

            anchor_indices = []
            positive_indices = []

            for _, row in positive_pairs.iterrows():
                anchor_idx = int(row['anchor_idx'])
                other_idx = int(row['other_idx'])

                anchor_indices.append(anchor_idx)
                positive_indices.append(other_idx)

            # Get images
            anchor_images = [dataset[idx]['image'] for idx in anchor_indices]
            positive_images = [dataset[idx]['image'] for idx in positive_indices]

            # Forward pass
            anchor_tensors = torch.stack([model.preprocess(img) for img in anchor_images]).to(model.device)
            positive_tensors = torch.stack([model.preprocess(img) for img in positive_images]).to(model.device)

            anchor_output = model.forward_image(anchor_tensors, return_attributes=True)
            positive_output = model.forward_image(positive_tensors, return_attributes=False)

            # Loss
            losses = criterion(anchor_output, positive_output)
            loss = losses['total']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(".4f")

        # Mini eval
        eval_results = run_mini_eval(model, dataset, n_queries=50)
        eval_results['epoch'] = epoch
        eval_results['avg_loss'] = avg_loss
        results.append(eval_results)

        print(".3f"
              ".3f")

    # Save results
    with open(ablation_dir / "results.json", 'w') as f:
        json.dump({
            'ablation': ablation_name,
            'config': config,
            'results': results
        }, f, indent=2)

    return {
        'ablation': ablation_name,
        'description': config['description'],
        'final_results': results[-1] if results else {}
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Step 7 ablation experiments"
    )

    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to pairs.parquet"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Epochs per ablation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print("Step 7 Ablations Runner")
    print("=" * 30)
    print(f"Pairs: {args.pairs}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs per ablation: {args.epochs}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    pair_df = load_pairs(args.pairs)
    dataset = load_dataset_split(args.dataset, "train")

    # Base config
    base_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'device': args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        'seed': args.seed,
        'w_anchor': 0.0
    }

    # Define ablations
    ablations = [
        "contrastive_only",
        "contrastive_plus_category",
        "contrastive_plus_all",
        "contrastive_optimized"
    ]

    # Run all ablations
    all_results = []

    for ablation_name in ablations:
        config = create_ablation_config(ablation_name, base_config)
        result = run_ablation(ablation_name, config, pair_df, dataset, output_dir)
        all_results.append(result)

    # Create comparison table
    print(f"\n{'='*60}")
    print("ABLATION COMPARISON")
    print(f"{'='*60}")

    print("Ablation".ljust(25), "Description".ljust(25), "Total@10".ljust(10), "Cat@10".ljust(10), "Mat@10".ljust(10))
    print("-" * 80)

    for result in all_results:
        name = result['ablation'][:24]
        desc = result['description'][:24]
        final = result.get('final_results', {})

        total = ".3f"
        cat = ".3f"
        mat = ".3f"

        print(f"{name:<25}{desc:<25}{total:<10}{cat:<10}{mat:<10}")

    # Save summary
    with open(output_dir / "ablation_comparison.json", 'w') as f:
        json.dump({
            'ablations': all_results,
            'summary_table': "See above print output"
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
