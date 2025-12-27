#!/usr/bin/env python
"""
Quick Ablation Study for Step 7

Optimized for fast iteration:
- Frozen encoders (train only projection)
- Partial epochs (0.25-0.5)
- Small dev eval
- Cached results
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
from src.training.sampler import OptimizedItemSampler
from src.training.losses import create_combined_loss
from scripts.step7_train_protocol import (
    build_attribute_label_mappings, prepare_attribute_labels_batch
)


def create_ablation_config(ablation_name: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create configuration for a specific ablation."""
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
            'w_category': 0.5,
            'w_material': 0.1,
            'w_neckline': 0.2,
            'w_pattern': 0.1,
            'w_sleeve': 0.1,
            'description': 'Optimized weights: more category, less material'
        })
    else:
        raise ValueError(f"Unknown ablation: {ablation_name}")

    return config


def quick_dev_eval(model, dataset, n_queries=100):
    """Very fast dev evaluation for ablation studies"""
    model.eval()

    # Sample random queries
    indices = torch.randperm(len(dataset))[:n_queries].tolist()
    queries = [dataset[idx] for idx in indices]

    # Build simple cache for evaluation
    if not hasattr(quick_dev_eval, '_embeddings'):
        print("  Computing embeddings for quick eval...")
        all_embeddings = []
        for i in range(0, len(dataset), 64):  # Larger batches for speed
            batch_items = [dataset[j] for j in range(i, min(i+64, len(dataset)))]
            batch_images = [item['image'] for item in batch_items]
            batch_tensors = torch.stack([model.preprocess(img) for img in batch_images]).to(model.device)

            with torch.no_grad():
                embeddings = model.forward_image(batch_tensors, return_attributes=False)['embedding']
            all_embeddings.append(embeddings.cpu())

        quick_dev_eval._embeddings = torch.cat(all_embeddings)
        quick_dev_eval._embeddings_norm = quick_dev_eval._embeddings / quick_dev_eval._embeddings.norm(dim=1, keepdim=True)

    embeddings = quick_dev_eval._embeddings_norm
    total_correct = 0

    for query_item in queries:
        query_idx = indices[queries.index(query_item)]
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
        similarities[query_idx] = -float('inf')  # Exclude self

        # Top-10 accuracy
        top10_indices = torch.topk(similarities, 10).indices.tolist()
        top10_items = [dataset[idx] for idx in top10_indices]

        if any(item.get('category2') == query_category for item in top10_items):
            total_correct += 1

    return total_correct / n_queries


def run_quick_ablation(ablation_name: str, config: Dict[str, Any], dataset, output_dir: Path):
    """Run a quick ablation experiment"""
    print(f"\n{'='*50}")
    print(f"Quick Ablation: {ablation_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*50}")

    ablation_dir = output_dir / ablation_name
    ablation_dir.mkdir(exist_ok=True)

    # Create model with frozen encoders
    model = ProtocolModel(
        model_name="ViT-B-32",
        pretrained="openai",
        projection_hidden=None,
        use_attribute_heads=True,
        device=config['device']
    )

    # Freeze encoders for faster training
    if config.get('freeze_encoders', False):
        print("Freezing encoders - training only projection layers...")
        for name, param in model.named_parameters():
            if 'projection' not in name and 'attribute' not in name:
                param.requires_grad = False

    # Build attribute mappings
    label_mappings = build_attribute_label_mappings(dataset)

    # Create optimized sampler
    sampler = OptimizedItemSampler(dataset, batch_size=config['batch_size'], seed=config['seed'])

    # Create loss
    criterion = create_combined_loss(
        contrastive_weight=1.0,
        attribute_weight=1.0,
        w_category=config['w_category'],
        w_material=config['w_material'],
        w_neckline=config['w_neckline'],
        w_pattern=config['w_pattern'],
        w_sleeve=config['w_sleeve']
    )

    # Optimizer (only trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config['lr'])

    # Quick training: partial epochs
    results = []

    # CPU optimization: smaller batches and fewer evaluations
    if torch.cuda.is_available():
        batches_per_epoch = min(50, len(dataset) // config['batch_size'])
        eval_frequency = 25
    else:
        # CPU: smaller effective batch size and less frequent eval
        batches_per_epoch = min(20, len(dataset) // config['batch_size'])
        eval_frequency = 50  # Less frequent on CPU

    total_batches = int(config['epochs'] * batches_per_epoch)

    print(f"Training for {total_batches} batches ({config['epochs']} epochs × {batches_per_epoch} batches)")

    model.train()
    global_step = 0

    for batch_idx in range(total_batches):
        # Sample batch
        batch_pairs = sampler.sample_batch()

        if not batch_pairs:
            continue

        # Prepare batch
        anchor_indices = [int(p[0]) for p in batch_pairs]
        positive_indices = [int(p[1]) for p in batch_pairs]

        anchor_items = [dataset[idx] for idx in anchor_indices]
        positive_items = [dataset[idx] for idx in positive_indices]

        # Prepare data
        anchor_images = [item['image'] for item in anchor_items]
        positive_images = [item['image'] for item in positive_items]

        attribute_labels = prepare_attribute_labels_batch(
            anchor_items, label_mappings, model.device
        )

        # Forward pass
        anchor_tensors = torch.stack([model.preprocess(img) for img in anchor_images]).to(model.device)
        positive_tensors = torch.stack([model.preprocess(img) for img in positive_images]).to(model.device)

        anchor_output = model.forward_image(anchor_tensors, return_attributes=True)
        positive_output = model.forward_image(positive_tensors, return_attributes=False)

        losses = criterion(anchor_output, positive_output, attribute_labels=attribute_labels)
        loss = losses['total']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        # Quick eval (frequency based on hardware)
        if batch_idx % eval_frequency == 0:
            print(f"Batch {batch_idx}/{total_batches}: loss={loss.item():.4f}")

            # Skip dev eval on CPU (too slow), just track loss
            if torch.cuda.is_available():
                eval_acc = quick_dev_eval(model, dataset, n_queries=50)
                print(".3f")
            else:
                eval_acc = 0.0  # Placeholder for CPU runs
                print("  (CPU mode - skipping dev eval for speed)")

            results.append({
                'batch': batch_idx,
                'global_step': global_step,
                'loss': loss.item(),
                'contrastive_loss': losses['contrastive'].item(),
                'attribute_loss': losses.get('attribute_total', torch.tensor(0.0)).item(),
                'dev_acc': eval_acc
            })

            results.append({
                'batch': batch_idx,
                'global_step': global_step,
                'loss': loss.item(),
                'contrastive_loss': losses['contrastive'].item(),
                'attribute_loss': losses.get('attribute_total', torch.tensor(0.0)).item(),
                'dev_acc': eval_acc
            })

    # Save results
    with open(ablation_dir / "quick_results.json", 'w') as f:
        json.dump({
            'ablation': ablation_name,
            'config': config,
            'results': results,
            'final_dev_acc': results[-1]['dev_acc'] if results else 0.0
        }, f, indent=2)

    return {
        'ablation': ablation_name,
        'description': config['description'],
        'final_dev_acc': results[-1]['dev_acc'] if results else 0.0,
        'best_dev_acc': max((r['dev_acc'] for r in results if r['dev_acc'] > 0), default=0.0),
        'final_loss': results[-1]['loss'] if results else float('inf'),
        'cpu_mode': not torch.cuda.is_available()
    }


def main():
    parser = argparse.ArgumentParser(description="Quick ablation experiments")

    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--epochs", type=float, default=0.25, help="Partial epochs (e.g., 0.25)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--freeze_encoders", action="store_true", help="Freeze CLIP encoders")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("Quick Step 7 Ablations")
    print("=" * 30)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs} (partial)")
    print(f"Freeze encoders: {args.freeze_encoders}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_from_disk(args.dataset)['train']

    # Base config
    base_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'device': args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        'seed': args.seed,
        'freeze_encoders': args.freeze_encoders,
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
        result = run_quick_ablation(ablation_name, config, dataset, output_dir)
        all_results.append(result)

    # Create comparison table
    print(f"\n{'='*60}")
    print("QUICK ABLATION COMPARISON")
    print(f"{'='*60}")

    # Check if CPU mode
    cpu_mode = any(r.get('cpu_mode', False) for r in all_results)

    print("Ablation".ljust(25), "Description".ljust(30), "Dev@10".ljust(10), "Best@10".ljust(10))
    print("-" * 80)

    for result in all_results:
        name = result['ablation'][:24]
        desc = result['description'][:29]

        if cpu_mode and result['final_dev_acc'] == 0.0:
            final_acc = "N/A (CPU)"
            best_acc = "N/A (CPU)"
        else:
            final_acc = ".3f"
            best_acc = ".3f"

        print(f"{name:<25}{desc:<30}{final_acc:<10}{best_acc:<10}")

    if cpu_mode:
        print("\n📝 Note: CPU mode detected - dev evaluation was skipped for speed")
        print("   Loss curves are still valid for comparing configurations")

    # Save summary
    with open(output_dir / "quick_ablation_comparison.json", 'w') as f:
        json.dump({
            'ablations': all_results,
            'summary_table': "See above print output",
            'note': 'Quick ablation with frozen encoders and partial epochs'
        }, f, indent=2)

    print(f"\nQuick ablation results saved to: {output_dir}")
    print("Use these results to guide full training with the best configuration!")


if __name__ == "__main__":
    main()
