#!/usr/bin/env python
"""
Step 7 Protocol Model Training

Main training script for attribute-aware retrieval fine-tuning.

Usage:
    python scripts/step7_train_protocol.py \
        --pairs artifacts/step7/pairs.parquet \
        --dataset data/processed_v2/hf \
        --output artifacts/step7/runs/run_001 \
        --epochs 10 \
        --batch_size 64
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from datasets import load_from_disk
from tqdm import tqdm

from typing import Dict, List, Optional
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.protocol_model import ProtocolModel
from src.training.sampler import BalancedPairSampler
from src.training.losses import create_combined_loss
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Step 7 protocol model"
    )
    
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to pairs parquet"
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
        help="Output directory for run"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5, lower for stability)"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (fraction of steps for linear warmup)"
    )
    parser.add_argument(
        "--hard_negative_curriculum",
        action='store_true',
        default=True,
        help="Enable hard negative curriculum (start easy, get harder)"
    )
    parser.add_argument(
        "--random_negative_ratio",
        type=float,
        default=0.3,
        help="Fraction of negatives that are random (vs hard)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help="Checkpoint every N epochs"
    )
    # Loss weights
    parser.add_argument(
        "--w_category",
        type=float,
        default=0.2,
        help="Weight for category attribute loss"
    )
    parser.add_argument(
        "--w_material",
        type=float,
        default=0.2,
        help="Weight for material attribute loss"
    )
    parser.add_argument(
        "--w_neckline",
        type=float,
        default=0.1,
        help="Weight for neckline attribute loss"
    )
    parser.add_argument(
        "--w_pattern",
        type=float,
        default=0.1,
        help="Weight for pattern attribute loss"
    )
    parser.add_argument(
        "--w_sleeve",
        type=float,
        default=0.1,
        help="Weight for sleeve attribute loss"
    )
    parser.add_argument(
        "--w_anchor",
        type=float,
        default=0.0,
        help="Weight for anchor-to-baseline regularization (0 = disabled)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use"
    )
    parser.add_argument(
        "--overfit_one_batch_steps",
        type=int,
        default=0,
        help="If > 0, overfit on single batch for N steps (sanity check)"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pairs(pairs_path: str) -> pd.DataFrame:
    """Load pair dataset."""
    print(f"Loading pairs from: {pairs_path}")
    df = pd.read_parquet(pairs_path)
    print(f"Loaded {len(df)} pairs")
    return df


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


def get_git_hash() -> str:
    """Get current git hash."""
    import subprocess
    try:
        hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        return hash
    except:
        return "unknown"


def build_attribute_label_mappings(dataset) -> Dict[str, Dict[str, int]]:
    """
    Build label mappings from string attribute values to indices.

    Args:
        dataset: HuggingFace dataset with attribute columns

    Returns:
        Dict mapping attribute_name -> {string_value: int_index}
    """
    attribute_columns = [
        'attr_material_primary',
        'attr_pattern_primary',
        'attr_neckline_primary',
        'attr_sleeve_primary'
    ]

    mappings = {}

    for attr_col in attribute_columns:
        if attr_col not in dataset.column_names:
            print(f"Warning: {attr_col} not found in dataset")
            continue

        # Collect all unique non-null values
        values = set()
        for item in dataset:
            val = item.get(attr_col)
            if val is not None and val != "unknown":
                values.add(val)

        # Sort for reproducibility
        sorted_values = sorted(values)

        # Create mapping (reserve 0 for unknown/null)
        mapping = {"unknown": 0}
        for i, val in enumerate(sorted_values, 1):
            mapping[val] = i

        attr_name = attr_col.replace('attr_', '').replace('_primary', '')
        mappings[attr_name] = mapping

        print(f"Built mapping for {attr_name}: {len(mapping)} classes")

    return mappings


def prepare_attribute_labels_batch(anchor_batch: List[Dict], label_mappings: Dict[str, Dict[str, int]], device) -> Optional[Dict[str, torch.Tensor]]:
    """
    Prepare attribute labels for a batch of anchor items.

    Args:
        anchor_batch: List of anchor item metadata
        label_mappings: Attribute label mappings
        device: Torch device

    Returns:
        Dict of attribute_name -> tensor of indices, or None if no valid labels
    """
    if not anchor_batch:
        return None

    # Collect labels for each attribute
    attr_labels = defaultdict(list)

    for item in anchor_batch:
        for attr_name, mapping in label_mappings.items():
            col_name = f"attr_{attr_name}_primary"
            val = item.get(col_name, "unknown")

            # Map to index
            idx = mapping.get(val, 0)  # Default to 0 (unknown)
            attr_labels[attr_name].append(idx)

    # Convert to tensors
    result = {}
    for attr_name, indices in attr_labels.items():
        if indices:  # Only include if we have labels
            result[attr_name] = torch.tensor(indices, dtype=torch.long, device=device)

    return result if result else None


def train_epoch(
    model: ProtocolModel,
    sampler: BalancedPairSampler,
    dataset,
    criterion,
    optimizer,
    epoch: int,
    n_batches: int,
    log_file: Path,
    label_mappings: Dict[str, Dict[str, int]],
    hard_negative_curriculum: bool = True,
    random_negative_ratio: float = 0.3
):
    """Train for one epoch."""
    model.train()

    epoch_losses = []
    sampler.reset_epoch_stats()

    # Hard negative curriculum: start easy, get harder
    if hard_negative_curriculum:
        # Curriculum factor: 0.0 (easy) to 1.0 (hard)
        curriculum_factor = min(1.0, epoch / 5.0)  # Fully hard after 5 epochs
    else:
        curriculum_factor = 1.0

    pbar = tqdm(range(n_batches), desc=f"Epoch {epoch} (hardness: {curriculum_factor:.2f})")
    
    for batch_idx in pbar:
        # Sample batch
        batch_df = sampler.sample_batch()
        
        # Load images and texts for anchors and positives/negatives
        anchor_images = []
        other_images = []
        anchor_texts = []
        other_texts = []
        anchor_batch = []  # Metadata for attribute labels

        for _, row in batch_df.iterrows():
            anchor_idx = int(row['anchor_idx'])
            other_idx = int(row['other_idx'])

            anchor_item = dataset[anchor_idx]
            other_item = dataset[other_idx]

            anchor_images.append(anchor_item['image'])
            other_images.append(other_item['image'])
            anchor_texts.append(anchor_item.get('text', ''))
            other_texts.append(other_item.get('text', ''))
            anchor_batch.append(anchor_item)  # Store metadata for attribute labels

        # Preprocess images
        anchor_imgs_tensor = torch.stack([
            model.preprocess(img) for img in anchor_images
        ]).to(model.device)

        other_imgs_tensor = torch.stack([
            model.preprocess(img) for img in other_images
        ]).to(model.device)

        # Forward pass
        anchor_output = model.forward_image(anchor_imgs_tensor, return_attributes=True)
        other_output = model.forward_image(other_imgs_tensor, return_attributes=False)

        # Prepare attribute labels (from anchor metadata)
        attribute_labels = prepare_attribute_labels_batch(anchor_batch, label_mappings, model.device)
        
        # Compute loss
        losses = criterion(
            anchor_output,
            other_output,
            attribute_labels=attribute_labels
        )

        loss = losses['total']

        # HARD ASSERTIONS: Ensure attribute losses are non-zero when use_attribute_heads=True
        if model.use_attribute_heads:
            attr_total = losses.get('attribute_total', torch.tensor(0.0))
            assert attr_total is not None, "attribute_total loss is None when use_attribute_heads=True"

            # Check that attribute loss is non-zero at least once in first few batches
            if epoch == 0 and batch_idx < 5:
                attr_val = attr_total.item()
                assert attr_val > 0, f"attribute_total loss is zero ({attr_val}) in early training when use_attribute_heads=True"

                # Also check individual attribute losses are present
                attr_loss_keys = [k for k in losses.keys() if k.startswith('attribute_') and k != 'attribute_total']
                assert len(attr_loss_keys) > 0, "No individual attribute losses found when use_attribute_heads=True"

        # Per-head logging (every 10 batches)
        if batch_idx % 10 == 0:
            log_entry = {
                'epoch': epoch,
                'batch': batch_idx,
                'loss': loss.item(),
                'contrastive_loss': losses['contrastive'].item(),
                'attribute_loss': losses.get('attribute_total', torch.tensor(0.0)).item(),
                'learning_rate': scheduler.get_last_lr()[0],
                'timestamp': datetime.now().isoformat()
            }

            # Add per-head losses and accuracies
            if 'attributes' in anchor_output and attribute_labels:
                attr_preds = anchor_output['attributes']
                for attr_name in ['material', 'pattern', 'neckline', 'sleeve']:
                    if attr_name in attr_preds and attr_name in attribute_labels:
                        # Loss
                        loss_key = f'attribute_{attr_name}'
                        if loss_key in losses:
                            log_entry[loss_key] = losses[loss_key].item()

                        # Batch accuracy (approx)
                        preds = attr_preds[attr_name].argmax(dim=1)
                        targets = attribute_labels[attr_name]
                        accuracy = (preds == targets).float().mean().item()
                        log_entry[f'{attr_name}_acc'] = accuracy

            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        # Log
        epoch_losses.append(loss.item())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'cont': losses['contrastive'].item(),
            'attr': losses['attribute_total'].item()
        })
        
    
    # Epoch stats
    avg_loss = np.mean(epoch_losses)
    sampler_stats = sampler.get_epoch_stats()
    
    print(f"\nEpoch {epoch} complete:")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Avg weak categories per batch: {sampler_stats['avg_weak_category_per_batch']:.1f}")
    print(f"  Avg rare materials per batch: {sampler_stats['avg_rare_material_per_batch']:.1f}")
    print(f"  Avg neckline known rate: {sampler_stats['avg_neckline_known_rate']:.2f}")

    # Early stopping check (after first few epochs)
    if epoch >= 2:
        # Simple heuristic: if loss is not decreasing, might be overfitting
        # In a real implementation, you'd do a mini-evaluation here
        pass  # TODO: Add proper mini-eval and early stopping

    return avg_loss, sampler_stats


def run_overfit_test(
    model: ProtocolModel,
    sampler: BalancedPairSampler,
    dataset,
    criterion,
    optimizer,
    n_steps: int,
    label_mappings: Dict[str, Dict[str, int]],
    output_dir: Path
):
    """
    Run single-batch overfit test to verify training setup.

    Freezes to one batch and trains for n_steps, checking that:
    - Total loss decreases
    - Attribute losses decrease (if enabled)
    - Batch accuracies increase (if attributes enabled)
    """
    print("Setting up overfit test...")

    # Get one batch and freeze it
    sampler_iter = iter(sampler)
    anchor_batch, other_batch, _ = next(sampler_iter)

    # Extract data for this fixed batch
    anchor_images = [dataset[idx]['image'] for idx in anchor_batch]
    anchor_batch_data = [dataset[idx] for idx in anchor_batch]
    other_images = [dataset[idx]['image'] for idx in other_batch]

    # Preprocess once (fixed for all steps)
    anchor_imgs_tensor = torch.stack([
        model.preprocess(img) for img in anchor_images
    ]).to(model.device)

    other_imgs_tensor = torch.stack([
        model.preprocess(img) for img in other_images
    ]).to(model.device)

    # Prepare attribute labels once
    attribute_labels = prepare_attribute_labels_batch(anchor_batch_data, label_mappings, model.device)

    # Track metrics over steps
    metrics_history = []

    print(f"Running overfit test for {n_steps} steps...")
    for step in range(n_steps):
        # Forward pass
        anchor_output = model.forward_image(anchor_imgs_tensor, return_attributes=True)
        other_output = model.forward_image(other_imgs_tensor, return_attributes=False)

        # Compute loss
        losses = criterion(
            anchor_output,
            other_output,
            attribute_labels=attribute_labels
        )

        loss = losses['total']

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect metrics
        step_metrics = {
            'step': step,
            'total_loss': loss.item(),
            'contrastive_loss': losses['contrastive'].item(),
            'attribute_loss': losses.get('attribute_total', torch.tensor(0.0)).item(),
        }

        # Add per-head losses and accuracies
        if 'attributes' in anchor_output and attribute_labels:
            attr_preds = anchor_output['attributes']
            for attr_name in ['material', 'pattern', 'neckline', 'sleeve']:
                if attr_name in attr_preds and attr_name in attribute_labels:
                    # Loss
                    loss_key = f'attribute_{attr_name}'
                    if loss_key in losses:
                        step_metrics[loss_key] = losses[loss_key].item()

                    # Batch accuracy
                    preds = attr_preds[attr_name].argmax(dim=1)
                    targets = attribute_labels[attr_name]
                    accuracy = (preds == targets).float().mean().item()
                    step_metrics[f'{attr_name}_acc'] = accuracy

        metrics_history.append(step_metrics)

        if step % 10 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, attr={step_metrics['attribute_loss']:.4f}")

    # Analyze results
    results = analyze_overfit_results(metrics_history, model.use_attribute_heads)

    # Save results
    results_path = output_dir / "overfit_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Overfit test results saved to: {results_path}")

    # Print summary
    print("\n" + "="*50)
    print("OVERFIT TEST RESULTS")
    print("="*50)
    print(f"Total loss: {results['total_loss_decreased']}")
    print(f"Attribute loss: {results['attribute_loss_decreased']}")
    print(f"Attribute accuracies: {results['attribute_accuracies_improved']}")
    print(f"Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
    print("="*50)

    if not results['overall_pass']:
        raise RuntimeError("Overfit test failed! Check training setup before running full training.")


def analyze_overfit_results(metrics_history: List[Dict], use_attribute_heads: bool) -> Dict:
    """
    Analyze overfit test results to check training health.
    """
    if len(metrics_history) < 2:
        return {'overall_pass': False, 'error': 'Not enough steps'}

    first = metrics_history[0]
    last = metrics_history[-1]

    results = {
        'total_loss_decreased': last['total_loss'] < first['total_loss'],
        'initial_loss': first['total_loss'],
        'final_loss': last['total_loss'],
        'loss_reduction': first['total_loss'] - last['total_loss'],
    }

    if use_attribute_heads:
        results['attribute_loss_decreased'] = last['attribute_loss'] < first['attribute_loss']
        results['initial_attr_loss'] = first['attribute_loss']
        results['final_attr_loss'] = last['attribute_loss']

        # Check attribute accuracies improved
        acc_improved = []
        for attr in ['material', 'pattern', 'neckline', 'sleeve']:
            acc_key = f'{attr}_acc'
            if acc_key in first and acc_key in last:
                improved = last[acc_key] > first[acc_key]
                acc_improved.append(improved)
                results[f'{attr}_acc_improved'] = improved
                results[f'{attr}_acc_initial'] = first[acc_key]
                results[f'{attr}_acc_final'] = last[acc_key]

        results['attribute_accuracies_improved'] = all(acc_improved) if acc_improved else False
        results['overall_pass'] = (
            results['total_loss_decreased'] and
            results['attribute_loss_decreased'] and
            results['attribute_accuracies_improved']
        )
    else:
        results['overall_pass'] = results['total_loss_decreased']

    return results


def main():
    args = parse_args()
    
    print("=" * 60)
    print("STEP 7 PROTOCOL MODEL TRAINING")
    print("=" * 60)
    print(f"Pairs: {args.pairs}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save config
    config = {
        'pairs': args.pairs,
        'dataset': args.dataset,
        'split': args.split,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'git_hash': get_git_hash(),
        'timestamp': datetime.now().isoformat(),
        'loss_weights': {
            'w_category': args.w_category,
            'w_material': args.w_material,
            'w_neckline': args.w_neckline,
            'w_pattern': args.w_pattern,
            'w_sleeve': args.w_sleeve,
            'w_anchor': args.w_anchor,
        }
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfig saved to: {config_path}")
    
    # Load data
    pair_df = load_pairs(args.pairs)
    dataset = load_dataset_split(args.dataset, args.split)

    # Build attribute label mappings
    print("\nBuilding attribute label mappings...")
    label_mappings = build_attribute_label_mappings(dataset)
    print(f"Label mappings built for {len(label_mappings)} attributes")

    # Initialize model
    print("\nInitializing model...")
    model = ProtocolModel(
        model_name="ViT-B-32",
        pretrained="openai",
        projection_hidden=None,  # Use linear projections for reliable checkpoint loading
        use_attribute_heads=True,
        device=args.device
    )
    
    # Initialize sampler
    print("Initializing sampler...")
    sampler = BalancedPairSampler(
        pair_df=pair_df,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Initialize loss and optimizer
    print("Initializing loss and optimizer...")
    criterion = create_combined_loss(
        contrastive_weight=1.0,
        attribute_weight=1.0,  # Total attribute weight (individual weights control distribution)
        anchor_weight=args.w_anchor,
        w_category=args.w_category,
        w_material=args.w_material,
        w_neckline=args.w_neckline,
        w_pattern=args.w_pattern,
        w_sleeve=args.w_sleeve
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # Training log
    log_file = output_dir / "train_log.jsonl"
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    # Single-batch overfit test mode
    if args.overfit_one_batch_steps > 0:
        print(f"\nRunning single-batch overfit test for {args.overfit_one_batch_steps} steps...")
        run_overfit_test(
            model=model,
            sampler=sampler,
            dataset=dataset,
            criterion=criterion,
            optimizer=optimizer,
            n_steps=args.overfit_one_batch_steps,
            label_mappings=label_mappings,
            output_dir=output_dir
        )
        print("\nOverfit test completed. Check overfit_results.json for results.")
        return

    n_batches_per_epoch = len(pair_df) // args.batch_size

    # Learning rate scheduler (warmup + cosine decay)
    total_steps = args.epochs * n_batches_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(1, args.epochs + 1):
        avg_loss, stats = train_epoch(
            model,
            sampler,
            dataset,
            criterion,
            optimizer,
            epoch,
            n_batches_per_epoch,
            log_file,
            label_mappings,
            args.hard_negative_curriculum,
            args.random_negative_ratio
        )
        
        # Save checkpoint
        if epoch % args.checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
            model.save_checkpoint(
                str(checkpoint_path),
                epoch=epoch,
                optimizer_state=optimizer.state_dict()
            )
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint = checkpoint_dir / "final.pt"
    model.save_checkpoint(
        str(final_checkpoint),
        epoch=args.epochs,
        optimizer_state=optimizer.state_dict()
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Training log: {log_file}")
    print("\nNext steps:")
    print("  1. Run validation: scripts/step7_validate_checkpoint.py")
    print("  2. Run full eval: scripts/step7_eval_full.py")


if __name__ == "__main__":
    main()

