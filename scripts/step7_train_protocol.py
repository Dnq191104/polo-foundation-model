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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.protocol_model import ProtocolModel
from src.training.sampler import BalancedPairSampler
from src.training.losses import create_combined_loss


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
        default=1e-4,
        help="Learning rate (default: 1e-4)"
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use"
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


def train_epoch(
    model: ProtocolModel,
    sampler: BalancedPairSampler,
    dataset,
    criterion,
    optimizer,
    epoch: int,
    n_batches: int,
    log_file: Path
):
    """Train for one epoch."""
    model.train()
    
    epoch_losses = []
    sampler.reset_epoch_stats()
    
    pbar = tqdm(range(n_batches), desc=f"Epoch {epoch}")
    
    for batch_idx in pbar:
        # Sample batch
        batch_df = sampler.sample_batch()
        
        # Load images and texts for anchors and positives/negatives
        anchor_images = []
        other_images = []
        anchor_texts = []
        other_texts = []
        
        for _, row in batch_df.iterrows():
            anchor_idx = int(row['anchor_idx'])
            other_idx = int(row['other_idx'])
            
            anchor_item = dataset[anchor_idx]
            other_item = dataset[other_idx]
            
            anchor_images.append(anchor_item['image'])
            other_images.append(other_item['image'])
            anchor_texts.append(anchor_item.get('text', ''))
            other_texts.append(other_item.get('text', ''))
        
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
        # TODO: Map string labels to indices - for now, skip attribute loss
        attribute_labels = None  # Would need label mappings
        
        # Compute loss
        losses = criterion(
            anchor_output,
            other_output,
            attribute_labels=attribute_labels
        )
        
        loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log
        epoch_losses.append(loss.item())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'cont': losses['contrastive'].item(),
            'attr': losses['attribute_total'].item()
        })
        
        # Log to file
        log_entry = {
            'epoch': epoch,
            'batch': batch_idx,
            'loss': loss.item(),
            'contrastive_loss': losses['contrastive'].item(),
            'attribute_loss': losses['attribute_total'].item(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    # Epoch stats
    avg_loss = np.mean(epoch_losses)
    sampler_stats = sampler.get_epoch_stats()
    
    print(f"\nEpoch {epoch} complete:")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Avg weak categories per batch: {sampler_stats['avg_weak_category_per_batch']:.1f}")
    print(f"  Avg rare materials per batch: {sampler_stats['avg_rare_material_per_batch']:.1f}")
    print(f"  Avg neckline known rate: {sampler_stats['avg_neckline_known_rate']:.2f}")
    
    return avg_loss, sampler_stats


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
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfig saved to: {config_path}")
    
    # Load data
    pair_df = load_pairs(args.pairs)
    dataset = load_dataset_split(args.dataset, args.split)
    
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
        attribute_weight=0.3,
        neckline_weight=2.0
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
    
    n_batches_per_epoch = len(pair_df) // args.batch_size
    
    for epoch in range(1, args.epochs + 1):
        avg_loss, stats = train_epoch(
            model,
            sampler,
            dataset,
            criterion,
            optimizer,
            epoch,
            n_batches_per_epoch,
            log_file
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

