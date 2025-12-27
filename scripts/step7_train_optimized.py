#!/usr/bin/env python
"""
Optimized Step 7 Training with Curriculum Learning

Key optimizations:
- On-the-fly balanced sampling (no pre-generated pairs)
- Curriculum scheduling within single run
- Mixed precision training
- Gradient accumulation
- Dev eval for fast iteration
- Comprehensive caching
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from datasets import load_from_disk
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.protocol_model import ProtocolModel
from src.training.sampler import OptimizedItemSampler
from src.training.losses import create_combined_loss
from scripts.step7_train_protocol import (
    load_dataset_split, build_attribute_label_mappings,
    prepare_attribute_labels_batch, get_git_hash
)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Step 7 training")

    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--curriculum_schedule", action="store_true", help="Use curriculum scheduling")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Checkpoint every N epochs")
    parser.add_argument("--dev_eval_every", type=int, default=2, help="Dev eval every N epochs")
    parser.add_argument("--dev_eval_ids", type=str, help="Path to dev eval IDs")

    # Loss weights
    parser.add_argument("--w_category", type=float, default=0.5, help="Category weight")
    parser.add_argument("--w_material", type=float, default=0.1, help="Material weight")
    parser.add_argument("--w_neckline", type=float, default=0.2, help="Neckline weight")
    parser.add_argument("--w_pattern", type=float, default=0.1, help="Pattern weight")
    parser.add_argument("--w_sleeve", type=float, default=0.1, help="Sleeve weight")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def create_dev_eval_set(dataset, n_queries=500, seed=42):
    """Create stratified dev eval set"""
    np.random.seed(seed)
    dev_queries = []

    # Group by category
    category_items = {}
    for i, item in enumerate(dataset):
        cat = item.get('category2', 'unknown')
        if cat not in category_items:
            category_items[cat] = []
        category_items[cat].append(i)

    # Sample proportionally from each category
    total_samples = 0
    for cat, ids in category_items.items():
        n_cat_samples = max(5, int(n_queries * len(ids) / len(dataset)))
        sampled_ids = np.random.choice(ids, min(n_cat_samples, len(ids)), replace=False)
        dev_queries.extend(sampled_ids.tolist())
        total_samples += len(sampled_ids)

    # Trim to exact size
    dev_queries = dev_queries[:n_queries]

    return dev_queries


def dev_evaluate(model, dataset, dev_ids, device):
    """Fast dev evaluation"""
    model.eval()

    # Sample queries
    queries = [dataset[idx] for idx in dev_ids[:100]]  # Small subset for speed

    # Build simple index for evaluation
    all_embeddings = []
    batch_size = 32

    print("Computing dev embeddings...")
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_items = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]
            batch_images = [item['image'] for item in batch_items]
            batch_tensors = torch.stack([model.preprocess(img) for img in batch_images]).to(device)

            embeddings = model.forward_image(batch_tensors, return_attributes=False)['embedding']
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings)
    embeddings_norm = all_embeddings / all_embeddings.norm(dim=1, keepdim=True)

    # Evaluate
    total_correct = 0
    category_correct = 0

    for query_item in queries:
        query_img = query_item['image']
        query_tensor = model.preprocess(query_img).unsqueeze(0).to(device)

        with torch.no_grad():
            query_emb = model.forward_image(query_tensor, return_attributes=False)['embedding']
        query_emb = query_emb / query_emb.norm(dim=1, keepdim=True)
        query_emb = query_emb.cpu()

        # Find nearest neighbors
        similarities = (query_emb @ embeddings_norm.T).squeeze()
        query_idx = dev_ids[queries.index(query_item)] if query_item in queries else 0
        similarities[query_idx] = -float('inf')  # Exclude self

        top10_indices = torch.topk(similarities, 10).indices.tolist()
        top10_items = [dataset[idx] for idx in top10_indices]

        # Check category match
        query_category = query_item.get('category2', '')
        if any(item.get('category2') == query_category for item in top10_items):
            category_correct += 1
            total_correct += 1

    return {
        'total_acc': total_correct / len(queries),
        'category_acc': category_correct / len(queries)
    }


def train_optimized(args):
    """Optimized training with curriculum learning"""

    print("=" * 60)
    print("OPTIMIZED STEP 7 TRAINING")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation})")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Curriculum: {args.curriculum_schedule}")
    print("=" * 60)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_from_disk(args.dataset)['train']

    # Build attribute mappings
    label_mappings = build_attribute_label_mappings(dataset)

    # Create optimized sampler
    sampler = OptimizedItemSampler(dataset, batch_size=args.batch_size, seed=args.seed)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProtocolModel(
        model_name="ViT-B-32",
        pretrained="openai",
        projection_hidden=None,
        use_attribute_heads=True,
        device=device
    )

    # Create loss function
    criterion = create_combined_loss(
        contrastive_weight=1.0,
        attribute_weight=1.0,
        w_category=args.w_category,
        w_material=args.w_material,
        w_neckline=args.w_neckline,
        w_pattern=args.w_pattern,
        w_sleeve=args.w_sleeve
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Mixed precision
    scaler = GradScaler() if args.mixed_precision else None

    # Learning rate scheduler (warmup + cosine)
    total_steps = args.epochs * (len(dataset) // args.batch_size)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create dev eval set if needed
    if args.dev_eval_ids and os.path.exists(args.dev_eval_ids):
        with open(args.dev_eval_ids, 'r') as f:
            dev_ids = json.load(f)
    else:
        print("Creating dev eval set...")
        dev_ids = create_dev_eval_set(dataset, n_queries=500, seed=args.seed)
        if args.dev_eval_ids:
            with open(args.dev_eval_ids, 'w') as f:
                json.dump(dev_ids, f)

    # Training log
    log_file = output_dir / "train_log.jsonl"
    global_step = 0

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):

        # Update curriculum weights if enabled
        if args.curriculum_schedule:
            sampler.update_curriculum_weights(epoch, args.epochs)

        model.train()
        epoch_losses = []
        accumulation_step = 0
        optimizer.zero_grad()

        # Progress bar
        n_batches = len(dataset) // args.batch_size
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch}/{args.epochs}")

        for batch_idx in pbar:
            # Sample batch
            batch_pairs = sampler.sample_batch()

            if not batch_pairs:
                continue

            # Prepare batch data
            anchor_indices = [p[0] for p in batch_pairs]
            positive_indices = [p[1] for p in batch_pairs]
            labels = [p[2] for p in batch_pairs]

            # Get items
            anchor_items = [dataset[idx] for idx in anchor_indices]
            positive_items = [dataset[idx] for idx in positive_indices]

            # Prepare images
            anchor_images = [item['image'] for item in anchor_items]
            positive_images = [item['image'] for item in positive_items]

            # Prepare attribute labels for anchors
            anchor_batch_metadata = anchor_items
            attribute_labels = prepare_attribute_labels_batch(
                anchor_batch_metadata, label_mappings, device
            )

            # Forward pass
            anchor_tensors = torch.stack([model.preprocess(img) for img in anchor_images]).to(device)
            positive_tensors = torch.stack([model.preprocess(img) for img in positive_images]).to(device)

            if args.mixed_precision:
                with autocast():
                    anchor_output = model.forward_image(anchor_tensors, return_attributes=True)
                    positive_output = model.forward_image(positive_tensors, return_attributes=False)

                    losses = criterion(anchor_output, positive_output, attribute_labels=attribute_labels)
                    loss = losses['total'] / args.gradient_accumulation
            else:
                anchor_output = model.forward_image(anchor_tensors, return_attributes=True)
                positive_output = model.forward_image(positive_tensors, return_attributes=False)

                losses = criterion(anchor_output, positive_output, attribute_labels=attribute_labels)
                loss = losses['total'] / args.gradient_accumulation

            # Backward pass
            if args.mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_step += 1

            # Update weights every N steps
            if accumulation_step % args.gradient_accumulation == 0:
                if args.mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_losses.append(loss.item() * args.gradient_accumulation)

            # Update progress bar
            avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

            # Log every 50 steps
            if batch_idx % 50 == 0:
                log_entry = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'global_step': global_step,
                    'loss': loss.item() * args.gradient_accumulation,
                    'contrastive_loss': losses['contrastive'].item(),
                    'attribute_loss': losses.get('attribute_total', torch.tensor(0.0)).item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'timestamp': datetime.now().isoformat()
                }

                with open(log_file, 'a') as f:
                    json.dump(log_entry, f)
                    f.write('\n')

        # Epoch complete - dev eval
        if epoch % args.dev_eval_every == 0:
            print(f"\nRunning dev evaluation for epoch {epoch}...")
            dev_results = dev_evaluate(model, dataset, dev_ids, device)
            print(".3f"
                  ".3f")

            # Log dev results
            log_entry = {
                'epoch': epoch,
                'dev_eval': dev_results,
                'timestamp': datetime.now().isoformat()
            }
            with open(log_file, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')

        # Checkpoint
        if epoch % args.checkpoint_every == 0:
            checkpoint_path = output_dir / f"checkpoints/epoch_{epoch}.pt"
            checkpoint_path.parent.mkdir(exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': sum(epoch_losses) / len(epoch_losses),
                'args': vars(args),
                'git_hash': get_git_hash()
            }, checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")

    # Final checkpoint
    final_checkpoint = output_dir / "checkpoints/final.pt"
    final_checkpoint.parent.mkdir(exist_ok=True)

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args),
        'git_hash': get_git_hash()
    }, final_checkpoint)

    print(f"\nTraining complete! Final checkpoint: {final_checkpoint}")

    # Save training summary
    summary = {
        'training_completed': True,
        'final_epoch': args.epochs,
        'total_steps': global_step,
        'final_checkpoint': str(final_checkpoint),
        'args': vars(args),
        'git_hash': get_git_hash(),
        'completed_at': datetime.now().isoformat()
    }

    with open(output_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    train_optimized(args)


if __name__ == "__main__":
    main()
