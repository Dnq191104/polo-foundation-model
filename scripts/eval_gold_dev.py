#!/usr/bin/env python
"""
Gold Dev Eval Metrics

Compute comprehensive metrics for the gold dev evaluation set.
Focuses on category accuracy (Recall@K) with per-category breakdowns
and weak-category averages for rapid iteration signal.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.protocol_model import ProtocolModel


def load_gold_eval_data(gold_eval_path):
    """Load gold eval IDs."""
    with open(gold_eval_path, 'r') as f:
        data = json.load(f)

    eval_ids = data.get('eval_ids', data if isinstance(data, list) else [])
    metadata = data.get('metadata', {})

    return eval_ids, metadata


def load_overrides(overrides_path):
    """Load overrides mapping from separate file."""
    if not Path(overrides_path).exists():
        return {}

    with open(overrides_path, 'r') as f:
        data = json.load(f)

    return data.get('overrides', {})


def evaluate_gold_dev(
    model,
    dataset,
    eval_ids,
    overrides=None,
    device='cuda',
    batch_size=64
):
    """
    Evaluate gold dev set with comprehensive category metrics.

    Args:
        model: ProtocolModel instance
        dataset: Dataset to evaluate on
        eval_ids: List of query indices
        verified_labels: Dict of verified label corrections
        device: Device to use
        batch_size: Batch size for embedding computation

    Returns:
        Dict with evaluation results
    """

    model.eval()
    print(f"Evaluating {len(eval_ids)} gold dev queries...")

    # Weak categories for special tracking
    weak_categories = ['shorts', 'rompers', 'cardigans']

    # Compute all embeddings (cached for efficiency)
    if not hasattr(evaluate_gold_dev, '_embeddings'):
        print("Computing dataset embeddings...")
        all_embeddings = []

        for i in tqdm(range(0, len(dataset), batch_size), desc="Embedding"):
            batch_items = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]
            batch_images = [item['image'] for item in batch_items]
            batch_tensors = torch.stack([model.preprocess(img) for img in batch_images]).to(device)

            with torch.no_grad():
                embeddings = model.forward_image(batch_tensors, return_attributes=False)['embedding']
            all_embeddings.append(embeddings.cpu())

        evaluate_gold_dev._embeddings = torch.cat(all_embeddings)
        evaluate_gold_dev._embeddings_norm = evaluate_gold_dev._embeddings / evaluate_gold_dev._embeddings.norm(dim=1, keepdim=True)

    embeddings_norm = evaluate_gold_dev._embeddings_norm

    # Track results
    results = {
        'overall': defaultdict(list),
        'by_category': defaultdict(lambda: defaultdict(list)),
        'weak_categories': {cat: defaultdict(list) for cat in weak_categories},
        'queries': []
    }

    for query_idx in eval_ids:
        query_item = dataset[query_idx]

        # Apply overrides if available
        if overrides and query_idx in overrides:
            override = overrides[query_idx]
            # Apply override corrections to query_item
            for key, value in override.items():
                if key in query_item or key.startswith('attr_'):
                    query_item[key] = value

        query_category = query_item.get('category2', '')

        # Get query embedding
        query_img = query_item['image']
        query_tensor = model.preprocess(query_img).unsqueeze(0).to(device)

        with torch.no_grad():
            query_emb = model.forward_image(query_tensor, return_attributes=False)['embedding']
        query_emb = query_emb / query_emb.norm(dim=1, keepdim=True)
        query_emb = query_emb.cpu()

        # Compute similarities
        similarities = (query_emb @ embeddings_norm.T).squeeze()
        similarities[query_idx] = -float('inf')  # Exclude self

        # Get top-K results
        for k in [1, 5, 10]:
            topk_indices = torch.topk(similarities, k).indices.tolist()
            topk_items = [dataset[idx] for idx in topk_indices]

            # Category accuracy (Recall@K)
            category_correct = any(item.get('category2') == query_category for item in topk_items)
            results['overall'][f'category_recall@{k}'].append(category_correct)

            # Per-category tracking
            results['by_category'][query_category][f'category_recall@{k}'].append(category_correct)

            # Weak category tracking
            if query_category in weak_categories:
                results['weak_categories'][query_category][f'category_recall@{k}'].append(category_correct)

        # Store query result
        query_result = {
            'query_idx': query_idx,
            'category': query_category,
            'item_ID': query_item.get('item_ID', ''),
            'verified': query_idx in verified_labels if verified_labels else False
        }
        results['queries'].append(query_result)

    # Compute aggregate metrics
    metrics = {
        'overall': {},
        'by_category': {},
        'weak_categories': {},
        'summary': {}
    }

    # Overall metrics
    for k in [1, 5, 10]:
        recall_values = results['overall'][f'category_recall@{k}']
        metrics['overall'][f'category_recall@{k}'] = sum(recall_values) / len(recall_values)

    # Per-category metrics
    for category, cat_results in results['by_category'].items():
        cat_metrics = {}
        for k in [1, 5, 10]:
            recall_values = cat_results[f'category_recall@{k}']
            if recall_values:
                cat_metrics[f'category_recall@{k}'] = sum(recall_values) / len(recall_values)
                cat_metrics[f'n_queries@{k}'] = len(recall_values)
        metrics['by_category'][category] = cat_metrics

    # Weak category metrics and average
    weak_averages = {}
    for k in [1, 5, 10]:
        weak_recalls = []
        for cat in weak_categories:
            if cat in results['weak_categories'] and results['weak_categories'][cat][f'category_recall@{k}']:
                cat_recall = sum(results['weak_categories'][cat][f'category_recall@{k}']) / len(results['weak_categories'][cat][f'category_recall@{k}'])
                metrics['weak_categories'][cat] = metrics['weak_categories'].get(cat, {})
                metrics['weak_categories'][cat][f'category_recall@{k}'] = cat_recall
                weak_recalls.append(cat_recall)

        if weak_recalls:
            weak_averages[f'weak_avg_recall@{k}'] = sum(weak_recalls) / len(weak_recalls)

    # Summary statistics
    metrics['summary'] = {
        'n_queries': len(eval_ids),
        'n_verified': sum(1 for q in results['queries'] if q['verified']),
        'weak_categories': weak_categories,
        'top_1_accuracy': metrics['overall'].get('category_recall@1', 0),
        'top_5_accuracy': metrics['overall'].get('category_recall@5', 0),
        **weak_averages
    }

    return metrics


def generate_metrics_report(metrics, metadata=None):
    """Generate human-readable metrics report."""

    lines = []
    lines.append("=" * 60)
    lines.append("GOLD DEV EVAL METRICS REPORT")
    lines.append("=" * 60)

    if metadata:
        lines.append(f"Eval set: {metadata.get('n_queries', 'N/A')} queries")
        lines.append(f"Created: {metadata.get('created_at', 'N/A')}")
        if 'verified_at' in metadata:
            lines.append(f"Verified: {metadata.get('verified_at', 'N/A')}")
            lines.append(f"Corrections: {metadata.get('n_corrections', 0)}")
        lines.append("")

    # Overall metrics
    lines.append("OVERALL CATEGORY RECALL:")
    lines.append("-" * 40)
    overall = metrics['overall']
    lines.append(".3f")
    lines.append(".3f")
    lines.append(".3f")
    lines.append("")

    # Weak categories
    if metrics['weak_categories']:
        lines.append("WEAK CATEGORIES (shorts/rompers/cardigans):")
        lines.append("-" * 40)
        for cat, cat_metrics in metrics['weak_categories'].items():
            if cat_metrics:
                recall1 = cat_metrics.get('category_recall@1', 0)
                recall5 = cat_metrics.get('category_recall@5', 0)
                recall10 = cat_metrics.get('category_recall@10', 0)
                lines.append(".3f")

        # Weak average
        summary = metrics['summary']
        if 'weak_avg_recall@1' in summary:
            lines.append("")
            lines.append("WEAK CATEGORY AVERAGES:")
            lines.append(".3f")
            lines.append(".3f")
            lines.append(".3f")
        lines.append("")

    # Top categories by performance
    lines.append("TOP 10 CATEGORIES BY RECALL@10:")
    lines.append("-" * 40)

    by_category = metrics['by_category']
    sorted_cats = sorted(
        by_category.items(),
        key=lambda x: x[1].get('category_recall@10', 0),
        reverse=True
    )

    for i, (cat, cat_metrics) in enumerate(sorted_cats[:10]):
        recall10 = cat_metrics.get('category_recall@10', 0)
        n_queries = cat_metrics.get('n_queries@10', 0)
        lines.append(".3f")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate gold dev set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--gold_eval", type=str, required=True, help="Gold eval JSON file")
    parser.add_argument("--overrides", type=str, help="Overrides JSON file (optional)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--report", type=str, help="Output text report file")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    args = parser.parse_args()

    # Load gold eval data
    eval_ids, metadata = load_gold_eval_data(args.gold_eval)

    # Load overrides if provided
    overrides = None
    if args.overrides:
        overrides = load_overrides(args.overrides)
        print(f"Overrides: {args.overrides} ({len(overrides)} corrections)")
    else:
        # Auto-detect overrides file
        overrides_path = Path(args.gold_eval).parent / "gold_dev_eval_overrides.json"
        if overrides_path.exists():
            overrides = load_overrides(str(overrides_path))
            print(f"Auto-loaded overrides: {overrides_path} ({len(overrides)} corrections)")

    print("=" * 60)
    print("GOLD DEV EVAL")
    print("=" * 60)
    print(f"Gold eval: {args.gold_eval}")
    print(f"Queries: {len(eval_ids)}")
    if overrides:
        print(f"Overrides: {len(overrides)} corrections applied")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 60)

    # Load model
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = ProtocolModel(
        model_name="ViT-B-32",
        pretrained="openai",
        projection_hidden=None,
        use_attribute_heads=True,
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load dataset
    dataset = load_from_disk(args.dataset)[args.split]

    # Run evaluation
    metrics = evaluate_gold_dev(
        model, dataset, eval_ids, overrides, device, args.batch_size
    )

    # Add metadata to results
    results = {
        'metrics': metrics,
        'config': {
            'gold_eval_path': args.gold_eval,
            'checkpoint': args.checkpoint,
            'dataset': args.dataset,
            'split': args.split,
            'device': device,
            'batch_size': args.batch_size
        },
        'metadata': metadata
    }

    # Generate and print report
    report = generate_metrics_report(metrics, metadata)
    print("\n" + report)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Metrics saved to: {output_path}")

    # Save text report
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            f.write(report)

        print(f"Report saved to: {report_path}")

    print("\n[SUCCESS] Gold dev evaluation completed!")


if __name__ == "__main__":
    main()
