#!/usr/bin/env python
"""
Step 7 Hard-Set Evaluation

Evaluates model performance on targeted "hard" cases:
- Shorts/cardigans (weak categories)
- Denim/leather (rare materials)
- Neckline confusers (subtle attribute differences)

Quick iteration tool for Step 7 development.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.protocol_model import ProtocolModel


def create_hard_query_set(dataset, n_queries_per_type: int = 50) -> Dict[str, List[Dict]]:
    """
    Create a targeted set of "hard" queries for evaluation.

    Focuses on:
    - Weak categories: shorts, cardigans
    - Rare materials: denim, leather
    - Neckline confusers: items with similar appearance but different necklines
    """
    print("Creating hard query set...")

    # Define target criteria
    weak_categories = ['shorts', 'cardigans']
    rare_materials = ['denim', 'leather']
    neckline_types = ['crewneck', 'v-neck', 'scoop', 'square', 'boat']

    query_sets = {
        'weak_categories': [],
        'rare_materials': [],
        'neckline_confusers': []
    }

    # Collect weak category queries
    for category in weak_categories:
        category_items = [item for item in dataset
                         if item.get('category2') == category and
                         item.get('attr_material_primary') not in ['unknown', None]]

        # Sample diverse materials within this category
        material_counts = defaultdict(int)
        for item in category_items:
            material_counts[item.get('attr_material_primary', 'unknown')] += 1

        # Get items from top materials to ensure diversity
        selected_items = []
        for material in sorted(material_counts.keys(), key=lambda x: material_counts[x], reverse=True):
            material_items = [item for item in category_items
                            if item.get('attr_material_primary') == material]
            selected_items.extend(material_items[:max(1, n_queries_per_type // len(material_counts))])

        query_sets['weak_categories'].extend(selected_items[:n_queries_per_type])

    # Collect rare material queries
    for material in rare_materials:
        material_items = [item for item in dataset
                         if item.get('attr_material_primary') == material and
                         item.get('category2') not in ['unknown', None]]

        # Sample diverse categories within this material
        category_counts = defaultdict(int)
        for item in material_items:
            category_counts[item.get('category2', 'unknown')] += 1

        selected_items = []
        for category in sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True):
            category_items = [item for item in material_items
                            if item.get('category2') == category]
            selected_items.extend(category_items[:max(1, n_queries_per_type // len(category_counts))])

        query_sets['rare_materials'].extend(selected_items[:n_queries_per_type])

    # Collect neckline confusers (items that look similar but have different necklines)
    neckline_items = [item for item in dataset
                     if item.get('attr_neckline_primary') in neckline_types and
                     item.get('category2') in ['tees', 'blouses', 'sweaters']]

    # Group by visual similarity (same category + material)
    confuser_groups = defaultdict(list)
    for item in neckline_items:
        key = (item.get('category2', ''), item.get('attr_material_primary', ''))
        confuser_groups[key].append(item)

    # Select groups with multiple different necklines
    confusers = []
    for group_key, group_items in confuser_groups.items():
        necklines_in_group = set(item.get('attr_neckline_primary') for item in group_items)
        if len(necklines_in_group) >= 2:  # At least 2 different necklines
            # Take one item per neckline type
            for neckline in necklines_in_group:
                neckline_items = [item for item in group_items
                                if item.get('attr_neckline_primary') == neckline]
                if neckline_items:
                    confusers.append(neckline_items[0])
                    if len(confusers) >= n_queries_per_type:
                        break
            if len(confusers) >= n_queries_per_type:
                break

    query_sets['neckline_confusers'] = confusers[:n_queries_per_type]

    # Summary
    print(f"Created hard query set:")
    print(f"  - Weak categories: {len(query_sets['weak_categories'])} queries")
    print(f"  - Rare materials: {len(query_sets['rare_materials'])} queries")
    print(f"  - Neckline confusers: {len(query_sets['neckline_confusers'])} queries")
    print(f"  - Total: {sum(len(queries) for queries in query_sets.values())} queries")

    return query_sets


def evaluate_hard_set(
    model: ProtocolModel,
    query_sets: Dict[str, List[Dict]],
    catalog_embeddings: np.ndarray,
    catalog_metadata: List[Dict],
    top_k: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on the hard query sets.
    """
    print("Evaluating hard query sets...")

    model.eval()

    # Normalize catalog embeddings
    catalog_embeddings = catalog_embeddings / np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)

    results = {}

    for set_name, queries in query_sets.items():
        if not queries:
            continue

        print(f"Evaluating {set_name} ({len(queries)} queries)...")

        total_correct = 0
        category_correct = 0
        material_correct = 0
        attribute_correct = 0

        for query_item in tqdm(queries, desc=f"{set_name}"):
            query_idx = query_item['idx']

            # Get query embedding
            query_img = query_item['image']
            query_tensor = model.preprocess(query_img).unsqueeze(0).to(model.device)
            with torch.no_grad():
                query_emb = model.forward_image(query_tensor, return_attributes=False)['embedding']
            query_emb = query_emb / query_emb.norm(dim=1, keepdim=True)
            query_emb = query_emb.cpu().numpy().squeeze()

            # Compute similarities
            similarities = np.dot(catalog_embeddings, query_emb)

            # Exclude self
            similarities[query_idx] = -float('inf')

            # Get top-k
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            top_k_items = [catalog_metadata[idx] for idx in top_k_indices]

            # Check correctness based on set type
            query_category = query_item.get('category2')
            query_material = query_item.get('attr_material_primary')
            query_neckline = query_item.get('attr_neckline_primary')

            if set_name == 'weak_categories':
                # For weak categories, check category match
                correct = any(item.get('category2') == query_category for item in top_k_items)
            elif set_name == 'rare_materials':
                # For rare materials, check material match
                correct = any(item.get('attr_material_primary') == query_material for item in top_k_items)
            elif set_name == 'neckline_confusers':
                # For neckline confusers, check neckline match
                correct = any(item.get('attr_neckline_primary') == query_neckline for item in top_k_items)
            else:
                # General correctness
                correct = (any(item.get('category2') == query_category for item in top_k_items) or
                          any(item.get('attr_material_primary') == query_material for item in top_k_items))

            if correct:
                total_correct += 1

            # Additional metrics
            if any(item.get('category2') == query_category for item in top_k_items):
                category_correct += 1
            if any(item.get('attr_material_primary') == query_material for item in top_k_items):
                material_correct += 1
            if set_name == 'neckline_confusers':
                if any(item.get('attr_neckline_primary') == query_neckline for item in top_k_items):
                    attribute_correct += 1

        # Calculate accuracies
        n_queries = len(queries)
        results[set_name] = {
            'n_queries': n_queries,
            'accuracy': total_correct / n_queries,
            'category_accuracy': category_correct / n_queries,
            'material_accuracy': material_correct / n_queries,
        }

        if set_name == 'neckline_confusers':
            results[set_name]['neckline_accuracy'] = attribute_correct / n_queries

        print(".3f")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Step 7 hard-set evaluation"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--catalog",
        type=str,
        required=True,
        help="Path to catalog directory (with catalog_img.npy, catalog_meta.parquet)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--n_queries",
        type=int,
        default=50,
        help="Queries per hard set type"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device"
    )

    args = parser.parse_args()

    print("Step 7 Hard-Set Evaluation")
    print("=" * 35)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Catalog: {args.catalog}")
    print(f"Output: {args.output}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProtocolModel.load_checkpoint(args.model, device=device)

    # Load dataset
    dataset = load_from_disk(args.dataset)
    if isinstance(dataset, dict):
        # Use train split for evaluation
        dataset = dataset.get('train', dataset.get('validation', list(dataset.values())[0]))

    # Load catalog
    catalog_dir = Path(args.catalog)
    catalog_embeddings = np.load(catalog_dir / "catalog_img.npy")
    catalog_meta_df = pd.read_parquet(catalog_dir / "catalog_meta.parquet")
    catalog_metadata = catalog_meta_df.to_dict('records')

    print(f"Loaded catalog: {len(catalog_embeddings)} items")

    # Create hard query sets
    query_sets = create_hard_query_set(dataset, args.n_queries)

    # Evaluate
    results = evaluate_hard_set(
        model=model,
        query_sets=query_sets,
        catalog_embeddings=catalog_embeddings,
        catalog_metadata=catalog_metadata,
        top_k=args.top_k
    )

    # Save results
    results_path = output_dir / "hardset_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print("HARD-SET EVALUATION RESULTS")
    print(f"{'='*50}")

    for set_name, metrics in results.items():
        print(f"\n{set_name.upper()}:")
        print(f"  Queries: {metrics['n_queries']}")
        print(".3f")
        print(".3f")
        print(".3f")

        if 'neckline_accuracy' in metrics:
            print(".3f")

    # Overall assessment
    avg_accuracy = np.mean([metrics['accuracy'] for metrics in results.values()])
    print(f"\nOverall average accuracy: {avg_accuracy:.3f}")

    if avg_accuracy > 0.6:
        print("✓ GOOD: Model performs well on targeted hard cases!")
    elif avg_accuracy > 0.4:
        print("⚠️ OK: Moderate performance on hard cases.")
    else:
        print("❌ POOR: Model struggles with targeted weaknesses.")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
