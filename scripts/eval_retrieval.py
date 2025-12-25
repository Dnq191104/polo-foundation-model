#!/usr/bin/env python
"""
Evaluate Retrieval System

Runs retrieval on validation queries and computes attribute-aware metrics.

Usage:
    python scripts/eval_retrieval.py \
        --catalog_dir artifacts/retrieval/openclip_vitb32_v0 \
        --query_dataset data/processed_v2/hf \
        --query_split validation \
        --output metrics_report.json \
        --top_k 10
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.embedder import OpenCLIPEmbedder
from src.retrieval.engine import RetrievalEngine
from src.utils.eval_diagnostics import RetrievalDiagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval system"
    )
    
    parser.add_argument(
        "--catalog_dir",
        type=str,
        required=True,
        help="Path to catalog embeddings directory"
    )
    parser.add_argument(
        "--query_dataset",
        type=str,
        required=True,
        help="Path to query dataset directory"
    )
    parser.add_argument(
        "--query_split",
        type=str,
        default="validation",
        help="Query split to use (default: validation)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics_report.json",
        help="Output JSON file for metrics"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for detailed reports (optional)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of results to evaluate (default: 10)"
    )
    parser.add_argument(
        "--candidate_n",
        type=int,
        default=200,
        help="Number of candidates for stage 1 (default: 200)"
    )
    parser.add_argument(
        "--weight_image",
        type=float,
        default=0.7,
        help="Image weight in fusion (default: 0.7)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model name (default: ViT-B-32)"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Pretrained weights (default: openai)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Maximum number of queries to evaluate (default: all)"
    )
    parser.add_argument(
        "--save_per_query",
        action="store_true",
        help="Save per-query results to parquet"
    )
    parser.add_argument(
        "--exclude_self",
        action="store_true",
        help="Exclude query item_ID from results"
    )
    
    return parser.parse_args()


def load_queries(dataset_dir: str, split: str, max_queries: int = None):
    """Load query dataset."""
    print(f"Loading queries from: {dataset_dir}")
    ds = load_from_disk(dataset_dir)
    
    if isinstance(ds, dict) or hasattr(ds, 'keys'):
        if split not in ds:
            available = list(ds.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")
        ds = ds[split]
    
    if max_queries:
        ds = ds.select(range(min(max_queries, len(ds))))
    
    print(f"Loaded {len(ds)} query items")
    return ds


def encode_query(
    item: dict,
    embedder: OpenCLIPEmbedder
) -> tuple:
    """
    Encode a single query item.
    
    Returns:
        Tuple of (img_vec, txt_vec, metadata)
    """
    # Encode image
    img = item["image"]
    img_vec = embedder.encode_image(img, normalize=True)
    
    # Encode text
    text = item.get("text", "")
    txt_vec = embedder.encode_text(text, normalize=True) if text else None
    
    # Extract metadata
    metadata = {
        "item_ID": item.get("item_ID", ""),
        "category2": item.get("category2", ""),
        "text": text,
        "attr_material_primary": item.get("attr_material_primary", "unknown"),
        "attr_pattern_primary": item.get("attr_pattern_primary", "unknown"),
        "attr_neckline_primary": item.get("attr_neckline_primary", "unknown"),
        "attr_sleeve_primary": item.get("attr_sleeve_primary", "unknown"),
    }
    
    return img_vec, txt_vec, metadata


def run_retrieval_batch(
    queries,
    embedder: OpenCLIPEmbedder,
    engine: RetrievalEngine,
    top_k: int = 10,
    candidate_n: int = 200,
    weight_image: float = 0.7,
    show_progress: bool = True
) -> tuple:
    """
    Run retrieval on a batch of queries.
    
    Returns:
        Tuple of (query_items, retrieved_items_list, query_times)
    """
    query_items = []
    retrieved_items_list = []
    query_times = []
    
    iterator = tqdm(queries, desc="Running retrieval") if show_progress else queries
    
    for query in iterator:
        # Encode query
        img_vec, txt_vec, metadata = encode_query(query, embedder)
        query_items.append(metadata)
        
        # Run retrieval
        start_time = time.time()
        results = engine.search(
            img_vec,
            txt_vec,
            query_item_id=metadata["item_ID"],
            top_k=top_k,
            candidate_n=candidate_n,
            weight_image=weight_image,
            return_metadata=True
        )
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        # Extract retrieved item metadata
        retrieved = [r["metadata"] for r in results]
        retrieved_items_list.append(retrieved)
    
    return query_items, retrieved_items_list, query_times


def compute_metrics(
    query_items: List[Dict[str, Any]],
    retrieved_items_list: List[List[Dict[str, Any]]],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Compute retrieval metrics with attribute-aware handling.
    
    Special handling:
    - Skip material metrics when query material is "unknown"
    """
    diagnostics = RetrievalDiagnostics(unknown_value="unknown")
    
    # Overall metrics
    overall = diagnostics.evaluate_batch(query_items, retrieved_items_list, k=top_k)
    
    # Compute material match rate only on queries with known material
    material_known_queries = []
    material_known_results = []
    
    for query, results in zip(query_items, retrieved_items_list):
        if query.get("attr_material_primary", "unknown") != "unknown":
            material_known_queries.append(query)
            material_known_results.append(results)
    
    if material_known_queries:
        material_metrics = diagnostics.evaluate_batch(
            material_known_queries,
            material_known_results,
            k=top_k
        )
        overall[f"material_match@{top_k}_known_only"] = material_metrics.get(f"material_match@{top_k}", 0)
        overall["n_queries_with_known_material"] = len(material_known_queries)
    else:
        overall[f"material_match@{top_k}_known_only"] = 0.0
        overall["n_queries_with_known_material"] = 0
    
    # Category breakdown
    category_metrics = diagnostics.slice_by_category(
        query_items,
        retrieved_items_list,
        k=top_k
    )
    
    # Material breakdown (for queries with known material)
    material_metrics_by_mat = {}
    if material_known_queries:
        material_metrics_by_mat = diagnostics.slice_by_material(
            material_known_queries,
            material_known_results,
            k=top_k
        )
    
    return {
        "overall": overall,
        "by_category": category_metrics,
        "by_material": material_metrics_by_mat,
    }


def generate_report(
    metrics: Dict[str, Any],
    query_times: List[float],
    config: Dict[str, Any]
) -> str:
    """Generate formatted evaluation report."""
    lines = []
    lines.append("=" * 60)
    lines.append("RETRIEVAL EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Catalog: {config['catalog_dir']}")
    lines.append(f"Query dataset: {config['query_dataset']}")
    lines.append(f"Query split: {config['query_split']}")
    lines.append(f"Total queries: {config['n_queries']}")
    lines.append(f"Top-k: {config['top_k']}")
    lines.append(f"Weight (image/text): {config['weight_image']:.2f}/{1-config['weight_image']:.2f}")
    lines.append("")
    
    # Timing
    lines.append("PERFORMANCE:")
    lines.append("-" * 40)
    avg_time = np.mean(query_times) * 1000  # ms
    p50_time = np.percentile(query_times, 50) * 1000
    p95_time = np.percentile(query_times, 95) * 1000
    lines.append(f"  Average query time: {avg_time:.1f}ms")
    lines.append(f"  P50: {p50_time:.1f}ms, P95: {p95_time:.1f}ms")
    lines.append("")
    
    # Overall metrics
    overall = metrics["overall"]
    lines.append("OVERALL METRICS:")
    lines.append("-" * 40)
    
    # Key metrics
    top_k = config['top_k']
    cat_match = overall.get(f"category_match@{top_k}", 0) * 100
    mat_match_all = overall.get(f"material_match@{top_k}", 0) * 100
    mat_match_known = overall.get(f"material_match@{top_k}_known_only", 0) * 100
    pat_match = overall.get(f"pattern_match@{top_k}", 0) * 100
    
    lines.append(f"  Category match@{top_k}: {cat_match:.1f}%")
    lines.append(f"  Material match@{top_k} (all): {mat_match_all:.1f}%")
    lines.append(f"  Material match@{top_k} (known only): {mat_match_known:.1f}%")
    lines.append(f"    (computed on {overall.get('n_queries_with_known_material', 0)} queries)")
    lines.append(f"  Pattern match@{top_k}: {pat_match:.1f}%")
    
    # Other attributes
    for attr in ["neckline", "sleeve"]:
        key = f"{attr}_match@{top_k}"
        if key in overall:
            val = overall[key] * 100
            lines.append(f"  {attr.capitalize()} match@{top_k}: {val:.1f}%")
    
    lines.append("")
    
    # Category breakdown (top categories)
    lines.append(f"BY CATEGORY (Top 10):")
    lines.append("-" * 40)
    
    by_category = metrics["by_category"]
    sorted_cats = sorted(
        by_category.items(),
        key=lambda x: -x[1].get("n_queries", 0)
    )
    
    for i, (category, cat_metrics) in enumerate(sorted_cats[:10]):
        n = cat_metrics.get("n_queries", 0)
        cat_m = cat_metrics.get(f"category_match@{top_k}", 0) * 100
        mat_m = cat_metrics.get(f"material_match@{top_k}", 0) * 100
        lines.append(f"  {category} (n={n}):")
        lines.append(f"    category: {cat_m:.1f}%, material: {mat_m:.1f}%")
    
    lines.append("")
    
    # Material breakdown
    if metrics["by_material"]:
        lines.append(f"BY QUERY MATERIAL (Top 10):")
        lines.append("-" * 40)
        
        by_material = metrics["by_material"]
        sorted_mats = sorted(
            by_material.items(),
            key=lambda x: -x[1].get("n_queries", 0)
        )
        
        for i, (material, mat_metrics) in enumerate(sorted_mats[:10]):
            n = mat_metrics.get("n_queries", 0)
            mat_m = mat_metrics.get(f"material_match@{top_k}", 0) * 100
            lines.append(f"  {material} (n={n}): {mat_m:.1f}%")
        
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def save_per_query_results(
    output_path: str,
    query_items: List[Dict[str, Any]],
    retrieved_items_list: List[List[Dict[str, Any]]],
    query_times: List[float]
):
    """Save per-query results to parquet for detailed analysis."""
    rows = []
    
    for i, (query, results, qtime) in enumerate(
        zip(query_items, retrieved_items_list, query_times)
    ):
        # Top-1 result
        top1 = results[0] if results else {}
        
        row = {
            "query_idx": i,
            "query_item_id": query.get("item_ID", ""),
            "query_category2": query.get("category2", ""),
            "query_material": query.get("attr_material_primary", "unknown"),
            "query_pattern": query.get("attr_pattern_primary", "unknown"),
            "query_time_ms": qtime * 1000,
            "top1_item_id": top1.get("item_ID", ""),
            "top1_category2": top1.get("category2", ""),
            "top1_material": top1.get("attr_material_primary", "unknown"),
            "top1_pattern": top1.get("attr_pattern_primary", "unknown"),
        }
        
        # Count category matches in top-k
        n_cat_match = sum(
            1 for r in results
            if r.get("category2") == query.get("category2")
        )
        row["n_category_match_in_topk"] = n_cat_match
        
        # Count material matches in top-k
        query_mat = query.get("attr_material_primary", "unknown")
        if query_mat != "unknown":
            n_mat_match = sum(
                1 for r in results
                if r.get("attr_material_primary") == query_mat
            )
            row["n_material_match_in_topk"] = n_mat_match
        else:
            row["n_material_match_in_topk"] = None
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"Saved per-query results: {output_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("EVALUATE RETRIEVAL SYSTEM")
    print("=" * 60)
    print(f"Catalog: {args.catalog_dir}")
    print(f"Query dataset: {args.query_dataset}")
    print(f"Query split: {args.query_split}")
    print(f"Top-k: {args.top_k}")
    print(f"Exclude self: {args.exclude_self}")
    print("=" * 60)
    
    # Load catalog
    print("\nInitializing retrieval engine...")
    engine = RetrievalEngine(
        args.catalog_dir,
        exclude_self=args.exclude_self
    )
    
    # Load embedder for queries
    print("\nInitializing query embedder...")
    embedder = OpenCLIPEmbedder(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device
    )
    
    # Load queries
    queries = load_queries(
        args.query_dataset,
        args.query_split,
        max_queries=args.max_queries
    )
    
    # Run retrieval
    print("\nRunning retrieval...")
    query_items, retrieved_items_list, query_times = run_retrieval_batch(
        queries,
        embedder,
        engine,
        top_k=args.top_k,
        candidate_n=args.candidate_n,
        weight_image=args.weight_image,
        show_progress=True
    )
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(query_items, retrieved_items_list, top_k=args.top_k)
    
    # Generate report
    config = {
        "catalog_dir": args.catalog_dir,
        "query_dataset": args.query_dataset,
        "query_split": args.query_split,
        "n_queries": len(query_items),
        "top_k": args.top_k,
        "candidate_n": args.candidate_n,
        "weight_image": args.weight_image,
        "exclude_self": args.exclude_self,
    }
    
    report = generate_report(metrics, query_times, config)
    print("\n" + report)
    
    # Save metrics
    output = {
        "config": config,
        "metrics": metrics,
        "query_times": {
            "mean_ms": float(np.mean(query_times) * 1000),
            "std_ms": float(np.std(query_times) * 1000),
            "p50_ms": float(np.percentile(query_times, 50) * 1000),
            "p95_ms": float(np.percentile(query_times, 95) * 1000),
            "p99_ms": float(np.percentile(query_times, 99) * 1000),
        },
    }
    
    # Save main metrics
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to: {args.output}")
    
    # Save to output_dir if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save overall metrics
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(output["metrics"]["overall"], f, indent=2)
        
        # Save category breakdown
        category_path = output_dir / "metrics_by_category.json"
        with open(category_path, "w") as f:
            json.dump(output["metrics"]["by_category"], f, indent=2)
        
        # Save material breakdown
        if output["metrics"]["by_material"]:
            material_path = output_dir / "metrics_by_material.json"
            with open(material_path, "w") as f:
                json.dump(output["metrics"]["by_material"], f, indent=2)
        
        print(f"Detailed reports saved to: {output_dir}")
    
    # Save per-query results if requested
    if args.save_per_query:
        per_query_path = args.output.replace(".json", "_per_query.parquet")
        if args.output_dir:
            per_query_path = str(Path(args.output_dir) / "per_query_results.parquet")
        
        save_per_query_results(
            per_query_path,
            query_items,
            retrieved_items_list,
            query_times
        )
    
    print("\n[SUCCESS] Evaluation completed successfully!")


if __name__ == "__main__":
    main()

