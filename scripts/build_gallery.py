#!/usr/bin/env python
"""
Build Qualitative Gallery

Creates HTML gallery for visual inspection of retrieval results.
Samples queries across major categories and displays top-10 results.

Usage:
    python scripts/build_gallery.py \
        --catalog_dir artifacts/retrieval/openclip_vitb32_v0 \
        --query_dataset data/processed_v2/hf \
        --query_split validation \
        --output gallery.html \
        --n_samples 100
"""

import argparse
import sys
import base64
from io import BytesIO
from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image
from datasets import load_from_disk
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.embedder import OpenCLIPEmbedder
from src.retrieval.engine import RetrievalEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retrieval gallery for qualitative review"
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
        default="gallery.html",
        help="Output HTML file (default: gallery.html)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of query samples (default: 100)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of results per query (default: 10)"
    )
    parser.add_argument(
        "--stratify_by",
        type=str,
        default="category2",
        help="Column to stratify by (default: category2)"
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
        "--weight_image",
        type=float,
        default=0.7,
        help="Image weight in fusion (default: 0.7)"
    )
    parser.add_argument(
        "--candidate_n",
        type=int,
        default=200,
        help="Number of candidates for stage 1 (default: 200)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    
    return parser.parse_args()


def image_to_base64(img: Image.Image, max_size: int = 256) -> str:
    """Convert PIL Image to base64 string for HTML embedding."""
    # Resize to max dimension
    img = img.copy()
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode()
    
    return f"data:image/jpeg;base64,{img_b64}"


def stratified_sample(ds, stratify_by: str, n_samples: int, seed: int = 42):
    """Sample dataset with stratification by a column."""
    import random
    random.seed(seed)
    
    # Group by stratification column
    groups = defaultdict(list)
    for i, item in enumerate(ds):
        key = item.get(stratify_by, "unknown")
        groups[key].append(i)
    
    # Determine samples per group
    n_groups = len(groups)
    samples_per_group = max(1, n_samples // n_groups)
    
    # Sample from each group
    sampled_indices = []
    for key, indices in groups.items():
        n = min(samples_per_group, len(indices))
        sampled = random.sample(indices, n)
        sampled_indices.extend(sampled)
    
    # If we haven't reached n_samples, add more from largest groups
    if len(sampled_indices) < n_samples:
        remaining = n_samples - len(sampled_indices)
        # Get all indices not yet sampled
        all_indices = set(range(len(ds)))
        available = list(all_indices - set(sampled_indices))
        if available:
            additional = random.sample(available, min(remaining, len(available)))
            sampled_indices.extend(additional)
    
    # Trim to exactly n_samples
    sampled_indices = sampled_indices[:n_samples]
    
    # Sort by stratification key for organized gallery
    sampled_items = [(ds[i], i) for i in sampled_indices]
    sampled_items.sort(key=lambda x: x[0].get(stratify_by, ""))
    
    return [item for item, idx in sampled_items]


def generate_html_header() -> str:
    """Generate HTML header with CSS."""
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Retrieval Gallery</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .summary {
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .query-block {
            background-color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            page-break-inside: avoid;
        }
        .query-header {
            display: flex;
            align-items: start;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
        }
        .query-image {
            flex-shrink: 0;
            margin-right: 20px;
        }
        .query-image img {
            border: 2px solid #333;
            border-radius: 4px;
        }
        .query-info {
            flex-grow: 1;
        }
        .query-info h3 {
            margin-top: 0;
            color: #2196F3;
        }
        .metadata {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 3px;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .metadata-row {
            margin: 5px 0;
        }
        .metadata-label {
            font-weight: bold;
            display: inline-block;
            width: 100px;
        }
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }
        .result-item {
            background-color: #fafafa;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }
        .result-item img {
            max-width: 100%;
            border-radius: 3px;
            margin-bottom: 10px;
        }
        .result-rank {
            font-weight: bold;
            color: #2196F3;
            font-size: 1.1em;
        }
        .result-score {
            color: #666;
            font-size: 0.9em;
        }
        .result-meta {
            font-size: 0.85em;
            margin-top: 5px;
            text-align: left;
        }
        .category-match {
            background-color: #e8f5e9;
            border-color: #4caf50;
        }
        .material-match {
            background-color: #e3f2fd;
        }
        .text-truncate {
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .category-section {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 3px solid #2196F3;
        }
        .category-section h2 {
            color: #2196F3;
        }
    </style>
</head>
<body>
"""


def generate_html_footer() -> str:
    """Generate HTML footer."""
    return """
</body>
</html>
"""


def generate_summary_section(config: dict, category_counts: dict) -> str:
    """Generate summary section HTML."""
    html = ['<div class="summary">']
    html.append('<h2>Gallery Summary</h2>')
    html.append(f'<p><strong>Catalog:</strong> {config["catalog_dir"]}</p>')
    html.append(f'<p><strong>Query dataset:</strong> {config["query_dataset"]}</p>')
    html.append(f'<p><strong>Total queries:</strong> {config["n_samples"]}</p>')
    html.append(f'<p><strong>Top-k:</strong> {config["top_k"]}</p>')
    html.append(f'<p><strong>Weight (image/text):</strong> {config["weight_image"]:.2f}/{1-config["weight_image"]:.2f}</p>')
    
    # Category breakdown
    html.append('<h3>Queries by Category:</h3>')
    html.append('<ul>')
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:15]:
        html.append(f'<li><strong>{cat}:</strong> {count}</li>')
    html.append('</ul>')
    html.append('</div>')
    
    return '\n'.join(html)


def generate_query_block(
    query: dict,
    query_idx: int,
    results: list,
    catalog_images: dict
) -> str:
    """Generate HTML for a single query block."""
    html = ['<div class="query-block">']
    
    # Query header
    html.append('<div class="query-header">')
    
    # Query image
    query_img_b64 = image_to_base64(query["image"])
    html.append('<div class="query-image">')
    html.append(f'<img src="{query_img_b64}" width="200" />')
    html.append('</div>')
    
    # Query info
    html.append('<div class="query-info">')
    html.append(f'<h3>Query {query_idx + 1}</h3>')
    
    # Metadata
    html.append('<div class="metadata">')
    html.append(f'<div class="metadata-row"><span class="metadata-label">Item ID:</span> {query.get("item_ID", "N/A")}</div>')
    html.append(f'<div class="metadata-row"><span class="metadata-label">Category:</span> {query.get("category2", "N/A")}</div>')
    html.append(f'<div class="metadata-row"><span class="metadata-label">Material:</span> {query.get("attr_material_primary", "unknown")}</div>')
    html.append(f'<div class="metadata-row"><span class="metadata-label">Pattern:</span> {query.get("attr_pattern_primary", "unknown")}</div>')
    html.append(f'<div class="metadata-row"><span class="metadata-label">Neckline:</span> {query.get("attr_neckline_primary", "unknown")}</div>')
    html.append(f'<div class="metadata-row"><span class="metadata-label">Sleeve:</span> {query.get("attr_sleeve_primary", "unknown")}</div>')
    
    # Text description
    text = query.get("text", "")
    if text:
        html.append(f'<div class="metadata-row"><span class="metadata-label">Text:</span><br/><span class="text-truncate">{text}</span></div>')
    html.append('</div>')  # metadata
    
    html.append('</div>')  # query-info
    html.append('</div>')  # query-header
    
    # Results grid
    html.append('<div class="results-grid">')
    
    query_cat = query.get("category2", "")
    query_mat = query.get("attr_material_primary", "unknown")
    
    for result in results:
        meta = result["metadata"]
        result_cat = meta.get("category2", "")
        result_mat = meta.get("attr_material_primary", "unknown")
        
        # Determine highlight class
        highlight_class = ""
        if result_cat == query_cat:
            highlight_class = "category-match"
        if result_mat != "unknown" and result_mat == query_mat:
            if highlight_class:
                highlight_class += " material-match"
            else:
                highlight_class = "material-match"
        
        html.append(f'<div class="result-item {highlight_class}">')
        
        # Get result image
        result_idx = result["idx"]
        if result_idx in catalog_images:
            result_img_b64 = catalog_images[result_idx]
        else:
            result_img_b64 = ""
        
        if result_img_b64:
            html.append(f'<img src="{result_img_b64}" />')
        else:
            html.append('<div style="width:180px;height:180px;background:#ddd;"></div>')
        
        # Result info
        html.append(f'<div class="result-rank">#{result["rank"]}</div>')
        html.append(f'<div class="result-score">Score: {result["score"]:.3f}</div>')
        html.append(f'<div class="result-score">Img: {result["img_score"]:.3f} | Txt: {result["txt_score"]:.3f}</div>')
        
        html.append('<div class="result-meta">')
        html.append(f'<strong>{result_cat}</strong><br/>')
        html.append(f'Mat: {result_mat}<br/>')
        html.append(f'Pat: {meta.get("attr_pattern_primary", "unknown")}')
        html.append('</div>')
        
        html.append('</div>')  # result-item
    
    html.append('</div>')  # results-grid
    html.append('</div>')  # query-block
    
    return '\n'.join(html)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("BUILD RETRIEVAL GALLERY")
    print("=" * 60)
    print(f"Catalog: {args.catalog_dir}")
    print(f"Query dataset: {args.query_dataset}")
    print(f"Samples: {args.n_samples}")
    print(f"Stratify by: {args.stratify_by}")
    print("=" * 60)
    
    # Load catalog
    print("\nInitializing retrieval engine...")
    engine = RetrievalEngine(args.catalog_dir, exclude_self=False)
    
    # Load embedder for queries
    print("Initializing query embedder...")
    embedder = OpenCLIPEmbedder(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device
    )
    
    # Load query dataset
    print(f"\nLoading query dataset from: {args.query_dataset}")
    ds = load_from_disk(args.query_dataset)
    
    if isinstance(ds, dict) or hasattr(ds, 'keys'):
        if args.query_split not in ds:
            available = list(ds.keys())
            raise ValueError(f"Split '{args.query_split}' not found. Available: {available}")
        ds = ds[args.query_split]
    
    print(f"Loaded {len(ds)} query items")
    
    # Stratified sampling
    print(f"\nSampling {args.n_samples} queries (stratified by {args.stratify_by})...")
    sampled_queries = stratified_sample(
        ds,
        args.stratify_by,
        args.n_samples,
        seed=args.seed
    )
    print(f"Sampled {len(sampled_queries)} queries")
    
    # Count categories
    category_counts = Counter([q.get(args.stratify_by, "unknown") for q in sampled_queries])
    
    # Pre-load catalog images for results
    print("\nPre-loading catalog images (this may take a moment)...")
    catalog_images = {}
    unique_indices = set()
    
    # First pass: collect all result indices we'll need
    print("Running retrieval to collect result indices...")
    all_results = []
    for query in tqdm(sampled_queries, desc="Running retrieval"):
        # Encode query
        img_vec = embedder.encode_image(query["image"], normalize=True)
        text = query.get("text", "")
        txt_vec = embedder.encode_text(text, normalize=True) if text else None
        
        # Run retrieval
        results = engine.search(
            img_vec,
            txt_vec,
            query_item_id=query.get("item_ID"),
            top_k=args.top_k,
            candidate_n=args.candidate_n,
            weight_image=args.weight_image,
            return_metadata=True
        )
        
        all_results.append(results)
        for result in results:
            unique_indices.add(result["idx"])
    
    print(f"Loading {len(unique_indices)} unique catalog images...")
    
    # Load catalog dataset to get images
    catalog_ds_path = Path(args.catalog_dir).parent.parent.parent / "data" / "processed_v2" / "hf"
    if not catalog_ds_path.exists():
        # Try alternate path from manifest
        if engine.manifest and "dataset" in engine.manifest:
            catalog_ds_path = Path(engine.manifest["dataset"]["path"])
    
    if catalog_ds_path.exists():
        catalog_ds = load_from_disk(str(catalog_ds_path))
        if isinstance(catalog_ds, dict):
            catalog_ds = catalog_ds["train"]
        
        for idx in tqdm(unique_indices, desc="Loading catalog images"):
            if idx < len(catalog_ds):
                img = catalog_ds[int(idx)]["image"]
                catalog_images[idx] = image_to_base64(img, max_size=200)
    else:
        print(f"Warning: Could not load catalog images from {catalog_ds_path}")
    
    # Generate HTML
    print("\nGenerating HTML gallery...")
    html_parts = []
    
    # Header
    html_parts.append(generate_html_header())
    html_parts.append('<h1>Retrieval Gallery</h1>')
    
    # Summary
    config = {
        "catalog_dir": args.catalog_dir,
        "query_dataset": args.query_dataset,
        "n_samples": len(sampled_queries),
        "top_k": args.top_k,
        "weight_image": args.weight_image,
    }
    html_parts.append(generate_summary_section(config, category_counts))
    
    # Group queries by category for organized display
    current_category = None
    for i, (query, results) in enumerate(zip(sampled_queries, all_results)):
        query_cat = query.get(args.stratify_by, "unknown")
        
        # Add category section header if new category
        if query_cat != current_category:
            html_parts.append(f'<div class="category-section">')
            html_parts.append(f'<h2>{query_cat}</h2>')
            html_parts.append('</div>')
            current_category = query_cat
        
        # Add query block
        html_parts.append(generate_query_block(query, i, results, catalog_images))
    
    # Footer
    html_parts.append(generate_html_footer())
    
    # Write to file
    html_content = '\n'.join(html_parts)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✓ Gallery saved to: {args.output}")
    print(f"  Total queries: {len(sampled_queries)}")
    print(f"  Categories: {len(category_counts)}")
    print(f"  File size: {len(html_content) / 1024 / 1024:.1f} MB")
    print("\nOpen the HTML file in a web browser to view the gallery.")


if __name__ == "__main__":
    main()

