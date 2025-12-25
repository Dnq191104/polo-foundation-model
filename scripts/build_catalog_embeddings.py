#!/usr/bin/env python
"""
Build Catalog Embeddings

Offline job to encode catalog items (images + text) and save embeddings
for fast retrieval.

Usage:
    python scripts/build_catalog_embeddings.py \
        --dataset_dir data/processed_v2/hf \
        --split train \
        --output artifacts/retrieval/openclip_vitb32_v0 \
        --batch_size 32
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from datasets import load_from_disk
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.embedder import OpenCLIPEmbedder
from src.retrieval.io import save_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build catalog embeddings for retrieval"
    )
    
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to HuggingFace dataset directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to encode (default: train)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for embeddings"
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run identifier for tracking (default: timestamp)"
    )
    parser.add_argument(
        "--text_batch_size",
        type=int,
        default=64,
        help="Batch size for text encoding (default: 64)"
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
    print(f"Columns: {ds.column_names}")
    
    return ds


def extract_metadata(ds) -> pd.DataFrame:
    """Extract metadata columns for saving."""
    # Required columns
    required = ["item_ID", "category2", "text"]
    
    # Optional attribute columns
    optional = [
        "attr_material_primary",
        "attr_pattern_primary",
        "attr_neckline_primary",
        "attr_sleeve_primary"
    ]
    
    # Check what's available
    available_cols = []
    for col in required:
        if col not in ds.column_names:
            raise ValueError(f"Required column '{col}' not found in dataset")
        available_cols.append(col)
    
    for col in optional:
        if col in ds.column_names:
            available_cols.append(col)
        else:
            print(f"Warning: Optional column '{col}' not found")
    
    # Extract as DataFrame
    metadata = pd.DataFrame({
        col: ds[col] for col in available_cols
    })
    
    return metadata


def encode_catalog(
    ds,
    embedder: OpenCLIPEmbedder,
    batch_size: int = 32,
    text_batch_size: int = 64
) -> tuple:
    """
    Encode all catalog items.
    
    Returns:
        Tuple of (img_embeddings, txt_embeddings)
    """
    n_items = len(ds)
    
    print(f"\nEncoding {n_items} catalog items...")
    print("=" * 60)
    
    # Extract images and text
    print("Extracting images and text...")
    images = []
    texts = []
    
    for i, item in enumerate(ds):
        img = item["image"]
        text = item.get("text", "")
        
        # Convert image to PIL if needed
        if not isinstance(img, Image.Image):
            img = img.convert("RGB") if hasattr(img, 'convert') else Image.fromarray(img)
        
        images.append(img)
        texts.append(text if text else "")
        
        if (i + 1) % 1000 == 0:
            print(f"  Extracted {i + 1}/{n_items} items...")
    
    print(f"Extracted {len(images)} images and {len(texts)} texts")
    
    # Encode images
    print("\nEncoding images...")
    start_time = time.time()
    img_embeddings = embedder.encode_image_batch(
        images,
        batch_size=batch_size,
        normalize=True,
        show_progress=True
    )
    img_time = time.time() - start_time
    print(f"Image encoding completed in {img_time:.1f}s ({img_time/n_items*1000:.1f}ms per item)")
    
    # Encode text
    print("\nEncoding text...")
    start_time = time.time()
    txt_embeddings = embedder.encode_text_batch(
        texts,
        batch_size=text_batch_size,
        normalize=True,
        show_progress=True
    )
    txt_time = time.time() - start_time
    print(f"Text encoding completed in {txt_time:.1f}s ({txt_time/n_items*1000:.1f}ms per item)")
    
    # Validate shapes
    assert img_embeddings.shape[0] == n_items, "Image embedding count mismatch"
    assert txt_embeddings.shape[0] == n_items, "Text embedding count mismatch"
    assert img_embeddings.shape[1] == txt_embeddings.shape[1], "Embedding dimension mismatch"
    
    print(f"\nEncoding summary:")
    print(f"  Shape: {img_embeddings.shape}")
    print(f"  Total time: {img_time + txt_time:.1f}s")
    print(f"  Throughput: {n_items / (img_time + txt_time):.1f} items/s")
    
    return img_embeddings, txt_embeddings


def build_manifest(
    args: argparse.Namespace,
    embedder: OpenCLIPEmbedder,
    dataset_path: str,
    split: str,
    n_items: int,
    embedding_dim: int
) -> dict:
    """Build manifest with configuration and metadata."""
    if args.run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = args.run_id
    
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "split": split,
            "n_items": n_items,
        },
        "model": embedder.get_model_info(),
        "embedding_dim": embedding_dim,
        "encoding": {
            "batch_size": args.batch_size,
            "text_batch_size": args.text_batch_size,
        },
        "retrieval_config": {
            "default_weight_image": 0.7,
            "default_weight_text": 0.3,
            "default_candidate_n": 200,
            "default_top_k": 10,
        },
    }
    
    return manifest


def main():
    args = parse_args()
    
    print("=" * 60)
    print("BUILD CATALOG EMBEDDINGS")
    print("=" * 60)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model_name} ({args.pretrained})")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Load dataset
    ds = load_dataset_split(args.dataset_dir, args.split)
    
    # Extract metadata
    metadata = extract_metadata(ds)
    print(f"\nMetadata extracted: {metadata.shape}")
    print(f"Columns: {list(metadata.columns)}")
    
    # Check for missing item_IDs
    missing_ids = metadata["item_ID"].isna().sum()
    if missing_ids > 0:
        raise ValueError(f"Found {missing_ids} missing item_IDs")
    
    # Initialize embedder
    print(f"\nInitializing embedder...")
    embedder = OpenCLIPEmbedder(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device
    )
    
    # Encode catalog
    img_embeddings, txt_embeddings = encode_catalog(
        ds,
        embedder,
        batch_size=args.batch_size,
        text_batch_size=args.text_batch_size
    )
    
    # Build manifest
    manifest = build_manifest(
        args,
        embedder,
        args.dataset_dir,
        args.split,
        len(ds),
        img_embeddings.shape[1]
    )
    
    # Save catalog
    print(f"\nSaving catalog to: {args.output}")
    save_catalog(
        args.output,
        img_embeddings,
        txt_embeddings,
        metadata,
        manifest,
        validate=True
    )
    
    # Validation
    print("\nValidating saved catalog...")
    from src.retrieval.io import validate_catalog
    if validate_catalog(args.output, verbose=True):
        print("\n✓ Catalog build completed successfully!")
    else:
        print("\n✗ Catalog validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

