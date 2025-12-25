"""
I/O utilities for catalog embeddings and metadata.

Handles saving/loading of:
- Embedding arrays (.npy)
- Metadata (parquet)
- Manifest (json)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any
import hashlib
import subprocess


def get_git_hash() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def compute_array_checksum(arr: np.ndarray) -> str:
    """Compute MD5 checksum of numpy array."""
    return hashlib.md5(arr.tobytes()).hexdigest()


def save_catalog(
    output_dir: str,
    img_embeddings: np.ndarray,
    txt_embeddings: np.ndarray,
    metadata: pd.DataFrame,
    manifest: Dict[str, Any],
    validate: bool = True
) -> None:
    """
    Save catalog embeddings, metadata, and manifest.
    
    Args:
        output_dir: Directory to save files
        img_embeddings: Image embeddings (N, D), normalized
        txt_embeddings: Text embeddings (N, D), normalized
        metadata: Metadata DataFrame with item_ID, category2, etc.
        manifest: Configuration manifest
        validate: Validate data before saving
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Validate
    if validate:
        assert img_embeddings.shape[0] == txt_embeddings.shape[0], \
            "Image and text embeddings must have same number of rows"
        assert img_embeddings.shape[0] == len(metadata), \
            "Embeddings and metadata must have same number of rows"
        assert img_embeddings.shape[1] == txt_embeddings.shape[1], \
            "Image and text embeddings must have same dimension"
        
        # Check normalization
        img_norms = np.linalg.norm(img_embeddings, axis=1)
        txt_norms = np.linalg.norm(txt_embeddings, axis=1)
        
        assert np.allclose(img_norms, 1.0, atol=1e-4), \
            f"Image embeddings not normalized: mean norm = {img_norms.mean():.6f}"
        assert np.allclose(txt_norms, 1.0, atol=1e-4), \
            f"Text embeddings not normalized: mean norm = {txt_norms.mean():.6f}"
        
        # Check for required columns
        required_cols = ["item_ID", "category2"]
        for col in required_cols:
            assert col in metadata.columns, f"Missing required column: {col}"
        
        # Check for missing item_IDs
        missing_ids = metadata["item_ID"].isna().sum()
        assert missing_ids == 0, f"Found {missing_ids} missing item_IDs"
    
    # Save embeddings
    img_path = output_path / "catalog_img.npy"
    txt_path = output_path / "catalog_txt.npy"
    
    np.save(img_path, img_embeddings)
    np.save(txt_path, txt_embeddings)
    
    print(f"Saved image embeddings: {img_path} (shape: {img_embeddings.shape})")
    print(f"Saved text embeddings: {txt_path} (shape: {txt_embeddings.shape})")
    
    # Save metadata
    meta_path = output_path / "catalog_meta.parquet"
    metadata.to_parquet(meta_path, index=False)
    print(f"Saved metadata: {meta_path} ({len(metadata)} items)")
    
    # Augment manifest with checksums and git info
    manifest["checksums"] = {
        "img_embeddings": compute_array_checksum(img_embeddings),
        "txt_embeddings": compute_array_checksum(txt_embeddings),
    }
    manifest["git_hash"] = get_git_hash()
    
    # Save manifest
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved manifest: {manifest_path}")
    print(f"\nCatalog saved successfully to: {output_dir}")


def load_catalog(
    catalog_dir: str,
    load_metadata: bool = True,
    load_manifest: bool = True,
    validate_checksums: bool = False
) -> Dict[str, Any]:
    """
    Load catalog embeddings, metadata, and manifest.
    
    Args:
        catalog_dir: Directory containing catalog files
        load_metadata: Load metadata DataFrame
        load_manifest: Load manifest dict
        validate_checksums: Validate array checksums against manifest
        
    Returns:
        Dictionary with keys: img_embeddings, txt_embeddings, metadata, manifest
    """
    catalog_path = Path(catalog_dir)
    
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog directory not found: {catalog_dir}")
    
    result = {}
    
    # Load embeddings
    img_path = catalog_path / "catalog_img.npy"
    txt_path = catalog_path / "catalog_txt.npy"
    
    if not img_path.exists():
        raise FileNotFoundError(f"Image embeddings not found: {img_path}")
    if not txt_path.exists():
        raise FileNotFoundError(f"Text embeddings not found: {txt_path}")
    
    result["img_embeddings"] = np.load(img_path)
    result["txt_embeddings"] = np.load(txt_path)
    
    print(f"Loaded image embeddings: {img_path} (shape: {result['img_embeddings'].shape})")
    print(f"Loaded text embeddings: {txt_path} (shape: {result['txt_embeddings'].shape})")
    
    # Load metadata
    if load_metadata:
        meta_path = catalog_path / "catalog_meta.parquet"
        if meta_path.exists():
            result["metadata"] = pd.read_parquet(meta_path)
            print(f"Loaded metadata: {meta_path} ({len(result['metadata'])} items)")
        else:
            print(f"Warning: Metadata not found at {meta_path}")
            result["metadata"] = None
    
    # Load manifest
    if load_manifest:
        manifest_path = catalog_path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                result["manifest"] = json.load(f)
            print(f"Loaded manifest: {manifest_path}")
        else:
            print(f"Warning: Manifest not found at {manifest_path}")
            result["manifest"] = None
    
    # Validate checksums if requested
    if validate_checksums and result.get("manifest"):
        manifest = result["manifest"]
        if "checksums" in manifest:
            img_checksum = compute_array_checksum(result["img_embeddings"])
            txt_checksum = compute_array_checksum(result["txt_embeddings"])
            
            expected_img = manifest["checksums"].get("img_embeddings")
            expected_txt = manifest["checksums"].get("txt_embeddings")
            
            if expected_img and img_checksum != expected_img:
                raise ValueError(
                    f"Image embeddings checksum mismatch!\n"
                    f"Expected: {expected_img}\n"
                    f"Got: {img_checksum}"
                )
            if expected_txt and txt_checksum != expected_txt:
                raise ValueError(
                    f"Text embeddings checksum mismatch!\n"
                    f"Expected: {expected_txt}\n"
                    f"Got: {txt_checksum}"
                )
            
            print("Checksum validation passed ✓")
    
    return result


def validate_catalog(catalog_dir: str, verbose: bool = True) -> bool:
    """
    Validate catalog structure and data integrity.
    
    Args:
        catalog_dir: Directory containing catalog files
        verbose: Print validation details
        
    Returns:
        True if validation passes
    """
    try:
        data = load_catalog(
            catalog_dir,
            load_metadata=True,
            load_manifest=True,
            validate_checksums=True
        )
        
        # Check shapes match
        n_img = data["img_embeddings"].shape[0]
        n_txt = data["txt_embeddings"].shape[0]
        n_meta = len(data["metadata"]) if data["metadata"] is not None else 0
        
        if verbose:
            print(f"\nValidation summary:")
            print(f"  Image embeddings: {n_img} items")
            print(f"  Text embeddings: {n_txt} items")
            print(f"  Metadata: {n_meta} items")
        
        assert n_img == n_txt, f"Shape mismatch: img={n_img}, txt={n_txt}"
        if data["metadata"] is not None:
            assert n_img == n_meta, f"Shape mismatch: embeddings={n_img}, meta={n_meta}"
        
        # Check normalization
        img_norms = np.linalg.norm(data["img_embeddings"], axis=1)
        txt_norms = np.linalg.norm(data["txt_embeddings"], axis=1)
        
        if verbose:
            print(f"  Image norm: mean={img_norms.mean():.6f}, std={img_norms.std():.6f}")
            print(f"  Text norm: mean={txt_norms.mean():.6f}, std={txt_norms.std():.6f}")
        
        assert np.allclose(img_norms, 1.0, atol=1e-4), "Image embeddings not normalized"
        assert np.allclose(txt_norms, 1.0, atol=1e-4), "Text embeddings not normalized"
        
        if verbose:
            print("\n✓ Catalog validation passed")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"\n✗ Catalog validation failed: {e}")
        return False

