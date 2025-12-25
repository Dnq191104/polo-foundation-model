"""
Retrieval Engine for Fashion Search

Implements two-stage retrieval:
1. Image-based candidate generation (top-N)
2. Multimodal fusion re-ranking (image + text)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .io import load_catalog


class RetrievalEngine:
    """
    Two-stage retrieval engine with multimodal fusion.
    
    Stage 1: Image-only candidate generation (fast)
    Stage 2: Image+text fusion re-ranking (interpretable)
    
    Args:
        catalog_dir: Directory containing catalog embeddings
        exclude_self: Exclude query item_ID from results (for evaluation)
    """
    
    def __init__(
        self,
        catalog_dir: str,
        exclude_self: bool = False
    ):
        """Load catalog and prepare for search."""
        self.catalog_dir = catalog_dir
        self.exclude_self = exclude_self
        
        # Load catalog
        print(f"Loading catalog from: {catalog_dir}")
        catalog = load_catalog(
            catalog_dir,
            load_metadata=True,
            load_manifest=True,
            validate_checksums=False
        )
        
        self.img_embeddings = catalog["img_embeddings"]
        self.txt_embeddings = catalog["txt_embeddings"]
        self.metadata = catalog["metadata"]
        self.manifest = catalog["manifest"]
        
        self.n_items = len(self.img_embeddings)
        self.embedding_dim = self.img_embeddings.shape[1]
        
        # Create item_ID to index mapping for exclusion
        if "item_ID" in self.metadata.columns:
            self.item_id_to_idx = {
                item_id: idx
                for idx, item_id in enumerate(self.metadata["item_ID"])
            }
        else:
            self.item_id_to_idx = {}
        
        print(f"Catalog loaded: {self.n_items} items, dim={self.embedding_dim}")
    
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return vec
        return vec / norm
    
    def _compute_similarity(
        self,
        query_vec: np.ndarray,
        catalog_vecs: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and catalog.
        
        Assumes both are L2-normalized, so cosine = dot product.
        
        Args:
            query_vec: Query vector (D,)
            catalog_vecs: Catalog vectors (N, D)
            
        Returns:
            Similarities (N,)
        """
        return catalog_vecs @ query_vec
    
    def search_image_only(
        self,
        query_img_vec: np.ndarray,
        top_k: int = 10,
        exclude_item_id: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using image embedding only.
        
        Args:
            query_img_vec: Query image embedding (D,), normalized
            top_k: Number of results to return
            exclude_item_id: Item ID to exclude from results
            
        Returns:
            Tuple of (indices, scores)
        """
        # Compute similarities
        similarities = self._compute_similarity(query_img_vec, self.img_embeddings)
        
        # Exclude self if requested
        if exclude_item_id and exclude_item_id in self.item_id_to_idx:
            exclude_idx = self.item_id_to_idx[exclude_item_id]
            similarities[exclude_idx] = -np.inf
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def search_multimodal(
        self,
        query_img_vec: np.ndarray,
        query_txt_vec: Optional[np.ndarray] = None,
        top_k: int = 10,
        candidate_n: int = 200,
        weight_image: float = 0.7,
        exclude_item_id: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Search using image+text fusion (two-stage).
        
        Stage 1: Get top candidate_n by image similarity
        Stage 2: Re-rank candidates by fused score
        
        Args:
            query_img_vec: Query image embedding (D,), normalized
            query_txt_vec: Query text embedding (D,), normalized (optional)
            top_k: Number of final results
            candidate_n: Number of candidates from stage 1
            weight_image: Weight for image similarity (0-1)
            exclude_item_id: Item ID to exclude from results
            
        Returns:
            Tuple of (indices, fused_scores, score_breakdown)
            where score_breakdown = {"img_scores": ..., "txt_scores": ...}
        """
        # Stage 1: Image-based candidate generation
        candidate_n = min(candidate_n, self.n_items)
        candidate_indices, img_similarities = self.search_image_only(
            query_img_vec,
            top_k=candidate_n,
            exclude_item_id=exclude_item_id
        )
        
        # Handle text: if no text provided or weight is 1.0, return image-only
        if query_txt_vec is None or weight_image >= 1.0:
            final_indices = candidate_indices[:top_k]
            final_scores = img_similarities[:top_k]
            score_breakdown = {
                "img_scores": img_similarities[:top_k],
                "txt_scores": np.zeros_like(final_scores),
            }
            return final_indices, final_scores, score_breakdown
        
        # Stage 2: Compute text similarity for candidates
        candidate_txt_embeddings = self.txt_embeddings[candidate_indices]
        txt_similarities = self._compute_similarity(query_txt_vec, candidate_txt_embeddings)
        
        # Fused score
        weight_text = 1.0 - weight_image
        fused_scores = weight_image * img_similarities + weight_text * txt_similarities
        
        # Re-rank candidates by fused score
        rerank_indices = np.argsort(fused_scores)[::-1][:top_k]
        
        final_indices = candidate_indices[rerank_indices]
        final_scores = fused_scores[rerank_indices]
        
        score_breakdown = {
            "img_scores": img_similarities[rerank_indices],
            "txt_scores": txt_similarities[rerank_indices],
        }
        
        return final_indices, final_scores, score_breakdown
    
    def search(
        self,
        query_img_vec: np.ndarray,
        query_txt_vec: Optional[np.ndarray] = None,
        query_item_id: Optional[str] = None,
        top_k: int = 10,
        candidate_n: int = 200,
        weight_image: float = 0.7,
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Unified search interface.
        
        Args:
            query_img_vec: Query image embedding (D,), normalized
            query_txt_vec: Query text embedding (D,), normalized (optional)
            query_item_id: Query item ID (for self-exclusion if enabled)
            top_k: Number of results to return
            candidate_n: Number of candidates for stage 1
            weight_image: Image weight in fusion
            return_metadata: Include metadata in results
            
        Returns:
            List of result dicts with keys:
                - idx: catalog index
                - score: final similarity score
                - img_score: image similarity (if multimodal)
                - txt_score: text similarity (if multimodal)
                - metadata: item metadata (if return_metadata=True)
        """
        # Determine exclusion
        exclude_id = None
        if self.exclude_self and query_item_id:
            exclude_id = query_item_id
        
        # Run search
        indices, scores, breakdown = self.search_multimodal(
            query_img_vec,
            query_txt_vec,
            top_k=top_k,
            candidate_n=candidate_n,
            weight_image=weight_image,
            exclude_item_id=exclude_id
        )
        
        # Build results
        results = []
        for i, (idx, score) in enumerate(zip(indices, scores)):
            result = {
                "rank": i + 1,
                "idx": int(idx),
                "score": float(score),
                "img_score": float(breakdown["img_scores"][i]),
                "txt_score": float(breakdown["txt_scores"][i]),
            }
            
            if return_metadata and self.metadata is not None:
                # Add metadata as dict
                meta_row = self.metadata.iloc[idx].to_dict()
                result["metadata"] = meta_row
            
            results.append(result)
        
        return results
    
    def get_catalog_item(self, idx: int) -> Dict[str, Any]:
        """
        Get catalog item by index.
        
        Args:
            idx: Catalog index
            
        Returns:
            Dict with metadata and embeddings
        """
        item = {
            "idx": idx,
            "img_embedding": self.img_embeddings[idx],
            "txt_embedding": self.txt_embeddings[idx],
        }
        
        if self.metadata is not None:
            item["metadata"] = self.metadata.iloc[idx].to_dict()
        
        return item
    
    def get_catalog_item_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get catalog item by item_ID.
        
        Args:
            item_id: Item ID to look up
            
        Returns:
            Dict with item data, or None if not found
        """
        if item_id not in self.item_id_to_idx:
            return None
        
        idx = self.item_id_to_idx[item_id]
        return self.get_catalog_item(idx)
    
    def get_info(self) -> Dict[str, Any]:
        """Get engine configuration info."""
        return {
            "catalog_dir": self.catalog_dir,
            "n_items": self.n_items,
            "embedding_dim": self.embedding_dim,
            "exclude_self": self.exclude_self,
            "manifest": self.manifest,
        }

