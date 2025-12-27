"""
Balanced Sampler for Step 7 Training

Oversamples weak classes, rare materials, and neckline-known samples.
Caps dominant groups to prevent overfitting on common patterns.
"""

import random
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
import functools
import hashlib
import pickle
import os

import pandas as pd
import numpy as np
from datasets import Dataset


def cache_result(cache_dir='artifacts/cache'):
    """Cache expensive deterministic functions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name + args
            key_data = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            key = hashlib.md5(key_data.encode()).hexdigest()
            cache_path = f"{cache_dir}/{key}.pkl"

            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)

            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)

            return result
        return wrapper
    return decorator


class OptimizedItemSampler:
    """
    On-the-fly balanced sampling without pre-generated pairs.

    Benefits:
    - No expensive pair generation
    - Dynamic balancing weights
    - Minimal memory usage
    - Fast iteration
    """

    def __init__(self, dataset, batch_size=64, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Build cached indices
        self.category_to_ids = self._build_category_index()
        self.material_to_ids = self._build_material_index()

        # Default balanced weights
        self.category_weights = self._compute_balanced_weights()
        self.material_weights = None  # Can be set dynamically

    @cache_result()
    def _build_category_index(self):
        """Build category → [item_ids] mapping"""
        index = {}
        for i, item in enumerate(self.dataset):
            category = item.get('category2', 'unknown')
            if category not in index:
                index[category] = []
            index[category].append(i)
        return index

    @cache_result()
    def _build_material_index(self):
        """Build material → [item_ids] mapping"""
        index = {}
        for i, item in enumerate(self.dataset):
            material = item.get('attr_material_primary', 'unknown')
            if material not in index:
                index[material] = []
            index[material].append(i)
        return index

    def _compute_balanced_weights(self):
        """Compute balanced weights for categories"""
        categories = list(self.category_to_ids.keys())
        n_categories = len(categories)

        # Start with uniform weights
        weights = {cat: 1.0 / n_categories for cat in categories}

        # Boost weak categories
        weak_categories = ['shorts', 'rompers', 'cardigans', 'graphic']
        for cat in weak_categories:
            if cat in weights:
                weights[cat] *= 3.0  # 3x boost

        # Cap dominant categories
        dominant_categories = ['tees', 't-shirts']
        for cat in dominant_categories:
            if cat in weights:
                weights[cat] *= 0.3  # Reduce by 70%

        # Renormalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        return weights

    def update_curriculum_weights(self, epoch, max_epochs=10):
        """Update weights based on curriculum schedule"""
        progress = epoch / max_epochs

        if progress < 0.2:  # Epochs 1-2: Easy categories only
            easy_cats = ['tees', 'sweaters', 'jackets']
            weights = {cat: (1.0 if cat in easy_cats else 0.0)
                      for cat in self.category_weights.keys()}
        elif progress < 0.4:  # Epochs 3-4: Add medium
            medium_cats = ['tees', 'sweaters', 'jackets', 'dresses', 'blouses']
            weights = {cat: (0.6 if cat in ['tees', 'sweaters', 'jackets'] else
                           0.3 if cat in ['dresses', 'blouses'] else 0.05)
                      for cat in self.category_weights.keys()}
        else:  # Epochs 5+: All balanced
            weights = self._compute_balanced_weights()

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}

        self.category_weights = weights

    def sample_batch(self):
        """Sample a balanced batch on-the-fly"""
        batch_pairs = []

        # Sample categories according to current weights
        categories = list(self.category_weights.keys())
        weights = list(self.category_weights.values())

        sampled_categories = self.rng.choice(
            categories,
            size=self.batch_size // 2,  # Half batch for positive pairs
            p=weights
        )

        for category in sampled_categories:
            cat_ids = self.category_to_ids.get(category, [])
            if len(cat_ids) >= 2:
                # Sample positive pair within category
                anchor_id, pos_id = self.rng.choice(cat_ids, 2, replace=False)
                batch_pairs.append((anchor_id, pos_id, 1))  # 1 = positive label

        # Sample negative pairs from different categories
        all_categories = list(self.category_to_ids.keys())
        for _ in range(self.batch_size // 2):
            # Pick two different categories
            cat1, cat2 = self.rng.choice(all_categories, 2, replace=False)

            cat1_ids = self.category_to_ids.get(cat1, [])
            cat2_ids = self.category_to_ids.get(cat2, [])

            if cat1_ids and cat2_ids:
                anchor_id = self.rng.choice(cat1_ids)
                neg_id = self.rng.choice(cat2_ids)
                batch_pairs.append((anchor_id, neg_id, 0))  # 0 = negative label

        return batch_pairs


class BalancedPairSampler:
    """
    Balanced batch sampler for pair dataset.
    
    Strategy:
    - Oversample weak categories (cardigans, shorts)
    - Oversample rare materials (denim, leather)
    - Oversample neckline-known samples
    - Cap dominant groups (cotton, tees)
    """
    
    # Target categories to boost
    WEAK_CATEGORIES = ['cardigans', 'shorts']
    RARE_MATERIALS = ['denim', 'leather']
    
    # Categories/materials to cap
    DOMINANT_CATEGORIES = ['tees', 't-shirts']
    DOMINANT_MATERIALS = ['cotton']
    
    def __init__(
        self,
        pair_df: pd.DataFrame,
        batch_size: int = 64,
        weak_boost: float = 3.0,
        rare_boost: float = 3.0,
        neckline_boost: float = 2.0,
        dominant_cap: float = 0.3,
        seed: int = 42
    ):
        """
        Initialize balanced sampler.
        
        Args:
            pair_df: Pair dataset DataFrame
            batch_size: Batch size
            weak_boost: Multiplier for weak categories
            rare_boost: Multiplier for rare materials
            neckline_boost: Multiplier for neckline-known samples
            dominant_cap: Max fraction of batch from dominant groups
            seed: Random seed
        """
        self.pair_df = pair_df
        self.batch_size = batch_size
        self.weak_boost = weak_boost
        self.rare_boost = rare_boost
        self.neckline_boost = neckline_boost
        self.dominant_cap = dominant_cap
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Build sampling weights
        self._build_weights()
        
        # Track stats
        self.batch_stats = []
    
    def _build_weights(self):
        """Build sampling weights for each pair."""
        weights = np.ones(len(self.pair_df))
        
        for i, row in self.pair_df.iterrows():
            weight = 1.0
            
            anchor_cat = row.get('anchor_category', '')
            anchor_mat = row.get('anchor_material', 'unknown')
            anchor_neck = row.get('anchor_neckline', 'unknown')
            
            # Boost weak categories
            if anchor_cat in self.WEAK_CATEGORIES:
                weight *= self.weak_boost
            
            # Boost rare materials
            if anchor_mat in self.RARE_MATERIALS:
                weight *= self.rare_boost
            
            # Boost neckline-known
            if anchor_neck != 'unknown':
                weight *= self.neckline_boost
            
            # Cap dominant groups
            if anchor_cat in self.DOMINANT_CATEGORIES:
                weight *= (1.0 / self.weak_boost)  # Reduce
            
            if anchor_mat in self.DOMINANT_MATERIALS:
                weight *= (1.0 / self.rare_boost)  # Reduce
            
            weights[i] = weight
        
        # Normalize
        weights = weights / weights.sum()
        
        self.weights = weights
        
        print(f"Sampling weights built: min={weights.min():.6f}, max={weights.max():.6f}")
    
    def sample_batch(self) -> pd.DataFrame:
        """
        Sample a balanced batch.
        
        Returns:
            Batch DataFrame
        """
        # Sample with replacement using weights
        indices = self.rng.choices(
            range(len(self.pair_df)),
            weights=self.weights,
            k=self.batch_size
        )
        
        batch = self.pair_df.iloc[indices].copy()
        
        # Compute and log stats
        stats = self._compute_batch_stats(batch)
        self.batch_stats.append(stats)
        
        return batch
    
    def _compute_batch_stats(self, batch: pd.DataFrame) -> Dict[str, Any]:
        """Compute batch statistics."""
        stats = {
            'batch_size': len(batch),
            'positive_rate': (batch['label'] == 1).sum() / len(batch),
        }
        
        # Category distribution
        cat_counts = batch['anchor_category'].value_counts()
        stats['categories'] = cat_counts.to_dict()
        stats['weak_category_count'] = sum(
            cat_counts.get(cat, 0) for cat in self.WEAK_CATEGORIES
        )
        
        # Material distribution
        mat_counts = batch['anchor_material'].value_counts()
        stats['materials'] = mat_counts.to_dict()
        stats['rare_material_count'] = sum(
            mat_counts.get(mat, 0) for mat in self.RARE_MATERIALS
        )
        
        # Neckline coverage
        neck_known = (batch['anchor_neckline'] != 'unknown').sum()
        stats['neckline_known_count'] = neck_known
        stats['neckline_known_rate'] = neck_known / len(batch)
        
        return stats
    
    def get_epoch_stats(self) -> Dict[str, Any]:
        """Get aggregated stats for current epoch."""
        if not self.batch_stats:
            return {}
        
        # Aggregate across batches
        total_batches = len(self.batch_stats)
        
        avg_positive_rate = np.mean([s['positive_rate'] for s in self.batch_stats])
        avg_weak_cat = np.mean([s['weak_category_count'] for s in self.batch_stats])
        avg_rare_mat = np.mean([s['rare_material_count'] for s in self.batch_stats])
        avg_neck_rate = np.mean([s['neckline_known_rate'] for s in self.batch_stats])
        
        # Category distribution
        all_cats = defaultdict(int)
        for stats in self.batch_stats:
            for cat, count in stats['categories'].items():
                all_cats[cat] += count
        
        # Material distribution
        all_mats = defaultdict(int)
        for stats in self.batch_stats:
            for mat, count in stats['materials'].items():
                all_mats[mat] += count
        
        return {
            'total_batches': total_batches,
            'avg_positive_rate': avg_positive_rate,
            'avg_weak_category_per_batch': avg_weak_cat,
            'avg_rare_material_per_batch': avg_rare_mat,
            'avg_neckline_known_rate': avg_neck_rate,
            'category_distribution': dict(all_cats),
            'material_distribution': dict(all_mats),
        }
    
    def reset_epoch_stats(self):
        """Reset stats for new epoch."""
        self.batch_stats = []
    
    def log_batch_stats(self, batch_idx: int):
        """Log current batch stats."""
        if not self.batch_stats or batch_idx >= len(self.batch_stats):
            return
        
        stats = self.batch_stats[batch_idx]
        
        print(f"  Batch {batch_idx}:")
        print(f"    Positive rate: {stats['positive_rate']:.2f}")
        print(f"    Weak categories: {stats['weak_category_count']}")
        print(f"    Rare materials: {stats['rare_material_count']}")
        print(f"    Neckline known: {stats['neckline_known_count']} ({stats['neckline_known_rate']:.2f})")


class SequentialPairSampler:
    """
    Simple sequential sampler for validation.
    
    Just iterates through pairs in order.
    """
    
    def __init__(self, pair_df: pd.DataFrame, batch_size: int = 64):
        """
        Initialize sequential sampler.
        
        Args:
            pair_df: Pair dataset DataFrame
            batch_size: Batch size
        """
        self.pair_df = pair_df
        self.batch_size = batch_size
        self.current_idx = 0
    
    def sample_batch(self) -> Optional[pd.DataFrame]:
        """
        Sample next batch sequentially.
        
        Returns:
            Batch DataFrame or None if end reached
        """
        if self.current_idx >= len(self.pair_df):
            return None
        
        end_idx = min(self.current_idx + self.batch_size, len(self.pair_df))
        batch = self.pair_df.iloc[self.current_idx:end_idx].copy()
        
        self.current_idx = end_idx
        
        return batch
    
    def reset(self):
        """Reset to beginning."""
        self.current_idx = 0


def create_balanced_sampler(
    pair_df: pd.DataFrame,
    batch_size: int = 64,
    weak_boost: float = 3.0,
    rare_boost: float = 3.0,
    neckline_boost: float = 2.0,
    seed: int = 42
) -> BalancedPairSampler:
    """
    Factory function to create BalancedPairSampler.
    
    Args:
        pair_df: Pair dataset DataFrame
        batch_size: Batch size
        weak_boost: Weak category boost
        rare_boost: Rare material boost
        neckline_boost: Neckline-known boost
        seed: Random seed
        
    Returns:
        Configured BalancedPairSampler instance
    """
    return BalancedPairSampler(
        pair_df=pair_df,
        batch_size=batch_size,
        weak_boost=weak_boost,
        rare_boost=rare_boost,
        neckline_boost=neckline_boost,
        seed=seed
    )

