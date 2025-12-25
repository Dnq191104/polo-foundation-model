"""
Attribute-Based Pair Generation Module

Generates training pairs based on attribute signals for contrastive learning.
Supports strong positives, medium positives, hard negatives, and easy negatives.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Iterator, Set, Any

from datasets import Dataset
from .graphic_policy import GraphicPolicy


class AttributePairGenerator:
    """
    Generates training pairs based on fashion attributes.
    
    Pair types:
    - Strong positive: same category2 AND same material tag
    - Medium positive: same category2 AND share at least 1 design tag
    - Hard negative: same category2 but different material
    - Easy negative: different category2
    
    Designed for contrastive learning where the model should:
    - Pull together items with similar materials and design
    - Push apart items with different materials (even in same category)
    """
    
    def __init__(
        self,
        unknown_value: str = "unknown",
        seed: int = 42,
        design_attributes: Optional[List[str]] = None,
        graphic_policy: Optional[GraphicPolicy] = None
    ):
        """
        Initialize pair generator.
        
        Args:
            unknown_value: Value indicating no tag found (excluded from matching)
            seed: Random seed for reproducibility
            design_attributes: List of design attribute names (default: pattern, neckline, sleeve)
            graphic_policy: Policy for handling category2='graphic' (default: pattern-only)
        """
        self.unknown_value = unknown_value
        self.seed = seed
        self.rng = random.Random(seed)
        
        self.design_attributes = design_attributes or ['pattern', 'neckline', 'sleeve']
        self.graphic_policy = graphic_policy or GraphicPolicy(remap_to_pattern=True)
    
    def _build_index(
        self,
        dataset: Dataset,
        category_col: str = 'category2',
        material_col: str = 'attr_material_primary'
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Build index mapping (category, material) -> list of indices.
        
        Applies graphic policy: items with category2='graphic' are indexed
        separately but excluded from category-based matching.
        
        Args:
            dataset: Dataset with extracted attributes
            category_col: Category column name
            material_col: Material attribute column name
            
        Returns:
            Nested dict: category -> material -> [indices]
        """
        index = defaultdict(lambda: defaultdict(list))
        
        for i in range(len(dataset)):
            category = dataset[i][category_col]
            
            # Skip graphic items for category-based indexing
            if self.graphic_policy.should_exclude_for_category_matching(category):
                continue
            
            material = dataset[i].get(material_col, self.unknown_value)
            index[category][material].append(i)
        
        return index
    
    def _get_design_tags(
        self,
        item: Dict[str, Any],
        use_primary: bool = False
    ) -> Set[str]:
        """
        Get all design tags from an item.
        
        Args:
            item: Dataset item
            use_primary: Use primary tags only (vs full list)
            
        Returns:
            Set of design tag strings
        """
        tags = set()
        
        for attr_name in self.design_attributes:
            if use_primary:
                col = f'attr_{attr_name}_primary'
                value = item.get(col, self.unknown_value)
                if value and value != self.unknown_value:
                    tags.add(f"{attr_name}:{value}")
            else:
                col = f'attr_{attr_name}'
                values = item.get(col, [])
                if isinstance(values, list):
                    for v in values:
                        if v and v != self.unknown_value:
                            tags.add(f"{attr_name}:{v}")
        
        return tags
    
    def _share_design_tag(
        self,
        item1: Dict[str, Any],
        item2: Dict[str, Any]
    ) -> bool:
        """Check if two items share at least one design tag."""
        tags1 = self._get_design_tags(item1)
        tags2 = self._get_design_tags(item2)
        return bool(tags1 & tags2)  # Non-empty intersection
    
    def generate_strong_positives(
        self,
        dataset: Dataset,
        n_pairs: Optional[int] = None,
        category_col: str = 'category2',
        material_col: str = 'attr_material_primary'
    ) -> List[Tuple[int, int]]:
        """
        Generate strong positive pairs: same category2 AND same material.
        
        Graphic policy: Excludes category2='graphic' items from category-based matching.
        
        Args:
            dataset: Dataset with extracted attributes
            n_pairs: Max number of pairs (None for all possible)
            category_col: Category column name
            material_col: Material attribute column name
            
        Returns:
            List of (anchor_idx, positive_idx) tuples
        """
        index = self._build_index(dataset, category_col, material_col)
        pairs = []
        
        for category, material_dict in index.items():
            for material, indices in material_dict.items():
                # Skip unknown materials
                if material == self.unknown_value:
                    continue
                
                # Need at least 2 items for a pair
                if len(indices) < 2:
                    continue
                
                # Generate pairs within this group
                for i, idx1 in enumerate(indices):
                    for idx2 in indices[i + 1:]:
                        # Double-check graphic policy (already filtered in index, but be explicit)
                        cat1 = dataset[idx1][category_col]
                        cat2 = dataset[idx2][category_col]
                        if self.graphic_policy.validate_pair_categories(cat1, cat2, 'positive'):
                            pairs.append((idx1, idx2))
        
        # Shuffle and limit
        self.rng.shuffle(pairs)
        if n_pairs is not None:
            pairs = pairs[:n_pairs]
        
        return pairs
    
    def generate_medium_positives(
        self,
        dataset: Dataset,
        n_pairs: Optional[int] = None,
        category_col: str = 'category2',
        exclude_strong: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Generate medium positive pairs: same category2 AND share design tag.
        
        Args:
            dataset: Dataset with extracted attributes
            n_pairs: Max number of pairs
            category_col: Category column name
            exclude_strong: Exclude pairs that would also be strong positives
            
        Returns:
            List of (anchor_idx, positive_idx) tuples
        """
        pairs = []
        
        # Group by category
        category_indices = defaultdict(list)
        for i in range(len(dataset)):
            category = dataset[i][category_col]
            category_indices[category].append(i)
        
        for category, indices in category_indices.items():
            if len(indices) < 2:
                continue
            
            # Sample pairs to check (full enumeration is expensive)
            max_check = min(len(indices) * 10, len(indices) * (len(indices) - 1) // 2)
            candidates = []
            
            for _ in range(max_check):
                idx1, idx2 = self.rng.sample(indices, 2)
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                candidates.append((idx1, idx2))
            
            # Deduplicate
            candidates = list(set(candidates))
            
            for idx1, idx2 in candidates:
                item1 = dataset[idx1]
                item2 = dataset[idx2]
                
                # Check if they share design tags
                if not self._share_design_tag(item1, item2):
                    continue
                
                # Optionally exclude strong positives (same material)
                if exclude_strong:
                    mat1 = item1.get('attr_material_primary', self.unknown_value)
                    mat2 = item2.get('attr_material_primary', self.unknown_value)
                    if mat1 == mat2 and mat1 != self.unknown_value:
                        continue
                
                pairs.append((idx1, idx2))
        
        # Shuffle and limit
        self.rng.shuffle(pairs)
        if n_pairs is not None:
            pairs = pairs[:n_pairs]
        
        return pairs
    
    def generate_hard_negatives(
        self,
        dataset: Dataset,
        n_pairs: Optional[int] = None,
        category_col: str = 'category2',
        material_col: str = 'attr_material_primary'
    ) -> List[Tuple[int, int]]:
        """
        Generate hard negative pairs: same category2 but different material.
        
        These are "hard" because the items look similar (same category) but
        have different materials, forcing the model to learn material distinctions.
        
        Args:
            dataset: Dataset with extracted attributes
            n_pairs: Max number of pairs
            category_col: Category column name
            material_col: Material attribute column name
            
        Returns:
            List of (anchor_idx, negative_idx) tuples
        """
        index = self._build_index(dataset, category_col, material_col)
        pairs = []
        
        for category, material_dict in index.items():
            # Get materials with known values
            materials = [m for m in material_dict.keys() if m != self.unknown_value]
            
            if len(materials) < 2:
                continue
            
            # Generate pairs across different materials
            for i, mat1 in enumerate(materials):
                for mat2 in materials[i + 1:]:
                    indices1 = material_dict[mat1]
                    indices2 = material_dict[mat2]
                    
                    # Sample pairs between the two material groups
                    n_sample = min(10, len(indices1), len(indices2))
                    sample1 = self.rng.sample(indices1, n_sample)
                    sample2 = self.rng.sample(indices2, n_sample)
                    
                    for idx1, idx2 in zip(sample1, sample2):
                        pairs.append((idx1, idx2))
        
        # Shuffle and limit
        self.rng.shuffle(pairs)
        if n_pairs is not None:
            pairs = pairs[:n_pairs]
        
        return pairs
    
    def generate_easy_negatives(
        self,
        dataset: Dataset,
        n_pairs: Optional[int] = None,
        category_col: str = 'category2'
    ) -> List[Tuple[int, int]]:
        """
        Generate easy negative pairs: different category2.
        
        These are "easy" because the items are visually different categories,
        which the model should easily distinguish.
        
        Graphic policy: Excludes pairs where one item is category2='graphic'.
        
        Args:
            dataset: Dataset with extracted attributes
            n_pairs: Max number of pairs
            category_col: Category column name
            
        Returns:
            List of (anchor_idx, negative_idx) tuples
        """
        # Group by category, excluding graphic
        category_indices = defaultdict(list)
        for i in range(len(dataset)):
            category = dataset[i][category_col]
            # Skip graphic items
            if not self.graphic_policy.should_exclude_for_category_matching(category):
                category_indices[category].append(i)
        
        categories = list(category_indices.keys())
        
        if len(categories) < 2:
            return []
        
        pairs = []
        
        # Generate pairs across categories
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i + 1:]:
                indices1 = category_indices[cat1]
                indices2 = category_indices[cat2]
                
                # Sample pairs
                n_sample = min(5, len(indices1), len(indices2))
                sample1 = self.rng.sample(indices1, n_sample)
                sample2 = self.rng.sample(indices2, n_sample)
                
                for idx1, idx2 in zip(sample1, sample2):
                    # Validate with graphic policy
                    if self.graphic_policy.validate_pair_categories(cat1, cat2, 'negative'):
                        pairs.append((idx1, idx2))
        
        # Shuffle and limit
        self.rng.shuffle(pairs)
        if n_pairs is not None:
            pairs = pairs[:n_pairs]
        
        return pairs
    
    def generate_training_batch(
        self,
        dataset: Dataset,
        n_strong: int = 100,
        n_medium: int = 100,
        n_hard: int = 100,
        n_easy: int = 100
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Generate a balanced batch of training pairs.
        
        Args:
            dataset: Dataset with extracted attributes
            n_strong: Number of strong positive pairs
            n_medium: Number of medium positive pairs
            n_hard: Number of hard negative pairs
            n_easy: Number of easy negative pairs
            
        Returns:
            Dict with 'strong_positive', 'medium_positive', 'hard_negative', 'easy_negative' keys
        """
        return {
            'strong_positive': self.generate_strong_positives(dataset, n_strong),
            'medium_positive': self.generate_medium_positives(dataset, n_medium),
            'hard_negative': self.generate_hard_negatives(dataset, n_hard),
            'easy_negative': self.generate_easy_negatives(dataset, n_easy),
        }
    
    def generate_triplets(
        self,
        dataset: Dataset,
        n_triplets: int = 1000,
        hard_negative_ratio: float = 0.5
    ) -> List[Tuple[int, int, int]]:
        """
        Generate training triplets (anchor, positive, negative).
        
        Uses strong positives and a mix of hard/easy negatives.
        
        Args:
            dataset: Dataset with extracted attributes
            n_triplets: Number of triplets to generate
            hard_negative_ratio: Proportion of hard vs easy negatives
            
        Returns:
            List of (anchor_idx, positive_idx, negative_idx) tuples
        """
        # Generate pairs
        strong_pairs = self.generate_strong_positives(dataset, n_triplets * 2)
        hard_negs = self.generate_hard_negatives(dataset, n_triplets)
        easy_negs = self.generate_easy_negatives(dataset, n_triplets)
        
        # Convert hard/easy to sets for lookup
        all_hard_neg_indices = set()
        for idx1, idx2 in hard_negs:
            all_hard_neg_indices.add(idx1)
            all_hard_neg_indices.add(idx2)
        
        all_easy_neg_indices = set()
        for idx1, idx2 in easy_negs:
            all_easy_neg_indices.add(idx1)
            all_easy_neg_indices.add(idx2)
        
        triplets = []
        
        for anchor_idx, positive_idx in strong_pairs[:n_triplets]:
            # Decide hard or easy negative
            use_hard = self.rng.random() < hard_negative_ratio
            
            if use_hard and hard_negs:
                neg_pair = self.rng.choice(hard_negs)
                neg_idx = neg_pair[0] if neg_pair[1] == anchor_idx else neg_pair[1]
            elif easy_negs:
                neg_pair = self.rng.choice(easy_negs)
                neg_idx = neg_pair[0] if neg_pair[1] == anchor_idx else neg_pair[1]
            else:
                continue
            
            triplets.append((anchor_idx, positive_idx, neg_idx))
        
        return triplets
    
    def get_pair_stats(
        self,
        dataset: Dataset
    ) -> Dict[str, Any]:
        """
        Get statistics about possible pairs in the dataset.
        
        Args:
            dataset: Dataset with extracted attributes
            
        Returns:
            Dict with pair generation statistics
        """
        index = self._build_index(dataset)
        
        stats = {
            'total_samples': len(dataset),
            'categories': len(index),
            'materials_per_category': {},
            'potential_strong_pairs': 0,
            'potential_hard_pairs': 0,
        }
        
        for category, material_dict in index.items():
            non_unknown_materials = [m for m in material_dict.keys() if m != self.unknown_value]
            stats['materials_per_category'][category] = len(non_unknown_materials)
            
            # Strong pairs: within same material group
            for material, indices in material_dict.items():
                if material != self.unknown_value and len(indices) >= 2:
                    n = len(indices)
                    stats['potential_strong_pairs'] += n * (n - 1) // 2
            
            # Hard pairs: across different materials
            if len(non_unknown_materials) >= 2:
                for i, mat1 in enumerate(non_unknown_materials):
                    for mat2 in non_unknown_materials[i + 1:]:
                        n1 = len(material_dict[mat1])
                        n2 = len(material_dict[mat2])
                        stats['potential_hard_pairs'] += n1 * n2
        
        return stats


def create_pair_generator(
    unknown_value: str = "unknown",
    seed: int = 42
) -> AttributePairGenerator:
    """
    Factory function to create an AttributePairGenerator.
    
    Args:
        unknown_value: Value indicating no tag found
        seed: Random seed
        
    Returns:
        Configured AttributePairGenerator instance
    """
    return AttributePairGenerator(unknown_value=unknown_value, seed=seed)

