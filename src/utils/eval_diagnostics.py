"""
Evaluation Diagnostics Module

Provides functions for evaluating retrieval quality using attribute signals.
Computes material/pattern match rates and category-based analysis.
"""

from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any, Tuple, Union

from datasets import Dataset


class RetrievalDiagnostics:
    """
    Computes diagnostic metrics for retrieval results using attribute signals.
    
    Key metrics:
    - Material match rate: % of retrieved items matching query's material
    - Pattern match rate: % of retrieved items matching query's pattern
    - Category match rate: % of retrieved items in same category
    - Sliced analysis by category2
    """
    
    def __init__(
        self,
        unknown_value: str = "unknown",
        attr_names: Optional[List[str]] = None
    ):
        """
        Initialize diagnostics calculator.
        
        Args:
            unknown_value: Value indicating no tag found
            attr_names: Attribute names to track (default: material, pattern)
        """
        self.unknown_value = unknown_value
        self.attr_names = attr_names or ['material', 'pattern', 'neckline', 'sleeve']
    
    def compute_attribute_match_rate(
        self,
        query_tags: Union[str, List[str]],
        retrieved_tags: List[Union[str, List[str]]],
        match_type: str = 'any'
    ) -> float:
        """
        Compute the rate of attribute match between query and retrieved items.
        
        Args:
            query_tags: Query item's tags (single or list)
            retrieved_tags: Retrieved items' tags (list of single/list)
            match_type: 'any' (share any tag) or 'primary' (exact primary match)
            
        Returns:
            Match rate as fraction (0.0 to 1.0)
        """
        if not retrieved_tags:
            return 0.0
        
        # Normalize query tags to set
        if isinstance(query_tags, str):
            query_set = {query_tags}
        elif isinstance(query_tags, list):
            query_set = set(query_tags)
        else:
            query_set = set()
        
        # Remove unknown
        query_set.discard(self.unknown_value)
        
        if not query_set:
            # Query has no tags - can't compute meaningful match
            return 0.0
        
        matches = 0
        for ret_tags in retrieved_tags:
            # Normalize retrieved tags
            if isinstance(ret_tags, str):
                ret_set = {ret_tags}
            elif isinstance(ret_tags, list):
                ret_set = set(ret_tags)
            else:
                ret_set = set()
            
            ret_set.discard(self.unknown_value)
            
            if match_type == 'any':
                # Match if any overlap
                if query_set & ret_set:
                    matches += 1
            elif match_type == 'primary':
                # Match if first element matches
                query_primary = list(query_set)[0] if query_set else None
                ret_primary = list(ret_set)[0] if ret_set else None
                if query_primary and query_primary == ret_primary:
                    matches += 1
        
        return matches / len(retrieved_tags)
    
    def compute_material_match_rate(
        self,
        query_item: Dict[str, Any],
        retrieved_items: List[Dict[str, Any]],
        use_primary: bool = True
    ) -> float:
        """
        Compute material match rate for retrieval results.
        
        Args:
            query_item: Query item with attr_material or attr_material_primary
            retrieved_items: List of retrieved items
            use_primary: Use primary tag only
            
        Returns:
            Match rate as fraction (0.0 to 1.0)
        """
        col = 'attr_material_primary' if use_primary else 'attr_material'
        
        query_tags = query_item.get(col, self.unknown_value)
        retrieved_tags = [item.get(col, self.unknown_value) for item in retrieved_items]
        
        match_type = 'primary' if use_primary else 'any'
        return self.compute_attribute_match_rate(query_tags, retrieved_tags, match_type)
    
    def compute_pattern_match_rate(
        self,
        query_item: Dict[str, Any],
        retrieved_items: List[Dict[str, Any]],
        use_primary: bool = True
    ) -> float:
        """
        Compute pattern match rate for retrieval results.
        
        Args:
            query_item: Query item with attr_pattern or attr_pattern_primary
            retrieved_items: List of retrieved items
            use_primary: Use primary tag only
            
        Returns:
            Match rate as fraction (0.0 to 1.0)
        """
        col = 'attr_pattern_primary' if use_primary else 'attr_pattern'
        
        query_tags = query_item.get(col, self.unknown_value)
        retrieved_tags = [item.get(col, self.unknown_value) for item in retrieved_items]
        
        match_type = 'primary' if use_primary else 'any'
        return self.compute_attribute_match_rate(query_tags, retrieved_tags, match_type)
    
    def compute_category_match_rate(
        self,
        query_item: Dict[str, Any],
        retrieved_items: List[Dict[str, Any]],
        category_col: str = 'category2'
    ) -> float:
        """
        Compute category match rate for retrieval results.
        
        Args:
            query_item: Query item
            retrieved_items: List of retrieved items
            category_col: Category column name
            
        Returns:
            Match rate as fraction (0.0 to 1.0)
        """
        if not retrieved_items:
            return 0.0
        
        query_cat = query_item.get(category_col, '')
        if not query_cat:
            return 0.0
        
        matches = sum(1 for item in retrieved_items if item.get(category_col) == query_cat)
        return matches / len(retrieved_items)
    
    def evaluate_retrieval(
        self,
        query_item: Dict[str, Any],
        retrieved_items: List[Dict[str, Any]],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results with all attribute metrics.
        
        Args:
            query_item: Query item
            retrieved_items: List of retrieved items (top-k)
            k: Number of items to evaluate (truncates retrieved_items)
            
        Returns:
            Dict with metric names and values
        """
        # Truncate to k
        top_k = retrieved_items[:k]
        
        metrics = {
            f'material_match@{k}': self.compute_material_match_rate(query_item, top_k),
            f'pattern_match@{k}': self.compute_pattern_match_rate(query_item, top_k),
            f'category_match@{k}': self.compute_category_match_rate(query_item, top_k),
        }
        
        # Add other attributes
        for attr_name in self.attr_names:
            if attr_name not in ['material', 'pattern']:
                col = f'attr_{attr_name}_primary'
                query_tags = query_item.get(col, self.unknown_value)
                retrieved_tags = [item.get(col, self.unknown_value) for item in top_k]
                metrics[f'{attr_name}_match@{k}'] = self.compute_attribute_match_rate(
                    query_tags, retrieved_tags, 'primary'
                )
        
        return metrics
    
    def evaluate_batch(
        self,
        query_items: List[Dict[str, Any]],
        retrieved_items_list: List[List[Dict[str, Any]]],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate a batch of retrieval results.
        
        Args:
            query_items: List of query items
            retrieved_items_list: List of retrieval results per query
            k: Number of items to evaluate per query
            
        Returns:
            Dict with averaged metrics
        """
        if len(query_items) != len(retrieved_items_list):
            raise ValueError("Query and retrieved lists must have same length")
        
        if not query_items:
            return {}
        
        # Collect metrics for each query
        all_metrics = defaultdict(list)
        
        for query_item, retrieved_items in zip(query_items, retrieved_items_list):
            metrics = self.evaluate_retrieval(query_item, retrieved_items, k)
            for name, value in metrics.items():
                all_metrics[name].append(value)
        
        # Average
        avg_metrics = {
            name: sum(values) / len(values)
            for name, values in all_metrics.items()
        }
        
        return avg_metrics
    
    def slice_by_category(
        self,
        query_items: List[Dict[str, Any]],
        retrieved_items_list: List[List[Dict[str, Any]]],
        k: int = 10,
        category_col: str = 'category2'
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval results sliced by query category.
        
        Args:
            query_items: List of query items
            retrieved_items_list: List of retrieval results per query
            k: Number of items to evaluate per query
            category_col: Category column name
            
        Returns:
            Dict mapping category to metrics dict
        """
        # Group by category
        category_groups = defaultdict(lambda: {'queries': [], 'results': []})
        
        for query_item, retrieved_items in zip(query_items, retrieved_items_list):
            category = query_item.get(category_col, 'unknown')
            category_groups[category]['queries'].append(query_item)
            category_groups[category]['results'].append(retrieved_items)
        
        # Evaluate each category
        sliced_metrics = {}
        for category, data in category_groups.items():
            metrics = self.evaluate_batch(data['queries'], data['results'], k)
            metrics['n_queries'] = len(data['queries'])
            sliced_metrics[category] = metrics
        
        return sliced_metrics
    
    def slice_by_material(
        self,
        query_items: List[Dict[str, Any]],
        retrieved_items_list: List[List[Dict[str, Any]]],
        k: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval results sliced by query material.
        
        Args:
            query_items: List of query items
            retrieved_items_list: List of retrieval results per query
            k: Number of items to evaluate per query
            
        Returns:
            Dict mapping material to metrics dict
        """
        # Group by material
        material_groups = defaultdict(lambda: {'queries': [], 'results': []})
        
        for query_item, retrieved_items in zip(query_items, retrieved_items_list):
            material = query_item.get('attr_material_primary', self.unknown_value)
            material_groups[material]['queries'].append(query_item)
            material_groups[material]['results'].append(retrieved_items)
        
        # Evaluate each material
        sliced_metrics = {}
        for material, data in material_groups.items():
            if material == self.unknown_value:
                continue  # Skip unknown
            metrics = self.evaluate_batch(data['queries'], data['results'], k)
            metrics['n_queries'] = len(data['queries'])
            sliced_metrics[material] = metrics
        
        return sliced_metrics
    
    def generate_evaluation_report(
        self,
        query_items: List[Dict[str, Any]],
        retrieved_items_list: List[List[Dict[str, Any]]],
        k: int = 10,
        include_category_slice: bool = True,
        include_material_slice: bool = True
    ) -> str:
        """
        Generate a formatted evaluation report.
        
        Args:
            query_items: List of query items
            retrieved_items_list: List of retrieval results per query
            k: Number of items to evaluate per query
            include_category_slice: Include category breakdown
            include_material_slice: Include material breakdown
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("RETRIEVAL EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Total queries: {len(query_items)}")
        lines.append(f"Top-k: {k}")
        lines.append("")
        
        # Overall metrics
        overall = self.evaluate_batch(query_items, retrieved_items_list, k)
        
        lines.append("OVERALL METRICS:")
        lines.append("-" * 40)
        for name, value in sorted(overall.items()):
            lines.append(f"  {name}: {value:.3f}")
        lines.append("")
        
        # Category slice
        if include_category_slice:
            lines.append("BY CATEGORY:")
            lines.append("-" * 40)
            
            sliced = self.slice_by_category(query_items, retrieved_items_list, k)
            # Sort by count
            sorted_cats = sorted(sliced.items(), key=lambda x: -x[1].get('n_queries', 0))
            
            for category, metrics in sorted_cats[:10]:  # Top 10
                n = metrics.get('n_queries', 0)
                mat_match = metrics.get(f'material_match@{k}', 0)
                cat_match = metrics.get(f'category_match@{k}', 0)
                lines.append(f"  {category} (n={n}):")
                lines.append(f"    material_match: {mat_match:.3f}, category_match: {cat_match:.3f}")
            lines.append("")
        
        # Material slice
        if include_material_slice:
            lines.append("BY QUERY MATERIAL:")
            lines.append("-" * 40)
            
            sliced = self.slice_by_material(query_items, retrieved_items_list, k)
            sorted_mats = sorted(sliced.items(), key=lambda x: -x[1].get('n_queries', 0))
            
            for material, metrics in sorted_mats[:10]:  # Top 10
                n = metrics.get('n_queries', 0)
                mat_match = metrics.get(f'material_match@{k}', 0)
                lines.append(f"  {material} (n={n}): material_match@{k}={mat_match:.3f}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def compute_material_match_rate(
    query_tags: Union[str, List[str]],
    retrieved_tags: List[Union[str, List[str]]],
    unknown_value: str = "unknown"
) -> float:
    """
    Standalone function to compute material match rate.
    
    Args:
        query_tags: Query item's material tag(s)
        retrieved_tags: Retrieved items' material tags
        unknown_value: Value indicating no tag
        
    Returns:
        Match rate as fraction
    """
    diag = RetrievalDiagnostics(unknown_value=unknown_value)
    return diag.compute_attribute_match_rate(query_tags, retrieved_tags)


def compute_pattern_match_rate(
    query_tags: Union[str, List[str]],
    retrieved_tags: List[Union[str, List[str]]],
    unknown_value: str = "unknown"
) -> float:
    """
    Standalone function to compute pattern match rate.
    
    Args:
        query_tags: Query item's pattern tag(s)
        retrieved_tags: Retrieved items' pattern tags
        unknown_value: Value indicating no tag
        
    Returns:
        Match rate as fraction
    """
    diag = RetrievalDiagnostics(unknown_value=unknown_value)
    return diag.compute_attribute_match_rate(query_tags, retrieved_tags)


def slice_results_by_category(
    query_items: List[Dict[str, Any]],
    retrieved_items_list: List[List[Dict[str, Any]]],
    k: int = 10,
    category_col: str = 'category2'
) -> Dict[str, Dict[str, float]]:
    """
    Standalone function to slice results by category.
    
    Args:
        query_items: List of query items
        retrieved_items_list: List of retrieval results
        k: Top-k to evaluate
        category_col: Category column
        
    Returns:
        Dict mapping category to metrics
    """
    diag = RetrievalDiagnostics()
    return diag.slice_by_category(query_items, retrieved_items_list, k, category_col)


def create_diagnostics(
    unknown_value: str = "unknown",
    attr_names: Optional[List[str]] = None
) -> RetrievalDiagnostics:
    """
    Factory function to create RetrievalDiagnostics.
    
    Args:
        unknown_value: Value indicating no tag
        attr_names: Attribute names to track
        
    Returns:
        Configured RetrievalDiagnostics instance
    """
    return RetrievalDiagnostics(unknown_value=unknown_value, attr_names=attr_names)

