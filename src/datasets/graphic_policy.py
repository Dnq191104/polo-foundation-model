"""
Graphic Category Policy for Step 7

Implements policy to treat category2='graphic' as pattern only, not as a category label.
This avoids taxonomy noise where 'graphic' is really a pattern descriptor.
"""

from typing import Dict, Any, List


class GraphicPolicy:
    """
    Policy handler for category2='graphic' items.
    
    Policy: Treat 'graphic' as a pattern-only signal, never as a category label.
    
    Implementation:
    - Never use 'graphic' as a positive/negative for category matching
    - Remap category2='graphic' to pattern attribute if missing
    - Optionally exclude 'graphic' queries from Category@10 metrics
    """
    
    GRAPHIC_CATEGORY = 'graphic'
    
    def __init__(self, remap_to_pattern: bool = True):
        """
        Initialize graphic policy.
        
        Args:
            remap_to_pattern: If True, remap category2='graphic' to attr_pattern_primary
        """
        self.remap_to_pattern = remap_to_pattern
    
    def is_graphic_category(self, category2: str) -> bool:
        """Check if category2 is 'graphic'."""
        return category2 and category2.lower() == self.GRAPHIC_CATEGORY
    
    def should_exclude_for_category_matching(self, category2: str) -> bool:
        """
        Check if item should be excluded from category-based matching.
        
        For pair generation, this means:
        - Don't create positives where both items have category2='graphic'
        - Don't create negatives based on category2='graphic' difference
        
        Args:
            category2: Category value
            
        Returns:
            True if should be excluded from category matching
        """
        return self.is_graphic_category(category2)
    
    def apply_pattern_remapping(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply pattern remapping to item if category2='graphic'.
        
        If category2='graphic' and attr_pattern_primary is 'unknown',
        set attr_pattern_primary='graphic'.
        
        Args:
            item: Dataset item (will be modified in place)
            
        Returns:
            Modified item
        """
        if not self.remap_to_pattern:
            return item
        
        category2 = item.get('category2', '')
        if self.is_graphic_category(category2):
            # Check if pattern is unknown
            pattern_primary = item.get('attr_pattern_primary', 'unknown')
            if pattern_primary == 'unknown':
                item['attr_pattern_primary'] = 'graphic'
        
        return item
    
    def filter_for_category_metrics(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out graphic items for category-based metrics.
        
        Use this to exclude 'graphic' queries from Category@10 evaluation
        to avoid taxonomy noise.
        
        Args:
            items: List of items
            
        Returns:
            Filtered list excluding graphic items
        """
        return [
            item for item in items
            if not self.is_graphic_category(item.get('category2', ''))
        ]
    
    def get_effective_category(self, category2: str) -> str:
        """
        Get effective category for matching purposes.
        
        Returns None for 'graphic' to indicate it should not be used
        for category-based matching.
        
        Args:
            category2: Original category
            
        Returns:
            Effective category for matching (None for 'graphic')
        """
        if self.is_graphic_category(category2):
            return None
        return category2
    
    def validate_pair_categories(
        self,
        cat1: str,
        cat2: str,
        pair_type: str
    ) -> bool:
        """
        Validate if a pair should be created based on categories.
        
        Rules:
        - Positive pairs: Never create if either is 'graphic'
        - Negative pairs: Never create based on 'graphic' difference
        
        Args:
            cat1: First item category
            cat2: Second item category
            pair_type: Type of pair ('positive' or 'negative')
            
        Returns:
            True if pair is valid under graphic policy
        """
        has_graphic = (
            self.is_graphic_category(cat1) or
            self.is_graphic_category(cat2)
        )
        
        if pair_type == 'positive':
            # Don't create positive pairs involving graphic
            return not has_graphic
        
        elif pair_type == 'negative':
            # Don't create negative pairs based on graphic difference
            # If one is graphic and other isn't, this is not a valid negative
            if has_graphic and not (
                self.is_graphic_category(cat1) and
                self.is_graphic_category(cat2)
            ):
                return False
            return True
        
        return True


def create_graphic_policy(remap_to_pattern: bool = True) -> GraphicPolicy:
    """
    Factory function to create GraphicPolicy.
    
    Args:
        remap_to_pattern: Whether to remap graphic to pattern attribute
        
    Returns:
        Configured GraphicPolicy instance
    """
    return GraphicPolicy(remap_to_pattern=remap_to_pattern)

