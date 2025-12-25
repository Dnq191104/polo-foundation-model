"""
Group Key Derivation for Strong Positives

Derives product group keys from item_ID, text, and other fields to identify
images that belong to the same product (e.g., different views, poses, variants).
"""

import re
from typing import Dict, Any, Tuple, Optional
from urllib.parse import urlparse


class GroupKeyDeriver:
    """
    Derives product group keys using heuristics.
    
    Strategies:
    1. item_ID pattern matching (e.g., remove suffixes like _view1, _A, etc.)
    2. URL/filename stem extraction
    3. SKU-like token extraction from text
    4. Fuzzy title matching
    
    Returns group_key with confidence score and reason.
    """
    
    def __init__(self):
        """Initialize group key deriver."""
        # Patterns for item_ID suffixes that indicate variants
        self.suffix_patterns = [
            r'[-_]([0-9]+)$',  # _0, _1, -001
            r'[-_]([a-zA-Z])$',  # _A, _B, -a
            r'[-_](view|front|back|side|detail|zoom)[-_]?([0-9]+)?$',  # _view1, _front
            r'[-_](model|worn|flat|ghost)[-_]?([0-9]+)?$',  # _model, _worn
            r'[-_](xs|s|m|l|xl|xxl)$',  # Size variants (unlikely but possible)
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.suffix_patterns]
    
    def derive_from_item_id(self, item_id: str) -> Tuple[str, float, str]:
        """
        Derive group key from item_ID by removing view/variant suffixes.
        
        Args:
            item_id: Item ID string
            
        Returns:
            Tuple of (group_key, confidence, reason)
        """
        if not item_id:
            return item_id, 0.0, "empty_item_id"
        
        # Try each pattern
        for pattern in self.compiled_patterns:
            match = pattern.search(item_id)
            if match:
                # Remove the matched suffix
                group_key = pattern.sub('', item_id)
                return group_key, 0.8, f"removed_suffix_{match.group()}"
        
        # No pattern matched - use item_ID as-is with low confidence
        return item_id, 0.3, "no_suffix_pattern"
    
    def derive_from_url(self, url_field: Optional[str]) -> Tuple[Optional[str], float, str]:
        """
        Derive group key from URL by extracting filename stem.
        
        Args:
            url_field: URL or path string
            
        Returns:
            Tuple of (group_key, confidence, reason)
        """
        if not url_field:
            return None, 0.0, "no_url"
        
        try:
            # Parse URL and get path
            parsed = urlparse(url_field)
            path = parsed.path
            
            # Extract filename without extension
            filename = path.split('/')[-1]
            stem = filename.rsplit('.', 1)[0]
            
            if stem and len(stem) > 5:  # Reasonable length
                # Try to remove common suffixes
                for pattern in self.compiled_patterns:
                    match = pattern.search(stem)
                    if match:
                        group_key = pattern.sub('', stem)
                        return group_key, 0.7, f"url_stem_cleaned"
                
                return stem, 0.5, "url_stem_raw"
            
        except Exception:
            pass
        
        return None, 0.0, "url_parse_failed"
    
    def derive_from_text_sku(self, text: str) -> Tuple[Optional[str], float, str]:
        """
        Derive group key from SKU-like tokens in text.
        
        Looks for patterns like:
        - Style #12345
        - SKU: ABC-123
        - Item: 98765
        
        Args:
            text: Product description text
            
        Returns:
            Tuple of (group_key, confidence, reason)
        """
        if not text:
            return None, 0.0, "no_text"
        
        # Common SKU patterns
        sku_patterns = [
            r'(?:style|sku|item|product)\s*[#:\s]\s*([A-Z0-9\-_]+)',
            r'\b([A-Z]{2,}[-_][0-9]{3,})\b',  # ABC-123, XY_456
            r'\b([0-9]{6,})\b',  # Long numeric codes
        ]
        
        for pattern_str in sku_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            match = pattern.search(text)
            if match:
                sku = match.group(1)
                return sku, 0.6, f"sku_pattern_{pattern_str[:20]}"
        
        return None, 0.0, "no_sku_found"
    
    def derive_group_key(
        self,
        item: Dict[str, Any],
        item_id_col: str = 'item_ID',
        url_col: Optional[str] = None,
        text_col: str = 'text'
    ) -> Dict[str, Any]:
        """
        Derive best group key from all available signals.
        
        Strategy:
        1. Try item_ID pattern matching (highest priority)
        2. Try URL stem extraction
        3. Try SKU extraction from text
        4. Fallback to item_ID as-is
        
        Args:
            item: Dataset item
            item_id_col: Column name for item ID
            url_col: Optional column name for URL/path
            text_col: Column name for text description
            
        Returns:
            Dict with keys:
                - group_key: Derived group key
                - confidence: Confidence score (0-1)
                - method: Method used
                - reason: Detailed reason string
        """
        item_id = item.get(item_id_col, '')
        text = item.get(text_col, '')
        url = item.get(url_col, '') if url_col else None
        
        # Try strategies in order
        candidates = []
        
        # 1. Item ID pattern
        id_key, id_conf, id_reason = self.derive_from_item_id(item_id)
        candidates.append({
            'group_key': id_key,
            'confidence': id_conf,
            'method': 'item_id_pattern',
            'reason': id_reason
        })
        
        # 2. URL stem (if available)
        if url:
            url_key, url_conf, url_reason = self.derive_from_url(url)
            if url_key:
                candidates.append({
                    'group_key': url_key,
                    'confidence': url_conf,
                    'method': 'url_stem',
                    'reason': url_reason
                })
        
        # 3. Text SKU
        sku_key, sku_conf, sku_reason = self.derive_from_text_sku(text)
        if sku_key:
            candidates.append({
                'group_key': sku_key,
                'confidence': sku_conf,
                'method': 'text_sku',
                'reason': sku_reason
            })
        
        # Select best by confidence
        best = max(candidates, key=lambda x: x['confidence'])
        
        return best
    
    def build_group_index(
        self,
        dataset,
        min_confidence: float = 0.6
    ) -> Dict[str, Any]:
        """
        Build group index for entire dataset.
        
        Args:
            dataset: HuggingFace dataset
            min_confidence: Minimum confidence to accept a group key
            
        Returns:
            Dict with:
                - index: Dict mapping group_key -> list of indices
                - metadata: Per-item group info
                - stats: Statistics about groups
        """
        from collections import defaultdict, Counter
        
        group_index = defaultdict(list)
        metadata = []
        
        for i in range(len(dataset)):
            item = dataset[i]
            group_info = self.derive_group_key(item)
            
            # Only use if confidence is high enough
            if group_info['confidence'] >= min_confidence:
                group_key = group_info['group_key']
                group_index[group_key].append(i)
            else:
                # Low confidence - treat as singleton
                group_key = f"singleton_{i}"
                group_index[group_key].append(i)
                group_info['group_key'] = group_key
                group_info['is_singleton'] = True
            
            metadata.append(group_info)
        
        # Compute statistics
        group_sizes = [len(indices) for indices in group_index.values()]
        size_dist = Counter(group_sizes)
        
        multi_item_groups = sum(1 for size in group_sizes if size > 1)
        total_groups = len(group_index)
        
        stats = {
            'total_items': len(dataset),
            'total_groups': total_groups,
            'multi_item_groups': multi_item_groups,
            'singleton_groups': total_groups - multi_item_groups,
            'avg_group_size': sum(group_sizes) / len(group_sizes) if group_sizes else 0,
            'max_group_size': max(group_sizes) if group_sizes else 0,
            'size_distribution': dict(size_dist),
        }
        
        return {
            'index': dict(group_index),
            'metadata': metadata,
            'stats': stats,
        }


def create_group_key_deriver() -> GroupKeyDeriver:
    """
    Factory function to create a GroupKeyDeriver.
    
    Returns:
        Configured GroupKeyDeriver instance
    """
    return GroupKeyDeriver()

