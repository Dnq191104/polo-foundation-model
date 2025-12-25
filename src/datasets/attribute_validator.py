"""
Fashion Attribute Validation Module

Provides tools for validating attribute extraction coverage, quality,
and generating reports for human spot-checking.
"""

import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from datasets import Dataset


class AttributeValidator:
    """
    Validates attribute extraction quality and coverage.
    
    Provides methods for:
    - Computing coverage statistics
    - Generating coverage reports by category
    - Sampling for human spot-check
    - Exporting data for manual review
    """
    
    # Standard attribute names
    ATTRIBUTE_NAMES = ['material', 'pattern', 'neckline', 'sleeve']
    
    def __init__(self, unknown_value: str = "unknown"):
        """
        Initialize validator.
        
        Args:
            unknown_value: Value that indicates no tag was found
        """
        self.unknown_value = unknown_value
    
    def compute_coverage(
        self, 
        dataset: Dataset,
        attr_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute coverage statistics for each attribute type.
        
        Args:
            dataset: Dataset with extracted attributes
            attr_names: Attribute names to check (default: standard set)
            
        Returns:
            Dict mapping attribute name to coverage stats
        """
        if attr_names is None:
            attr_names = self.ATTRIBUTE_NAMES
        
        results = {}
        total = len(dataset)
        
        for attr_name in attr_names:
            primary_col = f'attr_{attr_name}_primary'
            list_col = f'attr_{attr_name}'
            
            stats = {
                'total': total,
                'tagged': 0,
                'unknown': 0,
                'coverage_pct': 0.0,
                'multi_tag_count': 0,
                'multi_tag_pct': 0.0,
            }
            
            if primary_col in dataset.column_names:
                values = dataset[primary_col]
                tagged = sum(1 for v in values if v != self.unknown_value)
                stats['tagged'] = tagged
                stats['unknown'] = total - tagged
                stats['coverage_pct'] = (tagged / total * 100) if total > 0 else 0.0
            
            if list_col in dataset.column_names:
                lists = dataset[list_col]
                multi = sum(1 for v in lists if isinstance(v, list) and len(v) > 1)
                stats['multi_tag_count'] = multi
                stats['multi_tag_pct'] = (multi / total * 100) if total > 0 else 0.0
            
            results[attr_name] = stats
        
        return results
    
    def compute_tag_distribution(
        self,
        dataset: Dataset,
        attr_name: str,
        use_primary: bool = True
    ) -> Dict[str, int]:
        """
        Compute distribution of tags for an attribute type.
        
        Args:
            dataset: Dataset with extracted attributes
            attr_name: Attribute type (e.g., 'material')
            use_primary: Use primary tag column vs list column
            
        Returns:
            Dict mapping tag name to count
        """
        if use_primary:
            col = f'attr_{attr_name}_primary'
            if col not in dataset.column_names:
                return {}
            values = dataset[col]
            return dict(Counter(values))
        else:
            col = f'attr_{attr_name}'
            if col not in dataset.column_names:
                return {}
            counter = Counter()
            for tag_list in dataset[col]:
                if isinstance(tag_list, list):
                    counter.update(tag_list)
            return dict(counter)
    
    def compute_coverage_by_category(
        self,
        dataset: Dataset,
        category_column: str = 'category2',
        attr_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute coverage statistics broken down by category.
        
        Args:
            dataset: Dataset with extracted attributes
            category_column: Column to group by
            attr_names: Attribute names to check
            
        Returns:
            Dict mapping category to attribute coverage stats
        """
        if attr_names is None:
            attr_names = self.ATTRIBUTE_NAMES
        
        if category_column not in dataset.column_names:
            return {}
        
        # Group by category
        categories = set(dataset[category_column])
        results = {}
        
        for category in categories:
            # Filter to this category
            cat_ds = dataset.filter(lambda x: x[category_column] == category)
            
            # Compute coverage for this subset
            coverage = self.compute_coverage(cat_ds, attr_names)
            results[category] = coverage
        
        return results
    
    def generate_coverage_report(
        self,
        dataset: Dataset,
        attr_names: Optional[List[str]] = None,
        by_category: bool = False,
        category_column: str = 'category2'
    ) -> str:
        """
        Generate a human-readable coverage report.
        
        Args:
            dataset: Dataset with extracted attributes
            attr_names: Attribute names to check
            by_category: Whether to break down by category
            category_column: Column to group by
            
        Returns:
            Formatted report string
        """
        if attr_names is None:
            attr_names = self.ATTRIBUTE_NAMES
        
        lines = []
        lines.append("=" * 60)
        lines.append("ATTRIBUTE EXTRACTION COVERAGE REPORT")
        lines.append("=" * 60)
        lines.append(f"Total samples: {len(dataset)}")
        lines.append("")
        
        # Overall coverage
        coverage = self.compute_coverage(dataset, attr_names)
        
        lines.append("OVERALL COVERAGE:")
        lines.append("-" * 40)
        lines.append(f"{'Attribute':<15} {'Tagged':<10} {'Coverage %':<12} {'Multi-tag %':<12}")
        lines.append("-" * 40)
        
        for attr_name in attr_names:
            stats = coverage.get(attr_name, {})
            tagged = stats.get('tagged', 0)
            cov_pct = stats.get('coverage_pct', 0)
            multi_pct = stats.get('multi_tag_pct', 0)
            lines.append(f"{attr_name:<15} {tagged:<10} {cov_pct:<12.1f} {multi_pct:<12.1f}")
        
        lines.append("")
        
        # Tag distribution for each attribute
        lines.append("TAG DISTRIBUTION:")
        lines.append("-" * 40)
        
        for attr_name in attr_names:
            dist = self.compute_tag_distribution(dataset, attr_name)
            sorted_dist = sorted(dist.items(), key=lambda x: -x[1])[:10]  # Top 10
            
            lines.append(f"\n{attr_name.upper()}:")
            for tag, count in sorted_dist:
                pct = count / len(dataset) * 100
                lines.append(f"  {tag:<20} {count:>6} ({pct:>5.1f}%)")
        
        # By category breakdown
        if by_category:
            lines.append("")
            lines.append("=" * 60)
            lines.append("COVERAGE BY CATEGORY:")
            lines.append("=" * 60)
            
            cat_coverage = self.compute_coverage_by_category(
                dataset, category_column, attr_names
            )
            
            # Sort categories by count
            cat_counts = Counter(dataset[category_column])
            sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
            
            for category, count in sorted_cats[:15]:  # Top 15 categories
                lines.append(f"\n{category} (n={count}):")
                cat_stats = cat_coverage.get(category, {})
                
                for attr_name in attr_names:
                    stats = cat_stats.get(attr_name, {})
                    cov_pct = stats.get('coverage_pct', 0)
                    lines.append(f"  {attr_name}: {cov_pct:.1f}%")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def sample_for_spotcheck(
        self,
        dataset: Dataset,
        n: int = 100,
        seed: int = 42,
        stratify_by: Optional[str] = None
    ) -> Dataset:
        """
        Sample items for human spot-check.
        
        Args:
            dataset: Dataset with extracted attributes
            n: Number of samples
            seed: Random seed
            stratify_by: Column to stratify by (optional)
            
        Returns:
            Sampled subset of dataset
        """
        random.seed(seed)
        
        if stratify_by and stratify_by in dataset.column_names:
            # Stratified sampling
            categories = list(set(dataset[stratify_by]))
            samples_per_cat = max(1, n // len(categories))
            
            indices = []
            for category in categories:
                cat_indices = [
                    i for i, c in enumerate(dataset[stratify_by])
                    if c == category
                ]
                sampled = random.sample(cat_indices, min(samples_per_cat, len(cat_indices)))
                indices.extend(sampled)
            
            # Shuffle and trim to n
            random.shuffle(indices)
            indices = indices[:n]
        else:
            # Simple random sampling
            all_indices = list(range(len(dataset)))
            indices = random.sample(all_indices, min(n, len(dataset)))
        
        return dataset.select(indices)
    
    def format_spotcheck_item(
        self,
        item: Dict[str, Any],
        attr_names: Optional[List[str]] = None
    ) -> str:
        """
        Format a single item for human review.
        
        Args:
            item: Dataset item
            attr_names: Attribute names to display
            
        Returns:
            Formatted string for display
        """
        if attr_names is None:
            attr_names = self.ATTRIBUTE_NAMES
        
        lines = []
        lines.append("-" * 60)
        
        # Basic info
        if 'item_ID' in item:
            lines.append(f"Item ID: {item['item_ID']}")
        if 'category2' in item:
            lines.append(f"Category: {item['category2']}")
        
        # Text
        lines.append("")
        lines.append("TEXT:")
        text = item.get('text', '')
        # Wrap text at 80 chars
        import textwrap
        wrapped = textwrap.fill(text, width=80)
        lines.append(wrapped)
        
        # Extracted attributes
        lines.append("")
        lines.append("EXTRACTED ATTRIBUTES:")
        for attr_name in attr_names:
            list_col = f'attr_{attr_name}'
            primary_col = f'attr_{attr_name}_primary'
            
            tags = item.get(list_col, [])
            primary = item.get(primary_col, self.unknown_value)
            
            if isinstance(tags, list):
                tags_str = ", ".join(tags) if tags else "(none)"
            else:
                tags_str = str(tags) if tags else "(none)"
            
            lines.append(f"  {attr_name}: {tags_str} [primary: {primary}]")
        
        lines.append("")
        lines.append("Review: [ ] Correct  [ ] Partial  [ ] Wrong")
        lines.append("Notes: _________________________________")
        
        return "\n".join(lines)
    
    def export_spotcheck_csv(
        self,
        dataset: Dataset,
        output_path: str,
        n: int = 100,
        seed: int = 42,
        attr_names: Optional[List[str]] = None
    ) -> str:
        """
        Export a sample for spot-checking to CSV.
        
        Args:
            dataset: Dataset with extracted attributes
            output_path: Path to save CSV
            n: Number of samples
            seed: Random seed
            attr_names: Attribute names to include
            
        Returns:
            Path to saved file
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for CSV export. Install with: pip install pandas")
        
        if attr_names is None:
            attr_names = self.ATTRIBUTE_NAMES
        
        sample = self.sample_for_spotcheck(dataset, n, seed)
        
        # Build rows
        rows = []
        for i in range(len(sample)):
            item = sample[i]
            row = {
                'index': i,
                'item_ID': item.get('item_ID', ''),
                'category2': item.get('category2', ''),
                'text': item.get('text', ''),
            }
            
            # Add attribute columns
            for attr_name in attr_names:
                list_col = f'attr_{attr_name}'
                primary_col = f'attr_{attr_name}_primary'
                
                tags = item.get(list_col, [])
                primary = item.get(primary_col, self.unknown_value)
                
                if isinstance(tags, list):
                    row[f'{attr_name}_tags'] = "; ".join(tags)
                else:
                    row[f'{attr_name}_tags'] = str(tags)
                row[f'{attr_name}_primary'] = primary
            
            # Review columns
            row['review_status'] = ''  # correct / partial / wrong
            row['error_type'] = ''     # missed_tag / wrong_tag / conflict_rule
            row['notes'] = ''
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        print(f"Exported {len(rows)} samples to: {output_path}")
        return output_path
    
    def analyze_spotcheck_results(
        self,
        csv_path: str,
        attr_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze completed spot-check results.
        
        Args:
            csv_path: Path to completed spot-check CSV
            attr_names: Attribute names to analyze
            
        Returns:
            Analysis results dict
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for analysis. Install with: pip install pandas")
        
        if attr_names is None:
            attr_names = self.ATTRIBUTE_NAMES
        
        df = pd.read_csv(csv_path)
        
        results = {
            'total_reviewed': len(df),
            'status_counts': {},
            'error_types': {},
            'by_attribute': {},
        }
        
        # Status counts
        if 'review_status' in df.columns:
            status_counts = df['review_status'].value_counts().to_dict()
            results['status_counts'] = status_counts
            
            # Calculate usable rate
            correct = status_counts.get('correct', 0)
            partial = status_counts.get('partial', 0)
            total = len(df)
            results['usable_rate'] = (correct + partial) / total * 100 if total > 0 else 0
        
        # Error types
        if 'error_type' in df.columns:
            error_counts = df['error_type'].value_counts().to_dict()
            results['error_types'] = error_counts
        
        # By attribute analysis (if individual columns exist)
        for attr_name in attr_names:
            primary_col = f'{attr_name}_primary'
            if primary_col in df.columns:
                # Count unknowns
                unknown_count = (df[primary_col] == self.unknown_value).sum()
                results['by_attribute'][attr_name] = {
                    'unknown_count': int(unknown_count),
                    'unknown_pct': unknown_count / len(df) * 100 if len(df) > 0 else 0,
                }
        
        return results


def create_validator(unknown_value: str = "unknown") -> AttributeValidator:
    """
    Factory function to create an AttributeValidator.
    
    Args:
        unknown_value: Value indicating no tag found
        
    Returns:
        Configured AttributeValidator instance
    """
    return AttributeValidator(unknown_value)

