import yaml
import re
from typing import Dict, List, Any
from pathlib import Path


class FashionLabelMapper:
    """Maps DeepFashion categories to unified fashion labels."""
    
    def __init__(self, schema_path: str = None):
        """Initialize mapper with schema."""
        if schema_path is None:
            # Auto-detect schema path
            from pathlib import Path
            current_dir = Path.cwd()
            
            if current_dir.name == 'notebooks':
                # We're in notebooks/, go up to find schema
                schema_path = str(current_dir.parent / "src" / "datasets" / "schema.yaml")
            else:
                # Assume we're in project root
                schema_path = "src/datasets/schema.yaml"
        
        self.schema = self._load_schema(schema_path)
        self.category_mappings = self._build_category_mappings()
        self.neckline_mappings = self._build_neckline_mappings()
        self.sleeve_mappings = self._build_sleeve_mappings()
    
    def _load_schema(self, schema_path: str) -> Dict[str, List[str]]:
        """Load the label schema from YAML file."""
        with open(schema_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _build_category_mappings(self) -> Dict[str, str]:
        """Build mapping from DeepFashion category2 to unified category."""
        return {
            # Tops
            "t-shirts": "top",
            "shirts": "top", 
            "blouses": "top",
            "tops": "top",
            "sweaters": "top",
            "hoodies": "top",
            "cardigans": "top",
            "tank tops": "top",
            "crop tops": "top",
            "bralettes": "top",
            "bodysuits": "top",
            "tunics": "top",
            
            # Outerwear
            "jackets": "outerwear",
            "coats": "outerwear",
            "blazers": "outerwear",
            "vests": "outerwear",
            "ponchos": "outerwear",
            
            # Bottoms
            "pants": "bottom",
            "jeans": "bottom",
            "shorts": "bottom",
            "leggings": "bottom",
            "skirt": "bottom",
            "skirts": "bottom",
            
            # Dresses
            "dresses": "dress",
            "jumpsuits": "dress",
            "rompers": "dress",
            
            # Other
            "suits": "other",
            "swimwear": "other",
            "sleepwear": "other",
            "underwear": "other",
            "socks": "other",
            "accessories": "other"
        }
    
    def _build_neckline_mappings(self) -> Dict[str, str]:
        """Build mapping to neckline types."""
        return {
            "polo": "polo_collar",
            "crew": "crew",
            "v-neck": "v_neck",
            "shirt": "shirt_collar",
            "hoodie": "hood",
            "hood": "hood"
        }
    
    def _build_sleeve_mappings(self) -> Dict[str, str]:
        """Build mapping to sleeve lengths."""
        return {
            "sleeveless": "sleeveless",
            "short": "short",
            "long": "long",
            "three-quarter": "three_quarter"
        }
    
    def map_category2(self, category2: str) -> Dict[str, str]:
        """Map a single DeepFashion category2 to unified labels."""
        category2_lower = category2.lower().strip()
        
        # Initialize with defaults
        labels = {
            "category": "other",
            "neckline": "unknown", 
            "sleeve_length": "unknown",
            "pattern": "unknown"
        }
        
        # Map category
        for key, value in self.category_mappings.items():
            if key in category2_lower or key.replace("-", " ") in category2_lower:
                labels["category"] = value
                break
        
        # Map neckline
        for key, value in self.neckline_mappings.items():
            if key in category2_lower:
                labels["neckline"] = value
                break
        
        # Map sleeve length
        for key, value in self.sleeve_mappings.items():
            if key in category2_lower or key.replace("_", "-") in category2_lower:
                labels["sleeve_length"] = value
                break
        
        return labels
    
    def map_dataset_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Map a single dataset example."""
        mapped_labels = self.map_category2(example["category2"])
        
        # Add the mapped labels to the example
        example["unified_category"] = mapped_labels["category"]
        example["unified_neckline"] = mapped_labels["neckline"]
        example["unified_sleeve_length"] = mapped_labels["sleeve_length"]
        example["unified_pattern"] = mapped_labels["pattern"]
        
        return example


def apply_mapping_to_dataset(dataset, mapper: FashionLabelMapper):
    """Apply the mapper to an entire HuggingFace dataset."""
    def map_function(example):
        return mapper.map_dataset_example(example)
    
    return dataset.map(map_function)


# Convenience function to create mapper and apply to dataset
def map_deepfashion_labels(dataset, schema_path: str = "src/datasets/schema.yaml"):
    """Complete pipeline to map DeepFashion labels."""
    mapper = FashionLabelMapper(schema_path)
    return apply_mapping_to_dataset(dataset, mapper)