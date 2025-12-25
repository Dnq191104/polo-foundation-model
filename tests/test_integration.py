"""
Integration Tests for Attribute Extraction System

Tests end-to-end functionality with sample data.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.attribute_extractor import AttributeExtractor
from src.datasets.attribute_validator import AttributeValidator
from src.datasets.pair_generator import AttributePairGenerator
from src.utils.eval_diagnostics import RetrievalDiagnostics


# Sample dataset for testing
SAMPLE_DATA = [
    {
        'text': 'Cotton crew neck t-shirt with graphic print.',
        'category2': 'blouses',
        'item_ID': 'item_001',
    },
    {
        'text': 'Denim jacket with button closure and collar.',
        'category2': 'jackets',
        'item_ID': 'item_002',
    },
    {
        'text': 'Silk v-neck blouse in solid color.',
        'category2': 'blouses',
        'item_ID': 'item_003',
    },
    {
        'text': 'Cotton short sleeve polo with striped pattern.',
        'category2': 'blouses',
        'item_ID': 'item_004',
    },
    {
        'text': 'Wool long sleeve sweater with crew neck.',
        'category2': 'sweaters',
        'item_ID': 'item_005',
    },
    {
        'text': 'Polyester sleeveless tank top.',
        'category2': 'blouses',
        'item_ID': 'item_006',
    },
    {
        'text': 'Cotton t-shirt with color block design.',
        'category2': 'blouses',
        'item_ID': 'item_007',
    },
    {
        'text': 'Leather jacket with zipper closure.',
        'category2': 'jackets',
        'item_ID': 'item_008',
    },
    {
        'text': 'Linen shirt with button-down collar.',
        'category2': 'blouses',
        'item_ID': 'item_009',
    },
    {
        'text': 'Knit cardigan with long sleeves.',
        'category2': 'sweaters',
        'item_ID': 'item_010',
    },
]


class MockDataset:
    """Mock dataset for testing without HuggingFace dependency."""
    
    def __init__(self, data: list):
        self.data = data
        self.column_names = list(data[0].keys()) if data else []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Support column access like HuggingFace Dataset
        if isinstance(idx, str):
            return [item.get(idx) for item in self.data]
        if isinstance(idx, slice):
            return MockDataset(self.data[idx])
        return self.data[idx]
    
    def __iter__(self):
        return iter(self.data)
    
    def select(self, indices):
        return MockDataset([self.data[i] for i in indices])
    
    def filter(self, func):
        return MockDataset([item for item in self.data if func(item)])
    
    def map(self, func, **kwargs):
        return MockDataset([func(item) for item in self.data])


class TestIntegrationExtraction:
    """Integration tests for attribute extraction."""
    
    @pytest.fixture
    def extractor(self):
        return AttributeExtractor()
    
    @pytest.fixture
    def sample_dataset(self):
        return MockDataset(SAMPLE_DATA.copy())
    
    def test_extract_all_samples(self, extractor, sample_dataset):
        """Test extraction on all sample items."""
        for item in sample_dataset:
            result = extractor.extract(item['text'])
            assert isinstance(result, dict)
            assert 'material' in result
            assert 'pattern' in result
    
    def test_extraction_coverage(self, extractor, sample_dataset):
        """Test that extraction achieves reasonable coverage."""
        material_count = 0
        pattern_count = 0
        
        for item in sample_dataset:
            result = extractor.extract(item['text'])
            if result['material']:
                material_count += 1
            if result['pattern']:
                pattern_count += 1
        
        # Expect high material coverage on clean sample data
        assert material_count / len(sample_dataset) >= 0.8, "Material coverage too low"
        # Pattern coverage may be lower since not all items have explicit patterns
        assert pattern_count / len(sample_dataset) >= 0.3, "Pattern coverage too low"
    
    def test_process_dataset(self, extractor, sample_dataset):
        """Test processing entire dataset."""
        processed = sample_dataset.map(extractor.process_example)
        
        # Check first item has attributes
        first = processed[0]
        assert 'attr_material' in first
        assert 'attr_material_primary' in first


class TestIntegrationValidator:
    """Integration tests for attribute validation."""
    
    @pytest.fixture
    def validator(self):
        return AttributeValidator()
    
    @pytest.fixture
    def processed_dataset(self):
        extractor = AttributeExtractor()
        data = [extractor.process_example(item.copy()) for item in SAMPLE_DATA]
        return MockDataset(data)
    
    def test_compute_coverage(self, validator, processed_dataset):
        """Test coverage computation."""
        coverage = validator.compute_coverage(processed_dataset)
        
        assert 'material' in coverage
        assert 'pattern' in coverage
        assert coverage['material']['coverage_pct'] > 0
    
    def test_tag_distribution(self, validator, processed_dataset):
        """Test tag distribution computation."""
        dist = validator.compute_tag_distribution(processed_dataset, 'material')
        
        assert isinstance(dist, dict)
        assert 'cotton' in dist or 'unknown' in dist
    
    def test_sample_for_spotcheck(self, validator, processed_dataset):
        """Test sampling for spot-check."""
        sample = validator.sample_for_spotcheck(processed_dataset, n=5)
        
        assert len(sample) == 5
    
    def test_format_spotcheck_item(self, validator, processed_dataset):
        """Test formatting item for spot-check."""
        item = processed_dataset[0]
        formatted = validator.format_spotcheck_item(item)
        
        assert isinstance(formatted, str)
        assert 'TEXT:' in formatted
        assert 'EXTRACTED ATTRIBUTES:' in formatted


class TestIntegrationPairGenerator:
    """Integration tests for pair generation."""
    
    @pytest.fixture
    def generator(self):
        return AttributePairGenerator(seed=42)
    
    @pytest.fixture
    def processed_dataset(self):
        extractor = AttributeExtractor()
        data = [extractor.process_example(item.copy()) for item in SAMPLE_DATA]
        return MockDataset(data)
    
    def test_generate_strong_positives(self, generator, processed_dataset):
        """Test strong positive pair generation."""
        pairs = generator.generate_strong_positives(processed_dataset)
        
        # May not have many strong positives with small dataset
        assert isinstance(pairs, list)
        for p in pairs:
            assert len(p) == 2
            assert isinstance(p[0], int)
            assert isinstance(p[1], int)
    
    def test_generate_hard_negatives(self, generator, processed_dataset):
        """Test hard negative pair generation."""
        pairs = generator.generate_hard_negatives(processed_dataset)
        
        assert isinstance(pairs, list)
    
    def test_generate_easy_negatives(self, generator, processed_dataset):
        """Test easy negative pair generation."""
        pairs = generator.generate_easy_negatives(processed_dataset)
        
        assert isinstance(pairs, list)
        # Should have pairs since we have multiple categories
        assert len(pairs) > 0
    
    def test_get_pair_stats(self, generator, processed_dataset):
        """Test pair statistics generation."""
        stats = generator.get_pair_stats(processed_dataset)
        
        assert 'total_samples' in stats
        assert stats['total_samples'] == len(processed_dataset)
        assert 'categories' in stats


class TestIntegrationDiagnostics:
    """Integration tests for evaluation diagnostics."""
    
    @pytest.fixture
    def diagnostics(self):
        return RetrievalDiagnostics()
    
    @pytest.fixture
    def processed_data(self):
        extractor = AttributeExtractor()
        return [extractor.process_example(item.copy()) for item in SAMPLE_DATA]
    
    def test_compute_material_match_rate(self, diagnostics, processed_data):
        """Test material match rate computation."""
        query = processed_data[0]  # Cotton item
        retrieved = processed_data[1:5]  # Mix of materials
        
        rate = diagnostics.compute_material_match_rate(query, retrieved)
        
        assert 0.0 <= rate <= 1.0
    
    def test_compute_category_match_rate(self, diagnostics, processed_data):
        """Test category match rate computation."""
        query = processed_data[0]  # Blouses
        retrieved = processed_data[1:5]
        
        rate = diagnostics.compute_category_match_rate(query, retrieved)
        
        assert 0.0 <= rate <= 1.0
    
    def test_evaluate_retrieval(self, diagnostics, processed_data):
        """Test full retrieval evaluation."""
        query = processed_data[0]
        retrieved = processed_data[1:]
        
        metrics = diagnostics.evaluate_retrieval(query, retrieved, k=5)
        
        assert 'material_match@5' in metrics
        assert 'category_match@5' in metrics
    
    def test_slice_by_category(self, diagnostics, processed_data):
        """Test slicing results by category."""
        queries = processed_data[:3]
        results = [[processed_data[i+3] for i in range(3)] for _ in range(3)]
        
        sliced = diagnostics.slice_by_category(queries, results, k=3)
        
        assert isinstance(sliced, dict)


class TestDeterminism:
    """Test that extraction is deterministic."""
    
    def test_extraction_deterministic(self):
        """Test that same input gives same output."""
        extractor = AttributeExtractor()
        text = "Cotton crew neck t-shirt with striped pattern."
        
        result1 = extractor.extract(text)
        result2 = extractor.extract(text)
        
        assert result1 == result2
    
    def test_pair_generation_deterministic(self):
        """Test that pair generation is deterministic with same seed."""
        extractor = AttributeExtractor()
        data = [extractor.process_example(item.copy()) for item in SAMPLE_DATA]
        dataset = MockDataset(data)
        
        gen1 = AttributePairGenerator(seed=42)
        gen2 = AttributePairGenerator(seed=42)
        
        pairs1 = gen1.generate_strong_positives(dataset, n_pairs=10)
        pairs2 = gen2.generate_strong_positives(dataset, n_pairs=10)
        
        assert pairs1 == pairs2


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])

