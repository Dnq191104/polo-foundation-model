#!/usr/bin/env python
"""Test the updated extractor on audit samples."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.attribute_extractor import AttributeExtractor

def load_spotcheck_csv(csv_path):
    """Load samples from spotcheck CSV file."""
    import csv
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'index': int(row['index']),
                'item_ID': row['item_ID'],
                'category2': row['category2'],
                'text': row['text'],
                'expected_material': row['material_primary'],
                'expected_pattern': row['pattern_primary'],
                'expected_neckline': row['neckline_primary'],
                'expected_sleeve': row['sleeve_primary']
            })
    return samples

def main():
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        samples = load_spotcheck_csv(csv_path)
        print(f"Loaded {len(samples)} samples from {csv_path}")
    else:
        # Use hardcoded samples for testing
        samples = [

    e = AttributeExtractor()

    # Test samples from the user's audit (if not loading from CSV)
    if 'samples' not in locals():
        samples = [
        # Index 0 - sweatshirts (Pattern Missed: "graphic patterns" not detected)
        {'category2': 'sweatshirts', 'text': 'The female wears a long-sleeve shirt with graphic patterns. The shirt is with cotton fabric and its neckline is lapel.'},
        
        # Index 1 - tees (Material Bleed: included "denim" from shorts, should be cotton only)
        {'category2': 'tees', 'text': 'This lady wears a tank tank shirt with pure color patterns. The tank shirt is with cotton fabric. This lady wears a three-point shorts. The shorts are with denim fabric and solid color patterns. This female is wearing a ring on her finger.'},
        
        # Index 2 - tees (PASS: correctly extracted cotton, graphic, short sleeves)
        {'category2': 'tees', 'text': 'The lady is wearing a short-sleeve shirt with graphic patterns. The shirt is with cotton fabric. The neckline of the shirt is round. There is an accessory on her wrist. This woman wears a ring. There is an accessory in his her neck.'},
        
        # Index 3 - shorts (Logic Error: pulled "sleeveless" from shirt - shorts shouldn't have sleeve)
        {'category2': 'shorts', 'text': 'This woman wears a tank tank shirt with solid color patterns and a three-point shorts. The tank shirt is with cotton fabric and its neckline is round. The shorts are with cotton fabric and striped patterns.'},
        
        # Index 4 - blouses (Material Bleed: included "denim" from pants, should be cotton only)
        {'category2': 'blouses', 'text': 'The lady wears a medium-sleeve shirt with graphic patterns. The shirt is with cotton fabric. The pants the lady wears is of three-point length. The pants are with denim fabric and pure color patterns. There is an accessory on her wrist. There is a ring on her finger. There is an accessory in his her neck.'},
        
        # Index 5 - blouses (Attribution Error: "floral" from pants, "cotton" from pants; blouse is chiffon+graphic)
        {'category2': 'blouses', 'text': 'The lady is wearing a sleeveless tank shirt with graphic patterns. The tank shirt is with chiffon fabric. It has a round neckline. The lady wears a three-point pants. The pants are with cotton fabric and floral patterns. There is an accessory on her wrist. This woman has neckwear. The person is wearing a ring on her finger.'},
        
        # Index 6 - tees (PASS)
        {'category2': 'tees', 'text': 'The upper clothing has short sleeves, cotton fabric and complicated patterns. There is an accessory on her wrist.'},
        
        # Index 7 - shirts (PASS: correctly mapped "lapel" to "collared")
        {'category2': 'shirts', 'text': 'The T-shirt this guy wears has short sleeves and it is with cotton fabric and pure color patterns. The neckline of the T-shirt is lapel. This gentleman wears a hat.'},
        
        # Index 8 - shorts (Attribution Error: pulled "crew" and "long_sleeve" from shirt for shorts)
        {'category2': 'shorts', 'text': 'The shirt this lady wears has long sleeves, its fabric is cotton, and it has pure color patterns. The shirt has a crew neckline. This lady wears a three-point pants, with cotton fabric and solid color patterns. The outer clothing this person wears is with cotton fabric and solid color patterns.'},
        
        # Index 9 - blouses (PASS: correctly isolated chiffon, v-neck)
        {'category2': 'blouses', 'text': 'This woman wears a long-sleeve sweater with solid color patterns. The sweater is with chiffon fabric. It has a v-shape neckline. The shorts this woman wears is of three-point length. The shorts are with chiffon fabric and pure color patterns. There is a ring on her finger.'},
    ]

    print("Testing updated extractor on audit samples:")
    print("=" * 80)
    
    expected_results = [
        # Index 0: Expected cotton, graphic, lapel->collared, long_sleeve
        {'material': 'cotton', 'pattern': 'graphic', 'neckline': 'collared', 'sleeve': 'long_sleeve'},
        # Index 1: Expected cotton only (not denim), solid/pure color, sleeveless
        {'material': 'cotton', 'pattern': 'solid', 'neckline': None, 'sleeve': 'sleeveless'},
        # Index 2: Expected cotton, graphic, round, short_sleeve
        {'material': 'cotton', 'pattern': 'graphic', 'neckline': 'round', 'sleeve': 'short_sleeve'},
        # Index 3: Expected cotton, stripe for SHORTS (no neckline/sleeve for shorts)
        {'material': 'cotton', 'pattern': 'stripe', 'neckline': None, 'sleeve': None},
        # Index 4: Expected cotton, graphic for blouse (not denim)
        {'material': 'cotton', 'pattern': 'graphic', 'neckline': None, 'sleeve': None},
        # Index 5: Expected chiffon, graphic, round, sleeveless for blouse (not cotton/floral)
        {'material': 'chiffon', 'pattern': 'graphic', 'neckline': 'round', 'sleeve': 'sleeveless'},
        # Index 6: Expected cotton, unknown pattern, short_sleeve
        {'material': 'cotton', 'pattern': None, 'neckline': None, 'sleeve': 'short_sleeve'},
        # Index 7: Expected cotton, solid, collared, short_sleeve
        {'material': 'cotton', 'pattern': 'solid', 'neckline': 'collared', 'sleeve': 'short_sleeve'},
        # Index 8: Expected cotton, solid for SHORTS (no neckline/sleeve)
        {'material': 'cotton', 'pattern': 'solid', 'neckline': None, 'sleeve': None},
        # Index 9: Expected chiffon, solid, v_neck, long_sleeve for blouse
        {'material': 'chiffon', 'pattern': 'solid', 'neckline': 'v_neck', 'sleeve': 'long_sleeve'},
    ]

    for i, sample in enumerate(samples):
        result = e.extract_with_primary(sample['text'], sample['category2'])
        expected = expected_results[i]
        
        print(f"\nIndex {i} - Category: {sample['category2']}")
        print(f"  Material: {result['attr_material']} (primary: {result['attr_material_primary']})")
        print(f"  Pattern: {result['attr_pattern']} (primary: {result['attr_pattern_primary']})")
        print(f"  Neckline: {result['attr_neckline']} (primary: {result['attr_neckline_primary']})")
        print(f"  Sleeve: {result['attr_sleeve']} (primary: {result['attr_sleeve_primary']})")
        
        # Check if expectations are met
        checks = []
        if expected['material']:
            if expected['material'] in result['attr_material']:
                checks.append(f"[OK] Material '{expected['material']}' found")
            else:
                checks.append(f"[FAIL] Expected material '{expected['material']}' not found")
        if expected['pattern']:
            if expected['pattern'] in result['attr_pattern']:
                checks.append(f"[OK] Pattern '{expected['pattern']}' found")
            else:
                checks.append(f"[FAIL] Expected pattern '{expected['pattern']}' not found")
        
        # Check shorts don't have neckline/sleeve
        if sample['category2'] == 'shorts':
            if not result['attr_neckline']:
                checks.append("[OK] No neckline for shorts (correct)")
            else:
                checks.append(f"[FAIL] Shorts have neckline: {result['attr_neckline']}")
            if not result['attr_sleeve']:
                checks.append("[OK] No sleeve for shorts (correct)")
            else:
                checks.append(f"[FAIL] Shorts have sleeve: {result['attr_sleeve']}")
        
        for check in checks:
            print(f"  {check}")

if __name__ == '__main__':
    main()

