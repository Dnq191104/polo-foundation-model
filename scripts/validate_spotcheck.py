#!/usr/bin/env python
"""Validate attribute extraction accuracy on spotcheck CSV."""

import csv
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.attribute_extractor import AttributeExtractor

# Expected results from the audit (what should be extracted)
EXPECTED_RESULTS = {
    0: {'material': 'cotton', 'pattern': 'graphic', 'neckline': 'collared', 'sleeve': 'long_sleeve'},
    1: {'material': 'cotton', 'pattern': 'solid', 'neckline': None, 'sleeve': 'sleeveless'},
    2: {'material': 'cotton', 'pattern': 'graphic', 'neckline': 'round', 'sleeve': 'short_sleeve'},
    3: {'material': 'cotton', 'pattern': 'stripe', 'neckline': None, 'sleeve': None},  # shorts
    4: {'material': 'cotton', 'pattern': 'graphic', 'neckline': None, 'sleeve': None},
    5: {'material': 'chiffon', 'pattern': 'graphic', 'neckline': 'round', 'sleeve': 'sleeveless'},
    6: {'material': 'cotton', 'pattern': None, 'neckline': None, 'sleeve': 'short_sleeve'},
    7: {'material': 'cotton', 'pattern': 'solid', 'neckline': 'collared', 'sleeve': 'short_sleeve'},
    8: {'material': 'cotton', 'pattern': 'solid', 'neckline': None, 'sleeve': None},  # shorts
    9: {'material': 'chiffon', 'pattern': 'solid', 'neckline': 'v_neck', 'sleeve': 'long_sleeve'},
    10: {'material': 'knit', 'pattern': 'solid', 'neckline': 'crew', 'sleeve': 'three_quarter'},  # knitting fabric
    # Add more as needed from the audit...
}

def get_audit_samples():
    """Get key samples from the audit that were problematic."""
    return [
        # Index 0: sweatshirts with graphic patterns (was missing pattern)
        {'index': 0, 'category2': 'sweatshirts', 'text': 'The female wears a long-sleeve shirt with graphic patterns. The shirt is with cotton fabric and its neckline is lapel.',
         'expected': {'material': 'cotton', 'pattern': 'graphic', 'neckline': 'collared', 'sleeve': 'long_sleeve'}},

        # Index 1: tees (was contaminating with denim from shorts)
        {'index': 1, 'category2': 'tees', 'text': 'This lady wears a tank tank shirt with pure color patterns. The tank shirt is with cotton fabric. This lady wears a three-point shorts. The shorts are with denim fabric and solid color patterns. This female is wearing a ring on her finger.',
         'expected': {'material': 'cotton', 'pattern': 'solid', 'neckline': None, 'sleeve': 'sleeveless'}},

        # Index 3: shorts (was extracting sleeve from shirt)
        {'index': 3, 'category2': 'shorts', 'text': 'This woman wears a tank tank shirt with solid color patterns and a three-point shorts. The tank tank shirt is with cotton fabric and its neckline is round. The shorts are with cotton fabric and striped patterns.',
         'expected': {'material': 'cotton', 'pattern': 'stripe', 'neckline': None, 'sleeve': None}},

        # Index 10: sweaters (knitting fabric not mapped)
        {'index': 10, 'category2': 'sweaters', 'text': 'Her shirt has medium sleeves, knitting fabric and solid color patterns. It has a crew neckline.',
         'expected': {'material': 'knit', 'pattern': 'solid', 'neckline': 'crew', 'sleeve': 'three_quarter'}},

        # Index 13: dresses (dead zone - should extract from tank top)
        {'index': 13, 'category2': 'dresses', 'text': 'The lady is wearing a tank tank top with graphic patterns. The tank top is with cotton fabric. It has a suspenders neckline. The lady wears a long trousers. The trousers are with cotton fabric and graphic patterns.',
         'expected': {'material': 'cotton', 'pattern': 'graphic', 'neckline': 'strapless', 'sleeve': None}},

        # Index 22: skirts (complicated patterns not mapped)
        {'index': 22, 'category2': 'skirts', 'text': 'The person is wearing a short-sleeve T-shirt with complicated patterns. The T-shirt is with cotton fabric.',
         'expected': {'material': 'cotton', 'pattern': 'abstract', 'neckline': None, 'sleeve': 'short_sleeve'}},
    ]

def evaluate_accuracy(samples, extractor):
    """Evaluate accuracy of attribute extraction."""
    results = []

    for sample in samples:
        result = extractor.extract_with_primary(sample['text'], sample['category2'])

        # Check each attribute
        checks = {}
        for attr in ['material', 'pattern', 'neckline', 'sleeve']:
            expected = sample['expected'].get(attr)
            actual = result[f'attr_{attr}_primary']

            # Handle None/unknown cases
            if expected is None and actual == 'unknown':
                correct = True  # Both indicate no attribute
            elif expected is None and actual != 'unknown':
                correct = False  # Should have no attribute but extracted something
            elif expected is not None and actual == 'unknown':
                correct = False  # Should extract something but got unknown
            elif expected is not None and actual != 'unknown':
                correct = expected == actual
            else:
                correct = True  # Both unknown or both None

            checks[attr] = {
                'expected': expected,
                'actual': actual,
                'correct': correct
            }

        sample_result = {
            'index': sample['index'],
            'category2': sample['category2'],
            'checks': checks,
            'all_correct': all(c['correct'] for c in checks.values())
        }
        results.append(sample_result)

    return results

def print_accuracy_report(results):
    """Print detailed accuracy report."""
    print("=" * 80)
    print("SPOTCHECK VALIDATION REPORT")
    print("=" * 80)

    total_samples = len(results)
    correct_samples = sum(1 for r in results if r['all_correct'])

    print(f"Total samples: {total_samples}")
    print(".1f")
    print()

    # Accuracy by attribute
    attributes = ['material', 'pattern', 'neckline', 'sleeve']
    print("ACCURACY BY ATTRIBUTE:")
    print("-" * 50)

    for attr in attributes:
        total_checks = sum(1 for r in results if r['checks'][attr]['expected'] != 'unknown')
        correct_checks = sum(1 for r in results if r['checks'][attr]['correct'])

        if total_checks > 0:
            accuracy = correct_checks / total_checks * 100
            print("10")
        else:
            print("10")

    print()
    print("ERROR ANALYSIS:")
    print("-" * 50)

    # Count errors by type
    errors = {'missed': 0, 'wrong': 0, 'unknown_issue': 0}
    for result in results:
        for attr, check in result['checks'].items():
            if not check['correct']:
                if check['expected'] != 'unknown' and check['actual'] == 'unknown':
                    errors['missed'] += 1
                elif check['expected'] == 'unknown' and check['actual'] != 'unknown':
                    errors['unknown_issue'] += 1
                else:
                    errors['wrong'] += 1

    print(f"Missed tags (should extract but got unknown): {errors['missed']}")
    print(f"Wrong tags (extracted wrong value): {errors['wrong']}")
    print(f"Unknown issues (should be unknown but extracted): {errors['unknown_issue']}")

    print()
    print("SAMPLES WITH ISSUES:")
    print("-" * 50)

    issues_found = 0
    for result in results:
        if not result['all_correct']:
            issues_found += 1
            if issues_found > 20:  # Limit output
                print("... and more")
                break

            wrong_attrs = [attr for attr, check in result['checks'].items() if not check['correct']]
            print("2")

    print()
    if correct_samples / total_samples >= 0.8:
        print("[SUCCESS] Accuracy meets 80% target!")
    else:
        print("[NEEDS IMPROVEMENT] Accuracy below 80% target")

def main():
    print("Loading audit samples...")
    samples = get_audit_samples()
    print(f"Loaded {len(samples)} key audit samples")

    print("Initializing attribute extractor...")
    extractor = AttributeExtractor()

    print("Evaluating accuracy...")
    results = evaluate_accuracy(samples, extractor)

    print_accuracy_report(results)

if __name__ == '__main__':
    main()
