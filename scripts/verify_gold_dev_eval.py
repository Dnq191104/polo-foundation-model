#!/usr/bin/env python
"""
Gold Dev Eval Verification Tool

Export queries for manual verification and import corrected labels.
Supports CSV export for easy editing and HTML gallery for visual review.
"""

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_gold_eval_ids(eval_ids_path):
    """Load gold eval IDs with metadata."""
    with open(eval_ids_path, 'r') as f:
        data = json.load(f)

    # Handle both old format (list) and new format (dict with metadata)
    if isinstance(data, list):
        return data, {}
    else:
        return data.get('eval_ids', []), data.get('metadata', {})


def export_for_verification(dataset_path, eval_ids, output_dir, split='validation'):
    """
    Export gold eval queries for manual verification.

    Creates:
    - CSV file with metadata for easy editing
    - HTML gallery for visual review
    """

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)[split]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect query data
    queries_data = []

    for query_idx in eval_ids:
        item = dataset[query_idx]

        # Convert image to base64 for HTML
        img = item['image']
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        query_data = {
            'query_idx': query_idx,
            'item_ID': item.get('item_ID', ''),
            'current_category2': item.get('category2', ''),
            'verified_category2': '',  # For manual entry
            'current_material': item.get('attr_material_primary', ''),
            'verified_material': '',  # For manual entry
            'current_pattern': item.get('attr_pattern_primary', ''),
            'verified_pattern': '',  # For manual entry
            'current_neckline': item.get('attr_neckline_primary', ''),
            'verified_neckline': '',  # For manual entry
            'current_sleeve': item.get('attr_sleeve_primary', ''),
            'verified_sleeve': '',  # For manual entry
            'text': item.get('text', ''),
            'img_base64': img_base64,
            'verification_notes': ''  # For reviewer notes
        }

        queries_data.append(query_data)

    # Create CSV for editing
    csv_data = []
    for q in queries_data:
        csv_row = {
            'query_idx': q['query_idx'],
            'item_ID': q['item_ID'],
            'current_category2': q['current_category2'],
            'verified_category2': q['verified_category2'],
            'current_material': q['current_material'],
            'verified_material': q['verified_material'],
            'current_pattern': q['current_pattern'],
            'verified_pattern': q['verified_pattern'],
            'current_neckline': q['current_neckline'],
            'verified_neckline': q['verified_neckline'],
            'current_sleeve': q['current_sleeve'],
            'verified_sleeve': q['verified_sleeve'],
            'text': q['text'],
            'verification_notes': q['verification_notes']
        }
        csv_data.append(csv_row)

    csv_df = pd.DataFrame(csv_data)
    csv_path = output_dir / "gold_eval_verification.csv"
    csv_df.to_csv(csv_path, index=False)

    # Create HTML gallery
    html_content = create_verification_html(queries_data)
    html_path = output_dir / "gold_eval_verification.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Exported {len(queries_data)} queries for verification:")
    print(f"  CSV: {csv_path}")
    print(f"  HTML: {html_path}")

    # Create verification instructions
    instructions_path = output_dir / "VERIFICATION_INSTRUCTIONS.txt"
    with open(instructions_path, 'w') as f:
        f.write("""GOLD DEV EVAL VERIFICATION INSTRUCTIONS

1. Review each query in the HTML gallery or CSV file
2. Check if the current labels (category, attributes) are correct
3. If incorrect, enter the correct labels in the 'verified_*' columns
4. Add notes in 'verification_notes' if needed
5. Save the edited CSV file
6. Run this script with --import to update the gold eval set

VERIFICATION FOCUS AREAS:
- Category accuracy (especially weak categories: shorts, rompers, cardigans)
- Material correctness
- Pattern detection
- Neckline/sleeve attributes for upper body garments

WEAK CATEGORIES TO PAY EXTRA ATTENTION TO:
- shorts
- rompers
- cardigans

These categories are boosted in the eval set to ensure good coverage.

IMPORT COMMAND:
python scripts/verify_gold_dev_eval.py --import_csv gold_eval_corrected.csv --gold_eval gold_dev_eval_ids.json --output gold_dev_eval_verified.json
""")

    print(f"  Instructions: {instructions_path}")

    return csv_path, html_path


def create_verification_html(queries_data):
    """Create HTML gallery for verification."""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Gold Dev Eval Verification</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .query {{ border: 1px solid #ccc; margin: 20px 0; padding: 15px; border-radius: 8px; }}
        .query-header {{ font-weight: bold; color: #333; margin-bottom: 10px; }}
        .query-image {{ max-width: 200px; max-height: 200px; border: 1px solid #ddd; margin: 10px 0; }}
        .current-labels {{ background: #f0f0f0; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .verification-fields {{ background: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .attribute-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
        .attribute-item {{ background: #f8f9fa; padding: 8px; border-radius: 4px; }}
        .weak-category {{ border-left: 4px solid #ff6b6b; }}
        .text-preview {{ font-style: italic; color: #666; max-width: 400px; word-wrap: break-word; }}
        .notes {{ width: 100%; height: 60px; border: 1px solid #ccc; border-radius: 4px; padding: 5px; }}
    </style>
</head>
<body>
    <h1>Gold Dev Eval Verification Gallery</h1>
    <p><strong>Total queries:</strong> {len(queries_data)}</p>
    <p><strong>Instructions:</strong> Review each query below. Check if labels are correct. Note any issues for correction.</p>
    <hr>
"""

    for i, query in enumerate(queries_data):
        is_weak = query['current_category2'] in ['shorts', 'rompers', 'cardigans']
        weak_class = 'weak-category' if is_weak else ''

        html += f"""
    <div class="query {weak_class}">
        <div class="query-header">
            Query #{i+1} - Item ID: {query['item_ID']} - Index: {query['query_idx']}
            {' [WEAK CATEGORY]' if is_weak else ''}
        </div>

        <img src="data:image/jpeg;base64,{query['img_base64']}" class="query-image" alt="Query image">

        <div class="current-labels">
            <strong>Current Labels:</strong><br>
            Category: {query['current_category2']}<br>
            Material: {query['current_material']}<br>
            Pattern: {query['current_pattern']}<br>
            Neckline: {query['current_neckline']}<br>
            Sleeve: {query['current_sleeve']}
        </div>

        <div class="text-preview">
            <strong>Text:</strong> {query['text'][:200]}{'...' if len(query['text']) > 200 else ''}
        </div>

        <div class="verification-fields">
            <strong>Verification Notes:</strong><br>
            <textarea class="notes" placeholder="Enter verification notes here...">{query['verification_notes']}</textarea>
        </div>
    </div>
"""

    html += """
</body>
</html>
"""

    return html


def import_verified_labels(csv_path, gold_eval_path, overrides_output_path):
    """
    Import verified labels from corrected CSV and create overrides file.
    Keeps the original gold eval IDs immutable.
    """

    print(f"Loading verified labels from {csv_path}...")
    verified_df = pd.read_csv(csv_path)

    print(f"Loading gold eval from {gold_eval_path}...")
    with open(gold_eval_path, 'r') as f:
        gold_data = json.load(f)

    eval_ids = gold_data.get('eval_ids', gold_data if isinstance(gold_data, list) else [])

    # Collect overrides mapping
    overrides = {}
    corrections = []

    print(f"Processing {len(verified_df)} verified entries...")

    for _, row in verified_df.iterrows():
        query_idx = int(row['query_idx'])

        # Skip if this ID is not in our gold eval set
        if query_idx not in eval_ids:
            print(f"⚠️  Warning: query_idx {query_idx} not found in gold eval set, skipping")
            continue

        # Check for corrections
        override_corrections = {}
        correction_record = {
            'query_idx': query_idx,
            'original': {},
            'verified': {},
            'notes': row.get('verification_notes', '')
        }

        # Category correction
        orig_cat = row.get('current_category2', '')
        verified_cat = row.get('verified_category2', '')
        if pd.notna(verified_cat) and verified_cat.strip() and verified_cat != orig_cat:
            override_corrections['category2'] = verified_cat.strip()
            correction_record['original']['category2'] = orig_cat
            correction_record['verified']['category2'] = verified_cat.strip()

        # Attribute corrections
        attrs = ['material', 'pattern', 'neckline', 'sleeve']
        for attr in attrs:
            orig_val = row.get(f'current_{attr}', '')
            verified_val = row.get(f'verified_{attr}', '')
            if pd.notna(verified_val) and verified_val.strip() and verified_val != orig_val:
                override_corrections[f'attr_{attr}_primary'] = verified_val.strip()
                correction_record['original'][f'attr_{attr}_primary'] = orig_val
                correction_record['verified'][f'attr_{attr}_primary'] = verified_val.strip()

        # Only store if there are actual corrections
        if override_corrections:
            overrides[query_idx] = override_corrections
            corrections.append(correction_record)

    print(f"Found {len(corrections)} label corrections for {len(overrides)} items")

    # Create overrides structure
    overrides_data = {
        'overrides': overrides,
        'metadata': {
            'created_at': str(pd.Timestamp.now()),
            'source_csv': csv_path,
            'source_gold_eval': gold_eval_path,
            'n_corrections': len(corrections),
            'n_overridden_items': len(overrides),
            'corrections': corrections
        }
    }

    # Save overrides file
    overrides_output_path = Path(overrides_output_path)
    overrides_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(overrides_output_path, 'w') as f:
        json.dump(overrides_data, f, indent=2)

    print(f"Overrides saved to: {overrides_output_path}")
    print(f"Corrections applied: {len(corrections)}")
    print(f"Items with overrides: {len(overrides)}")
    print(f"Ready for evaluation with eval_gold_dev.py --overrides {overrides_output_path}")

    return overrides_output_path


def main():
    parser = argparse.ArgumentParser(description="Gold dev eval verification tool")

    # Export mode
    parser.add_argument("--export", action="store_true", help="Export queries for verification")
    parser.add_argument("--gold_eval", type=str, help="Path to gold_dev_eval_ids.json")
    parser.add_argument("--dataset", type=str, help="Dataset path (for export)")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="artifacts/step7/verification", help="Output directory for export")

    # Import mode
    parser.add_argument("--import_csv", type=str, help="Import corrected CSV file")
    parser.add_argument("--overrides_output", type=str, default="artifacts/step7/gold_dev_eval_overrides.json", help="Output path for overrides JSON")

    args = parser.parse_args()

    if args.export:
        if not args.gold_eval:
            print("Error: --gold_eval required for export")
            sys.exit(1)
        if not args.dataset:
            print("Error: --dataset required for export")
            sys.exit(1)

        eval_ids, _ = load_gold_eval_ids(args.gold_eval)
        export_for_verification(args.dataset, eval_ids, args.output_dir, args.split)

    elif args.import_csv:
        if not args.gold_eval:
            print("Error: --gold_eval required for import")
            sys.exit(1)

        import_verified_labels(args.import_csv, args.gold_eval, args.overrides_output)

    else:
        print("Error: Must specify --export or --import")
        sys.exit(1)


if __name__ == "__main__":
    main()
