#!/usr/bin/env python
"""
Compute Step 7 Scoreboard

Compares checkpoint metrics against Step 6 baseline and computes deltas.

Usage:
    python scripts/step7_compute_scoreboard.py \
        --baseline artifacts/retrieval/openclip_vitb32_v0/eval/metrics.json \
        --checkpoint artifacts/step7/runs/run_001/eval/metrics.json \
        --output artifacts/step7/runs/run_001/eval/scoreboard
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.scoreboard import Step7Scoreboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Step 7 scoreboard vs baseline"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline metrics JSON (Step 6)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint metrics JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for scoreboard files"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k value for metrics (default: 10)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("STEP 7 SCOREBOARD COMPUTATION")
    print("=" * 60)
    print(f"Baseline: {args.baseline}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()
    
    # Create scoreboard
    scoreboard = Step7Scoreboard(args.baseline)
    
    # Compute and save
    text_path = scoreboard.save_scoreboard(
        args.checkpoint,
        args.output,
        top_k=args.top_k
    )
    
    # Print scoreboard
    with open(text_path, 'r') as f:
        print(f.read())
    
    print(f"\nScoreboard saved to: {args.output}")
    print(f"  - {Path(args.output) / 'scoreboard.txt'}")
    print(f"  - {Path(args.output) / 'scoreboard_deltas.json'}")


if __name__ == "__main__":
    main()

