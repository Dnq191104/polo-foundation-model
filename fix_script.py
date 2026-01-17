#!/usr/bin/env python3
import re

# Read the file
with open('scripts/step7_run_ablations.py', 'r') as f:
    content = f.read()

# Replace the problematic line
old_line = "positive_indices = batch_df['positive_indices'].tolist()[:8]"
new_line = "positive_indices = batch_df['other_idx'].tolist()[:8]"

content = content.replace(old_line, new_line)

# Write back
with open('scripts/step7_run_ablations.py', 'w') as f:
    f.write(content)

print("Fixed the column name!")






