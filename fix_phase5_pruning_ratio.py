#!/usr/bin/env python3
"""
Fix Phase 5 Cell 6 - Reduce FLAP pruning ratio
15% is too aggressive after Phases 3-4. Use 8% instead.
"""

import json
import re

with open('cse465v5-s2st-corrected.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Phase 5 Cell 6 (RUN PHASE 5)
cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        if 'Phase 5 Cell 6' in source and 'RUN PHASE 5' in source and 'FLAP_RATIO' in source:
            cell_idx = i
            break

if cell_idx is None:
    print("ERROR: Could not find Phase 5 Cell 6")
    exit(1)

print(f"Found Phase 5 Cell 6 at cell index {cell_idx}")

# Get current source
source = nb['cells'][cell_idx]['source']
if isinstance(source, list):
    source = ''.join(source)

# Replace FLAP_RATIO and MIN_KEEP_FRAC
source = re.sub(
    r'FLAP_RATIO\s*=\s*0\.15',
    'FLAP_RATIO    = 0.08',
    source
)
source = re.sub(
    r'MIN_KEEP_FRAC\s*=\s*0\.70',
    'MIN_KEEP_FRAC = 0.80',
    source
)

# Add a comment explaining the change
source = re.sub(
    r'(FLAP_RATIO\s*=\s*0\.08.*?\n)',
    r'\1# REDUCED from 0.15 → 0.08 to prevent decoder collapse after Phases 3-4\n',
    source
)

nb['cells'][cell_idx]['source'] = source

with open('cse465v5-s2st-corrected.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Phase 5 Cell 6 updated with conservative pruning ratios")
print("\nNEW SETTINGS:")
print("  FLAP_RATIO    = 0.08  (was 0.15) - prune only 8% of neurons")
print("  MIN_KEEP_FRAC = 0.80  (was 0.70) - keep at least 80% per layer")
print("\nEXPECTED RESULTS:")
print("  - Params saved: ~50M (instead of ~103M)")
print("  - ChrF drop: 2-5 points (instead of 40+ points)")
print("  - No repeated characters or loops")
print("  - Model remains functional for Phase 7 fine-tuning")
