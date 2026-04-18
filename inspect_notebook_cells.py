#!/usr/bin/env python3
"""Inspect cells around index 105 to understand the structure."""

import json

with open('cse465v5-s2st-corrected.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check cells around index 105
for idx in range(103, 110):
    if idx < len(nb['cells']):
        cell = nb['cells'][idx]
        source = ''.join(cell.get('source', []))
        
        print(f'\n{"="*80}')
        print(f'CELL {idx} ({cell["cell_type"]})')
        print(f'{"="*80}')
        
        if cell['cell_type'] == 'markdown':
            print(source[:500])
        else:
            # Show first 1000 chars for code cells
            preview = source[:1000]
            if len(source) > 1000:
                preview += f'\n... [truncated, total length: {len(source)} chars]'
            print(preview)
