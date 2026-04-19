#!/usr/bin/env python3
"""Verify the notebook was updated correctly"""

import json

with open('full-kd.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check Phase 8 Cell 1 (should have Full Model KD code)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '## Phase 8 — Cell 1:' in source:
            code_cell = nb['cells'][i+1]
            code = ''.join(code_cell['source'])
            print('Phase 8 Cell 1 preview (first 500 chars):')
            print(code[:500])
            print('...\n')
            print('Checking for Full KD markers:')
            markers = [
                ('FULL MODEL KNOWLEDGE DISTILLATION', 'FULL MODEL KNOWLEDGE DISTILLATION' in code),
                ('ALL Parameters Trainable', 'ALL Parameters Trainable' in code),
                ('trainable_params', 'trainable_params' in code),
                ('model_p8_student.train()', 'model_p8_student.train()' in code),
            ]
            for marker, found in markers:
                status = '✓' if found else '✗'
                print(f'  {status} Contains "{marker}": {found}')
            break

# Check benchmark cell
print('\n' + '='*60)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'Phase 8 Benchmark Cell 2' in source:
            print('Benchmark Cell 2 - Model name check:')
            has_new = 'phase8_full_kd' in source
            has_old = "'phase8_kd'" in source
            print(f'  {"✓" if has_new else "✗"} Contains phase8_full_kd: {has_new}')
            print(f'  {"✓" if not has_old else "✗"} Old phase8_kd removed: {not has_old}')
            break

# Check main header
print('\n' + '='*60)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if 'Phase 8:' in source and 'Knowledge Distillation' in source:
            print('Phase 8 Main Header:')
            if 'Full Model Knowledge Distillation' in source:
                print('  ✓ Updated to "Full Model Knowledge Distillation"')
            else:
                print('  ✗ Still shows old title')
            break

print('\n' + '='*60)
print('Verification complete!')
