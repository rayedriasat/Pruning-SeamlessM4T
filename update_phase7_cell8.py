#!/usr/bin/env python3
"""
Script to update Phase 7 Cell 8 in cse465v5-s2st-corrected.ipynb
with the complete fixed loss functions.

This replaces the broken compute_t2u_loss function that causes dimension mismatch errors.
"""

import json
import sys

# Read the complete fix code
with open('PHASE7_CELL8_COMPLETE_FIX.py', 'r', encoding='utf-8') as f:
    fixed_code = f.read()

# Load the notebook
notebook_path = 'cse465v5-s2st-corrected.ipynb'
print(f'Loading {notebook_path}...')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Phase 7 Cell 8
# We need to search for cells that contain the loss functions
# Based on inspection, this is cell 105 which has compute_t2u_loss
target_cell_idx = None
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        # Look for the cell that defines compute_t2u_loss
        if 'def compute_t2u_loss' in source:
            target_cell_idx = idx
            print(f'Found Phase 7 Cell 8 (loss functions) at index {idx}')
            break

if target_cell_idx is None:
    print('ERROR: Could not find Phase 7 Cell 8 (cell with compute_t2u_loss)')
    sys.exit(1)

# Replace the cell content
print(f'Replacing cell {target_cell_idx} with fixed code...')

# Convert the fixed code to notebook format (list of lines)
fixed_lines = fixed_code.split('\n')
# Add newline to each line except the last
nb_source = [line + '\n' for line in fixed_lines[:-1]]
if fixed_lines[-1]:  # Add last line without newline if it's not empty
    nb_source.append(fixed_lines[-1])

# Update the cell
nb['cells'][target_cell_idx]['source'] = nb_source

# Clear any existing outputs
nb['cells'][target_cell_idx]['outputs'] = []
nb['cells'][target_cell_idx]['execution_count'] = None

# Save the updated notebook
backup_path = 'cse465v5-s2st-corrected.ipynb.backup'
print(f'Creating backup at {backup_path}...')
with open(backup_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Saving updated notebook to {notebook_path}...')
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('✓ Done!')
print()
print('NEXT STEPS:')
print('1. Reload the notebook in Kaggle/Jupyter')
print('2. Run Phase 7 Cell 8 (the cell we just updated)')
print('3. Run Phase 7 Cell 9 (the training loop)')
print()
print('The dimension mismatch error should now be fixed.')
