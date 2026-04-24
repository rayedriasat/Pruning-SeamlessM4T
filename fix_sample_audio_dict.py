#!/usr/bin/env python3
"""
Fix missing sample_id_to_audio dictionary in Phase 6a training.

The Phase 6a training needs to look up audio waveforms by sample ID,
but the dictionary wasn't created. This script adds a cell to create it.
"""

import json
import sys

def fix_sample_audio_dict(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find Phase 6a training cell (cell 75)
    phase6a_cell_idx = None
    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'Phase 6a' in source and 'BATCH_SIZE' in source and 'valid_kd = [x for x in kd_data' in source:
            phase6a_cell_idx = i
            break
    
    if phase6a_cell_idx is None:
        print("ERROR: Could not find Phase 6a training cell")
        return False
    
    print(f"Found Phase 6a training in cell {phase6a_cell_idx}")
    
    # Check if the cell already has sample_id_to_audio creation
    phase6a_source = ''.join(nb['cells'][phase6a_cell_idx]['source'])
    if 'sample_id_to_audio = {' in phase6a_source:
        print("✓ sample_id_to_audio already exists in the cell")
        return True
    
    # Get the current cell source
    current_source = nb['cells'][phase6a_cell_idx]['source']
    if isinstance(current_source, str):
        current_lines = current_source.split('\n')
    else:
        current_lines = current_source
    
    # Find where to insert the dictionary creation (after imports, before data validation)
    insert_idx = None
    for i, line in enumerate(current_lines):
        if 'valid_kd = [x for x in kd_data' in line:
            insert_idx = i
            break
    
    if insert_idx is None:
        print("ERROR: Could not find insertion point")
        return False
    
    # Create the dictionary creation code
    dict_creation = [
        '',
        '# ── Create audio lookup dictionary ────────────────────────────────────────────',
        '# Map sample IDs to audio waveforms for training',
        'print("Creating sample_id_to_audio dictionary...")',
        'sample_id_to_audio = {}',
        '',
        '# Add from ft_samples (training data)',
        'if "ft_samples" in globals() and ft_samples is not None:',
        '    for s in ft_samples:',
        '        if "id" in s and "wav" in s:',
        '            sample_id_to_audio[s["id"]] = s["wav"]',
        '    print(f"  Added {len(sample_id_to_audio)} samples from ft_samples")',
        '',
        '# Add from all_train_samples if available',
        'if "all_train_samples" in globals() and all_train_samples is not None:',
        '    for pair_key, samples in all_train_samples.items():',
        '        for s in samples:',
        '            if "id" in s and "wav" in s:',
        '                sample_id_to_audio[s["id"]] = s["wav"]',
        '    print(f"  Total samples in dictionary: {len(sample_id_to_audio)}")',
        '',
        '# Fallback: create from kd_data if it has wav field',
        'if len(sample_id_to_audio) == 0:',
        '    print("  Warning: ft_samples not found, trying to reconstruct from eval_samples...")',
        '    if "eval_samples" in globals() and eval_samples is not None:',
        '        for s in eval_samples:',
        '            if "id" in s and "wav" in s:',
        '                sample_id_to_audio[s["id"]] = s["wav"]',
        '        print(f"  Added {len(sample_id_to_audio)} samples from eval_samples")',
        '',
        'if len(sample_id_to_audio) == 0:',
        '    raise RuntimeError("Could not create sample_id_to_audio dictionary. "',
        '                       "Make sure ft_samples or eval_samples is loaded.")',
        '',
        'print(f"✓ sample_id_to_audio ready with {len(sample_id_to_audio)} samples")',
        '',
        '# ── Data validation ─────────────────────────────────────────────────────────────',
    ]
    
    # Insert the dictionary creation code
    new_lines = current_lines[:insert_idx] + dict_creation + current_lines[insert_idx:]
    
    # Update the cell
    nb['cells'][phase6a_cell_idx]['source'] = new_lines
    
    # Save backup
    backup_path = notebook_path + '.backup_before_audio_dict_fix'
    import shutil
    shutil.copy(notebook_path, backup_path)
    print(f"Backup saved to: {backup_path}")
    
    # Save notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Fixed sample_id_to_audio in {notebook_path}")
    print("\nAdded code to create sample_id_to_audio dictionary from:")
    print("  1. ft_samples (training data) - primary source")
    print("  2. all_train_samples (if available)")
    print("  3. eval_samples (fallback)")
    print("\nThe dictionary maps sample IDs to audio waveforms for training.")
    
    return True

if __name__ == '__main__':
    notebook_path = 'Alteration/seamless-final.ipynb'
    success = fix_sample_audio_dict(notebook_path)
    sys.exit(0 if success else 1)
