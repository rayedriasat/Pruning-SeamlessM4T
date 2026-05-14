#!/usr/bin/env python3
"""
Apply Phase 6 fix to pragmata-recovery.ipynb
Fixes the CUDA device-side assertion error by using pre-tokenized teacher sequences
"""

import json
import sys

def fix_text_recovery_step(cell_source):
    """Fix the text_recovery_step function to use pre-tokenized teacher sequences"""
    
    old_code = """def text_recovery_step(sample, cache_entry, use_teacher_text):
    audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
    target_text = cache_entry['teacher_text_str'] if use_teacher_text else sample['ref']
    labels = build_target_labels(processor, [target_text], sample['tgt_lang'], student_device)

    outputs = model_student(
        **audio_inputs,
        labels=labels,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return outputs.loss"""
    
    new_code = """def text_recovery_step(sample, cache_entry, use_teacher_text):
    audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
    
    if use_teacher_text:
        # Use pre-tokenized teacher sequences directly from cache
        labels = cache_entry['teacher_text_sequences'].unsqueeze(0).to(student_device)
        # Mask padding tokens
        labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    else:
        # Use ground truth reference text
        labels = build_target_labels(processor, [sample['ref']], sample['tgt_lang'], student_device)

    outputs = model_student(
        **audio_inputs,
        labels=labels,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return outputs.loss"""
    
    # Join cell source if it's a list
    if isinstance(cell_source, list):
        source_str = ''.join(cell_source)
    else:
        source_str = cell_source
    
    # Check if old code exists
    if old_code in source_str:
        print("✓ Found text_recovery_step function - applying fix...")
        source_str = source_str.replace(old_code, new_code)
        
        # Split back into lines for notebook format
        if isinstance(cell_source, list):
            return source_str.split('\n')
        return source_str
    
    return cell_source


def fix_cache_validation(cell_source):
    """Add validation to cache building function"""
    
    # Join cell source if it's a list
    if isinstance(cell_source, list):
        source_str = ''.join(cell_source)
    else:
        source_str = cell_source
    
    # Look for the cache building function
    marker = "teacher_text_sequences = out.sequences[0].detach().cpu()"
    
    if marker in source_str:
        print("✓ Found cache building function - adding validation...")
        
        validation_code = """
    
    # Validate sequence length
    if teacher_text_sequences.numel() == 0:
        return None, 'empty_teacher_sequence'
    
    if teacher_text_sequences.numel() > 512:  # max position embeddings
        return None, f'teacher_sequence_too_long:{teacher_text_sequences.numel()}'
"""
        
        # Insert validation after the marker
        source_str = source_str.replace(
            marker,
            marker + validation_code
        )
        
        # Split back into lines for notebook format
        if isinstance(cell_source, list):
            return source_str.split('\n')
        return source_str
    
    return cell_source


def main():
    notebook_path = 'AAA/pragmata-recovery.ipynb'
    
    print(f"Loading notebook: {notebook_path}")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error loading notebook: {e}")
        sys.exit(1)
    
    print(f"Found {len(notebook['cells'])} cells")
    
    fixes_applied = 0
    
    # Process each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = cell.get('source', [])
        if not source:
            continue
        
        # Try to apply fixes
        new_source = fix_text_recovery_step(source)
        if new_source != source:
            cell['source'] = new_source
            fixes_applied += 1
            print(f"  Cell {i}: Applied text_recovery_step fix")
            continue
        
        new_source = fix_cache_validation(source)
        if new_source != source:
            cell['source'] = new_source
            fixes_applied += 1
            print(f"  Cell {i}: Applied cache validation fix")
    
    if fixes_applied == 0:
        print("\n⚠ No fixes applied - code patterns not found or already fixed")
        sys.exit(0)
    
    # Save fixed notebook
    output_path = notebook_path
    backup_path = notebook_path + '.backup_before_phase6_fix'
    
    print(f"\nCreating backup: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Saving fixed notebook: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"\n✓ Successfully applied {fixes_applied} fix(es)")
    print("\nNext steps:")
    print("1. Restart your Kaggle kernel")
    print("2. Run all cells up to Phase 6")
    print("3. The CUDA assertion error should be resolved")


if __name__ == '__main__':
    main()
