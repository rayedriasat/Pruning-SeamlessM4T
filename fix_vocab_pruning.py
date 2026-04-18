import json

# Load the notebook
with open('cse465v6-s2st-optimised.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the apply_vocab_pruning function and add the old_to_new mapping storage
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    # Fix 1: Add old_to_new mapping storage in apply_vocab_pruning
    if 'def apply_vocab_pruning(mdl, used_token_ids):' in source and 'mdl2._vocab_remap_to_old = used_token_ids' in source:
        old_line = '    mdl2._vocab_remap_to_old = used_token_ids   # new_id i → old_id used_token_ids[i]\n\n    return mdl2, old_to_new'
        new_line = '''    mdl2._vocab_remap_to_old = used_token_ids   # new_id i → old_id used_token_ids[i]
    
    # 4. CRITICAL: Store the old_to_new mapping for input token remapping
    mdl2._vocab_old_to_new = old_to_new

    return mdl2, old_to_new'''
        
        if old_line in source:
            new_source = source.replace(old_line, new_line)
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"✓ Fixed apply_vocab_pruning in cell {i}")

# Find and fix the _save_custom_state function to save old_to_new mapping
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    if 'def _save_custom_state(mdl, path):' in source and "_vocab_remap_to_old" in source:
        old_line = "    for attr in ['_vocab_remap_to_old']:"
        new_line = "    for attr in ['_vocab_remap_to_old', '_vocab_old_to_new']:"
        
        if old_line in source:
            new_source = source.replace(old_line, new_line)
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"✓ Fixed _save_custom_state in cell {i}")

# Find and add input token remapping in run_s2st and run_s2tt_only
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    # Fix run_s2st to remap input tokens
    if '@torch.no_grad()\ndef run_s2st(mdl, audio_array' in source:
        # Add remapping logic after inputs are created
        old_block = '''    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}

    # ── 1. Speech generation (full S2ST path) ────────────────────────────────
    with torch.autocast(DEVICE, dtype=DTYPE):'''
        
        new_block = '''    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}
    
    # CRITICAL: Remap input_ids if vocabulary was pruned
    if hasattr(mdl, '_vocab_old_to_new') and 'input_ids' in inputs:
        old_to_new = mdl._vocab_old_to_new
        input_ids = inputs['input_ids']
        # Remap each token ID, keeping special tokens and unmapped tokens as-is
        remapped = input_ids.clone()
        for old_id, new_id in old_to_new.items():
            remapped[input_ids == old_id] = new_id
        inputs['input_ids'] = remapped

    # ── 1. Speech generation (full S2ST path) ────────────────────────────────
    with torch.autocast(DEVICE, dtype=DTYPE):'''
        
        if old_block in source:
            new_source = source.replace(old_block, new_block)
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"✓ Fixed run_s2st input remapping in cell {i}")

# Fix run_s2tt_only similarly
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    if '@torch.no_grad()\ndef run_s2tt_only(mdl, audio_array' in source:
        old_block = '''    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}

    # Temporarily replace vocoder with no-op to skip audio generation'''
        
        new_block = '''    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}
    
    # CRITICAL: Remap input_ids if vocabulary was pruned
    if hasattr(mdl, '_vocab_old_to_new') and 'input_ids' in inputs:
        old_to_new = mdl._vocab_old_to_new
        input_ids = inputs['input_ids']
        remapped = input_ids.clone()
        for old_id, new_id in old_to_new.items():
            remapped[input_ids == old_id] = new_id
        inputs['input_ids'] = remapped

    # Temporarily replace vocoder with no-op to skip audio generation'''
        
        if old_block in source:
            new_source = source.replace(old_block, new_block)
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"✓ Fixed run_s2tt_only input remapping in cell {i}")

# Save the modified notebook
with open('cse465v6-s2st-optimised.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n✅ Vocabulary pruning fix applied successfully!")
print("\nWhat was fixed:")
print("1. Added _vocab_old_to_new mapping storage in apply_vocab_pruning()")
print("2. Updated _save_custom_state() to save the old_to_new mapping")
print("3. Added input token ID remapping in run_s2st() before model.generate()")
print("4. Added input token ID remapping in run_s2tt_only() before model.generate()")
print("\nThis ensures that processor-generated token IDs (from original vocab)")
print("are remapped to the pruned vocabulary space before being fed to the model.")
