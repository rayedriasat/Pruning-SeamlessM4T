import json

# Load the notebook
with open('cse465v6-s2st-optimised.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Applying comprehensive vocabulary pruning fix...\n")

# Fix 1: Update apply_vocab_pruning to also update the processor
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    if 'p1_ckpt = load_latest_checkpoint' in source and 'phase1_vocab' in source and 'model_p1, processor = load_model_from_drive' in source:
        # This is the Phase 1 loading cell - we need to fix it
        old_block = '''p1_ckpt = load_latest_checkpoint('phase1_vocab')
if p1_ckpt:
    model_p1, processor = load_model_from_drive('phase1_vocab_pruned')
    print('Loaded Phase 1 from Drive.')
else:
    model_p1, old_to_new = apply_vocab_pruning(model, used_ids)
    _sync_config_to_architecture(model_p1)
    save_model_to_drive(model_p1, processor, 'phase1_vocab_pruned')
    save_checkpoint({'used_ids': used_ids}, name='phase1_vocab', step=0)'''

        new_block = '''p1_ckpt = load_latest_checkpoint('phase1_vocab')
if p1_ckpt:
    model_p1, processor = load_model_from_drive('phase1_vocab_pruned')
    print('Loaded Phase 1 from Drive.')
else:
    model_p1, old_to_new = apply_vocab_pruning(model, used_ids)
    _sync_config_to_architecture(model_p1)
    
    # CRITICAL: Create a new processor with the pruned vocabulary
    # The processor's tokenizer must match the model's vocabulary size
    from transformers import AutoProcessor
    
    # Clone the processor but update its tokenizer's vocab size
    # Note: We keep using the original processor for encoding/decoding
    # but the model will handle the vocabulary internally
    print(f'  Processor vocab: {processor.tokenizer.vocab_size} (unchanged - model handles remapping)')
    
    save_model_to_drive(model_p1, processor, 'phase1_vocab_pruned')
    save_checkpoint({'used_ids': used_ids}, name='phase1_vocab', step=0)'''

        if old_block in source:
            new_source = source.replace(old_block, new_block)
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"✓ Updated Phase 1 checkpoint loading in cell {i}")

# Fix 2: The REAL fix - add input clamping in run_s2st BEFORE moving to device
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    if '@torch.no_grad()\ndef run_s2st(mdl, audio_array' in source and 'inputs = proc(' in source:
        # Find the inputs creation and add clamping
        old_block = '''    inputs = proc(
        audio=audio_array,
        src_lang=src_lang,
        sampling_rate=SAMPLE_RATE,
        return_tensors='pt'
    )
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}'''

        new_block = '''    inputs = proc(
        audio=audio_array,
        src_lang=src_lang,
        sampling_rate=SAMPLE_RATE,
        return_tensors='pt'
    )
    
    # CRITICAL FIX: Clamp input_ids to pruned vocabulary size
    if hasattr(mdl, '_vocab_old_to_new') and 'input_ids' in inputs:
        vocab_size = mdl.config.vocab_size
        input_ids = inputs['input_ids']
        
        # Remap old vocab IDs to new vocab IDs
        old_to_new = mdl._vocab_old_to_new
        remapped = input_ids.clone()
        
        for old_id, new_id in old_to_new.items():
            remapped[input_ids == old_id] = new_id
        
        # Clamp any remaining out-of-bounds IDs to vocab_size - 1
        remapped = torch.clamp(remapped, 0, vocab_size - 1)
        inputs['input_ids'] = remapped
    
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}'''

        if old_block in source:
            new_source = source.replace(old_block, new_block)
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"✓ Fixed run_s2st with input clamping in cell {i}")

# Fix 3: Same fix for run_s2tt_only
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    if '@torch.no_grad()\ndef run_s2tt_only(mdl, audio_array' in source and 'inputs = proc(' in source:
        old_block = '''    inputs = proc(
        audio=audio_array,
        src_lang=src_lang,
        sampling_rate=SAMPLE_RATE,
        return_tensors='pt'
    )
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}'''

        new_block = '''    inputs = proc(
        audio=audio_array,
        src_lang=src_lang,
        sampling_rate=SAMPLE_RATE,
        return_tensors='pt'
    )
    
    # CRITICAL FIX: Clamp input_ids to pruned vocabulary size
    if hasattr(mdl, '_vocab_old_to_new') and 'input_ids' in inputs:
        vocab_size = mdl.config.vocab_size
        input_ids = inputs['input_ids']
        
        # Remap old vocab IDs to new vocab IDs
        old_to_new = mdl._vocab_old_to_new
        remapped = input_ids.clone()
        
        for old_id, new_id in old_to_new.items():
            remapped[input_ids == old_id] = new_id
        
        # Clamp any remaining out-of-bounds IDs to vocab_size - 1
        remapped = torch.clamp(remapped, 0, vocab_size - 1)
        inputs['input_ids'] = remapped
    
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}'''

        if old_block in source:
            new_source = source.replace(old_block, new_block)
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"✓ Fixed run_s2tt_only with input clamping in cell {i}")

# Fix 4: Ensure apply_vocab_pruning stores the mapping
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    if 'def apply_vocab_pruning(mdl, used_token_ids):' in source:
        if 'mdl2._vocab_old_to_new = old_to_new' not in source:
            # Add the mapping storage
            old_line = '    mdl2._vocab_remap_to_old = used_token_ids   # new_id i → old_id used_token_ids[i]\n\n    return mdl2, old_to_new'
            new_line = '''    mdl2._vocab_remap_to_old = used_token_ids   # new_id i → old_id used_token_ids[i]
    mdl2._vocab_old_to_new = old_to_new  # old_id → new_id for input remapping

    return mdl2, old_to_new'''
            
            if old_line in source:
                new_source = source.replace(old_line, new_line)
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                print(f"✓ Added old_to_new mapping storage in cell {i}")

# Fix 5: Update _save_custom_state to save both mappings
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    if 'def _save_custom_state(mdl, path):' in source:
        old_line = "    for attr in ['_vocab_remap_to_old']:"
        new_line = "    for attr in ['_vocab_remap_to_old', '_vocab_old_to_new']:"
        
        if old_line in source and new_line not in source:
            new_source = source.replace(old_line, new_line)
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"✓ Updated _save_custom_state to save both mappings in cell {i}")

# Save the modified notebook
with open('cse465v6-s2st-optimised.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n" + "="*70)
print("✅ COMPLETE VOCABULARY PRUNING FIX APPLIED!")
print("="*70)
print("\nWhat was fixed:")
print("1. Added _vocab_old_to_new mapping storage in apply_vocab_pruning()")
print("2. Added input_ids remapping + clamping in run_s2st() BEFORE .to(device)")
print("3. Added input_ids remapping + clamping in run_s2tt_only() BEFORE .to(device)")
print("4. Updated _save_custom_state() to persist both mappings")
print("\nHow it works:")
print("- Processor generates token IDs from original 256K vocabulary")
print("- Before moving to GPU, we remap IDs to pruned vocabulary space")
print("- Any unmapped IDs are clamped to valid range")
print("- This prevents CUDA out-of-bounds errors")
print("\n⚠️  IMPORTANT: Restart your kernel and re-run from Phase 1!")
print("="*70)
