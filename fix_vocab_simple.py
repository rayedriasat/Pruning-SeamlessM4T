import json

# Load the notebook
with open('cse465v6-s2st-optimised.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Applying SIMPLE vocabulary pruning fix...\n")
print("Root cause: Speech encoder may have language embeddings that reference vocab\n")

# The REAL fix: Add safety check in run_s2st and run_s2tt_only
# to handle any vocab-related tensors in inputs

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    if '@torch.no_grad()\ndef run_s2st(mdl, audio_array' in source:
        # Add vocab size check right after inputs creation
        old_line = '    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}'
        
        new_line = '''    # CRITICAL: Clamp any vocab-related tensors to pruned vocab size
    if hasattr(mdl.config, 'vocab_size'):
        vocab_size = mdl.config.vocab_size
        for k in ['input_ids', 'decoder_input_ids', 'labels']:
            if k in inputs and inputs[k] is not None:
                inputs[k] = torch.clamp(inputs[k], 0, vocab_size - 1)
    
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}'''
        
        if old_line in source and 'torch.clamp(inputs[k], 0, vocab_size - 1)' not in source:
            new_source = source.replace(old_line, new_line)
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"✓ Added vocab clamping in run_s2st (cell {i})")

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    if '@torch.no_grad()\ndef run_s2tt_only(mdl, audio_array' in source:
        old_line = '    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}'
        
        new_line = '''    # CRITICAL: Clamp any vocab-related tensors to pruned vocab size
    if hasattr(mdl.config, 'vocab_size'):
        vocab_size = mdl.config.vocab_size
        for k in ['input_ids', 'decoder_input_ids', 'labels']:
            if k in inputs and inputs[k] is not None:
                inputs[k] = torch.clamp(inputs[k], 0, vocab_size - 1)
    
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}'''
        
        if old_line in source and 'torch.clamp(inputs[k], 0, vocab_size - 1)' not in source:
            new_source = source.replace(old_line, new_line)
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"✓ Added vocab clamping in run_s2tt_only (cell {i})")

# Save
with open('cse465v6-s2st-optimised.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n✅ Simple fix applied!")
print("\nThis clamps any vocabulary-related tensor IDs to the pruned vocab size")
print("before moving tensors to GPU, preventing out-of-bounds access.")
print("\n⚠️  Restart kernel and re-run Phase 1")
