import json

# Read notebook
with open('Alteration/seamless-final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix the Phase 5 cell with the hook
fixed = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Find the cell with the hook definition
        if 'def _hook_t2u_enc_in(module, inp, out):' in source and 't2u_enc_inputs' in source:
            print('Found Phase 5 hook cell')
            
            # Replace the problematic hook with safer version
            old_hook = """    def _hook_t2u_enc_in(module, inp, out):
        x = inp[0] if isinstance(inp, tuple) else inp
        t2u_enc_inputs['last'] = x.detach().cpu()"""
            
            new_hook = """    def _hook_t2u_enc_in(module, inp, out):
        \"\"\"Safely capture T2U encoder inputs\"\"\"
        try:
            if inp is None:
                return
            # Extract tensor from input
            if isinstance(inp, tuple):
                if len(inp) == 0:
                    return
                x = inp[0]
            elif isinstance(inp, torch.Tensor):
                x = inp
            else:
                return
            # Validate and store
            if x is not None and isinstance(x, torch.Tensor):
                t2u_enc_inputs['last'] = x.detach().cpu()
        except Exception as e:
            print(f'  [Hook] Error: {e}')"""
            
            if old_hook in source:
                source = source.replace(old_hook, new_hook)
                
                # Also add validation in the extraction loop
                old_loop = """                with torch.no_grad():
                    out = teacher.generate(**inp, tgt_lang=tgt_m4t,
                                           return_intermediate_token_ids=True)
                t2u_in = t2u_enc_inputs.get('last')
                uid = getattr(out,'unit_ids',None)"""
                
                new_loop = """                with torch.no_grad():
                    out = teacher.generate(**inp, tgt_lang=tgt_m4t,
                                           return_intermediate_token_ids=True)
                t2u_in = t2u_enc_inputs.get('last')
                if t2u_in is None:
                    print(f'  [{i+1}] Warning: T2U input not captured, skipping')
                    continue
                uid = getattr(out,'unit_ids',None)"""
                
                if old_loop in source:
                    source = source.replace(old_loop, new_loop)
                    print('Added validation in extraction loop')
                
                # Update cell source - split back into lines
                cell['source'] = source.split('\n')
                # Ensure proper line endings for notebook format
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                
                fixed = True
                print('Fixed hook function')
                break

if fixed:
    # Save the fixed notebook
    with open('Alteration/seamless-final.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print('✓ Notebook saved with fixes')
else:
    print('✗ Could not find the hook cell to fix')
