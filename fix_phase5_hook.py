"""
Fix for Phase 5 KD extraction hook - tuple index out of range error
This script provides the corrected hook function to paste into your notebook
"""

print("""
CORRECTED HOOK FUNCTION FOR PHASE 5:
=====================================

Replace the hook definition in Phase 5 with this safer version:

# Hook to capture T2U encoder inputs (text dec outputs fed to T2U)
t2u_enc_inputs = {}

def _hook_t2u_enc_in(module, inp, out):
    '''Safely capture T2U encoder inputs'''
    # Handle different input formats
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
            print(f"  [Hook] Unexpected input type: {type(inp)}")
            return
        
        # Validate and store
        if x is not None and isinstance(x, torch.Tensor):
            t2u_enc_inputs['last'] = x.detach().cpu()
        else:
            print(f"  [Hook] Invalid tensor: {type(x)}")
    except Exception as e:
        print(f"  [Hook] Error: {e}")

_hook_handle = teacher.t2u_model.model.encoder.register_forward_hook(_hook_t2u_enc_in)

=====================================

ADDITIONAL FIX - Update the KD extraction loop:

Replace the extraction loop with better error handling:

for i, s in enumerate(samples_here):
    t2u_enc_inputs.clear()
    try:
        # Extract speaker embedding
        spk_emb = extract_speaker_emb(s['wav'])
        
        # Prepare input
        inp = processor(audio=s['wav'], sampling_rate=16000, return_tensors='pt')
        dev = _model_input_device(teacher)
        inp = {k: v.to(dev) for k,v in inp.items() if isinstance(v,torch.Tensor)}
        
        # Generate with teacher
        with torch.no_grad():
            out = teacher.generate(**inp, tgt_lang=tgt_m4t,
                                   return_intermediate_token_ids=True)
        
        # Extract captured T2U input
        t2u_in = t2u_enc_inputs.get('last')
        
        # Validate T2U input was captured
        if t2u_in is None:
            print(f'  [{i+1}] Warning: T2U input not captured, skipping')
            continue
        
        # Extract unit IDs
        uid = getattr(out, 'unit_ids', None)
        if uid is not None:
            uid = uid[0].cpu()
        
        # Store KD sample
        kd_data.append({
            'id': s['id'], 
            'src_lang': src_m4t, 
            'tgt_lang': tgt_m4t,
            't2u_input': t2u_in,
            'unit_ids':  uid,
            'n_tokens':  t2u_in.shape[1] if t2u_in is not None else 0,
            'spk_emb':   spk_emb,
        })
        
        if (i+1) % 50 == 0:
            print(f'  [{i+1}/{len(samples_here)}] {len(kd_data)} total KD samples')
            
    except Exception as e:
        print(f'  [{i+1}] Error: {e}')
        import traceback
        traceback.print_exc()
        continue
    
    # Clear CUDA cache periodically
    if (i+1) % 20 == 0:
        torch.cuda.empty_cache()

=====================================

ROOT CAUSE:
The error occurs because the hook receives an empty tuple or None when the 
T2U encoder is called. This happens when:
1. The model architecture doesn't pass inputs as expected
2. The generate() call doesn't trigger the T2U encoder properly
3. The hook is attached to the wrong module

ALTERNATIVE APPROACH - Use pre_forward_hook instead:

def _hook_t2u_enc_in_pre(module, inp):
    '''Pre-forward hook - captures inputs before processing'''
    try:
        if isinstance(inp, tuple) and len(inp) > 0:
            x = inp[0]
            if isinstance(x, torch.Tensor):
                t2u_enc_inputs['last'] = x.detach().cpu()
    except Exception as e:
        print(f"  [Pre-hook] Error: {e}")

_hook_handle = teacher.t2u_model.model.encoder.register_forward_pre_hook(_hook_t2u_enc_in_pre)

=====================================
""")

print("\nDEBUGGING STEPS:")
print("1. Check if T2U encoder is being called:")
print("   - Add print statement in hook: print(f'Hook called: inp type={type(inp)}, len={len(inp) if isinstance(inp, tuple) else 'N/A'}')")
print("\n2. Verify model structure:")
print("   - Check: teacher.t2u_model.model.encoder")
print("   - Try: list(teacher.t2u_model.named_modules()) to see all modules")
print("\n3. Test with single sample first:")
print("   - Run extraction on just 1 sample to see detailed error")
print("\n4. Alternative: Extract from text_decoder output instead:")
print("   - Hook: teacher.text_decoder (this feeds into T2U)")
