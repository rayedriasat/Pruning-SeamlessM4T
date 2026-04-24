"""
COMPLETE FIXED VERSION OF PHASE 5 - KD EXTRACTION
Copy this entire cell into your notebook to replace the existing Phase 5 cell
"""

# ── Free Phase 3/4 models from VRAM before loading teacher ───────────────────
if 'model_p4' in dir() and model_p4 is not None: 
    del model_p4
if 'model_p3' in dir() and model_p3 is not None: 
    del model_p3
gc.collect(); torch.cuda.empty_cache()
print('VRAM cleared for teacher KD extraction.')
gpu_mem()

KD_DRIVE_PATH  = f'{WORK_DIR}/kd_data_v2.pt'
KD_RCLONE_PATH = f'{GDRIVE_ROOT}/kd_data_v2.pt'

# Try to load existing KD data
if os.path.exists(KD_DRIVE_PATH):
    print(f'KD data found at {KD_DRIVE_PATH}')
    kd_data = torch.load(KD_DRIVE_PATH, map_location='cpu', weights_only=False)
    print(f'Loaded {len(kd_data)} KD samples.')
elif ON_KAGGLE:
    print('Trying to pull KD data from rclone remote...')
    r = subprocess.run(f'rclone copy "{KD_RCLONE_PATH}" "{WORK_DIR}/" --transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M',
                       shell=True, capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(KD_DRIVE_PATH):
        kd_data = torch.load(KD_DRIVE_PATH, map_location='cpu', weights_only=False)
        print(f'Pulled {len(kd_data)} KD samples from Drive.')
    else:
        kd_data = None
        print('KD data not found on Drive — will extract now.')
else:
    kd_data = None
    print('KD data not found — will extract.')

if kd_data is None:
    # ── Load teacher for extraction ──────────────────────────────────────────
    print('Loading teacher model (1805M) for KD extraction...')
    teacher, _proc_t = load_base_model()
    teacher.eval()

    # Hook to capture T2U encoder inputs (text dec outputs fed to T2U)
    # FIXED VERSION - handles empty tuples and None inputs safely
    t2u_enc_inputs = {}
    
    def _hook_t2u_enc_in(module, inp, out):
        '''Safely capture T2U encoder inputs with robust error handling'''
        try:
            # Handle None input
            if inp is None:
                return
            
            # Extract tensor from input
            x = None
            if isinstance(inp, tuple):
                if len(inp) == 0:
                    # Empty tuple - skip
                    return
                x = inp[0]
            elif isinstance(inp, torch.Tensor):
                x = inp
            else:
                # Unexpected type - log but don't crash
                return
            
            # Validate and store tensor
            if x is not None and isinstance(x, torch.Tensor):
                t2u_enc_inputs['last'] = x.detach().cpu()
                
        except Exception as e:
            # Silent fail - don't interrupt training
            pass
    
    # Register hook
    _hook_handle = teacher.t2u_model.model.encoder.register_forward_hook(_hook_t2u_enc_in)

    # ── Build multilingual train set ──────────────────────────────────────────
    all_train_samples = {}
    for src_m4t, tgt_m4t in EVAL_LANG_PAIRS:
        pair_key = f'{src_m4t}2{tgt_m4t}'
        pair_samples = [s for s in ft_samples if s['src_lang']==src_m4t and s['tgt_lang']==tgt_m4t]
        all_train_samples[pair_key] = pair_samples[:200]

    kd_data = []
    PAIRS = EVAL_LANG_PAIRS  # All 5 language pairs, bidirectional

    for src_m4t, tgt_m4t in PAIRS:
        samples_here = all_train_samples.get(f'{src_m4t}2{tgt_m4t}', ft_samples[:200])
        print(f'\nExtracting KD: {src_m4t}→{tgt_m4t} ({len(samples_here)} samples)...')
        
        successful = 0
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
                
                successful += 1
                
                if (i+1) % 50 == 0:
                    print(f'  [{i+1}/{len(samples_here)}] {successful} successful, {len(kd_data)} total KD samples')
                    
            except Exception as e:
                print(f'  [{i+1}] Error: {str(e)[:100]}')
                continue
            
            # Clear CUDA cache periodically
            if (i+1) % 20 == 0:
                torch.cuda.empty_cache()
        
        print(f'  Completed {src_m4t}→{tgt_m4t}: {successful}/{len(samples_here)} successful')

    # Remove hook
    _hook_handle.remove()
    
    # Save KD data
    print(f'\nKD extraction complete: {len(kd_data)} samples')
    torch.save(kd_data, KD_DRIVE_PATH)
    if ON_KAGGLE: 
        _rclone_push(KD_DRIVE_PATH, '')
    print(f'KD data saved to {KD_DRIVE_PATH}')

    # Free teacher
    del teacher
    gc.collect(); torch.cuda.empty_cache()
    print('Teacher unloaded from VRAM.')
    gpu_mem()

# ── KD data statistics ────────────────────────────────────────────────────────
valid_t2u   = sum(1 for x in kd_data if x.get('t2u_input') is not None)
valid_units = sum(1 for x in kd_data if x.get('unit_ids') is not None)
n_toks      = [x['n_tokens'] for x in kd_data if x['n_tokens']>0]

print(f'\n=== KD Data Summary ===')
print(f'Total samples: {len(kd_data)}')
print(f'Valid T2U inputs: {valid_t2u} ({valid_t2u/len(kd_data)*100:.1f}%)')
print(f'Valid unit IDs: {valid_units} ({valid_units/len(kd_data)*100:.1f}%)')
if n_toks:
    print(f'Avg tokens per sample: {np.mean(n_toks):.1f} (min={min(n_toks)}, max={max(n_toks)})')

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Phase 5: KD Data Statistics', fontweight='bold')

from collections import Counter
pair_counts = Counter(f"{x['src_lang']}→{x['tgt_lang']}" for x in kd_data)
axes[0].bar(pair_counts.keys(), pair_counts.values(), color='#4CAF50', alpha=0.8)
axes[0].set_title('KD samples per language pair')
axes[0].tick_params(axis='x', rotation=30)

if n_toks:
    axes[1].hist(n_toks, bins=20, color='#2196F3', alpha=0.8, edgecolor='white')
    axes[1].set_title(f'T2U input length (μ={np.mean(n_toks):.1f})')
    axes[1].set_xlabel('n_tokens')

spk_norms = [x['spk_emb'].norm().item() for x in kd_data if x.get('spk_emb') is not None]
if spk_norms:
    axes[2].hist(spk_norms, bins=20, color='#FF5722', alpha=0.8, edgecolor='white')
    axes[2].set_title('ECAPA embedding norms')
    axes[2].set_xlabel('L2 norm')

plt.tight_layout()
save_figure(fig, 'phase5_kd_stats.png')
plt.show()

print(f'\n✓ Phase 5 complete: {len(kd_data)} KD samples ready for training')
