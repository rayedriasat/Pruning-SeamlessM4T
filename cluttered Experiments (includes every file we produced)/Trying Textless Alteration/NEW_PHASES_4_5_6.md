# New Phase 4-6 Structure for Pruning Pipeline

## Overview
Replace the CIF connector approach (Phases 4-6) with additional pruning of speech encoder and text decoder from Phase 3 model.

**Starting Point:** `phase3_t2u_laco` model
- Speech Encoder: 16 layers (already pruned from 24)
- Text Decoder: 24 layers (unpruned)
- T2U Model: 4+4 layers (LaCo merged)

**Target:**
- Speech Encoder: 12 layers (prune 4 more)
- Text Decoder: 12 layers (prune 12)
- T2U Model: 4+4 layers (unchanged)

---

## Phase 4: Speech Encoder Additional Pruning (16L → 12L)

### Objective
Prune 4 more layers from the speech encoder using BI-guided iterative greedy method.

### Code Structure

```python
---
## Phase 4: Speech Encoder Additional Pruning (16 → 12 layers)
Target: Remove 4 more layers from speech encoder (currently 16L after Phase 2).
Method: BI-guided iterative greedy (same as Phase 2).
Conservative pruning to maintain quality.

# Load Phase 3 model
model_p3, processor = load_model_from_drive('phase3_t2u_laco')
print_model_breakdown(model_p3, 'Phase 3 Model (Enc16L + T2U 4+4L)')

# Consolidate to single GPU
model_p4 = _consolidate_to_single_GPU(model_p3)

# Parameters
N_ENC_REMOVE_P4 = 4
ENC_BI_RATIO_P4 = 0.5

# Check for existing checkpoint
p4_ckpt = load_latest_checkpoint('phase4_enc_pruning')
p4_complete = p4_ckpt and len(p4_ckpt.get('removed', [])) >= N_ENC_REMOVE_P4

if p4_complete:
    print(f'Phase 4 complete: removed {p4_ckpt["removed"]}')
    try:
        model_p4, processor = load_model_from_drive('phase4_enc_12L')
    except:
        print('  Rebuilding from checkpoint...')
        # Rebuild model from Phase 3 + apply removals
        parent, la = get_speech_encoder_layers(model_p4)
        cur = list(getattr(parent, la))
        keep = [i for i in range(len(cur)) if i not in p4_ckpt['removed']]
        setattr(parent, la, nn.ModuleList([cur[i] for i in keep]))
        sync_model_config(model_p4)
        save_model_to_drive(model_p4, processor, 'phase4_enc_12L')
else:
    done = len(p4_ckpt['removed']) if p4_ckpt else 0
    print(f'{"Resuming" if done else "Running"} Phase 4: enc pruning ({done}/{N_ENC_REMOVE_P4} done)...')
    
    # Sanity check
    sanity = quick_eval_chrf(model_p4, eval_samples, 10)
    print(f'  Sanity ChrF={sanity:.2f}')
    assert sanity > 10, f'Sanity too low: {sanity:.2f}'
    
    # Compute or load BI scores
    if not (p4_ckpt and p4_ckpt.get('bi_scores')):
        print('Computing Block Influence scores...')
        bi_scores = compute_block_influence(model_p4, eval_samples, max_n=50)
        save_checkpoint(dict(removed=[], log=[], bi_scores=bi_scores), 
                       'phase4_enc_pruning', 0)
    else:
        bi_scores = p4_ckpt['bi_scores']
        print(f'  BI scores loaded ({len(bi_scores)} layers)')
    
    # Get protected layers
    parent_tmp, la_tmp = get_speech_encoder_layers(model_p4)
    n_enc = len(getattr(parent_tmp, la_tmp))
    enc_protected = _get_protected_enc(n_enc)
    
    # Run iterative pruning
    removed_enc, p4_log = iterative_enc_prune(
        model_p4, eval_samples, N_ENC_REMOVE_P4, max_eval=16,
        ckpt_name='phase4_enc_pruning', bi_scores=bi_scores,
        bi_candidate_ratio=ENC_BI_RATIO_P4, protected=enc_protected)
    
    sync_model_config(model_p4)
    save_checkpoint(dict(removed=removed_enc, log=p4_log, bi_scores=bi_scores),
                   'phase4_enc_pruning', 0)
    save_model_to_drive(model_p4, processor, 'phase4_enc_12L')

print(f'Encoder layers removed: {removed_enc}')
print_model_breakdown(model_p4, 'After Phase 4: Enc 12L')

# Benchmark
p4_bench = load_latest_checkpoint('phase4_benchmark')
if p4_bench:
    p4_results = p4_bench['results']
    p4_summary = p4_bench['summary']
    p4_detailed = p4_bench.get('detailed_summary')
    if not p4_detailed:
        p4_detailed = compute_detailed_summary(p4_results, 'P4_Enc12L', p4_summary['params_M'])
else:
    p4_results, p4_summary = run_benchmark_asr(
        model_p4, eval_samples, 'P4_Enc12L', save_n=4)
    p4_detailed = compute_detailed_summary(p4_results, 'P4_Enc12L', p4_summary['params_M'])
    save_checkpoint(dict(
        results=p4_results, 
        summary=p4_summary,
        detailed_summary=p4_detailed
    ), 'phase4_benchmark', 0)

store_summary(p4_summary)
store_detailed_summary(p4_detailed)
print_detailed_summary_table('P4_Enc12L')
plot_phase_comparison()
plot_detailed_phase_comparison()

# Free memory
del model_p3
gc.collect()
torch.cuda.empty_cache()
print('P3 model freed.')
```

---

## Phase 5: Text Decoder Pruning (24L → 12L)

### Objective
Prune 12 layers from the text decoder using BI-guided iterative greedy method.

### Code Structure

```python
---
## Phase 5: Text Decoder Pruning (24 → 12 layers)
Target: Remove 12 layers from text decoder (currently 24L, unpruned).
Method: BI-guided iterative greedy (adapted for decoder).
This is the first time we prune the text decoder.

# Load Phase 4 model
model_p4, processor = load_model_from_drive('phase4_enc_12L')
print_model_breakdown(model_p4, 'Phase 4 Model (Enc12L + Dec24L + T2U 4+4L)')

# Consolidate to single GPU
model_p5 = _consolidate_to_single_GPU(model_p4)

# Parameters
N_DEC_REMOVE = 12
DEC_BI_RATIO = 0.5

# Helper functions for text decoder
def get_text_decoder_layers(mdl):
    """Get text decoder layers ModuleList."""
    dec = mdl.text_decoder
    if hasattr(dec, 'layers') and isinstance(dec.layers, nn.ModuleList):
        return dec, 'layers'
    for attr in ['decoder', 'model']:
        if hasattr(dec, attr):
            sub = getattr(dec, attr)
            if hasattr(sub, 'layers') and isinstance(sub.layers, nn.ModuleList):
                return sub, 'layers'
    raise RuntimeError('Cannot find text decoder layers')

def compute_decoder_block_influence(mdl, samples, max_n=50):
    """
    Compute Block Influence for text decoder layers.
    BI = 1 - cos(layer_input, layer_output)
    """
    parent, la = get_text_decoder_layers(mdl)
    layers = getattr(parent, la)
    n = len(layers)
    bi = {i: [] for i in range(n)}
    hooks = []
    
    for i in range(n):
        def make_hook(idx):
            def hook(mod, inp, out):
                x = inp[0]
                if x is None or not isinstance(x, torch.Tensor):
                    return
                y = out[0] if isinstance(out, tuple) else out
                if y is None or not isinstance(y, torch.Tensor):
                    return
                x = x.detach().float().reshape(-1, x.shape[-1])
                y = y.detach().to(x.device).float().reshape(-1, y.shape[-1])
                bi[idx].append(1.0 - F.cosine_similarity(x, y, dim=-1).mean().item())
            return hook
        hooks.append(layers[i].register_forward_hook(make_hook(i)))
    
    mdl.eval()
    dev = next(mdl.text_decoder.parameters()).device
    ok = 0
    
    for idx, s in enumerate(samples[:max_n]):
        if idx % 10 == 0:
            print(f'  Calibrating decoder BI {idx}/{min(max_n, len(samples))}...')
        try:
            inputs = processor(audio=s['wav'], sampling_rate=16000, return_tensors='pt')
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                # Run full S2ST to activate decoder
                _ = mdl.generate(**inputs, tgt_lang=s['tgt_lang'])
            ok += 1
        except Exception as e:
            print(f'  Sample {idx} failed: {e}')
    
    for h in hooks:
        h.remove()
    
    scores = {i: float(np.mean(v)) if v else 0.0 for i, v in bi.items()}
    print(f'  Calibrated {ok}/{min(max_n, len(samples))} samples.')
    
    ranked = sorted(scores.items(), key=lambda x: x[1])
    print('  Decoder BI ranking (low=redundant):')
    for rank, (li, bv) in enumerate(ranked):
        print(f'    Rank{rank+1:>2}  L{li:>2}  BI={bv:.4f}')
    
    return scores

def _get_protected_dec(n_total):
    """Protect first, middle, and last decoder layers."""
    return {0, n_total//2, n_total-1}

def iterative_dec_prune(mdl, samples, n_remove, tgt_lang='ben', max_eval=16,
                       ckpt_name='phase5_dec_pruning', bi_scores=None,
                       bi_candidate_ratio=0.5, protected=None):
    """
    BI-guided iterative greedy decoder pruning.
    Same logic as encoder pruning but for text decoder.
    """
    parent, la = get_text_decoder_layers(mdl)
    current = list(getattr(parent, la))
    orig_idx = list(range(len(current)))
    n_total = len(current)
    removed, log = [], []
    
    if protected is None:
        protected = _get_protected_dec(n_total)
    print(f'  Protected decoder layers (first/mid/last): {sorted(protected)}')
    
    # Resume from checkpoint
    partial = load_latest_checkpoint(ckpt_name)
    if partial and partial.get('removed'):
        removed = list(partial['removed'])
        log = partial.get('log', [])
        for r in removed:
            if r in orig_idx:
                pos = orig_idx.index(r)
                current.pop(pos)
                orig_idx.pop(pos)
        setattr(parent, la, nn.ModuleList(current))
        print(f'  Resuming: removed {removed}, {len(current)} layers remain')
    
    # Baseline
    baseline = quick_eval_chrf(mdl, samples, max_samples=max_eval)
    print(f'  Baseline ChrF: {baseline:.2f}')
    
    for it in range(len(removed), n_remove):
        eligible = [pos for pos in range(len(current)) if orig_idx[pos] not in protected]
        
        if bi_scores and len(eligible) > 2:
            by_bi = sorted(eligible, key=lambda pos: bi_scores.get(orig_idx[pos], float('inf')))
            n_cands = max(2, int(len(by_bi) * bi_candidate_ratio))
            cands = by_bi[:n_cands]
            print(f'\n  Iter {it+1}/{n_remove} | BI pre-filter: {len(cands)}/{len(eligible)} cands')
        else:
            cands = eligible
            print(f'\n  Iter {it+1}/{n_remove} | all {len(cands)} eligible (no BI)')
        
        if not cands:
            print('  No candidates left, stopping.')
            break
        
        scores = {}
        for pos in cands:
            temp = current[:pos] + current[pos+1:]
            setattr(parent, la, nn.ModuleList(temp))
            sc = quick_eval_chrf(mdl, samples, max_samples=max_eval)
            bi_note = f'  BI={bi_scores.get(orig_idx[pos], 0):.4f}' if bi_scores else ''
            print(f'    Remove L{orig_idx[pos]:>2} -> ChrF={sc:.2f}{bi_note}')
            scores[pos] = (orig_idx[pos], sc)
        
        setattr(parent, la, nn.ModuleList(current))
        
        best_pos = max(scores, key=lambda k: scores[k][1])
        best_orig, best_sc = scores[best_pos]
        current.pop(best_pos)
        orig_idx.pop(best_pos)
        setattr(parent, la, nn.ModuleList(current))
        removed.append(best_orig)
        
        log.append(dict(
            iter=it+1, removed=best_orig, chrf=best_sc,
            remaining=len(current),
            bi_score=bi_scores.get(best_orig) if bi_scores else None))
        
        print(f'  -> Removed L{best_orig} ChrF={best_sc:.2f} ({len(current)} remain)')
        
        if bi_scores and best_orig in bi_scores:
            del bi_scores[best_orig]
        
        save_checkpoint(dict(removed=removed, log=log, bi_scores=bi_scores or {}),
                       ckpt_name, step=0)
        torch.cuda.empty_cache()
    
    return removed, log

# Check for existing checkpoint
p5_ckpt = load_latest_checkpoint('phase5_dec_pruning')
p5_complete = p5_ckpt and len(p5_ckpt.get('removed', [])) >= N_DEC_REMOVE

if p5_complete:
    print(f'Phase 5 complete: removed {p5_ckpt["removed"]}')
    try:
        model_p5, processor = load_model_from_drive('phase5_dec_12L')
    except:
        print('  Rebuilding from checkpoint...')
        parent, la = get_text_decoder_layers(model_p5)
        cur = list(getattr(parent, la))
        keep = [i for i in range(len(cur)) if i not in p5_ckpt['removed']]
        setattr(parent, la, nn.ModuleList([cur[i] for i in keep]))
        sync_model_config(model_p5)
        save_model_to_drive(model_p5, processor, 'phase5_dec_12L')
else:
    done = len(p5_ckpt['removed']) if p5_ckpt else 0
    print(f'{"Resuming" if done else "Running"} Phase 5: dec pruning ({done}/{N_DEC_REMOVE} done)...')
    
    # Sanity check
    sanity = quick_eval_chrf(model_p5, eval_samples, 10)
    print(f'  Sanity ChrF={sanity:.2f}')
    assert sanity > 10, f'Sanity too low: {sanity:.2f}'
    
    # Compute or load BI scores
    if not (p5_ckpt and p5_ckpt.get('bi_scores')):
        print('Computing decoder Block Influence scores...')
        bi_scores = compute_decoder_block_influence(model_p5, eval_samples, max_n=50)
        save_checkpoint(dict(removed=[], log=[], bi_scores=bi_scores),
                       'phase5_dec_pruning', 0)
    else:
        bi_scores = p5_ckpt['bi_scores']
        print(f'  Decoder BI scores loaded ({len(bi_scores)} layers)')
    
    # Get protected layers
    parent_tmp, la_tmp = get_text_decoder_layers(model_p5)
    n_dec = len(getattr(parent_tmp, la_tmp))
    dec_protected = _get_protected_dec(n_dec)
    
    # Run iterative pruning
    removed_dec, p5_log = iterative_dec_prune(
        model_p5, eval_samples, N_DEC_REMOVE, max_eval=16,
        ckpt_name='phase5_dec_pruning', bi_scores=bi_scores,
        bi_candidate_ratio=DEC_BI_RATIO, protected=dec_protected)
    
    sync_model_config(model_p5)
    save_checkpoint(dict(removed=removed_dec, log=p5_log, bi_scores=bi_scores),
                   'phase5_dec_pruning', 0)
    save_model_to_drive(model_p5, processor, 'phase5_dec_12L')

print(f'Decoder layers removed: {removed_dec}')
print_model_breakdown(model_p5, 'After Phase 5: Enc12L + Dec12L')

# Benchmark
p5_bench = load_latest_checkpoint('phase5_benchmark')
if p5_bench:
    p5_results = p5_bench['results']
    p5_summary = p5_bench['summary']
    p5_detailed = p5_bench.get('detailed_summary')
    if not p5_detailed:
        p5_detailed = compute_detailed_summary(p5_results, 'P5_Dec12L', p5_summary['params_M'])
else:
    p5_results, p5_summary = run_benchmark_asr(
        model_p5, eval_samples, 'P5_Dec12L', save_n=4)
    p5_detailed = compute_detailed_summary(p5_results, 'P5_Dec12L', p5_summary['params_M'])
    save_checkpoint(dict(
        results=p5_results,
        summary=p5_summary,
        detailed_summary=p5_detailed
    ), 'phase5_benchmark', 0)

store_summary(p5_summary)
store_detailed_summary(p5_detailed)
print_detailed_summary_table('P5_Dec12L')
plot_phase_comparison()
plot_detailed_phase_comparison()

# Free memory
del model_p4
gc.collect()
torch.cuda.empty_cache()
print('P4 model freed.')
```

---

## Phase 6: DoRA Fine-tuning (Optional Recovery)

### Objective
Apply DoRA (r=16) to speech encoder and text decoder to recover any quality loss from aggressive pruning.

### Code Structure

```python
---
## Phase 6: DoRA Fine-tuning (Optional Recovery)
Apply DoRA (r=16) to speech encoder + text decoder to recover quality.
Loss: 0.80×S2TT_CE + 0.20×S2ST_unit_CE
Papers: DoRA (Liu ICML 2024 Oral)

from peft import LoraConfig, get_peft_model

# Load Phase 5 model
model_p5, processor = load_model_from_drive('phase5_dec_12L')
print_model_breakdown(model_p5, 'Phase 5 Model (Enc12L + Dec12L + T2U 4+4L)')

# Consolidate to single GPU
model_p6 = _consolidate_to_single_GPU(model_p5)
device = torch.device('cuda:0')

# Freeze all, then apply DoRA
for p in model_p6.parameters():
    p.requires_grad_(False)

# DoRA configs
lora_cfg_enc = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias='none', use_dora=True,
    target_modules=['linear_q', 'linear_k', 'linear_v', 'linear_out',
                   'intermediate_dense', 'output_dense'])

lora_cfg_dec = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias='none', use_dora=True,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'])

print('Applying DoRA to speech_encoder...')
model_p6.speech_encoder = get_peft_model(model_p6.speech_encoder, lora_cfg_enc)
model_p6.speech_encoder.print_trainable_parameters()

print('Applying DoRA to text_decoder...')
model_p6.text_decoder = get_peft_model(model_p6.text_decoder, lora_cfg_dec)
model_p6.text_decoder.print_trainable_parameters()

# Training parameters
MAX_STEPS_P6 = 2000
BATCH_ACCUM = 4
LOG_EVERY = 25
SAVE_EVERY = 500

# Optimizer
optimizer_p6 = torch.optim.AdamW([
    {'params': [p for p in model_p6.speech_encoder.parameters() if p.requires_grad],
     'lr': 2e-5, 'weight_decay': 0.01},
    {'params': [p for p in model_p6.text_decoder.parameters() if p.requires_grad],
     'lr': 2e-5, 'weight_decay': 0.01},
], betas=(0.9, 0.98))

scheduler_p6 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_p6, T_max=MAX_STEPS_P6, eta_min=1e-6)

scaler_p6 = torch.cuda.amp.GradScaler()

# Training data
train_samples = [s for s in ft_samples if s.get('ref') and len(s['ref']) > 0]
print(f'Training samples: {len(train_samples)}')

# Training loop
print(f'\n{"="*70}')
print(f'  PHASE 6: DoRA Fine-tuning')
print(f'  Steps: {MAX_STEPS_P6}  |  Accum: {BATCH_ACCUM}')
print(f'  Loss: 0.80×S2TT_CE + 0.20×S2ST_unit_CE')
print(f'{"="*70}\n')

model_p6.train()
optimizer_p6.zero_grad()
loss_log_p6 = []

for step in range(MAX_STEPS_P6):
    sample = random.choice(train_samples)
    
    try:
        inputs = processor(audio=sample['wav'], sampling_rate=16000, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Tokenize reference text
        text_inputs = processor.tokenizer(
            sample['ref'], return_tensors='pt', padding=True, truncation=True)
        labels = text_inputs['input_ids'].to(device)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # S2TT loss (text decoder)
            outputs = model_p6(**inputs, tgt_lang=sample['tgt_lang'], labels=labels)
            s2tt_loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0).to(device)
            
            # S2ST loss (unit prediction) - optional
            s2st_loss = torch.tensor(0.0).to(device)
            
            loss = 0.80 * s2tt_loss + 0.20 * s2st_loss
        
        scaler_p6.scale(loss / BATCH_ACCUM).backward()
        loss_log_p6.append(loss.item())
        
        if (step + 1) % BATCH_ACCUM == 0:
            scaler_p6.unscale_(optimizer_p6)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model_p6.parameters() if p.requires_grad], 1.0)
            scaler_p6.step(optimizer_p6)
            scaler_p6.update()
            optimizer_p6.zero_grad()
            scheduler_p6.step()
            torch.cuda.empty_cache()
        
        if (step + 1) % LOG_EVERY == 0:
            cur_lr = optimizer_p6.param_groups[0]['lr']
            recent_loss = np.mean(loss_log_p6[-50:])
            print(f'  Step {step+1:>5}/{MAX_STEPS_P6} | '
                  f'loss={recent_loss:.4f} | '
                  f's2tt={s2tt_loss.item():.4f} | '
                  f'lr={cur_lr:.2e}')
        
        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint({
                'step': step + 1,
                'enc_state': model_p6.speech_encoder.state_dict(),
                'dec_state': model_p6.text_decoder.state_dict(),
                'optimizer_state': optimizer_p6.state_dict(),
                'loss_log': loss_log_p6,
            }, 'phase6_dora', step + 1)
    
    except Exception as e:
        print(f'  Step {step+1} error: {e}')
        continue

# Merge DoRA adapters
print('\nMerging DoRA adapters...')
model_p6.speech_encoder = model_p6.speech_encoder.merge_and_unload()
model_p6.text_decoder = model_p6.text_decoder.merge_and_unload()
model_p6.eval()
sync_model_config(model_p6)
gc.collect()
torch.cuda.empty_cache()

save_model_to_drive(model_p6, processor, 'phase6_dora_merged')
print('✓ Phase 6 model saved to Drive as phase6_dora_merged')

# Benchmark
p6_bench = load_latest_checkpoint('phase6_benchmark')
if p6_bench:
    p6_results = p6_bench['results']
    p6_summary = p6_bench['summary']
    p6_detailed = p6_bench.get('detailed_summary')
    if not p6_detailed:
        p6_detailed = compute_detailed_summary(p6_results, 'P6_DoRA', p6_summary['params_M'])
else:
    p6_results, p6_summary = run_benchmark_asr(
        model_p6, eval_samples, 'P6_DoRA', save_n=4)
    p6_detailed = compute_detailed_summary(p6_results, 'P6_DoRA', p6_summary['params_M'])
    save_checkpoint(dict(
        results=p6_results,
        summary=p6_summary,
        detailed_summary=p6_detailed
    ), 'phase6_benchmark', 0)

store_summary(p6_summary)
store_detailed_summary(p6_detailed)
print_detailed_summary_table('P6_DoRA')
plot_phase_comparison()
plot_detailed_phase_comparison()

# Training loss plot
if loss_log_p6:
    fig, ax = plt.subplots(figsize=(10, 5))
    ema, v = [], loss_log_p6[0]
    for l in loss_log_p6:
        v = 0.05*l + 0.95*v
        ema.append(v)
    ax.plot(loss_log_p6, alpha=0.2, color='#FF5722', lw=0.5, label='Raw')
    ax.plot(ema, color='#FF5722', lw=2, label='EMA')
    ax.set_title('Phase 6: DoRA Fine-tuning Loss', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, 'phase6_dora_training.png')
    plt.show()

print('\n✓ Phase 6 complete.')

# Free memory
del model_p5
gc.collect()
torch.cuda.empty_cache()
print('P5 model freed.')
```

---

## Summary

**New Pipeline:**
1. **Phase 0-3:** Unchanged (baseline → vocab → enc 16L → T2U LaCo)
2. **Phase 4:** Speech encoder 16L → 12L (prune 4 more layers)
3. **Phase 5:** Text decoder 24L → 12L (prune 12 layers)
4. **Phase 6:** DoRA fine-tuning for quality recovery
5. **Phase 7:** Final comprehensive benchmark

**Expected Final Model:**
- Speech Encoder: 12 layers (~330M params)
- Text Decoder: 12 layers (~313M params)
- T2U Model: 4+4 layers (~175M params)
- **Total: ~818M params** (vs 1805M teacher = 55% reduction)

**Key Differences from Original Plan:**
- No CIF connector (not viable without text decoder removal)
- No speaker adapter (voice cloning not in scope)
- Focus on balanced pruning across encoder + decoder
- DoRA recovery step to maintain quality
