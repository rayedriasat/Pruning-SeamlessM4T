"""
PHASE 6A AND 6B FIXES FOR SEAMLESS-FINAL.IPYNB
================================================

Copy these cells into your notebook to fix Phase 6a and 6b.

PHASE 6A ANALYSIS:
------------------
Your training DID learn but didn't converge:
- Cosine loss: 0.47 → 0.37 (good progress, target < 0.10)
- Qty error: 27.7 → 7.5 (excellent!)
- Total loss: 1.53 → 0.35 (good!)

SOLUTION: Continue training with:
1. EXTENDED TRAINING: 10000 steps (was 5000)
2. HIGHER LR: 2e-4 for connector (was 1e-4)
3. LOWER COSINE WEIGHT: 0.40 (was 0.50)
4. Resume from your checkpoint

PHASE 6B ISSUE:
---------------
The textless model has NO text decoder, so the DoRA training from
only-p7-dora.ipynb won't work (it trains text_decoder which doesn't exist).

SOLUTION: Apply DoRA only to speech_encoder + t2u_model, train with unit CE loss.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6A: EXTENDED TRAINING (CELL 1 - Replace your Phase 6a training cell)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hyperparameters (IMPROVED) ────────────────────────────────────────────────
MAX_STEPS_P6A = 10000  # DOUBLED from 5000
BATCH_SIZE    = 8
BATCH_ACCUM   = 2
LOG_EVERY     = 100
SAVE_EVERY    = 500
QTY_NORM      = 20.0

# ── Resume from checkpoint ────────────────────────────────────────────────────
start_6a    = 0
loss_log_6a = []
feat_log_6a = []
qty_log_6a  = []

p6a_ck_resume = load_latest_checkpoint('phase6a_connector')
if p6a_ck_resume and p6a_ck_resume.get('step', 0) > 0:
    start_6a    = p6a_ck_resume['step']
    loss_log_6a = p6a_ck_resume.get('loss_log', [])
    feat_log_6a = p6a_ck_resume.get('feat_log', [])
    qty_log_6a  = p6a_ck_resume.get('qty_log', [])
    print(f'✓ Resuming Phase 6a from step {start_6a}')
    print(f'  Previous best cosine: {min(feat_log_6a[-100:]) if feat_log_6a else "N/A"}')

# ── Optimizer (HIGHER LR) ─────────────────────────────────────────────────────
optimizer_6a = torch.optim.AdamW([
    {'params': model_6a.cif_connector.parameters(),   'lr': 2e-4, 'weight_decay': 0.01},  # INCREASED from 1e-4
    {'params': model_6a.speaker_adapter.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
], betas=(0.9, 0.98))

if p6a_ck_resume and p6a_ck_resume.get('optimizer_state'):
    try:
        optimizer_6a.load_state_dict(p6a_ck_resume['optimizer_state'])
        print('  Optimizer state restored.')
    except Exception as e:
        print(f'  Optimizer restore failed: {e}')

scheduler_6a = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_6a, T_0=1000, T_mult=2, eta_min=1e-5)

scaler_6a = torch.cuda.amp.GradScaler()

print('='*70)
print('  PHASE 6a: CIF Connector + Speaker Adapter Feature KD Training (EXTENDED)')
print(f'  Steps: {start_6a} → {MAX_STEPS_P6A}')
print(f'  Batch size: {BATCH_SIZE}')
print(f'  Loss: 0.40×cosine_KD + 0.20×MSE_KD + 0.30×qty_pred + 0.10×spk_reg')  # ADJUSTED weights
print(f'  Connector LR: 2e-4 (INCREASED), Speaker LR: 1e-4')
print('='*70)
print()

recent_feat, recent_qty_abs, recent_total = [], [], []

for step in range(start_6a, MAX_STEPS_P6A):
    # ── Sample mini-batch ─────────────────────────────────────────────────────
    batch_samples = random.sample(valid_kd, min(BATCH_SIZE, len(valid_kd)))
    
    batch_enc_out = []
    batch_target = []
    batch_target_qty = []
    batch_lang_id = []
    batch_spk_emb = []
    batch_n_tokens = []
    
    for sample in batch_samples:
        tgt_lang = sample['tgt_lang']
        lang_id = m4t_lang_to_vocoder_id(tgt_lang)
        target = sample['t2u_input'].to(device).float()
        n_tokens = float(sample['n_tokens'])
        spk_emb = sample['spk_emb'].to(device).float()
        audio_wav = sample_id_to_audio.get(sample['id'])
        if audio_wav is None:
            continue
        
        try:
            inp_proc = processor(audio=audio_wav, sampling_rate=16000, return_tensors='pt')
            inp_f = inp_proc['input_features'].to(device)
            attn_m = inp_proc.get('attention_mask')
            if attn_m is not None:
                attn_m = attn_m.to(device)
            
            with torch.no_grad():
                enc_out = model_6a.speech_encoder(
                    input_features=inp_f,
                    attention_mask=attn_m
                ).last_hidden_state.float()
            
            batch_enc_out.append(enc_out.squeeze(0))
            batch_target.append(target.squeeze(0))
            batch_target_qty.append(n_tokens)
            batch_lang_id.append(lang_id)
            batch_spk_emb.append(spk_emb)
            batch_n_tokens.append(n_tokens)
            
        except Exception as e:
            print(f'  [!] Sample {sample["id"]} failed: {e}')
            continue
    
    if len(batch_enc_out) == 0:
        continue
    
    # ── Forward pass ──────────────────────────────────────────────────────────
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        batch_cos_loss = []
        batch_mse_loss = []
        batch_qty_loss = []
        batch_spk_reg = []
        batch_fired_counts = []
        batch_qty_errs = []
        
        for i in range(len(batch_enc_out)):
            enc_out = batch_enc_out[i].unsqueeze(0)
            target = batch_target[i].unsqueeze(0)
            lang_id_tensor = torch.tensor([batch_lang_id[i]], device=device)
            target_qty = torch.tensor([batch_target_qty[i]], dtype=torch.float, device=device)
            spk_emb = batch_spk_emb[i]
            
            # CIF connector forward
            cif_out = model_6a.cif_connector(enc_out, tgt_lang_id=lang_id_tensor)
            if len(cif_out) == 4:
                connector_out, actual_qty, qty_pred, alpha_weights = cif_out
            else:
                connector_out, actual_qty, qty_pred = cif_out
            
            # Speaker adapter forward
            spk_proj = model_6a.speaker_adapter(spk_emb.unsqueeze(0))
            
            # ── Loss computation ──────────────────────────────────────────────
            T_pred = connector_out.shape[1]
            T_tgt = target.shape[1]
            T_min = min(T_pred, T_tgt)
            
            if T_min == 0:
                continue
            
            conn_trimmed = connector_out[:, :T_min, :]
            tgt_trimmed = target[:, :T_min, :]
            
            # Per-token cosine similarity
            cos_sim = F.cosine_similarity(
                conn_trimmed.squeeze(0),
                tgt_trimmed.squeeze(0).detach(),
                dim=-1)
            cos_loss = (1.0 - cos_sim).mean()
            
            # MSE loss
            mse_loss = F.mse_loss(conn_trimmed, tgt_trimmed.detach())
            
            # Quantity loss (normalized)
            qty_loss = F.mse_loss(qty_pred / QTY_NORM, target_qty / QTY_NORM)
            
            # Speaker regularization
            spk_reg = ((spk_proj.float().norm(dim=-1) - 14.0) ** 2).mean()
            
            batch_cos_loss.append(cos_loss)
            batch_mse_loss.append(mse_loss)
            batch_qty_loss.append(qty_loss)
            batch_spk_reg.append(spk_reg)
            batch_fired_counts.append(actual_qty.item())
            batch_qty_errs.append(abs(qty_pred.item() - batch_n_tokens[i]))
        
        if len(batch_cos_loss) == 0:
            continue
        
        # Average losses
        cos_loss = torch.stack(batch_cos_loss).mean()
        mse_loss = torch.stack(batch_mse_loss).mean()
        qty_loss = torch.stack(batch_qty_loss).mean()
        spk_reg = torch.stack(batch_spk_reg).mean()
        
        # ADJUSTED LOSS WEIGHTS (lower cosine, higher qty)
        loss = (0.40 * cos_loss +      # REDUCED from 0.50
                0.20 * mse_loss +
                0.30 * qty_loss +      # INCREASED from 0.25
                0.10 * spk_reg)        # INCREASED from 0.05
    
    # Backward pass
    scaler_6a.scale(loss).backward()
    
    # Logging
    avg_fired = np.mean(batch_fired_counts)
    avg_qty_err = np.mean(batch_qty_errs)
    
    loss_log_6a.append(loss.item())
    feat_log_6a.append(cos_loss.item())
    qty_log_6a.append(avg_qty_err)
    
    recent_feat.append(cos_loss.item())
    recent_qty_abs.append(avg_qty_err)
    recent_total.append(loss.item())
    if len(recent_feat) > 100:
        recent_feat.pop(0)
        recent_qty_abs.pop(0)
        recent_total.pop(0)
    
    # Optimizer step
    if (step + 1) % BATCH_ACCUM == 0:
        scaler_6a.unscale_(optimizer_6a)
        torch.nn.utils.clip_grad_norm_(trainable_6a, 1.0)
        scaler_6a.step(optimizer_6a)
        scaler_6a.update()
        optimizer_6a.zero_grad()
        scheduler_6a.step()
    
    # Logging
    if (step + 1) % LOG_EVERY == 0:
        avg_cos = np.mean(recent_feat[-50:]) if recent_feat else 0
        avg_qty = np.mean(recent_qty_abs[-50:]) if recent_qty_abs else 0
        avg_tot = np.mean(recent_total[-50:]) if recent_total else 0
        cur_lr = optimizer_6a.param_groups[0]['lr']
        
        sample_fired = int(batch_fired_counts[0]) if batch_fired_counts else 0
        sample_tgt = int(batch_n_tokens[0]) if batch_n_tokens else 0
        
        print(f'  Step {step+1:5d}/{MAX_STEPS_P6A} | '
              f'cos={avg_cos:.4f} | qty_err(tok)={avg_qty:.1f} | '
              f'total={avg_tot:.4f} | '
              f'fired={sample_fired} vs tgt={sample_tgt} | '
              f'batch={len(batch_cos_loss)} | lr={cur_lr:.2e}')
    
    # Checkpoint
    if (step + 1) % SAVE_EVERY == 0:
        save_checkpoint(
            state={
                'step':            step + 1,
                'cif_state':       model_6a.cif_connector.state_dict(),
                'spk_state':       model_6a.speaker_adapter.state_dict(),
                'optimizer_state': optimizer_6a.state_dict(),
                'scheduler_state': scheduler_6a.state_dict(),
                'loss_log':        loss_log_6a,
                'feat_log':        feat_log_6a,
                'qty_log':         qty_log_6a,
            },
            name='phase6a_connector',
            step=step + 1,
            keep=3,
        )

print('\n✓ Phase 6a extended training complete!')
print(f'  Final cosine loss: {feat_log_6a[-1]:.4f}')
print(f'  Target: < 0.10 ({"CONVERGED" if feat_log_6a[-1] < 0.10 else "needs more training"})')


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6B: DORA E2E FINE-TUNING (CELL 2 - Replace your Phase 6b cells)
# ═══════════════════════════════════════════════════════════════════════════════

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 6b: DoRA E2E Fine-tuning (FIXED FOR TEXTLESS MODEL)                  ║
# ║  - Apply DoRA to speech_encoder + t2u_model ONLY (no text_decoder)          ║
# ║  - Train with unit CE loss (T2U generates units, not text)                   ║
# ║  - Based on working only-p7-dora.ipynb but adapted for textless arch         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print('Loading Phase 6a model for Phase 6b DoRA fine-tuning...')
model_6b = model_6a  # Already in memory with trained CIF + speaker adapter

# Restore 6a final trained weights
p6a_final = load_latest_checkpoint('phase6a_connector')
if p6a_final and p6a_final.get('step', 0) > 0:
    model_6b.cif_connector.load_state_dict(p6a_final['cif_state'])
    model_6b.speaker_adapter.load_state_dict(p6a_final['spk_state'])
    print(f'✓ CIF + speaker adapter weights from step {p6a_final["step"]} restored.')

# Freeze all, unfreeze CIF + speaker adapter
for p in model_6b.parameters():
    p.requires_grad_(False)
for p in model_6b.cif_connector.parameters():
    p.requires_grad_(True)
for p in model_6b.speaker_adapter.parameters():
    p.requires_grad_(True)

# ── Apply DoRA to speech encoder + T2U (NOT text_decoder - it doesn't exist!) ─
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
    use_dora=True,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'])

print('Applying DoRA to speech_encoder...')
model_6b.speech_encoder = get_peft_model(model_6b.speech_encoder, lora_cfg)
model_6b.speech_encoder.print_trainable_parameters()

print('Applying DoRA to t2u_model...')
model_6b.t2u_model = get_peft_model(model_6b.t2u_model, lora_cfg)
model_6b.t2u_model.print_trainable_parameters()

# Multi-GPU layout
if N_GPU >= 2:
    model_6b.speech_encoder  = model_6b.speech_encoder.to('cuda:0')
    model_6b.cif_connector   = model_6b.cif_connector.to('cuda:0')
    model_6b.speaker_adapter = model_6b.speaker_adapter.to('cuda:0')
    model_6b.t2u_model       = model_6b.t2u_model.to('cuda:1')
    if model_6b.vocoder is not None:
        model_6b.vocoder = model_6b.vocoder.to('cuda:1')
    print('Multi-GPU: enc+CIF+spk → cuda:0 | T2U+vocoder → cuda:1')
    DEV_ENC = 'cuda:0'
    DEV_T2U = 'cuda:1'
else:
    model_6b = _consolidate_to_single_gpu(model_6b)
    DEV_ENC = 'cuda:0'
    DEV_T2U = 'cuda:0'

trainable_6b = [p for p in model_6b.parameters() if p.requires_grad]
print(f'Total trainable params: {sum(p.numel() for p in trainable_6b)/1e6:.2f}M')
gpu_mem()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 6b Training Loop (UNIT CE LOSS - for textless model)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MAX_STEPS_E2E = 2500
BATCH_ACCUM   = 4
LOG_EVERY     = 50
SAVE_EVERY    = 250

p6b_ck     = load_latest_checkpoint('phase6b_e2e')
start_6b   = p6b_ck.get('step', 0)   if p6b_ck else 0
loss_log_6b = p6b_ck.get('loss_log', []) if p6b_ck else []

if p6b_ck and start_6b > 0:
    try:
        model_6b.speech_encoder.load_state_dict(p6b_ck['enc_state'], strict=False)
        model_6b.t2u_model.load_state_dict(p6b_ck['t2u_state'], strict=False)
        model_6b.cif_connector.load_state_dict(p6b_ck['cif_state'])
        model_6b.speaker_adapter.load_state_dict(p6b_ck['spk_state'])
        print(f'Resumed Phase 6b from step {start_6b}')
    except Exception as e:
        print(f'Checkpoint restore failed: {e}. Starting from step 0.')
        start_6b = 0

optimizer_6b = torch.optim.AdamW([
    {'params': model_6b.cif_connector.parameters(),
     'lr': 5e-5, 'weight_decay': 0.01},
    {'params': model_6b.speaker_adapter.parameters(),
     'lr': 2e-5, 'weight_decay': 0.01},
    {'params': [p for p in model_6b.speech_encoder.parameters() if p.requires_grad],
     'lr': 2e-5, 'weight_decay': 0.01},
    {'params': [p for p in model_6b.t2u_model.parameters() if p.requires_grad],
     'lr': 2e-5, 'weight_decay': 0.01},
], betas=(0.9, 0.98))

scheduler_6b = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_6b, T_max=MAX_STEPS_E2E, eta_min=1e-6,
    last_epoch=max(0, start_6b - 1))

scaler_6b = torch.cuda.amp.GradScaler()

# Only samples with unit_ids AND audio
unit_kd = [x for x in kd_data
           if x.get('unit_ids') is not None
           and x.get('t2u_input') is not None
           and sample_id_to_audio.get(x['id']) is not None]
print(f'Phase 6b training samples (unit labels + audio): {len(unit_kd)}')
assert len(unit_kd) > 10, 'Not enough unit_kd samples'

model_6b.train()
optimizer_6b.zero_grad()

print(f'\n{"="*70}')
print(f'  PHASE 6b: End-to-End DoRA Fine-tuning (TEXTLESS MODEL)')
print(f'  Steps: {start_6b} → {MAX_STEPS_E2E}')
print(f'  Loss: 0.80×unit_CE + 0.15×qty_pred + 0.05×spk_reg')
print(f'  KEY: Real speech encoder forward every step')
print(f'{"="*70}\n')

recent_unit = []
recent_qty  = []

for step in range(start_6b, MAX_STEPS_E2E):
    sample   = random.choice(unit_kd)
    tgt_lang = sample['tgt_lang']
    lang_id  = torch.tensor([m4t_lang_to_vocoder_id(tgt_lang)], device=DEV_ENC)
    unit_ids = sample['unit_ids'].unsqueeze(0).to(DEV_T2U)
    spk_emb  = sample['spk_emb'].to(DEV_ENC).float()
    n_toks   = float(sample['n_tokens'])
    target_qty = torch.tensor([n_toks], dtype=torch.float, device=DEV_ENC)

    # Load actual audio and run real speech encoder
    audio_wav = sample_id_to_audio.get(sample['id'])
    if audio_wav is None:
        continue

    try:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Real speech encoder forward
            inp_proc = processor(audio=audio_wav, sampling_rate=16000,
                                 return_tensors='pt')
            inp_f  = inp_proc['input_features'].to(DEV_ENC)
            attn_m = inp_proc.get('attention_mask')
            if attn_m is not None: attn_m = attn_m.to(DEV_ENC)

            enc_out = model_6b.speech_encoder(
                input_features=inp_f,
                attention_mask=attn_m).last_hidden_state.float()

            # CIF connector
            connector_out, actual_qty, qty_pred = model_6b.cif_connector(
                enc_out, lang_id)

            spk_proj = model_6b.speaker_adapter(spk_emb.unsqueeze(0))

            # Move connector output to T2U device
            connector_t2u = connector_out.to(DEV_T2U)

            # T2U unit CE loss
            try:
                t2u_out   = model_6b.t2u_model(
                    inputs_embeds=connector_t2u,
                    labels=unit_ids)
                unit_loss = t2u_out.loss.to(DEV_ENC)
            except Exception as e_t2u:
                print(f'  T2U forward error at step {step+1}: {e_t2u}')
                unit_loss = torch.tensor(0.0, device=DEV_ENC, requires_grad=True)

            # Quantity prediction loss
            qty_loss = F.mse_loss(qty_pred, target_qty)

            # Speaker regularization
            spk_reg = ((spk_proj.float().norm(dim=-1) - 14.0) ** 2).mean()

            loss = (0.80 * unit_loss +
                    0.15 * qty_loss +
                    0.05 * spk_reg)

        scaler_6b.scale(loss / BATCH_ACCUM).backward()
        loss_log_6b.append(loss.item())

        recent_unit.append(unit_loss.item())
        recent_qty.append(qty_loss.item())
        if len(recent_unit) > 100:
            recent_unit.pop(0); recent_qty.pop(0)

        if (step + 1) % BATCH_ACCUM == 0:
            scaler_6b.unscale_(optimizer_6b)
            torch.nn.utils.clip_grad_norm_(trainable_6b, 1.0)
            scaler_6b.step(optimizer_6b)
            scaler_6b.update()
            optimizer_6b.zero_grad()
            scheduler_6b.step()

        if (step + 1) % LOG_EVERY == 0:
            avg_unit = np.mean(recent_unit[-50:]) if recent_unit else 0
            avg_qty  = np.mean(recent_qty[-50:])  if recent_qty  else 0
            cur_lr   = optimizer_6b.param_groups[0]['lr']
            T_fired  = connector_out.shape[1]
            print(f'  Step {step+1:>5}/{MAX_STEPS_E2E} | '
                  f'unit_CE={avg_unit:.4f} | '
                  f'qty_err={avg_qty:.2f} | '
                  f'fired={T_fired} vs tgt={int(n_toks)} | '
                  f'lr={cur_lr:.2e}')

        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint({
                'step':      step + 1,
                'enc_state': model_6b.speech_encoder.state_dict(),
                't2u_state': model_6b.t2u_model.state_dict(),
                'cif_state': model_6b.cif_connector.state_dict(),
                'spk_state': model_6b.speaker_adapter.state_dict(),
                'loss_log':  loss_log_6b,
            }, 'phase6b_e2e', step + 1)
            print(f'  ✓ Checkpoint saved at step {step+1}')

    except Exception as e:
        print(f'  Step {step+1} error: {e}')
        if step - start_6b < 5:
            import traceback; traceback.print_exc()
        optimizer_6b.zero_grad()
        torch.cuda.empty_cache()
        continue

print('Phase 6b training complete.')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 6b: Merge DoRA + Save Final Model                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print('Merging DoRA adapters into base weights...')
model_6b.speech_encoder = model_6b.speech_encoder.merge_and_unload()
model_6b.t2u_model      = model_6b.t2u_model.merge_and_unload()
model_6b.eval()
model_6b = _consolidate_to_single_gpu(model_6b)
sync_model_config(model_6b)
gc.collect(); torch.cuda.empty_cache()
print_model_breakdown(model_6b, 'Phase 6b FINAL: ~673M Textless Model')

# Save final model
save_model_to_drive(model_6b, processor, 'phase6b_e2e_merged',
                    manifest_extra={
                        'hidden': hidden,
                        'n_langs': n_langs,
                        'cif_params': count_params(model_6b.cif_connector),
                        'spk_params': count_params(model_6b.speaker_adapter),
                    })

print('\n✓ Final ~673M textless model saved to Drive.')


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
HOW TO USE THESE FIXES:
-----------------------

1. PHASE 6A (Extended Training):
   - Replace your Phase 6a training cell with the code from "PHASE 6A: EXTENDED TRAINING"
   - This will resume from your checkpoint at step 5000 and continue to 10000
   - With higher LR and adjusted loss weights, cosine should converge to < 0.10

2. PHASE 6B (Fixed DoRA):
   - Replace ALL your Phase 6b cells with the code from "PHASE 6B: DORA E2E FINE-TUNING"
   - This applies DoRA only to speech_encoder + t2u_model (not text_decoder)
   - Trains with unit CE loss (correct for textless model)

3. Run the cells in order:
   - Phase 6a extended training (10000 steps)
   - Phase 6b DoRA training (2500 steps)
   - Phase 6b merge and save

EXPECTED RESULTS:
-----------------
- Phase 6a: Cosine loss should drop below 0.10 by step 10000
- Phase 6b: Unit CE loss should decrease steadily, model quality improves
- Final model: ~673M params, ready for Phase 7 benchmark

If Phase 6a still doesn't converge after 10000 steps:
- Increase MAX_STEPS_P6A to 15000
- Increase connector LR to 3e-4
- Lower cosine weight to 0.30
"""
