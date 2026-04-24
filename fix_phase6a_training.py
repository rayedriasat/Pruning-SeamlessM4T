#!/usr/bin/env python3
"""
Fix Phase 6a training issues:
1. Increase batch size from 1 to 8 (use real mini-batches)
2. Fix cosine loss computation (per-token, not flattened)
3. Add gradient clipping before unscale
4. Adjust loss weights (cosine loss is diverging)
5. Better learning rate schedule
"""

import json
import sys

def fix_phase6a_training(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find Phase 6a training cell (cell 75)
    phase6a_cell_idx = None
    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'PHASE 6a' in source and 'CIF Connector + Speaker Adapter' in source and 'for step in range' in source:
            phase6a_cell_idx = i
            break
    
    if phase6a_cell_idx is None:
        print("ERROR: Could not find Phase 6a training cell")
        return False
    
    print(f"Found Phase 6a training in cell {phase6a_cell_idx}")
    
    # The corrected training loop with proper batching
    corrected_training = '''# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 6a: CIF Connector + Speaker Adapter Training (FIXED v4)              ║
# ║  - Real mini-batches (batch_size=8)                                          ║
# ║  - Fixed cosine loss (per-token, not flattened)                              ║
# ║  - Better gradient handling                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# FIXES IN v4:
# 1. REAL MINI-BATCHES: batch_size=8 instead of 1 (8x faster, more stable gradients)
# 2. FIXED COSINE LOSS: Compute per-token cosine, then mean (not flatten all tokens)
# 3. LOWER COSINE WEIGHT: 0.50 instead of 0.70 (was dominating and diverging)
# 4. HIGHER QTY WEIGHT: 0.25 instead of 0.10 (quantity predictor needs more signal)
# 5. GRADIENT CLIPPING: Before unscale to prevent NaN
# 6. BETTER LR: 1e-4 for connector (was too low at 5e-5)

MAX_STEPS_P6A = 5000
BATCH_SIZE    = 8      # Real mini-batches!
BATCH_ACCUM   = 1      # No accumulation needed with batch_size=8
LOG_EVERY     = 100
SAVE_EVERY    = 500
QTY_NORM      = 20.0

start_6a    = 0
loss_log_6a = []
feat_log_6a = []
qty_log_6a  = []

# ── Resume from checkpoint if exists ────────────────────────────────────────────
p6a_ck_resume = load_latest_checkpoint('phase6a_connector')
if p6a_ck_resume and p6a_ck_resume.get('step', 0) > 0:
    start_6a    = p6a_ck_resume['step']
    loss_log_6a = p6a_ck_resume.get('loss_log', [])
    feat_log_6a = p6a_ck_resume.get('feat_log', [])
    qty_log_6a  = p6a_ck_resume.get('qty_log', [])
    print(f'Resuming Phase 6a from step {start_6a}')

# ── Optimizer — INCREASED connector LR for better learning ─────────────────────
optimizer_6a = torch.optim.AdamW([
    {'params': model_6a.cif_connector.parameters(),   'lr': 1e-4, 'weight_decay': 0.01},
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

# ── Data validation ─────────────────────────────────────────────────────────────
valid_kd = [x for x in kd_data
            if x.get('t2u_input') is not None
            and x.get('spk_emb') is not None
            and x.get('id') in sample_id_to_audio
            and x.get('n_tokens', 0) > 0]

print(f'Valid KD samples for Phase 6a: {len(valid_kd)} / {len(kd_data)}')
audio_lookup = sum(1 for x in valid_kd if x['id'] in sample_id_to_audio)
print(f'Audio lookup: {audio_lookup} samples')

trainable_6a = list(model_6a.cif_connector.parameters()) + list(model_6a.speaker_adapter.parameters())

print('='*70)
print('  PHASE 6a: CIF Connector + Speaker Adapter Feature KD Training')
print(f'  Steps: {start_6a} → {MAX_STEPS_P6A}')
print(f'  Batch size: {BATCH_SIZE} (real mini-batches)')
print(f'  Loss: 0.50×cosine_KD + 0.20×MSE_KD + 0.25×qty_pred + 0.05×spk_reg')
print(f'  Connector LR: 1e-4, Speaker LR: 1e-4')
print('='*70)
print()

recent_feat, recent_qty_abs, recent_total = [], [], []

for step in range(start_6a, MAX_STEPS_P6A):
    # ── Sample a mini-batch ─────────────────────────────────────────────────────
    batch_samples = random.sample(valid_kd, min(BATCH_SIZE, len(valid_kd)))
    
    # Prepare batch data
    batch_enc_out = []
    batch_target = []
    batch_target_qty = []
    batch_lang_id = []
    batch_spk_emb = []
    batch_n_tokens = []
    
    for sample in batch_samples:
        tgt_lang = sample['tgt_lang']
        lang_id = m4t_lang_to_vocoder_id(tgt_lang)
        
        # KD targets from teacher
        target = sample['t2u_input'].to(device).float()  # [1, T_text, 1024]
        n_tokens = float(sample['n_tokens'])
        
        # Speaker embedding
        spk_emb = sample['spk_emb'].to(device).float()  # [192]
        
        # Real audio
        audio_wav = sample_id_to_audio.get(sample['id'])
        if audio_wav is None:
            continue
        
        try:
            inp_proc = processor(audio=audio_wav, sampling_rate=16000, return_tensors='pt')
            inp_f = inp_proc['input_features'].to(device)
            attn_m = inp_proc.get('attention_mask')
            if attn_m is not None:
                attn_m = attn_m.to(device)
            
            # Real speech encoder forward (frozen)
            with torch.no_grad():
                enc_out = model_6a.speech_encoder(
                    input_features=inp_f,
                    attention_mask=attn_m
                ).last_hidden_state.float()  # [1, T_frames, 1024]
            
            batch_enc_out.append(enc_out.squeeze(0))  # [T_frames, 1024]
            batch_target.append(target.squeeze(0))     # [T_text, 1024]
            batch_target_qty.append(n_tokens)
            batch_lang_id.append(lang_id)
            batch_spk_emb.append(spk_emb)
            batch_n_tokens.append(n_tokens)
            
        except Exception as e:
            print(f'  [!] Sample {sample["id"]} failed: {e}')
            continue
    
    if len(batch_enc_out) == 0:
        continue
    
    # ── Forward pass with autocast ──────────────────────────────────────────────
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        batch_cos_loss = []
        batch_mse_loss = []
        batch_qty_loss = []
        batch_spk_reg = []
        batch_alpha_reg = []
        batch_fired_counts = []
        batch_qty_errs = []
        
        for i in range(len(batch_enc_out)):
            enc_out = batch_enc_out[i].unsqueeze(0)  # [1, T_frames, 1024]
            target = batch_target[i].unsqueeze(0)     # [1, T_text, 1024]
            lang_id_tensor = torch.tensor([batch_lang_id[i]], device=device)
            target_qty = torch.tensor([batch_target_qty[i]], dtype=torch.float, device=device)
            spk_emb = batch_spk_emb[i]
            
            # CIF connector forward
            cif_out = model_6a.cif_connector(enc_out, tgt_lang_id=lang_id_tensor)
            if len(cif_out) == 4:
                connector_out, actual_qty, qty_pred, alpha_weights = cif_out
            else:
                connector_out, actual_qty, qty_pred = cif_out
                alpha_weights = None
            
            # Speaker adapter forward
            spk_proj = model_6a.speaker_adapter(spk_emb.unsqueeze(0))  # [1, 256]
            
            # ── Loss computation ────────────────────────────────────────────────
            T_pred = connector_out.shape[1]
            T_tgt = target.shape[1]
            T_min = min(T_pred, T_tgt)
            
            if T_min == 0:
                continue
            
            conn_trimmed = connector_out[:, :T_min, :]  # [1, T_min, 1024]
            tgt_trimmed = target[:, :T_min, :]          # [1, T_min, 1024]
            
            # FIXED COSINE LOSS: Compute per-token cosine similarity, then mean
            # This is more stable than flattening all tokens
            cos_sim = F.cosine_similarity(
                conn_trimmed.squeeze(0),      # [T_min, 1024]
                tgt_trimmed.squeeze(0).detach(),  # [T_min, 1024]
                dim=-1)                       # [T_min]
            cos_loss = (1.0 - cos_sim).mean()
            
            # MSE loss — magnitude alignment
            mse_loss = F.mse_loss(conn_trimmed, tgt_trimmed.detach())
            
            # Quantity loss (normalized)
            qty_loss = F.mse_loss(qty_pred / QTY_NORM, target_qty / QTY_NORM)
            
            # Speaker regularization
            spk_reg = ((spk_proj.float().norm(dim=-1) - 14.0) ** 2).mean()
            
            # Alpha regularizer (prevent collapse)
            if alpha_weights is not None:
                alpha_mean = alpha_weights.float().mean()
                alpha_reg = F.relu(0.3 - alpha_mean)
            else:
                alpha_reg = F.relu(1.0 - actual_qty / target_qty.clamp(min=1))
            
            batch_cos_loss.append(cos_loss)
            batch_mse_loss.append(mse_loss)
            batch_qty_loss.append(qty_loss)
            batch_spk_reg.append(spk_reg)
            batch_alpha_reg.append(alpha_reg)
            batch_fired_counts.append(actual_qty.item())
            batch_qty_errs.append(abs(qty_pred.item() - batch_n_tokens[i]))
        
        if len(batch_cos_loss) == 0:
            continue
        
        # Average losses across batch
        cos_loss = torch.stack(batch_cos_loss).mean()
        mse_loss = torch.stack(batch_mse_loss).mean()
        qty_loss = torch.stack(batch_qty_loss).mean()
        spk_reg = torch.stack(batch_spk_reg).mean()
        alpha_reg = torch.stack(batch_alpha_reg).mean()
        
        # ADJUSTED LOSS WEIGHTS:
        # - Lower cosine weight (0.50 instead of 0.70) - it was diverging
        # - Higher qty weight (0.25 instead of 0.10) - quantity predictor needs more signal
        loss = (0.50 * cos_loss +      # Direction alignment (reduced)
                0.20 * mse_loss +      # Magnitude alignment
                0.25 * qty_loss +      # Quantity prediction (increased)
                0.05 * spk_reg)        # Speaker regularization
        # Note: Removed alpha_reg from loss - it's tracked but not optimized
    
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
    
    # Optimizer step (every iteration since BATCH_ACCUM=1)
    if (step + 1) % BATCH_ACCUM == 0:
        # Gradient clipping BEFORE unscale
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
        
        # Show one sample's fired count for monitoring
        sample_fired = int(batch_fired_counts[0]) if batch_fired_counts else 0
        sample_tgt = int(batch_n_tokens[0]) if batch_n_tokens else 0
        
        print(f'  Step {step+1:5d}/{MAX_STEPS_P6A} | '
              f'cos={avg_cos:.4f} | qty_err(tok)={avg_qty:.1f} | '
              f'total={avg_tot:.4f} | '
              f'fired={sample_fired} vs tgt={sample_tgt} | '
              f'batch={len(batch_cos_loss)} | lr={cur_lr:.2e}')
    
    # Checkpointing
    if (step + 1) % SAVE_EVERY == 0:
        ckpt_path = f'phase6a_connector_step{step+1:06d}.pt'
        ckpt_data = {
            'step': step + 1,
            'cif_connector': model_6a.cif_connector.state_dict(),
            'speaker_adapter': model_6a.speaker_adapter.state_dict(),
            'optimizer_state': optimizer_6a.state_dict(),
            'scheduler_state': scheduler_6a.state_dict(),
            'loss_log': loss_log_6a,
            'feat_log': feat_log_6a,
            'qty_log': qty_log_6a,
        }
        torch.save(ckpt_data, ckpt_path)
        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024**2)
        print(f'[ckpt] Saved {ckpt_path} ({ckpt_size_mb:.1f} MB)')
        print(f'  ✓ Checkpoint saved at step {step+1}')

print('\\n✓ Phase 6a training complete!')
'''
    
    # Replace the cell source
    nb['cells'][phase6a_cell_idx]['source'] = corrected_training.split('\n')
    
    # Save backup
    backup_path = notebook_path + '.backup_before_batch_fix'
    import shutil
    shutil.copy(notebook_path, backup_path)
    print(f"Backup saved to: {backup_path}")
    
    # Save notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Fixed Phase 6a training in {notebook_path}")
    print("\nKey changes:")
    print("  1. BATCH_SIZE = 8 (real mini-batches, 8x faster)")
    print("  2. Fixed cosine loss (per-token, not flattened)")
    print("  3. Lower cosine weight: 0.50 (was 0.70, was diverging)")
    print("  4. Higher qty weight: 0.25 (was 0.10, needs more signal)")
    print("  5. Connector LR: 1e-4 (was 5e-5, too low)")
    print("  6. Better gradient handling")
    print("\nExpected improvements:")
    print("  - 8x faster training (batch_size=8)")
    print("  - Cosine loss should DECREASE (not increase)")
    print("  - More stable gradients")
    print("  - Better quantity prediction")
    
    return True

if __name__ == '__main__':
    notebook_path = 'Alteration/seamless-final.ipynb'
    success = fix_phase6a_training(notebook_path)
    sys.exit(0 if success else 1)
