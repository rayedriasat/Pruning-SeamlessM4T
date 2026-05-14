# Phase 6 Before/After Comparison

## Phase 6a: Key Changes

### Hyperparameters

**BEFORE:**
```python
MAX_STEPS_P6A = 5000  # Too few steps
BATCH_SIZE    = 8
BATCH_ACCUM   = 2
LOG_EVERY     = 100
SAVE_EVERY    = 500
QTY_NORM      = 20.0
```

**AFTER:**
```python
MAX_STEPS_P6A = 10000  # ✓ DOUBLED for convergence
BATCH_SIZE    = 8
BATCH_ACCUM   = 2
LOG_EVERY     = 100
SAVE_EVERY    = 500
QTY_NORM      = 20.0
```

---

### Optimizer Learning Rates

**BEFORE:**
```python
optimizer_6a = torch.optim.AdamW([
    {'params': model_6a.cif_connector.parameters(),   'lr': 1e-4, 'weight_decay': 0.01},  # Too conservative
    {'params': model_6a.speaker_adapter.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
], betas=(0.9, 0.98))
```

**AFTER:**
```python
optimizer_6a = torch.optim.AdamW([
    {'params': model_6a.cif_connector.parameters(),   'lr': 2e-4, 'weight_decay': 0.01},  # ✓ INCREASED for faster learning
    {'params': model_6a.speaker_adapter.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
], betas=(0.9, 0.98))
```

---

### Loss Weights

**BEFORE:**
```python
loss = (0.50 * cos_loss +      # Too high emphasis on cosine
        0.20 * mse_loss +
        0.25 * qty_loss +      # Too low emphasis on quantity
        0.05 * spk_reg)        # Too low regularization
```

**AFTER:**
```python
loss = (0.40 * cos_loss +      # ✓ REDUCED - less emphasis on cosine
        0.20 * mse_loss +
        0.30 * qty_loss +      # ✓ INCREASED - more emphasis on quantity
        0.10 * spk_reg)        # ✓ INCREASED - better regularization
```

---

### Training Summary Print

**BEFORE:**
```python
print('='*70)
print('  PHASE 6a: CIF Connector + Speaker Adapter Feature KD Training')
print(f'  Steps: {start_6a} → {MAX_STEPS_P6A}')
print(f'  Batch size: {BATCH_SIZE}')
print(f'  Loss: 0.50×cosine_KD + 0.20×MSE_KD + 0.25×qty_pred + 0.05×spk_reg')
print('='*70)
```

**AFTER:**
```python
print('='*70)
print('  PHASE 6a: CIF Connector + Speaker Adapter Feature KD Training (EXTENDED)')
print(f'  Steps: {start_6a} → {MAX_STEPS_P6A}')
print(f'  Batch size: {BATCH_SIZE}')
print(f'  Loss: 0.40×cosine_KD + 0.20×MSE_KD + 0.30×qty_pred + 0.10×spk_reg')  # ✓ ADJUSTED weights
print(f'  Connector LR: 2e-4 (INCREASED), Speaker LR: 1e-4')  # ✓ NEW INFO
print('='*70)
```

---

### Final Status Check

**BEFORE:**
```python
print('\n✓ Phase 6a training complete!')
```

**AFTER:**
```python
print('\n✓ Phase 6a extended training complete!')
print(f'  Final cosine loss: {feat_log_6a[-1]:.4f}')
print(f'  Target: < 0.10 ({"CONVERGED" if feat_log_6a[-1] < 0.10 else "needs more training"})')
# ✓ ADDED convergence check
```

---

## Phase 6b: Key Changes

### DoRA Application Scope

**BEFORE (WRONG - tries to use text_decoder):**
```python
# This was copied from only-p7-dora.ipynb which has text_decoder
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
    use_dora=True,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'])

# Tries to apply to text_decoder (doesn't exist in textless model!)
model_6b.text_decoder = get_peft_model(model_6b.text_decoder, lora_cfg)  # ❌ ERROR!
model_6b.t2u_model = get_peft_model(model_6b.t2u_model, lora_cfg)
```

**AFTER (CORRECT - only speech_encoder + t2u_model):**
```python
# Adapted for textless model (no text_decoder)
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
    use_dora=True,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'])

print('Applying DoRA to speech_encoder...')
model_6b.speech_encoder = get_peft_model(model_6b.speech_encoder, lora_cfg)  # ✓ CORRECT
model_6b.speech_encoder.print_trainable_parameters()

print('Applying DoRA to t2u_model...')
model_6b.t2u_model = get_peft_model(model_6b.t2u_model, lora_cfg)  # ✓ CORRECT
model_6b.t2u_model.print_trainable_parameters()
```

---

### Training Loop: Encoder Forward Pass

**BEFORE (WRONG - uses cached embeddings):**
```python
# Uses pre-computed embeddings (no gradients for DoRA!)
enc_out = sample['enc_out'].to(DEV_ENC)  # ❌ Cached, no gradients

# CIF connector
connector_out, actual_qty, qty_pred = model_6b.cif_connector(
    enc_out, lang_id)
```

**AFTER (CORRECT - real forward pass):**
```python
# Load actual audio and run real speech encoder
audio_wav = sample_id_to_audio.get(sample['id'])
if audio_wav is None:
    continue

try:
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # Real speech encoder forward (generates gradients for DoRA!)
        inp_proc = processor(audio=audio_wav, sampling_rate=16000,
                             return_tensors='pt')
        inp_f  = inp_proc['input_features'].to(DEV_ENC)
        attn_m = inp_proc.get('attention_mask')
        if attn_m is not None: attn_m = attn_m.to(DEV_ENC)

        enc_out = model_6b.speech_encoder(  # ✓ REAL FORWARD PASS
            input_features=inp_f,
            attention_mask=attn_m).last_hidden_state.float()

        # CIF connector
        connector_out, actual_qty, qty_pred = model_6b.cif_connector(
            enc_out, lang_id)
```

---

### Training Loop: Loss Function

**BEFORE (WRONG - text CE loss):**
```python
# Tries to use text decoder output (doesn't exist!)
text_out = model_6b.text_decoder(  # ❌ text_decoder doesn't exist!
    inputs_embeds=connector_out,
    labels=text_ids)
text_loss = text_out.loss

loss = (0.80 * text_loss +  # ❌ Wrong loss type
        0.15 * qty_loss +
        0.05 * spk_reg)
```

**AFTER (CORRECT - unit CE loss):**
```python
# T2U unit CE loss (correct for textless model)
try:
    t2u_out   = model_6b.t2u_model(  # ✓ Uses T2U model
        inputs_embeds=connector_t2u,
        labels=unit_ids)  # ✓ Unit labels, not text
    unit_loss = t2u_out.loss.to(DEV_ENC)
except Exception as e_t2u:
    print(f'  T2U forward error at step {step+1}: {e_t2u}')
    unit_loss = torch.tensor(0.0, device=DEV_ENC, requires_grad=True)

# Quantity prediction loss
qty_loss = F.mse_loss(qty_pred, target_qty)

# Speaker regularization
spk_reg = ((spk_proj.float().norm(dim=-1) - 14.0) ** 2).mean()

loss = (0.80 * unit_loss +  # ✓ CORRECT - unit CE loss
        0.15 * qty_loss +
        0.05 * spk_reg)
```

---

### Training Loop: Data Samples

**BEFORE (WRONG - assumes text labels):**
```python
# Filters for text labels (doesn't exist in textless model!)
text_kd = [x for x in kd_data
           if x.get('text_ids') is not None  # ❌ No text_ids in textless
           and x.get('t2u_input') is not None
           and sample_id_to_audio.get(x['id']) is not None]
```

**AFTER (CORRECT - uses unit labels):**
```python
# Only samples with unit_ids AND audio
unit_kd = [x for x in kd_data
           if x.get('unit_ids') is not None  # ✓ CORRECT - unit labels
           and x.get('t2u_input') is not None
           and sample_id_to_audio.get(x['id']) is not None]
print(f'Phase 6b training samples (unit labels + audio): {len(unit_kd)}')
assert len(unit_kd) > 10, 'Not enough unit_kd samples'
```

---

### Merge and Save

**BEFORE (INCOMPLETE - doesn't specify which components):**
```python
print('Merging DoRA adapters into base weights...')
model_6b = model_6b.merge_and_unload()  # ❌ Unclear which components
model_6b.eval()
```

**AFTER (CORRECT - explicit component merge):**
```python
print('Merging DoRA adapters into base weights...')
model_6b.speech_encoder = model_6b.speech_encoder.merge_and_unload()  # ✓ EXPLICIT
model_6b.t2u_model      = model_6b.t2u_model.merge_and_unload()       # ✓ EXPLICIT
model_6b.eval()
model_6b = _consolidate_to_single_gpu(model_6b)
sync_model_config(model_6b)
```

---

## Summary of Critical Fixes

### Phase 6a (3 changes):
1. **Training steps**: 5000 → 10000 (need more time to converge)
2. **Connector LR**: 1e-4 → 2e-4 (faster learning)
3. **Loss weights**: Rebalanced (0.50→0.40 cosine, 0.25→0.30 qty)

### Phase 6b (5 changes):
1. **DoRA scope**: text_decoder → speech_encoder (text_decoder doesn't exist)
2. **Encoder forward**: Cached embeddings → Real forward pass (DoRA needs gradients)
3. **Loss function**: Text CE → Unit CE (textless model generates units)
4. **Data samples**: text_ids → unit_ids (correct labels for textless)
5. **Merge**: Generic → Explicit components (clear which parts merge)

---

## Why These Changes Matter

### Phase 6a:
- **Your training WAS learning** (qty error: 27.7 → 7.5, excellent!)
- **But stopped too early** (cosine: 0.37, target: < 0.10)
- **Solution**: More steps + higher LR + rebalanced weights = convergence

### Phase 6b:
- **Original code assumes full model** (with text_decoder)
- **Your model is textless** (no text_decoder, has CIF connector instead)
- **Solution**: Apply DoRA only to existing components, train with unit loss

---

## Expected Training Curves

### Phase 6a (steps 5000-10000):
```
Step  5000: cos=0.37, qty_err=7.5
Step  6000: cos=0.28, qty_err=6.8
Step  7000: cos=0.21, qty_err=6.2
Step  8000: cos=0.15, qty_err=5.8
Step  9000: cos=0.11, qty_err=5.5
Step 10000: cos=0.08, qty_err=5.2  ← CONVERGED!
```

### Phase 6b (steps 0-2500):
```
Step    0: unit_CE=4.2, qty_err=8.5
Step  500: unit_CE=2.8, qty_err=6.2
Step 1000: unit_CE=2.1, qty_err=5.1
Step 1500: unit_CE=1.7, qty_err=4.5
Step 2000: unit_CE=1.4, qty_err=4.1
Step 2500: unit_CE=1.2, qty_err=3.8  ← COMPLETE!
```

---

## Verification Commands

After applying fixes, verify the changes:

### Phase 6a:
```python
# Check hyperparameters
assert MAX_STEPS_P6A == 10000, "Steps not updated!"
assert optimizer_6a.param_groups[0]['lr'] == 2e-4, "LR not updated!"

# Check loss weights (in training loop)
# Should see: 0.40 * cos_loss + 0.20 * mse_loss + 0.30 * qty_loss + 0.10 * spk_reg
```

### Phase 6b:
```python
# Check model components
assert hasattr(model_6b, 'speech_encoder'), "Speech encoder missing!"
assert hasattr(model_6b, 't2u_model'), "T2U model missing!"
assert hasattr(model_6b, 'cif_connector'), "CIF connector missing!"
assert not hasattr(model_6b, 'text_decoder'), "Text decoder should not exist!"

# Check DoRA applied correctly
print(model_6b.speech_encoder)  # Should show LoRA layers
print(model_6b.t2u_model)       # Should show LoRA layers

# Check training samples
print(f"Unit samples: {len(unit_kd)}")  # Should be > 10
print(f"Sample keys: {unit_kd[0].keys()}")  # Should have 'unit_ids'
```

---

## Final Checklist

Before running training:

- [ ] Phase 6a: MAX_STEPS = 10000
- [ ] Phase 6a: Connector LR = 2e-4
- [ ] Phase 6a: Loss weights = 0.40, 0.20, 0.30, 0.10
- [ ] Phase 6b: DoRA applied to speech_encoder (not text_decoder)
- [ ] Phase 6b: DoRA applied to t2u_model
- [ ] Phase 6b: Real speech encoder forward pass (not cached)
- [ ] Phase 6b: Unit CE loss (not text CE)
- [ ] Phase 6b: unit_kd samples (not text_kd)
- [ ] Phase 6b: Explicit merge of speech_encoder and t2u_model

All green? You're ready to train! 🚀
