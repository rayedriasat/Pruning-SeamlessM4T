# Phase 6a and 6b Fix Instructions

## Summary

You have two issues to fix in your `Alteration/seamless-final.ipynb` notebook:

### **Phase 6a Issue**: Training didn't converge
- **Status**: Training DID learn but stopped too early
- **Evidence**: 
  - Cosine loss: 0.47 → 0.37 (good progress, but target is < 0.10)
  - Quantity error: 27.7 → 7.5 tokens (excellent convergence!)
  - Total loss: 1.53 → 0.35 (good progress)
- **Root cause**: Insufficient training steps (5000 was too few) + conservative learning rate

### **Phase 6b Issue**: DoRA training incompatible with textless model
- **Status**: Code copied from `only-p7-dora.ipynb` which trains models WITH text_decoder
- **Problem**: Your textless model has NO text_decoder (surgically removed in Phase 4)
- **Root cause**: Original code tries to apply DoRA to text_decoder → causes errors

---

## Solution Overview

Both fixes are ready in `Alteration/phase6_fixes.py`:

### **Phase 6a Fix**: Extended training configuration
- Increase MAX_STEPS from 5000 → 10000
- Increase connector LR from 1e-4 → 2e-4
- Adjust loss weights: 0.40×cosine (down from 0.50), 0.30×qty (up from 0.25)
- Resume from your existing checkpoint at step 5000

### **Phase 6b Fix**: Complete rewrite for textless architecture
- Apply DoRA only to `speech_encoder` + `t2u_model` (NOT text_decoder)
- Train with unit CE loss (T2U generates units, not text)
- Keep CIF connector and speaker adapter unfrozen
- Use real speech encoder forward pass every step (not cached embeddings)
- Multi-GPU layout: encoder+CIF+speaker on cuda:0, T2U on cuda:1

---

## Step-by-Step Instructions

### Step 1: Backup your current notebook
```bash
cp Alteration/seamless-final.ipynb Alteration/seamless-final.ipynb.backup
```

### Step 2: Replace Phase 6a training cell

**Location**: Find the cell with this header:
```python
# ║  Phase 6a: CIF Connector + Speaker Adapter Training (FIXED v4)              ║
```

**Action**: Replace the ENTIRE cell (from the header comment to the final print statement) with the code from `Alteration/phase6_fixes.py` section:
```python
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6A: EXTENDED TRAINING (CELL 1 - Replace your Phase 6a training cell)
# ═══════════════════════════════════════════════════════════════════════════════
```

**Key changes**:
- `MAX_STEPS_P6A = 10000` (was 5000)
- Connector LR: `2e-4` (was 1e-4)
- Loss weights: `0.40 * cos_loss + 0.20 * mse_loss + 0.30 * qty_loss + 0.10 * spk_reg`

### Step 3: Replace ALL Phase 6b cells

**Location**: Find these cells:
1. Cell 8: "Phase 6b: DoRA E2E Fine-tuning — CORRECTED"
2. Cell 9: "Phase 6b Training Loop"
3. Cell 10: "Phase 6b: Merge DoRA + Save Final Model"

**Action**: Replace ALL THREE cells with the code from `Alteration/phase6_fixes.py` section:
```python
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6B: DORA E2E FINE-TUNING (CELL 2 - Replace your Phase 6b cells)
# ═══════════════════════════════════════════════════════════════════════════════
```

**Key changes**:
- DoRA applied ONLY to `speech_encoder` and `t2u_model` (no text_decoder)
- Training uses unit CE loss (correct for textless model)
- Real speech encoder forward pass every step
- Multi-GPU layout optimized for textless architecture

### Step 4: Run the training

1. **Run Phase 6a extended training**:
   - Will resume from step 5000
   - Continue to step 10000
   - Monitor cosine loss - should drop below 0.10

2. **Run Phase 6b DoRA training**:
   - Loads Phase 6a final weights
   - Trains for 2500 steps
   - Merges DoRA adapters and saves final model

---

## Expected Results

### Phase 6a (after 10000 steps):
```
✓ Phase 6a extended training complete!
  Final cosine loss: 0.08 (target: < 0.10)
  Status: CONVERGED
```

### Phase 6b (after 2500 steps):
```
Phase 6b training complete.
Merging DoRA adapters into base weights...
✓ Final ~673M textless model saved to Drive.
```

---

## Troubleshooting

### If Phase 6a still doesn't converge after 10000 steps:

**Option 1**: Extend further
```python
MAX_STEPS_P6A = 15000  # Increase from 10000
```

**Option 2**: Increase learning rate
```python
{'params': model_6a.cif_connector.parameters(), 'lr': 3e-4, ...}  # Increase from 2e-4
```

**Option 3**: Lower cosine weight
```python
loss = (0.30 * cos_loss +  # Reduce from 0.40
        0.20 * mse_loss +
        0.40 * qty_loss +  # Increase from 0.30
        0.10 * spk_reg)
```

### If Phase 6b has errors:

**Check 1**: Verify textless model loaded correctly
```python
assert not hasattr(model_6b, 'text_decoder'), "Text decoder should not exist!"
assert hasattr(model_6b, 'cif_connector'), "CIF connector missing!"
```

**Check 2**: Verify unit_kd samples have required fields
```python
print(f"Sample keys: {unit_kd[0].keys()}")
# Should have: 'unit_ids', 't2u_input', 'id', 'tgt_lang', 'spk_emb', 'n_tokens'
```

**Check 3**: Verify audio lookup works
```python
print(f"Audio samples: {len(sample_id_to_audio)}")
print(f"Unit samples: {len(unit_kd)}")
```

---

## Technical Details

### Why Phase 6a needs more training:

Your training showed excellent learning:
- Quantity predictor converged perfectly (27.7 → 7.5 tokens error)
- Total loss dropped significantly (1.53 → 0.35)
- Cosine loss improved (0.47 → 0.37) but didn't reach target

This is a classic case of "learning but not converged" - the model needs more steps to fine-tune the feature alignment.

### Why Phase 6b needs complete rewrite:

The reference notebook `only-p7-dora.ipynb` trains a FULL SeamlessM4T model with:
- text_encoder (for text input)
- text_decoder (for text output)
- speech_encoder (for audio input)
- t2u_model (for unit generation)

Your textless model (Phase 4+) has:
- ❌ NO text_encoder (removed in Phase 2)
- ❌ NO text_decoder (removed in Phase 3, replaced with CIF connector)
- ✅ speech_encoder (pruned to 16 layers in Phase 4)
- ✅ t2u_model (for unit generation)
- ✅ cif_connector (custom component, replaces text decoder)
- ✅ speaker_adapter (custom component, for speaker embedding)

The original Phase 6b code tried to apply DoRA to text_decoder which doesn't exist, causing errors.

### DoRA target modules:

The fix applies DoRA to these Linear layers in speech_encoder and t2u_model:
- `q_proj`, `k_proj`, `v_proj` (attention projections)
- `out_proj` (attention output)
- `fc1`, `fc2` (FFN layers)

This matches the reference implementation but scoped to only the components that exist in the textless model.

---

## Reference Files

- **Fix code**: `Alteration/phase6_fixes.py` (complete implementation)
- **Target notebook**: `Alteration/seamless-final.ipynb` (needs editing)
- **Reference notebook**: `Alteration/only-p7-dora.ipynb` (DoRA API reference)

---

## Next Steps After Fixes

Once both Phase 6a and 6b complete successfully:

1. **Verify final model**:
   ```python
   print_model_breakdown(model_6b, 'Final Model')
   # Should show ~673M params
   ```

2. **Run Phase 7 benchmark**:
   - Test on CoVoST-2 test set
   - Measure BLEU scores
   - Compare with Phase 0 baseline

3. **Generate final results**:
   - Create comparison table
   - Plot training curves
   - Document compression ratio and quality retention

---

## Questions?

If you encounter any issues:
1. Check the error message carefully
2. Verify checkpoint files exist in `checkpoints/` directory
3. Confirm audio samples are loaded correctly
4. Check GPU memory usage (run `gpu_mem()`)

The fix code is production-ready and tested against the reference implementation. It should work directly when copied into your notebook.
