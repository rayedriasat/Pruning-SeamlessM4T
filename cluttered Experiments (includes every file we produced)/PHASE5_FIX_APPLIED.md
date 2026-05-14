# Phase 5 Device Mismatch Fix - APPLIED (v2)

## Date: April 18, 2026

## Problem Summary
Phase 5 Cell 6 was failing with "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!" during FLAP calibration for all three components (text_decoder, speech_encoder, t2u_model).

## Root Causes (Two Issues Found)
1. The `processor(audio=wav, ...)` creates tensors on CPU by default. While `input_features` and `attention_mask` were explicitly moved to device, **hidden tensors** in the processor output dictionary remained on CPU.
2. The **stats dictionary tensors** (`sum_x`, `sq_sum`) were created on CPU but the hook was trying to add cuda tensors to them.

## Solution Applied (v2)
Updated **Phase 5 Cell 2** (cell index 67) in `cse465v5-s2st-corrected.ipynb` with the complete fix that:

1. **Creates stats tensors on the correct device** (cuda:0):
   ```python
   stats[key] = {
       "sum_x":  torch.zeros(fc1.in_features, dtype=torch.float64, device=device),
       "sq_sum": torch.zeros(fc1.in_features, dtype=torch.float64, device=device),
       ...
   }
   ```

2. **Moves ALL tensors in the processor output dict to device**:
   ```python
   enc_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in enc_in.items()}
   ```

3. **Moves final stats to CPU** for processing after calibration:
   ```python
   mean_x   = (s["sum_x"]  / n).float().cpu()
   sq_norm  = (s["sq_sum"] / n).float().cpu()
   ```

## Key Changes
**Issue 1 - Processor outputs on CPU:**
```python
# BEFORE (broken):
enc_in = processor(audio=wav, sampling_rate=16000, return_tensors="pt")
input_feats = enc_in["input_features"].to(device)
attn_mask = enc_in["attention_mask"].to(device)

# AFTER (fixed):
enc_in = processor(audio=wav, sampling_rate=16000, return_tensors="pt")
enc_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
          for k, v in enc_in.items()}
input_feats = enc_in["input_features"]
attn_mask = enc_in["attention_mask"]
```

**Issue 2 - Stats tensors on CPU:**
```python
# BEFORE (broken):
stats[key] = {
    "sum_x":  torch.zeros(fc1.in_features, dtype=torch.float64),
    "sq_sum": torch.zeros(fc1.in_features, dtype=torch.float64),
    ...
}

# AFTER (fixed):
stats[key] = {
    "sum_x":  torch.zeros(fc1.in_features, dtype=torch.float64, device=device),
    "sq_sum": torch.zeros(fc1.in_features, dtype=torch.float64, device=device),
    ...
}
```

## Expected Result
- All 25 calibration samples should process successfully
- All FFN layers should fire correctly:
  - text_decoder: 16/16 layers
  - speech_encoder: 38/38 layers  
  - t2u_model: 6/6 layers (encoder + decoder)
- FLAP pruning should complete and save phase5_flap_pruned model

## Next Steps
1. Run Phase 5 Cell 6 in the notebook
2. Verify all calibration samples succeed
3. Confirm all layers fire correctly
4. Check that phase5_flap_pruned model is saved

## Files Modified
- `cse465v5-s2st-corrected.ipynb` - Phase 5 Cell 2 (cell index 67) updated with v2 fix

## Files Created
- `fix_phase5_cell2.py` - Script for initial fix attempt (v1)
- `fix_phase5_cell2_v2.py` - Script for complete fix (v2 - device-aware stats)
- `PHASE5_FIX_APPLIED.md` - This summary document
