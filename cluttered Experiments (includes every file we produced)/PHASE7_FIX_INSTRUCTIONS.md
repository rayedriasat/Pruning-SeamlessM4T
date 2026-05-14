# Phase 7 Training Loop Fix - Dimension Mismatch Error

## Problem Summary

The Phase 7 training loop fails immediately with:
```
RuntimeError: The size of tensor a (67) must match the size of tensor b (533) at non-singleton dimension 3
```

## Root Cause

The **speech encoder downsamples** the input sequence:
- Input features: 533 frames
- Speech encoder output: 67 frames (8x downsampling)

The bug: `encoder_attention_mask` was created with the **input** length (533) but passed to `text_decoder` which expects it to match the **encoder output** length (67).

## The Fix

### Location
**Phase 7 Cell 8** - Replace the `compute_t2u_loss` function

### Key Changes

1. **Create attention mask AFTER encoding** (not before):
   ```python
   # OLD (WRONG):
   encoder_attention_mask = att  # Uses input length (533)
   
   # NEW (CORRECT):
   B, T_enc, H = enc_hidden.shape
   encoder_attention_mask = torch.ones(
       (B, T_enc), dtype=torch.long, device=enc_hidden.device
   )  # Uses encoder output length (67)
   ```

2. **Add sequence length alignment** for unit_labels:
   ```python
   # Align unit_labels with T2U output length
   ul = unit_labels[:, :T_out] if unit_labels.shape[1] > T_out else unit_labels
   if ul.shape[1] < T_out:
       pad_len = T_out - ul.shape[1]
       ul = F.pad(ul, (0, pad_len), value=-100)
   ```

## How to Apply

### Option 1: Copy from fix_phase7_t2u_loss.py
1. Open `fix_phase7_t2u_loss.py` (created in this directory)
2. Copy the `compute_t2u_loss` function
3. In your notebook, find **Phase 7 Cell 8** (the cell with loss functions)
4. Replace the existing `compute_t2u_loss` function with the fixed version

### Option 2: Manual Edit
In **Phase 7 Cell 8**, find this line:
```python
text_dec_out = base.text_decoder(
    input_ids=dec_input_ids,
    encoder_hidden_states=enc_hidden,
    encoder_attention_mask=att,  # ← WRONG: uses input length
    return_dict=True,
)
```

Replace with:
```python
# CRITICAL FIX: Create encoder_attention_mask matching enc_hidden length
B, T_enc, H = enc_hidden.shape
encoder_attention_mask = torch.ones(
    (B, T_enc), dtype=torch.long, device=enc_hidden.device
)

text_dec_out = base.text_decoder(
    input_ids=dec_input_ids,
    encoder_hidden_states=enc_hidden,
    encoder_attention_mask=encoder_attention_mask,  # ← CORRECT
    return_dict=True,
)
```

## Verification

After applying the fix, the training loop should:
1. ✓ Start without dimension mismatch errors
2. ✓ Show S2TT and T2U loss values
3. ✓ Progress through training steps

Expected output:
```
Step    50/2000  S2TT=2.3456  T2U=3.1234  t=0.5min
Step   100/2000  S2TT=2.1234  T2U=2.9876  t=1.0min
...
```

## Why This Matters

The T2U model is responsible for converting text tokens → speech units → audio waveform.

Without this fix:
- Training crashes immediately
- No T2U gradients flow
- Audio output remains broken

With this fix:
- Training proceeds normally
- T2U learns to generate correct speech units
- Audio output quality recovers

## Technical Details

### SeamlessM4T Architecture
```
Input Audio (16kHz)
    ↓
Speech Encoder (downsamples 8x)
    ↓ [B, 67, 1024]  ← Output sequence length
Text Decoder
    ↓
T2U Model
    ↓
Vocoder
    ↓
Output Audio
```

### Attention Mask Requirements
- `speech_encoder`: accepts mask with input length
- `text_decoder`: requires mask matching **encoder output** length
- Mismatch → dimension error in cross-attention layers

## Related Files
- `fix_phase7_t2u_loss.py` - Complete fixed function
- `cse465v5-s2st-corrected.ipynb` - Your notebook (Phase 7 Cell 8)

## Questions?
If training still fails after applying this fix, check:
1. Did you replace the entire `compute_t2u_loss` function?
2. Are you running Phase 7 Cell 9 (training loop) after updating Cell 8?
3. Do you have sufficient GPU memory? (T4 with 15GB should work)
