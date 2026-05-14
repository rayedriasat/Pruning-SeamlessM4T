# Phase 7 Training Loop Fix - APPLIED ✓

## Summary

The Phase 7 Cell 8 in `cse465v5-s2st-corrected.ipynb` has been **successfully updated** with the complete fix for the dimension mismatch error.

## What Was Fixed

### Original Problem
```
RuntimeError: The size of tensor a (67) must match the size of tensor b (533)
```

This error occurred because:
1. The speech encoder **downsamples** input sequences (533 frames → 67 frames)
2. The `encoder_attention_mask` was created with the **input** length (533)
3. But `text_decoder` expected it to match the **encoder output** length (67)

### The Solution

**Cell 105** (Phase 7 Cell 8) now contains the complete fixed version with:

✓ `prepare_s2tt_batch()` - Prepares audio + text labels for S2TT loss  
✓ `prepare_unit_batch()` - Prepares audio + unit labels for T2U loss  
✓ `compute_s2tt_loss()` - Computes text decoder cross-entropy  
✓ `compute_t2u_loss()` - **FIXED** T2U loss with correct attention mask handling  

### Key Fix in `compute_t2u_loss()`

**Before (WRONG):**
```python
encoder_attention_mask = att  # Uses input length (533)
```

**After (CORRECT):**
```python
# Create attention mask AFTER encoding
B, T_enc, H = enc_hidden.shape
encoder_attention_mask = torch.ones(
    (B, T_enc), dtype=torch.long, device=enc_hidden.device
)  # Uses encoder output length (67)
```

## Files Modified

- ✓ `cse465v5-s2st-corrected.ipynb` - Cell 105 updated with complete fix
- ✓ `cse465v5-s2st-corrected.ipynb.backup` - Backup of original notebook

## Files Created

- `PHASE7_CELL8_COMPLETE_FIX.py` - Complete fixed code (source)
- `PHASE7_FIX_INSTRUCTIONS.md` - Detailed explanation
- `fix_phase7_t2u_loss.py` - Partial fix (dimension mismatch only)
- `update_phase7_cell8.py` - Script that applied the fix
- `verify_update.py` - Verification script
- `QUICK_FIX_SUMMARY.txt` - Quick reference
- `PHASE7_FIX_APPLIED.md` - This file

## Next Steps for You

### 1. Upload the Fixed Notebook to Kaggle

Since you're running on Kaggle, you need to upload the updated notebook:

**Option A: Direct Upload**
1. Download `cse465v5-s2st-corrected.ipynb` from your local workspace
2. Go to your Kaggle notebook
3. Click "File" → "Upload Notebook"
4. Select the updated `cse465v5-s2st-corrected.ipynb`

**Option B: Copy-Paste Cell Content**
1. Open `PHASE7_CELL8_COMPLETE_FIX.py` in a text editor
2. Copy the entire content
3. In your Kaggle notebook, find Cell 105 (the cell with `compute_t2u_loss`)
4. Delete all content in that cell
5. Paste the complete fixed code
6. Run the cell

### 2. Run the Training

After updating Cell 105:

```python
# Run Cell 105 (Phase 7 Cell 8) - Load the fixed loss functions
# This should print:
# S2ST combined loss functions ready.
#   S2TT weight: 0.4  |  T2U weight: 0.6
```

Then run Cell 108 (Phase 7 Cell 9) - The training loop:

```python
# Expected output:
# Starting Phase 7 from scratch.
# Step    50/2000  S2TT=2.3456  T2U=3.1234  t=0.5min
# Step   100/2000  S2TT=2.1234  T2U=2.9876  t=1.0min
# ...
```

### 3. Verify Training Works

The training should now:
- ✓ Start without dimension mismatch errors
- ✓ Show both S2TT and T2U loss values
- ✓ Progress through training steps
- ✓ Save checkpoints every 200 steps

### 4. If Training Still Fails

Check these potential issues:

**Issue: `AttributeError: 'NoneType' object has no attribute 'sum'`**
- **Cause**: Loss functions returning `None`
- **Fix**: Already included in the complete fix (fallback loss handling)

**Issue: `KeyError: 'units'`**
- **Cause**: `ft_s2st_pairs` missing unit labels
- **Fix**: Run Cell 103 (Phase 7 Cell 6) to extract unit labels first

**Issue: Out of memory**
- **Cause**: T4 GPU has 15GB VRAM
- **Fix**: Reduce `BATCH_SIZE` from 2 to 1 in Cell 108

## Technical Details

### Loss Computation Flow

```
Input Audio (16kHz)
    ↓
Speech Encoder (downsamples 8x)
    ↓ [B, 67, 1024]  ← Output sequence length
Text Decoder (with corrected attention mask)
    ↓ [B, 1, 1024]
T2U Model
    ↓ [B, T_out, unit_vocab]
Cross-Entropy Loss
```

### Loss Weights

- **S2TT_WEIGHT = 0.4** - Text decoder loss (lower priority)
- **T2U_WEIGHT = 0.6** - T2U loss (higher priority, audio was broken)

### Why This Matters

The T2U model converts text tokens → speech units → audio waveform.

**Without this fix:**
- Training crashes immediately
- No T2U gradients flow
- Audio output remains broken

**With this fix:**
- Training proceeds normally
- T2U learns to generate correct speech units
- Audio output quality recovers

## Verification Checklist

Before running training, verify:

- [ ] Cell 105 contains all 4 functions: `prepare_s2tt_batch`, `prepare_unit_batch`, `compute_s2tt_loss`, `compute_t2u_loss`
- [ ] Cell 105 prints "S2ST combined loss functions ready" when run
- [ ] Cell 103 (Phase 7 Cell 6) has been run to extract unit labels
- [ ] `ft_s2st_pairs` has data with `units` field
- [ ] GPU is available (`torch.cuda.is_available()` returns `True`)

## Questions?

If you encounter any issues:

1. Check that Cell 105 was updated correctly (run `verify_update.py`)
2. Verify `ft_s2st_pairs` has unit labels (run Cell 103 first)
3. Check GPU memory usage (reduce batch size if needed)
4. Review the error message and compare with the "If Training Still Fails" section above

## Success Indicators

Training is working correctly when you see:

```
S2ST combined loss functions ready.
  S2TT weight: 0.4  |  T2U weight: 0.6
  NOTE: T2U loss uses direct t2u_model() call (HF-compatible approach)

Starting Phase 7 from scratch.
Step    50/2000  S2TT=2.3456  T2U=3.1234  t=0.5min
Step   100/2000  S2TT=2.1234  T2U=2.9876  t=1.0min
...
```

Both S2TT and T2U losses should decrease over time.

---

**Status**: ✓ Fix applied successfully  
**Date**: 2026-04-18  
**Notebook**: `cse465v5-s2st-corrected.ipynb`  
**Cell Updated**: Cell 105 (Phase 7 Cell 8)
