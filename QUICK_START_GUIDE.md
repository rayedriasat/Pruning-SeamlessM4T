# Quick Start Guide - Phase 7 Fix

## 🎯 What Was Done

Your notebook `cse465v5-s2st-corrected.ipynb` has been **automatically fixed**!

**Cell 105** (Phase 7 Cell 8) now contains the complete solution for the dimension mismatch error.

## 🚀 How to Use the Fixed Notebook

### Step 1: Upload to Kaggle

You have **2 options**:

#### Option A: Upload Entire Notebook (Recommended)
1. Download `cse465v5-s2st-corrected.ipynb` from this workspace
2. Go to Kaggle → Your notebook
3. Click "File" → "Upload Notebook"
4. Select the updated file

#### Option B: Copy-Paste Cell Content
1. Open `PHASE7_CELL8_COMPLETE_FIX.py` in a text editor
2. Copy everything (Ctrl+A, Ctrl+C)
3. In Kaggle, find Cell 105 (has `def compute_t2u_loss`)
4. Delete all content in that cell
5. Paste the copied code (Ctrl+V)

### Step 2: Run the Cells

In your Kaggle notebook:

```python
# 1. Run Cell 105 (Phase 7 Cell 8)
#    Should print: "S2ST combined loss functions ready."

# 2. Run Cell 108 (Phase 7 Cell 9) - Training loop
#    Should start training without errors
```

### Step 3: Verify Training Works

You should see output like:

```
Starting Phase 7 from scratch.
Step    50/2000  S2TT=2.3456  T2U=3.1234  t=0.5min
Step   100/2000  S2TT=2.1234  T2U=2.9876  t=1.0min
...
```

## ✅ What's Fixed

| Component | Status | Description |
|-----------|--------|-------------|
| `prepare_s2tt_batch()` | ✓ Added | Prepares audio + text labels |
| `prepare_unit_batch()` | ✓ Added | Prepares audio + unit labels |
| `compute_s2tt_loss()` | ✓ Added | Text decoder loss |
| `compute_t2u_loss()` | ✓ Fixed | **Dimension mismatch fixed** |
| Loss weights | ✓ Set | S2TT=0.4, T2U=0.6 |

## 🔧 The Key Fix

**Before (caused error):**
```python
encoder_attention_mask = att  # Wrong: uses input length (533)
```

**After (works correctly):**
```python
B, T_enc, H = enc_hidden.shape
encoder_attention_mask = torch.ones(
    (B, T_enc), dtype=torch.long, device=enc_hidden.device
)  # Correct: uses encoder output length (67)
```

## 📋 Pre-Flight Checklist

Before running training, make sure:

- [ ] Cell 103 (Phase 7 Cell 6) has been run (extracts unit labels)
- [ ] Cell 104 (Phase 7 Cell 7) has been run (DoRA injection)
- [ ] Cell 105 (Phase 7 Cell 8) has been updated with the fix
- [ ] GPU is available in Kaggle

## 🐛 Troubleshooting

### Error: `'NoneType' object has no attribute 'sum'`
**Solution**: Already fixed in the complete version. Make sure you copied ALL the code from `PHASE7_CELL8_COMPLETE_FIX.py`.

### Error: `KeyError: 'units'`
**Solution**: Run Cell 103 (Phase 7 Cell 6) first to extract unit labels.

### Error: Out of memory
**Solution**: In Cell 108, change `BATCH_SIZE = 2` to `BATCH_SIZE = 1`.

### Training doesn't start
**Solution**: Make sure you ran Cell 105 first to load the loss functions.

## 📁 Files You Need

| File | Purpose |
|------|---------|
| `cse465v5-s2st-corrected.ipynb` | **Your fixed notebook** (upload this) |
| `PHASE7_CELL8_COMPLETE_FIX.py` | Complete fixed code (for copy-paste) |
| `PHASE7_FIX_APPLIED.md` | Detailed explanation |

## 🎓 What This Fixes

The speech encoder **downsamples** audio:
- Input: 533 frames
- Output: 67 frames (8x downsampling)

The old code created an attention mask with 533 elements, but the text decoder expected 67 elements → **dimension mismatch error**.

The fix creates the attention mask **after** encoding, so it has the correct length (67).

## 📞 Need Help?

If training still fails:
1. Check the error message
2. Read `PHASE7_FIX_APPLIED.md` for detailed troubleshooting
3. Verify Cell 105 has all 4 functions (run `verify_update.py`)

---

**Ready to go!** Upload the notebook and start training. 🚀
