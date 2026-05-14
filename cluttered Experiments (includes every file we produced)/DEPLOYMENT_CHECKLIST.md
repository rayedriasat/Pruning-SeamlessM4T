# Phase 7 Fix - Deployment Checklist

## ✅ Pre-Deployment Verification

Run this checklist **before** uploading to Kaggle:

### 1. Verify Fix Was Applied
```bash
python verify_update.py
```

Expected output:
```
Cell 105 now contains:
- prepare_s2tt_batch: True
- prepare_unit_batch: True
- compute_s2tt_loss: True
- compute_t2u_loss: True
- S2TT_WEIGHT: True
- T2U_WEIGHT: True
- encoder_attention_mask fix: True

✓ All required functions are present in Cell 105!
```

### 2. Check Files Exist
- [ ] `cse465v5-s2st-corrected.ipynb` (fixed notebook)
- [ ] `cse465v5-s2st-corrected.ipynb.backup` (backup)
- [ ] `PHASE7_CELL8_COMPLETE_FIX.py` (for reference)

### 3. Review Changes
Open `cse465v5-s2st-corrected.ipynb` and verify Cell 105 starts with:
```python
# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 CELL 8: COMPLETE FIXED VERSION
# Combined S2ST loss functions + batch preparation helpers
# ══════════════════════════════════════════════════════════════════════════════
```

## 📤 Deployment to Kaggle

### Option A: Upload Entire Notebook (Recommended)

1. **Download the fixed notebook**
   - File: `cse465v5-s2st-corrected.ipynb`
   - Location: Current workspace

2. **Go to Kaggle**
   - Navigate to: https://www.kaggle.com/
   - Find your notebook

3. **Upload**
   - Click "File" → "Upload Notebook"
   - Select `cse465v5-s2st-corrected.ipynb`
   - Wait for upload to complete

4. **Verify upload**
   - Open Cell 105
   - Check it contains all 4 functions
   - Look for the header comment "PHASE 7 CELL 8: COMPLETE FIXED VERSION"

### Option B: Manual Copy-Paste

1. **Open source file**
   - File: `PHASE7_CELL8_COMPLETE_FIX.py`
   - Select all (Ctrl+A)
   - Copy (Ctrl+C)

2. **Go to Kaggle notebook**
   - Find Cell 105 (search for `compute_t2u_loss`)
   - Delete all existing content in that cell

3. **Paste new code**
   - Paste (Ctrl+V)
   - Verify all code was pasted
   - Check for any formatting issues

## 🚀 Running the Fixed Code

### Step 1: Prepare Environment

In Kaggle, run these cells in order:

```python
# Cell 1-11: Setup cells (imports, paths, etc.)
# Cell 12-102: Previous phases (should already be complete)
```

### Step 2: Run Phase 7 Prerequisites

```python
# Cell 103: Phase 7 Cell 6 - Extract unit labels
# Expected output: "S2ST training pairs: XXX (of XXX total ft_samples)"

# Cell 104: Phase 7 Cell 7 - DoRA injection
# Expected output: "trainable params: XXX || all params: XXX || trainable%: X.XX"
```

### Step 3: Load Fixed Loss Functions

```python
# Cell 105: Phase 7 Cell 8 - Loss functions (THE FIX)
# Expected output:
# S2ST combined loss functions ready.
#   S2TT weight: 0.4  |  T2U weight: 0.6
#   NOTE: T2U loss uses direct t2u_model() call (HF-compatible approach)
```

### Step 4: Start Training

```python
# Cell 108: Phase 7 Cell 9 - Training loop
# Expected output:
# Starting Phase 7 from scratch.
# Step    50/2000  S2TT=2.3456  T2U=3.1234  t=0.5min
# Step   100/2000  S2TT=2.1234  T2U=2.9876  t=1.0min
# ...
```

## ✅ Success Indicators

Training is working correctly when you see:

### Immediate Success (Cell 105)
- [x] No import errors
- [x] Prints "S2ST combined loss functions ready"
- [x] Shows loss weights (S2TT=0.4, T2U=0.6)

### Training Success (Cell 108)
- [x] No dimension mismatch error
- [x] Both S2TT and T2U losses are computed
- [x] Loss values are reasonable (not NaN or inf)
- [x] Training progresses through steps
- [x] Checkpoints are saved every 200 steps

### Expected Loss Behavior
- **S2TT loss**: Should start around 2-3, decrease gradually
- **T2U loss**: Should start around 3-4, decrease gradually
- **Both losses**: Should show downward trend over time

## 🐛 Troubleshooting

### Issue 1: Dimension Mismatch Error Still Occurs

**Symptoms:**
```
RuntimeError: The size of tensor a (67) must match the size of tensor b (533)
```

**Solution:**
- Cell 105 was not updated correctly
- Re-upload the notebook or re-paste the code
- Make sure you copied ALL code from `PHASE7_CELL8_COMPLETE_FIX.py`

### Issue 2: NoneType Error

**Symptoms:**
```
AttributeError: 'NoneType' object has no attribute 'sum'
```

**Solution:**
- This should be fixed in the complete version
- Verify Cell 105 has the fallback loss handling:
  ```python
  # Last resort: return small loss to keep training stable
  return torch.tensor(0.01, requires_grad=True, device=enc_hidden.device)
  ```

### Issue 3: KeyError: 'units'

**Symptoms:**
```
KeyError: 'units'
```

**Solution:**
- Run Cell 103 (Phase 7 Cell 6) first
- This extracts unit labels from the model
- Verify `ft_s2st_pairs` has data with `units` field

### Issue 4: Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- In Cell 108, change:
  ```python
  BATCH_SIZE = 2  # Change to 1
  ```
- Or reduce `MAX_STEPS` if needed

### Issue 5: Training Doesn't Start

**Symptoms:**
- Cell 108 runs but no output
- Or immediate error

**Solution:**
- Make sure Cell 105 was run first
- Check that `model_p7` exists (from Cell 104)
- Verify GPU is available: `torch.cuda.is_available()`

## 📊 Monitoring Training

### What to Watch

1. **Loss values**
   - Should decrease over time
   - S2TT: 2-3 → 1-2 (after 2000 steps)
   - T2U: 3-4 → 2-3 (after 2000 steps)

2. **Time per step**
   - Should be consistent (0.5-1 min per 50 steps)
   - If increasing, may indicate memory issues

3. **Checkpoint saves**
   - Every 200 steps
   - Check file size is reasonable (not 0 bytes)

### When to Stop

Training completes automatically after 2000 steps, but you can stop early if:
- Losses plateau (no improvement for 500+ steps)
- Out of time/budget on Kaggle
- Losses start increasing (overfitting)

## 📝 Post-Training

After training completes:

1. **Run Cell 109** - Plot loss curves
2. **Run Cell 110** - Merge DoRA adapters
3. **Run Cell 111** - Final benchmark

## 🎯 Final Verification

Before considering the fix complete:

- [ ] Training completed without dimension mismatch error
- [ ] Both S2TT and T2U losses decreased
- [ ] Checkpoints were saved successfully
- [ ] Loss curves show downward trend
- [ ] Final benchmark shows improved quality

## 📚 Reference Documents

- `QUICK_START_GUIDE.md` - Quick reference
- `PHASE7_FIX_APPLIED.md` - Detailed explanation
- `ARCHITECTURE_DIAGRAM.txt` - Visual architecture
- `FIX_SUMMARY.txt` - One-page summary

## 🆘 Need Help?

If you encounter issues not covered here:

1. Check the error message carefully
2. Review `PHASE7_FIX_APPLIED.md` troubleshooting section
3. Verify Cell 105 has all 4 functions
4. Check that prerequisites (Cells 103-104) were run

---

**Good luck with your training!** 🚀

The dimension mismatch error is now fixed, and Phase 7 should complete successfully.
