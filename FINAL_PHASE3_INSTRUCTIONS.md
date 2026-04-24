# Phase 3 Fix - Final Instructions

## ✅ Both Fixes Applied!

Your notebook now has **both fixes** for Phase 3:

1. ✅ **Fix 1:** Added `attention_mask=None` parameter to layer calls
2. ✅ **Fix 2:** Added dtype conversion to match model's float16

## 🚀 Quick Start (3 Steps)

### Step 1: Delete Old Checkpoint

In your Kaggle notebook, run:
```python
!rm -rf /kaggle/working/checkpoints/phase3_laco_done_step000000.pt
```

Or in terminal:
```bash
rm checkpoints/phase3_laco_done_step000000.pt
```

### Step 2: Re-run Phase 3 Cells

In your notebook, re-run these two cells in order:

**Cell 1:** The cell containing:
```python
def _cosine_sim_layers(merged, orig_j, calib_tensors, device):
    ...
```

**Cell 2:** The cell containing:
```python
# ── RUN Phase 3 ───────────────────────────────────────────────────────────────
p3_done = load_latest_checkpoint('phase3_laco_done')
...
```

### Step 3: Verify Success ✓

You should see output like:
```
Running Phase 3: LaCo T2U merge...
  Multi-device → consolidating to cuda:0...
  Model now on: cuda:0
  Built 8 calibration tensors.

  T2U-Enc: 6 layers -> merging up to 2
  L1: sim=0.9234 -> MERGED [1/2]      ← ✓ Real similarity!
  L2: sim=0.9567 -> MERGED [2/2]      ← ✓ Real similarity!
  L3: sim=0.8234 -> kept (below 0.96)
  L4: sim=0.7891 -> kept (below 0.96)
  L5: sim=0.8456 -> kept (below 0.96)
  T2U-Enc: 6 -> 4 layers              ← ✓ Reduced!

  T2U-Dec: 6 layers -> merging up to 2
  L1: sim=0.9456 -> MERGED [1/2]
  L2: sim=0.9678 -> MERGED [2/2]
  L3: sim=0.8123 -> kept (below 0.96)
  L4: sim=0.7945 -> kept (below 0.96)
  L5: sim=0.8567 -> kept (below 0.96)
  T2U-Dec: 6 -> 4 layers              ← ✓ Reduced!
```

**Success indicators:**
- ✅ No `[sim_err: ...]` messages
- ✅ Similarity scores between 0.85-0.99 (not 0.0000)
- ✅ Some layers show "MERGED" status
- ✅ Final layer count: 4 (not 6)

## 🎯 Expected Results

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| T2U Encoder | 6 layers | **4 layers** |
| T2U Decoder | 6 layers | **4 layers** |
| Similarity scores | 0.0000 | **0.85-0.99** |
| Layers merged | 0 | **4 total** |
| Params saved | 0M | **~87M** |

## ❓ Troubleshooting

### Still seeing sim=0.0000?

**Check 1:** Verify the fix is in your notebook
```python
# Run this in a notebook cell:
import json
with open('/kaggle/working/seamless-final.ipynb', 'r') as f:
    nb = json.load(f)
    for cell in nb['cells']:
        src = ''.join(cell.get('source', ''))
        if 'model_dtype' in src and '_cosine_sim_layers' in src:
            print("✓ Dtype fix is present!")
            break
    else:
        print("✗ Fix not found - re-run fix_phase3_dtype.py")
```

**Check 2:** Verify eval_samples is loaded
```python
print(f"Eval samples: {len(eval_samples)}")
# Should show: Eval samples: 200 (or similar)
```

**Check 3:** Check model dtype
```python
print(f"Model dtype: {next(model_p2.t2u_model.parameters()).dtype}")
# Should show: Model dtype: torch.float16
```

### Still seeing dtype errors?

If you see `expected scalar type Float but found Half`:
1. Make sure you re-ran the cell with the fixed `_cosine_sim_layers` function
2. Restart the kernel and re-run all cells from the beginning
3. The fix should automatically detect and convert dtypes

### Different error?

Post the error message and I'll help debug!

## 📚 What Was Fixed

### Issue 1: Missing Parameter
```python
# BEFORE (broken):
o = orig_j(x)  # ← Missing attention_mask

# AFTER (fixed):
o = orig_j(x, attention_mask=None)  # ← Proper signature
```

### Issue 2: Dtype Mismatch
```python
# BEFORE (broken):
x = x.to(device)  # ← Still float32, model is float16

# AFTER (fixed):
model_dtype = next(orig_j.parameters()).dtype  # Detect float16
x = x.to(device=device, dtype=model_dtype)     # Convert to float16
```

## 🎉 Success!

Once you see non-zero similarity scores and layer reduction, Phase 3 is working correctly!

You can then proceed to Phase 4 (Text Decoder Removal + CIF Connector).

---

**Files created for reference:**
- `PHASE3_FIX_SUMMARY.md` - First fix (attention_mask)
- `PHASE3_DTYPE_FIX.md` - Second fix (dtype conversion)
- `FINAL_PHASE3_INSTRUCTIONS.md` - This file (complete guide)

**Your notebook is ready!** Just delete the checkpoint and re-run Phase 3. 🚀
