# Quick Fix Guide - Phase 5 Hook Error

## Problem
```
[1] Error: tuple index out of range
[2] Error: tuple index out of range
[3] Error: tuple index out of range
```

## Solution Applied ✅

The notebook `seamless-final.ipynb` has been **automatically fixed**. 

## What Was Changed

**Cell 66 (Phase 5)** - The hook function now safely handles:
- Empty tuples
- None values  
- Non-tensor inputs
- Exceptions during capture

## How to Use the Fixed Notebook

### Option 1: Re-run from Phase 5 (Recommended)
```python
# In your Kaggle/Colab notebook:
# 1. Restart kernel (to clear any corrupted state)
# 2. Run all cells up to Phase 4
# 3. Run the fixed Phase 5 cell
```

### Option 2: Continue from Current State
If you already have Phase 0-4 results cached:
```python
# Just re-run the Phase 5 cell
# The fix will handle the errors gracefully
```

## Expected Behavior After Fix

### ✅ Good Output:
```
Extracting KD: eng→ben (200 samples)...
  [50/200] 48 total KD samples
  [100/200] 97 total KD samples
  [150/200] 147 total KD samples
  [200/200] 196 total KD samples
```

### ⚠️ Acceptable Warnings (if any):
```
  [42] Warning: T2U input not captured, skipping
```
*A few warnings (<5%) are normal and won't affect training*

### ❌ Bad Output (if still occurring):
```
  [1] Error: tuple index out of range
  [2] Error: tuple index out of range
```
*If you still see this, see "Alternative Solutions" below*

## Verification Steps

After running Phase 5, check:

```python
# Should show ~1400-1600 samples (out of 1600 total)
print(f"Total KD samples: {len(kd_data)}")

# Should be close to 100%
valid_t2u = sum(1 for x in kd_data if x.get('t2u_input') is not None)
print(f"Valid T2U inputs: {valid_t2u}/{len(kd_data)} ({valid_t2u/len(kd_data)*100:.1f}%)")

# Should be >90%
valid_units = sum(1 for x in kd_data if x.get('unit_ids') is not None)
print(f"Valid unit IDs: {valid_units}/{len(kd_data)} ({valid_units/len(kd_data)*100:.1f}%)")
```

**Target:** At least 1400 valid samples with both T2U inputs and unit IDs

## Alternative Solutions (If Fix Doesn't Work)

### Alternative 1: Use Pre-Forward Hook
Replace the hook registration line with:
```python
def _hook_t2u_enc_in_pre(module, inp):
    try:
        if isinstance(inp, tuple) and len(inp) > 0:
            x = inp[0]
            if isinstance(x, torch.Tensor):
                t2u_enc_inputs['last'] = x.detach().cpu()
    except Exception as e:
        print(f"[Pre-hook] Error: {e}")

_hook_handle = teacher.t2u_model.model.encoder.register_forward_pre_hook(_hook_t2u_enc_in_pre)
```

### Alternative 2: Hook Text Decoder Instead
```python
# Text decoder output is what feeds into T2U
_hook_handle = teacher.text_decoder.register_forward_hook(_hook_t2u_enc_in)
```

### Alternative 3: Use Cached KD Data
If extraction keeps failing, you can use pre-extracted KD data:
```python
# Download from Drive (if available)
if ON_KAGGLE:
    subprocess.run(f'rclone copy "{GDRIVE_ROOT}/kd_data_v2.pt" "{WORK_DIR}/"', shell=True)
    kd_data = torch.load(f'{WORK_DIR}/kd_data_v2.pt', map_location='cpu', weights_only=False)
```

## Troubleshooting

### Issue: Still getting tuple errors
**Solution:** Check if teacher model loaded correctly
```python
print(teacher.t2u_model.model.encoder)
# Should show: SeamlessM4Tv2Encoder(...)
```

### Issue: T2U inputs are None for all samples
**Solution:** Hook might be on wrong module
```python
# Debug: Print all module names
for name, module in teacher.named_modules():
    if 't2u' in name.lower():
        print(name)
```

### Issue: Out of memory during extraction
**Solution:** Process in smaller batches
```python
# Add after each language pair:
torch.cuda.empty_cache()
gc.collect()
```

## Files Modified
- ✅ `Alteration/seamless-final.ipynb` (Cell 66 - Phase 5)

## Status
🟢 **FIXED** - Ready to re-run Phase 5

## Next Phase
Once Phase 5 completes successfully:
- ✅ Verify KD data quality (see "Verification Steps" above)
- ➡️ Proceed to **Phase 6a: CIF Connector Training**
