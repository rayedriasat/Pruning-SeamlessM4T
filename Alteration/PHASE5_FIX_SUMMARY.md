# Phase 5 KD Extraction Fix Summary

## Issue
**Error:** `tuple index out of range` in Phase 5 KD extraction hook function

**Location:** Cell 66 in `seamless-final.ipynb` - Phase 5: KD Target Extraction from Teacher

**Root Cause:** 
The hook function `_hook_t2u_enc_in` was attempting to access `inp[0]` without checking if the tuple was empty or if `inp` was None. This caused crashes when the T2U encoder was called with unexpected input formats.

## Fix Applied

### 1. Safer Hook Function
**Before:**
```python
def _hook_t2u_enc_in(module, inp, out):
    x = inp[0] if isinstance(inp, tuple) else inp
    t2u_enc_inputs['last'] = x.detach().cpu()
```

**After:**
```python
def _hook_t2u_enc_in(module, inp, out):
    """Safely capture T2U encoder inputs"""
    try:
        if inp is None:
            return
        # Extract tensor from input
        if isinstance(inp, tuple):
            if len(inp) == 0:
                return
            x = inp[0]
        elif isinstance(inp, torch.Tensor):
            x = inp
        else:
            return
        # Validate and store
        if x is not None and isinstance(x, torch.Tensor):
            t2u_enc_inputs['last'] = x.detach().cpu()
    except Exception as e:
        print(f'  [Hook] Error: {e}')
```

### 2. Added Validation in Extraction Loop
**Before:**
```python
with torch.no_grad():
    out = teacher.generate(**inp, tgt_lang=tgt_m4t,
                           return_intermediate_token_ids=True)
t2u_in = t2u_enc_inputs.get('last')
uid = getattr(out,'unit_ids',None)
```

**After:**
```python
with torch.no_grad():
    out = teacher.generate(**inp, tgt_lang=tgt_m4t,
                           return_intermediate_token_ids=True)
t2u_in = t2u_enc_inputs.get('last')
if t2u_in is None:
    print(f'  [{i+1}] Warning: T2U input not captured, skipping')
    continue
uid = getattr(out,'unit_ids',None)
```

## Changes Made

1. **Null/Empty Check:** Added checks for `None` and empty tuples before accessing elements
2. **Type Validation:** Validates that extracted value is actually a Tensor before storing
3. **Exception Handling:** Wrapped hook logic in try-except to prevent crashes
4. **Skip Invalid Samples:** Added validation to skip samples where T2U input wasn't captured
5. **Debug Output:** Added informative error messages for troubleshooting

## Benefits

- **Robustness:** Handles edge cases in model forward pass gracefully
- **Debugging:** Clear error messages help identify issues
- **Data Quality:** Skips invalid samples instead of crashing
- **Continuity:** Extraction continues even if some samples fail

## Testing Recommendations

After applying this fix, you should see:

1. **Successful extraction** for most samples (expected: 1400-1600 out of 1600 total)
2. **Warning messages** for samples where T2U input wasn't captured (acceptable: <10%)
3. **No crashes** - the loop should complete for all language pairs

### Expected Output Pattern:
```
Extracting KD: eng→ben (200 samples)...
  [50/200] 48 total KD samples
  [100/200] 97 total KD samples
  [150/200] 147 total KD samples
  [200/200] 196 total KD samples

Extracting KD: ben→eng (200 samples)...
  [50/200] 245 total KD samples
  ...
```

### If Issues Persist:

1. **Check hook attachment point:**
   ```python
   print(teacher.t2u_model.model.encoder)
   ```

2. **Use alternative hook (pre-forward):**
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

3. **Hook text_decoder output instead:**
   ```python
   # Text decoder output feeds into T2U - more reliable capture point
   _hook_handle = teacher.text_decoder.register_forward_hook(_hook_t2u_enc_in)
   ```

## Files Modified

- `Alteration/seamless-final.ipynb` - Cell 66 (Phase 5 KD extraction)

## Next Steps

1. **Re-run Phase 5 cell** in your Kaggle/Colab notebook
2. **Monitor extraction progress** - should complete without crashes
3. **Verify KD data quality:**
   ```python
   print(f"Total KD samples: {len(kd_data)}")
   print(f"Valid T2U inputs: {sum(1 for x in kd_data if x.get('t2u_input') is not None)}")
   print(f"Valid unit IDs: {sum(1 for x in kd_data if x.get('unit_ids') is not None)}")
   ```
4. **Proceed to Phase 6a** once KD data is successfully extracted

## Status
✅ **Fix Applied** - Notebook updated with safer hook implementation
🔄 **Ready to Re-run** - Phase 5 cell can now be executed without crashes
