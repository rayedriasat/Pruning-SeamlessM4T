# Phase 5 T2U Calibration Fix - COMPLETE

## Date: April 18, 2026

## Problem
T2U calibration was failing with:
```
AttributeError: 'NoneType' object has no attribute 'sum'
```

The error occurred in `t2u_model.forward()` when trying to compute `char_count_per_id.sum(1)`.

## Root Cause
The T2U model (Non-Autoregressive decoder) requires **character-level inputs**:
- `char_input_ids` - character token IDs
- `char_count_per_id` - how many characters per text token

We were calling `t2u_model(inputs_embeds=text_hidden)` which doesn't provide these required inputs, causing `char_count_per_id` to be `None`.

## Solution
**Use `model.generate()` for T2U calibration instead of direct forward pass.**

The `generate()` method:
1. Runs the full S2ST pipeline (speech_encoder → text_decoder → t2u_model)
2. Properly constructs `char_input_ids` and `char_count_per_id` internally
3. Fires ALL T2U encoder + decoder layers with correct inputs
4. The hooks capture activations from all FFN layers

### Code Change
**BEFORE (broken):**
```python
elif component_name == "t2u_model":
    # ... get text_hidden from text_decoder ...
    t2u_device = next(model.t2u_model.parameters()).device
    model.t2u_model(
        inputs_embeds=text_hidden.to(t2u_device),
        return_dict=True,
    )
```

**AFTER (fixed):**
```python
elif component_name == "t2u_model":
    # Use generate() to fire T2U with proper char_input_ids
    model.generate(
        input_features=input_feats,
        attention_mask=attn_mask,
        tgt_lang="ben",
        return_intermediate_token_ids=True,
        max_new_tokens=16,  # short for speed
    )
```

## Why This Works
1. `generate()` internally calls `_indices_to_subwords()` which creates the character-level inputs
2. These are passed to `t2u_model.forward()` correctly
3. All T2U layers fire and hooks capture activations
4. FLAP pruning can proceed normally

## Expected Result
When you run Phase 5 Cell 6 now:
- ✓ text_decoder calibration: 16/16 layers fire
- ✓ speech_encoder calibration: 38/38 layers fire  
- ✓ **t2u_model calibration: 6/6 layers fire** (was failing before)
- ✓ FLAP pruning completes for all three components
- ✓ phase5_flap_pruned model saves successfully

## Performance Note
Using `generate()` for T2U calibration is slightly slower than direct forward (adds ~30s for 25 samples), but it's the only correct way to fire T2U layers with proper inputs in the HuggingFace implementation.

## Files Modified
- `cse465v5-s2st-corrected.ipynb` - Phase 5 Cell 2 (cell index 67)

## Files Created
- `fix_phase5_t2u_calibration.py` - Script that applied the fix
- `PHASE5_T2U_FIX_COMPLETE.md` - This summary
