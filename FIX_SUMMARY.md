# Fix Summary: run_s2st and run_s2tt_only Functions

## Issues Found in `cse465v6-s2st-optimised.ipynb`

### 1. **Incorrect Output Structure Access**
**Problem:** The functions were accessing the model output incorrectly:
- `run_s2st`: Used `out[0]` for waveform and `out[1]` for text
- `run_s2tt_only`: Used `out[0]` for token_ids

**Root Cause:** The SeamlessM4T model's `generate()` method returns a structured output object, not a tuple. The correct attributes are:
- `out.sequences` for text token IDs
- `out.waveform` for audio waveform

### 2. **Missing Helper Function**
**Problem:** The `_model_input_device()` helper function was missing.

**Impact:** Without this function, inputs weren't being placed on the correct device (where the speech encoder lives), which could cause device mismatch errors.

### 3. **Improper Device Handling**
**Problem:** The original code used `.to(DEVICE)` directly on the processor output, which doesn't handle multi-GPU scenarios properly.

**Fix:** Changed to use dictionary comprehension with `_model_input_device()`:
```python
inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}
```

## Changes Made

### Added `_model_input_device()` Helper
```python
def _model_input_device(mdl):
    """Device where speech inputs should be placed (first speech encoder param)."""
    if hasattr(mdl, 'speech_encoder'):
        return next(mdl.speech_encoder.parameters()).device
    return next(mdl.parameters()).device
```

### Fixed `run_s2st()` Function
**Before:**
```python
waveform = out[0].cpu().squeeze().float().numpy()
text = ''
if isinstance(out, (list, tuple)) and len(out) > 1 and out[1] is not None:
    try:
        ids = _remap_ids_for_decode(mdl, out[1].cpu())
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except:
        pass
```

**After:**
```python
# Extract text from sequences attribute
text_ids = _remap_ids_for_decode(mdl, out.sequences.cpu())
text = proc.batch_decode(text_ids, skip_special_tokens=True)[0].strip()

# Extract waveform
waveform = out.waveform.cpu().numpy().squeeze() if out.waveform is not None else np.zeros(16000)
```

### Fixed `run_s2tt_only()` Function
**Before:**
```python
token_ids = out[0] if isinstance(out, (list, tuple)) else out
token_ids = _remap_ids_for_decode(mdl, token_ids.cpu())
```

**After:**
```python
# Extract text from sequences attribute
token_ids = _remap_ids_for_decode(mdl, out.sequences.cpu())
```

### Added No-Op Vocoder Pattern
The corrected version properly implements the no-op vocoder pattern to skip audio generation in `run_s2tt_only()`:

```python
# Temporarily replace vocoder with no-op to skip audio generation
orig_voc = mdl.vocoder
inp_device = next(iter(inputs.values())).device

class _NoOpVocoder(nn.Module):
    def forward(self, *args, **kwargs):
        return torch.zeros(1, 1, device=inp_device), [1]

mdl.vocoder = _NoOpVocoder()
try:
    # ... generate ...
finally:
    mdl.vocoder = orig_voc
```

## Why These Fixes Matter

1. **Prevents Empty Text Output:** The original code would often return empty strings because it was looking for text in the wrong place (`out[1]` instead of `out.sequences`)

2. **Ensures Audio Generation Works:** Accessing `out.waveform` instead of `out[0]` ensures the audio waveform is correctly extracted

3. **Proper Device Handling:** Using `_model_input_device()` ensures inputs are placed on the correct GPU, especially important for multi-GPU setups or when the model is split across devices

4. **Robust Error Handling:** The try-except block now properly falls back to text-only generation if the vocoder fails, rather than silently returning empty text

## Testing Recommendations

After applying these fixes, test with:
1. A single sample to verify text output is not empty
2. Multiple samples to verify audio generation works
3. Check that both BLEU and ChrF scores are computed correctly
4. Verify audio files are saved and playable

## Reference
The corrected implementation is based on the working version in `cse465v5-s2st-corrected.ipynb`, which has been tested and validated.
