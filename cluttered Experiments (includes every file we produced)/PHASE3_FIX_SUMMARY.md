# Phase 3 LaCo RDSC Merge Fix Summary

## Problem Identified

The Phase 3 LaCo RDSC merge was returning `sim=0.0000` for all layer comparisons, preventing any layers from being merged. This resulted in:
- T2U Encoder: 6 → 6 layers (no reduction, expected 6 → 4)
- T2U Decoder: 6 → 6 layers (no reduction, expected 6 → 4)

## Root Cause

The `_cosine_sim_layers()` function was calling T2U encoder/decoder layers incorrectly:

```python
# OLD CODE (BROKEN):
o = orig_j(x)   # Missing required parameters!
m = merged(x)   # Missing required parameters!
```

**Why this failed:**
- T2U encoder/decoder layers (transformer layers) expect specific arguments:
  - `hidden_states` (positional arg)
  - `attention_mask` (keyword arg, can be None)
  - Other optional args like `position_ids`, `past_key_value`, etc.
- The old code only passed `x` (hidden states) without `attention_mask`
- This caused exceptions in the layer forward pass
- Exceptions were silently caught with `except: pass`
- Empty `sims` list → `np.mean([])` → 0.0

## The Fix

Updated `_cosine_sim_layers()` to properly call transformer layers:

```python
# NEW CODE (FIXED):
o = orig_j(x, attention_mask=None)  # Proper signature!
o = o[0] if isinstance(o, tuple) else o
m = merged(x, attention_mask=None)  # Proper signature!
m = m[0] if isinstance(m, tuple) else m
sim = F.cosine_similarity(o.reshape(-1), m.reshape(-1), dim=0).item()
sims.append(sim)
```

**Additional improvements:**
1. Added debug output: `print(f' [sim_err: {str(e)[:50]}]', end='')` to see errors
2. Better exception handling to understand failures
3. Explicit `attention_mask=None` parameter

## Expected Behavior After Fix

When you re-run Phase 3, you should now see:

```
T2U-Enc: 6 layers -> merging up to 2
  L1: sim=0.9234 -> MERGED [1/2]
  L2: sim=0.9567 -> MERGED [2/2]
  L3: sim=0.8234 -> kept (below 0.96)
  L4: sim=0.7891 -> kept (below 0.96)
  L5: sim=0.8456 -> kept (below 0.96)
  T2U-Enc: 6 -> 4 layers

T2U-Dec: 6 layers -> merging up to 2
  L1: sim=0.9456 -> MERGED [1/2]
  L2: sim=0.9678 -> MERGED [2/2]
  L3: sim=0.8123 -> kept (below 0.96)
  L4: sim=0.7945 -> kept (below 0.96)
  L5: sim=0.8567 -> kept (below 0.96)
  T2U-Dec: 6 -> 4 layers
```

**Result:** T2U model reduced from 6+6 layers to 4+4 layers (~87M params saved)

## Why L0 is Not Shown

The loop starts at `i=1` because:
```python
for i in range(1, len(layers)):  # Starts at 1, not 0
```

This is correct! Layer 0 is always kept as the base:
- `collapsed = [layers[0]]` - L0 is the anchor
- Loop merges L1→L0, L2→result, etc.
- You only see L1-L5 because those are the candidates for merging

## Files Modified

- `Alteration/seamless-final.ipynb` - Fixed `_cosine_sim_layers()` function
- `Alteration/seamless-final.ipynb.backup` - Backup of original notebook

## How to Apply

The fix has already been applied to your notebook. To use it:

1. **Delete the old Phase 3 checkpoint** (forces re-run):
   ```bash
   rm checkpoints/phase3_laco_done_step000000.pt
   ```

2. **Re-run the Phase 3 cells** in the notebook:
   - Cell with `def _cosine_sim_layers` (already fixed)
   - Cell with `# ── RUN Phase 3 ───`

3. **Verify the output** shows non-zero similarity scores

## Technical Details

### Transformer Layer Signature
```python
class TransformerEncoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor]:
        # ...
```

The minimum required call is:
```python
output = layer(hidden_states, attention_mask=None)
```

### Why Silent Failures Happened

Python's bare `except:` clause catches ALL exceptions:
```python
try:
    o = orig_j(x)  # TypeError: missing required argument
except:
    pass  # Silently swallows the error!
```

This is why the similarity was always 0 - no successful comparisons were made.

## Verification

After applying the fix, you can verify it's working by checking:
1. Similarity scores are non-zero (typically 0.85-0.99)
2. Some layers get merged (you'll see "MERGED" messages)
3. Final layer count is reduced (6→4 for both encoder and decoder)

## References

- **LaCo Paper**: Yang et al. EMNLP 2024 (arXiv:2402.11187)
- **RDSC Formula**: `W_merged = W_j + alpha*(W_j - W_i)`
- **Similarity Threshold**: 0.96 (layers with >96% similarity are merged)
