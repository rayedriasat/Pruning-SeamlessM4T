# CIF Connector Firing Bug Fix (v3)

## Problem
The CIF connector was collapsing and firing only 1 token regardless of input length during Phase 6a training:
- Expected: 15-30 tokens for typical 20-token targets
- Actual: `fired=1` consistently across all steps
- `qty_err(tok)=31.2` (quantity predictor off by ~31 tokens)

## Root Cause
The bug was in the **residual calculation after firing** in the CIF accumulation loop.

### The Buggy Code (v2)
```python
while acc_w >= self.threshold:
    # Fire one token
    fired.append(acc / max(acc_w, 1e-6))
    acc_w -= self.threshold
    # Keep residual for next token
    if acc_w > 0:
        acc = acc * (acc_w / max(acc_w + self.threshold, 1e-6))  # ❌ BUG HERE
    else:
        acc = torch.zeros_like(acc)
```

### Why This Was Wrong
After firing, `acc_w` has already been reduced by `self.threshold`:
- Before firing: `acc_w = 1.2` (example)
- After `acc_w -= self.threshold`: `acc_w = 0.2` (with threshold=1.0)
- Then the code does: `acc * (0.2 / (0.2 + 1.0)) = acc * 0.167`

But this is incorrect! The denominator `acc_w + self.threshold` is adding back the threshold that was just subtracted, which doesn't make mathematical sense.

The correct residual should be: `acc * (0.2 / 1.2) = acc * 0.167` (using the **original** acc_w before firing)

### The Mathematical Error
The residual representation should be proportional to the **leftover weight fraction** of the **original accumulated weight**:

```
residual = acc * (leftover_weight / original_weight)
         = acc * (acc_w_after / acc_w_before)
```

But the buggy code was doing:
```
residual = acc * (acc_w_after / (acc_w_after + threshold))
```

This is wrong because it's not using the original weight before firing.

## The Fix (v3)

### Corrected Code
```python
while acc_w >= self.threshold:
    # Fire one token with the accumulated representation
    fired.append(acc.clone())
    
    # CRITICAL FIX: Proper residual calculation
    # After firing, we have leftover weight = acc_w - threshold
    # The residual representation should be proportional to this leftover
    acc_w_before_fire = acc_w  # ✓ Store original weight
    acc_w -= self.threshold
    
    if acc_w > 1e-6:
        # Keep residual proportional to leftover weight
        acc = acc * (acc_w / acc_w_before_fire)  # ✓ Correct ratio
    else:
        # No significant residual
        acc = torch.zeros_like(acc)
        acc_w = 0.0
```

### Key Changes
1. **Store `acc_w_before_fire`** before reducing by threshold
2. **Use the correct ratio**: `acc_w / acc_w_before_fire` instead of `acc_w / (acc_w + threshold)`
3. **Clone the accumulator** when firing: `fired.append(acc.clone())` to avoid reference issues

## Expected Results After Fix

After applying this fix and restarting Phase 6a training, you should see:

1. **Multiple tokens firing**: `fired=15-30` for typical 20-token targets (not just 1)
2. **Quantity error decreasing**: `qty_err(tok)` should drop below 10 after 1000 steps
3. **Cosine loss improving**: `cos` loss should drop below 0.03 by step 2000
4. **Proper CIF behavior**: The connector should fire approximately `qty_pred` tokens

## How to Apply

The fix has been automatically applied to `Alteration/seamless-final.ipynb`:
- **Backup saved**: `Alteration/seamless-final.ipynb.backup_before_v3_fix`
- **Cell modified**: Cell 59 (CIFConnector class)
- **Version**: v3 (with proper residual calculation)

### Next Steps
1. **Restart the notebook kernel** to reload the corrected CIFConnector class
2. **Re-run Phase 6a training** from the beginning
3. **Monitor the logs** for `fired=` values - should now be 15-30 instead of 1
4. **Verify convergence**: `qty_err` should decrease and `cos` loss should improve

## Technical Details

### Why This Bug Caused Collapse
The incorrect residual calculation meant that after firing one token, the residual was too large. This prevented the accumulator from reaching the threshold again, causing the CIF to fire only once per sequence regardless of length.

### CIF Algorithm Recap
The Continuous Integrate-and-Fire (CIF) algorithm:
1. Predicts per-frame weights `alpha[t]` that sum to the target quantity
2. Accumulates weights: `acc_w += alpha[t]`
3. When `acc_w >= threshold`, fires a token
4. Keeps the residual: `acc_w -= threshold` and proportionally reduces `acc`
5. Continues until all frames are processed

The residual handling is critical - it allows multiple tokens to fire from the accumulated weight.

## Files Modified
- `Alteration/seamless-final.ipynb` (Cell 59: CIFConnector class)
- Backup: `Alteration/seamless-final.ipynb.backup_before_v3_fix`

## Related Issues
- Previous fix (v2) addressed: Sigmoid vs Softplus, threshold value (0.95), weight scaling
- This fix (v3) addresses: Residual calculation after firing
