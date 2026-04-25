# Phase 6a Training Issues - Root Cause Analysis

## Current Status (Step 800)
- **Cosine loss: 0.4082** (target: < 0.10)
- **Progress: VERY SLOW** (only 0.09 drop in 800 steps)
- **Fired tokens: 36-50** (targets: 19-35) - **OVER-FIRING**

## Root Causes Identified

### 1. **CIF Threshold Still Too High** ❌
**Current:** `threshold=0.70`
**Problem:** Still causing over-firing (36-50 tokens vs target 19-35)
**Fix:** Lower to `threshold=0.50`

### 2. **Loss Weights Suboptimal** ❌
**Current:**
```python
loss = (0.50 * cos_loss +      # Cosine (direction)
        0.20 * mse_loss +      # MSE (magnitude)
        0.20 * qty_loss +      # Quantity
        0.10 * spk_reg)        # Speaker
```

**Problem:** 
- Cosine weight (0.50) is TOO HIGH - dominates gradient, prevents convergence
- MSE weight (0.20) is TOO LOW - magnitude alignment is critical
- Quantity weight (0.20) is TOO LOW - CIF needs strong qty signal

**Fix:**
```python
loss = (0.30 * cos_loss +      # REDUCED - let other losses help
        0.40 * mse_loss +      # INCREASED - magnitude is key
        0.25 * qty_loss +      # INCREASED - CIF needs this
        0.05 * spk_reg)        # REDUCED - less important
```

### 3. **Learning Rate Too High** ❌
**Current:** `lr=5e-4` for connector
**Problem:** With high cosine weight, this causes oscillation
**Fix:** `lr=3e-4` (more stable with new loss weights)

### 4. **Scheduler Too Aggressive** ❌
**Current:** `CosineAnnealingWarmRestarts(T_0=1000, T_mult=2)`
**Problem:** LR drops too fast (already at 4.88e-04 → 5.50e-05 by step 800)
**Fix:** Use `CosineAnnealingLR` with longer T_max

### 5. **BATCH_ACCUM=2 Unnecessary** ⚠️
**Current:** Effective batch size = 8×2 = 16
**Problem:** Too large, slows convergence
**Fix:** Set to 1 (batch_size=8 is enough)

## Complete Fix

### Fix 1: CIF Threshold
```python
# In CIFConnector class __init__
def __init__(self, d_model=1024, n_refiner_layers=2, n_langs=45, threshold=0.50):  # CHANGED from 0.70
```

### Fix 2: Loss Weights
```python
# In Phase 6a training loop
loss = (0.30 * cos_loss +      # Direction (REDUCED from 0.50)
        0.40 * mse_loss +      # Magnitude (INCREASED from 0.20) - KEY FIX
        0.25 * qty_loss +      # Quantity (INCREASED from 0.20)
        0.05 * spk_reg)        # Speaker (REDUCED from 0.10)
```

### Fix 3: Learning Rate
```python
optimizer_6a = torch.optim.AdamW([
    {'params': model_6a.cif_connector.parameters(),   'lr': 3e-4, 'weight_decay': 0.01},  # REDUCED from 5e-4
    {'params': model_6a.speaker_adapter.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},  # Keep same
], betas=(0.9, 0.98))
```

### Fix 4: Scheduler
```python
scheduler_6a = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_6a, T_max=5000, eta_min=1e-5)  # CHANGED from WarmRestarts
```

### Fix 5: Batch Accumulation
```python
BATCH_ACCUM = 1  # CHANGED from 2
```

## Expected Results After Fix

| Steps | Cosine Loss | Fired Tokens | Status |
|-------|-------------|--------------|--------|
| 0-500 | 0.50 → 0.25 | 30-40 → 25-35 | Rapid drop |
| 500-1500 | 0.25 → 0.12 | 25-35 → 22-30 | Steady |
| 1500-3000 | 0.12 → 0.06 | 22-30 → 20-28 | Converging |
| 3000-5000 | 0.06 → 0.04 | 20-28 → 18-25 | Fine-tuning |

## Why This Will Work

1. **Lower threshold (0.50)** → More tokens fired → Better alignment with targets
2. **Higher MSE weight (0.40)** → Stronger magnitude matching → Faster convergence
3. **Balanced loss** → No single loss dominates → Stable training
4. **Lower LR (3e-4)** → More stable updates → Less oscillation
5. **Better scheduler** → Gradual decay → Consistent progress

## Action Items

1. ✅ Delete all `phase6a*.pt` checkpoints
2. ✅ Apply all 5 fixes above
3. ✅ Restart training from step 0
4. ✅ Monitor first 500 steps - should see cosine < 0.25

## Critical: Why MSE Weight Matters

The teacher's T2U input embeddings have **both direction AND magnitude** information.
- **Cosine loss** only matches direction (angle between vectors)
- **MSE loss** matches magnitude (vector norms)

**Current problem:** With cosine=0.50 and MSE=0.20, the model learns direction but ignores magnitude.
**Solution:** MSE=0.40 forces the model to match both, leading to much better feature alignment.

This is why cosine is stuck at 0.40 - the model is learning direction, but the magnitude mismatch prevents full convergence.
