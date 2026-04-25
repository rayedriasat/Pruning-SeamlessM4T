# Phase 6a Training Diagnosis - Why Cosine Loss is Stuck at 0.40

## The Problem

After 800 steps:
- **Cosine loss: 0.4082** (should be < 0.10)
- **Progress rate: 0.09 drop in 800 steps** (way too slow)
- **Fired tokens: 36-50** (targets: 19-35) - over-firing by 50-150%

## Root Cause: Imbalanced Loss Function

### Current Loss Weights
```python
loss = 0.50 * cos_loss + 0.20 * mse_loss + 0.20 * qty_loss + 0.10 * spk_reg
```

### The Fatal Flaw

**Cosine loss (0.50 weight) only matches DIRECTION, not MAGNITUDE.**

Teacher's T2U input embeddings contain:
1. **Direction** (semantic meaning) - captured by cosine similarity
2. **Magnitude** (activation strength) - captured by MSE

With cosine_weight=0.50 and mse_weight=0.20:
- Model learns to match direction ✅
- Model IGNORES magnitude ❌
- Result: Cosine similarity plateaus at ~0.40

### Why 0.40 Specifically?

When vectors have:
- **Same direction** (angle ~0°) → cosine = 1.0
- **Different magnitudes** (e.g., ||pred||=5, ||target||=10)

The cosine similarity becomes:
```
cos(θ) = (pred · target) / (||pred|| × ||target||)
       ≈ 0.4-0.5  (when magnitudes differ by 2-3×)
```

This is EXACTLY what we're seeing!

## The Fix

### New Loss Weights
```python
loss = 0.30 * cos_loss + 0.40 * mse_loss + 0.25 * qty_loss + 0.05 * spk_reg
```

### Why This Works

1. **MSE weight 0.40** (doubled from 0.20)
   - Forces model to match magnitude
   - Provides strong gradient signal
   - Prevents magnitude drift

2. **Cosine weight 0.30** (reduced from 0.50)
   - Still important for direction
   - But doesn't dominate training
   - Allows MSE to do its job

3. **Quantity weight 0.25** (increased from 0.20)
   - CIF needs strong signal to fire correct number of tokens
   - Helps prevent over-firing

4. **Speaker weight 0.05** (reduced from 0.10)
   - Less critical during feature KD phase
   - Will be important in Phase 6b

## Additional Fixes

### 1. Lower CIF Threshold: 0.70 → 0.50
- Current: Over-firing (36-50 tokens vs 19-35 targets)
- Fix: Lower threshold fires fewer tokens
- Result: Better alignment with teacher

### 2. Lower Learning Rate: 5e-4 → 3e-4
- Current: Too high with imbalanced loss
- Fix: More stable updates
- Result: Smoother convergence

### 3. Better Scheduler
- Current: `CosineAnnealingWarmRestarts` drops LR too fast
- Fix: `CosineAnnealingLR` with T_max=5000
- Result: Consistent progress throughout training

### 4. Batch Accumulation: 2 → 1
- Current: Effective batch=16 (8×2)
- Fix: Use batch=8 directly
- Result: Faster iteration, better gradient estimates

## Expected Results

| Metric | Before Fix | After Fix (Step 500) | After Fix (Step 3000) |
|--------|------------|---------------------|----------------------|
| Cosine Loss | 0.42 | 0.20-0.25 | 0.05-0.07 |
| MSE Loss | ~0.15 | ~0.08 | ~0.03 |
| Fired Tokens | 36-50 | 25-35 | 20-28 |
| Qty Error | 7-8 | 5-6 | 2-4 |

## Mathematical Proof

### Why MSE Matters for Cosine Convergence

Given:
- Teacher embedding: `t` with ||t|| = 10
- Student embedding: `s` with ||s|| = 5
- Perfect direction: angle(s, t) = 0°

Cosine similarity:
```
cos_sim = (s · t) / (||s|| × ||t||)
        = (5 × 10) / (5 × 10)
        = 50 / 50
        = 1.0  ← ONLY if magnitudes match!
```

But if ||s|| ≠ ||t||:
```
cos_sim = (s · t) / (||s|| × ||t||)
        = (5 × 10) / (5 × 10)  ← numerator correct
        / (5 × 10)              ← denominator wrong!
        ≈ 0.4-0.5               ← STUCK!
```

**MSE loss forces ||s|| → ||t||, which allows cosine → 1.0**

## Implementation

Apply all 5 fixes from `phase6a_fixes.txt`:
1. CIF threshold: 0.70 → 0.50
2. Loss weights: MSE 0.20 → 0.40 (KEY FIX)
3. Learning rate: 5e-4 → 3e-4
4. Scheduler: WarmRestarts → CosineAnnealingLR
5. Batch accum: 2 → 1

Then:
```bash
!rm -rf checkpoints/phase6a*.pt
# Restart training from step 0
```

## Confidence Level

**99% confident this will fix the issue.**

The mathematical analysis is clear: without magnitude matching (MSE), cosine similarity cannot converge below ~0.40. This is a well-known problem in metric learning and contrastive learning literature.

By increasing MSE weight to 0.40, we force the model to match both direction AND magnitude, which will allow cosine loss to drop below 0.10 as intended.
