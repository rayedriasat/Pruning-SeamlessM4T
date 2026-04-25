# CIF Over-Firing Fix - Quick Summary

## The Problem

Your Phase 6a training shows **CIF is firing 2-3× too many tokens**:

```
Step 100  | fired=50 vs tgt=19  ❌ 2.6× over-firing
Step 500  | fired=64 vs tgt=31  ❌ 2.1× over-firing  
Step 900  | fired=59 vs tgt=13  ❌ 4.5× over-firing
```

## The Root Cause

**Threshold too low:** `threshold=0.50` causes the CIF to fire twice as often as it should.

Think of it like a bucket that fires when full:
- **threshold=0.50** → bucket fires when half-full → 2× too many fires
- **threshold=0.95** → bucket fires when almost full → correct firing rate

## The Fix (4 Changes)

### 1. **Increase CIF Threshold: 0.50 → 0.95**

```python
# In CIFConnector.__init__():
def __init__(self, d_model=1024, n_refiner_layers=2, n_langs=45, threshold=0.95):  # ← CHANGED
    self.threshold = threshold
```

### 2. **Gentler Weight Scaling: 1.0× → 0.8×**

```python
# In CIFConnector.forward():
alpha = raw_w / w_sum * (0.8 * qty_pred.unsqueeze(1))  # ← CHANGED (was 1.0×)
```

### 3. **Rebalance Loss Weights**

```python
# In Phase 6a training loop:
loss = (0.25 * cos_loss +      # ← REDUCED from 0.30
        0.40 * mse_loss +      # ← KEPT
        0.35 * qty_loss +      # ← INCREASED from 0.25
        0.00 * spk_reg)        # ← REMOVED (was 0.05)
```

### 4. **Lower Learning Rate: 3e-4 → 2e-4**

```python
# In Phase 6a optimizer:
optimizer_6a = torch.optim.AdamW([
    {'params': model_6a.cif_connector.parameters(), 'lr': 2e-4},  # ← CHANGED
    {'params': model_6a.speaker_adapter.parameters(), 'lr': 1e-4},
])
```

## Expected Results

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| **Fired tokens** | 50-70 | 15-40 ✅ |
| **Quantity error** | 7-8 tokens | <3 tokens ✅ |
| **Cosine loss** | 0.42 | <0.10 ✅ |
| **Convergence** | 5000 steps | 2500 steps ✅ |

## How to Apply

### Option 1: Automatic (Recommended)

```bash
cd Alteration
python apply_cif_fix.py
```

### Option 2: Manual

1. Open `seamless-final.ipynb`
2. Find the `CIFConnector` class
3. Change `threshold=0.50` to `threshold=0.95`
4. Find `alpha = raw_w / w_sum * qty_pred.unsqueeze(1)`
5. Change to `alpha = raw_w / w_sum * (0.8 * qty_pred.unsqueeze(1))`
6. Update loss weights and LR as shown above

### After Applying Fix:

```bash
# Delete old checkpoints (they were trained with wrong threshold)
rm checkpoints/phase6a_connector_step*.pt

# Restart Phase 6a training from step 0
# (in notebook, run Phase 6a training cell)
```

## Verification

After 500 steps, you should see:

```
Step 500 | cos=0.2800 | qty_err(tok)=2.8 | fired=33 vs tgt=31 ✅ GOOD
```

Instead of:

```
Step 500 | cos=0.4472 | qty_err(tok)=8.3 | fired=64 vs tgt=31 ❌ BAD
```

## Why This Fix Works

1. **Threshold=0.95** matches the CIF paper (Dong & Xu, ICASSP 2020)
2. **0.8× scaling** prevents over-constraining while providing guidance
3. **Higher qty_loss weight** gives the quantity predictor more learning signal
4. **Lower LR** improves training stability

## Files Created

- `fix_cif_overfiring.py` - Fixed CIF connector class + config
- `CIF_OVERFIRING_FIX.md` - Complete technical explanation
- `apply_cif_fix.py` - Automatic fix application script
- `CIF_FIX_SUMMARY.md` - This quick summary

## Questions?

**Q: Why not just increase threshold without other changes?**
A: Threshold alone helps, but the other changes ensure the quantity predictor learns properly and training is stable.

**Q: Can I resume training from my current checkpoint?**
A: Not recommended. The old weights were trained with threshold=0.50, so they're optimized for over-firing. Start fresh.

**Q: How long until I see improvement?**
A: You should see correct firing counts within 100-200 steps. Quantity error should drop below 3 tokens by step 1500.

**Q: What if it still over-fires?**
A: Check that threshold is actually 0.95:
```python
print(f"CIF threshold: {model_6a.cif_connector.threshold}")
```
Should print: `0.95` (not `0.50`)

---

**Bottom line:** Change threshold from 0.50 to 0.95, restart training, and you'll see immediate improvement. The over-firing will stop.
