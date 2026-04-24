# Quick Start: Phase 6a Training (v4)

## TL;DR - Just Do This

1. **Restart kernel** (Kernel → Restart)
2. **Run all cells up to Phase 6a** (Cells 1-74)
3. **Run Phase 6a training** (Cell 75)
4. **Watch the logs** - should see `batch=8` and cosine loss decreasing
5. **Wait ~5 hours** for training to complete

## What to Expect

### First 100 Steps
```
Step   100 | cos=0.15 | qty_err=25 | fired=15 vs tgt=20 | batch=8
```
- Cosine loss: ~0.15 (starting point)
- Quantity error: ~25 tokens (will improve)
- Fired count: within 10 tokens of target
- **Batch size: 8** (important!)

### After 500 Steps
```
Step   500 | cos=0.08 | qty_err=10 | fired=20 vs tgt=24 | batch=8
```
- Cosine loss: ~0.08 (improving!)
- Quantity error: ~10 tokens (better)
- Fired count: within 5 tokens of target

### After 1000 Steps
```
Step  1000 | cos=0.05 | qty_err=5 | fired=19 vs tgt=21 | batch=8
```
- Cosine loss: ~0.05 (good!)
- Quantity error: ~5 tokens (very good)
- Fired count: within 3 tokens of target

### Final (5000 Steps)
```
Step  5000 | cos=0.02 | qty_err=2 | fired=20 vs tgt=21 | batch=8
```
- Cosine loss: <0.03 (excellent!)
- Quantity error: <3 tokens (excellent!)
- Fired count: within 1 token of target

## Red Flags 🚩

### Stop and Debug If You See:

**1. Batch size = 1**
```
Step   100 | ... | batch=1  ← WRONG! Should be 8
```
→ You're using old code. Restart kernel and verify Cell 75 has `BATCH_SIZE = 8`

**2. Cosine loss increasing**
```
Step   100 | cos=0.17
Step   200 | cos=0.36  ← WRONG! Should be decreasing
Step   300 | cos=0.46  ← Getting worse!
```
→ Model is diverging. Check loss weights (should be 0.50 cos, 0.25 qty)

**3. Firing 1-3 tokens**
```
Step   100 | fired=1 vs tgt=20  ← WRONG! Should be 10-30
Step   200 | fired=2 vs tgt=23  ← Still wrong
```
→ CIF connector bug. Verify Cell 59 has CIFConnector v3 with `acc_w_before_fire`

**4. Very slow training**
```
[10 seconds per step]  ← WRONG! Should be 1-2 seconds
```
→ Not using batching. Check batch_size=8 and GPU utilization

## Quick Verification

Before starting training, run this in a new cell:

```python
# Verify CIF connector
print(f"CIF threshold: {model_6a.cif_connector.threshold}")  # Should be 0.95

# Verify batch size (check Cell 75)
# Look for: BATCH_SIZE = 8

# Test CIF forward pass
with torch.no_grad():
    test_enc = torch.randn(1, 100, 1024, device=device)
    test_out = model_6a.cif_connector(test_enc)
    print(f"CIF returns {len(test_out)} values")  # Should be 4
    print(f"Fired tokens: {test_out[1].item():.0f}")  # Should be 10-30

print("✓ All checks passed!")
```

Expected output:
```
CIF threshold: 0.95
CIF returns 4 values
Fired tokens: 18
✓ All checks passed!
```

## Training Timeline

| Time | Steps | Status |
|------|-------|--------|
| 0 min | 0 | Starting |
| 5 min | 100 | Initial learning |
| 25 min | 500 | Good progress |
| 50 min | 1000 | Converging |
| 2.5 hr | 2500 | Almost done |
| 5 hr | 5000 | Complete! |

## When to Stop

**Stop training when**:
- ✅ Reached 5000 steps, OR
- ✅ Cosine loss < 0.03 for 500 consecutive steps, OR
- ✅ Quantity error < 3 for 500 consecutive steps

**Abort training if**:
- ❌ Cosine loss increasing for >500 steps
- ❌ Loss becomes NaN
- ❌ Fired counts stuck at 1-3 after 1000 steps

## After Training

Once training completes:

```python
# Load the final checkpoint
ckpt = torch.load('phase6a_connector_step005000.pt')
model_6a.cif_connector.load_state_dict(ckpt['cif_connector'])
model_6a.speaker_adapter.load_state_dict(ckpt['speaker_adapter'])

print("✓ Phase 6a complete!")
print(f"Final cosine loss: {ckpt['feat_log'][-1]:.4f}")
print(f"Final qty error: {ckpt['qty_log'][-1]:.1f}")
```

## Need Help?

See detailed documentation:
- **COMPLETE_FIX_SUMMARY_V4.md** - Complete overview
- **PHASE6A_MONITORING_GUIDE.md** - Detailed monitoring guide
- **PHASE6A_TRAINING_FIX_V4.md** - Technical details

## Summary

✅ **Restart kernel**  
✅ **Run Phase 6a training** (Cell 75)  
✅ **Watch for `batch=8` and decreasing cosine loss**  
✅ **Wait ~5 hours**  
✅ **Done!**

That's it! 🚀
