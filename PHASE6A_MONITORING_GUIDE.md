# Phase 6a Training Monitoring Guide

## Quick Health Check

### ✅ GOOD Training (What You Want to See)

```
Step   100 | cos=0.15 | qty_err=25 | fired=15 vs tgt=20 | batch=8
Step   200 | cos=0.12 | qty_err=20 | fired=18 vs tgt=23 | batch=8  ← Improving
Step   300 | cos=0.10 | qty_err=15 | fired=22 vs tgt=24 | batch=8  ← Better
Step   500 | cos=0.08 | qty_err=10 | fired=20 vs tgt=24 | batch=8  ← Good!
Step  1000 | cos=0.05 | qty_err=5  | fired=19 vs tgt=21 | batch=8  ← Excellent!
```

**Signs of healthy training**:
- ✓ Cosine loss **decreasing** over time
- ✓ Quantity error **decreasing** (30 → 5 tokens)
- ✓ Fired counts **within 5 tokens** of target
- ✓ Batch size = 8 (not 1)
- ✓ Training speed: ~1-2 seconds per step

### ❌ BAD Training (Problems)

```
Step   100 | cos=0.17 | qty_err=30 | fired=2  vs tgt=20 | batch=1
Step   200 | cos=0.36 | qty_err=30 | fired=3  vs tgt=23 | batch=1  ← Diverging!
Step   300 | cos=0.46 | qty_err=28 | fired=3  vs tgt=24 | batch=1  ← Worse!
Step   500 | cos=0.49 | qty_err=26 | fired=5  vs tgt=24 | batch=1  ← Still bad!
```

**Signs of problems**:
- ❌ Cosine loss **increasing** (diverging!)
- ❌ Quantity error **not decreasing**
- ❌ Fired counts **way off** (2-5 vs 20-24)
- ❌ Batch size = 1 (should be 8)
- ❌ Training speed: ~5-10 seconds per step (too slow)

## Metric Targets by Step

| Step | Cosine Loss | Qty Error | Fired vs Target | Status |
|------|-------------|-----------|-----------------|--------|
| 100 | 0.10-0.20 | 20-30 | ±10 tokens | Starting |
| 500 | 0.05-0.10 | 10-20 | ±5 tokens | Learning |
| 1000 | 0.03-0.07 | 5-10 | ±3 tokens | Good |
| 2000 | 0.02-0.05 | 2-5 | ±2 tokens | Very good |
| 5000 | <0.03 | <3 | ±1 token | Excellent |

## What Each Metric Means

### Cosine Loss (`cos=`)
- **What it measures**: Direction alignment between CIF output and teacher embeddings
- **Range**: 0.0 (perfect) to 2.0 (opposite directions)
- **Target**: Should **decrease** from ~0.15 to <0.03
- **Problem if**: Increasing or stuck above 0.20

### Quantity Error (`qty_err(tok)=`)
- **What it measures**: Absolute difference between predicted and actual token count
- **Range**: 0 (perfect) to 50+ (very wrong)
- **Target**: Should **decrease** from ~30 to <5
- **Problem if**: Not decreasing or stuck above 20

### Fired vs Target (`fired=X vs tgt=Y`)
- **What it measures**: How many tokens CIF actually fired vs expected
- **Target**: Should be **within ±5 tokens** of target
- **Problem if**: Consistently off by >10 tokens or firing only 1-3 tokens

### Batch Size (`batch=`)
- **What it measures**: How many samples processed per step
- **Target**: Should be **8**
- **Problem if**: Shows 1 (means batching didn't work)

### Learning Rate (`lr=`)
- **What it measures**: Current learning rate (with cosine schedule)
- **Range**: Starts at 1e-4, decays to 1e-5
- **Target**: Should gradually decrease
- **Problem if**: Stuck at very low value (<1e-6)

## Troubleshooting

### Problem: Cosine Loss Increasing

**Symptoms**:
```
cos=0.17 → 0.36 → 0.46 → 0.49  (getting worse!)
```

**Possible causes**:
1. Learning rate too high (but 1e-4 should be fine)
2. Gradient explosion (check for NaN)
3. Loss weights imbalanced (should be 0.50 cos, 0.25 qty)

**Solutions**:
- Check for NaN in gradients: `torch.isnan(loss).any()`
- Verify loss weights in code: `0.50 * cos_loss + 0.25 * qty_loss`
- Try lower LR: 5e-5 instead of 1e-4

### Problem: Quantity Not Improving

**Symptoms**:
```
qty_err=30 → 30 → 28 → 26  (barely changing)
```

**Possible causes**:
1. Quantity loss weight too low
2. CIF threshold wrong (should be 0.95)
3. Weight predictor not learning

**Solutions**:
- Verify qty weight: should be 0.25 (not 0.10)
- Check CIF threshold: `model_6a.cif_connector.threshold` should be 0.95
- Increase qty weight to 0.30 if still not learning

### Problem: Firing Only 1-3 Tokens

**Symptoms**:
```
fired=1 vs tgt=20
fired=2 vs tgt=23
fired=3 vs tgt=24
```

**Possible causes**:
1. CIF connector bug (should be fixed in v3)
2. Threshold too high
3. Alpha weights collapsing

**Solutions**:
- Verify you're using CIFConnector v3 (with `acc_w_before_fire`)
- Check threshold: should be 0.95, not 1.0
- Check alpha_weights: `print(alpha_weights.mean())` should be >0.3

### Problem: Batch Size = 1

**Symptoms**:
```
batch=1  (should be 8!)
```

**Possible causes**:
1. Using old training code (v3 instead of v4)
2. Not enough valid samples

**Solutions**:
- Verify you're using Phase 6a training v4 (with `BATCH_SIZE = 8`)
- Check valid samples: `print(len(valid_kd))` should be >100
- Restart kernel and re-run training cell

### Problem: Training Very Slow

**Symptoms**:
- 5-10 seconds per step (should be 1-2 seconds)

**Possible causes**:
1. Batch size = 1 (not using batching)
2. CPU bottleneck (audio loading)
3. Not using BF16 autocast

**Solutions**:
- Verify batch_size = 8
- Check GPU utilization: `nvidia-smi` should show >50% usage
- Verify autocast: `with torch.cuda.amp.autocast(dtype=torch.bfloat16):`

## Expected Training Timeline

With the v4 fixes (batch_size=8, proper losses):

| Time | Steps | Cosine Loss | Qty Error | Status |
|------|-------|-------------|-----------|--------|
| 0 min | 0 | 0.20 | 30 | Starting |
| 5 min | 100 | 0.15 | 25 | Initial learning |
| 25 min | 500 | 0.08 | 10 | Good progress |
| 50 min | 1000 | 0.05 | 5 | Converging |
| 2 hr | 2000 | 0.03 | 3 | Very good |
| 5 hr | 5000 | <0.03 | <3 | Complete |

**Total training time**: ~5 hours (with batch_size=8)

Compare to old (batch_size=1): ~40 hours! 🐌

## When to Stop Training

### Early Stopping Criteria (Good Enough)

Stop training if you reach **any** of these:
- ✓ Cosine loss < 0.03 for 500 consecutive steps
- ✓ Quantity error < 3 tokens for 500 consecutive steps
- ✓ Fired counts within ±2 tokens for 500 consecutive steps

### Continue Training If

Keep training if:
- Cosine loss still decreasing
- Quantity error still decreasing
- Haven't reached 5000 steps yet

### Abort Training If

Stop and debug if:
- ❌ Cosine loss increasing for >500 steps
- ❌ Loss becomes NaN
- ❌ Quantity error not decreasing after 1000 steps
- ❌ Fired counts stuck at 1-3 tokens after 1000 steps

## Quick Commands

### Check Current Metrics
```python
# In notebook after training starts
print(f"Recent cosine loss: {np.mean(recent_feat[-50:]):.4f}")
print(f"Recent qty error: {np.mean(recent_qty_abs[-50:]):.1f}")
print(f"Recent total loss: {np.mean(recent_total[-50:]):.4f}")
```

### Check CIF Connector
```python
# Verify CIF threshold
print(f"CIF threshold: {model_6a.cif_connector.threshold}")  # Should be 0.95

# Check a forward pass
with torch.no_grad():
    test_enc = torch.randn(1, 100, 1024, device=device)
    test_out = model_6a.cif_connector(test_enc)
    print(f"CIF output: {len(test_out)} values")  # Should be 4
    print(f"Fired tokens: {test_out[1].item():.0f}")  # Should be 10-30
```

### Check Batch Size
```python
# Should see BATCH_SIZE = 8 in the training cell
# Look for this line:
# BATCH_SIZE = 8
```

### Monitor GPU
```bash
# In terminal
watch -n 1 nvidia-smi
```

Should show:
- GPU utilization: 50-80%
- Memory usage: 10-15 GB
- Temperature: <80°C

## Summary

**Healthy training checklist**:
- ✅ Cosine loss decreasing
- ✅ Quantity error decreasing
- ✅ Fired counts within ±5 tokens
- ✅ Batch size = 8
- ✅ Training speed: 1-2 sec/step
- ✅ GPU utilization: >50%

If all checkboxes are ✅, your training is working correctly! 🎉
