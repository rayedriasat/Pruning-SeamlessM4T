# How to Add Phase 8 to Your Notebook

## 🚀 Quick Integration (5 Minutes)

### Step 1: Open Your Phase 7 Notebook

Open `only-p7-cse465v5-s2st-corrected.ipynb` in Kaggle or Colab.

### Step 2: Scroll to the End

Find the last cell in Phase 7 (should be Cell 13: "Load merged model in future sessions").

### Step 3: Add Phase 8 Cells

Copy all cells from `phase8_cells.py` and paste them after Phase 7 Cell 13.

You should now have:
- **Cells 1-23**: Setup (from Phase 7)
- **Cells 24-34**: Phase 7 training (from Phase 7)
- **Cells 35-44**: Phase 8 training (NEW - from `phase8_cells.py`)

### Step 4: Run Phase 8

```python
# Run cells sequentially:
# Cell 35: Load Phase 7 model (1 min)
# Cell 36: Load unit labels (30 sec)
# Cell 37: T2U data prep (instant)
# Cell 38: Freeze encoders (instant)
# Cell 39: Verification (30 sec)
# Cell 40: Training loop (35 min) ← MAIN TRAINING
# Cell 41: Loss plot (10 sec)
# Cell 42: Save model (2 min)
# Cell 43: Benchmark (10 min)
# Cell 44: Final results (30 sec)
```

**Total time**: ~50 minutes

## 📋 Cell Mapping

| Phase 8 Cell | Description | Time | Can Skip? |
|--------------|-------------|------|-----------|
| 35 | Load Phase 7 model | 1 min | ❌ No |
| 36 | Load unit labels | 30 sec | ❌ No |
| 37 | T2U data prep | instant | ❌ No |
| 38 | Freeze encoders | instant | ❌ No |
| 39 | Verification | 30 sec | ⚠️ Recommended |
| 40 | Training loop | 35 min | ❌ No |
| 41 | Loss plot | 10 sec | ✅ Yes |
| 42 | Save model | 2 min | ❌ No |
| 43 | Benchmark | 10 min | ⚠️ Recommended |
| 44 | Final results | 30 sec | ✅ Yes |

## ⚙️ Configuration

### Default Settings (Good for Most Cases)
```python
MAX_STEPS  = 1000
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR         = 5e-5
```

### If You Have Limited Time
```python
MAX_STEPS  = 500   # Minimum for decent results
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR         = 5e-5
```

### If You Have OOM Errors
```python
MAX_STEPS  = 1000
BATCH_SIZE = 1     # Reduce batch size
GRAD_ACCUM = 8     # Increase accumulation
LR         = 5e-5
```

### If You Want Best Quality
```python
MAX_STEPS  = 2000  # More training
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR         = 3e-5  # Lower LR for stability
```

## 🎯 What to Watch During Training

### Cell 39 (Verification)
```
✓ T2U forward pass successful!
  Text hidden shape: torch.Size([2, 45, 1024])
  Unit labels shape: torch.Size([2, 187])
  Loss value:        7.2341
```
- **Loss should be 6-8**: If >10 or <4, something is wrong
- **Shapes should match**: Text ~20-100, Units ~50-300

### Cell 40 (Training)
```
Step   25/1000  Loss=7.2341  LR=4.88e-05  Time=0.8min
Step   50/1000  Loss=6.1234  LR=4.75e-05  Time=1.6min
Step  100/1000  Loss=4.8765  LR=4.50e-05  Time=3.3min
```
- **Loss should decrease**: From ~7-8 to ~2-3
- **No NaN/Inf**: If you see these, stop and reduce LR
- **Time per step**: ~2 seconds on T4

### Cell 43 (Benchmark)
```
Phase 8  ChrF (txt):  64.8  (maintained)
Phase 8  ASR-BLEU:    18.5  (audio recovery: +15.3)
Phase 8  ASR-ChrF:    38.2
```
- **ASR-BLEU > 15**: Good audio quality
- **ASR-ChrF > 35**: Good audio quality
- **txt-ChrF ~65**: Text quality maintained

## 🐛 Troubleshooting

### "Phase 7 model not found"
**Fix**: Make sure Phase 7 Cell 11 (merge adapters) and Cell 12 (save model) completed successfully.

```python
# Check if model exists
import os
print(os.path.exists(f"{MODEL_DIR}/phase7_dora_merged"))
```

### "Unit cache not found"
**Fix**: Re-run Phase 7 Cell 6 to extract unit labels.

```python
# Check if cache exists
import os
print(os.path.exists(f"{CKPT_DIR}/unit_labels_cache.pt"))
```

### "T2U forward pass fails"
**Fix**: Check unit sequence lengths and filter invalid ones.

```python
# Add this before Cell 39
lens = [p["units"].numel() for p in ft_s2st_pairs]
print(f"Unit lengths: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.0f}")
ft_s2st_pairs = [p for p in ft_s2st_pairs if 3 <= p["units"].numel() <= 500]
print(f"After filtering: {len(ft_s2st_pairs)} pairs")
```

### "Out of memory"
**Fix**: Reduce batch size in Cell 40.

```python
# Change these lines in Cell 40
BATCH_SIZE = 1     # Was 2
GRAD_ACCUM = 8     # Was 4
```

### "Loss not decreasing"
**Fix**: Check learning rate and gradient clipping.

```python
# Change these lines in Cell 40
LR = 1e-5          # Was 5e-5 (lower)
GRAD_CLIP = 0.5    # Was 1.0 (stricter)
```

### "ASR-BLEU still low (<10)"
**Fix**: Train longer or check unit labels.

```python
# Option 1: Train longer
MAX_STEPS = 2000   # Was 1000

# Option 2: Check unit labels
print("Sample units:", unit_labels[0][:20])
print("Should be integers in range [0, 10000]")
```

## ✅ Success Checklist

Before starting Phase 8:
- [ ] Phase 7 training complete (Cell 9 finished)
- [ ] Phase 7 model merged (Cell 11 finished)
- [ ] Phase 7 model saved (Cell 12 finished)
- [ ] Unit labels cached (Phase 7 Cell 6 finished)
- [ ] GPU has >12 GB VRAM free

During Phase 8:
- [ ] Cell 39 verification passed
- [ ] Cell 40 loss decreasing smoothly
- [ ] No OOM or NaN errors
- [ ] Checkpoints saving every 100 steps

After Phase 8:
- [ ] Final loss < 3.0
- [ ] ASR-BLEU > 15
- [ ] Audio samples sound good
- [ ] Model saved to Drive

## 🎉 You're Done!

If all checkboxes are checked, your compressed SeamlessM4T model is ready for deployment!

**Final model specs**:
- **Size**: ~1.1 GB (52% compression)
- **Speed**: 2-3x faster inference
- **Quality**: 90% text, 75% audio retention
- **Languages**: EN→BN (extensible to other pairs)

---

## 📞 Need Help?

1. Read `PHASE8_README.md` for detailed troubleshooting
2. Check `PHASE8_T2U_NAR_TRAINING.md` for conceptual explanations
3. Review error messages in training logs
4. Verify all prerequisites are met

**Good luck with Phase 8!** 🚀
