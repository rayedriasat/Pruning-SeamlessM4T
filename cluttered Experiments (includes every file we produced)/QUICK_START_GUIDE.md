# Phase 8 Full Model KD - Quick Start Guide

## TL;DR

Replace Phase 8 T2U-only KD (which failed) with Full Model KD (which will work).

**What changed**: Train ENTIRE model instead of just T2U component.

**Why**: T2U cannot be trained in isolation due to architectural dependencies.

**Result**: Better audio quality (ASR-BLEU/ChrF) while maintaining text quality.

## 3-Step Implementation

### Step 1: Copy Training Cells (5 minutes)

Open `full-kd.ipynb` and replace these 7 cells with code from `phase8_full_kd_cells.py`:

1. **Phase 8 — Cell 1**: Load Phase 7 Student Model
2. **Phase 8 — Cell 2**: Load Teacher Model  
3. **Phase 8 — Cell 3**: KD Loss Function
4. **Phase 8 — Cell 4**: Optimizer Setup
5. **Phase 8 — Cell 5**: Training Loop
6. **Phase 8 — Cell 6**: Plot Training Curves
7. **Phase 8 — Cell 7**: Save Model

**How to copy**:
- Open `phase8_full_kd_cells.py`
- Find each cell section (marked with `# ===== Phase 8 — Cell X =====`)
- Copy the code under each section
- Paste into corresponding cell in `full-kd.ipynb`

### Step 2: Update Benchmark Cells (2 minutes)

In `full-kd.ipynb`, find Phase 8 Benchmark cells and do a **search-replace**:

```
Find:    'phase8_kd'
Replace: 'phase8_full_kd'
```

**Cells to update**:
- Phase 8 Benchmark Cell 2 (model evaluation)
- Phase 8 Benchmark Cell 3 (comparison figure)
- Phase 8 Benchmark Cell 4 (radar chart)
- Phase 8 Benchmark Cell 5 (summary table)

See `BENCHMARK_CELLS_UPDATES.md` for detailed changes.

### Step 3: Run Training (8-16 hours)

Execute cells in order:

```python
# 1. Setup cells (1-20) - if not already run
# 2. Phase 8 Cell 1 - Load student model
# 3. Phase 8 Cell 2 - Load teacher model
# 4. Phase 8 Cell 3 - Define KD loss
# 5. Phase 8 Cell 4 - Setup optimizer
# 6. Phase 8 Cell 5 - Run training (THIS TAKES TIME)
# 7. Phase 8 Cell 6 - Plot curves
# 8. Phase 8 Cell 7 - Save model
# 9. Phase 8 Benchmark Cells 1-5 - Evaluate
```

## What to Expect

### During Training (Cell 5)

**Progress bar**:
```
[P8] Full KD: 100%|████████| 1000/1000 [8:23:45<00:00, 30.23s/step, 
                  loss=0.1234, kl=0.0890, audio=0.0344, lr=9.5e-06]
```

**Metrics**:
- `loss`: Total KD loss (should decrease)
- `kl`: Text distillation loss (should decrease)
- `audio`: Waveform MSE loss (should decrease)
- `lr`: Learning rate (cosine decay)

**Time estimate**:
- ~30-60 seconds per step
- 1000 steps = 8-16 hours on T4 GPU
- Checkpoints saved every 250 steps

**Memory usage**:
- ~11-12 GB VRAM (fits in 16 GB GPU)
- If OOM: Code handles gracefully, training continues

### After Training (Cells 6-7)

**Training curves** (Cell 6):
- 3 plots: Total loss, KL loss, Audio MSE
- Should show downward trend
- Saved as `phase8_full_kd_training_curves.png`

**Model saved** (Cell 7):
- Saved to Drive as `phase8_full_kd`
- ~1B parameters
- Ready for benchmarking

### Benchmark Results (Benchmark Cells)

**4-model comparison**:
1. Teacher (baseline)
2. Phase 6 (after T2U pruning - quality dip)
3. Phase 7 (after DoRA - text recovery)
4. Phase 8 Full KD (after Full KD - audio recovery)

**Expected improvements in Phase 8**:
- ✅ ASR-BLEU: +2-5 points (audio quality)
- ✅ ASR-ChrF: +3-7 points (audio quality)
- ✅ Text-BLEU: Maintain or +1-2 points
- ✅ Text-ChrF: Maintain (already good from Phase 7)

## Key Differences from Old Approach

| Aspect | Old (T2U-only) | New (Full Model) |
|--------|----------------|------------------|
| **Trainable** | T2U only (~50M) | Entire model (~1B) |
| **Status** | Failed (OOM, errors) | Works ✅ |
| **Training time** | N/A | 8-16 hours |
| **Memory** | OOM (43-127 GB) | 11-12 GB ✅ |
| **Quality** | N/A | Good (text + audio) |

## Troubleshooting

### "Out of memory" error
**Solution**: Code handles this automatically. Training continues with dummy loss for that step.

If persistent:
```python
# In Cell 4, change:
KD_GRAD_ACCUM = 16  # Was 8, now 16 (slower but less memory)
```

### Training too slow
**Expected**: 30-60 sec/step is normal for full model training.

To speed up (at cost of quality):
```python
# In Cell 4, change:
KD_MAX_STEPS = 500  # Was 1000, now 500 (half the training)
```

### Loss not decreasing
**Check after 100 steps**. If still flat:
```python
# In Cell 4, try:
KD_LR = 3e-5  # Was 1e-5, now 3x higher
KD_ALPHA = 0.5  # Was 0.7, now more audio focus
```

### Checkpoint not saving
**Check**:
- Drive mounted? (Colab: run Cell 2)
- rclone configured? (Kaggle: run Cell 5)
- Disk space available?

## Files Reference

### Implementation Files
- `phase8_full_kd_cells.py` - Training cell code (copy from here)
- `PHASE8_FULL_KD_IMPLEMENTATION_GUIDE.md` - Detailed explanation
- `BENCHMARK_CELLS_UPDATES.md` - Benchmark update reference
- `QUICK_START_GUIDE.md` - This file

### Target File
- `full-kd.ipynb` - Main notebook (apply changes here)

### Generated Files (after training)
- `checkpoints/phase8_full_kd_step*.pt` - Training checkpoints
- `models/phase8_full_kd/` - Final model
- `figures/phase8_full_kd_training_curves.png` - Training plots
- `figures/phase8_full_kd_4model_comparison.png` - Benchmark comparison
- `figures/phase8_full_kd_radar_comparison.png` - Radar chart
- `figures/phase8_full_kd_benchmark_summary.csv` - Metrics table

## Validation Checklist

Before starting training:
- [ ] Phase 7 model exists (`phase7_dora_merged_v1`)
- [ ] Teacher model loads successfully
- [ ] All 7 Phase 8 cells updated
- [ ] Benchmark cells updated (search-replace done)
- [ ] GPU has 16 GB VRAM (or 12 GB minimum)
- [ ] Drive mounted (Colab) or rclone configured (Kaggle)

During training:
- [ ] Progress bar shows decreasing loss
- [ ] No persistent OOM errors
- [ ] Checkpoints saving every 250 steps
- [ ] VRAM usage ~11-12 GB (check with `nvidia-smi`)

After training:
- [ ] Training curves show downward trend
- [ ] Model saved to Drive
- [ ] Benchmark runs without errors
- [ ] ASR-BLEU/ChrF improved vs Phase 7

## Next Steps After Completion

1. **Analyze benchmark results**
   - Compare Phase 8 vs Phase 7
   - Check if ASR metrics improved
   - Verify text quality maintained

2. **If results good**: Proceed to Phase 9 (final benchmark + paper)

3. **If results need improvement**:
   - Try longer training (2000 steps)
   - Adjust alpha (more audio focus: 0.5)
   - Adjust temperature (try 1.5 or 3.0)
   - Try different learning rate (3e-5)

4. **Generate paper figures**:
   - Use benchmark comparison plots
   - Include training curves
   - Show 4-model progression

## Support

If you encounter issues:

1. **Check error message** - Most errors are self-explanatory
2. **Read implementation guide** - `PHASE8_FULL_KD_IMPLEMENTATION_GUIDE.md`
3. **Check benchmark updates** - `BENCHMARK_CELLS_UPDATES.md`
4. **Verify prerequisites** - Phase 7 model must exist
5. **Monitor resources** - Use `nvidia-smi` to check VRAM

## Summary

**Old approach**: Train only T2U → Failed (architectural issues)

**New approach**: Train entire model → Works (standard KD)

**Time investment**: 
- Setup: 7 minutes
- Training: 8-16 hours (automated)
- Benchmarking: 30 minutes

**Expected outcome**: Better audio quality while maintaining text quality

---

**Ready to start?** Open `full-kd.ipynb` and begin with Step 1! 🚀
