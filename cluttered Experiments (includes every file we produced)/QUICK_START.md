# Quick Start: Resume Phase 6 Training

## ✓ Fix Applied Successfully

The CUDA assertion error has been fixed in `AAA/pragmata-recovery.ipynb`.

## What Was Fixed

**Problem**: Re-tokenizing teacher text created invalid position indices  
**Solution**: Use pre-tokenized `teacher_text_sequences` directly from cache

## Resume Training Now

### 1. Upload Fixed Notebook
```bash
# The fixed file is ready:
AAA/pragmata-recovery.ipynb
```

### 2. In Kaggle

1. **Upload** the fixed notebook
2. **Restart kernel** (important!)
3. **Run all cells** up to Phase 6

### 3. Verify It's Working

You should see:
```
[6b1] Text decoder warmup (LoRA only)
  max_audio=20s | trainable=15.60M
  [6b1] step   50/300 | loss=2.xxxx | KD=50% | lr=1.00e-04
```

**No CUDA errors!** ✓

## Your Data is Safe

- ✓ Teacher cache: 9600 entries in Google Drive
- ✓ All checkpoints preserved
- ✓ No need to regenerate anything

## Files Created

1. `AAA/pragmata-recovery.ipynb` - Fixed notebook
2. `AAA/pragmata-recovery.ipynb.backup_before_phase6_fix` - Backup
3. `PHASE6_FIX_SUMMARY.md` - Detailed explanation
4. `PHASE6_FIX.md` - Technical details
5. `apply_phase6_fix.py` - Fix script (already run)
6. `verify_fix.py` - Verification script (already run)

## If You Need to Revert

```bash
# Restore from backup:
cp AAA/pragmata-recovery.ipynb.backup_before_phase6_fix AAA/pragmata-recovery.ipynb
```

## Expected Training Time

- **Stage 6B1**: ~10-15 minutes (300 steps)
- **Stage 6B2**: ~30-40 minutes (900 steps)
- **Stage 6C**: ~25-35 minutes (600 steps)
- **Stage 6D**: ~10-15 minutes (200 steps)

**Total**: ~1.5-2 hours for full Phase 6

## Monitoring

Watch for:
- ✓ Loss decreasing
- ✓ No CUDA errors
- ✓ Checkpoints saving every 300 steps
- ✓ Quick eval ChrF improving

## Need Help?

Check these files:
- `PHASE6_FIX_SUMMARY.md` - Full explanation
- `AAA/planning.md` - Original plan
- `AAA/modeling_seamless_m4t_v2.py` - Model architecture reference

---

**Ready to go!** Upload the fixed notebook and resume training. 🚀
