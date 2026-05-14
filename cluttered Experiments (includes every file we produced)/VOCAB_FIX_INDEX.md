# Vocab Mismatch Fix - Complete Index

## 🚀 Quick Start

**If you just want to fix it NOW**: Read `KAGGLE_CELLS_TO_PASTE.md`

**If you want to understand what happened**: Read `VOCAB_MISMATCH_SUMMARY.md`

**If you want step-by-step instructions**: Read `APPLY_VOCAB_FIX_NOW.md`

---

## 📁 All Solution Files

### 1. **KAGGLE_CELLS_TO_PASTE.md** ⭐ START HERE
- **Purpose**: Ready-to-paste cells for Kaggle
- **Contains**: Complete code cells for both options
- **Use when**: You want to fix it immediately
- **Time to read**: 2 minutes
- **Time to implement**: 5-40 minutes depending on option

### 2. **APPLY_VOCAB_FIX_NOW.md**
- **Purpose**: Complete step-by-step guide
- **Contains**: Detailed instructions for both options
- **Use when**: You want clear guidance with explanations
- **Time to read**: 5 minutes
- **Time to implement**: 5-40 minutes depending on option

### 3. **QUICK_FIX_REFERENCE.md**
- **Purpose**: Quick reference card
- **Contains**: Decision tree, quick commands, test code
- **Use when**: You need a quick reminder or command
- **Time to read**: 1 minute

### 4. **VOCAB_MISMATCH_SUMMARY.md**
- **Purpose**: Complete explanation of what happened
- **Contains**: Debug evidence, root cause, why it happened
- **Use when**: You want to understand the problem deeply
- **Time to read**: 5 minutes

### 5. **SOLUTION_VOCAB_MISMATCH.md**
- **Purpose**: Detailed technical solution
- **Contains**: Both options with technical details
- **Use when**: You want technical depth
- **Time to read**: 10 minutes

### 6. **FINAL_FIX_text_recovery_step.py**
- **Purpose**: Updated function with remapping (Option 1)
- **Contains**: Python code for text_recovery_step
- **Use when**: Implementing Option 1
- **Time to read**: 2 minutes

### 7. **TEST_SOLUTION.md**
- **Purpose**: Test cells to determine which option
- **Contains**: Diagnostic and test code
- **Use when**: You want to test before implementing
- **Time to read**: 3 minutes

### 8. **README_FINAL_SOLUTION.md**
- **Purpose**: Quick overview of both solutions
- **Contains**: Summary and quick start
- **Use when**: You want a brief overview
- **Time to read**: 3 minutes

### 9. **This file (VOCAB_FIX_INDEX.md)**
- **Purpose**: Navigation guide for all files
- **Contains**: Index and reading recommendations

---

## 🎯 Reading Path by Goal

### Goal: Fix it ASAP
1. `KAGGLE_CELLS_TO_PASTE.md` - Paste cells and run
2. Done!

### Goal: Understand and fix
1. `VOCAB_MISMATCH_SUMMARY.md` - Understand the problem
2. `APPLY_VOCAB_FIX_NOW.md` - Follow step-by-step guide
3. Done!

### Goal: Deep understanding
1. `VOCAB_MISMATCH_SUMMARY.md` - What happened
2. `SOLUTION_VOCAB_MISMATCH.md` - Technical details
3. `APPLY_VOCAB_FIX_NOW.md` - Implementation
4. Done!

### Goal: Quick reference
1. `QUICK_FIX_REFERENCE.md` - Commands and decision tree
2. Done!

---

## 🔍 Problem Summary

**What**: Teacher cache has 256K vocab tokens, student model only has 22K vocab

**Why**: Phase 1 pruned vocab to 22K, but Phase 6A cache used base teacher (256K vocab)

**Result**: CUDA assertion error when embedding lookup fails

**Solution**: Either remap tokens (fast) or rebuild cache with student model (better)

---

## ✅ Solution Options

### Option 1: Token Remapping
- **Time**: 5 minutes
- **Requires**: `_vocab_remap_to_old` attribute
- **Pros**: Fast
- **Cons**: Potential information loss, more complex code

### Option 2: Rebuild Cache ⭐ RECOMMENDED
- **Time**: 40 minutes
- **Requires**: Nothing (always works)
- **Pros**: Better quality, simpler, self-consistent
- **Cons**: Takes longer

---

## 📊 Decision Matrix

| Criterion | Option 1 | Option 2 |
|-----------|----------|----------|
| Speed | ✓✓✓ (5 min) | ✗ (40 min) |
| Quality | ✓ (good) | ✓✓✓ (best) |
| Simplicity | ✓ (needs remapping) | ✓✓✓ (no remapping) |
| Availability | ✗ (needs attribute) | ✓✓✓ (always works) |
| Self-consistency | ✗ (learns from base) | ✓✓✓ (learns from self) |
| **Recommended** | If urgent | **YES** |

---

## 🎓 Key Learnings

1. **Vocab consistency matters**: All components must use the same vocabulary
2. **Cache validation**: Always verify cached data matches model expectations
3. **Self-consistency**: Student learning from itself is more principled
4. **Debug first**: The debug cells revealed the exact problem

---

## 📝 Implementation Checklist

### Before Starting
- [ ] Read `VOCAB_MISMATCH_SUMMARY.md` to understand the problem
- [ ] Decide: Option 1 (fast) or Option 2 (better)
- [ ] Open `KAGGLE_CELLS_TO_PASTE.md` for code

### Option 1 Implementation
- [ ] Run Cell 1 to check for remapping info
- [ ] Verify you have `_vocab_remap_to_old`
- [ ] Update `text_recovery_step` function (Cell 2A)
- [ ] Test the fix (Cell 2B)
- [ ] Start training (Cell 4)

### Option 2 Implementation
- [ ] Run Cell 1 to confirm approach
- [ ] Clear old cache files (Cell 3A)
- [ ] Set student as teacher and rebuild (Cell 3B)
- [ ] Wait ~40 minutes for rebuild
- [ ] Verify new cache (Cell 3C)
- [ ] Start training (Cell 4)

### After Fix
- [ ] Verify no CUDA errors
- [ ] Monitor training loss
- [ ] Continue with Phase 6B stages

---

## 🆘 Troubleshooting

### "No _vocab_remap_to_old attribute"
→ Use Option 2 (rebuild cache)

### "Remapping test failed"
→ Use Option 2 (rebuild cache)

### "Cache rebuild taking too long"
→ Normal, should take 30-40 minutes for ~9600 samples

### "Still getting CUDA errors after fix"
→ Check that tokens are in bounds (0 to 22766)
→ Verify you're using the updated function
→ Report the error with debug output

---

## 📞 Support

If you encounter issues:

1. Check `QUICK_FIX_REFERENCE.md` for common commands
2. Re-read `APPLY_VOCAB_FIX_NOW.md` for detailed steps
3. Verify your implementation matches the provided code
4. Check that all cells ran without errors

---

## ✨ Expected Outcome

After applying the fix, you should see:

```
[6b1] Text decoder warmup (LoRA only)
  max_audio=20s | trainable=15.60M
  [6b1] step   50/300 | loss=2.3456 | KD=50% | lr=1.00e-04
  [6b1] step  100/300 | loss=2.1234 | KD=50% | lr=9.50e-05
  [6b1] step  150/300 | loss=1.9876 | KD=50% | lr=8.50e-05
```

No more CUDA assertion errors! ✓

---

## 🎯 Recommendation

**Use Option 2 (Rebuild Cache)** because:
- ✓ Best training quality
- ✓ Self-consistent (student learns from itself)
- ✓ No information loss
- ✓ Simpler code
- ✓ Worth the 40-minute wait

The only reason to use Option 1 is if you need to start training in the next 5 minutes and can't wait 40 minutes for the rebuild.

---

**Status**: All solution files created ✓  
**Ready to implement**: Yes ✓  
**Recommended path**: Option 2 (rebuild cache) ✓
