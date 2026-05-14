# Vocab Mismatch Issue - Complete Summary

## What Happened

Your debug outputs revealed the exact problem:

### Debug Evidence
```
DEBUG 1: Cache Entry Analysis
- Teacher text sequences shape: torch.Size([23])
- Min token ID: 3
- Max token ID: 256022  ← WAY OUT OF BOUNDS!
- Out of bounds tokens: 5
- Bad token IDs: [28442, 30428, 178607, 247676, 256022]
- Vocab size is: 22767  ← Student model only has 22K tokens

DEBUG 2: Forward Pass Test
- Labels max: 256022  ← Out of bounds!
- Out of bounds decoder IDs: 5
- Result: CUDA error: device-side assert triggered

DEBUG 3: Tokenization Comparison
- Teacher sequences from cache: Range [3, 256022]
- Re-tokenized text: Range [3, 256022]
- Conclusion: Re-tokenizing doesn't help (processor still has 256K vocab)

DEBUG 4: Configuration Check
- Processor Tokenizer vocab size: 256001  ← Full vocabulary
- Model Student vocab size: 22767         ← Pruned vocabulary
- ⚠ MISMATCH: Tokenizer vocab (256001) != Model vocab (22767)
```

## Root Cause

**Phase 1**: You pruned the vocabulary from 256K → 22K tokens to reduce model size

**Phase 6A**: You built the teacher cache using the **base teacher model** which still has the full 256K vocabulary

**Phase 6B**: You're trying to train the **Phase 5 student model** which only has 22K vocabulary

**Result**: Teacher cache contains tokens like `256022` that don't exist in the student's embedding table → CUDA assertion error when trying to look them up

## Why Re-tokenizing Didn't Work

The initial fix attempted to re-tokenize the teacher text instead of using cached sequences. This didn't work because:

1. The **processor's tokenizer** still has the full 256K vocabulary
2. Re-tokenizing produces the same out-of-bounds tokens
3. The mismatch is between **processor vocab** (256K) and **model vocab** (22K)

## The Real Solution

You need to ensure the teacher sequences use the **same vocabulary** as the student model. Two ways to do this:

### Option 1: Remap Tokens
- Map cached tokens from 256K vocab → 22K vocab
- Uses `_vocab_remap_to_old` attribute from Phase 1
- Fast (5 minutes) but requires remapping info

### Option 2: Rebuild Cache ⭐ RECOMMENDED
- Use **student model** as its own teacher
- Generates tokens in correct 22K vocabulary
- Slower (40 minutes) but better quality

## Why Option 2 is Better

Even though Option 2 takes longer, it provides:

1. **Better training quality**: Student learns from its own output distribution, not a different model's distribution
2. **No information loss**: No tokens mapped to `<unk>` because they were pruned
3. **Self-consistency**: Student generates and learns from the same vocabulary
4. **Simpler code**: No remapping logic needed in training loop
5. **One-time cost**: 40 minutes now, but cleaner training forever

## Implementation Path

### Path A: Quick Fix (Option 1)
```
1. Check for _vocab_remap_to_old attribute
2. Update text_recovery_step with remapping logic
3. Test with one sample
4. Start training
Time: 5 minutes
```

### Path B: Better Fix (Option 2) ⭐
```
1. Clear old cache files
2. Set model_teacher = model_student
3. Rebuild cache (~40 minutes)
4. Verify tokens are in bounds
5. Start training
Time: 40 minutes
```

## What You Should Do

**My recommendation**: Use **Option 2** (rebuild cache)

**Reasoning**:
- You're doing research for a paper (INTERSPEECH/IWSLT 2026)
- Training quality matters more than 35 minutes of setup time
- Self-consistent training (student learns from itself) is more principled
- Avoids potential issues with unmapped tokens
- Cleaner implementation without remapping complexity

**However**, if you need to start training immediately and have the remapping info, Option 1 will work.

## Files Created for You

1. **APPLY_VOCAB_FIX_NOW.md** - Complete step-by-step guide for both options
2. **QUICK_FIX_REFERENCE.md** - Quick reference card with commands
3. **SOLUTION_VOCAB_MISMATCH.md** - Detailed technical explanation
4. **FINAL_FIX_text_recovery_step.py** - Updated function with remapping (Option 1)
5. **TEST_SOLUTION.md** - Test cells to determine which option to use
6. **This file** - Summary of what happened and why

## Next Steps

1. **Read**: `APPLY_VOCAB_FIX_NOW.md` for complete instructions
2. **Decide**: Option 1 (fast) or Option 2 (better)
3. **Implement**: Follow the step-by-step guide
4. **Test**: Verify with one sample before full training
5. **Train**: Start Phase 6B with confidence

## Expected Timeline

### Option 1 (Remapping)
- Check for remapping info: 1 minute
- Update function: 2 minutes
- Test: 2 minutes
- **Total: ~5 minutes**

### Option 2 (Rebuild)
- Clear cache: 1 minute
- Update teacher model: 1 minute
- Rebuild cache: 35-40 minutes
- Verify: 2 minutes
- **Total: ~40 minutes**

## After the Fix

You'll see normal training output:
```
[6b1] Text decoder warmup (LoRA only)
  max_audio=20s | trainable=15.60M
  [6b1] step   50/300 | loss=2.3456 | KD=50% | lr=1.00e-04
  [6b1] step  100/300 | loss=2.1234 | KD=50% | lr=9.50e-05
  [6b1] step  150/300 | loss=1.9876 | KD=50% | lr=8.50e-05
```

No more CUDA assertion errors! ✓

## Key Insight

This issue highlights an important principle in model compression:

> **When you prune a model's vocabulary, you must ensure all downstream components (tokenizers, caches, training data) use the same pruned vocabulary.**

In your case, the teacher cache was built with the original vocabulary, creating a mismatch with the pruned student model. The fix ensures vocabulary consistency throughout the training pipeline.

---

**Status**: Root cause identified ✓  
**Solution**: Two options provided ✓  
**Recommendation**: Option 2 (rebuild cache) ✓  
**Ready to proceed**: Yes ✓
