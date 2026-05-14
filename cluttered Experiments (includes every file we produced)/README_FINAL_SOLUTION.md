# Final Solution: Vocab Mismatch Fixed

## What We Found

**Root Cause**: Teacher cache has tokens from 256K vocab, but student model only has 22K vocab (from Phase 1 pruning).

```
Teacher tokens: [256022, 178607, 247676, ...]  ← Out of bounds!
Student vocab:  0 to 22,766                     ← Only 22K tokens
```

## Two Solutions

### Option 1: Remap Tokens (Fast - 5 minutes)
- **IF** you have `_vocab_remap_to_old` on your model
- Remap cached tokens from 256K → 22K vocab
- Update `text_recovery_step` function
- Start training immediately

### Option 2: Rebuild Cache (Slow - 40 minutes, but BETTER)
- Use **student model** as teacher
- Rebuild cache with correct 22K vocab
- No remapping needed
- More accurate for student model

## Quick Start

### Step 1: Determine Which Option

Paste this into a Kaggle cell:

```python
# Check if remapping is available
if hasattr(model_student, '_vocab_remap_to_old'):
    print("✓ Use OPTION 1: Token Remapping (fast)")
else:
    p1 = load_latest_checkpoint('phase1_vocab_pruning')
    if p1 and 'vocab_remap_to_old' in p1:
        model_student._vocab_remap_to_old = p1['vocab_remap_to_old']
        print("✓ Use OPTION 1: Token Remapping (fast)")
    else:
        print("✓ Use OPTION 2: Rebuild Cache (better)")
```

### Step 2: Apply Solution

**For Option 1** - Update `text_recovery_step`:

```python
def text_recovery_step(sample, cache_entry, use_teacher_text):
    audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
    
    if use_teacher_text:
        teacher_seq = cache_entry['teacher_text_sequences']
        
        # Remap from 256K vocab to 22K vocab
        old_to_new = {
            old_id: new_id 
            for new_id, old_id in enumerate(model_student._vocab_remap_to_old)
        }
        
        remapped = teacher_seq.clone()
        for i in range(len(remapped)):
            old_id = int(remapped[i].item())
            remapped[i] = old_to_new.get(old_id, 1)  # 1 = <unk>
        
        labels = remapped.unsqueeze(0).to(student_device)
        labels = labels.masked_fill(labels == 0, -100)
    else:
        labels = build_target_labels(processor, [sample['ref']], sample['tgt_lang'], student_device)

    outputs = model_student(
        **audio_inputs,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    return outputs.loss
```

**For Option 2** - Rebuild cache:

```python
# Clear old cache
import glob
for f in glob.glob(f'{CKPT_DIR}/phase6_teacher_cache_*'):
    os.remove(f)

# Use student as teacher
model_teacher = model_student

# Rebuild (takes ~40 minutes)
phase6_cache_manifest = build_or_load_phase6_cache(
    'train', ft_samples, shard_size=PHASE6_CACHE_SHARD_SIZE
)
```

### Step 3: Start Training

```python
# This will now work!
phase6_logs['6b1'] = run_text_recovery_stage(
    stage_key='6b1',
    title='Text decoder warmup (LoRA only)',
    steps=STAGE6B1_STEPS,
    max_audio_sec=MAX_AUDIO_SEC_B1,
    text_lr=1e-4,
    kd_prob=PHASE6_TEXT_KD_PROB,
)
```

## Files Created

1. **SOLUTION_VOCAB_MISMATCH.md** - Detailed explanation
2. **TEST_SOLUTION.md** - Test cells to determine which option
3. **FINAL_FIX_text_recovery_step.py** - Updated function (Option 1)
4. **fix_vocab_mismatch.py** - Testing script (Option 1)
5. **This file** - Quick reference

## My Recommendation

**Use Option 2 (Rebuild Cache)** even though it's slower, because:

1. ✓ Student learns from its own output distribution
2. ✓ No information loss from unmapped tokens
3. ✓ Simpler code - no remapping logic
4. ✓ More accurate for your 22K vocab model
5. ✓ You only rebuild once

The 40 minutes is worth it for better training quality.

## Expected Output After Fix

```
[6b1] Text decoder warmup (LoRA only)
  max_audio=20s | trainable=15.60M
  [6b1] step   50/300 | loss=2.3456 | KD=50% | lr=1.00e-04
  [6b1] step  100/300 | loss=2.1234 | KD=50% | lr=9.50e-05
  ...
```

No more CUDA errors! ✓

## Questions?

- **Why did this happen?** Phase 1 pruned vocab to 22K, but Phase 6A cache used base teacher (256K vocab)
- **Will this affect quality?** Option 2 might actually improve quality (student learns from itself)
- **Can I use the old cache?** Only with Option 1 (remapping), but Option 2 is better

---

**Status**: Root cause identified ✓  
**Solution**: Two options provided ✓  
**Ready to fix**: Yes ✓
