# SOLUTION: Vocab Mismatch Between Teacher Cache and Student Model

## Root Cause (Confirmed by Debug Output)

**The teacher cache contains tokens from the FULL vocabulary (256,001 tokens), but your student model only has 22,767 tokens (from Phase 1 vocab pruning).**

### Evidence:
```
Teacher cache tokens: [256022, 178607, 247676, 30428, 28442]
Student vocab size: 22,767
Processor vocab size: 256,001
Out of bounds tokens: 5
```

These tokens (256022, 178607, etc.) are **way beyond** the student's vocabulary, causing the embedding lookup to fail with CUDA assertion errors.

## Why This Happened

1. **Phase 1**: You pruned the vocabulary from 256K → 22K tokens
2. **Phase 6A**: You built the teacher cache using the **base teacher model** (still has 256K vocab)
3. **Phase 6B**: You're trying to train the **Phase 5 student model** (only has 22K vocab)
4. **Result**: Teacher tokens don't exist in student vocabulary → CUDA error

## Solution Options

### Option 1: Remap Teacher Tokens (RECOMMENDED)

Remap the cached teacher tokens from the old 256K vocab to the new 22K vocab.

**Requirements:**
- The `_vocab_remap_to_old` attribute must be saved on your model
- This was created during Phase 1 vocab pruning

**Steps:**

#### Step 1: Check if remapping info exists

Add this cell to your notebook:

```python
# Check for vocab remapping info
if hasattr(model_student, '_vocab_remap_to_old'):
    print(f"✓ Found vocab remap: {len(model_student._vocab_remap_to_old)} tokens")
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(model_student._vocab_remap_to_old)}
else:
    print("✗ No vocab remap found")
    # Try loading from Phase 1 checkpoint
    p1_ckpt = load_latest_checkpoint('phase1_vocab_pruning')
    if p1_ckpt and 'vocab_remap_to_old' in p1_ckpt:
        print("✓ Found in Phase 1 checkpoint")
        model_student._vocab_remap_to_old = p1_ckpt['vocab_remap_to_old']
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(model_student._vocab_remap_to_old)}
    else:
        print("✗ Cannot find vocab remap - use Option 2")
```

#### Step 2: Update text_recovery_step function

Replace your `text_recovery_step` function with this:

```python
def text_recovery_step(sample, cache_entry, use_teacher_text):
    audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
    
    if use_teacher_text:
        # Get teacher sequences from cache
        teacher_seq = cache_entry['teacher_text_sequences']
        
        # Remap from old vocab (256K) to new vocab (22K)
        if not hasattr(model_student, '_vocab_remap_to_old'):
            raise RuntimeError("No vocab remapping info - cannot use teacher cache")
        
        old_to_new = {
            old_id: new_id 
            for new_id, old_id in enumerate(model_student._vocab_remap_to_old)
        }
        
        # Remap each token
        remapped_seq = teacher_seq.clone()
        for i in range(len(remapped_seq)):
            old_id = int(remapped_seq[i].item())
            if old_id in old_to_new:
                remapped_seq[i] = old_to_new[old_id]
            else:
                # Token was pruned - map to <unk>
                remapped_seq[i] = 1
        
        labels = remapped_seq.unsqueeze(0).to(student_device)
        labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    else:
        # Use ground truth reference text
        labels = build_target_labels(processor, [sample['ref']], sample['tgt_lang'], student_device)

    outputs = model_student(
        **audio_inputs,
        labels=labels,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return outputs.loss
```

#### Step 3: Test it

```python
# Test remapping
sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)
loss = text_recovery_step(sample, cache_entry, use_teacher_text=True)
print(f"✓ Success! Loss: {loss.item():.4f}")
```

---

### Option 2: Rebuild Cache with Student Model (SIMPLER)

If Option 1 doesn't work (no remapping info), rebuild the cache using the **student model** as the teacher.

**Advantages:**
- Simpler - no remapping needed
- Tokens are already in correct vocab
- More accurate for student model

**Steps:**

#### Step 1: Update cache building to use student model

In your Phase 6A cache building cell, change:

```python
# OLD (uses base teacher with 256K vocab)
model_teacher, _ = load_model_from_drive('phase0_v1_baseline')

# NEW (uses student model with 22K vocab)
model_teacher = model_student  # Use student as its own teacher
```

#### Step 2: Clear old cache and rebuild

```python
# Clear old cache files
import glob
old_cache_files = glob.glob(f'{CKPT_DIR}/phase6_teacher_cache_*')
for f in old_cache_files:
    os.remove(f)
    print(f"Removed {f}")

# Rebuild cache (will take ~30-40 minutes)
phase6_cache_manifest = build_or_load_phase6_cache('train', ft_samples, shard_size=PHASE6_CACHE_SHARD_SIZE)
```

#### Step 3: Continue with training

The new cache will have tokens in the correct 22K vocab range, and training will work.

---

## Recommended Approach

**Try Option 1 first** (remapping) - it's faster if you have the remapping info.

**If Option 1 fails** (no remapping info found), use **Option 2** (rebuild cache).

## Why Option 2 Might Be Better

Even if Option 1 works, Option 2 has advantages:

1. **More accurate**: Student model generates tokens it actually knows
2. **No information loss**: No mapping to `<unk>` for pruned tokens
3. **Simpler**: No remapping logic needed
4. **Self-consistent**: Student learns from its own output distribution

The only downside is rebuild time (~30-40 minutes), but you only do it once.

## Quick Decision Tree

```
Do you have _vocab_remap_to_old on model_student?
├─ YES → Use Option 1 (remapping)
│         Test it - if it works, great!
│         If quality is poor, consider Option 2
│
└─ NO → Use Option 2 (rebuild cache with student model)
         This is the cleanest solution
```

## Files to Help You

1. `fix_vocab_mismatch.py` - Test remapping (Option 1)
2. `FINAL_FIX_text_recovery_step.py` - Updated function with remapping
3. This file - Complete solution guide

## Next Steps

1. **Decide**: Option 1 (remap) or Option 2 (rebuild)?
2. **Implement**: Follow the steps above
3. **Test**: Run one training step to verify
4. **Train**: Proceed with Phase 6B

The CUDA error will be gone once tokens are in the correct vocab range!
