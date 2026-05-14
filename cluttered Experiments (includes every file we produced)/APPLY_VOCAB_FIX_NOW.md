# APPLY VOCAB FIX NOW - Step by Step Guide

## Root Cause Confirmed ✓

Your debug output clearly shows:
- **Teacher cache tokens**: Range [3, 256022] - from 256K vocabulary
- **Student model vocab**: Only 22,767 tokens
- **Out of bounds tokens**: 5 per sample (e.g., 256022, 178607, 247676, 30428, 28442)
- **Result**: CUDA assertion error when embedding lookup fails

## Two Solutions Available

### Option 1: Token Remapping (Fast - 5 minutes)
- **IF** you have `_vocab_remap_to_old` attribute saved on your model
- Remap cached tokens from 256K → 22K vocabulary
- Can start training immediately

### Option 2: Rebuild Cache (Slow - 40 minutes, but RECOMMENDED)
- Use **student model** as its own teacher
- Generates tokens in correct 22K vocabulary
- More accurate for your pruned model
- No remapping complexity

---

## STEP 1: Determine Which Option You Have

**Paste this into a NEW cell in your Kaggle notebook:**

```python
print("="*70)
print("CHECKING VOCAB REMAPPING AVAILABILITY")
print("="*70)

# Check if model has vocab remapping info
has_remap = hasattr(model_student, '_vocab_remap_to_old')

print(f"\n1. Model has _vocab_remap_to_old attribute: {has_remap}")

if not has_remap:
    # Try loading from Phase 1 checkpoint
    print("\n2. Checking Phase 1 checkpoint for remapping info...")
    try:
        p1_ckpt = load_latest_checkpoint('phase1_vocab_pruning')
        if p1_ckpt and 'vocab_remap_to_old' in p1_ckpt:
            print("   ✓ Found vocab_remap_to_old in Phase 1 checkpoint")
            model_student._vocab_remap_to_old = p1_ckpt['vocab_remap_to_old']
            has_remap = True
        else:
            print("   ✗ Not found in Phase 1 checkpoint")
    except Exception as e:
        print(f"   ✗ Error loading Phase 1 checkpoint: {str(e)[:100]}")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if has_remap:
    print("\n✓ You can use OPTION 1: Token Remapping (fast)")
    print("\nYou have the vocab remapping info available.")
    print("\nHOWEVER, I still recommend OPTION 2 (rebuild cache) because:")
    print("  • Student learns from its own output distribution")
    print("  • No information loss from unmapped tokens")
    print("  • More accurate for your 22K vocab model")
    print("  • Only takes 40 minutes (one-time cost)")
    
    # Show remapping stats
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(model_student._vocab_remap_to_old)}
    print(f"\nRemapping stats:")
    print(f"  Old vocab size: {max(old_to_new.keys()) + 1}")
    print(f"  New vocab size: {len(model_student._vocab_remap_to_old)}")
    print(f"  Tokens mapped: {len(old_to_new)}")
    
    print("\n→ Proceed to OPTION 1 or OPTION 2 below")
    
else:
    print("\n✓ Use OPTION 2: Rebuild Cache with Student Model")
    print("\nYou don't have vocab remapping info, so you must rebuild.")
    print("\nThis is actually the BETTER solution:")
    print("  • Student model generates tokens it actually knows")
    print("  • More accurate for training")
    print("  • Simpler implementation")
    
    print("\n→ Proceed to OPTION 2 below")

print("\n" + "="*70)
```

**Run this cell and see which option you have available.**

---

## OPTION 1: Token Remapping (If Available)

### Step 1A: Update text_recovery_step Function

**Find the cell with `def text_recovery_step(...)` and REPLACE it with:**

```python
def text_recovery_step(sample, cache_entry, use_teacher_text):
    """
    Text recovery training step with vocab remapping for teacher sequences.
    
    The teacher cache contains tokens from the full 256K vocab, but the student
    model only has 22K tokens (from Phase 1 vocab pruning). We remap the teacher
    tokens to the student's vocabulary space.
    """
    audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
    
    if use_teacher_text:
        # Get teacher sequences from cache
        teacher_seq = cache_entry['teacher_text_sequences']
        
        # Remap from old vocab (256K) to new vocab (22K)
        if not hasattr(model_student, '_vocab_remap_to_old'):
            raise RuntimeError(
                "Model does not have _vocab_remap_to_old attribute. "
                "Cannot remap teacher tokens. Use Option 2 instead."
            )
        
        # Build reverse mapping: old_id -> new_id
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
                remapped_seq[i] = 1  # <unk> token ID
        
        labels = remapped_seq.unsqueeze(0).to(student_device)
        # Mask padding tokens
        labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    else:
        # Use ground truth reference text (already in correct vocab)
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

### Step 1B: Test the Fix

**Add a NEW cell to test:**

```python
print("Testing token remapping fix...")

# Get a sample
sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)

# Show original tokens
teacher_seq = cache_entry['teacher_text_sequences']
print(f"\nOriginal tokens (first 10): {teacher_seq[:10].tolist()}")
print(f"Original range: [{teacher_seq.min().item()}, {teacher_seq.max().item()}]")

# Test remapping
old_to_new = {old_id: new_id for new_id, old_id in enumerate(model_student._vocab_remap_to_old)}
remapped = teacher_seq.clone()
for i in range(len(remapped)):
    old_id = int(remapped[i].item())
    remapped[i] = old_to_new.get(old_id, 1)  # 1 = <unk>

print(f"\nRemapped tokens (first 10): {remapped[:10].tolist()}")
print(f"Remapped range: [{remapped.min().item()}, {remapped.max().item()}]")

# Check bounds
out_of_bounds = (remapped >= model_student.config.vocab_size) | (remapped < 0)
print(f"\nOut of bounds tokens: {out_of_bounds.sum().item()}")

if out_of_bounds.sum().item() == 0:
    print("✓ All tokens in bounds!")
    
    # Test forward pass
    try:
        loss = text_recovery_step(sample, cache_entry, use_teacher_text=True)
        print(f"✓ Forward pass SUCCESS! Loss: {loss.item():.4f}")
        print("\n→ Fix verified! You can now start Phase 6B training.")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)[:200]}")
        print("\n→ Use Option 2 instead (rebuild cache)")
else:
    print("✗ Remapping failed - tokens still out of bounds")
    print("\n→ Use Option 2 instead (rebuild cache)")
```

### Step 1C: Start Training

If the test passes, proceed with Phase 6B training:

```python
# This should now work!
phase6_logs['6b1'] = run_text_recovery_stage(
    stage_key='6b1',
    title='Text decoder warmup (LoRA only)',
    steps=STAGE6B1_STEPS,
    max_audio_sec=MAX_AUDIO_SEC_B1,
    text_lr=1e-4,
    kd_prob=PHASE6_TEXT_KD_PROB,
)
```

---

## OPTION 2: Rebuild Cache (RECOMMENDED)

This is the cleaner, more accurate solution. The student model generates tokens in its own 22K vocabulary.

### Step 2A: Clear Old Cache Files

**Add a NEW cell:**

```python
print("Clearing old cache files...")
import glob

old_files = glob.glob(f'{CKPT_DIR}/phase6_teacher_cache_*')
print(f"Found {len(old_files)} cache files to remove:")

for f in old_files:
    print(f"  Removing {os.path.basename(f)}")
    os.remove(f)

print(f"\n✓ Cleared {len(old_files)} cache files")
```

### Step 2B: Rebuild Cache with Student Model

**Find the Phase 6A cache building cell and UPDATE it:**

Look for the cell that has:
```python
model_teacher, _ = load_model_from_drive('phase0_v1_baseline')
```

**Change it to:**

```python
# Use student model as its own teacher (generates tokens in correct 22K vocab)
print("Using student model as teacher for cache generation...")
model_teacher = model_student
model_teacher.eval()

print(f"Teacher vocab size: {model_teacher.config.vocab_size}")
print(f"Student vocab size: {model_student.config.vocab_size}")
print("✓ Vocab sizes match - no remapping needed!")
```

### Step 2C: Rebuild the Cache

**Run the cache building cell:**

```python
print("Rebuilding Phase 6 teacher cache...")
print("This will take approximately 30-40 minutes")
print(f"Processing {len(ft_samples)} samples...")

phase6_cache_manifest = build_or_load_phase6_cache(
    'train', 
    ft_samples, 
    shard_size=PHASE6_CACHE_SHARD_SIZE
)

print("\n✓ Cache rebuilt successfully!")
print(f"  Total entries: {phase6_cache_manifest['total_cached']}")
print(f"  Shards: {len(phase6_cache_manifest['shards'])}")
```

### Step 2D: Verify the New Cache

**Add a NEW cell to verify:**

```python
print("Verifying new cache has correct vocab range...")

# Load a cache entry
sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)

teacher_seq = cache_entry['teacher_text_sequences']
print(f"\nTeacher sequence shape: {teacher_seq.shape}")
print(f"Token range: [{teacher_seq.min().item()}, {teacher_seq.max().item()}]")
print(f"Model vocab size: {model_student.config.vocab_size}")

# Check bounds
out_of_bounds = (teacher_seq >= model_student.config.vocab_size) | (teacher_seq < 0)
print(f"Out of bounds tokens: {out_of_bounds.sum().item()}")

if out_of_bounds.sum().item() == 0:
    print("✓ All tokens in bounds!")
    
    # Test forward pass
    try:
        loss = text_recovery_step(sample, cache_entry, use_teacher_text=True)
        print(f"✓ Forward pass SUCCESS! Loss: {loss.item():.4f}")
        print("\n→ Cache verified! You can now start Phase 6B training.")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)[:200]}")
else:
    print(f"✗ Still have {out_of_bounds.sum().item()} out of bounds tokens!")
```

### Step 2E: Start Training

Once verified, proceed with Phase 6B:

```python
# This should now work!
phase6_logs['6b1'] = run_text_recovery_stage(
    stage_key='6b1',
    title='Text decoder warmup (LoRA only)',
    steps=STAGE6B1_STEPS,
    max_audio_sec=MAX_AUDIO_SEC_B1,
    text_lr=1e-4,
    kd_prob=PHASE6_TEXT_KD_PROB,
)
```

---

## My Recommendation

**Use OPTION 2 (Rebuild Cache)** even if Option 1 is available, because:

1. ✓ **More accurate**: Student learns from its own output distribution
2. ✓ **No information loss**: No mapping to `<unk>` for pruned tokens
3. ✓ **Simpler code**: No remapping logic in training loop
4. ✓ **Self-consistent**: Student model generates and learns from same vocab
5. ✓ **One-time cost**: 40 minutes now, but better training quality

The only advantage of Option 1 is speed (5 minutes vs 40 minutes), but Option 2 is worth the wait for better training results.

---

## Expected Output After Fix

```
[6b1] Text decoder warmup (LoRA only)
  max_audio=20s | trainable=15.60M
  [6b1] step   50/300 | loss=2.3456 | KD=50% | lr=1.00e-04
  [6b1] step  100/300 | loss=2.1234 | KD=50% | lr=9.50e-05
  [6b1] step  150/300 | loss=1.9876 | KD=50% | lr=8.50e-05
  ...
```

No more CUDA assertion errors! ✓

---

## Summary

- **Root cause**: Teacher cache (256K vocab) vs Student model (22K vocab)
- **Option 1**: Remap tokens (fast, if available)
- **Option 2**: Rebuild cache with student model (better quality)
- **Recommendation**: Use Option 2 for best results
- **Time cost**: 40 minutes one-time rebuild
- **Benefit**: Clean, accurate training with no vocab mismatches

Choose your option and follow the steps above!
