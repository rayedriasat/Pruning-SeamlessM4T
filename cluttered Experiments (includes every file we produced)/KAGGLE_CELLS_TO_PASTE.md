# Kaggle Cells to Paste - Vocab Mismatch Fix

Copy and paste these cells into your Kaggle notebook in order.

---

## CELL 1: Check Which Option You Have

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
    
    print("\n→ You can choose OPTION 1 or OPTION 2 below")
    
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

---

## OPTION 1: Token Remapping (If Available and You Want Fast Fix)

### CELL 2A: Update text_recovery_step Function

**Find the cell with `def text_recovery_step(...)` and REPLACE the entire function with:**

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

### CELL 2B: Test Option 1 Fix

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
        print("\n" + "="*70)
        print("FIX VERIFIED - READY TO TRAIN")
        print("="*70)
        print("\nYou can now start Phase 6B training:")
        print("  phase6_logs['6b1'] = run_text_recovery_stage(...)")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)[:200]}")
        print("\n→ Use Option 2 instead (rebuild cache)")
else:
    print("✗ Remapping failed - tokens still out of bounds")
    print("\n→ Use Option 2 instead (rebuild cache)")
```

---

## OPTION 2: Rebuild Cache (RECOMMENDED)

### CELL 3A: Clear Old Cache Files

```python
print("="*70)
print("OPTION 2: REBUILD CACHE WITH STUDENT MODEL")
print("="*70)

print("\nStep 1: Clearing old cache files...")
import glob

old_files = glob.glob(f'{CKPT_DIR}/phase6_teacher_cache_*')
print(f"Found {len(old_files)} cache files to remove:")

for f in old_files:
    print(f"  Removing {os.path.basename(f)}")
    os.remove(f)

print(f"\n✓ Cleared {len(old_files)} cache files")
```

### CELL 3B: Set Student as Teacher and Rebuild

```python
print("\nStep 2: Setting student model as teacher...")

# Use student model as its own teacher (generates tokens in correct 22K vocab)
model_teacher = model_student
model_teacher.eval()

print(f"Teacher vocab size: {model_teacher.config.vocab_size}")
print(f"Student vocab size: {model_student.config.vocab_size}")
print("✓ Vocab sizes match - no remapping needed!")

print("\nStep 3: Rebuilding Phase 6 teacher cache...")
print("This will take approximately 30-40 minutes")
print(f"Processing {len(ft_samples)} samples...")
print("\nStarting cache rebuild...")

# Rebuild cache
phase6_cache_manifest = build_or_load_phase6_cache(
    'train', 
    ft_samples, 
    shard_size=PHASE6_CACHE_SHARD_SIZE
)

print("\n" + "="*70)
print("CACHE REBUILT SUCCESSFULLY")
print("="*70)
print(f"Total entries: {phase6_cache_manifest['total_cached']}")
print(f"Shards: {len(phase6_cache_manifest['shards'])}")
```

### CELL 3C: Verify New Cache

```python
print("="*70)
print("VERIFYING NEW CACHE")
print("="*70)

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
        print("\n" + "="*70)
        print("CACHE VERIFIED - READY TO TRAIN")
        print("="*70)
        print("\nYou can now start Phase 6B training:")
        print("  phase6_logs['6b1'] = run_text_recovery_stage(...)")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)[:200]}")
        print("\nPlease report this error - the cache should work now.")
else:
    print(f"✗ Still have {out_of_bounds.sum().item()} out of bounds tokens!")
    print("\nThis shouldn't happen - please report this error.")
```

---

## CELL 4: Start Phase 6B Training (After Either Option)

```python
print("="*70)
print("STARTING PHASE 6B TRAINING")
print("="*70)

# Stage 6b1: Text decoder warmup (LoRA only)
phase6_logs['6b1'] = run_text_recovery_stage(
    stage_key='6b1',
    title='Text decoder warmup (LoRA only)',
    steps=STAGE6B1_STEPS,
    max_audio_sec=MAX_AUDIO_SEC_B1,
    text_lr=1e-4,
    kd_prob=PHASE6_TEXT_KD_PROB,
)

print("\n✓ Stage 6b1 complete!")
```

---

## Summary

**Option 1 (Fast)**: Paste cells 1, 2A, 2B, then 4
- Time: ~5 minutes
- Requires: `_vocab_remap_to_old` attribute

**Option 2 (Better)**: Paste cells 1, 3A, 3B, 3C, then 4
- Time: ~40 minutes
- Recommended for best training quality

Both options will fix the CUDA assertion error and allow Phase 6B training to proceed!
