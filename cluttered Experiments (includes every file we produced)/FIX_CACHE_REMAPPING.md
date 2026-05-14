# Fix: Remap ALL Cache Entries

## Problem

The test passed with ONE sample, but training fails because:
- You have 9,600 cache entries across 5 shards
- Only the test sample was verified
- Other cache entries still have out-of-bounds tokens
- Remapping during forward pass is inefficient and incomplete

## Solution: Remap Cache Once at Load Time

### Option A: Remap Cache Shards (10 minutes)

Add this function to remap all cache entries:

```python
def remap_phase6_cache_vocab():
    """
    Remap all Phase 6 cache entries from 256K vocab to 22K vocab.
    This modifies the cache files in-place.
    """
    import glob
    
    if not hasattr(model_student, '_vocab_remap_to_old'):
        raise RuntimeError("No vocab remapping info available")
    
    # Build remapping dictionary
    old_to_new = {
        old_id: new_id 
        for new_id, old_id in enumerate(model_student._vocab_remap_to_old)
    }
    
    print("="*70)
    print("REMAPPING ALL CACHE ENTRIES")
    print("="*70)
    print(f"Vocab mapping: {len(old_to_new)} tokens")
    print(f"Old vocab size: {max(old_to_new.keys()) + 1}")
    print(f"New vocab size: {len(model_student._vocab_remap_to_old)}")
    
    # Find all cache shards
    cache_files = sorted(glob.glob(f'{CKPT_DIR}/phase6_teacher_cache_shard_*.pt'))
    print(f"\nFound {len(cache_files)} cache shards to remap")
    
    total_entries = 0
    total_remapped_tokens = 0
    total_unk_tokens = 0
    
    for cache_file in cache_files:
        print(f"\nProcessing {os.path.basename(cache_file)}...")
        
        # Load shard
        shard_data = torch.load(cache_file, map_location='cpu')
        
        # Remap each entry
        for entry in shard_data:
            teacher_seq = entry['teacher_text_sequences']
            remapped_seq = teacher_seq.clone()
            
            for i in range(len(remapped_seq)):
                old_id = int(remapped_seq[i].item())
                if old_id in old_to_new:
                    remapped_seq[i] = old_to_new[old_id]
                    total_remapped_tokens += 1
                else:
                    # Token was pruned - map to <unk>
                    remapped_seq[i] = 1
                    total_unk_tokens += 1
            
            # Update entry
            entry['teacher_text_sequences'] = remapped_seq
            total_entries += 1
        
        # Save remapped shard
        torch.save(shard_data, cache_file)
        print(f"  ✓ Remapped {len(shard_data)} entries")
    
    print("\n" + "="*70)
    print("REMAPPING COMPLETE")
    print("="*70)
    print(f"Total entries remapped: {total_entries}")
    print(f"Total tokens remapped: {total_remapped_tokens}")
    print(f"Tokens mapped to <unk>: {total_unk_tokens}")
    print(f"Percentage <unk>: {100*total_unk_tokens/(total_remapped_tokens+total_unk_tokens):.2f}%")
    
    return total_entries
```

### Option B: Rebuild Cache with Student Model (40 minutes) ⭐ RECOMMENDED

This is still the better solution:

```python
# Clear old cache
import glob
for f in glob.glob(f'{CKPT_DIR}/phase6_teacher_cache_*'):
    os.remove(f)
    print(f"Removed {os.path.basename(f)}")

# Use student as teacher
model_teacher = model_student
model_teacher.eval()

# Rebuild cache
phase6_cache_manifest = build_or_load_phase6_cache(
    'train', 
    ft_samples, 
    shard_size=PHASE6_CACHE_SHARD_SIZE
)

print(f"✓ Cache rebuilt with {phase6_cache_manifest['total_cached']} entries")
```

## Implementation Steps

### If Using Option A (Remap Cache):

**Step 1**: Add the `remap_phase6_cache_vocab()` function above

**Step 2**: Run the remapping:

```python
# Remap all cache entries
total = remap_phase6_cache_vocab()
print(f"\n✓ Remapped {total} cache entries")
```

**Step 3**: Simplify `text_recovery_step` (remove per-step remapping):

```python
def text_recovery_step(sample, cache_entry, use_teacher_text):
    """
    Text recovery training step.
    Cache entries are already remapped to 22K vocab.
    """
    audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
    
    if use_teacher_text:
        # Cache entries are already remapped - use directly
        labels = cache_entry['teacher_text_sequences'].unsqueeze(0).to(student_device)
        labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    else:
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

**Step 4**: Verify all cache entries:

```python
# Test multiple random samples
print("Verifying remapped cache...")
for i in range(10):
    sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)
    teacher_seq = cache_entry['teacher_text_sequences']
    
    out_of_bounds = (teacher_seq >= model_student.config.vocab_size) | (teacher_seq < 0)
    
    if out_of_bounds.sum().item() > 0:
        print(f"✗ Sample {i}: {out_of_bounds.sum().item()} out of bounds tokens!")
        print(f"  Range: [{teacher_seq.min().item()}, {teacher_seq.max().item()}]")
        break
else:
    print("✓ All 10 random samples verified - cache is clean!")
```

**Step 5**: Start training

### If Using Option B (Rebuild Cache) ⭐ RECOMMENDED:

Just run the rebuild code above and start training. No remapping complexity needed.

## Why Option B is Still Better

Even though Option A is faster (10 min vs 40 min), Option B provides:

1. **No <unk> tokens**: Student generates tokens it knows
2. **Self-consistent**: Student learns from its own distribution
3. **Simpler code**: No remapping logic at all
4. **Better quality**: More accurate for your pruned model

## Expected Results

After either fix, verify with:

```python
# Test 10 random samples
for i in range(10):
    sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)
    try:
        loss = text_recovery_step(sample, cache_entry, use_teacher_text=True)
        print(f"Sample {i}: Loss = {loss.item():.4f} ✓")
    except Exception as e:
        print(f"Sample {i}: FAILED - {str(e)[:100]}")
        break
else:
    print("\n✓ All samples passed - ready to train!")
```

Then start training:

```python
phase6_logs['6b1'] = run_text_recovery_stage(
    stage_key='6b1',
    title='Text decoder warmup (LoRA only)',
    steps=STAGE6B1_STEPS,
    max_audio_sec=MAX_AUDIO_SEC_B1,
    text_lr=1e-4,
    kd_prob=PHASE6_TEXT_KD_PROB,
)
```

## My Recommendation

**Use Option B (Rebuild Cache)** because:
- ✓ Cleaner solution
- ✓ Better training quality
- ✓ No <unk> tokens
- ✓ Self-consistent learning
- ✓ Worth the 30 extra minutes

The issue is that your current approach only remaps during forward pass, but you need to remap **all 9,600 cache entries** once, not just the ones you test with.
