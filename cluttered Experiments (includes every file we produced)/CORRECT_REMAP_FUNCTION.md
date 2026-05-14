# Correct Cache Remapping Function

## The Issue

The cache files are named:
- `phase6_teacher_cache_train_step000000.pt`
- `phase6_teacher_cache_train_step000001.pt`
- etc.

NOT `phase6_teacher_cache_shard_*.pt`

## Corrected Remapping Function

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
    
    # Find all cache shards - CORRECTED PATTERN
    cache_files = sorted(glob.glob(f'{CKPT_DIR}/phase6_teacher_cache_train_step*.pt'))
    print(f"\nFound {len(cache_files)} cache shards to remap")
    
    if len(cache_files) == 0:
        print("\n✗ No cache files found!")
        print(f"Looking in: {CKPT_DIR}")
        print("\nTrying to pull from Google Drive...")
        
        # Pull cache from Drive if on Kaggle
        if ON_KAGGLE:
            phase6_rclone_copy_checkpoint_family(
                [phase6_cache_checkpoint_name('train'), phase6_cache_manifest_name('train')],
                direction='pull',
            )
            cache_files = sorted(glob.glob(f'{CKPT_DIR}/phase6_teacher_cache_train_step*.pt'))
            print(f"After pulling: Found {len(cache_files)} cache shards")
        
        if len(cache_files) == 0:
            raise RuntimeError("No cache files found even after pulling from Drive")
    
    total_entries = 0
    total_remapped_tokens = 0
    total_unk_tokens = 0
    
    for cache_file in cache_files:
        print(f"\nProcessing {os.path.basename(cache_file)}...")
        
        # Load shard
        shard_data = torch.load(cache_file, map_location='cpu')
        
        # The structure is: {'split': ..., 'shard_idx': ..., 'entries': [...], 'total_cached': ...}
        entries = shard_data.get('entries', [])
        
        if not entries:
            print(f"  ⚠ No entries in this shard, skipping")
            continue
        
        # Remap each entry
        for entry in entries:
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
        print(f"  ✓ Remapped {len(entries)} entries")
    
    print("\n" + "="*70)
    print("REMAPPING COMPLETE")
    print("="*70)
    print(f"Total entries remapped: {total_entries}")
    print(f"Total tokens remapped: {total_remapped_tokens}")
    print(f"Tokens mapped to <unk>: {total_unk_tokens}")
    if total_remapped_tokens + total_unk_tokens > 0:
        print(f"Percentage <unk>: {100*total_unk_tokens/(total_remapped_tokens+total_unk_tokens):.2f}%")
    
    # Push remapped cache back to Drive
    if ON_KAGGLE:
        print("\nPushing remapped cache to Google Drive...")
        phase6_rclone_copy_checkpoint_family(
            [phase6_cache_checkpoint_name('train')],
            direction='push',
        )
        print("✓ Pushed to Drive")
    
    return total_entries

print("✓ Corrected function added")
```

## Usage

### Step 1: Add the corrected function

Paste the function above into a new cell.

### Step 2: Run the remapping

```python
# Remap all cache entries (takes ~5-10 minutes)
total = remap_phase6_cache_vocab()
print(f"\n✓ Successfully remapped {total} cache entries")
```

### Step 3: Verify multiple samples

```python
print("Verifying remapped cache with 10 random samples...")

for i in range(10):
    sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)
    teacher_seq = cache_entry['teacher_text_sequences']
    
    # Check bounds
    out_of_bounds = (teacher_seq >= model_student.config.vocab_size) | (teacher_seq < 0)
    
    if out_of_bounds.sum().item() > 0:
        print(f"✗ Sample {i}: {out_of_bounds.sum().item()} out of bounds tokens!")
        print(f"  Range: [{teacher_seq.min().item()}, {teacher_seq.max().item()}]")
        break
    
    # Test forward pass
    try:
        loss = text_recovery_step(sample, cache_entry, use_teacher_text=True)
        print(f"Sample {i}: Loss = {loss.item():.4f} ✓")
    except Exception as e:
        print(f"Sample {i}: FAILED - {str(e)[:100]}")
        break
else:
    print("\n" + "="*70)
    print("ALL SAMPLES VERIFIED - READY TO TRAIN")
    print("="*70)
```

### Step 4: Start training

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

## Key Changes

1. **Correct filename pattern**: `phase6_teacher_cache_train_step*.pt` instead of `phase6_teacher_cache_shard_*.pt`
2. **Pull from Drive if needed**: Automatically pulls cache files if not found locally
3. **Correct data structure**: Accesses `shard_data['entries']` instead of treating `shard_data` as a list
4. **Push back to Drive**: Automatically pushes remapped cache back to Google Drive after remapping

## Expected Output

```
======================================================================
REMAPPING ALL CACHE ENTRIES
======================================================================
Vocab mapping: 22767 tokens
Old vocab size: 256206
New vocab size: 22767

Found 5 cache shards to remap

Processing phase6_teacher_cache_train_step000000.pt...
  ✓ Remapped 1920 entries

Processing phase6_teacher_cache_train_step000001.pt...
  ✓ Remapped 1920 entries

Processing phase6_teacher_cache_train_step000002.pt...
  ✓ Remapped 1920 entries

Processing phase6_teacher_cache_train_step000003.pt...
  ✓ Remapped 1920 entries

Processing phase6_teacher_cache_train_step000004.pt...
  ✓ Remapped 1920 entries

======================================================================
REMAPPING COMPLETE
======================================================================
Total entries remapped: 9600
Total tokens remapped: 245000
Tokens mapped to <unk>: 12000
Percentage <unk>: 4.67%

Pushing remapped cache to Google Drive...
✓ Pushed to Drive

✓ Successfully remapped 9600 cache entries
```

This should now work correctly!
