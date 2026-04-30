# Quick Test: Which Solution to Use?

Paste this cell into your Kaggle notebook to determine which solution to use:

```python
print("="*70)
print("Determining best solution for vocab mismatch")
print("="*70)

# Check if model has vocab remapping info
has_remap = hasattr(model_student, '_vocab_remap_to_old')

print(f"\n1. Model has _vocab_remap_to_old: {has_remap}")

if not has_remap:
    # Try loading from Phase 1 checkpoint
    print("\n2. Checking Phase 1 checkpoint...")
    p1_ckpt = load_latest_checkpoint('phase1_vocab_pruning')
    if p1_ckpt and 'vocab_remap_to_old' in p1_ckpt:
        print("   ✓ Found vocab_remap_to_old in Phase 1 checkpoint")
        model_student._vocab_remap_to_old = p1_ckpt['vocab_remap_to_old']
        has_remap = True
    else:
        print("   ✗ Not found in Phase 1 checkpoint")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if has_remap:
    print("\n✓ Use OPTION 1: Token Remapping")
    print("\nYou have the vocab remapping info. This is the faster solution.")
    print("\nNext steps:")
    print("1. Replace text_recovery_step with the remapping version")
    print("2. Test with one sample")
    print("3. Start training")
    print("\nSee: FINAL_FIX_text_recovery_step.py")
    
    # Show remapping stats
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(model_student._vocab_remap_to_old)}
    print(f"\nRemapping stats:")
    print(f"  Old vocab size: {max(old_to_new.keys()) + 1}")
    print(f"  New vocab size: {len(model_student._vocab_remap_to_old)}")
    print(f"  Tokens mapped: {len(old_to_new)}")
    
else:
    print("\n✓ Use OPTION 2: Rebuild Cache with Student Model")
    print("\nYou don't have vocab remapping info. Rebuild the cache.")
    print("\nNext steps:")
    print("1. Clear old cache files")
    print("2. Set model_teacher = model_student")
    print("3. Rebuild cache (~30-40 minutes)")
    print("4. Start training")
    print("\nThis is actually the BETTER solution - more accurate!")

print("\n" + "="*70)
```

---

# If Option 1 (Remapping):

```python
# Test the remapping solution
print("Testing token remapping...")

# Get a sample
sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)

# Show original tokens
teacher_seq = cache_entry['teacher_text_sequences']
print(f"\nOriginal tokens: {teacher_seq[:10].tolist()}")
print(f"Range: [{teacher_seq.min().item()}, {teacher_seq.max().item()}]")

# Remap
old_to_new = {old_id: new_id for new_id, old_id in enumerate(model_student._vocab_remap_to_old)}
remapped = teacher_seq.clone()
for i in range(len(remapped)):
    old_id = int(remapped[i].item())
    remapped[i] = old_to_new.get(old_id, 1)  # 1 = <unk>

print(f"\nRemapped tokens: {remapped[:10].tolist()}")
print(f"Range: [{remapped.min().item()}, {remapped.max().item()}]")

# Check bounds
out_of_bounds = (remapped >= model_student.config.vocab_size) | (remapped < 0)
print(f"\nOut of bounds: {out_of_bounds.sum().item()} tokens")

if out_of_bounds.sum().item() == 0:
    print("✓ All tokens in bounds - remapping works!")
    
    # Test forward pass
    audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
    labels = remapped.unsqueeze(0).to(student_device)
    labels = labels.masked_fill(labels == 0, -100)
    
    try:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model_student(
                **audio_inputs,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
        print(f"✓ Forward pass SUCCESS! Loss: {outputs.loss.item():.4f}")
        print("\n→ Update text_recovery_step and start training!")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)[:100]}")
else:
    print("✗ Remapping failed - use Option 2 instead")
```

---

# If Option 2 (Rebuild Cache):

```python
# Clear old cache and rebuild with student model

print("Clearing old cache files...")
import glob
old_files = glob.glob(f'{CKPT_DIR}/phase6_teacher_cache_*')
for f in old_files:
    os.remove(f)
    print(f"  Removed {os.path.basename(f)}")

print(f"\nCleared {len(old_files)} cache files")

# Use student as teacher
print("\nSetting model_teacher = model_student...")
model_teacher = model_student
model_teacher.eval()

print("\nRebuilding cache (this will take ~30-40 minutes)...")
print("The new cache will have tokens in the correct 22K vocab range")

# Rebuild
phase6_cache_manifest = build_or_load_phase6_cache(
    'train', 
    ft_samples, 
    shard_size=PHASE6_CACHE_SHARD_SIZE
)

print("\n✓ Cache rebuilt successfully!")
print(f"  Total entries: {phase6_cache_manifest['total_cached']}")
print("\n→ Now you can start Phase 6B training!")
```

---

Run the first cell to see which option you should use, then run the corresponding test/fix cell.
