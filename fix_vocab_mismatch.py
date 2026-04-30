"""
Fix for Phase 6: Remap teacher cache tokens to student vocabulary

The teacher cache contains tokens from the full 256K vocab, but the student
model only has 22K tokens (from Phase 1 vocab pruning). We need to remap
the teacher tokens to the student's vocabulary space.
"""

# Add this cell BEFORE Phase 6B training in your notebook

print("="*70)
print("Fixing vocab mismatch in teacher cache")
print("="*70)

# Check if student has vocab remapping info
if hasattr(model_student, '_vocab_remap_to_old'):
    print(f"\n✓ Found vocab remap: {len(model_student._vocab_remap_to_old)} tokens")
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(model_student._vocab_remap_to_old)}
    print(f"  Mapping from {len(old_to_new)} old IDs to {model_student.config.vocab_size} new IDs")
else:
    print("\n⚠ No vocab remap found on model_student")
    print("  This means Phase 1 vocab pruning info was not saved properly")
    print("  We'll need to rebuild the cache with the student model as teacher")
    
    # Check if we can load the remap from Phase 1 checkpoint
    p1_ckpt = load_latest_checkpoint('phase1_vocab_pruning')
    if p1_ckpt and 'vocab_remap_to_old' in p1_ckpt:
        print("\n✓ Found vocab remap in Phase 1 checkpoint")
        model_student._vocab_remap_to_old = p1_ckpt['vocab_remap_to_old']
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(model_student._vocab_remap_to_old)}
        print(f"  Loaded mapping: {len(old_to_new)} old IDs -> {model_student.config.vocab_size} new IDs")
    else:
        print("\n✗ Cannot find vocab remap - need to rebuild cache")
        print("  Solution: Use student model as teacher for cache building")
        raise RuntimeError("Vocab remap not found - see solution below")

# Function to remap teacher sequences
def remap_teacher_sequences(teacher_seq, old_to_new, student_vocab_size, unk_id=1):
    """
    Remap teacher token IDs from old vocab to new vocab.
    
    Args:
        teacher_seq: torch.Tensor of token IDs from teacher (old vocab)
        old_to_new: dict mapping old token ID -> new token ID
        student_vocab_size: size of student vocabulary
        unk_id: ID to use for unmapped tokens (default: 1 for <unk>)
    
    Returns:
        torch.Tensor of remapped token IDs (new vocab)
    """
    remapped = teacher_seq.clone()
    
    for i in range(len(remapped)):
        old_id = int(remapped[i].item())
        
        # Check if this token exists in the mapping
        if old_id in old_to_new:
            new_id = old_to_new[old_id]
            remapped[i] = new_id
        else:
            # Token was pruned - map to <unk>
            remapped[i] = unk_id
    
    return remapped

# Test the remapping on one sample
print("\n" + "="*70)
print("Testing remapping on sample")
print("="*70)

sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)

teacher_seq_old = cache_entry['teacher_text_sequences']
print(f"\nOriginal teacher sequence:")
print(f"  Length: {teacher_seq_old.numel()}")
print(f"  Range: [{teacher_seq_old.min().item()}, {teacher_seq_old.max().item()}]")
print(f"  First 10: {teacher_seq_old[:10].tolist()}")

# Check out of bounds
out_of_bounds = (teacher_seq_old >= model_student.config.vocab_size) | (teacher_seq_old < 0)
print(f"  Out of bounds: {out_of_bounds.sum().item()} tokens")

# Remap
teacher_seq_new = remap_teacher_sequences(
    teacher_seq_old, 
    old_to_new, 
    model_student.config.vocab_size
)

print(f"\nRemapped teacher sequence:")
print(f"  Length: {teacher_seq_new.numel()}")
print(f"  Range: [{teacher_seq_new.min().item()}, {teacher_seq_new.max().item()}]")
print(f"  First 10: {teacher_seq_new[:10].tolist()}")

# Check out of bounds
out_of_bounds_new = (teacher_seq_new >= model_student.config.vocab_size) | (teacher_seq_new < 0)
print(f"  Out of bounds: {out_of_bounds_new.sum().item()} tokens")

if out_of_bounds_new.sum().item() == 0:
    print("\n✓ Remapping successful - all tokens in bounds!")
else:
    print("\n✗ Remapping failed - still have out of bounds tokens")

# Test forward pass with remapped sequence
print("\n" + "="*70)
print("Testing forward pass with remapped sequence")
print("="*70)

audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
labels = teacher_seq_new.unsqueeze(0).to(student_device)
labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)

try:
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model_student(
            **audio_inputs,
            labels=labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    print(f"✓ SUCCESS! Forward pass works with remapped tokens")
    print(f"  Loss: {outputs.loss.item():.4f}")
except RuntimeError as e:
    print(f"✗ FAILED: {str(e)[:200]}")

print("\n" + "="*70)
print("Fix complete - now update text_recovery_step to use remapping")
print("="*70)
