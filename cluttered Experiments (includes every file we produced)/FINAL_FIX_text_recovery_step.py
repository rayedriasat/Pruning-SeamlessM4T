"""
FINAL FIX for text_recovery_step with vocab remapping

Replace the text_recovery_step function in your notebook with this version.
"""

def text_recovery_step(sample, cache_entry, use_teacher_text):
    """
    Text recovery training step with vocab remapping for teacher sequences.
    
    The teacher cache contains tokens from the full 256K vocab, but the student
    model only has 22K tokens (from Phase 1 vocab pruning). We need to remap
    the teacher tokens to the student's vocabulary space.
    """
    audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
    
    if use_teacher_text:
        # Get teacher sequences from cache
        teacher_seq = cache_entry['teacher_text_sequences']
        
        # Remap from old vocab (256K) to new vocab (22K)
        if hasattr(model_student, '_vocab_remap_to_old'):
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
        else:
            # No remapping available - this will fail
            raise RuntimeError(
                "Model does not have _vocab_remap_to_old attribute. "
                "Cannot remap teacher tokens from 256K vocab to 22K vocab. "
                "Solution: Rebuild cache using student model as teacher."
            )
        
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


# Also update phase6_pick_training_pair to add remapping helper
def remap_teacher_sequences_cached(cache_entry, old_to_new):
    """
    Remap teacher sequences in cache entry from old vocab to new vocab.
    This modifies the cache entry in-place for efficiency.
    """
    if 'teacher_text_sequences_remapped' not in cache_entry:
        teacher_seq = cache_entry['teacher_text_sequences']
        remapped = teacher_seq.clone()
        
        for i in range(len(remapped)):
            old_id = int(remapped[i].item())
            if old_id in old_to_new:
                remapped[i] = old_to_new[old_id]
            else:
                remapped[i] = 1  # <unk>
        
        cache_entry['teacher_text_sequences_remapped'] = remapped
    
    return cache_entry['teacher_text_sequences_remapped']


print("""
UPDATED text_recovery_step function with vocab remapping.

Replace the existing text_recovery_step in your notebook with the version above.

Key changes:
1. Checks for _vocab_remap_to_old attribute on model_student
2. Remaps teacher tokens from 256K vocab to 22K vocab
3. Maps pruned tokens to <unk> (ID=1)
4. Raises clear error if remapping info is missing
""")
