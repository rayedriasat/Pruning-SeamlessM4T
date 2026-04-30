# Phase 6 AcceleratorError Fix

## Root Cause

The CUDA device-side assertion error occurs because:

1. Teacher cache stores `teacher_text_sequences` as a 1D tensor (from `out.sequences[0]`)
2. When using teacher text for KD, the code decodes this to `teacher_text_str` and re-tokenizes it
3. Re-tokenization can produce different token IDs or sequence lengths than the original
4. This causes position embedding index out of bounds errors

## The Fix

Instead of re-tokenizing the decoded teacher text, use the pre-tokenized `teacher_text_sequences` directly from the cache.

### Changes Required in pragmata-recovery.ipynb

#### 1. Update `text_recovery_step` function

**OLD CODE (BROKEN):**
```python
def text_recovery_step(sample, cache_entry, use_teacher_text):
    audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
    target_text = cache_entry['teacher_text_str'] if use_teacher_text else sample['ref']
    labels = build_target_labels(processor, [target_text], sample['tgt_lang'], student_device)

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

**NEW CODE (FIXED):**
```python
def text_recovery_step(sample, cache_entry, use_teacher_text):
    audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
    
    if use_teacher_text:
        # Use pre-tokenized teacher sequences directly from cache
        labels = cache_entry['teacher_text_sequences'].unsqueeze(0).to(student_device)
        # Mask padding tokens
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

## Why This Works

1. **No re-tokenization**: Uses the exact token IDs that the teacher model generated
2. **Correct sequence length**: Preserves the original sequence length from teacher generation
3. **Proper device placement**: Moves tensor to student_device before use
4. **Correct shape**: Adds batch dimension with `unsqueeze(0)`
5. **Proper masking**: Masks padding tokens with -100 for loss computation

## Additional Safety Check

Add validation in the cache building function to ensure sequences are valid:

```python
def build_teacher_cache_entry(model_teacher, sample):
    # ... existing code ...
    
    teacher_text_sequences = out.sequences[0].detach().cpu()
    
    # Validate sequence length
    if teacher_text_sequences.numel() == 0:
        return None, 'empty_teacher_sequence'
    
    if teacher_text_sequences.numel() > 512:  # max position embeddings
        return None, f'teacher_sequence_too_long:{teacher_text_sequences.numel()}'
    
    # ... rest of existing code ...
```

## Testing

After applying the fix:
1. The position embeddings will receive valid indices
2. No CUDA assertion errors
3. Teacher KD will use exact teacher token sequences
4. Training should proceed normally
