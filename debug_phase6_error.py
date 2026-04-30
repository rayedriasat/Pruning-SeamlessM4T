"""
Debug cells to identify the root cause of the CUDA assertion error.
Add these cells to your notebook BEFORE the Phase 6B training cell.
"""

# ============================================================================
# DEBUG CELL 1: Inspect a single cache entry
# ============================================================================
print("="*70)
print("DEBUG 1: Inspecting cache entry structure")
print("="*70)

# Get one sample
sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)

print(f"\nSample keys: {list(sample.keys())}")
print(f"Cache entry keys: {list(cache_entry.keys())}")

print(f"\n--- Sample Info ---")
print(f"ID: {sample['id']}")
print(f"Source lang: {sample['src_lang']}")
print(f"Target lang: {sample['tgt_lang']}")
print(f"Audio shape: {sample['wav'].shape}")
print(f"Reference text: {sample['ref'][:100]}...")

print(f"\n--- Cache Entry Info ---")
print(f"Teacher text str: {cache_entry['teacher_text_str'][:100]}...")
print(f"Teacher text sequences shape: {cache_entry['teacher_text_sequences'].shape}")
print(f"Teacher text sequences dtype: {cache_entry['teacher_text_sequences'].dtype}")
print(f"Teacher text sequences device: {cache_entry['teacher_text_sequences'].device}")
print(f"Teacher text sequences min: {cache_entry['teacher_text_sequences'].min().item()}")
print(f"Teacher text sequences max: {cache_entry['teacher_text_sequences'].max().item()}")
print(f"Teacher text sequences length: {cache_entry['teacher_text_sequences'].numel()}")

print(f"\n--- First 20 tokens ---")
print(cache_entry['teacher_text_sequences'][:20])


# ============================================================================
# DEBUG CELL 2: Check processor tokenizer config
# ============================================================================
print("\n" + "="*70)
print("DEBUG 2: Processor tokenizer configuration")
print("="*70)

print(f"\nTokenizer type: {type(processor.tokenizer)}")
print(f"Vocab size: {processor.tokenizer.vocab_size}")
print(f"Pad token: {processor.tokenizer.pad_token} (id={processor.tokenizer.pad_token_id})")
print(f"EOS token: {processor.tokenizer.eos_token} (id={processor.tokenizer.eos_token_id})")
print(f"BOS token: {processor.tokenizer.bos_token} (id={processor.tokenizer.bos_token_id})")
print(f"UNK token: {processor.tokenizer.unk_token} (id={processor.tokenizer.unk_token_id})")

# Check model config
print(f"\n--- Model Config ---")
print(f"Model vocab size: {model_student.config.vocab_size}")
print(f"Model pad token id: {model_student.config.pad_token_id}")
print(f"Model eos token id: {model_student.config.eos_token_id}")
print(f"Model decoder start token id: {model_student.config.decoder_start_token_id}")
print(f"Max position embeddings: {model_student.config.max_position_embeddings}")

# Check text decoder embedding
print(f"\n--- Text Decoder Embeddings ---")
print(f"Embed tokens num_embeddings: {model_student.text_decoder.embed_tokens.num_embeddings}")
print(f"Embed tokens embedding_dim: {model_student.text_decoder.embed_tokens.embedding_dim}")
print(f"Embed tokens padding_idx: {model_student.text_decoder.embed_tokens.padding_idx}")

# Check position embeddings
print(f"\n--- Position Embeddings ---")
print(f"Position embed num_positions: {model_student.text_decoder.embed_positions.num_positions}")
print(f"Position embed padding_idx: {model_student.text_decoder.embed_positions.padding_idx}")


# ============================================================================
# DEBUG CELL 3: Test label creation with teacher sequences
# ============================================================================
print("\n" + "="*70)
print("DEBUG 3: Testing label creation with teacher sequences")
print("="*70)

# Method 1: Direct use of teacher sequences (our fix)
print("\n--- Method 1: Direct teacher sequences ---")
labels_direct = cache_entry['teacher_text_sequences'].unsqueeze(0).to(student_device)
print(f"Labels shape: {labels_direct.shape}")
print(f"Labels dtype: {labels_direct.dtype}")
print(f"Labels device: {labels_direct.device}")
print(f"Labels min: {labels_direct.min().item()}")
print(f"Labels max: {labels_direct.max().item()}")
print(f"Labels contains pad token: {(labels_direct == processor.tokenizer.pad_token_id).any().item()}")

# Check if any token IDs are out of bounds
vocab_size = model_student.config.vocab_size
out_of_bounds = (labels_direct >= vocab_size) | (labels_direct < 0)
print(f"Out of bounds tokens: {out_of_bounds.sum().item()}")
if out_of_bounds.any():
    print(f"  Out of bounds values: {labels_direct[out_of_bounds].unique()}")

# Mask padding
labels_masked = labels_direct.masked_fill(labels_direct == processor.tokenizer.pad_token_id, -100)
print(f"\nAfter masking padding:")
print(f"Labels min: {labels_masked.min().item()}")
print(f"Labels max (excluding -100): {labels_masked[labels_masked != -100].max().item() if (labels_masked != -100).any() else 'N/A'}")


# ============================================================================
# DEBUG CELL 4: Test label creation with reference text
# ============================================================================
print("\n" + "="*70)
print("DEBUG 4: Testing label creation with reference text")
print("="*70)

print(f"\n--- Method 2: Reference text tokenization ---")
print(f"Reference text: {sample['ref'][:100]}...")

labels_ref = build_target_labels(processor, [sample['ref']], sample['tgt_lang'], student_device)
print(f"Labels shape: {labels_ref.shape}")
print(f"Labels dtype: {labels_ref.dtype}")
print(f"Labels device: {labels_ref.device}")
print(f"Labels min: {labels_ref.min().item()}")
print(f"Labels max (excluding -100): {labels_ref[labels_ref != -100].max().item() if (labels_ref != -100).any() else 'N/A'}")

# Check if any token IDs are out of bounds
out_of_bounds_ref = ((labels_ref >= vocab_size) | (labels_ref < -100)) & (labels_ref != -100)
print(f"Out of bounds tokens: {out_of_bounds_ref.sum().item()}")
if out_of_bounds_ref.any():
    print(f"  Out of bounds values: {labels_ref[out_of_bounds_ref].unique()}")


# ============================================================================
# DEBUG CELL 5: Test actual forward pass with CUDA_LAUNCH_BLOCKING
# ============================================================================
print("\n" + "="*70)
print("DEBUG 5: Testing forward pass with detailed error info")
print("="*70)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("\nTesting with direct teacher sequences...")
audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
labels_test = cache_entry['teacher_text_sequences'].unsqueeze(0).to(student_device)
labels_test = labels_test.masked_fill(labels_test == processor.tokenizer.pad_token_id, -100)

print(f"Audio inputs keys: {list(audio_inputs.keys())}")
for k, v in audio_inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")

print(f"\nLabels: shape={labels_test.shape}, dtype={labels_test.dtype}, device={labels_test.device}")
print(f"Labels range: [{labels_test[labels_test != -100].min().item()}, {labels_test[labels_test != -100].max().item()}]")

try:
    print("\nAttempting forward pass...")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model_student(
            **audio_inputs,
            labels=labels_test,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    print(f"✓ Forward pass succeeded! Loss: {outputs.loss.item():.4f}")
except RuntimeError as e:
    print(f"✗ Forward pass failed with error:")
    print(f"  {str(e)}")
    
    # Additional debugging
    print(f"\n--- Detailed debugging ---")
    
    # Check decoder_input_ids that would be created
    print(f"\nChecking what decoder_input_ids would be created from labels...")
    # The model shifts labels to create decoder_input_ids
    # decoder_input_ids = shift_tokens_right(labels, pad_token_id, decoder_start_token_id)
    
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import shift_tokens_right
    
    decoder_input_ids = shift_tokens_right(
        labels_test,
        model_student.config.pad_token_id,
        model_student.config.decoder_start_token_id
    )
    
    print(f"Decoder input IDs shape: {decoder_input_ids.shape}")
    print(f"Decoder input IDs min: {decoder_input_ids.min().item()}")
    print(f"Decoder input IDs max: {decoder_input_ids.max().item()}")
    print(f"Decoder input IDs first 20: {decoder_input_ids[0, :20]}")
    
    # Check if decoder_input_ids are in valid range
    out_of_bounds_decoder = (decoder_input_ids >= vocab_size) | (decoder_input_ids < 0)
    print(f"Out of bounds decoder input IDs: {out_of_bounds_decoder.sum().item()}")
    if out_of_bounds_decoder.any():
        print(f"  Out of bounds values: {decoder_input_ids[out_of_bounds_decoder].unique()}")
        print(f"  Positions: {torch.where(out_of_bounds_decoder)}")

print("\n" + "="*70)
print("DEBUG COMPLETE")
print("="*70)
