# Kaggle Debug Cells - Paste These Into Your Notebook

Add these cells **BEFORE** the Phase 6B training cell to diagnose the exact issue.

## Cell 1: Basic Cache Inspection

```python
print("="*70)
print("DEBUG: Inspecting cache entry")
print("="*70)

# Get one sample
sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)

print(f"\n--- Cache Entry ---")
print(f"Keys: {list(cache_entry.keys())}")
print(f"Teacher text sequences shape: {cache_entry['teacher_text_sequences'].shape}")
print(f"Teacher text sequences dtype: {cache_entry['teacher_text_sequences'].dtype}")
print(f"Min token ID: {cache_entry['teacher_text_sequences'].min().item()}")
print(f"Max token ID: {cache_entry['teacher_text_sequences'].max().item()}")
print(f"Sequence length: {cache_entry['teacher_text_sequences'].numel()}")
print(f"First 20 tokens: {cache_entry['teacher_text_sequences'][:20].tolist()}")

print(f"\n--- Model Config ---")
print(f"Vocab size: {model_student.config.vocab_size}")
print(f"Pad token ID: {model_student.config.pad_token_id}")
print(f"EOS token ID: {model_student.config.eos_token_id}")
print(f"Decoder start token ID: {model_student.config.decoder_start_token_id}")
print(f"Max position embeddings: {model_student.config.max_position_embeddings}")

# Check for out-of-bounds tokens
vocab_size = model_student.config.vocab_size
out_of_bounds = (cache_entry['teacher_text_sequences'] >= vocab_size) | (cache_entry['teacher_text_sequences'] < 0)
print(f"\n--- Validation ---")
print(f"Out of bounds tokens: {out_of_bounds.sum().item()}")
if out_of_bounds.any():
    bad_tokens = cache_entry['teacher_text_sequences'][out_of_bounds].unique()
    print(f"  Bad token IDs: {bad_tokens.tolist()}")
    print(f"  Vocab size is: {vocab_size}")
```

## Cell 2: Test Forward Pass with Error Details

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("="*70)
print("DEBUG: Testing forward pass")
print("="*70)

# Prepare inputs
audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
labels = cache_entry['teacher_text_sequences'].unsqueeze(0).to(student_device)
labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)

print(f"\nLabels shape: {labels.shape}")
print(f"Labels min: {labels[labels != -100].min().item()}")
print(f"Labels max: {labels[labels != -100].max().item()}")
print(f"Labels first 20: {labels[0, :20].tolist()}")

# Check what decoder_input_ids will be created
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import shift_tokens_right

decoder_input_ids = shift_tokens_right(
    labels,
    model_student.config.pad_token_id,
    model_student.config.decoder_start_token_id
)

print(f"\nDecoder input IDs shape: {decoder_input_ids.shape}")
print(f"Decoder input IDs min: {decoder_input_ids.min().item()}")
print(f"Decoder input IDs max: {decoder_input_ids.max().item()}")
print(f"Decoder input IDs first 20: {decoder_input_ids[0, :20].tolist()}")

# Check for out-of-bounds
out_of_bounds = (decoder_input_ids >= vocab_size) | (decoder_input_ids < 0)
print(f"\nOut of bounds decoder IDs: {out_of_bounds.sum().item()}")
if out_of_bounds.any():
    bad_ids = decoder_input_ids[out_of_bounds].unique()
    print(f"  Bad IDs: {bad_ids.tolist()}")
    positions = torch.where(out_of_bounds)
    print(f"  At positions: batch={positions[0][:5].tolist()}, seq={positions[1][:5].tolist()}")

# Try forward pass
try:
    print("\nAttempting forward pass...")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model_student(
            **audio_inputs,
            labels=labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    print(f"✓ SUCCESS! Loss: {outputs.loss.item():.4f}")
except RuntimeError as e:
    print(f"✗ FAILED!")
    print(f"Error: {str(e)[:200]}")
```

## Cell 3: Compare Teacher vs Processor Tokenization

```python
print("="*70)
print("DEBUG: Comparing tokenization methods")
print("="*70)

# Method 1: Teacher sequences from cache
teacher_seq = cache_entry['teacher_text_sequences']
print(f"\n--- Teacher Sequences (from cache) ---")
print(f"Length: {teacher_seq.numel()}")
print(f"Range: [{teacher_seq.min().item()}, {teacher_seq.max().item()}]")
print(f"Tokens: {teacher_seq[:30].tolist()}")

# Method 2: Re-tokenize the teacher text
teacher_text = cache_entry['teacher_text_str']
print(f"\n--- Teacher Text ---")
print(f"Text: {teacher_text[:100]}...")

retokenized = processor.tokenizer(
    text_target=[teacher_text],
    tgt_lang=sample['tgt_lang'],
    return_tensors='pt',
    padding=True,
)
retok_ids = retokenized['input_ids'][0]

print(f"\n--- Re-tokenized ---")
print(f"Length: {retok_ids.numel()}")
print(f"Range: [{retok_ids.min().item()}, {retok_ids.max().item()}]")
print(f"Tokens: {retok_ids[:30].tolist()}")

# Compare
print(f"\n--- Comparison ---")
print(f"Length match: {teacher_seq.numel() == retok_ids.numel()}")
if teacher_seq.numel() == retok_ids.numel():
    matches = (teacher_seq == retok_ids.cpu()).sum().item()
    print(f"Token matches: {matches}/{teacher_seq.numel()} ({100*matches/teacher_seq.numel():.1f}%)")
    if matches != teacher_seq.numel():
        diff_positions = torch.where(teacher_seq != retok_ids.cpu())[0][:10]
        print(f"First differences at positions: {diff_positions.tolist()}")
        for pos in diff_positions[:3]:
            print(f"  Pos {pos}: teacher={teacher_seq[pos].item()}, retok={retok_ids[pos].item()}")
```

## Cell 4: Check Processor Configuration

```python
print("="*70)
print("DEBUG: Processor and model configuration")
print("="*70)

print(f"\n--- Processor Tokenizer ---")
print(f"Type: {type(processor.tokenizer).__name__}")
print(f"Vocab size: {processor.tokenizer.vocab_size}")
print(f"Model max length: {processor.tokenizer.model_max_length}")

special_tokens = {
    'pad': (processor.tokenizer.pad_token, processor.tokenizer.pad_token_id),
    'eos': (processor.tokenizer.eos_token, processor.tokenizer.eos_token_id),
    'bos': (processor.tokenizer.bos_token, processor.tokenizer.bos_token_id),
    'unk': (processor.tokenizer.unk_token, processor.tokenizer.unk_token_id),
}

for name, (token, token_id) in special_tokens.items():
    print(f"{name:>3}: '{token}' (id={token_id})")

print(f"\n--- Model Student Config ---")
print(f"Vocab size: {model_student.config.vocab_size}")
print(f"Hidden size: {model_student.config.hidden_size}")
print(f"Max position embeddings: {model_student.config.max_position_embeddings}")
print(f"Pad token ID: {model_student.config.pad_token_id}")
print(f"EOS token ID: {model_student.config.eos_token_id}")
print(f"Decoder start token ID: {model_student.config.decoder_start_token_id}")

print(f"\n--- Text Decoder ---")
if hasattr(model_student.text_decoder, 'embed_tokens'):
    print(f"Embed tokens num_embeddings: {model_student.text_decoder.embed_tokens.num_embeddings}")
    print(f"Embed tokens padding_idx: {model_student.text_decoder.embed_tokens.padding_idx}")

if hasattr(model_student.text_decoder, 'embed_positions'):
    print(f"Position embed num_positions: {model_student.text_decoder.embed_positions.num_positions}")
    print(f"Position embed padding_idx: {model_student.text_decoder.embed_positions.padding_idx}")

# Check if there's a mismatch
print(f"\n--- Validation ---")
if processor.tokenizer.vocab_size != model_student.config.vocab_size:
    print(f"⚠ MISMATCH: Tokenizer vocab ({processor.tokenizer.vocab_size}) != Model vocab ({model_student.config.vocab_size})")
else:
    print(f"✓ Vocab sizes match: {processor.tokenizer.vocab_size}")

if processor.tokenizer.pad_token_id != model_student.config.pad_token_id:
    print(f"⚠ MISMATCH: Tokenizer pad ({processor.tokenizer.pad_token_id}) != Model pad ({model_student.config.pad_token_id})")
else:
    print(f"✓ Pad token IDs match: {processor.tokenizer.pad_token_id}")
```

---

## Instructions

1. **Add these 4 cells** to your notebook BEFORE the Phase 6B training cell
2. **Run them in order**
3. **Copy the output** and share it
4. This will show us:
   - Exact token IDs in the cache
   - Whether they're out of bounds
   - What decoder_input_ids are created
   - Where exactly the error occurs
   - Any config mismatches

The output will tell us the **exact root cause** without guessing.
