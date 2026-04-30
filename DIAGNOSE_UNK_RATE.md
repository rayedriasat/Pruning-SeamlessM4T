# Diagnose 10% <unk> Rate

You're right - if all samples are in your 5 languages and you pruned vocab based on those 5 languages, you should have ~0% <unk>, not 10%.

## Diagnostic Cells

### Cell 1: Check What Tokens Are Being Mapped to <unk>

```python
import glob
from collections import Counter

print("Analyzing which tokens are being mapped to <unk>...")

# Build reverse mapping
old_to_new = {
    old_id: new_id 
    for new_id, old_id in enumerate(model_student._vocab_remap_to_old)
}

# Track unmapped tokens
unmapped_tokens = Counter()
total_tokens = 0
unmapped_count = 0

# Sample a few cache entries
cache_files = sorted(glob.glob(f'{CKPT_DIR}/phase6_teacher_cache_train_step*.pt'))[:2]  # First 2 shards

for cache_file in cache_files:
    print(f"\nAnalyzing {os.path.basename(cache_file)}...")
    shard_data = torch.load(cache_file, map_location='cpu')
    entries = shard_data.get('entries', [])
    
    for entry in entries[:100]:  # First 100 entries per shard
        teacher_seq = entry['teacher_text_sequences']
        
        for token_id in teacher_seq.tolist():
            total_tokens += 1
            if token_id not in old_to_new:
                unmapped_tokens[token_id] += 1
                unmapped_count += 1

print(f"\n{'='*70}")
print(f"UNMAPPED TOKEN ANALYSIS")
print(f"{'='*70}")
print(f"Total tokens analyzed: {total_tokens}")
print(f"Unmapped tokens: {unmapped_count} ({100*unmapped_count/total_tokens:.2f}%)")
print(f"Unique unmapped token IDs: {len(unmapped_tokens)}")

# Show top 20 unmapped tokens
print(f"\nTop 20 most frequent unmapped tokens:")
for token_id, count in unmapped_tokens.most_common(20):
    # Try to decode this token from the processor (which has full vocab)
    try:
        token_str = processor.tokenizer.convert_ids_to_tokens([token_id])[0]
        print(f"  Token ID {token_id:>6}: {count:>4} occurrences | '{token_str}'")
    except:
        print(f"  Token ID {token_id:>6}: {count:>4} occurrences | (cannot decode)")
```

### Cell 2: Compare Teacher Output vs Ground Truth Vocab

```python
print("Comparing teacher-generated text vs ground truth reference text...")

# Sample 10 entries
sample_count = 10
teacher_vocab_usage = Counter()
reference_vocab_usage = Counter()

for i in range(sample_count):
    sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)
    
    # Teacher tokens
    teacher_seq = cache_entry['teacher_text_sequences']
    teacher_vocab_usage.update(teacher_seq.tolist())
    
    # Reference tokens (ground truth)
    ref_tokens = processor.tokenizer.encode(
        sample['ref'],
        return_tensors='pt',
        add_special_tokens=True,
    )[0]
    reference_vocab_usage.update(ref_tokens.tolist())

print(f"\n{'='*70}")
print(f"VOCABULARY USAGE COMPARISON")
print(f"{'='*70}")

teacher_unique = set(teacher_vocab_usage.keys())
reference_unique = set(reference_vocab_usage.keys())

print(f"Teacher unique tokens: {len(teacher_unique)}")
print(f"Reference unique tokens: {len(reference_unique)}")
print(f"Overlap: {len(teacher_unique & reference_unique)}")
print(f"Teacher-only tokens: {len(teacher_unique - reference_unique)}")
print(f"Reference-only tokens: {len(reference_unique - teacher_unique)}")

# Check if teacher-only tokens are in student vocab
teacher_only = teacher_unique - reference_unique
in_student_vocab = sum(1 for t in teacher_only if t < model_student.config.vocab_size)
out_of_bounds = len(teacher_only) - in_student_vocab

print(f"\nTeacher-only tokens analysis:")
print(f"  In student vocab: {in_student_vocab}")
print(f"  Out of bounds: {out_of_bounds}")

# Show some teacher-only tokens
print(f"\nSample teacher-only tokens:")
for token_id in list(teacher_only)[:20]:
    try:
        token_str = processor.tokenizer.convert_ids_to_tokens([token_id])[0]
        in_student = "✓" if token_id < model_student.config.vocab_size else "✗"
        print(f"  {in_student} Token ID {token_id:>6}: '{token_str}'")
    except:
        print(f"  Token ID {token_id:>6}: (cannot decode)")
```

### Cell 3: Check Phase 1 Vocab Pruning Strategy

```python
print("Checking Phase 1 vocab pruning details...")

# Try to load Phase 1 checkpoint
p1_ckpt = load_latest_checkpoint('phase1_vocab_pruning')

if p1_ckpt and 'vocab_remap_to_old' in p1_ckpt:
    vocab_remap = p1_ckpt['vocab_remap_to_old']
    print(f"\n✓ Found vocab remapping info")
    print(f"  New vocab size: {len(vocab_remap)}")
    print(f"  Old vocab size: {max(vocab_remap) + 1}")
    
    # Check if there's frequency info
    if 'token_frequencies' in p1_ckpt:
        print(f"  ✓ Has token frequency data")
        freqs = p1_ckpt['token_frequencies']
        print(f"  Total tokens with frequency data: {len(freqs)}")
    else:
        print(f"  ✗ No token frequency data in checkpoint")
    
    # Check what criterion was used
    if 'pruning_criterion' in p1_ckpt:
        print(f"  Pruning criterion: {p1_ckpt['pruning_criterion']}")
    else:
        print(f"  ✗ No pruning criterion recorded")
    
    # Check coverage
    if 'vocab_coverage' in p1_ckpt:
        print(f"  Vocab coverage: {p1_ckpt['vocab_coverage']:.2%}")
    else:
        print(f"  ✗ No coverage info")
else:
    print("✗ Could not load Phase 1 checkpoint or no vocab remap found")
    print("\nThis means we can't verify how vocab pruning was done.")
```

### Cell 4: Check if Issue is with Specific Languages

```python
print("Checking <unk> rate by language pair...")

from collections import defaultdict

unk_by_pair = defaultdict(lambda: {'total': 0, 'unk': 0})

# Sample 50 entries
for i in range(50):
    sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)
    
    pair = f"{cache_entry['src_lang']}->{cache_entry['tgt_lang']}"
    teacher_seq = cache_entry['teacher_text_sequences']
    
    for token_id in teacher_seq.tolist():
        unk_by_pair[pair]['total'] += 1
        if token_id not in old_to_new:
            unk_by_pair[pair]['unk'] += 1

print(f"\n{'='*70}")
print(f"<UNK> RATE BY LANGUAGE PAIR")
print(f"{'='*70}")

for pair in sorted(unk_by_pair.keys()):
    stats = unk_by_pair[pair]
    unk_rate = 100 * stats['unk'] / stats['total'] if stats['total'] > 0 else 0
    print(f"{pair:<12} | Total: {stats['total']:>5} | <unk>: {stats['unk']:>4} | Rate: {unk_rate:>5.2f}%")
```

## What These Diagnostics Will Tell Us

1. **Cell 1**: Which specific tokens are being mapped to <unk> - are they rare tokens, punctuation, special characters?

2. **Cell 2**: Does the teacher generate different vocabulary than the ground truth references?

3. **Cell 3**: How was vocab pruning done in Phase 1? Was it frequency-based? What was the threshold?

4. **Cell 4**: Is the <unk> rate uniform across language pairs, or is it higher for specific languages?

## Expected Findings

### Scenario A: Vocab Pruning Was Too Aggressive
- Cell 1 will show common tokens being mapped to <unk>
- Cell 2 will show teacher uses tokens that ARE in references
- **Solution**: Lower the pruning threshold or rebuild cache with student model

### Scenario B: Teacher Generates Different Vocabulary
- Cell 1 will show tokens that are valid but not in training data
- Cell 2 will show teacher-only tokens that aren't in references
- **Solution**: Rebuild cache with student model (self-consistent)

### Scenario C: Specific Language Issue
- Cell 4 will show one language pair has much higher <unk> rate
- **Solution**: Check if that language was properly included in vocab pruning

## My Hypothesis

I suspect **Scenario B**: The base teacher model (trained on 200+ languages) generates text differently than your ground truth references, even for the same 5 languages. It might use:
- Different subword splits
- Synonyms or paraphrases
- More formal/informal language
- Different punctuation patterns

This is why **rebuilding the cache with the student model** is better - the student will generate text using only the vocabulary it knows, giving you 0% <unk> and self-consistent training.

## Run These Diagnostics

Run cells 1-4 above and share the output. This will tell us exactly why you're seeing 10% <unk> rate.
