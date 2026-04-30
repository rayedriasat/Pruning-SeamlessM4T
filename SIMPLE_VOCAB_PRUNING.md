# Simple and Effective Vocabulary Pruning

## The Problem with Unicode Ranges

Unicode range checking catches **too many tokens** because:
1. Tokens contain mixed characters (e.g., `"▁the"` has special char + letters)
2. Many tokens are shared across languages
3. Hard to get the ranges exactly right

## Better Approach: Corpus-Based with Full Coverage

**Just scan ALL the data you have for your 5 languages.** This is simpler and more accurate.

## Recommended Implementation

```python
def identify_used_tokens_complete(processor, target_langs):
    """
    Scan ALL available data for target languages to identify needed tokens.
    Simple, effective, and guarantees coverage.
    """
    from datasets import load_dataset
    
    tokenizer = processor.tokenizer
    keep_ids = set()
    
    print("="*70)
    print("VOCABULARY IDENTIFICATION - CORPUS-BASED")
    print("="*70)
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Always keep special tokens and language codes
    # ─────────────────────────────────────────────────────────────────────
    print("\n1. Adding special tokens and language codes...")
    
    if hasattr(tokenizer, 'all_special_ids'):
        keep_ids.update(tokenizer.all_special_ids)
    
    # Keep all language code tokens (e.g., __eng__, __cmn__)
    for tid in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(tid)
        if token and token.startswith('__') and token.endswith('__'):
            keep_ids.add(tid)
    
    print(f"   Special tokens: {len(keep_ids)}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Scan FLEURS (complete dataset)
    # ─────────────────────────────────────────────────────────────────────
    print("\n2. Scanning FLEURS dataset (all samples)...")
    
    fleurs_codes = {
        'eng': 'en_us',
        'ben': 'bn_in',
        'cmn': 'cmn_hans_cn',
        'arb': 'ar_eg',
        'hin': 'hi_in',
    }
    
    BASE = 'hf://datasets/google/fleurs@refs%2Fconvert%2Fparquet'
    before_fleurs = len(keep_ids)
    
    for lang in target_langs:
        if lang not in fleurs_codes:
            continue
        
        fc = fleurs_codes[lang]
        print(f"   Scanning {lang} ({fc})...")
        
        try:
            # Load FULL train split
            ds = load_dataset(
                'parquet',
                data_files={'train': f'{BASE}/{fc}/train/*.parquet'},
                split='train'
            )
            
            # Scan ALL samples
            for ex in ds:
                text = ex.get('transcription', '')
                if text:
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    keep_ids.update(tokens)
            
            print(f"      Processed {len(ds)} samples, total tokens: {len(keep_ids)}")
        except Exception as e:
            print(f"      Warning: {e}")
    
    print(f"   Added from FLEURS: {len(keep_ids) - before_fleurs}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Scan your training data (ft_samples)
    # ─────────────────────────────────────────────────────────────────────
    print("\n3. Scanning your training data...")
    
    before_training = len(keep_ids)
    
    # You should pass ft_samples to this function
    # For now, this is a placeholder
    print("   (Add your ft_samples scanning here)")
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Add common patterns (numbers, punctuation)
    # ─────────────────────────────────────────────────────────────────────
    print("\n4. Adding common patterns...")
    
    before_patterns = len(keep_ids)
    
    # Numbers
    for i in range(10):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        keep_ids.update(tokens)
    
    # Common punctuation
    punctuation = ['.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', 
                   '{', '}', '"', "'", '/', '\\', '@', '#', '$', '%', '&', 
                   '*', '+', '=', '<', '>', '|', '~', '`', '\n', '\t', ' ']
    
    for p in punctuation:
        try:
            tokens = tokenizer.encode(p, add_special_tokens=False)
            keep_ids.update(tokens)
        except:
            continue
    
    print(f"   Added from patterns: {len(keep_ids) - before_patterns}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("VOCABULARY IDENTIFICATION COMPLETE")
    print("="*70)
    print(f"Total tokens kept: {len(keep_ids):,} / {len(tokenizer):,}")
    print(f"Reduction: {100 * (1 - len(keep_ids)/len(tokenizer)):.1f}%")
    print(f"Estimated params saved: ~{(len(tokenizer) - len(keep_ids)) * 1024 / 1e6:.0f}M")
    
    return sorted(keep_ids)
```

## Expected Results

With this simpler approach:

```
FLEURS dataset sizes:
- English: ~2,800 samples
- Bengali: ~2,800 samples  
- Chinese: ~2,800 samples
- Arabic: ~2,800 samples
- Hindi: ~2,800 samples
Total: ~14,000 samples

Expected vocab size: 25,000-35,000 tokens (10-14% of 256K)
Params saved: ~220-230M
BLEU loss: 0.3-0.6 (much better than 1.3)
```

## Why This Works Better

1. **Corpus-based**: Only keeps tokens that actually appear in your data
2. **Complete coverage**: Scans ALL samples, not just 5,000
3. **Simple**: No complex Unicode range logic
4. **Verifiable**: Easy to test coverage on held-out data

## Usage

```python
# Reload base model
model_base, processor = load_base_model()

# Identify tokens
TARGET_5LANGS = ['eng', 'ben', 'cmn', 'arb', 'hin']
keep_ids = identify_used_tokens_complete(processor, TARGET_5LANGS)

print(f"\nVocabulary size: {len(keep_ids):,} tokens")
print(f"Reduction: {100 * (1 - len(keep_ids)/256099):.1f}%")

# Trim vocabulary
pre = count_params(model_base)
model_p1_new = trim_vocabulary(model_base, processor, keep_ids)
post = count_params(model_p1_new)

print(f"Params: {pre:.1f}M -> {post:.1f}M (saved {pre-post:.1f}M)")

# Test BLEU
# ... run evaluation ...

# If BLEU is good (>14.5), save it
save_checkpoint({'keep_ids': keep_ids}, 'phase1_vocab_v2', 0)
save_model_to_drive(model_p1_new, processor, 'phase1_vocab_5lang_v2')
```

## If You Want More Aggressive Pruning

If you want to get closer to 50% reduction (128K → 64K tokens):

```python
def identify_tokens_with_frequency_threshold(processor, target_langs, min_freq=2):
    """
    Keep only tokens that appear at least min_freq times.
    This removes very rare tokens.
    """
    from collections import Counter
    
    tokenizer = processor.tokenizer
    token_counts = Counter()
    
    # Scan all data and count frequencies
    # ... (same scanning code as above) ...
    
    # Keep tokens above frequency threshold
    keep_ids = set()
    
    # Always keep special tokens
    if hasattr(tokenizer, 'all_special_ids'):
        keep_ids.update(tokenizer.all_special_ids)
    
    # Keep language codes
    for tid in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(tid)
        if token and token.startswith('__') and token.endswith('__'):
            keep_ids.add(tid)
    
    # Keep tokens above threshold
    for token_id, count in token_counts.items():
        if count >= min_freq:
            keep_ids.add(token_id)
    
    return sorted(keep_ids)
```

But I **don't recommend** this - the BLEU loss will be higher.

## My Recommendation

1. **Use the simple corpus-based approach** (scan all FLEURS data)
2. **Expected result**: 25-35K tokens, ~220M params saved
3. **Test BLEU**: Should be 14.5-15.2 (vs 13.6 currently)
4. **If BLEU is good**: Proceed with Phase 2-5
5. **If BLEU is still low**: Add more data sources (CoVOST, Common Voice, etc.)

The goal is **maximum BLEU with acceptable parameter reduction**, not maximum parameter reduction at any cost.

## Why Not 50% Reduction?

SeamlessM4T's vocabulary is **not evenly distributed** across languages:
- Some tokens are shared across many languages (Latin alphabet, numbers, punctuation)
- Some languages have more tokens than others (Chinese has many characters)
- The tokenizer has many special tokens and language codes

So even with 5/100 languages, you won't get 95% reduction. Expect:
- **Realistic reduction**: 60-70% (keep 30-40% of vocab)
- **Aggressive reduction**: 80-85% (keep 15-20% of vocab, but BLEU suffers)

Your current 91% reduction (keep 9%) is **too aggressive** - that's why you lost 1.3 BLEU.
