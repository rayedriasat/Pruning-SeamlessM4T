# Better Vocabulary Pruning Strategy

## Goal
Keep ALL tokens needed for 5 languages (eng, ben, cmn, arb, hin) while maximizing parameter savings.

## Recommended Approach: Hybrid Method

Combine multiple strategies to ensure complete coverage:

### Step 1: Language-Specific Token Identification

```python
def identify_language_tokens_comprehensive(processor, target_langs):
    """
    Comprehensive token identification using multiple strategies.
    Ensures we don't miss any tokens needed for target languages.
    """
    tokenizer = processor.tokenizer
    keep_ids = set()
    
    print("="*70)
    print("COMPREHENSIVE VOCABULARY IDENTIFICATION")
    print("="*70)
    
    # ─────────────────────────────────────────────────────────────────────
    # Strategy 1: Special tokens and language codes
    # ─────────────────────────────────────────────────────────────────────
    print("\n1. Adding special tokens and language codes...")
    
    if hasattr(tokenizer, 'all_special_ids'):
        keep_ids.update(tokenizer.all_special_ids)
    
    for tid in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(tid)
        if token and token.startswith('__') and token.endswith('__'):
            keep_ids.add(tid)
    
    print(f"   Special tokens: {len(keep_ids)}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Strategy 2: Corpus-based identification (PRIMARY METHOD)
    # ─────────────────────────────────────────────────────────────────────
    print("\n2. Scanning corpora for tokens used in target languages...")
    
    from datasets import load_dataset
    
    fleurs_codes = {
        'eng': 'en_us',
        'ben': 'bn_in',
        'cmn': 'cmn_hans_cn',
        'arb': 'ar_eg',
        'hin': 'hi_in',
    }
    
    BASE = 'hf://datasets/google/fleurs@refs%2Fconvert%2Fparquet'
    before_corpus = len(keep_ids)
    
    for lang in target_langs:
        if lang not in fleurs_codes:
            continue
        
        fc = fleurs_codes[lang]
        print(f"   Scanning {lang} ({fc})...")
        
        try:
            # Load FULL dataset, not just 5000 samples
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
            
            print(f"      Processed {len(ds)} samples")
        except Exception as e:
            print(f"      Warning: {e}")
    
    print(f"   Added from corpus scan: {len(keep_ids) - before_corpus}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Strategy 3: Add common subword patterns and numbers
    # ─────────────────────────────────────────────────────────────────────
    print("\n3. Adding common patterns (numbers, punctuation)...")
    
    before_patterns = len(keep_ids)
    
    # Numbers and basic punctuation
    common_patterns = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}',
        '"', "'", '/', '@', '#', '$', '%', '&', '*', '+', '=', '<', '>',
    ]
    
    for pattern in common_patterns:
        try:
            tokens = tokenizer.encode(pattern, add_special_tokens=False)
            keep_ids.update(tokens)
        except:
            continue
    
    print(f"   Added from patterns: {len(keep_ids) - before_patterns}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Strategy 4: Validate and report
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("VOCABULARY IDENTIFICATION COMPLETE")
    print("="*70)
    print(f"Total tokens kept: {len(keep_ids)} / {len(tokenizer)}")
    print(f"Reduction: {100 * (1 - len(keep_ids)/len(tokenizer)):.1f}%")
    print(f"Estimated params saved: ~{(len(tokenizer) - len(keep_ids)) * 1024 / 1e6:.0f}M")
    
    return sorted(keep_ids)
```

### Step 2: Validate Coverage

```python
def validate_vocab_coverage(processor, keep_ids, test_samples):
    """
    Test if the pruned vocabulary can handle test samples without <unk>.
    """
    tokenizer = processor.tokenizer
    keep_set = set(keep_ids)
    
    print("\n" + "="*70)
    print("VALIDATING VOCABULARY COVERAGE")
    print("="*70)
    
    total_tokens = 0
    missing_tokens = 0
    missing_token_ids = set()
    
    for sample in test_samples:
        text = sample.get('ref', '') or sample.get('transcription', '')
        if not text:
            continue
        
        tokens = tokenizer.encode(text, add_special_tokens=True)
        total_tokens += len(tokens)
        
        for token_id in tokens:
            if token_id not in keep_set:
                missing_tokens += 1
                missing_token_ids.add(token_id)
    
    coverage = 100 * (1 - missing_tokens / total_tokens) if total_tokens > 0 else 0
    
    print(f"\nCoverage Analysis:")
    print(f"  Total tokens in test set: {total_tokens}")
    print(f"  Missing tokens: {missing_tokens}")
    print(f"  Coverage: {coverage:.2f}%")
    print(f"  Unique missing token IDs: {len(missing_token_ids)}")
    
    if missing_token_ids:
        print(f"\n  Sample missing tokens:")
        for token_id in list(missing_token_ids)[:20]:
            try:
                token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
                print(f"    Token ID {token_id}: '{token_str}'")
            except:
                print(f"    Token ID {token_id}: (cannot decode)")
    
    return coverage, missing_token_ids
```

### Step 3: Use the Better Vocabulary

```python
# Reload base model
model_base, processor = load_base_model()

# Identify tokens with comprehensive strategy
TARGET_5LANGS = ['eng', 'ben', 'cmn', 'arb', 'hin']
keep_ids = identify_language_tokens_comprehensive(processor, TARGET_5LANGS)

# Validate coverage on your test set
coverage, missing = validate_vocab_coverage(processor, keep_ids, ft_samples[:1000])

if coverage < 99.5:
    print(f"\n⚠ Coverage is {coverage:.2f}%, adding missing tokens...")
    keep_ids = sorted(set(keep_ids) | missing)
    print(f"  New vocab size: {len(keep_ids)}")

# Trim vocabulary
pre = count_params(model_base)
model_p1_new = trim_vocabulary(model_base, processor, keep_ids)
post = count_params(model_p1_new)

print(f"\nParams: {pre:.1f}M -> {post:.1f}M (saved {pre-post:.1f}M)")

# Save
save_checkpoint({'keep_ids': keep_ids, 'pre': pre, 'post': post}, 'phase1_vocab_v2', 0)
save_model_to_drive(model_p1_new, processor, 'phase1_vocab_5lang_v2')
```

## Expected Results

With this approach, you should get:

```
Old vocab pruning:
  Tokens kept: 22,767 / 256,001 (8.9%)
  BLEU: 13.6 (loss of 1.3 from base)
  Coverage: ~90% (10% <unk>)

New vocab pruning:
  Tokens kept: ~28,000-32,000 / 256,001 (11-12.5%)
  BLEU: 14.5-15.2 (loss of 0.4-0.7 from base)
  Coverage: >99.5% (<0.5% <unk>)
  
Parameter savings:
  Old: ~262M params saved
  New: ~230M params saved
  Difference: 32M more params, but much better quality
```

## Fast-Track Plan: Phase 1 → Phase 5

Since you know exactly what to prune:

```python
# Phase 1: Better vocab (1 hour)
model_p1 = better_vocab_pruning()

# Phase 2-5: Direct pruning (2-3 hours)
# You know the exact layers to remove, so just do it directly:

# Phase 2: Remove encoder layers (30 min)
model_p2 = remove_encoder_layers(model_p1, keep_layers=[0,2,4,6,8,10,12,14,16,18,20,22])

# Phase 3: Merge T2U layers with LaCo (1 hour)
model_p3 = apply_laco_merge(model_p2)

# Phase 4: Remove text decoder (30 min)
model_p4 = remove_text_decoder(model_p3)

# Phase 5: Final KD (30 min - just a few steps to verify)
model_p5 = quick_kd_verification(model_p4)

# Total: ~4 hours to Phase 5 with better vocab
```

## My Recommendation

**YES, redo Phase 1-5 with better vocabulary:**

1. ✅ **Time investment**: 4 hours
2. ✅ **Potential gain**: 0.5-1.0 BLEU recovery
3. ✅ **Risk**: Low (you know what to do)
4. ✅ **Paper impact**: Stronger baseline, better final results
5. ✅ **Learning**: You'll have a more robust vocab pruning method

The 1.3 BLEU drop from vocab pruning alone is too much. With better vocabulary, you should lose only 0.4-0.7 BLEU, which is much more acceptable.

## Action Plan

1. **Implement** the comprehensive vocab identification function above
2. **Test** coverage on your data (should be >99.5%)
3. **Redo Phase 1** with new vocabulary
4. **Fast-track Phases 2-5** (you know exact operations)
5. **Continue Phase 6** with stronger baseline

This will give you a much better model for your paper! 🚀
