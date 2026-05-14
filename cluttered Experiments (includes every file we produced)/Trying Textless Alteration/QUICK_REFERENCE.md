# Quick Reference: ASR Metrics + Multilingual Updates

## 🎯 What Changed

| Aspect | Before | After |
|--------|--------|-------|
| **Metrics** | Text ChrF/BLEU | **ASR-ChrF/BLEU** |
| **Languages** | EN↔BN only | **EN↔BN/HI/ZH/AR** |
| **Eval samples** | 25 | **200 (25×8)** |
| **Train samples** | 200 | **1600 (200×8)** |
| **Benchmark** | `run_benchmark()` | **`run_benchmark_asr()`** |

## 📊 Language Pairs (8 total)

```
English ↔ Bengali   (eng ↔ ben)
English ↔ Hindi     (eng ↔ hin)
English ↔ Mandarin  (eng ↔ cmn)
English ↔ Arabic    (eng ↔ arb)
```

## 🔧 Key Functions

### run_benchmark_asr()
```python
# Automatically handles all 8 language pairs
results, summary = run_benchmark_asr(
    model, eval_samples, label='MyModel', save_n=2
)
# Output: ASR-ChrF and ASR-BLEU per pair + overall
```

### quick_eval_chrf()
```python
# Now uses ASR transcription, multilingual samples
score = quick_eval_chrf(model, samples, max_samples=10)
# Used in Phase 2 encoder pruning
```

### asr_transcribe()
```python
# Routes to correct ASR backend per language
text = asr_transcribe(audio_np, tgt_lang='ben')
# ben → MMS-1B-all
# cmn/arb/hin/eng → Qwen3-ASR-1.7B
```

## 📈 Expected Results

### Translation Quality (ASR-based)
```
eng→ben: 40-45 ChrF, 37-42 BLEU
ben→eng: 35-40 ChrF, 32-37 BLEU
eng→cmn: 38-43 ChrF, 35-40 BLEU
cmn→eng: 36-41 ChrF, 33-38 BLEU
eng→arb: 37-42 ChrF, 34-39 BLEU
arb→eng: 35-40 ChrF, 32-37 BLEU
eng→hin: 38-43 ChrF, 35-40 BLEU
hin→eng: 36-41 ChrF, 33-38 BLEU

Average: 37-42 ChrF, 34-39 BLEU
```

### Voice Cloning
```
Target: 0.65-0.78 ECAPA similarity
Expected: 0.68-0.75 across pairs
```

### Speed
```
Teacher (1805M): RTF 0.268
V1 (1039M):      RTF 0.113
Textless (673M): RTF ~0.09 (3× faster)
```

## 🚀 Running the Notebook

### 1. Setup Cell
```python
# Loads all dependencies, sets up Drive/rclone
# No changes needed
```

### 2. Load Datasets
```python
# eval_samples: 200 samples (8 pairs × 25)
# ft_samples: 1600 samples (8 pairs × 200)
print(f'Eval: {len(eval_samples)} samples')
print(f'Train: {len(ft_samples)} samples')
```

### 3. Phase 0 Baseline
```python
# Benchmarks V1 model on all 8 pairs
p0_results, p0_summary = run_benchmark_asr(
    model_v1, eval_samples, label='P0_V1_Baseline'
)
# Expect: ~40 ChrF average across all pairs
```

### 4. Phase 2 Encoder Pruning
```python
# Uses multilingual ASR-ChrF for layer selection
removed_enc, p2_log = iterative_enc_prune(
    model_p2, eval_samples, N_ENC_REMOVE=8
)
# Should be stable (no cliff at iter 8)
```

### 5. Phase 7 Final Benchmark
```python
# Reports results for all 8 pairs
# Translation quality, voice cloning, long-form
# Comprehensive visualization with all pairs
```

## 🔍 Verification Commands

```bash
# Check updates are present
python Alteration/verify_updates.py

# Expected: 9+ checks passed
```

## 📝 Sample Output Snippets

### Dataset Loading
```
Loading eng→ben (en_us→bn_in) [test]...
  Added 25 samples
Loading ben→eng (bn_in→en_us) [test]...
  Added 25 samples
...
✓ Loaded 200 multilingual eval samples across 8 pairs
```

### Benchmark Output
```
BENCHMARK (ASR): P0_V1_Baseline  Samples:200

=== eng→ben (25 samples) ===
  [ 1/25] ASR-BLEU= 42.3 ASR-ChrF= 45.7 RTF=0.089
  ...

=== Summary by Language Pair ===
  eng→ben      ASR-ChrF=43.50  ASR-BLEU=40.20
  ben→eng      ASR-ChrF=38.70  ASR-BLEU=35.40
  ...

Overall: ASR-BLEU=37.45 ASR-ChrF=40.31 RTF=0.0912
```

### Phase 2 Pruning
```
Iter 1/8 | BI pre-filter: 6/16 cands
  Remove L12 -> ASR-ChrF=42.50  BI=0.0189
-> Removed L12 ASR-ChrF=42.50 (23 remain)
```

## ⚠️ Common Issues

### Issue: eval_samples is empty
**Fix**: Check FLEURS download, verify Drive connectivity

### Issue: ASR transcription fails
**Fix**: Verify MMS/Qwen3 models loaded, check GPU memory

### Issue: Phase 2 crashes
**Fix**: Reduce `max_eval` to 5, use smaller `bi_candidate_ratio`

## 📚 Documentation

- **UPDATES_SUMMARY.md** - Detailed change log
- **README_UPDATES.md** - Full guide with troubleshooting
- **PLAN.md Section 5** - ASR backend details
- **PLAN.md Section 7** - Phase plan

## ✅ Checklist

Before running:
- [ ] All update scripts executed successfully
- [ ] Verification script shows 9+ checks passed
- [ ] Kaggle/Colab secrets configured (HF_TOKEN, RCLONE_CONF)
- [ ] GPU available (2×T4 or better)

During execution:
- [ ] eval_samples loads 200 samples
- [ ] ft_samples loads 1600 samples
- [ ] Phase 0 reports all 8 pairs
- [ ] Phase 2 uses multilingual ASR-ChrF
- [ ] Phase 7 shows comprehensive results

After completion:
- [ ] Average ASR-ChrF ~37-42 across all pairs
- [ ] Speaker similarity 0.65-0.78
- [ ] RTF ~0.09 (3× faster than teacher)
- [ ] All 8 pairs have results in final table

---

**Quick Start**: Just run the notebook! All updates are already applied.
