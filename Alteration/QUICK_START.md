# Quick Start - Run Your Benchmark Now!

## ⚠️ IMPORTANT: Restart Kernel First!

Before running anything, **restart your notebook kernel** to clear cached imports.

---

## 3 Simple Steps

### 1️⃣ Restart Kernel
Click: `Kernel` → `Restart Kernel` in Jupyter

### 2️⃣ Run Setup Cells (in order)
```
Cell 1:  Imports
Cell 2:  pip install (includes qwen-asr) ← Wait ~5 min
Cell 3-11: Model loading
Cell 12: Qwen3-ASR loading
Cell 13: ASR dispatcher ← NEW! Must run this!
Cell 14+: Benchmark functions
```

### 3️⃣ Run Benchmark
```python
p0_results, p0_summary = run_benchmark_asr(
    model_v1, eval_samples, label='P0_V1_Baseline', save_n=2
)
```

---

## What Was Fixed

✅ Added missing `asr_transcribe()` dispatcher function (Cell 13)
✅ Qwen3-ASR now uses correct `qwen-asr` package
✅ All 5 languages supported: English, Bengali, Hindi, Mandarin, Arabic
✅ 8 bidirectional translation pairs
✅ ASR-based metrics (ASR-ChrF, ASR-BLEU)

---

## Expected Output

```
[Qwen3-ASR] Loading Qwen/Qwen3-ASR-1.7B...
[Qwen3-ASR] Ready.
ASR stack ready (Qwen3-ASR-1.7B for all languages).

============================================================
  BENCHMARK (ASR): P0_V1_Baseline  Samples:200
============================================================

  === eng→ben (25 samples) ===
  [ 1/25] ASR-BLEU= 38.5 ASR-ChrF= 41.2 RTF=0.095
  ...

  === ben→eng (25 samples) ===
  [ 1/25] ASR-BLEU= 42.1 ASR-ChrF= 45.3 RTF=0.102
  ...

  (continues for all 8 language pairs)
```

---

## Troubleshooting

**"NameError: name 'asr_transcribe' is not defined"**
→ Restart kernel, re-run Cell 13

**"No module named 'qwen_asr'"**
→ Re-run Cell 2 (pip install), wait for completion

**"model type `qwen3_asr` not recognized"**
→ Restart kernel (clears old cached code)

---

## Documentation

- `COMPLETE_FIX_SUMMARY.md` - Full user guide
- `ASR_DISPATCHER_FIX.md` - Technical details
- `QWEN_ASR_FIX_SUMMARY.md` - Qwen3-ASR implementation

---

**Ready to go! Restart kernel and run!** 🚀
