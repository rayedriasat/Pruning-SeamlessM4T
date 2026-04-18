# Phase 6 Bug Fix: ASR-BLEU Scoring for T2U Pruning

## 🐛 The Bug

**Symptom:** All T2U layer removal candidates show identical ChrF scores (46.59), making it impossible to select which layer to prune.

**Root Cause:**
```python
# OLD CODE (BROKEN):
baseline_chrf = quick_eval_chrf(model, samples, tgt_lang, max_eval)
# ...
sc = quick_eval_chrf(model, samples, tgt_lang, max_eval)
```

`quick_eval_chrf()` calls `run_s2t_only()`, which:
1. Disables the vocoder (replaces it with NoOp)
2. Runs only: `speech_encoder → text_decoder`
3. Returns text tokens only (no audio generation)

**Why this breaks T2U pruning:**
- T2U layers are in the path: `text_decoder → t2u_model → vocoder → audio`
- Text-only evaluation **completely bypasses T2U**
- Removing T2U layers has **zero effect** on text decoder output
- Result: All candidates score identically (the bug you observed)

---

## ✅ The Fix

**Use ASR-ChrF scoring for pruning decisions** (fast, single metric):

```python
# NEW CODE (CORRECT):
def quick_eval_asr_chrf(model, samples, tgt_lang='ben', max_eval=10):
    """
    Evaluate S2ST quality using ASR-ChrF.
    Returns: avg_asr_chrf (used for pruning decisions)
    """
    _ensure_mms_loaded()  # Load MMS-ASR (Bengali)
    scores = []
    
    for s in samples[:max_eval]:
        # Full S2ST: generates audio
        pred_text, out_wav = run_s2st(model, s['wav'], tgt_lang=tgt_lang)
        
        if out_wav is not None and len(out_wav) > 1600:
            # Transcribe output audio with MMS-ASR
            _, asr_chrf = compute_asr_chrf(out_wav, s['ref'], sr=16000)
            scores.append(asr_chrf)
        else:
            scores.append(0.0)  # No audio = failure
    
    return float(np.mean(scores))
```

**Why this works:**
1. ✅ Runs **full S2ST pipeline** (speech → text → units → audio)
2. ✅ T2U layers **directly affect** the audio output
3. ✅ ASR-ChrF measures whether audio contains correct Bengali words
4. ✅ Different layer removals → different audio quality → different scores
5. ✅ **Single metric** (ChrF) for pruning decisions — simpler and faster
6. ✅ ASR-BLEU computed separately during final benchmarking

---

## 📊 Expected Behavior After Fix

**Before (broken):**
```
Iter 1/2 (6 layers remain)
  Remove L0 -> ChrF=46.59  ← All identical!
  Remove L1 -> ChrF=46.59
  Remove L2 -> ChrF=46.59
  Remove L3 -> ChrF=46.59
  Remove L4 -> ChrF=46.59
  Remove L5 -> ChrF=46.59
```

**After (fixed):**
```
Iter 1/2 (6 layers remain)
  Remove L0 -> ASR-ChrF=28.45  ← Now varies!
  Remove L1 -> ASR-ChrF=31.22
  Remove L2 -> ASR-ChrF=29.87
  Remove L3 -> ASR-ChrF=30.15  ← Best (highest)
  Remove L4 -> ASR-ChrF=27.93
  Remove L5 -> ASR-ChrF=26.51
-> Removed L3 (keeps highest audio quality)
```

---

## 🔧 How to Apply the Fix

### Option 1: Replace in Jupyter Notebook
1. Open `cse465v5-s2st-corrected.ipynb`
2. Find **Phase 6 Cell 3** (search for `def iterative_prune_t2u_stack`)
3. Replace the entire cell with the code from `phase6_fix_asr_scoring.py`

### Option 2: Delete Corrupt Checkpoints and Re-run
```python
# Run this in a notebook cell to wipe Phase 6 and restart:
import os, glob, shutil

# Delete local checkpoints
for f in glob.glob(f'{CKPT_DIR}/phase6*.pt'):
    os.remove(f)
    print(f'Deleted: {f}')

# Delete saved model
if os.path.exists(f'{MODEL_DIR}/phase6_t2u_iter_pruned'):
    shutil.rmtree(f'{MODEL_DIR}/phase6_t2u_iter_pruned')
    print('Deleted phase6 model')

# Clear from Drive (Kaggle only)
if ON_KAGGLE:
    subprocess.run(f'rclone purge "{GDRIVE_ROOT}/checkpoints/phase6*.pt"', shell=True)
    subprocess.run(f'rclone purge "{GDRIVE_ROOT}/models/phase6_t2u_iter_pruned"', shell=True)

print('Phase 6 cleared. Apply the fix, then re-run Phase 6 Cell 4.')
```

---

## 🎯 Why ASR-ChrF is the Right Metric

| Metric | Path Evaluated | Detects T2U Changes? | Use Case |
|--------|---------------|---------------------|----------|
| **Text ChrF** (old) | speech → text decoder | ❌ No (bypasses T2U) | - |
| **ASR-ChrF** (new) | speech → text → T2U → audio → ASR | ✅ Yes (full S2ST) | Pruning decisions |
| **ASR-BLEU** | speech → text → T2U → audio → ASR | ✅ Yes (full S2ST) | Final benchmarking |

**ASR-ChrF for pruning because:**
- ✅ Measures what the user actually hears (audio output)
- ✅ Single metric = simpler, faster decisions
- ✅ ChrF is more stable than BLEU for short sequences
- ✅ Correctly reflects T2U layer importance

**ASR-BLEU for benchmarking because:**
- ✅ Standard metric for S2ST quality comparison
- ✅ Reported alongside ASR-ChrF in final results table

---

## 📝 Summary

**Before:** Phase 6 was selecting layers randomly (all scored 46.59)  
**After:** Phase 6 selects layers that preserve audio quality (ASR-ChrF)

**Pruning metric:** ASR-ChrF only (fast, single metric for decisions)  
**Benchmark metrics:** ASR-BLEU + ASR-ChrF (comprehensive quality report)

This aligns Phase 6 with Phase 7's training objective and ensures the compressed model maintains high S2ST quality.
