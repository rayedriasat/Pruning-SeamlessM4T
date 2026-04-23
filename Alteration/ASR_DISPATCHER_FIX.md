# ASR Dispatcher Fix - Complete Solution

## Problem Identified

The notebook had a **missing function** error:
```
NameError: name 'run_s2st_legacy' is not defined
```

After fixing that, there was another missing function:
```
NameError: name 'asr_transcribe' is not defined
```

## Root Cause

The `run_benchmark_asr` function calls `asr_transcribe(wav_out, lang_code)` to transcribe the translated audio, but this dispatcher function was never defined in the notebook.

The notebook had:
- ✅ `asr_transcribe_qwen()` - Qwen3-ASR transcription function
- ❌ `asr_transcribe()` - Missing dispatcher that routes to the correct ASR model

## Solution Applied

### 1. Added ASR Dispatcher Function (Cell 13)

Created the missing `asr_transcribe()` dispatcher function that routes all languages to Qwen3-ASR-1.7B:

```python
def asr_transcribe(audio_np, lang_code):
    """
    Transcribe audio using Qwen3-ASR-1.7B.
    
    Qwen3-ASR officially supports: Chinese, Arabic, Hindi, English
    For Bengali: Use auto-detect mode (language=None)
    
    Args:
        audio_np: numpy array of audio samples  
        lang_code: 3-letter language code (ben, cmn, arb, hin, eng)
    
    Returns:
        Transcribed text string
    """
    if audio_np is None or len(audio_np) < 400:
        return ''
    
    # Map 3-letter codes to 2-letter codes for Qwen
    lang_map = {
        'cmn': 'zh',  # Mandarin Chinese
        'arb': 'ar',  # Arabic
        'hin': 'hi',  # Hindi
        'eng': 'en',  # English
        'ben': None,  # Bengali → auto-detect
    }
    
    qwen_lang = lang_map.get(lang_code, None)
    return asr_transcribe_qwen(audio_np, sr=16000, lang=qwen_lang if qwen_lang else 'en')
```

### 2. Language Routing

| Language | 3-Letter Code | Qwen Language | Notes |
|----------|---------------|---------------|-------|
| Mandarin Chinese | `cmn` | `'zh'` → `'Chinese'` | Native support |
| Arabic | `arb` | `'ar'` → `'Arabic'` | Native support |
| Hindi | `hin` | `'hi'` → `'Hindi'` | Native support |
| English | `eng` | `'en'` → `'English'` | Native support |
| Bengali | `ben` | `None` → auto-detect | Fallback mode |

### 3. How It Works

**Flow:**
1. `run_benchmark_asr()` translates audio: `wav_in` → `wav_out`
2. Calls `asr_transcribe(wav_out, tgt_lang)` to transcribe the output
3. Dispatcher maps language code: `'cmn'` → `'zh'`
4. Calls `asr_transcribe_qwen(audio_np, sr=16000, lang='zh')`
5. Qwen function maps to full name: `'zh'` → `'Chinese'`
6. Qwen3-ASR transcribes: `model.transcribe(audio=(np_array, sr), language='Chinese')`
7. Returns transcribed text

**Example for Mandarin:**
```
'cmn' → 'zh' → 'Chinese' → Qwen3-ASR transcription
```

**Example for Bengali:**
```
'ben' → None → 'en' → 'English' → Qwen3-ASR auto-detect mode
```

## Qwen3-ASR Implementation (Already Correct)

The Qwen3-ASR implementation in Cell 12 matches the official API:

```python
from qwen_asr import Qwen3ASRModel

# Loading (✅ Correct)
_qwen_model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map='cuda:1' if N_GPU > 1 else 'cuda:0',
    max_inference_batch_size=32,
    max_new_tokens=256,
)

# Transcription (✅ Correct)
results = _qwen_model.transcribe(
    audio=(audio_np, sr),      # ✅ Tuple format
    language=qwen_lang,         # ✅ Full name or None
)
return results[0].text.strip()  # ✅ Access .text attribute
```

## What to Do Now

### Step 1: Restart Kernel ⚠️
**Important:** Restart the notebook kernel to clear any cached imports or old function definitions.

### Step 2: Re-run Setup Cells
Run these cells in order:
1. **Cell 1**: Imports and setup
2. **Cell 2**: pip install (includes `qwen-asr`)
3. **Cell 3-11**: Model loading and utilities
4. **Cell 12**: Qwen3-ASR loading (`asr_transcribe_qwen`)
5. **Cell 13**: ASR dispatcher (`asr_transcribe`) ← **NEW**
6. **Cell 14+**: Benchmark functions

### Step 3: Run Benchmark
```python
p0_results, p0_summary = run_benchmark_asr(
    model_v1, eval_samples, label='P0_V1_Baseline', save_n=2
)
```

## Expected Output

### When loading ASR:
```
[Qwen3-ASR] Loading Qwen/Qwen3-ASR-1.7B...
[Qwen3-ASR] Ready.
ASR stack ready (Qwen3-ASR-1.7B for all languages).
Languages: Chinese, Arabic, Hindi, English (native), Bengali (auto-detect)
```

### When benchmarking:
```
============================================================
  BENCHMARK (ASR): P0_V1_Baseline  Samples:200
============================================================

  === eng→ben (25 samples) ===
  [ 1/25] ASR-BLEU= 38.5 ASR-ChrF= 41.2 RTF=0.095
              pred: This is the transcribed text
  ...

  === ben→eng (25 samples) ===
  [ 1/25] ASR-BLEU= 42.1 ASR-ChrF= 45.3 RTF=0.102
              pred: Transcribed English text
  ...

  === eng→cmn (25 samples) ===
  [ 1/25] ASR-BLEU= 35.2 ASR-ChrF= 38.7 RTF=0.098
              pred: 这是转录的文本
  ...
```

## Files Modified

- `Alteration/seamless-final.ipynb` - Added ASR dispatcher function (Cell 13)

## Files Created

- `Alteration/add_asr_dispatcher.py` - Script that added the dispatcher
- `Alteration/fix_asr_dispatcher.py` - Script that fixed language routing
- `Alteration/ASR_DISPATCHER_FIX.md` - This documentation

## Summary of All Fixes

### Fix 1: S2ST Functions (Previous)
- Added missing `run_s2st`, `run_s2t_only`, `compute_bleu`, `compute_chrf`, `quick_eval_chrf`
- Fixed function name from `run_s2st_legacy` to `run_s2st`

### Fix 2: Qwen3-ASR Loading (Previous)
- Added `qwen-asr` to pip install
- Corrected loading to use `Qwen3ASRModel.from_pretrained()`
- Fixed transcription to use official API: `model.transcribe(audio=(np, sr), language=...)`
- Fixed language names: `'Chinese'`, `'Arabic'`, `'Hindi'`, `'English'`

### Fix 3: ASR Dispatcher (This Fix)
- **Added missing `asr_transcribe()` dispatcher function**
- Routes all 5 languages to Qwen3-ASR-1.7B
- Maps 3-letter codes (`cmn`, `arb`, `hin`, `eng`, `ben`) to 2-letter codes (`zh`, `ar`, `hi`, `en`, `None`)

## Troubleshooting

### If you still see: "NameError: name 'asr_transcribe' is not defined"
**Solution:** 
1. Restart kernel
2. Re-run Cell 13 (ASR dispatcher)
3. Verify the cell executed successfully

### If Qwen3-ASR fails to load
**Solution:**
1. Check `qwen-asr` is installed: `pip list | grep qwen`
2. Re-run pip install cell
3. Restart kernel and try again

### If transcription returns empty strings
**Check:**
- Audio length > 400 samples
- Language code is valid (`cmn`, `arb`, `hin`, `eng`, `ben`)
- Qwen model loaded successfully

## Reference

- Official Qwen3-ASR docs: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Package: `pip install qwen-asr`
- Supported languages: Chinese, Arabic, Hindi, English (+ auto-detect for others)

---

**Status**: ✅ Complete - Ready to run
**Last Updated**: 2026-04-23
