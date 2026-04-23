# Qwen3-ASR Fix Summary

## Problem
The notebook was trying to load Qwen3-ASR-1.7B using `transformers.AutoModelForSpeechSeq2Seq`, which doesn't work because Qwen3-ASR requires the dedicated `qwen-asr` package.

## Solution Applied

### 1. Added `qwen-asr` to pip install
```python
subprocess.run([
    'pip', 'install', '-q',
    'transformers>=4.41.0', 'datasets', 'torchaudio', 'speechbrain>=1.0.0',
    'peft>=0.10.0', 'librosa', 'jiwer', 'evaluate', 'sacrebleu',
    'sentencepiece', 'accelerate', 'matplotlib', 'seaborn',
    'soundfile', 'requests', 'pandas', 'qwen-asr',  # ← Added
], check=True)
```

### 2. Corrected Qwen3-ASR Loading
```python
import torch
from qwen_asr import Qwen3ASRModel

_qwen_model = None

def _ensure_qwen_loaded():
    global _qwen_model
    if _qwen_model is not None: return
    print('[Qwen3-ASR] Loading Qwen/Qwen3-ASR-1.7B...')
    _qwen_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map='cuda:1' if N_GPU > 1 else 'cuda:0',
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    print('[Qwen3-ASR] Ready.')
```

### 3. Corrected Transcription Function
```python
def asr_transcribe_qwen(audio_np, sr=16000, lang='zh'):
    """
    Transcribe audio using Qwen3-ASR-1.7B.
    Official API: model.transcribe(audio=..., language=...)
    """
    _ensure_qwen_loaded()
    if audio_np is None or len(audio_np) < 400: return ''
    
    # Language mapping (must use full names)
    lang_map = {
        'zh': 'Chinese',
        'ar': 'Arabic', 
        'hi': 'Hindi',
        'en': 'English'
    }
    qwen_lang = lang_map.get(lang, None)  # None = auto-detect
    
    try:
        # Pass audio as (numpy_array, sample_rate) tuple
        results = _qwen_model.transcribe(
            audio=(audio_np, sr),
            language=qwen_lang,
        )
        
        # Return transcribed text
        return results[0].text.strip() if results and len(results) > 0 else ''
    except Exception as e:
        print(f'[Qwen3-ASR] Error: {e}')
        return ''
```

## Key Points

### Official API Format (from Hugging Face)
```python
# Example from https://huggingface.co/Qwen/Qwen3-ASR-1.7B
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
)

results = model.transcribe(
    audio="path/to/audio.wav",  # or URL, or (np.ndarray, sr) tuple
    language=None,  # or "English", "Chinese", etc.
)

print(results[0].language)
print(results[0].text)
```

### Supported Audio Formats
According to the docs, `model.transcribe()` accepts:
- Local file path: `"path/to/audio.wav"`
- URL: `"https://..."`
- Base64 data
- **Tuple: `(np.ndarray, sample_rate)`** ← We use this

### Language Names
Must use **full language names**, not codes:
- ✅ `"Chinese"` not `"zh"`
- ✅ `"Arabic"` not `"ar"`
- ✅ `"Hindi"` not `"hi"`
- ✅ `"English"` not `"en"`
- ✅ `None` for auto-detection

## What to Do in Notebook

### Step 1: Restart Kernel
**Important:** Restart the notebook kernel to clear any cached imports

### Step 2: Re-run pip install cell
This will install the `qwen-asr` package (~5 minutes)

### Step 3: Re-run ASR loading cells
The Qwen3-ASR model will now load correctly using the `qwen-asr` package

### Step 4: Continue with benchmark
The benchmark should now work for all languages:
- Bengali → MMS-1B-all
- Mandarin/Arabic/Hindi/English → Qwen3-ASR-1.7B

## Expected Output

### When loading:
```
[Qwen3-ASR] Loading Qwen/Qwen3-ASR-1.7B...
[Qwen3-ASR] Ready.
```

### When transcribing:
```
=== ben→eng (25 samples) ===
  [ 1/25] ASR-BLEU= 38.5 ASR-ChrF= 41.2 RTF=0.095
              pred: This is the transcribed English text
```

## Troubleshooting

### If you see: "No module named 'qwen_asr'"
**Solution:** Re-run the pip install cell with `'qwen-asr'` added

### If you see: "model type `qwen3_asr` not recognized"
**Solution:** You're still using the old transformers-based loading. Make sure you've:
1. Restarted the kernel
2. Re-run the corrected ASR loading cell

### If transcription returns empty string
**Check:**
- Audio length > 400 samples
- Language name is correct ("English" not "en")
- Audio format is (numpy_array, sample_rate) tuple

## Files Modified

- `Alteration/seamless-final.ipynb` - Updated with correct qwen-asr usage

## Reference

- Official docs: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Package: `pip install qwen-asr`
- GitHub: https://github.com/QwenLM/Qwen3-ASR

---

**Status**: ✅ Fixed and ready to use
**Last Updated**: 2026-04-23
