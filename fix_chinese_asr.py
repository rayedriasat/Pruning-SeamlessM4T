# Fix for Chinese ASR issue in cse465v6-s2st-optimised.ipynb

# The issue is that SEAMLESS_TO_MMS_LANG mapping is missing Chinese language code
# Current mapping: {'ben': 'ben', 'hin': 'hin', 'eng': 'eng'}
# Missing: 'cmn' (Chinese Mandarin)

# Solution: Add Chinese language code to the mapping and update the transcribe_audio function

# 1. Updated SEAMLESS_TO_MMS_LANG mapping
SEAMLESS_TO_MMS_LANG_FIXED = {
    'ben': 'ben',  # Bengali
    'hin': 'hin',  # Hindi  
    'eng': 'eng',  # English
    'cmn': 'cmn',  # Chinese Mandarin (ADDED)
    'arb': 'ara',  # Arabic (MMS uses 'ara' for Arabic)
}

# 2. Updated transcribe_audio function with better error handling
def transcribe_audio_fixed(audio_array, language='ben'):
    """
    Transcribe audio using MMS-1B with improved language support and error handling.
    """
    # Language mapping for MMS compatibility
    lang_map = {
        'bn': 'ben', 'hi': 'hin', 'en': 'eng', 
        'cmn': 'cmn', 'zh': 'cmn',  # Chinese variants
        'ar': 'ara', 'arb': 'ara'   # Arabic variants
    }
    
    # Get the correct MMS language code
    lang_iso = lang_map.get(language, language)
    
    try:
        model, proc = get_mms(lang_iso)
        arr = audio_array.astype(np.float32)
        inputs = proc(arr, sampling_rate=SAMPLE_RATE, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.autocast(DEVICE, dtype=DTYPE):
            logits = model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        return proc.decode(ids[0]).strip()
        
    except Exception as e:
        print(f'[MMS-ASR] Error loading/using language {lang_iso}: {e}')
        return ''

# 3. Alternative: Add asr_transcribe function that matches the benchmark expectation
def asr_transcribe(audio_np, tgt_lang, sr=16000):
    """
    ASR transcribe function that matches the benchmark function signature.
    Routes to the appropriate ASR backend based on target language.
    """
    if audio_np is None or len(audio_np) < 800:
        return ''
    
    # Language mapping for different ASR backends
    lang_mapping = {
        'ben': 'ben',  # Bengali -> MMS
        'hin': 'hin',  # Hindi -> MMS
        'eng': 'eng',  # English -> MMS
        'cmn': 'cmn',  # Chinese -> MMS
        'arb': 'ara',  # Arabic -> MMS (note: MMS uses 'ara')
    }
    
    mms_lang = lang_mapping.get(tgt_lang, tgt_lang)
    
    try:
        return transcribe_audio_fixed(audio_np, language=mms_lang)
    except Exception as e:
        print(f'[ASR] Error ({tgt_lang}): {e}')
        return ''

print("Chinese ASR Fix Ready!")
print("Issues identified:")
print("1. SEAMLESS_TO_MMS_LANG missing 'cmn' mapping")
print("2. transcribe_audio function needs better error handling for unsupported languages")
print("3. Benchmark function expects asr_transcribe but notebook has transcribe_audio")
print("")
print("Solutions provided:")
print("1. Updated SEAMLESS_TO_MMS_LANG_FIXED with Chinese support")
print("2. Enhanced transcribe_audio_fixed with better language mapping")
print("3. Added asr_transcribe function that matches benchmark expectations")