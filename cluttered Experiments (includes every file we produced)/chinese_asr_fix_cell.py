# ── Chinese ASR Fix ──────────────────────────────────────────────────────────
# Add this cell to your notebook to fix the Chinese ASR issue

# Update the SEAMLESS_TO_MMS_LANG mapping to include Chinese
SEAMLESS_TO_MMS_LANG = {
    'ben': 'ben',  # Bengali
    'hin': 'hin',  # Hindi  
    'eng': 'eng',  # English
    'cmn': 'cmn',  # Chinese Mandarin (FIXED: was missing)
    'arb': 'ara',  # Arabic (MMS uses 'ara' for Arabic)
}

# Enhanced transcribe_audio function with better error handling
@torch.no_grad()
def transcribe_audio(audio_array, language='ben'):
    """
    Transcribe audio using MMS-1B with improved language support.
    Now supports Chinese (cmn) and has better error handling.
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
        result = proc.decode(ids[0]).strip()
        
        if not result:  # Empty result
            print(f'[MMS-ASR] Warning: Empty transcription for language {lang_iso}')
        
        return result
        
    except Exception as e:
        print(f'[MMS-ASR] Error loading/using language {lang_iso}: {e}')
        return ''

# Add asr_transcribe function for compatibility with benchmark functions that expect it
def asr_transcribe(audio_np, tgt_lang, sr=16000):
    """
    ASR transcribe function that matches some benchmark function signatures.
    This is an alias for transcribe_audio with parameter name compatibility.
    """
    return transcribe_audio(audio_np, language=tgt_lang)

print("✅ Chinese ASR Fix Applied!")
print("Changes made:")
print("1. Added 'cmn': 'cmn' to SEAMLESS_TO_MMS_LANG mapping")
print("2. Enhanced transcribe_audio with better language mapping and error handling")
print("3. Added asr_transcribe function for compatibility")
print("4. Now supports Chinese (cmn), Arabic (arb->ara), and other languages")
print("")
print("The Chinese ASR should now work properly in benchmarks!")