# Complete ASR configuration fix - replace the relevant section in your notebook

# ── M4T lang → ASR backend mapping ──────────────────────────────────────────
M4T_FLEURS_MAP = {
    'eng': 'en_us', 'ben': 'bn_in', 'cmn': 'cmn_hans_cn',
    'arb': 'ar_eg', 'hin': 'hi_in',
}

# MMS language codes - UPDATED to include Chinese
MMS_LANG_MAP = {
    'ben': 'ben',  # Bengali
    'hin': 'hin',  # Hindi
    'arb': 'ara',  # Arabic (MMS uses 'ara' for Arabic)
    'cmn': 'cmn',  # Chinese Mandarin (ADDED)
}

# UPDATED: Whisper only for English, MMS for all others
LANG_ASR_CONFIG = {
    'ben': ('mms', 'ben'),       # MMS for Bengali
    'hin': ('mms', 'hin'),       # MMS for Hindi
    'arb': ('mms', 'ara'),       # MMS for Arabic
    'cmn': ('mms', 'cmn'),       # MMS for Chinese (CHANGED from Whisper)
    'eng': ('whisper', 'en'),    # Whisper for English only
}

def asr_transcribe(audio_np, tgt_lang_m4t, sr=16000):
    """Route to correct ASR backend: Whisper for EN only, MMS for all others"""    
    if audio_np is None or len(audio_np) < 800: return ''
    backend, lang_code = LANG_ASR_CONFIG.get(tgt_lang_m4t, ('mms', 'eng'))  # Default to MMS
    try:
        if backend == 'mms':
            return asr_transcribe_mms(audio_np, lang_code, sr)
        else:  # whisper (only for English now)
            return asr_transcribe_whisper(audio_np, lang_code, sr)
    except Exception as e:
        print(f'[ASR] Error ({tgt_lang_m4t}): {e}')
        return ''

print('Updated ASR stack:')
print('  - Whisper-medium: English only')
print('  - MMS-1b-all: Bengali, Hindi, Arabic, Chinese')
print('')
print('Changes made:')
print('1. Updated LANG_ASR_CONFIG to use MMS for Chinese instead of Whisper')
print('2. Added Chinese (cmn) to MMS_LANG_MAP')
print('3. Updated asr_transcribe function default to use MMS')
print('4. Updated documentation strings')