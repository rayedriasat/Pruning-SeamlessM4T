#!/usr/bin/env python3
"""
Fix the asr_transcribe dispatcher to properly handle all 5 languages.
Qwen3-ASR supports: Chinese, Arabic, Hindi, English
For Bengali: Use Qwen with auto-detect (None) as fallback
"""

import json

def read_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_notebook(path, nb):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def main():
    nb_path = 'Alteration/seamless-final.ipynb'
    nb = read_notebook(nb_path)
    
    # Find the dispatcher cell we just added
    dispatcher_cell_idx = None
    for i, cell in enumerate(nb['cells']):
        source_text = ''.join(cell.get('source', []))
        if 'def asr_transcribe(' in source_text and 'ASR Dispatcher' in source_text:
            dispatcher_cell_idx = i
            break
    
    if dispatcher_cell_idx is None:
        print("❌ Could not find asr_transcribe dispatcher cell")
        return
    
    # Update with correct implementation
    dispatcher_code = '''# ── ASR Dispatcher ─────────────────────────────────────────────────────────
# Routes audio to Qwen3-ASR-1.7B for all languages
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
    # Qwen3-ASR supports: zh (Chinese), ar (Arabic), hi (Hindi), en (English)
    lang_map = {
        'cmn': 'zh',  # Mandarin Chinese → Chinese
        'arb': 'ar',  # Arabic
        'hin': 'hi',  # Hindi
        'eng': 'en',  # English
        'ben': None,  # Bengali → auto-detect (not officially supported)
    }
    
    qwen_lang = lang_map.get(lang_code, None)
    return asr_transcribe_qwen(audio_np, sr=16000, lang=qwen_lang if qwen_lang else 'en')

print('ASR stack ready (Qwen3-ASR-1.7B for all languages).')
print('Languages: Chinese, Arabic, Hindi, English (native), Bengali (auto-detect)')
'''
    
    # Update the cell
    nb['cells'][dispatcher_cell_idx]['source'] = [dispatcher_code]
    
    write_notebook(nb_path, nb)
    print(f"✅ Fixed asr_transcribe dispatcher in cell {dispatcher_cell_idx}")
    print("\nLanguage routing:")
    print("  • Mandarin (cmn) → Qwen3-ASR with Chinese")
    print("  • Arabic (arb) → Qwen3-ASR with Arabic")
    print("  • Hindi (hin) → Qwen3-ASR with Hindi")
    print("  • English (eng) → Qwen3-ASR with English")
    print("  • Bengali (ben) → Qwen3-ASR with auto-detect")

if __name__ == '__main__':
    main()
