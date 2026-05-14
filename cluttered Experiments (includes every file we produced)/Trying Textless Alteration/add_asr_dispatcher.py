#!/usr/bin/env python3
"""
Add the missing asr_transcribe dispatcher function that routes to Qwen3-ASR.
This function is called by run_benchmark_asr but was never defined.
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
    
    # Find the cell with asr_transcribe_qwen
    qwen_cell_idx = None
    for i, cell in enumerate(nb['cells']):
        source_text = ''.join(cell.get('source', []))
        if 'def asr_transcribe_qwen(' in source_text:
            qwen_cell_idx = i
            break
    
    if qwen_cell_idx is None:
        print("❌ Could not find asr_transcribe_qwen cell")
        return
    
    # Add the dispatcher function right after the Qwen cell
    dispatcher_code = '''# ── ASR Dispatcher ─────────────────────────────────────────────────────────
# Routes audio to the correct ASR model based on language
def asr_transcribe(audio_np, lang_code):
    """
    Transcribe audio using Qwen3-ASR-1.7B for all languages.
    
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
        'ben': 'en',  # Bengali → use English ASR (Qwen supports limited languages)
        'cmn': 'zh',  # Mandarin Chinese
        'arb': 'ar',  # Arabic
        'hin': 'hi',  # Hindi
        'eng': 'en',  # English
    }
    
    qwen_lang = lang_map.get(lang_code, 'en')
    return asr_transcribe_qwen(audio_np, sr=16000, lang=qwen_lang)

print('ASR stack ready (Qwen3-ASR for all languages).')
print('Note: Using Qwen3-ASR-1.7B for transcription of all target languages.')
'''
    
    # Create new cell
    new_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [dispatcher_code]
    }
    
    # Insert after the Qwen cell
    nb['cells'].insert(qwen_cell_idx + 1, new_cell)
    
    write_notebook(nb_path, nb)
    print(f"✅ Added asr_transcribe dispatcher function after cell {qwen_cell_idx}")
    print("\nThe dispatcher routes all languages to Qwen3-ASR-1.7B:")
    print("  • Bengali (ben) → Qwen3-ASR with English")
    print("  • Mandarin (cmn) → Qwen3-ASR with Chinese")
    print("  • Arabic (arb) → Qwen3-ASR with Arabic")
    print("  • Hindi (hin) → Qwen3-ASR with Hindi")
    print("  • English (eng) → Qwen3-ASR with English")

if __name__ == '__main__':
    main()
