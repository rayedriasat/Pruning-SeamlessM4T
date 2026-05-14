#!/usr/bin/env python3
"""
Correct Qwen3-ASR implementation based on official example:
https://huggingface.co/Qwen/Qwen3-ASR-1.7B
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
    
    updates_made = 0
    
    # Find and fix the Qwen3-ASR transcription function
    for i, cell in enumerate(nb['cells']):
        source_text = ''.join(cell.get('source', []))
        
        if "def asr_transcribe_qwen(" in source_text and "qwen_asr" in source_text:
            # Fix the transcription function to match official API
            new_qwen_code = '''# ── Qwen3-ASR-1.7B for ZH / AR / HI / EN ───────────────────────────────────
# PLAN.md Section 5: Qwen3-ASR-1.7B is stronger than MMS for high-resource langs
# Official docs: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
_qwen_model = None

def _ensure_qwen_loaded():
    global _qwen_model
    if _qwen_model is not None: return
    import torch
    from qwen_asr import Qwen3ASRModel
    print('[Qwen3-ASR] Loading Qwen/Qwen3-ASR-1.7B...')
    _qwen_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map='cuda:1' if N_GPU > 1 else 'cuda:0',
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    print('[Qwen3-ASR] Ready.')

def asr_transcribe_qwen(audio_np, sr=16000, lang='zh'):
    """
    Transcribe audio using Qwen3-ASR-1.7B.
    Official API: model.transcribe(audio=..., language=...)
    Audio can be: local path, URL, base64, or (np.ndarray, sr) tuple
    """
    _ensure_qwen_loaded()
    if audio_np is None or len(audio_np) < 400: return ''
    
    # Language mapping for Qwen3-ASR (must use full language names)
    lang_map = {
        'zh': 'Chinese',
        'ar': 'Arabic', 
        'hi': 'Hindi',
        'en': 'English'
    }
    qwen_lang = lang_map.get(lang, None)  # None = auto-detect
    
    try:
        # Pass audio as (numpy_array, sample_rate) tuple
        # This is one of the supported formats per the docs
        results = _qwen_model.transcribe(
            audio=(audio_np, sr),
            language=qwen_lang,
        )
        
        # Return transcribed text
        return results[0].text.strip() if results and len(results) > 0 else ''
    except Exception as e:
        print(f'[Qwen3-ASR] Error: {e}')
        return ''
'''
            cell['source'] = [new_qwen_code]
            updates_made += 1
            print("✓ Fixed asr_transcribe_qwen to match official API")
    
    if updates_made > 0:
        write_notebook(nb_path, nb)
        print(f"\n✅ Successfully fixed {updates_made} function(s) in {nb_path}")
        print("\nCorrections:")
        print("  • Audio format: (numpy_array, sample_rate) tuple")
        print("  • Language: Full names ('Chinese', 'Arabic', 'Hindi', 'English')")
        print("  • Return: results[0].text (official API)")
        print("\nThis matches the official example from Hugging Face")
    else:
        print("\n✓ Code already matches official API")

if __name__ == '__main__':
    main()
