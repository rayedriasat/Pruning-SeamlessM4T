#!/usr/bin/env python3
"""
Fix Qwen3-ASR loading to use the correct qwen-asr package
According to https://huggingface.co/Qwen/Qwen3-ASR-1.7B
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
    
    # Step 1: Add qwen-asr to pip install
    for i, cell in enumerate(nb['cells']):
        source_text = ''.join(cell.get('source', []))
        
        if "subprocess.run([" in source_text and "'pip', 'install'" in source_text and "transformers" in source_text:
            # Add qwen-asr to the install list
            if "'qwen-asr'" not in source_text:
                new_source = source_text.replace(
                    "'soundfile', 'requests', 'pandas',",
                    "'soundfile', 'requests', 'pandas', 'qwen-asr',"
                )
                cell['source'] = [new_source]
                updates_made += 1
                print("✓ Added 'qwen-asr' to pip install")
    
    # Step 2: Replace Qwen3-ASR loading with correct qwen-asr package usage
    for i, cell in enumerate(nb['cells']):
        source_text = ''.join(cell.get('source', []))
        
        if "# ── Qwen3-ASR-1.7B for ZH / AR / HI / EN" in source_text or "def _ensure_qwen_loaded():" in source_text:
            # Replace with correct qwen-asr package usage
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
    _ensure_qwen_loaded()
    if audio_np is None or len(audio_np) < 400: return ''
    
    # Resample if needed
    if sr != 16000:
        audio_np = torchaudio.functional.resample(
            torch.tensor(audio_np), sr, 16000).numpy()
    
    # Language mapping for Qwen3-ASR
    lang_map = {
        'zh': 'Chinese',
        'ar': 'Arabic', 
        'hi': 'Hindi',
        'en': 'English'
    }
    qwen_lang = lang_map.get(lang, 'English')
    
    # Transcribe using qwen-asr package
    # Pass audio as (numpy_array, sample_rate) tuple
    results = _qwen_model.transcribe(
        audio=(audio_np, 16000),
        language=qwen_lang,
    )
    
    return results[0].text.strip() if results else ''
'''
            cell['source'] = [new_qwen_code]
            updates_made += 1
            print("✓ Replaced Qwen3-ASR loading with correct qwen-asr package usage")
    
    if updates_made > 0:
        write_notebook(nb_path, nb)
        print(f"\n✅ Successfully made {updates_made} updates to {nb_path}")
        print("\nChanges:")
        print("  • Added 'qwen-asr' package to pip install")
        print("  • Updated Qwen3-ASR loading to use Qwen3ASRModel.from_pretrained()")
        print("  • Fixed transcription to use model.transcribe() API")
        print("\nNext steps:")
        print("  1. Restart the notebook kernel")
        print("  2. Re-run the pip install cell (will install qwen-asr)")
        print("  3. Re-run the ASR loading cells")
        print("  4. Continue with benchmark")
        print("\nNote: qwen-asr package handles model loading correctly")
    else:
        print("\n⚠️  No updates made - checking current state...")
        
        has_qwen_asr = False
        has_qwen3asr_model = False
        
        for cell in nb['cells']:
            source_text = ''.join(cell.get('source', []))
            if "'qwen-asr'" in source_text:
                has_qwen_asr = True
            if "from qwen_asr import Qwen3ASRModel" in source_text:
                has_qwen3asr_model = True
        
        print(f"  qwen-asr in pip install: {'✓' if has_qwen_asr else '✗'}")
        print(f"  Qwen3ASRModel usage: {'✓' if has_qwen3asr_model else '✗'}")

if __name__ == '__main__':
    main()
