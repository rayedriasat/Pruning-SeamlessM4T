#!/usr/bin/env python3
"""
Fix the run_benchmark_asr function to use the correct S2ST function name
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
    
    # Fix run_benchmark_asr to use correct function names
    for cell in nb['cells']:
        source_text = ''.join(cell.get('source', []))
        
        if 'def run_benchmark_asr(' in source_text and 'run_s2st_legacy' in source_text:
            # Replace run_s2st_legacy with the correct function
            new_source = source_text.replace('run_s2st_legacy', 'run_s2st')
            cell['source'] = [new_source]
            updates_made += 1
            print("✓ Fixed run_benchmark_asr to use run_s2st")
        
        # Also check if run_s2st function exists, if not add it
        if 'def run_s2t_only(' in source_text and 'def run_s2st(' not in source_text:
            # Add run_s2st function after run_s2t_only
            lines = source_text.split('\n')
            
            # Find where to insert
            insert_idx = -1
            for i, line in enumerate(lines):
                if 'def run_s2t_only(' in line:
                    # Find the end of this function
                    for j in range(i+1, len(lines)):
                        if lines[j].strip() and not lines[j].startswith(' ') and not lines[j].startswith('\t'):
                            insert_idx = j
                            break
                        elif j == len(lines) - 1:
                            insert_idx = j + 1
                            break
                    break
            
            if insert_idx > 0:
                # Insert run_s2st function
                run_s2st_code = '''
def run_s2st(mdl, wav, tgt_lang='ben'):
    """Full S2ST: returns (text, audio_numpy). Falls back to text-only if vocoder fails."""
    inputs = processor(audio=wav, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}
    orig_voc = mdl.vocoder
    try:
        with torch.no_grad():
            out = mdl.generate(**inputs, tgt_lang=tgt_lang,
                               return_intermediate_token_ids=True)
        text_ids = _remap_ids_for_decode(mdl, out.sequences.cpu())
        text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
        wav_out = out.waveform.cpu().numpy().squeeze() if out.waveform is not None else np.zeros(16000)
        return text, wav_out
    except RuntimeError as e:
        print(f'  Vocoder failed: {e}')
        mdl.vocoder = orig_voc
        return run_s2t_only(mdl, wav, tgt_lang), np.zeros(16000)
'''
                lines.insert(insert_idx, run_s2st_code)
                cell['source'] = ['\\n'.join(lines)]
                updates_made += 1
                print("✓ Added run_s2st function")
    
    # Save updated notebook
    write_notebook(nb_path, nb)
    print(f"\n✅ Successfully made {updates_made} fixes to {nb_path}")
    
    if updates_made == 0:
        print("\n⚠️  No fixes needed - checking if functions exist...")
        
        has_run_s2st = False
        has_run_s2t_only = False
        
        for cell in nb['cells']:
            source_text = ''.join(cell.get('source', []))
            if 'def run_s2st(' in source_text:
                has_run_s2st = True
            if 'def run_s2t_only(' in source_text:
                has_run_s2t_only = True
        
        print(f"  run_s2st function: {'✓ Found' if has_run_s2st else '✗ Missing'}")
        print(f"  run_s2t_only function: {'✓ Found' if has_run_s2t_only else '✗ Missing'}")
        
        if not has_run_s2st:
            print("\n❌ run_s2st function is missing!")
            print("   This function should be defined in the 'Benchmark functions ready' cell")
            print("   Please add it manually or check the original notebook")

if __name__ == '__main__':
    main()
