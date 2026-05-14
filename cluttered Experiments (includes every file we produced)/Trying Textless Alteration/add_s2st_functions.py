#!/usr/bin/env python3
"""
Add missing S2ST inference functions to the notebook
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
    
    # Find the cell with "Benchmark functions ready" and add S2ST functions before it
    for i, cell in enumerate(nb['cells']):
        source_text = ''.join(cell.get('source', []))
        
        # Find the cell that prints "Benchmark functions ready"
        if 'print(\'Benchmark functions ready.\')' in source_text or 'print("Benchmark functions ready.")' in source_text:
            # Check if S2ST functions already exist in this cell
            if 'def run_s2st(' not in source_text:
                # Add the S2ST functions at the beginning of this cell
                s2st_functions = '''from sacrebleu.metrics import BLEU, CHRF
_bleu = BLEU(effective_order=True)
_chrf = CHRF()

def find_layers_attr(component):
    for attr in ['layers', 'layer', 'inner_layers', 'encoder_layers', 'decoder_layers']:
        if hasattr(component, attr): return attr
    return None

def compute_bleu(hyp, ref):
    if not hyp.strip() or not ref.strip(): return 0.0
    return _bleu.sentence_score(hyp.strip(), [ref.strip()]).score

def compute_chrf(hyp, ref):
    if not hyp.strip() or not ref.strip(): return 0.0
    return _chrf.sentence_score(hyp.strip(), [ref.strip()]).score

def _remap_ids_for_decode(mdl, ids):
    if hasattr(mdl, '_vocab_remap_to_old'):
        remap = mdl._vocab_remap_to_old
        ids = ids.clone()
        mask = (ids >= 0) & (ids < len(remap))
        ids[mask] = remap[ids[mask]]
    return ids

def _model_input_device(mdl):
    if hasattr(mdl, 'speech_encoder'):
        return next(mdl.speech_encoder.parameters()).device
    return next(mdl.parameters()).device

def run_s2st(mdl, wav, tgt_lang='ben'):
    """Full S2ST for models with text decoder (Phases 0-3)."""
    inputs = processor(audio=wav, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}
    with torch.no_grad():
        try:
            out  = mdl.generate(**inputs, tgt_lang=tgt_lang,
                                return_intermediate_token_ids=True)
            text_ids = _remap_ids_for_decode(mdl, out.sequences.cpu())
            text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
            wav_out = out.waveform.cpu().numpy().squeeze() if out.waveform is not None else np.zeros(16000)
            return text, wav_out
        except RuntimeError:
            return run_s2t_only(mdl, wav, tgt_lang), np.zeros(16000)

def run_s2t_only(mdl, wav, tgt_lang='ben'):
    """Text-only generation (for benchmarking text-decoder models)."""
    inputs = processor(audio=wav, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}
    orig_voc = mdl.vocoder
    inp_dev  = next(iter(inputs.values())).device
    class _Noop(nn.Module):
        def forward(self, *a, **kw): return torch.zeros(1,1,device=inp_dev), [1]
    mdl.vocoder = _Noop()
    try:
        with torch.no_grad():
            out = mdl.generate(**inputs, tgt_lang=tgt_lang,
                               return_intermediate_token_ids=True)
    finally:
        mdl.vocoder = orig_voc
    text_ids = _remap_ids_for_decode(mdl, out.sequences.cpu())
    return processor.batch_decode(text_ids, skip_special_tokens=True)[0]

def quick_eval_chrf(mdl, samples, max_samples=10):
    """Quick ASR-ChrF evaluation for pruning decisions."""
    scores = []
    for s in samples[:max_samples]:
        try:
            # Get target language from sample if available
            tgt = s.get('tgt_lang', 'ben')
            _, wav_out = run_s2st(mdl, s['wav'], tgt_lang=tgt)
            pred = asr_transcribe(wav_out, tgt)
            scores.append(compute_chrf(pred, s['ref']))
        except:
            scores.append(0.0)
    return float(np.mean(scores))

'''
                # Prepend to existing source
                cell['source'] = [s2st_functions + '\n' + source_text]
                print("✓ Added S2ST inference functions (run_s2st, run_s2t_only, quick_eval_chrf)")
                
                # Save and exit
                write_notebook(nb_path, nb)
                print(f"\n✅ Successfully added S2ST functions to {nb_path}")
                return
    
    print("\n❌ Could not find 'Benchmark functions ready' cell")
    print("   Please add the S2ST functions manually")

if __name__ == '__main__':
    main()
