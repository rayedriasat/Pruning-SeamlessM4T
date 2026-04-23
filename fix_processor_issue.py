#!/usr/bin/env python3
"""
Fix for the processor None issue in seamless-final.ipynb
"""

# The issue is that the global processor variable becomes None
# Here's the fix to add to your notebook:

def ensure_processor_loaded():
    """Ensure processor is loaded and available globally"""
    global processor
    if processor is None:
        print("WARNING: processor is None, reloading...")
        from transformers import SeamlessM4TProcessor
        MODEL_NAME = 'facebook/seamless-m4t-v2-large'
        processor = SeamlessM4TProcessor.from_pretrained(MODEL_NAME)
        print("Processor reloaded successfully")
    return processor

def run_s2st_fixed(mdl, wav, tgt_lang='ben'):
    """Fixed version of run_s2st that ensures processor is loaded"""
    global processor
    
    # Ensure processor is loaded
    if processor is None:
        ensure_processor_loaded()
    
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
            return run_s2t_only_fixed(mdl, wav, tgt_lang), np.zeros(16000)

def run_s2t_only_fixed(mdl, wav, tgt_lang='ben'):
    """Fixed version of run_s2t_only that ensures processor is loaded"""
    global processor
    
    # Ensure processor is loaded
    if processor is None:
        ensure_processor_loaded()
    
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

def quick_eval_chrf_fixed(mdl, samples, max_samples=10):
    """Fixed version of quick_eval_chrf that ensures processor is loaded"""
    global processor
    
    # Ensure processor is loaded
    if processor is None:
        ensure_processor_loaded()
    
    scores = []
    for s in samples[:max_samples]:
        try:
            # Get target language from sample if available
            tgt = s.get('tgt_lang', 'ben')
            _, wav_out = run_s2st_fixed(mdl, s['wav'], tgt_lang=tgt)
            pred = asr_transcribe(wav_out, tgt)
            scores.append(compute_chrf(pred, s['ref']))
        except:
            scores.append(0.0)
    return float(np.mean(scores))

print("Fixed functions created. To use in your notebook:")
print("1. Add the ensure_processor_loaded() function")
print("2. Replace run_s2st with run_s2st_fixed")
print("3. Replace run_s2t_only with run_s2t_only_fixed") 
print("4. Replace quick_eval_chrf with quick_eval_chrf_fixed")
print("5. Or simply add this check before calling run_benchmark_asr:")
print("   if processor is None:")
print("       processor = SeamlessM4TProcessor.from_pretrained('facebook/seamless-m4t-v2-large')")