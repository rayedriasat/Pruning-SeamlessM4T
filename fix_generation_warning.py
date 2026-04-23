# Fix for the max_new_tokens/max_length warning in Whisper ASR

def asr_transcribe_whisper_fixed(audio_np, lang='en', sr=16000):
    """
    Fixed version of asr_transcribe_whisper that removes the max_length warning
    """
    _ensure_whisper_loaded()
    if audio_np is None or len(audio_np) < 400: return ''
    
    # Resample if needed
    if sr != 16000:
        audio_np = torchaudio.functional.resample(
            torch.tensor(audio_np), sr, 16000).numpy()
    
    device = next(_whisper_model.parameters()).device
    dtype = next(_whisper_model.parameters()).dtype
    
    # Whisper language codes
    whisper_lang = 'zh' if lang == 'zh' else 'en'
    
    try:
        # Process audio - ensure correct dtype
        inputs = _whisper_processor(
            audio_np, 
            sampling_rate=16000, 
            return_tensors='pt',
            return_attention_mask=True)
        
        # Move to device and convert to model dtype
        input_features = inputs['input_features'].to(device).to(dtype)
        
        # Use modern task/language parameters - FIXED: removed max_length
        with torch.no_grad():
            predicted_ids = _whisper_model.generate(
                input_features,
                language=whisper_lang,
                task='transcribe',
                max_new_tokens=256,  # Only use max_new_tokens, not max_length
                num_beams=1,
                do_sample=False)
        
        transcription = _whisper_processor.batch_decode(
            predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except Exception as e:
        print(f'[Whisper] Error: {e}')
        import traceback
        traceback.print_exc()
        return ''

print("Fixed asr_transcribe_whisper function created.")
print("Replace the original function with this version to fix the warning.")