# PATCH: Fix the max_new_tokens/max_length warning
# Add this cell to replace the problematic function

def asr_transcribe_whisper(audio_np, lang='en', sr=16000):
    """
    Transcribe audio using Whisper-medium for English or Chinese.
    lang: 'en' for English, 'zh' for Chinese
    FIXED: Removed max_length parameter to avoid warning
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
        
        # FIXED: Only use max_new_tokens, removed max_length to avoid warning
        with torch.no_grad():
            predicted_ids = _whisper_model.generate(
                input_features,
                language=whisper_lang,
                task='transcribe',
                max_new_tokens=256,  # Only this parameter, no max_length
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

print("Fixed asr_transcribe_whisper function - warning should be gone now")