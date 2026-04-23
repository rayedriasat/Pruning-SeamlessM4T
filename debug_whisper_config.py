# Debug script to find where max_length=448 is coming from

def debug_whisper_generation_config():
    """Debug the Whisper model's generation configuration"""
    
    # Check if Whisper model is loaded
    if '_whisper_model' in globals() and _whisper_model is not None:
        print("=== Whisper Model Generation Config ===")
        gen_config = _whisper_model.generation_config
        print(f"Generation config type: {type(gen_config)}")
        
        # Check all attributes
        for attr in dir(gen_config):
            if not attr.startswith('_'):
                try:
                    value = getattr(gen_config, attr)
                    if 'length' in attr.lower() or 'token' in attr.lower():
                        print(f"{attr}: {value}")
                except:
                    pass
        
        print("\n=== Model Config ===")
        model_config = _whisper_model.config
        for attr in dir(model_config):
            if not attr.startswith('_'):
                try:
                    value = getattr(model_config, attr)
                    if 'length' in attr.lower() or 'token' in attr.lower():
                        print(f"{attr}: {value}")
                except:
                    pass
    else:
        print("Whisper model not loaded yet")

def fix_whisper_generation_config():
    """Fix the Whisper generation config to remove max_length"""
    
    if '_whisper_model' in globals() and _whisper_model is not None:
        print("Fixing Whisper generation config...")
        
        # Get the generation config
        gen_config = _whisper_model.generation_config
        
        # Remove or set max_length to None
        if hasattr(gen_config, 'max_length'):
            print(f"Original max_length: {gen_config.max_length}")
            gen_config.max_length = None
            print("Set max_length to None")
        
        # Ensure max_new_tokens is set
        gen_config.max_new_tokens = 256
        print(f"Set max_new_tokens to: {gen_config.max_new_tokens}")
        
        # Update the model's generation config
        _whisper_model.generation_config = gen_config
        
        print("Generation config updated!")
        return True
    else:
        print("Whisper model not loaded")
        return False

# Alternative fix: Override generation parameters completely
def asr_transcribe_whisper_override_config(audio_np, lang='en', sr=16000):
    """
    Whisper transcription with explicit generation config override
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
        # Process audio
        inputs = _whisper_processor(
            audio_np, 
            sampling_rate=16000, 
            return_tensors='pt',
            return_attention_mask=True)
        
        # Move to device and convert to model dtype
        input_features = inputs['input_features'].to(device).to(dtype)
        
        # EXPLICIT override of ALL generation parameters
        with torch.no_grad():
            predicted_ids = _whisper_model.generate(
                input_features,
                language=whisper_lang,
                task='transcribe',
                max_new_tokens=256,
                max_length=None,  # Explicitly set to None
                num_beams=1,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=False)
        
        transcription = _whisper_processor.batch_decode(
            predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except Exception as e:
        print(f'[Whisper] Error: {e}')
        import traceback
        traceback.print_exc()
        return ''

print("Debug functions created. Run:")
print("1. debug_whisper_generation_config() - to see current config")
print("2. fix_whisper_generation_config() - to fix the config")
print("3. Use asr_transcribe_whisper_override_config() as replacement function")