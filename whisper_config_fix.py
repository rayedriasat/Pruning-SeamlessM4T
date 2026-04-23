# SOLUTION: The max_length=448 is coming from Whisper's default generation config

# Add this cell right after loading Whisper model to fix the issue:

def fix_whisper_max_length_issue():
    """Fix the max_length issue in Whisper model"""
    global _whisper_model
    
    if _whisper_model is not None:
        print("Fixing Whisper generation config...")
        
        # The issue is in the model's generation_config
        gen_config = _whisper_model.generation_config
        
        # Check current values
        print(f"Current max_length: {getattr(gen_config, 'max_length', 'Not set')}")
        print(f"Current max_new_tokens: {getattr(gen_config, 'max_new_tokens', 'Not set')}")
        
        # Fix: Remove max_length from generation config
        if hasattr(gen_config, 'max_length'):
            delattr(gen_config, 'max_length')
            print("Removed max_length from generation config")
        
        # Or alternatively, set it to None
        gen_config.max_length = None
        
        # Ensure max_new_tokens is properly set
        gen_config.max_new_tokens = 256
        
        print("Whisper generation config fixed!")
    else:
        print("Whisper model not loaded yet")

# Alternative approach: Modify the _ensure_whisper_loaded function
def _ensure_whisper_loaded_fixed():
    global _whisper_model, _whisper_processor
    if _whisper_model is not None: return
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    print('[Whisper] Loading openai/whisper-medium...')
    _whisper_processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    _whisper_model = WhisperForConditionalGeneration.from_pretrained(
        'openai/whisper-medium', torch_dtype=torch.float16)
    _whisper_model = _whisper_model.eval()
    
    # FIX: Modify generation config right after loading
    gen_config = _whisper_model.generation_config
    gen_config.max_length = None  # Remove the default max_length
    gen_config.max_new_tokens = 256
    print(f"[Whisper] Fixed generation config: max_length=None, max_new_tokens=256")
    
    try:
        device = 'cuda:1' if N_GPU > 1 else 'cuda:0'
        _whisper_model = _whisper_model.to(device)
    except RuntimeError:
        pass
    print('[Whisper] Ready.')

print("Solutions created:")
print("1. Run fix_whisper_max_length_issue() after Whisper is loaded")
print("2. Or replace _ensure_whisper_loaded with _ensure_whisper_loaded_fixed")
print("3. The issue is Whisper's default generation_config has max_length=448")