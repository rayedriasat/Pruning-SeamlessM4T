# Phase 4 Load Fix - Textless Architecture

## Problem

The error occurs because:
1. Phase 4 saves model with `vocab_size=0` (text decoder removed)
2. `load_model_from_drive()` calls `SeamlessM4Tv2ForSpeechToSpeech.from_pretrained()`
3. HuggingFace tries to create `nn.Embedding(0, hidden_size)` → crashes

```python
self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
# When vocab_size=0, this fails: "index 0 is out of bounds for dimension 0 with size 0"
```

## Solution

Create a custom loader that:
1. Loads base model first (with text decoder)
2. Applies surgical removal
3. Loads saved weights for remaining components

## Implementation

Add this function after `load_model_from_drive()`:

```python
def load_textless_model_from_drive(stage_name):
    """
    Load textless model (Phase 4+) that has no text decoder.
    Cannot use standard from_pretrained() because vocab_size=0 breaks nn.Embedding.
    """
    from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor
    
    local = f'{MODEL_DIR}/{stage_name}'
    if ON_KAGGLE and (not os.path.exists(local) or not os.listdir(local)):
        print(f'[model] Not in local cache — pulling from remote...')
        _rclone_pull_model(stage_name)
    
    if not os.path.exists(local) or not os.listdir(local):
        raise RuntimeError(f'[model] Not found or empty: {local}')
    
    print(f'[model] Loading textless model {stage_name} from {local} ...')
    
    # Load manifest to get metadata
    manifest_path = os.path.join(local, _PRUNING_MANIFEST)
    if os.path.exists(manifest_path):
        manifest = torch.load(manifest_path, map_location='cpu', weights_only=False)
        hidden = manifest.get('hidden', 1024)
        n_langs = manifest.get('n_langs', 36)
        print(f'  Manifest: hidden={hidden}, n_langs={n_langs}')
    else:
        hidden = 1024
        n_langs = 36
        print(f'  No manifest, using defaults: hidden={hidden}, n_langs={n_langs}')
    
    # Step 1: Load base model (with text decoder)
    print('  [1/5] Loading base model skeleton...')
    base_model, processor = load_base_model()
    base_model = _consolidate_to_single_gpu(base_model)
    
    # Step 2: Apply surgical removal (recreates textless architecture)
    print('  [2/5] Applying textless surgery...')
    model = remove_text_decoder_and_install_cif(base_model)
    
    # Step 3: Load saved weights
    print('  [3/5] Loading saved weights...')
    weight_file = None
    for fname in ['model.safetensors', 'pytorch_model.bin']:
        fpath = os.path.join(local, fname)
        if os.path.exists(fpath):
            weight_file = fpath
            break
    
    if weight_file is None:
        raise RuntimeError(f'No weight file found in {local}')
    
    if weight_file.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(weight_file)
    else:
        state_dict = torch.load(weight_file, map_location='cpu', weights_only=False)
    
    # Step 4: Load weights (strict=False because text decoder keys are missing)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # Filter out expected missing keys (text decoder components)
    expected_missing = ['text_decoder', 'lm_head', 'shared']
    actual_missing = [k for k in missing if not any(em in k for em in expected_missing)]
    
    if actual_missing:
        print(f'  ⚠ Unexpected missing keys: {len(actual_missing)}')
        for k in actual_missing[:10]:
            print(f'    - {k}')
    else:
        print(f'  ✓ All expected keys loaded (text decoder keys skipped as expected)')
    
    if unexpected:
        print(f'  ⚠ Unexpected keys: {len(unexpected)}')
    
    # Step 5: Load custom state
    print('  [4/5] Loading custom state...')
    _load_custom_state(model, local)
    
    # Step 6: Sync config
    print('  [5/5] Syncing config...')
    sync_model_config(model)
    
    model.eval()
    print(f'[model] ✓ Loaded textless model {stage_name}')
    
    return model, processor
```

## Updated Phase 6a Load Cell

Replace the Phase 6a load cell with:

```python
print('Loading Phase 4 model for Phase 6a training...')

# Use textless-specific loader
try:
    model_6a, processor = load_textless_model_from_drive('phase4_textless_pretrain')
    print('✓ Loaded Phase 4 textless model')
    print(f'  Model has CIF: {hasattr(model_6a, "cif_connector")}')
    print(f'  Model has Speaker: {hasattr(model_6a, "speaker_adapter")}')
    print(f'  Model has text_decoder: {hasattr(model_6a, "text_decoder") and model_6a.text_decoder is not None}')
except Exception as e:
    print(f'ERROR: Could not load Phase 4 model: {e}')
    import traceback
    traceback.print_exc()
    print('\nYou must run Phase 4 first!')
    raise

# Consolidate to single GPU
model_6a = _consolidate_to_single_gpu(model_6a)
model_6a.eval()

# Restore Phase 6a checkpoint if exists
p6a_ck = load_latest_checkpoint('phase6a_connector')
if p6a_ck and p6a_ck.get('step', 0) > 0:
    try:
        model_6a.cif_connector.load_state_dict(p6a_ck['cif_state'])
        model_6a.speaker_adapter.load_state_dict(p6a_ck['spk_state'])
        print(f'  ✓ CIF + Speaker adapter weights restored from step {p6a_ck["step"]}')
    except Exception as e:
        print(f'  ⚠ Could not restore checkpoint: {e}')

device = torch.device('cuda:0')
model_6a = model_6a.to(device)
print_model_breakdown(model_6a, 'Phase 6a Model Ready')
gpu_mem()
```

## Why This Works

1. **Loads base model first** - Has valid vocab_size, creates all components
2. **Applies surgery** - Removes text decoder, adds CIF/Speaker
3. **Loads saved weights** - Only for components that exist (speech_encoder, t2u_model, cif_connector, speaker_adapter)
4. **Skips text decoder keys** - Expected to be missing, not an error

## Alternative: Fix Phase 4 Save to Keep Minimal Vocab

If you want to avoid the custom loader, modify Phase 4 save to keep a minimal vocab:

```python
# In remove_text_decoder_and_install_cif(), BEFORE deleting text_decoder:

# Keep minimal vocab for HF compatibility (1 token)
if hasattr(mdl, 'shared') and mdl.shared is not None:
    old_vocab_size = mdl.config.vocab_size
    mdl.config.vocab_size = 1  # Minimal valid size
    mdl.shared = nn.Embedding(1, mdl.config.hidden_size, padding_idx=0)
    print(f'  ✓ Reduced vocab: {old_vocab_size} → 1 (HF compatibility)')

# Then delete text_decoder as before
```

But the custom loader is cleaner and more explicit about the textless architecture.
