#!/usr/bin/env python3
"""
Add load_textless_model_from_drive() function to notebook
This fixes the Phase 4 load error caused by vocab_size=0
"""

import json
import sys

def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✓ Saved {path}")

def find_cell_by_content(nb, search_text):
    """Find cell index containing search_text"""
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if search_text in source:
                return i
    return None

def add_textless_loader(nb):
    """Add load_textless_model_from_drive() after load_model_from_drive()"""
    
    textless_loader_code = '''def load_textless_model_from_drive(stage_name):
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

print('Textless model loader ready.')
'''
    
    # Find the cell with load_model_from_drive
    idx = find_cell_by_content(nb, "def load_model_from_drive(stage_name):")
    if idx is None:
        print("⚠ Could not find load_model_from_drive cell")
        return False
    
    # Add the new function after load_model_from_drive
    source = ''.join(nb['cells'][idx]['source'])
    
    # Check if already added
    if 'load_textless_model_from_drive' in source:
        print("✓ load_textless_model_from_drive already exists")
        return True
    
    # Append to the cell
    new_source = source + '\n\n' + textless_loader_code
    nb['cells'][idx]['source'] = new_source.split('\n')
    
    print("✓ Added load_textless_model_from_drive()")
    return True

def update_phase6a_load(nb):
    """Update Phase 6a load to use textless loader"""
    
    updated_load = '''print('Loading Phase 4 model for Phase 6a training...')

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
    print('\\nYou must run Phase 4 first!')
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
'''
    
    # Find Phase 6a load cell
    idx = find_cell_by_content(nb, "model_6a, processor = load_model_from_drive('phase4_textless_pretrain')")
    if idx is None:
        print("⚠ Could not find Phase 6a load cell")
        return False
    
    nb['cells'][idx]['source'] = updated_load.split('\n')
    print("✓ Updated Phase 6a load to use textless loader")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python add_textless_loader.py <notebook_path>")
        print("Example: python add_textless_loader.py seamless-final.ipynb")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    backup_path = notebook_path.replace('.ipynb', '_backup_textless_loader.ipynb')
    
    print(f"Loading notebook: {notebook_path}")
    nb = load_notebook(notebook_path)
    
    print(f"Creating backup: {backup_path}")
    save_notebook(nb, backup_path)
    
    print("\nApplying fixes...")
    print("=" * 60)
    
    success = True
    success &= add_textless_loader(nb)
    success &= update_phase6a_load(nb)
    
    if success:
        print("=" * 60)
        print(f"\n✓ Textless loader added successfully!")
        save_notebook(nb, notebook_path)
        print(f"\nBackup saved to: {backup_path}")
        print(f"Fixed notebook saved to: {notebook_path}")
        print("\nNext steps:")
        print("1. Re-run the Model I/O helpers cell (to load the new function)")
        print("2. Re-run Phase 6a load cell")
    else:
        print("\n⚠ Some fixes failed. Check output above.")
        print(f"Backup is at: {backup_path}")
        sys.exit(1)

if __name__ == '__main__':
    main()
