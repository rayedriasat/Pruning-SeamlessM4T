#!/usr/bin/env python3
"""
Apply Phase 4 and 6a fixes to seamless-final.ipynb
Fixes:
1. Phase 4 save/load to use proper save_model_to_drive()
2. CIF Connector weight normalization bug
3. Phase 6a model loading
4. Phase 6a training loss weights
"""

import json
import re
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

def replace_cif_connector_class(nb):
    """Replace CIFConnector class with fixed version"""
    
    fixed_cif = '''class CIFConnector(nn.Module):
    """
    Continuous Integrate-and-Fire connector (Dong & Xu, ICASSP 2020).
    
    CRITICAL FIX: Weight normalization now correctly scales to predicted n_tokens.
    Previous version normalized to sum=1.0, causing severe under-firing.
    """
    def __init__(self, d_model=1024, n_refiner_layers=2, n_langs=45, threshold=1.0):
        super().__init__()
        self.d_model   = d_model
        self.threshold = threshold

        # Quantity predictor: predicts target number of output tokens
        self.qty_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()   # always positive
        )

        # Weight predictor: per-frame importance (unnormalized)
        self.weight_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()   # always positive
        )

        # Language conditioning
        self.lang_embed = nn.Embedding(n_langs, d_model // 8)
        self.lang_proj  = nn.Linear(d_model // 8, d_model)

        # Refiner transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True, norm_first=True)
        self.refiner  = nn.TransformerEncoder(enc_layer, num_layers=n_refiner_layers)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, encoder_out, tgt_lang_id=None):
        """
        Args:
            encoder_out : [B, T_frames, D]
            tgt_lang_id : [B] integer lang IDs
        Returns:
            out        : [B, T_units, D]   — fired token representations
            actual_qty : [B]               — how many tokens actually fired
            qty_pred   : [B]               — quantity predictor head output
            alpha      : [B, T_frames]     — normalized per-frame weights
        """
        B, T, D = encoder_out.shape

        # Language conditioning
        if tgt_lang_id is not None:
            le = self.lang_proj(self.lang_embed(tgt_lang_id.to(encoder_out.device)))
            encoder_out = encoder_out + le.unsqueeze(1)

        # Predict target quantity from mean-pooled encoder output
        mean_pool = encoder_out.mean(dim=1)                        # [B, D]
        qty_pred  = self.qty_predictor(mean_pool).squeeze(-1)      # [B]

        # Per-frame unnormalized weights
        raw_w = self.weight_predictor(encoder_out).squeeze(-1)     # [B, T]

        # CRITICAL FIX: Normalize weights to sum to qty_pred (not 1.0!)
        w_sum  = raw_w.sum(dim=1, keepdim=True).clamp(min=1e-6)   # [B, 1]
        alpha  = raw_w / w_sum * qty_pred.unsqueeze(1)             # [B, T]

        # CIF: accumulate until threshold, fire
        outputs = []
        for b in range(B):
            w   = alpha[b]; h = encoder_out[b]
            acc = torch.zeros(D, device=h.device, dtype=h.dtype)
            acc_w, fired = 0.0, []
            for t in range(T):
                acc_w += w[t].item()
                acc   += w[t] * h[t]
                if acc_w >= self.threshold:
                    fired.append(acc / acc_w)
                    acc   = torch.zeros_like(acc)
                    acc_w = 0.0
            if acc_w > 0.05:
                fired.append(acc / max(acc_w, 1e-6))
            if not fired:
                fired.append(h.mean(0))
            outputs.append(torch.stack(fired))

        max_len = max(o.shape[0] for o in outputs)
        padded  = torch.zeros(B, max_len, D, device=encoder_out.device,
                              dtype=encoder_out.dtype)
        for b, o in enumerate(outputs):
            padded[b, :o.shape[0]] = o

        refined    = self.refiner(padded)
        out        = self.out_proj(refined)
        actual_qty = torch.tensor([float(o.shape[0]) for o in outputs],
                                  dtype=torch.float, device=encoder_out.device)

        return out, actual_qty, qty_pred, alpha
'''
    
    # Find CIF connector cell
    idx = find_cell_by_content(nb, 'class CIFConnector(nn.Module):')
    if idx is None:
        print("⚠ Could not find CIFConnector class cell")
        return False
    
    source = ''.join(nb['cells'][idx]['source'])
    
    # Replace the class definition
    # Find start and end of class
    class_start = source.find('class CIFConnector(nn.Module):')
    if class_start == -1:
        print("⚠ Could not find CIFConnector class start")
        return False
    
    # Find the next class or major section
    rest = source[class_start:]
    # Find end by looking for next class or major comment
    class_end_markers = ['\nclass ', '\n# ── ', '\n_cif_test']
    class_end = len(rest)
    for marker in class_end_markers:
        pos = rest.find(marker, 100)  # Skip first 100 chars to avoid matching self
        if pos != -1 and pos < class_end:
            class_end = pos
    
    new_source = source[:class_start] + fixed_cif + source[class_start + class_end:]
    nb['cells'][idx]['source'] = new_source.split('\n')
    
    print("✓ Fixed CIFConnector class")
    return True

def fix_phase4_save(nb):
    """Fix Phase 4 save to use save_model_to_drive()"""
    
    fixed_phase4_save = '''p4_done = load_latest_checkpoint('phase4_done')
if p4_done:
    print('Phase 4 architectural surgery already done.')
    try:
        model_p4, processor = load_model_from_drive('phase4_textless_pretrain')
        print('✓ Loaded Phase 4 from Drive using proper load_model_from_drive()')
    except Exception as e:
        print(f'Load failed: {e}. Will rebuild.')
        model_p4 = None
else:
    print('Running Phase 4: architectural surgery...')
    model_p4 = _consolidate_to_single_gpu(model_p3)
    model_p4 = remove_text_decoder_and_install_cif(model_p4)
    print_model_breakdown(model_p4, 'Phase 4: Textless Architecture')
    
    # CORRECT: Use save_model_to_drive() with processor
    save_model_to_drive(model_p4, processor, 'phase4_textless_pretrain',
                        manifest_extra={
                            'hidden': model_p4.config.hidden_size,
                            'n_langs': getattr(model_p4.config, 'vocoder_num_langs', 36),
                            'cif_params': count_params(model_p4.cif_connector),
                            'spk_params': count_params(model_p4.speaker_adapter),
                        })
    
    save_checkpoint({'done': True, 'hidden': model_p4.config.hidden_size},
                    'phase4_done', 0)
    print('✓ Phase 4 saved using proper save_model_to_drive()')
    print_model_breakdown(model_p4, 'Phase 4 DONE: Textless ~750M')

gpu_mem()
'''
    
    # Find Phase 4 surgical cell
    idx = find_cell_by_content(nb, "p4_dir = f'{MODEL_DIR}/phase4_textless_pretrain'")
    if idx is None:
        print("⚠ Could not find Phase 4 surgical cell")
        return False
    
    nb['cells'][idx]['source'] = fixed_phase4_save.split('\n')
    print("✓ Fixed Phase 4 save cell")
    return True

def fix_phase6a_load(nb):
    """Fix Phase 6a model loading"""
    
    fixed_phase6a_load = '''print('Loading Phase 4 model for Phase 6a training...')

# CORRECT: Use load_model_from_drive()
try:
    model_6a, processor = load_model_from_drive('phase4_textless_pretrain')
    print('✓ Loaded Phase 4 from Drive using proper load_model_from_drive()')
    print(f'  Model has CIF: {hasattr(model_6a, "cif_connector")}')
    print(f'  Model has Speaker: {hasattr(model_6a, "speaker_adapter")}')
except Exception as e:
    print(f'ERROR: Could not load Phase 4 model: {e}')
    print('You must run Phase 4 first!')
    raise

# Consolidate to single GPU
model_6a = _consolidate_to_single_gpu(model_6a)
model_6a.eval()

# Restore Phase 6a checkpoint if exists
p6a_ck = load_latest_checkpoint('phase6a_connector')
if p6a_ck and p6a_ck.get('step', 0) > 0:
    model_6a.cif_connector.load_state_dict(p6a_ck['cif_state'])
    model_6a.speaker_adapter.load_state_dict(p6a_ck['spk_state'])
    print(f'  ✓ CIF + Speaker adapter weights restored from step {p6a_ck["step"]}')

device = torch.device('cuda:0')
model_6a = model_6a.to(device)
print_model_breakdown(model_6a, 'Phase 6a Model Ready')
gpu_mem()
'''
    
    # Find Phase 6a load cell (looks for the broken rebuild logic)
    idx = find_cell_by_content(nb, "p4_saved = torch.load(f'{p4_dir}/textless_model.pt'")
    if idx is None:
        print("⚠ Could not find Phase 6a load cell")
        return False
    
    nb['cells'][idx]['source'] = fixed_phase6a_load.split('\n')
    print("✓ Fixed Phase 6a load cell")
    return True

def fix_phase6a_loss_weights(nb):
    """Fix Phase 6a training loss weights"""
    
    # Find Phase 6a training cell
    idx = find_cell_by_content(nb, 'loss = (0.65 * cos_loss')
    if idx is None:
        print("⚠ Could not find Phase 6a training loss cell")
        return False
    
    source = ''.join(nb['cells'][idx]['source'])
    
    # Replace loss computation
    old_loss = '''loss = (0.65 * cos_loss       +
                    0.20 * mse_loss       +
                    qty_warmup_w * qty_loss +
                    0.05 * spk_reg        +
                    0.10 * alpha_reg)'''
    
    new_loss = '''loss = (0.70 * cos_loss       +   # PRIMARY: direction alignment
                    0.15 * mse_loss       +   # magnitude alignment
                    qty_warmup_w * qty_loss + # quantity (warmed up)
                    0.05 * spk_reg        +   # speaker regularization
                    0.10 * alpha_reg)         # collapse prevention'''
    
    source = source.replace(old_loss, new_loss)
    nb['cells'][idx]['source'] = source.split('\n')
    
    print("✓ Fixed Phase 6a loss weights")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python apply_phase4_6a_fixes.py <notebook_path>")
        print("Example: python apply_phase4_6a_fixes.py seamless-final.ipynb")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    backup_path = notebook_path.replace('.ipynb', '_backup_before_fix.ipynb')
    
    print(f"Loading notebook: {notebook_path}")
    nb = load_notebook(notebook_path)
    
    print(f"Creating backup: {backup_path}")
    save_notebook(nb, backup_path)
    
    print("\nApplying fixes...")
    print("=" * 60)
    
    success = True
    success &= replace_cif_connector_class(nb)
    success &= fix_phase4_save(nb)
    success &= fix_phase6a_load(nb)
    success &= fix_phase6a_loss_weights(nb)
    
    if success:
        print("=" * 60)
        print(f"\n✓ All fixes applied successfully!")
        save_notebook(nb, notebook_path)
        print(f"\nBackup saved to: {backup_path}")
        print(f"Fixed notebook saved to: {notebook_path}")
        print("\nNext steps:")
        print("1. Delete corrupted Phase 4 checkpoint:")
        print("   !rm -rf /kaggle/working/models/phase4_textless_pretrain")
        print("2. Re-run Phase 4 cell")
        print("3. Re-run Phase 6a cells")
    else:
        print("\n⚠ Some fixes failed. Check output above.")
        print(f"Backup is at: {backup_path}")
        sys.exit(1)

if __name__ == '__main__':
    main()
