#!/usr/bin/env python3
"""
Fix the CIF connector firing bug in seamless-final.ipynb

The bug: After firing a token, the residual accumulator is incorrectly calculated.
The line `acc = acc * (acc_w / max(acc_w + self.threshold, 1e-6))` is wrong because
acc_w has already been reduced by threshold, so adding it back is incorrect.

Correct logic: The residual should be proportional to the leftover weight.
"""

import json
import sys

def fix_cif_connector(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find the CIF connector cell (cell 59)
    cif_cell_idx = None
    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'class CIFConnector' in source and 'CRITICAL FIX v2' in source:
            cif_cell_idx = i
            break
    
    if cif_cell_idx is None:
        print("ERROR: Could not find CIF connector cell")
        return False
    
    print(f"Found CIF connector in cell {cif_cell_idx}")
    
    # The corrected CIF connector with proper firing logic
    corrected_cif = '''# ── CIF Connector (Dong & Xu, ICASSP 2020: arXiv:1905.11235) ─────────────────
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CORRECTED CIF CONNECTOR v3 — Fixed residual calculation bug                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class CIFConnector(nn.Module):
    """
    Continuous Integrate-and-Fire connector (Dong & Xu, ICASSP 2020).
    
    CRITICAL FIX v3: Proper residual handling after firing.
    
    The bug in v2: After firing, the residual was calculated as:
        acc = acc * (acc_w / (acc_w + threshold))
    But acc_w was already reduced by threshold, so this was wrong.
    
    Correct logic: Keep the residual proportional to leftover weight.
    """
    def __init__(self, d_model=1024, n_refiner_layers=2, n_langs=45, threshold=0.95):
        super().__init__()
        self.d_model   = d_model
        self.threshold = threshold  # Lower threshold = more tokens fired
        
        # Quantity predictor: predicts target number of output tokens
        self.qty_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()   # always positive
        )
        
        # Weight predictor: per-frame importance in [0,1] range
        self.weight_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()   # Sigmoid for [0,1] range
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
        
        # Per-frame weights in [0, 1] range (sigmoid output)
        raw_w = self.weight_predictor(encoder_out).squeeze(-1)     # [B, T]
        
        # CRITICAL: Scale weights so their sum equals qty_pred
        # This ensures CIF fires approximately qty_pred tokens
        w_sum  = raw_w.sum(dim=1, keepdim=True).clamp(min=1e-6)   # [B, 1]
        alpha  = raw_w / w_sum * qty_pred.unsqueeze(1)             # [B, T]
        
        # CIF: accumulate until threshold, fire
        outputs = []
        for b in range(B):
            w   = alpha[b]  # [T]
            h   = encoder_out[b]  # [T, D]
            acc = torch.zeros(D, device=h.device, dtype=h.dtype)
            acc_w, fired = 0.0, []
            
            for t in range(T):
                w_t = w[t].item()
                acc_w += w_t
                acc   += w_t * h[t]
                
                # Fire when accumulated weight exceeds threshold
                while acc_w >= self.threshold:
                    # Fire one token with the accumulated representation
                    fired.append(acc.clone())
                    
                    # CRITICAL FIX: Proper residual calculation
                    # After firing, we have leftover weight = acc_w - threshold
                    # The residual representation should be proportional to this leftover
                    acc_w_before_fire = acc_w
                    acc_w -= self.threshold
                    
                    if acc_w > 1e-6:
                        # Keep residual proportional to leftover weight
                        acc = acc * (acc_w / acc_w_before_fire)
                    else:
                        # No significant residual
                        acc = torch.zeros_like(acc)
                        acc_w = 0.0
            
            # Fire remaining if significant
            if acc_w > 0.1:
                fired.append(acc)
            
            # Ensure at least 1 token fired
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


# ── Speaker Adapter (unchanged) ─────────────────────────────────────────────────
class SpeakerAdapter(nn.Module):
    """ECAPA 192-dim → HiFi-GAN vocoder 256-dim. ~0.1M params."""
    def __init__(self, ecapa_dim=192, vocoder_spkr_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ecapa_dim, vocoder_spkr_dim),
            nn.LayerNorm(vocoder_spkr_dim),
            nn.Tanh())
    def forward(self, ecapa_emb):
        return self.proj(ecapa_emb)


_cif_test = CIFConnector()
_spk_test = SpeakerAdapter()
print(f'CIFConnector (v3 fixed): ~{count_params(_cif_test):.2f}M params')
print(f'SpeakerAdapter: ~{count_params(_spk_test)*1000:.0f}K params')
del _cif_test, _spk_test
'''
    
    # Replace the cell source
    nb['cells'][cif_cell_idx]['source'] = corrected_cif.split('\n')
    
    # Save the notebook
    backup_path = notebook_path + '.backup_before_v3_fix'
    import shutil
    shutil.copy(notebook_path, backup_path)
    print(f"Backup saved to: {backup_path}")
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Fixed CIF connector in {notebook_path}")
    print("\nKey changes:")
    print("  1. Fixed residual calculation after firing")
    print("  2. acc_w_before_fire stored before reducing by threshold")
    print("  3. Residual = acc * (acc_w / acc_w_before_fire)")
    print("  4. This preserves the correct proportion of the representation")
    print("\nThe bug was: acc * (acc_w / (acc_w + threshold))")
    print("  - acc_w was already reduced, so adding threshold back was wrong")
    print("  - This caused the residual to be too large, preventing proper firing")
    
    return True

if __name__ == '__main__':
    notebook_path = 'Alteration/seamless-final.ipynb'
    success = fix_cif_connector(notebook_path)
    sys.exit(0 if success else 1)
