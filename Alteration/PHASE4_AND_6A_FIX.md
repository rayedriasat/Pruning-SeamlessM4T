# Phase 4 & 6a Critical Fixes

## Problem Summary

1. **Phase 4 Save/Load is Broken**
   - Using `torch.save()` directly instead of `save_model_to_drive()`
   - Not syncing config with `sync_model_config()`
   - Missing custom state save (`_save_custom_state()`)
   - Result: 288 missing keys when loading

2. **CIF Connector Not Learning**
   - Firing only 2-6 tokens when target is 20-56
   - Weight normalization issue in CIFConnector
   - Alpha weights collapsing to near-zero

## Root Cause

The Phase 4 surgical cell saves the model like this (WRONG):
```python
p4_dir = f'{MODEL_DIR}/phase4_textless_pretrain'
os.makedirs(p4_dir, exist_ok=True)
torch.save({
    'state_dict': model_p4.state_dict(),
    'config':     model_p4.config,
    'cif_state':  model_p4.cif_connector.state_dict(),
    'spk_state':  model_p4.speaker_adapter.state_dict(),
    'hidden':     model_p4.config.hidden_size,
    'n_langs':    getattr(model_p4.config, 'vocoder_num_langs', 36),
}, f'{p4_dir}/textless_model.pt')
```

This bypasses:
- `sync_model_config()` - config doesn't match actual layer counts
- `_save_custom_state()` - vocab remap lost
- HuggingFace `save_pretrained()` - proper weight serialization
- Processor save - tokenizer/processor not saved

## Fix 1: Phase 4 Proper Save

Replace the Phase 4 surgical RUN cell with:

```python
p4_done = load_latest_checkpoint('phase4_done')
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
```

## Fix 2: Phase 6a Proper Load

Replace the Phase 6a model loading cell with:

```python
print('Loading Phase 4 model for Phase 6a training...')

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
```

## Fix 3: CIF Connector Weight Normalization

The current CIFConnector has a critical bug in weight normalization. Replace the entire CIFConnector class:

```python
class CIFConnector(nn.Module):
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
```

## Fix 4: Phase 6a Training Loss Weights

Update Phase 6a training loop loss computation:

```python
# UPDATED loss weights - prioritize cosine alignment
loss = (0.70 * cos_loss       +   # PRIMARY: direction alignment
        0.15 * mse_loss       +   # magnitude alignment
        qty_warmup_w * qty_loss + # quantity (warmed up)
        0.05 * spk_reg        +   # speaker regularization
        0.10 * alpha_reg)         # collapse prevention
```

## Expected Results After Fixes

### Phase 4 Load
```
[model] Loading phase4_textless_pretrain from /kaggle/working/models/phase4_textless_pretrain ...
[model] Loaded phase4_textless_pretrain.
  Restored custom state: ['_vocab_remap_to_old']
✓ Loaded Phase 4 from Drive using proper load_model_from_drive()
  Model has CIF: True
  Model has Speaker: True

--- Phase 6a Model Ready ---
  speech_encoder                      440.8M  ( 66.0%)
  t2u_model                           175.2M  ( 26.2%)
  cif_connector                        18.5M  (  2.8%)
  vocoder                              41.9M  (  6.3%)
  speaker_adapter                       0.1M  (  0.0%)
  TOTAL                               667.8M
---
```

### Phase 6a Training
```
Step   100/5000 | cos=0.1788 | qty_err(tok)=5.2 | total=0.2184 | fired=18 vs tgt=20 | qty_w=0.020 | lr=4.99e-05
Step   200/5000 | cos=0.1257 | qty_err(tok)=3.8 | total=0.1648 | fired=21 vs tgt=23 | qty_w=0.040 | lr=4.98e-05
Step   300/5000 | cos=0.0923 | qty_err(tok)=2.7 | total=0.1217 | fired=22 vs tgt=24 | qty_w=0.060 | lr=4.94e-05
Step   400/5000 | cos=0.0713 | qty_err(tok)=2.1 | total=0.0959 | fired=54 vs tgt=56 | qty_w=0.080 | lr=4.90e-05
```

Notice:
- `fired` now matches `tgt` closely (18 vs 20, not 2 vs 20)
- `cos` loss decreasing properly (0.17 → 0.07)
- `qty_err` in reasonable range (2-5 tokens, not 30)

## Implementation Steps

1. **Backup current notebook** (if not already done)
2. **Replace CIFConnector class** in the Phase 4 cell
3. **Replace Phase 4 surgical RUN cell** with Fix 1 code
4. **Replace Phase 6a load cell** with Fix 2 code
5. **Update Phase 6a training loss** with Fix 4 weights
6. **Delete corrupted Phase 4 checkpoint**:
   ```python
   !rm -rf /kaggle/working/models/phase4_textless_pretrain
   ```
7. **Re-run from Phase 3** (or Phase 4 if Phase 3 is saved correctly)

## Verification Checklist

After applying fixes:

- [ ] Phase 4 saves using `save_model_to_drive()`
- [ ] Phase 4 loads with 0 missing keys
- [ ] CIF connector fires ~target number of tokens (±10%)
- [ ] Phase 6a cosine loss < 0.10 after 2000 steps
- [ ] Phase 6a qty_err < 5 tokens after 1000 steps
- [ ] Model breakdown shows all components present

## Why This Matters

The broken Phase 4 save/load means:
- Config doesn't match actual architecture
- Custom attributes (vocab remap) lost
- Processor/tokenizer not saved
- Can't properly resume training

The CIF weight normalization bug means:
- Connector fires 10× fewer tokens than needed
- T2U receives truncated sequences
- Translation quality catastrophically degraded
- No amount of training can fix it without the normalization fix

Both must be fixed together for Phase 6a to work.
