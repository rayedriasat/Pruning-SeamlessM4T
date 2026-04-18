#!/usr/bin/env python3
"""
Fix Phase 5 Cell 2 in the notebook by replacing the calibration function
with the version that explicitly moves ALL processor outputs to device.
"""

import json
import sys

# Read the notebook
with open('cse465v5-s2st-corrected.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new calibration function code
new_code = '''# ── Phase 5 Cell 2: FIXED calibration with explicit device handling ──────────

import torch
import torch.nn as nn
import numpy as np


def collect_ffn_calibration_stats(model, component_name, calibration_wavs,
                                   processor, n_samples=64, device=None):
    """
    Collect per-channel input L2-norm stats for every FFN layer in component_name.
    
    CRITICAL FIX: Explicitly move ALL tensors to device, including processor outputs.
    """
    if device is None:
        device = next(model.parameters()).device

    ffn_pairs = find_all_ffn_layers(model, component_name)
    if not ffn_pairs:
        print(f"  [calib] No FFN pairs in {component_name}, skipping.")
        return {}

    # Build stats keyed by id(parent)
    stats = {}
    for (parent, fc1_attr, fc2_attr, name) in ffn_pairs:
        fc1 = getattr(parent, fc1_attr)
        key = id(parent)
        stats[key] = {
            "sum_x":  torch.zeros(fc1.in_features, dtype=torch.float64),
            "sq_sum": torch.zeros(fc1.in_features, dtype=torch.float64),
            "count":  0,
            "module": parent,
            "fc1":    fc1_attr,
            "fc2":    fc2_attr,
            "name":   name,
        }

    hooks = []
    def make_hook(key):
        def hook(module, inp, out):
            x = inp[0].detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() == 1:
                x = x.unsqueeze(0)
            s = stats[key]
            s["count"]  += x.shape[0]
            s["sum_x"]  += x.sum(dim=0).double()
            s["sq_sum"] += x.pow(2).sum(dim=0).double()
        return hook

    for (parent, fc1_attr, _, name) in ffn_pairs:
        fc1 = getattr(parent, fc1_attr)
        hooks.append(fc1.register_forward_hook(make_hook(id(parent))))

    model.eval()
    n_actual = min(n_samples, len(calibration_wavs))

    print(f"  [calib] Collecting activations from {n_actual} samples "
          f"for {component_name} (direct forward pass — every layer fires)...")

    with torch.no_grad():
        for i, wav in enumerate(calibration_wavs[:n_actual]):
            if i % 20 == 0:
                print(f"  [calib] {i}/{n_actual}")
            try:
                # ── CRITICAL FIX: Process on CPU, then move EVERYTHING to device ──
                enc_in = processor(audio=wav, sampling_rate=16000,
                                   return_tensors="pt")
                
                # Move ALL tensors in the dict to device (not just known keys)
                enc_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in enc_in.items()}
                
                input_feats = enc_in["input_features"]
                attn_mask = enc_in["attention_mask"]
                
                if component_name == "speech_encoder":
                    model.speech_encoder(
                        input_features=input_feats,
                        attention_mask=attn_mask,
                        return_dict=True,
                    )

                elif component_name == "text_decoder":
                    enc_out = model.speech_encoder(
                        input_features=input_feats,
                        attention_mask=attn_mask,
                        return_dict=True,
                    )
                    enc_hidden = enc_out.last_hidden_state

                    dec_device = next(model.text_decoder.parameters()).device
                    enc_hidden = enc_hidden.to(dec_device)

                    enc_len = enc_hidden.shape[1]
                    encoder_attention_mask = torch.ones(
                        (enc_hidden.shape[0], enc_len),
                        dtype=torch.long,
                        device=dec_device
                    )
                    
                    fake_ids = torch.randint(
                        low=0,
                        high=model.config.vocab_size,
                        size=(1, 32),
                        device=dec_device
                    )
                
                    model.text_decoder(
                        input_ids=fake_ids,
                        encoder_hidden_states=enc_hidden,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=True,
                    )

                elif component_name == "t2u_model":
                    enc_out = model.speech_encoder(
                        input_features=input_feats,
                        attention_mask=attn_mask,
                        return_dict=True,
                    )
                    enc_hidden = enc_out.last_hidden_state
                    
                    dec_device = next(model.text_decoder.parameters()).device
                    enc_hidden = enc_hidden.to(dec_device)

                    try:
                        ben_tok = model.generation_config.text_decoder_lang_to_code_id.get("ben", 4)
                    except Exception:
                        ben_tok = 4

                    fake_ids = torch.full((1, 32), ben_tok,
                                         dtype=torch.long, device=dec_device)

                    enc_len = enc_hidden.shape[1]
                    encoder_attention_mask = torch.ones(
                        (enc_hidden.shape[0], enc_len),
                        dtype=torch.long,
                        device=dec_device
                    )
                    
                    dec_out = model.text_decoder(
                        input_ids=fake_ids,
                        encoder_hidden_states=enc_hidden,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=True,
                    )
                    text_hidden = dec_out.last_hidden_state

                    t2u_device = next(model.t2u_model.parameters()).device
                    model.t2u_model(
                        inputs_embeds=text_hidden.to(t2u_device),
                        return_dict=True,
                    )

            except Exception as e:
                print(f"  [calib] sample {i} failed: {e}")
                import traceback
                if i == 0:  # Print full traceback for first failure only
                    traceback.print_exc()

    for h in hooks:
        h.remove()

    # Diagnostics
    fired      = sum(1 for s in stats.values() if s["count"] > 0)
    not_fired  = len(stats) - fired
    total_toks = sum(s["count"] for s in stats.values())
    print(f"  [calib] Layers fired: {fired}/{len(stats)}  "
          f"total token-vectors: {total_toks}")
    if not_fired > 0:
        names = [s["name"] for s in stats.values() if s["count"] == 0]
        print(f"  [calib] WARNING: {not_fired} layers still got 0 — "
              f"weight-norm fallback: {names[:3]}{'...' if len(names) > 3 else ''}")
    else:
        print(f"  [calib] All {fired} layers fired correctly.")

    # Finalize statistics
    for key, s in stats.items():
        n        = max(s["count"], 1)
        mean_x   = (s["sum_x"]  / n).float()
        sq_norm  = (s["sq_sum"] / n).float()
        var      = (sq_norm - mean_x.pow(2)).clamp(min=0)
        s["mean"]    = mean_x
        s["var"]     = var
        s["sq_norm"] = sq_norm

    print(f"  [calib] Done. {len(stats)} FFN layers instrumented.")
    return stats


print("Calibration helpers ready (FIXED: explicit device handling for ALL processor outputs).")'''

# Find Phase 5 Cell 2
found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = cell['source']
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source
        
        if 'Phase 5 Cell 2' in source_text and 'calibration' in source_text:
            print(f"Found Phase 5 Cell 2 at cell index {i}")
            # Replace the source
            cell['source'] = new_code
            found = True
            break

if not found:
    print("ERROR: Could not find Phase 5 Cell 2 in the notebook")
    sys.exit(1)

# Write the updated notebook
with open('cse465v5-s2st-corrected.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Phase 5 Cell 2 updated successfully")
print("The calibration function now moves ALL processor outputs to device")
