# ── Phase 5 Cell 2: FIXED calibration — component-specific forward passes ──────
#
# ROOT CAUSE of device mismatch:
#   The text_decoder forward pass creates encoder_attention_mask on CPU by default.
#   We must explicitly move ALL tensors (including masks) to the target device.
#
# FIX: Explicitly move encoder_attention_mask to the same device as enc_hidden.

import torch
import torch.nn as nn
import numpy as np


def collect_ffn_calibration_stats(model, component_name, calibration_wavs,
                                   processor, n_samples=64, device=None):
    """
    Collect per-channel input L2-norm stats for every FFN layer in component_name.

    Uses direct component forward passes (NOT generate()) so every single
    FFN layer fires on every sample. No generate() truncation issues.

    Per-channel stats:
        sq_sum = Σ x_j²    → rms = sqrt(sq_sum/count)  [Wanda-sp]
        sum_x  = Σ x_j     → mean                      [bias compensation]
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
                # ── Step 1: always encode speech (needed by all components) ──
                enc_in = processor(audio=wav, sampling_rate=16000,
                                   return_tensors="pt")
                input_feats = enc_in["input_features"].to(device)
                attn_mask = enc_in["attention_mask"].to(device)
                
                if component_name == "speech_encoder":
                    # Direct speech encoder forward — fires all Conformer FFNs
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
                    enc_hidden = enc_out.last_hidden_state  # [1, T_enc, H]

                    # CRITICAL FIX: Move enc_hidden to text_decoder device
                    dec_device = next(model.text_decoder.parameters()).device
                    enc_hidden = enc_hidden.to(dec_device)

                    enc_len = enc_hidden.shape[1]
                    
                    # CRITICAL FIX: Create encoder_attention_mask on SAME device as enc_hidden
                    encoder_attention_mask = torch.ones(
                        (enc_hidden.shape[0], enc_len),
                        dtype=torch.long,
                        device=dec_device  # ← FIX: was missing, defaulted to CPU
                    )
                    
                    # Create fake decoder input_ids on correct device
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
                    # T2U calibration:
                    #   1. Encode speech
                    #   2. Run text_decoder with fake ids → get text hidden states
                    #   3. Run t2u_model(inputs_embeds=text_hidden)
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
                    
                    # CRITICAL FIX: encoder_attention_mask on correct device
                    encoder_attention_mask = torch.ones(
                        (enc_hidden.shape[0], enc_len),
                        dtype=torch.long,
                        device=dec_device  # ← FIX
                    )
                    
                    dec_out = model.text_decoder(
                        input_ids=fake_ids,
                        encoder_hidden_states=enc_hidden,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=True,
                    )
                    text_hidden = dec_out.last_hidden_state  # [1, 32, H]

                    # T2U forward: pass text hidden states as inputs_embeds
                    t2u_device = next(model.t2u_model.parameters()).device
                    model.t2u_model(
                        inputs_embeds=text_hidden.to(t2u_device),
                        return_dict=True,
                    )

                else:
                    # Generic fallback for any other component:
                    # try calling it directly if it has a simple forward
                    comp = getattr(model, component_name)
                    enc_out = model.speech_encoder(
                        input_features=input_feats,
                        attention_mask=attn_mask,
                        return_dict=True,
                    )
                    try:
                        comp(enc_out.last_hidden_state, return_dict=True)
                    except Exception:
                        pass

            except Exception as e:
                print(f"  [calib] sample {i} failed: {e}")

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


print("Calibration helpers ready (FIXED device placement).")
print("  CRITICAL FIX: encoder_attention_mask now created on correct device")
print("  speech_encoder: model.speech_encoder(input_features)")
print("  text_decoder:   model.text_decoder(fake_ids, encoder_hidden_states)")
print("  t2u_model:      model.t2u_model(inputs_embeds=text_hidden)")
