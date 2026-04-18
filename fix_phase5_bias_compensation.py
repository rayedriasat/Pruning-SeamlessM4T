#!/usr/bin/env python3
"""
Fix Phase 5 Cell 3 - Disable FLAP bias compensation
Bias compensation is causing NaN/Inf corruption. Disable it and rely on fine-tuning.
"""

import json

with open('cse465v5-s2st-corrected.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Phase 5 Cell 3 (structural_prune_ffn)
cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        if 'def structural_prune_ffn' in source and 'Phase 5 Cell 3' in source:
            cell_idx = i
            break

if cell_idx is None:
    print("ERROR: Could not find Phase 5 Cell 3")
    exit(1)

print(f"Found Phase 5 Cell 3 at cell index {cell_idx}")

# Fixed version WITHOUT bias compensation
new_code = '''# ── Phase 5 Cell 3: Neuron importance scoring (Wanda-sp + FLAP) ──────────────
#
# Wanda-sp (structured Wanda): score(k) = sum_j |W1[k,j]| * ||X_j||_2
#   where ||X_j||_2 = sqrt(E[x_j^2]) — the RMS of channel j inputs
#
# FLAP-row: score(k) = sum_j Var(X_j) * W1[k,j]^2
#
# We use Wanda-sp as primary (proven robust, never zero if layer fires),
# and fall back to pure row-norm if sq_norm is also zero (truly dead layer).

import torch
import torch.nn as nn
import numpy as np


def wanda_neuron_scores(fc1_weight, sq_norm):
    """
    Wanda-sp per-neuron score (ICLR 2024, structured variant).

    score(k) = sum_j |W1[k,j]| * sqrt(E[x_j^2])
             = |W1| @ rms_x          where rms_x = sqrt(sq_norm)

    fc1_weight : [ffn_hidden, model_hidden]
    sq_norm    : [model_hidden]  E[x_j^2] per channel

    Returns [ffn_hidden] scores. Falls back to row-L2-norm if sq_norm ~ 0.
    """
    W1 = fc1_weight.float().cpu()
    rms = sq_norm.float().cpu().clamp(min=0).sqrt()   # [model_hidden]

    if rms.max().item() < 1e-10:
        # Layer never fired or truly dead — use weight row-norm only
        return W1.pow(2).sum(dim=1).sqrt()

    return (W1.abs() * rms.unsqueeze(0)).sum(dim=1)   # [ffn_hidden]


def flap_neuron_scores(fc1_weight, var_x):
    """
    FLAP per-neuron score (AAAI 2024, Eq. 5 applied to rows).
    score(k) = sum_j Var(X_j) * W1[k,j]^2
    Falls back to row-norm if var is zero.
    """
    W1 = fc1_weight.float().cpu()
    v  = var_x.float().cpu().clamp(min=0)

    if v.max().item() < 1e-10:
        return W1.pow(2).sum(dim=1)

    return (W1.pow(2) * v.unsqueeze(0)).sum(dim=1)


def neuron_importance_scores(fc1_weight, var_x, sq_norm=None):
    """
    Primary scoring function: use Wanda-sp if sq_norm available, else FLAP.
    """
    if sq_norm is not None and sq_norm.max().item() > 1e-10:
        return wanda_neuron_scores(fc1_weight, sq_norm)
    if var_x is not None and var_x.max().item() > 1e-10:
        return flap_neuron_scores(fc1_weight, var_x)
    # Pure weight magnitude fallback
    return fc1_weight.float().cpu().pow(2).sum(dim=1)


def standardize_scores(scores):
    """FLAP Eq.6: standardize to zero mean, unit std for cross-layer comparison."""
    mu    = scores.mean()
    sigma = scores.std(unbiased=False)
    if sigma < 1e-8:
        return torch.zeros_like(scores)
    return (scores - mu) / sigma


def structural_prune_ffn(parent, fc1_attr, fc2_attr,
                          channel_mean, keep_idx, device):
    """
    Structurally prune one FFN pair using pre-computed keep_idx.
    
    CRITICAL FIX: Bias compensation DISABLED.
    Reason: Bias compensation is extremely fragile and causes NaN/Inf corruption
    when pruned neurons have extreme activations. Production implementations
    (e.g., LLM-Pruner, Wanda) skip bias compensation and rely on fine-tuning.
    
    This is the safe, proven approach.
    """
    fc1 = getattr(parent, fc1_attr)
    fc2 = getattr(parent, fc2_attr)
    ffn_dim = fc1.out_features

    fc1_device = fc1.weight.device
    fc2_device = fc2.weight.device

    n_keep   = len(keep_idx)
    kidx_fc1 = keep_idx.to(fc1_device)
    kidx_fc2 = keep_idx.to(fc2_device)

    # Create new fc1 (input projection) - keep only selected neurons
    new_fc1 = nn.Linear(fc1.in_features, n_keep,
                         bias=(fc1.bias is not None),
                         device=fc1_device, dtype=fc1.weight.dtype)
    new_fc1.weight.data.copy_(fc1.weight.data[kidx_fc1])
    if fc1.bias is not None:
        new_fc1.bias.data.copy_(fc1.bias.data[kidx_fc1])

    # Create new fc2 (output projection) - keep only selected input dims
    new_fc2 = nn.Linear(n_keep, fc2.out_features,
                         bias=(fc2.bias is not None),
                         device=fc2_device, dtype=fc2.weight.dtype)
    new_fc2.weight.data.copy_(fc2.weight.data[:, kidx_fc2])
    
    # Keep original fc2 bias WITHOUT compensation
    # Fine-tuning (Phase 7) will recover any lost contribution
    if fc2.bias is not None:
        new_fc2.bias.data.copy_(fc2.bias.data)

    setattr(parent, fc1_attr, new_fc1)
    setattr(parent, fc2_attr, new_fc2)

    return n_keep, ffn_dim


print('Neuron scoring + structural pruning helpers ready.')
print('  NOTE: Bias compensation DISABLED (prevents NaN corruption)')


# ── Phase 5 Cell 4: apply_flap_to_component — robust top-k pruning ───────────

def apply_flap_to_component(model, component_name, calib_stats,
                              global_prune_ratio=0.20,
                              min_keep_frac=0.50,
                              device=None):
    """
    Structured FFN width pruning using Wanda-sp / FLAP neuron scores.

    global_prune_ratio : target fraction of neurons to prune (e.g. 0.20 = 20%)
    min_keep_frac      : per-layer floor — never prune a single layer below this
                         fraction of its original size (default 0.50 = keep ≥50%)

    Algorithm (faithful to FLAP paper):
      1. Score every neuron in every layer (Wanda-sp metric)
      2. Standardize scores per-layer (FLAP Eq.6) for cross-layer comparison
      3. Pool all standardized scores, find global threshold at global_prune_ratio
      4. Per layer: keep neurons above threshold, enforce min_keep_frac floor
      5. Structurally remove pruned rows/cols (NO bias compensation - see note above)
    """
    if device is None:
        device = next(model.parameters()).device

    if not calib_stats:
        print(f'  [FLAP] No calib stats for {component_name}, skipping.')
        return {}

    # ── Step 1 & 2: score + standardize ─────────────────────────────────────
    all_std_scores  = {}
    all_raw_scores  = {}

    n_zero_var = 0
    for key, s in calib_stats.items():
        fc1   = getattr(s['module'], s['fc1'])
        W1    = fc1.weight.float().cpu()
        var_x  = s.get('var',     torch.zeros(W1.shape[1]))
        sq_norm = s.get('sq_norm', torch.zeros(W1.shape[1]))

        raw = neuron_importance_scores(W1, var_x, sq_norm)
        if raw.max().item() < 1e-10:
            n_zero_var += 1

        all_raw_scores[key] = raw
        all_std_scores[key] = standardize_scores(raw)

    if n_zero_var:
        print(f'  [FLAP] WARNING: {n_zero_var}/{len(calib_stats)} layers used '
              f'weight-only fallback (calibration did not fire for those layers).')

    # ── Step 3: global threshold ─────────────────────────────────────────────
    all_std_flat  = torch.cat(list(all_std_scores.values()))
    total_neurons = len(all_std_flat)
    n_prune_total = int(total_neurons * global_prune_ratio)

    sorted_scores, _ = torch.sort(all_std_flat)
    threshold = sorted_scores[max(0, n_prune_total - 1)].item()

    print(f'  [FLAP] {component_name}: {total_neurons} total neurons, '
          f'pruning ≤{n_prune_total} ({global_prune_ratio*100:.0f}%), '
          f'threshold={threshold:.4f}')
    print(f'         score range [{all_std_flat.min():.3f}, {all_std_flat.max():.3f}]  '
          f'mean={all_std_flat.mean():.3f}  std={all_std_flat.std():.3f}')

    # ── Step 4 & 5: per-layer prune ──────────────────────────────────────────
    results       = {}
    total_kept    = 0
    total_orig    = 0

    for key, s in calib_stats.items():
        std_scores = all_std_scores[key]
        fc1 = getattr(s['module'], s['fc1'])
        ffn_dim = fc1.out_features

        # How many neurons score above the global threshold?
        n_above = int((std_scores > threshold).sum().item())

        # Enforce per-layer minimum
        min_keep = max(1, int(ffn_dim * min_keep_frac))
        n_keep   = max(min_keep, n_above)
        n_keep   = min(ffn_dim, n_keep)   # can't keep more than we have

        # Top-k selection (stable, threshold-independent)
        _, keep_idx = torch.topk(std_scores, n_keep)
        keep_idx = keep_idx.sort().values

        structural_prune_ffn(
            s['module'], s['fc1'], s['fc2'],
            channel_mean=s['mean'],
            keep_idx=keep_idx,
            device=device
        )

        pct_kept = n_keep / ffn_dim * 100
        total_kept += n_keep
        total_orig += ffn_dim
        results[s['name']] = {
            'kept': n_keep, 'original': ffn_dim, 'pct': pct_kept
        }

    avg_kept = total_kept / max(total_orig, 1) * 100
    print(f'  [FLAP] Done. Kept {total_kept}/{total_orig} neurons '
          f'({avg_kept:.1f}%) across {len(results)} layers.')
    return results


print('apply_flap_to_component ready (bias compensation disabled).')'''

nb['cells'][cell_idx]['source'] = new_code

with open('cse465v5-s2st-corrected.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Phase 5 Cell 3 updated - bias compensation DISABLED")
print("\nKEY CHANGES:")
print("  1. Removed all bias compensation code from structural_prune_ffn()")
print("  2. fc2.bias now keeps original values (no compensation added)")
print("  3. Fine-tuning (Phase 7) will recover any lost contribution")
print("\nThis is the SAFE approach used in production pruning implementations.")
