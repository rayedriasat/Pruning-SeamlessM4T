# cell 80
# ── Phase 7 Cell 2: DoRA target discovery (robust) ───────────────────────────
import torch.nn as nn

def discover_lora_targets(mdl, scope_keywords=('text_decoder', 't2u_model', 'speech_encoder')):
    """
    Walk ALL named Linear modules and collect the leaf names present
    in the scopes we care about.  Prints a full map so you can see
    exactly what exists after all pruning phases.
    """
    found_by_scope = {}
    for name, mod in mdl.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        scope = next((kw for kw in scope_keywords if kw in name), None)
        if scope is None:
            continue
        leaf = name.split('.')[-1]
        found_by_scope.setdefault(scope, set()).add(leaf)

    print("Linear layer leaf names by scope:")
    all_leaves = set()
    for scope, leaves in sorted(found_by_scope.items()):
        print(f"  {scope}: {sorted(leaves)}")
        all_leaves |= leaves

    # Restrict to attention + ffn projections only (skip embeddings, lm_head etc.)
    attn_ffn_candidates = {
        'q_proj', 'k_proj', 'v_proj', 'out_proj',   # standard attn
        'k_proj', 'v_proj',                           # cross-attn (same names)
        'fc1', 'fc2',                                 # FFN
        'q_proj', 'out_proj',
    }
    targets = sorted(all_leaves & attn_ffn_candidates)
    print(f"\nRecommended target_modules: {targets}")

    # Verify count
    count = sum(
        1 for name, mod in mdl.named_modules()
        if isinstance(mod, nn.Linear)
        and name.split('.')[-1] in targets
        and any(kw in name for kw in scope_keywords)
    )
    print(f"Total Linear layers that will receive LoRA/DoRA: {count}")
    return targets

targets = discover_lora_targets(model_p6)
