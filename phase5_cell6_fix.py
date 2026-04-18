# ── Phase 5 Cell 6: RUN PHASE 5 (FIXED) ──────────────────────────────────────
# ROOT CAUSE: model_p4 has device_map='auto' split across devices after loading.
# deepcopy() preserves this split, causing "cuda:0 vs cpu" errors in calibration.
# FIX: Consolidate model_p4 to single GPU BEFORE deepcopy.

import gc as _stdlib_gc
import copy as _copy
 
FLAP_RATIO    = 0.15   # prune 15% of neurons globally per component
MIN_KEEP_FRAC = 0.70   # never shrink any single layer below 70% of original

def force_model_to_single_device(model, device):
    """
    Aggressively consolidate model to a single device.
    Handles device_map='auto' models that are split across devices.
    """
    print(f"  Forcing entire model to {device}...")
    
    # Step 1: Move base model
    model = model.to(device)
    
    # Step 2: Explicitly move all submodules
    for name, module in model.named_modules():
        if module is not model:  # skip root
            try:
                module.to(device)
            except Exception:
                pass
    
    # Step 3: Move all parameters
    for name, param in model.named_parameters():
        if param.device != device:
            param.data = param.data.to(device)
    
    # Step 4: Move all buffers
    for name, buf in model.named_buffers():
        if buf is not None and buf.device != device:
            buf.data = buf.data.to(device)
    
    # Step 5: Clear device_map (forces single-device mode)
    if hasattr(model, 'hf_device_map'):
        model.hf_device_map = {k: device for k in model.hf_device_map}
    
    torch.cuda.empty_cache()
    
    # Verify consolidation
    devices = set()
    for p in model.parameters():
        devices.add(p.device)
    for b in model.buffers():
        devices.add(b.device)
    
    if len(devices) > 1:
        print(f"  WARNING: Model still on multiple devices: {devices}")
    else:
        print(f"  ✓ Model consolidated to {device}")
    
    return model

# ── Try loading completed Phase 5 from Drive ──────────────────────────────────
try:
    model_p5, processor = load_model_from_drive("phase5_flap_pruned")
    sync_model_config(model_p5)
    model_p5 = _consolidate_to_single_gpu(model_p5)
    device = torch.device("cuda:0")
    model_p5 = force_model_to_single_device(model_p5, device)
    print("Loaded Phase 5 from Drive.")
    p5_loaded = True
except Exception as _e:
    print(f"No Phase 5 on Drive ({_e}), pruning from model_p4...")
    p5_loaded = False
 
# ── Run Phase 5 pruning ───────────────────────────────────────────────────────
if not p5_loaded:
    device = torch.device("cuda:0")
    
    # CRITICAL FIX: Consolidate model_p4 BEFORE deepcopy
    print("Consolidating model_p4 to single GPU before deepcopy...")
    model_p4 = _consolidate_to_single_gpu(model_p4)
    model_p4 = force_model_to_single_device(model_p4, device)
    
    # Verify model_p4 is on single device
    p4_devices = set(p.device for p in model_p4.parameters())
    print(f"  model_p4 devices after consolidation: {p4_devices}")
    if len(p4_devices) > 1:
        raise RuntimeError(f"model_p4 still split across {p4_devices}! Cannot proceed.")
    
    # Now deepcopy will preserve single-device placement
    print("Deep-copying model_p4 → model_p5...")
    model_p5 = _copy.deepcopy(model_p4)
    model_p5 = force_model_to_single_device(model_p5, device)
    
    # Final verification
    p5_devices = set(p.device for p in model_p5.parameters())
    print(f"  model_p5 devices after deepcopy: {p5_devices}")
    
    pre_params = count_params(model_p5)
    print(f"\nPre-pruning: {pre_params:.1f}M params")
 
    prune_results = {}
 
    for comp_name in ["text_decoder", "speech_encoder", "t2u_model"]:
        print(f"\n{'='*60}")
        print(f"  Collecting calibration for {comp_name}")
        print(f"{'='*60}")
        
        # Verify component device before calibration
        comp = getattr(model_p5, comp_name)
        comp_device = next(comp.parameters()).device
        print(f"  {comp_name} device: {comp_device}")
        
        if comp_device != device:
            print(f"  ERROR: {comp_name} on wrong device! Re-consolidating...")
            model_p5 = force_model_to_single_device(model_p5, device)
        
        calib = collect_ffn_calibration_stats(
            model_p5, comp_name, calib_wavs, processor,
            n_samples=min(64, len(calib_wavs)),
            device=device,
        )
        _stdlib_gc.collect()
        torch.cuda.empty_cache()
 
        print(f"\n{'='*60}")
        print(f"  Applying FLAP to {comp_name}")
        print(f"{'='*60}")
        
        results = apply_flap_to_component(
            model_p5, comp_name, calib,
            global_prune_ratio=FLAP_RATIO,
            min_keep_frac=MIN_KEEP_FRAC,
            device=device,
        )
        prune_results[comp_name] = results
        _stdlib_gc.collect()
        torch.cuda.empty_cache()
 
    post_params = count_params(model_p5)
    print(f"\n{'='*60}")
    print(f"Width pruning complete:")
    print(f"  {pre_params:.1f}M → {post_params:.1f}M")
    print(f"  Saved: {pre_params - post_params:.1f}M params")
    print(f"{'='*60}")
 
    sync_model_config(model_p5)
 
    for comp, res in prune_results.items():
        if res:
            avg_kept = np.mean([v["pct"] for v in res.values()])
            print(f"  {comp}: avg {avg_kept:.1f}% neurons kept ({len(res)} layers)")
 
    save_checkpoint({"prune_results": prune_results, "flap_ratio": FLAP_RATIO},
                    name="phase5_flap", step=0)
    sync_model_config(model_p5)
    save_model_to_drive(model_p5, processor, "phase5_flap_pruned")

print_model_breakdown(model_p5, "After Phase 5: FLAP Width Pruned")
