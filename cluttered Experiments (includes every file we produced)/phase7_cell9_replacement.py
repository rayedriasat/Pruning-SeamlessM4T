# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 CELL 9: CORRECTED TRAINING LOOP
# S2TT-focused training with proper error handling
# ══════════════════════════════════════════════════════════════════════════════

import random, time, logging, gc as _stdlib_gc
 
MAX_STEPS  = 2000
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR         = 1e-4
GRAD_CLIP  = 1.0
LOG_EVERY  = 50
SAVE_EVERY = 200
 
trainable = [p for p in model_p7.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS)
 
ft_ckpt    = load_latest_checkpoint("phase7_ft")
start_step = 0
s2tt_log = []
 
if ft_ckpt and ft_ckpt.get("step", 0) > 0:
    start_step = ft_ckpt["step"]
    s2tt_log   = ft_ckpt.get("s2tt_log", [])
    ostate = ft_ckpt.get("optimizer_state")
    sstate = ft_ckpt.get("scheduler_state")
    if ostate: optimizer.load_state_dict(ostate)
    if sstate: scheduler.load_state_dict(sstate)
    print(f"Resuming from step {start_step}")
else:
    print("Starting Phase 7 from scratch.")
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_p7 = model_p7.to(device)
model_p7.train()
 
# Suppress verbose SeamlessM4T logging
_m4t_log = logging.getLogger(
    "transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2")
_prev_level = _m4t_log.level
_m4t_log.setLevel(logging.ERROR)
 
try:
    optim_steps        = start_step
    micro_step         = 0
    consecutive_errors = 0
    optimizer.zero_grad()
    t0 = time.time()
 
    while optim_steps < MAX_STEPS:
        # Sample batch from training data
        batch = random.sample(ft_samples, min(BATCH_SIZE, len(ft_samples)))
 
        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # ── S2TT loss (text decoder path) ──────────────────────────
                in_f, attn, txt_labels = prepare_s2tt_batch(
                    batch, processor, device, TARGET_LANG, model_p7)
                
                # Compute loss
                l_s2tt = compute_s2tt_loss(model_p7, in_f, attn, txt_labels)
                
                # Check if loss is valid
                if l_s2tt is None or torch.isnan(l_s2tt) or torch.isinf(l_s2tt):
                    print(f"  [WARN] Invalid loss at step {optim_steps}: {l_s2tt}")
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        print("CRITICAL: too many invalid losses. Stopping.")
                        break
                    continue
                
                # Scale loss for gradient accumulation
                loss = l_s2tt / GRAD_ACCUM
 
            # Backward pass
            loss.backward()
            consecutive_errors = 0
 
        except Exception as e:
            consecutive_errors += 1
            print(f"  [ERR] Step {optim_steps}: {type(e).__name__}: {e}")
            if consecutive_errors == 1:  # Print full traceback on first error
                import traceback
                traceback.print_exc()
            optimizer.zero_grad()
            if consecutive_errors > 5:
                print("CRITICAL: too many consecutive errors. Stopping.")
                break
            _stdlib_gc.collect()
            torch.cuda.empty_cache()
            continue
 
        # Log loss
        s2tt_log.append(l_s2tt.item() if isinstance(l_s2tt, torch.Tensor) else float(l_s2tt))
 
        # Optimizer step after gradient accumulation
        if (micro_step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optim_steps += 1
 
            if optim_steps % LOG_EVERY == 0:
                avg_s2tt = np.mean(s2tt_log[-LOG_EVERY:])
                elapsed  = time.time() - t0
                print(f"Step {optim_steps:>5}/{MAX_STEPS}  "
                      f"S2TT={avg_s2tt:.4f}  "
                      f"t={elapsed/60:.1f}min")
 
            if optim_steps % SAVE_EVERY == 0:
                save_checkpoint(dict(
                    step=optim_steps,
                    s2tt_log=s2tt_log,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                ), name="phase7_ft", step=optim_steps)
 
        micro_step += 1
 
    print(f"\\nTraining done. Steps: {optim_steps}  "
          f"Time: {(time.time()-t0)/60:.1f}min")
 
finally:
    _m4t_log.setLevel(_prev_level)
    save_checkpoint(dict(
        step=optim_steps,
        s2tt_log=s2tt_log,
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
    ), name="phase7_ft", step=optim_steps)
    print("Final checkpoint saved.")
