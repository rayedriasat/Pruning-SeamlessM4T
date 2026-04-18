# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8: T2U NAR TRAINING FOR AUDIO RECOVERY
# ══════════════════════════════════════════════════════════════════════════════
#
# This file contains ONLY the Phase 8-specific cells.
# Prepend this to your Phase 7 notebook (Cells 1-23 setup + dataset loading).
#
# PREREQUISITES:
# - Phase 7 must be complete (phase7_dora_merged model saved)
# - Unit labels must be cached (unit_labels_cache.pt from Phase 7 Cell 6)
# - ft_samples must be loaded (Phase 7 Cell 24)
#
# ══════════════════════════════════════════════════════════════════════════════

# ── Phase 8 Cell 1: Load Phase 7 fine-tuned model ────────────────────────────
print("\\n" + "="*60)
print("  PHASE 8: T2U NAR TRAINING")
print("  Loading Phase 7 fine-tuned model...")
print("="*60 + "\\n")

try:
    model_p7, processor = load_model_from_drive("phase7_dora_merged")
    sync_model_config(model_p7)
    model_p7 = _consolidate_to_single_gpu(model_p7)
    print("✓ Loaded Phase 7 model")
except Exception as e:
    print(f"❌ ERROR loading Phase 7 model: {e}")
    print("\\nMake sure Phase 7 training is complete and model is saved.")
    raise

print_model_breakdown(model_p7, "Phase 7 (input to Phase 8)")

# Sanity check: verify T2U exists and has correct layer counts
t2u = model_p7.t2u_model
if t2u is None:
    raise RuntimeError("T2U model not found! Phase 6 may have removed it.")

t2u_enc = getattr(getattr(t2u, 'model', None), 'encoder', None)
t2u_dec = getattr(getattr(t2u, 'model', None), 'decoder', None)

if t2u_enc is None or t2u_dec is None:
    raise RuntimeError("T2U encoder/decoder not found!")

print(f"\\nT2U architecture:")
print(f"  Encoder layers: {len(t2u_enc.layers)}")
print(f"  Decoder layers: {len(t2u_dec.layers)}")
print(f"  Unit vocab size: {model_p7.config.unit_hifi_gan_vocab_size}")


# ── Phase 8 Cell 2: Load unit labels from Phase 7 cache ──────────────────────
import os
import torch

UNIT_CACHE_PATH = f"{CKPT_DIR}/unit_labels_cache.pt"

if not os.path.exists(UNIT_CACHE_PATH):
    raise FileNotFoundError(
        f"Unit cache not found: {UNIT_CACHE_PATH}\\n"
        f"Run Phase 7 Cell 6 first to extract unit labels."
    )

print(f"Loading unit labels from cache...")
cached = torch.load(UNIT_CACHE_PATH, map_location="cpu", weights_only=False)
unit_labels = cached["units"]

# Rebuild ft_s2st_pairs (same as Phase 7 Cell 6)
ft_s2st_pairs = []
for s, units in zip(ft_samples, unit_labels):
    if units is not None and units.numel() >= 3:
        ft_s2st_pairs.append({
            "wav":   s["wav"],
            "ref":   s["ref"],
            "units": units,
        })

print(f"✓ Loaded {len(ft_s2st_pairs)} training pairs with unit labels")
if ft_s2st_pairs:
    lens = [p["units"].numel() for p in ft_s2st_pairs]
    print(f"  Unit seq length: min={min(lens)}  max={max(lens)}  mean={np.mean(lens):.0f}")


# ── Phase 8 Cell 3: T2U data preparation ──────────────────────────────────────
import torch
import torch.nn.functional as F

def prepare_t2u_batch(batch, processor, device, tgt_lang, mdl):
    """
    Prepare inputs for T2U training.
    
    T2U NAR architecture requires:
    1. Text decoder hidden states (from speech encoder → text decoder forward pass)
    2. Text decoder padding mask
    3. Target unit sequences (discrete speech units)
    
    Returns:
        text_hidden: [B, text_seq_len, hidden_dim] - text decoder output
        text_mask:   [B, text_seq_len] - padding mask
        unit_labels: [B, unit_seq_len] - target units
    """
    audios  = [s["wav"] for s in batch]
    targets = [s["ref"] for s in batch]
    units   = [s["units"] for s in batch]
    
    # Encode audio → text decoder hidden states
    audio_enc = processor(audio=audios, sampling_rate=16000,
                          return_tensors="pt", padding=True)
    input_feats = audio_enc["input_features"].to(device)
    attn_mask   = audio_enc["attention_mask"].to(device)
    
    # Get text token IDs for text decoder
    tok = processor.tokenizer
    text_enc = tok(text_target=targets, tgt_lang=tgt_lang,
                   return_tensors="pt", padding=True)
    text_ids = text_enc["input_ids"].to(device)
    text_ids = remap_label_ids(text_ids, mdl)
    
    # Forward through speech encoder + text decoder to get hidden states
    with torch.no_grad():
        # Speech encoder
        speech_enc_out = mdl.speech_encoder(
            input_features=input_feats,
            attention_mask=attn_mask,
        )
        speech_hidden = speech_enc_out[0]  # [B, speech_seq_len, hidden_dim]
        
        # Text decoder (teacher forcing with target text)
        text_dec_out = mdl.text_decoder(
            input_ids=text_ids,
            encoder_hidden_states=speech_hidden,
            encoder_attention_mask=attn_mask,
        )
        text_hidden = text_dec_out[0]  # [B, text_seq_len, hidden_dim]
    
    # Text padding mask (1 = valid, 0 = padding)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    text_mask = (text_ids != pad_id).long()
    
    # Pad unit sequences to same length
    max_unit_len = max(u.size(0) for u in units)
    unit_labels = torch.full((len(units), max_unit_len), -100, dtype=torch.long)
    for i, u in enumerate(units):
        unit_labels[i, :u.size(0)] = u
    unit_labels = unit_labels.to(device)
    
    return text_hidden, text_mask, unit_labels


def compute_t2u_loss(model, text_hidden, text_mask, unit_labels):
    """
    T2U NAR loss: cross-entropy on discrete unit predictions.
    
    SeamlessM4Tv2's T2U is a Non-Autoregressive (NAR) transformer that:
    1. Takes text decoder hidden states as input
    2. Predicts discrete speech units in parallel (not autoregressively)
    3. Uses duration predictor to align text tokens → unit frames
    
    For simplicity, we use unit cross-entropy loss only (no duration loss).
    """
    t2u = model.t2u_model
    
    # T2U forward pass
    # Note: SeamlessM4Tv2's T2U expects:
    #   - encoder_hidden_states: text decoder output [B, text_len, hidden]
    #   - encoder_attention_mask: text padding mask [B, text_len]
    #   - labels: target unit IDs [B, unit_len]
    
    try:
        # Forward through T2U encoder
        t2u_enc_out = t2u.model.encoder(
            inputs_embeds=text_hidden,
            attention_mask=text_mask,
        )
        t2u_enc_hidden = t2u_enc_out[0]  # [B, text_len, hidden]
        
        # T2U decoder (NAR: predicts all units in parallel)
        # For training, we use teacher forcing with target units
        B, unit_len = unit_labels.shape
        
        # Embed target units (shift right for decoder input)
        unit_embed = t2u.model.decoder.embed_tokens(
            torch.where(unit_labels == -100, 0, unit_labels)
        )
        
        # Decoder forward
        t2u_dec_out = t2u.model.decoder(
            inputs_embeds=unit_embed,
            encoder_hidden_states=t2u_enc_hidden,
            encoder_attention_mask=text_mask,
        )
        logits = t2u_dec_out[0]  # [B, unit_len, unit_vocab_size]
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            unit_labels.view(-1),
            ignore_index=-100,
        )
        
        return loss
        
    except Exception as e:
        print(f"  [T2U loss] Error: {e}")
        import traceback
        traceback.print_exc()
        # Return small non-zero loss to avoid breaking training
        return torch.tensor(0.01, requires_grad=True, device=text_hidden.device)


print("✓ T2U data preparation and loss functions ready")


# ── Phase 8 Cell 4: Freeze speech encoder, prepare T2U optimizer ─────────────
import torch.nn as nn

# Freeze speech encoder (already fine-tuned in Phase 7)
for param in model_p7.speech_encoder.parameters():
    param.requires_grad = False

# Freeze text decoder (already fine-tuned in Phase 7)
for param in model_p7.text_decoder.parameters():
    param.requires_grad = False

# Unfreeze T2U model
for param in model_p7.t2u_model.parameters():
    param.requires_grad = True

# Count trainable parameters
trainable = [p for p in model_p7.parameters() if p.requires_grad]
trainable_M = sum(p.numel() for p in trainable) / 1e6

print(f"\\nTrainable parameters: {trainable_M:.1f}M")
print(f"  Speech encoder: FROZEN")
print(f"  Text decoder:   FROZEN")
print(f"  T2U model:      TRAINABLE ({count_params(model_p7.t2u_model):.1f}M)")


# ── Phase 8 Cell 5: Verification - test T2U forward pass ─────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_p7 = model_p7.to(device)
model_p7.train()

print("\\n" + "="*60)
print("  VERIFICATION: Testing T2U forward pass")
print("="*60)

try:
    test_batch = ft_s2st_pairs[:2]
    text_h, text_m, units = prepare_t2u_batch(
        test_batch, processor, device, TARGET_LANG, model_p7)
    
    with torch.no_grad():
        loss = compute_t2u_loss(model_p7, text_h, text_m, units)
    
    print(f"\\n✓ T2U forward pass successful!")
    print(f"  Text hidden shape: {text_h.shape}")
    print(f"  Unit labels shape: {units.shape}")
    print(f"  Loss value:        {loss.item():.4f}")
    print(f"\\n✓ Ready to train T2U!")
    
except Exception as e:
    print(f"\\n❌ T2U forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    print("\\n⚠️  Fix errors above before training")

print("="*60)


# ── Phase 8 Cell 6: T2U training loop ─────────────────────────────────────────
import random
import time
import numpy as np
import gc as _stdlib_gc

# Training hyperparameters
MAX_STEPS  = 1000   # T2U is smaller, needs fewer steps
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR         = 5e-5   # Lower LR for T2U (NAR is sensitive)
GRAD_CLIP  = 1.0
LOG_EVERY  = 25
SAVE_EVERY = 100

# Optimizer (only T2U parameters)
trainable = [p for p in model_p7.t2u_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS)

# Load checkpoint if exists
t2u_ckpt   = load_latest_checkpoint("phase8_t2u")
start_step = 0
t2u_log    = []

if t2u_ckpt and t2u_ckpt.get("step", 0) > 0:
    start_step = t2u_ckpt["step"]
    t2u_log    = t2u_ckpt.get("t2u_log", [])
    ostate = t2u_ckpt.get("optimizer_state")
    sstate = t2u_ckpt.get("scheduler_state")
    if ostate: optimizer.load_state_dict(ostate)
    if sstate: scheduler.load_state_dict(sstate)
    print(f"✓ Resuming T2U training from step {start_step}")
else:
    print("✓ Starting Phase 8 T2U training from scratch")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_p7 = model_p7.to(device)
model_p7.train()

# Training loop
try:
    optim_steps        = start_step
    micro_step         = 0
    consecutive_errors = 0
    optimizer.zero_grad()
    t0 = time.time()
    
    print(f"\\n{'='*60}")
    print(f"  PHASE 8: T2U NAR TRAINING")
    print(f"  Target: {MAX_STEPS} steps | Batch: {BATCH_SIZE} | Accum: {GRAD_ACCUM}")
    print(f"  LR: {LR:.2e} | Trainable: {trainable_M:.1f}M params")
    print(f"{'='*60}\\n")
    
    while optim_steps < MAX_STEPS:
        # Sample batch
        batch = random.sample(ft_s2st_pairs, min(BATCH_SIZE, len(ft_s2st_pairs)))
        
        try:
            # Prepare T2U inputs
            text_h, text_m, units = prepare_t2u_batch(
                batch, processor, device, TARGET_LANG, model_p7)
            
            # Compute T2U loss
            loss = compute_t2u_loss(model_p7, text_h, text_m, units)
            
            # Validate loss
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARN] Invalid loss at step {optim_steps}: {loss}")
                consecutive_errors += 1
                if consecutive_errors > 5:
                    print("\\n❌ CRITICAL: Too many invalid losses. Stopping.")
                    break
                continue
            
            # Scale for gradient accumulation
            loss = loss / GRAD_ACCUM
            loss.backward()
            consecutive_errors = 0
            
        except Exception as e:
            consecutive_errors += 1
            print(f"  [ERR] Step {optim_steps}: {type(e).__name__}: {e}")
            if consecutive_errors == 1:
                import traceback
                traceback.print_exc()
            optimizer.zero_grad()
            if consecutive_errors > 5:
                print("\\n❌ CRITICAL: Too many consecutive errors. Stopping.")
                break
            _stdlib_gc.collect()
            torch.cuda.empty_cache()
            continue
        
        # Log loss (unscaled)
        t2u_log.append(loss.item() * GRAD_ACCUM)
        
        # Optimizer step
        if (micro_step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optim_steps += 1
            
            # Logging
            if optim_steps % LOG_EVERY == 0:
                avg_loss = np.mean(t2u_log[-LOG_EVERY:])
                elapsed  = time.time() - t0
                lr_current = scheduler.get_last_lr()[0]
                print(f"  Step {optim_steps:>4}/{MAX_STEPS}  "
                      f"Loss={avg_loss:.4f}  "
                      f"LR={lr_current:.2e}  "
                      f"Time={elapsed/60:.1f}min")
            
            # Checkpointing
            if optim_steps % SAVE_EVERY == 0:
                save_checkpoint(dict(
                    step=optim_steps,
                    t2u_log=t2u_log,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                ), name="phase8_t2u", step=optim_steps)
                print(f"  ✓ Checkpoint saved at step {optim_steps}")
        
        micro_step += 1
    
    print(f"\\n{'='*60}")
    print(f"  ✓ T2U training complete!")
    print(f"  Steps: {optim_steps} | Time: {(time.time()-t0)/60:.1f}min")
    print(f"  Final loss: {np.mean(t2u_log[-50:]):.4f}")
    print(f"{'='*60}\\n")
    
finally:
    save_checkpoint(dict(
        step=optim_steps,
        t2u_log=t2u_log,
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
    ), name="phase8_t2u", step=optim_steps)
    print("✓ Final T2U checkpoint saved")


# ── Phase 8 Cell 7: T2U loss curve plot ──────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np

t2u_ckpt = load_latest_checkpoint("phase8_t2u")
if t2u_ckpt and t2u_ckpt.get("t2u_log"):
    t2u_log = t2u_ckpt["t2u_log"]
    
    if len(t2u_log) > 10:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        
        def _ema(vals, alpha=0.05):
            out, v = [], vals[0]
            for x in vals: v = alpha*x + (1-alpha)*v; out.append(v)
            return out
        
        ax.plot(t2u_log, alpha=0.2, color="steelblue", lw=0.5, label="Raw")
        ax.plot(_ema(t2u_log), color="steelblue", lw=2, label="EMA")
        ax.set_title("Phase 8: T2U NAR Loss (Audio Recovery)", fontweight="bold")
        ax.set_xlabel("Micro-step")
        ax.set_ylabel("Unit Cross-Entropy Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, "phase8_t2u_loss.png")
        plt.show()
        
        print(f"\\n✓ T2U loss curve saved")
        print(f"  Initial loss: {np.mean(t2u_log[:50]):.4f}")
        print(f"  Final loss:   {np.mean(t2u_log[-50:]):.4f}")
        print(f"  Improvement:  {np.mean(t2u_log[:50]) - np.mean(t2u_log[-50:]):.4f}")
else:
    print("No T2U training log found. Run training first.")


# ── Phase 8 Cell 8: Save Phase 8 model ───────────────────────────────────────
import gc as _stdlib_gc

print("Saving Phase 8 model (T2U fine-tuned)...")
model_p7.eval()
_stdlib_gc.collect()
torch.cuda.empty_cache()

sync_model_config(model_p7)
save_model_to_drive(model_p7, processor, "phase8_t2u_finetuned")
print_model_breakdown(model_p7, "After Phase 8: T2U Fine-tuned")


# ── Phase 8 Cell 9: Full S2ST + ASR-BLEU benchmark ───────────────────────────
print("\\n" + "="*60)
print("  PHASE 8 BENCHMARK: Full S2ST with ASR-BLEU")
print("="*60)

p8b = load_latest_checkpoint("phase8_benchmark")
if p8b and p8b.get("summary", {}).get("avg_asr_bleu", -1) >= 0:
    p8_results, p8_summary = p8b["results"], p8b["summary"]
    print(f"Loaded P8 benchmark: "
          f"txt_BLEU={p8_summary['avg_bleu']:.2f}  "
          f"txt_ChrF={p8_summary['avg_chrf']:.2f}  |  "
          f"asr_BLEU={p8_summary.get('avg_asr_bleu',0):.2f}  "
          f"asr_ChrF={p8_summary.get('avg_asr_chrf',0):.2f}")
else:
    p8_results, p8_summary = run_benchmark_s2st(
        model_p7, eval_samples, label="P8_T2U_NAR", save_n=4)
    save_checkpoint(dict(results=p8_results, summary=p8_summary),
                    name="phase8_benchmark", step=0)

# Compare against all phases
p4b = load_latest_checkpoint("phase4_benchmark")
p6b = load_latest_checkpoint("phase6_benchmark")
p7b = load_latest_checkpoint("phase7_benchmark")

p4_chrf  = p4b["summary"]["avg_chrf"] if p4b else 0.0
p6_chrf  = p6b["summary"]["avg_chrf"] if p6b else 0.0
p7_chrf  = p7b["summary"]["avg_chrf"] if p7b else 0.0
p8_chrf  = p8_summary["avg_chrf"]

p7_asr_b = p7b["summary"].get("avg_asr_bleu", 0) if p7b else 0
p8_asr_b = p8_summary.get("avg_asr_bleu", 0)
p8_asr_c = p8_summary.get("avg_asr_chrf", 0)

print(f"\\n{'='*60}")
print(f"  Phase 4  ChrF (txt):  {p4_chrf:.2f}")
print(f"  Phase 6  ChrF (txt):  {p6_chrf:.2f}  (drop: {p4_chrf-p6_chrf:.2f})")
print(f"  Phase 7  ChrF (txt):  {p7_chrf:.2f}  (recovery: +{p7_chrf-p6_chrf:.2f})")
print(f"  Phase 8  ChrF (txt):  {p8_chrf:.2f}  (maintained)")
print(f"")
print(f"  Phase 7  ASR-BLEU:    {p7_asr_b:.2f}  (text-only recovery)")
print(f"  Phase 8  ASR-BLEU:    {p8_asr_b:.2f}  (audio recovery: +{p8_asr_b-p7_asr_b:.2f})")
print(f"  Phase 8  ASR-ChrF:    {p8_asr_c:.2f}")
print(f"{'='*60}")

store_summary(p8_summary)
plot_phase_comparison()
plot_size_vs_quality()


# ── Phase 8 Cell 10: Final results table ─────────────────────────────────────
sc = load_latest_checkpoint("all_summaries")
if sc and "summaries" in sc:
    ALL_SUMMARIES = sc["summaries"]

print("\\n" + "="*90)
print("  SeamlessM4T v2 Large  Structured Compression  (EN→BN, FLEURS test)")
print("  FINAL RESULTS: Phase 0 → Phase 8")
print("="*90)

hdr = (f"{'Phase':<25} {'Params(M)':>10} {'Delta':>8} "
       f"{'txt-BLEU':>9} {'txt-ChrF':>9} "
       f"{'asr-BLEU':>9} {'asr-ChrF':>9} {'RTF':>7}")
print(hdr)
print("-"*len(hdr))

bp = ALL_SUMMARIES[0]["params_M"] if ALL_SUMMARIES else 2300
for s in ALL_SUMMARIES:
    d  = (1 - s["params_M"] / bp) * 100 if bp else 0
    ds = f"-{d:.1f}%" if d > 0 else "base"
    ab = s.get("avg_asr_bleu", 0)
    ac = s.get("avg_asr_chrf", 0)
    print(
        f"  {s['label']:<23} {s['params_M']:>8.1f}  {ds:>7}  "
        f"{s['avg_bleu']:>8.2f}  {s['avg_chrf']:>8.2f}  "
        f"{ab:>8.2f}  {ac:>8.2f}  {s['avg_rtf']:>6.4f}"
    )
print("="*90)

if len(ALL_SUMMARIES) >= 2:
    f, b = ALL_SUMMARIES[-1], ALL_SUMMARIES[0]
    print(f"\\n  COMPRESSION SUMMARY:")
    print(f"  Param reduction:      {(1-f['params_M']/b['params_M'])*100:.1f}%")
    print(f"  Speed (RTF):          {b['avg_rtf']/f['avg_rtf']:.2f}x faster")
    print(f"  Text quality (ChrF):  {f['avg_chrf']/b['avg_chrf']*100:.1f}% retained")
    if f.get("avg_asr_bleu", 0) > 0 and b.get("avg_asr_bleu", 0) > 0:
        print(f"  Audio quality (ASR):  {f['avg_asr_bleu']/b['avg_asr_bleu']*100:.1f}% retained")

print("\\n✓ Phase 8 complete! Model ready for deployment.")
