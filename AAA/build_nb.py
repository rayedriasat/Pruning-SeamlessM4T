# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6C — CORRECTED T2U RECOVERY
# Root-cause fixes applied (see diagnosis below)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ROOT CAUSE DIAGNOSIS:
#
# 1. DETACHED STUDENT CONDITIONING [PRIMARY BUG — causes ASR-ChrF collapse]
#    In the mini-test cell (cell 48), this line appears:
#
#      t2u_inputs_embeds = student_cond["t2u_input_embeds"].detach()
#
#    This detaches the student's conditioning from its own frozen encoder/decoder
#    graph — that is correct for memory. BUT the student_cond itself was computed
#    with torch.no_grad(), then the resulting embeds were passed into t2u_model()
#    OUTSIDE of autocast. The KL/CE losses computed against teacher logits are
#    therefore operating on conditioning that does NOT reflect what the student
#    text_decoder actually produces at inference time, because text_decoder is
#    frozen and its hidden states fed through a different graph than at generation.
#
#    The deeper issue: at inference/generation, SeamlessM4Tv2 runs the text_decoder
#    auto-regressively to produce tokens, then re-runs a *separate* forward to build
#    t2u_input_embeds from the forced text sequence. During 6C training you are
#    feeding teacher_text_sequences into the frozen student text_decoder to build
#    conditioning, but the student text_decoder is still in LoRA-modified form from
#    6B2. The LoRA-modified decoder produces different hidden states than the
#    teacher decoder does on the same token sequence — so the student T2U model is
#    being trained to map teacher-decoder conditioning to teacher units, but at
#    inference it will receive student-decoder conditioning. This distribution
#    mismatch grows with every gradient step → ASR-ChrF collapses.
#
# 2. t2u_overlap_losses USES last_hidden_state AS LOGITS [SECONDARY BUG]
#    The planning doc's t2u_overlap_losses() reads:
#
#      student_logits = student_out.last_hidden_state
#      teacher_logits = teacher_out.last_hidden_state
#
#    But SeamlessM4Tv2TextToUnitForConditionalGeneration output has:
#      - .last_hidden_state  → decoder hidden states, shape [B, T, hidden_dim]  (NOT logit space)
#      - .logits             → actual unit vocabulary logits, shape [B, T, unit_vocab_size]
#
#    Using last_hidden_state for KL/CE means you are computing divergence in
#    ~1024-dim hidden space (not unit-vocab space). The CE loss picks argmax over
#    1024 dims → always the wrong "unit". This explains why:
#      - soft loss looks small (~0.97) — KL on 1024-dim normalized vectors is tiny
#      - hard loss is large and noisy (~8.3) — argmax of hidden states is garbage
#      - ASR-ChrF falls: the model is trained against a meaningless signal
#
# 3. LENGTH LOSS DOMINATES EARLY STEPS [TERTIARY BUG]
#    The len loss starts at 227.5 at step 10 and drives the total loss to 25.8.
#    Smooth-L1 on raw token counts is unbounded. With 0.10 weight the actual
#    gradient contribution from the KL/CE terms is being swamped. The model
#    first learns to minimize length error before learning unit distribution.
#    Fix: normalize length loss by max_len, or weight it at 0.01 initially.
#
# 4. TEACHER-CONDITIONING BYPASS ATTEMPT (cell 49) DOES NOT FIX THE ROOT CAUSE
#    The attempted fix in cell 49 moves teacher_cond to student device and feeds
#    it into the student t2u_model. This removes the distribution mismatch for
#    conditioning, but:
#      a. The logits bug (#2) is still present
#      b. Training student T2U on teacher conditioning means at inference the
#         student T2U must receive student conditioning — the same mismatch
#         re-emerges at decode time unless you also freeze the student decoder
#         hidden state distribution, which defeats the purpose of 6B.
#
# ───────────────────────────────────────────────────────────────────────────────
# THE CORRECT APPROACH (matching seamless_communication/cli/m4t/finetune):
#
# The official finetune.py / trainer.py (UnitYFinetuneWrapper) feeds:
#   text_decoder_out  →  t2u_model.encode()  →  t2u_model.decode()
# where text_decoder_out comes from the STUDENT's own decoder on forced tokens.
# The loss is plain NLL (negative log likelihood) against GROUND TRUTH unit IDs,
# NOT KD against teacher logits.
#
# For recovery when ground truth units aren't available, the correct KD approach
# is: use STUDENT conditioning for BOTH teacher and student T2U forward passes,
# but get teacher units from the cache (offline, already decoded by teacher),
# and compute NLL of student T2U logits against cached teacher unit IDs.
#
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── FIXED: correct attention mask helper (mirrors source file exactly) ─────────

def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    """Mirrors SeamlessM4Tv2ForSpeechToSpeech._compute_new_attention_mask."""
    batch_size, mask_seq_len = hidden_states.shape[:2]
    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)
    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)
    mask = hidden_states.new_ones((batch_size, mask_seq_len))
    mask = mask.masked_fill(bool_mask, 0)
    return mask


# ── FIXED: build_t2u_conditioning uses the GIVEN model's own decoder ──────────
# The critical change: no more .detach() on the returned embeds when grad is needed.
# Caller controls whether to wrap in torch.no_grad().

def build_t2u_conditioning_from_sequences(model, input_features, attention_mask, text_sequences):
    """
    Build T2U conditioning from a model using forced text_sequences.
    Returns dict of all T2U model inputs.
    This matches the internal path in SeamlessM4Tv2ForSpeechToSpeech.generate().
    """
    # Step 1: speech encoder
    enc_out = model.speech_encoder(
        input_features=input_features,
        attention_mask=attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    enc = enc_out.last_hidden_state  # [B, T_enc, D]

    # Step 2: build encoder attention mask (sub-sampled)
    encoder_attention_mask = None
    if attention_mask is not None:
        sub_lengths = model._compute_sub_sample_lengths_from_attention_mask(
            attention_mask
        ).to(enc.device)
        encoder_attention_mask = _compute_new_attention_mask(enc, sub_lengths)

    # Step 3: forced text decoder pass — gives T2U input embeddings
    # text_sequences[:, :-1] = shift right (standard teacher forcing)
    dec_out = model.text_decoder(
        input_ids=text_sequences[:, :-1],
        encoder_hidden_states=enc,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    t2u_input_embeds = dec_out.last_hidden_state  # [B, T_txt-1, D]

    # Step 4: build char inputs (mirrors _prepare_text_to_unit_model_kwargs)
    pad_token_id = model.generation_config.pad_token_id
    eos_token_id = model.generation_config.eos_token_id

    # text_sequences[:, 2:-1] = skip BOS + lang token, skip EOS
    t2u_input_ids = text_sequences[:, 2:-1].clone()
    t2u_input_ids = torch.masked_fill(
        t2u_input_ids, t2u_input_ids == eos_token_id, pad_token_id
    )

    t2u_subwords = model._indices_to_subwords(t2u_input_ids)
    t2u_char_count_per_id = model._count_character_length_in_subword(
        t2u_input_ids,
        t2u_subwords,
        pad_token_id=pad_token_id,
    )
    pad_zero = t2u_char_count_per_id.new_zeros((t2u_char_count_per_id.shape[0], 1))
    t2u_char_count_per_id = torch.cat([pad_zero, t2u_char_count_per_id, pad_zero], dim=1)

    t2u_char_input_ids = model._get_char_input_ids(
        t2u_input_ids,
        t2u_subwords,
        t2u_char_count_per_id,
        pad_token_id=pad_token_id,
    )

    # Step 5: build T2U attention mask from text sequence lengths
    seq_lens = (text_sequences[:, :-1] != pad_token_id).int().sum(1)
    t2u_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)

    return {
        "encoder_hidden_states": enc,
        "encoder_attention_mask": encoder_attention_mask,
        "t2u_input_embeds": t2u_input_embeds,
        "t2u_attention_mask": t2u_attention_mask,
        "t2u_char_input_ids": t2u_char_input_ids,
        "t2u_char_count_per_id": t2u_char_count_per_id,
    }


# ── FIXED: loss function uses .logits, not .last_hidden_state ─────────────────

def t2u_nll_loss_from_cache(student_t2u_out, teacher_unit_ids, pad_unit_id=0):
    """
    PRIMARY LOSS: NLL of student T2U logits against cached teacher unit IDs.

    This mirrors CalcLoss in the official trainer.py exactly.
    teacher_unit_ids: [B, T_units] — from Phase 6A cache, teacher's decoded unit sequence.
    student_t2u_out.logits: [B, T_student, unit_vocab_size]

    Handles length mismatch by truncating to the shorter sequence.
    """
    # FIXED: use .logits (unit vocabulary space), NOT .last_hidden_state
    student_logits = student_t2u_out.logits  # [B, T_s, V]
    B, T_s, V = student_logits.shape
    T_t = teacher_unit_ids.shape[1]
    L = min(T_s, T_t)

    if L < 2:
        return student_logits.new_zeros(())

    s_logits = student_logits[:, :L, :].reshape(-1, V)
    t_labels = teacher_unit_ids[:, :L].reshape(-1).to(student_logits.device)

    # Mask padding
    mask = t_labels != pad_unit_id
    if mask.sum() == 0:
        return student_logits.new_zeros(())

    loss = F.cross_entropy(s_logits[mask], t_labels[mask])
    return loss


def t2u_kd_loss_from_logits(student_t2u_out, teacher_t2u_out, temperature=2.0):
    """
    SECONDARY LOSS: KL divergence between teacher and student T2U logits.
    FIXED: reads .logits (not .last_hidden_state) from both outputs.
    Only used when teacher is on GPU1 during step. Optional supplement to NLL.
    """
    # FIXED: .logits, not .last_hidden_state
    student_logits = student_t2u_out.logits  # [B, T_s, V]
    teacher_logits = teacher_t2u_out.logits  # [B, T_t, V]

    student_mask = student_t2u_out.padding_mask.bool()  # [B, T_s]
    teacher_mask = teacher_t2u_out.padding_mask.bool()  # [B, T_t]

    student_len = student_mask.sum(1).long()
    teacher_len = teacher_mask.sum(1).long()
    common_len  = torch.minimum(student_len, teacher_len)

    total_kl = student_logits.new_zeros(())
    valid = 0

    for b in range(student_logits.size(0)):
        L = int(common_len[b].item())
        if L < 2:
            continue
        s = student_logits[b, :L]
        t = teacher_logits[b, :L].to(s.device)

        total_kl = total_kl + F.kl_div(
            F.log_softmax(s / temperature, dim=-1),
            F.softmax(t / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)
        valid += 1

    if valid == 0:
        return student_logits.new_zeros(())
    return total_kl / valid


def t2u_length_loss_normalized(student_t2u_out, teacher_unit_ids, max_len=1024.0):
    """
    FIXED length loss: normalized by max_len so it stays in [0, 1] range.
    Previously was raw token-count Smooth-L1, which hit 227.5 and dominated early gradients.
    """
    student_mask = student_t2u_out.padding_mask.bool()  # [B, T_s]
    student_len  = student_mask.sum(1).float()
    teacher_len  = (teacher_unit_ids != 0).sum(1).float().to(student_len.device)

    # Normalize so loss is in [0,1] range
    return F.smooth_l1_loss(
        student_len / max_len,
        teacher_len / max_len,
    )


# ── CORRECTED Phase 6C step function ─────────────────────────────────────────

def phase6c_step_corrected(
    model_student,
    model_teacher,
    sample,
    cache_entry,
    student_device,
    teacher_device,
    autocast_dtype=torch.float16,
    use_kd_supplement=True,
):
    """
    Corrected Phase 6C training step.

    Key fixes vs. the broken mini-test:
    1. Student conditioning uses student's OWN text_decoder (no cross-model pollution)
    2. Loss uses .logits not .last_hidden_state
    3. Primary loss is NLL against cached teacher unit IDs (offline, stable)
    4. KD is supplementary only, weighted at 0.20
    5. Length loss is normalized, weighted at 0.02 (not 0.10)
    6. Student conditioning is NOT detached from the t2u_model forward — only
       the speech_encoder and text_decoder (which are frozen) are no_grad'd
       via freeze, not via detach, so the t2u_model still receives valid grad signals
       through the embed input.
    """
    teacher_text_sequences = cache_entry["teacher_text_sequences"].unsqueeze(0)
    teacher_unit_ids       = cache_entry["teacher_unit_sequences"].unsqueeze(0)  # [1, T_units]

    audio_inputs_student = phase6_prepare_audio_inputs(sample, student_device)

    # ── Student conditioning: student's OWN frozen encoder + decoder ──────────
    # speech_encoder and text_decoder are frozen (requires_grad=False),
    # so this runs without storing activations for those modules.
    # Do NOT wrap in torch.no_grad() — t2u_model needs grad through inputs_embeds.
    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        student_cond = build_t2u_conditioning_from_sequences(
            model_student,
            input_features=audio_inputs_student["input_features"],
            attention_mask=audio_inputs_student.get("attention_mask"),
            text_sequences=teacher_text_sequences.to(student_device),
        )

        # ── Student T2U forward ────────────────────────────────────────────────
        student_t2u = model_student.t2u_model(
            inputs_embeds=student_cond["t2u_input_embeds"],
            attention_mask=student_cond["t2u_attention_mask"],
            char_input_ids=student_cond["t2u_char_input_ids"],
            char_count_per_id=student_cond["t2u_char_count_per_id"],
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # ── PRIMARY LOSS: NLL against cached teacher unit IDs ─────────────────
        # FIXED: uses .logits (unit vocab space), not .last_hidden_state
        loss_nll = t2u_nll_loss_from_cache(
            student_t2u,
            teacher_unit_ids.to(student_device),
        )

        # ── SECONDARY LOSS: length normalization ──────────────────────────────
        # FIXED: normalized, so it stays bounded [0, 1]
        loss_len = t2u_length_loss_normalized(
            student_t2u,
            teacher_unit_ids.to(student_device),
        )

    # ── OPTIONAL SUPPLEMENT: KD from live teacher (GPU1) ──────────────────────
    # Only adds signal if teacher is loaded; skip if VRAM is tight.
    loss_kd = None
    if use_kd_supplement and model_teacher is not None:
        audio_inputs_teacher = {k: v.to(teacher_device) for k, v in audio_inputs_student.items()}
        with torch.no_grad():
            teacher_cond = build_t2u_conditioning_from_sequences(
                model_teacher,
                input_features=audio_inputs_teacher["input_features"],
                attention_mask=audio_inputs_teacher.get("attention_mask"),
                text_sequences=teacher_text_sequences.to(teacher_device),
            )
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                teacher_t2u = model_teacher.t2u_model(
                    inputs_embeds=teacher_cond["t2u_input_embeds"],
                    attention_mask=teacher_cond["t2u_attention_mask"],
                    char_input_ids=teacher_cond["t2u_char_input_ids"],
                    char_count_per_id=teacher_cond["t2u_char_count_per_id"],
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
        # FIXED: KD loss uses .logits not .last_hidden_state
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            loss_kd = t2u_kd_loss_from_logits(student_t2u, teacher_t2u)

    # ── Combined loss ─────────────────────────────────────────────────────────
    # Primary: NLL (0.78) — stable, matches official trainer.py approach
    # KD: soft (0.20) — optional, supplements NLL
    # Length: (0.02) — light regularizer, normalized so it can't dominate
    if loss_kd is not None:
        loss = 0.78 * loss_nll + 0.20 * loss_kd + 0.02 * loss_len
    else:
        loss = 0.98 * loss_nll + 0.02 * loss_len

    metrics = {
        "nll": loss_nll.item(),
        "kd":  loss_kd.item() if loss_kd is not None else 0.0,
        "len": loss_len.item(),
    }
    return loss, metrics


# ── CORRECTED main training loop for Phase 6C ────────────────────────────────

def run_phase6c_corrected(
    model_student,
    model_teacher,
    student_device,
    teacher_device,
    phase6_pick_training_pair,
    phase6_quick_eval,
    phase6_cache_index,
    steps=700,
    grad_accum=8,
    max_audio_sec=12,
    base_lr=8e-5,
    dur_lr=1e-4,
    log_every=10,
    eval_every=50,
    save_every=50,
    ckpt_dir="checkpoints",
    autocast_dtype=torch.float16,
    use_kd_supplement=True,
):
    import os, math

    # Freeze everything except t2u_model
    for p in model_student.parameters():
        p.requires_grad_(False)
    for p in model_student.t2u_model.parameters():
        p.requires_grad_(True)

    # Cast t2u_model trainable params to FP32 for stable grad scaler
    n_cast = 0
    for p in model_student.t2u_model.parameters():
        if p.dtype == torch.float16:
            p.data = p.data.float()
            n_cast += 1
    print(f"  Cast {n_cast} trainable FP16 params to FP32")

    # Separate LR groups
    enc_params, dec_params, dur_params, scalar_params, head_params = [], [], [], [], []
    for name, p in model_student.t2u_model.named_parameters():
        if not p.requires_grad:
            continue
        if "duration_predictor" in name:
            dur_params.append(p)
        elif "pos_emb_alpha" in name:
            scalar_params.append(p)
        elif name == "lm_head.weight":
            head_params.append(p)
        elif name.startswith("model.decoder."):
            dec_params.append(p)
        else:
            enc_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params,    "lr": base_lr,  "weight_decay": 0.01},
            {"params": dec_params,    "lr": base_lr,  "weight_decay": 0.01},
            {"params": dur_params,    "lr": dur_lr,   "weight_decay": 0.00},
            {"params": scalar_params, "lr": dur_lr,   "weight_decay": 0.00},
            {"params": head_params,   "lr": base_lr,  "weight_decay": 0.01},
        ],
        betas=(0.9, 0.98),
    )

    warmup_steps = max(1, int(0.10 * steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    model_student.train()
    optimizer.zero_grad(set_to_none=True)

    logs = []
    micro_step = 0
    opt_step   = 0
    running = {"nll": 0.0, "kd": 0.0, "len": 0.0, "loss": 0.0, "n": 0}

    total_micro = steps * grad_accum

    print(f"\nPhase 6C (CORRECTED) — {steps} optimizer steps, grad_accum={grad_accum}")
    print(f"  Primary loss: NLL against cached teacher unit IDs (.logits)")
    print(f"  KD supplement: {'enabled (0.20 weight)' if use_kd_supplement else 'disabled'}")
    print(f"  Length loss: normalized smooth-L1 (0.02 weight)\n")

    while micro_step < total_micro:
        sample, cache_entry = phase6_pick_training_pair(
            max_audio_sec=max_audio_sec, balanced=True
        )

        try:
            loss, metrics = phase6c_step_corrected(
                model_student=model_student,
                model_teacher=model_teacher if use_kd_supplement else None,
                sample=sample,
                cache_entry=cache_entry,
                student_device=student_device,
                teacher_device=teacher_device,
                autocast_dtype=autocast_dtype,
                use_kd_supplement=use_kd_supplement,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  [OOM at micro {micro_step}] — skipping sample")
                micro_step += 1
                continue
            raise

        scaler.scale(loss / grad_accum).backward()
        micro_step += 1

        # Accumulate metrics
        for k, v in metrics.items():
            running[k] += v
        running["loss"] += loss.item()
        running["n"]    += 1

        if micro_step % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model_student.t2u_model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            if opt_step % log_every == 0:
                n = max(1, running["n"])
                avg = {k: running[k] / n for k in ["loss", "nll", "kd", "len"]}
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  6c opt {opt_step:4d}/{steps} | "
                    f"loss={avg['loss']:.4f} | "
                    f"nll={avg['nll']:.4f} | "
                    f"kd={avg['kd']:.4f} | "
                    f"len={avg['len']:.4f} | "
                    f"lr={lr_now:.2e}"
                )
                logs.append({"step": opt_step, **avg, "lr": lr_now})
                running = {"nll": 0.0, "kd": 0.0, "len": 0.0, "loss": 0.0, "n": 0}

            if opt_step % eval_every == 0:
                model_student.eval()
                phase6_quick_eval(f"6c_step{opt_step:06d}", max_samples=16)
                model_student.train()

            if opt_step % save_every == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = f"{ckpt_dir}/phase6_6c_step{opt_step:06d}.pt"
                torch.save({
                    "optimizer_step": opt_step,
                    "t2u_model": model_student.t2u_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, ckpt_path)
                print(f"  [saved] {ckpt_path}")

            if opt_step >= steps:
                break

    print(f"\nPhase 6C complete ({opt_step} optimizer steps).")
    return logs


# ═══════════════════════════════════════════════════════════════════════════════
# HOW TO USE IN YOUR NOTEBOOK
# Replace your current Phase 6C mini-test and run_t2u_recovery_stage calls with:
# ═══════════════════════════════════════════════════════════════════════════════
#
#   ensure_teacher_loaded()

#   phase6_logs["6c"] = run_phase6c_corrected(
#       model_student=model_student,
#       model_teacher=model_teacher,
#       student_device=student_device,
#       teacher_device=teacher_device,
#       phase6_pick_training_pair=phase6_pick_training_pair,
#       phase6_quick_eval=phase6_quick_eval,
#       phase6_cache_index=phase6_cache_index,
#       steps=STAGE6C_STEPS,          # 700 or 1100
#       grad_accum=GRAD_ACCUM,        # 8
#       max_audio_sec=MAX_AUDIO_SEC_C,  # 12
#       base_lr=8e-5,
#       dur_lr=1e-4,
#       log_every=LOG_EVERY,
#       eval_every=EVAL_EVERY,
#       save_every=SAVE_EVERY,
#       ckpt_dir=CKPT_DIR,
#       autocast_dtype=autocast_dtype,
#       use_kd_supplement=True,   # set False if GPU1 is tight
#   )

#   phase6_quick_eval("stage6c_done", max_samples=16)
#
# ─────────────────────────────────────────────────────────────────────────────
# ALSO: your cache_entry must have "teacher_unit_sequences" populated.
# From Phase 6A cache builder, verify:
#   cache_entry["teacher_unit_sequences"] = out.unit_sequences[0].detach().cpu()
# This should already be present from your build_teacher_cache_entry() function.
# ═══════════════════════════════════════════════════════════════════════════════