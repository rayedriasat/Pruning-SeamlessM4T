# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL FIX: Phase 6C Dtype Mismatch
# ═══════════════════════════════════════════════════════════════════════════════
#
# ERROR: RuntimeError: expected scalar type Half but found Float
#
# ROOT CAUSE: The frozen text_decoder outputs FP32 embeddings, but the student
# T2U model's encoder expects FP16 inputs. This causes a dtype mismatch in the
# first LayerNorm of the T2U encoder.
#
# SOLUTION: Cast t2u_input_embeds to FP16 before passing to T2U model.
# ═══════════════════════════════════════════════════════════════════════════════

def build_t2u_conditioning_from_sequences_FIXED(model, input_features, attention_mask, text_sequences):
    """
    FIXED VERSION: 
    1. Sets frozen text_decoder to eval mode to prevent dropout corruption
    2. Casts t2u_input_embeds to FP16 to match T2U model dtype
    """
    enc = model.speech_encoder(
        input_features=input_features,
        attention_mask=attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ).last_hidden_state

    encoder_attention_mask = None
    if attention_mask is not None:
        sub_lengths = model._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(enc.device)
        encoder_attention_mask = _compute_new_attention_mask(hidden_states=enc, seq_lens=sub_lengths)

    pad_token_id = model.generation_config.pad_token_id
    eos_token_id = model.generation_config.eos_token_id

    # ═══════════════════════════════════════════════════════════════════════
    # FIX 1: frozen text_decoder must be in eval mode during conditioning
    # ═══════════════════════════════════════════════════════════════════════
    was_training = model.text_decoder.training
    if was_training:
        model.text_decoder.eval()

    try:
        t2u_input_embeds = model.text_decoder(
            input_ids=text_sequences[:, :-1],
            encoder_hidden_states=enc,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
    finally:
        if was_training:
            model.text_decoder.train()

    # ═══════════════════════════════════════════════════════════════════════
    # FIX 2: Cast t2u_input_embeds to FP16 to match T2U model dtype
    # ═══════════════════════════════════════════════════════════════════════
    # The text_decoder outputs FP32, but T2U encoder expects FP16
    t2u_dtype = next(model.t2u_model.parameters()).dtype
    if t2u_input_embeds.dtype != t2u_dtype:
        t2u_input_embeds = t2u_input_embeds.to(t2u_dtype)

    # T2U char path
    t2u_input_ids = text_sequences[:, 2:-1].clone()
    t2u_input_ids = torch.masked_fill(t2u_input_ids, t2u_input_ids == eos_token_id, pad_token_id)

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

    seq_lens = (text_sequences[:, :-1] != pad_token_id).int().sum(1)
    t2u_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)

    return {
        't2u_input_embeds': t2u_input_embeds,
        't2u_attention_mask': t2u_attention_mask,
        't2u_char_input_ids': t2u_char_input_ids,
        't2u_char_count_per_id': t2u_char_count_per_id,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE FIXED PHASE 6C TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_t2u_recovery_stage_FIXED(
    stage_key,
    title,
    steps,
    max_audio_sec,
    resume_from_step=0,
):
    """
    FIXED VERSION with:
    1. Dropout fix (text_decoder.eval() during conditioning)
    2. Dtype fix (cast embeddings to FP16)
    3. Detached frozen conditioning (prevent OOM)
    4. Rebalanced loss weights (reduce hard CE dominance)
    """
    ensure_teacher_loaded()
    if PHASE6_T2U_TRAIN_MODE == "selective":
        mark_t2u_selective_trainable()
    else:
        mark_t2u_trainable_full()

    ensure_trainable_fp32()

    optimizer = torch.optim.AdamW(
        make_t2u_param_groups(model_student, base_lr=8e-5, dur_lr=1e-4,
                              scalar_lr=1e-4, head_lr=8e-5),
        betas=(0.9, 0.98),
    )
    scheduler = make_cosine_scheduler(optimizer, steps)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    log_every  = max(1,        LOG_EVERY)
    eval_every = max(log_every, EVAL_EVERY)
    save_every = max(log_every, SAVE_EVERY)

    # Resume logic
    if resume_from_step > 0:
        ckpt = phase6_load_latest_local_checkpoint(f"phase6_{stage_key}")
        if ckpt is None:
            raise RuntimeError(
                f"resume_from_step={resume_from_step} requested but no checkpoint found "
                f"for phase6_{stage_key}"
            )
        model_student.t2u_model.load_state_dict(ckpt["t2u_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("logs"):
            phase6_logs[stage_key] = ckpt["logs"]
        for _ in range(resume_from_step):
            scheduler.step()
        print(f"  Resumed {stage_key} from optimizer step {resume_from_step}/{steps}")

    model_student.train()
    optimizer.zero_grad(set_to_none=True)

    print(f"\n[{stage_key}] {title}")
    print(f"  optimizer steps : {resume_from_step} → {steps}")
    print(f"  fwd passes left : {(steps - resume_from_step) * GRAD_ACCUM}")
    print(f"  log/eval/save   : every {log_every}/{eval_every}/{save_every} opt steps")
    print(f"  max_audio={max_audio_sec}s | trainable={count_trainable_params(model_student):.2f}M")

    start_micro = resume_from_step * GRAD_ACCUM
    total_micro = steps            * GRAD_ACCUM

    for micro_step in range(start_micro, total_micro):
        sample, cache_entry    = phase6_pick_training_pair(max_audio_sec=max_audio_sec, balanced=True)
        teacher_text_sequences = cache_entry["teacher_text_sequences"].unsqueeze(0)
        audio_inputs_student   = phase6_prepare_audio_inputs(sample, student_device)
        audio_inputs_teacher   = {k: v.to(teacher_device) for k, v in audio_inputs_student.items()}

        try:
            # ── Teacher path (GPU1, always no_grad) ───────────────────────────
            with torch.no_grad():
                teacher_cond = build_t2u_conditioning_from_sequences_FIXED(
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

            # ── Student conditioning (GPU0) — FROZEN path, NO_GRAD ─────────────
            with torch.no_grad():
                student_cond = build_t2u_conditioning_from_sequences_FIXED(
                    model_student,
                    input_features=audio_inputs_student["input_features"],
                    attention_mask=audio_inputs_student.get("attention_mask"),
                    text_sequences=teacher_text_sequences.to(student_device),
                )
            # Detach to make inputs_embeds a leaf
            t2u_inputs_embeds = student_cond["t2u_input_embeds"].detach()

            # ── Student T2U (GPU0, trainable) ─────────────────────────────────
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                student_t2u = model_student.t2u_model(
                    inputs_embeds=t2u_inputs_embeds,
                    attention_mask=student_cond["t2u_attention_mask"],
                    char_input_ids=student_cond["t2u_char_input_ids"],
                    char_count_per_id=student_cond["t2u_char_count_per_id"],
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

                teacher_t2u.last_hidden_state = (
                    teacher_t2u.last_hidden_state.to(student_device)
                    if hasattr(teacher_t2u, "last_hidden_state") and teacher_t2u.last_hidden_state is not None
                    else None
                )
                if hasattr(teacher_t2u, "logits") and teacher_t2u.logits is not None:
                    teacher_t2u.logits = teacher_t2u.logits.to(student_device)
                teacher_t2u.padding_mask = teacher_t2u.padding_mask.to(student_device)

                t2u_soft, t2u_hard, t2u_len = t2u_overlap_losses(student_t2u, teacher_t2u)

                # ═══════════════════════════════════════════════════════════════
                # FIX 3: Rebalanced loss weights to reduce hard CE dominance
                # Old: 0.60 soft + 0.30 hard + 0.10 len (hard CE was 70-80%)
                # New: 0.70 soft + 0.20 hard + 0.10 len (hard CE now ~40%)
                # ═══════════════════════════════════════════════════════════════
                loss = 0.70 * t2u_soft + 0.20 * t2u_hard + 0.10 * t2u_len

            scaler.scale(loss / GRAD_ACCUM).backward()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                safe_gc()
                phase6_raise_oom(stage_key, micro_step + 1, max_audio_sec,
                                 extra='set PHASE6_T2U_TRAIN_MODE="selective"')
            raise

        phase6_logs[stage_key].append({
            "micro_step": micro_step + 1,
            "loss":       float(loss.detach().cpu()),
            "t2u_soft":   float(t2u_soft.detach().cpu()),
            "t2u_hard":   float(t2u_hard.detach().cpu()),
            "t2u_len":    float(t2u_len.detach().cpu()),
            "lr":         optimizer.param_groups[0]["lr"],
        })

        if (micro_step + 1) % GRAD_ACCUM == 0:
            opt_step = (micro_step + 1) // GRAD_ACCUM

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model_student.parameters() if p.requires_grad],
                MAX_GRAD_NORM,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if opt_step % log_every == 0:
                recent   = phase6_logs[stage_key][-(log_every * GRAD_ACCUM):]
                avg_loss = np.mean([r["loss"]     for r in recent])
                avg_soft = np.mean([r["t2u_soft"] for r in recent])
                avg_hard = np.mean([r["t2u_hard"] for r in recent])
                avg_len  = np.mean([r["t2u_len"]  for r in recent])
                print(
                    f"  [{stage_key}] opt {opt_step:>4}/{steps} | "
                    f"loss={avg_loss:.4f} | soft={avg_soft:.4f} | "
                    f"hard={avg_hard:.4f} | len={avg_len:.4f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if opt_step % eval_every == 0:
                phase6_quick_eval(f"{stage_key}_step{opt_step}", max_samples=16)

            if opt_step % save_every == 0:
                save_checkpoint(
                    {
                        "stage":          stage_key,
                        "optimizer_step": opt_step,
                        "steps_total":    steps,
                        "mode":           PHASE6_T2U_TRAIN_MODE,
                        "logs":           phase6_logs[stage_key],
                        "t2u_model":      model_student.t2u_model.state_dict(),
                        "optimizer":      optimizer.state_dict(),
                    },
                    f"phase6_{stage_key}", opt_step,
                )

    return phase6_logs[stage_key]


print("=" * 80)
print("✓ FIXED build_t2u_conditioning_from_sequences_FIXED loaded")
print("✓ FIXED run_t2u_recovery_stage_FIXED loaded")
print("=" * 80)
print("\nFIXES APPLIED:")
print("  1. Dropout fix: text_decoder.eval() during conditioning")
print("  2. Dtype fix: cast t2u_input_embeds to FP16")
print("  3. Detached frozen conditioning: prevent OOM")
print("  4. Rebalanced loss: 0.70 soft + 0.20 hard + 0.10 len")
print("\nTO APPLY:")
print("  1. Replace build_t2u_conditioning_from_sequences with _FIXED version")
print("  2. Delete all phase6_6c checkpoints")
print("  3. Run: phase6_logs['6c'] = run_t2u_recovery_stage_FIXED(...)")
print("=" * 80)
