# ═══════════════════════════════════════════════════════════════════════════════
# FIX FOR PHASE 6C: Frozen Text Decoder Dropout Corruption
# ═══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM: The frozen text_decoder is in train mode during T2U conditioning,
# causing dropout to corrupt t2u_input_embeds even inside no_grad().
#
# SOLUTION: Force text_decoder to eval mode before running it in conditioning.
#
# Replace the existing build_t2u_conditioning_from_sequences function with this:
# ═══════════════════════════════════════════════════════════════════════════════

def build_t2u_conditioning_from_sequences(model, input_features, attention_mask, text_sequences):
    """
    FIXED VERSION: Sets frozen text_decoder to eval mode during conditioning
    to prevent dropout corruption of t2u_input_embeds.
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
    # CRITICAL FIX: frozen text_decoder must be in eval mode during conditioning.
    # In train mode, dropout corrupts t2u_input_embeds even inside no_grad().
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

    # T2U char path — model internal methods expect NEW IDs (confirmed by debug)
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
# UPDATED PHASE 6C TRAINING STEP WITH FIX
# ═══════════════════════════════════════════════════════════════════════════════

def phase6c_t2u_step_FIXED(sample, cache_entry):
    """
    FIXED VERSION: Includes dropout fix and detaches frozen conditioning outputs.
    """
    audio_inputs_student = processor(
        audio=sample["wav"],
        sampling_rate=16000,
        return_tensors="pt",
    )
    audio_inputs_student = {k: v.to(student_device) for k, v in audio_inputs_student.items()}

    teacher_text_sequences = cache_entry["teacher_text_sequences"].unsqueeze(0)

    # Teacher path on GPU1
    audio_inputs_teacher = {k: v.to(teacher_device) for k, v in audio_inputs_student.items()}
    teacher_text_sequences_gpu = teacher_text_sequences.to(teacher_device)

    with torch.no_grad():
        teacher_cond = build_t2u_conditioning_from_sequences(
            model_teacher,
            input_features=audio_inputs_teacher["input_features"],
            attention_mask=audio_inputs_teacher.get("attention_mask"),
            text_sequences=teacher_text_sequences_gpu,
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

    # ── Student conditioning (GPU0) — FROZEN path, NO_GRAD ─────────────────
    # CRITICAL FIX: speech_encoder + text_decoder are frozen.
    # Running them outside no_grad() keeps every intermediate activation
    # in RAM to support backward() — which never comes for frozen params.
    # This is the primary OOM cause.  Detach before passing to T2U so
    # gradients only flow through T2U's own parameters.
    with torch.no_grad():
        student_cond = build_t2u_conditioning_from_sequences(
            model_student,
            input_features=audio_inputs_student["input_features"],
            attention_mask=audio_inputs_student.get("attention_mask"),
            text_sequences=teacher_text_sequences.to(student_device),
        )
    # Detach to make inputs_embeds a leaf — T2U grads stay local to T2U.
    t2u_inputs_embeds = student_cond["t2u_input_embeds"].detach()

    # ── Student T2U (GPU0, trainable) ─────────────────────────────────────
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

        # Use weights from planning.md
        loss = 0.60 * t2u_soft + 0.30 * t2u_hard + 0.10 * t2u_len

    return loss, {
        "t2u_soft": t2u_soft.item(),
        "t2u_hard": t2u_hard.item(),
        "t2u_len": t2u_len.item(),
    }


print("✓ Fixed build_t2u_conditioning_from_sequences loaded")
print("✓ Fixed phase6c_t2u_step_FIXED loaded")
print("\nTo apply the fix:")
print("1. Run the diagnostic cells to confirm dropout corruption")
print("2. Replace the old function with this fixed version")
print("3. Restart Phase 6C training from step 0")
