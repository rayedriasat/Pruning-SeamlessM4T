# Phase 6 Recovery Plan for `phase5_dec_14L`: DoRA + Source-Grounded KD

This plan is written against the actual module structure in `AAA/modeling_seamless_m4t_v2.py`.
It avoids the broken assumptions in the current Phase 6 cells of `AAA/pragmata-recovery.ipynb`.

## 1. Hard facts from the saved model file

These are the facts we must code around:

1. `SeamlessM4Tv2ForSpeechToSpeech.forward()` does **not** train `t2u_model`.
   - See `AAA/modeling_seamless_m4t_v2.py` around the warning in the S2S `forward()`.
   - The top-level `forward()` only computes the text loss from `speech_encoder -> text_decoder -> lm_head`.
   - Therefore `outputs.t2u_loss` does not exist in the standard HF forward path.

2. T2U is explicitly treated as **non auto-regressive** in generation.
   - See the comment near the T2U path in `generate()`: "The text-to-unit model is non auto-regressive."
   - This is why T2U cannot be trained like a normal autoregressive decoder.

3. The current Phase 6 notebook targets several wrong module names:
   - Wrong: `text_decoder.layers.*.encoder_attn.*`
   - Correct in this file: `text_decoder.layers.*.cross_attention.*`
   - Wrong: `t2u_model.model.decoder.layers.*.fc1/fc2`
   - Correct: the T2U decoder layer has `self_attn`, `conv1`, `conv2`, no FFN block.

4. The T2U decoder has an internal duration predictor and predicts output length before CE loss.
   - `SeamlessM4Tv2TextToUnitDecoder.forward()`:
     - upsamples by `char_count_per_id`
     - runs `duration_predictor`
     - builds `dur_out`
     - upsamples again
     - then produces unit logits
   - So naive `labels=teacher_unit_ids` is unstable unless the student-predicted length already matches the label length.

5. There is a source bug if `output_attentions=True` in the T2U decoder path.
   - `SeamlessM4Tv2TextToUnitDecoderLayer.forward()` returns `(hidden_states, self_attn_weights)`.
   - `SeamlessM4Tv2TextToUnitDecoder.forward()` tries to read `layer_outputs[2]`.
   - Keep `output_attentions=False` for T2U training unless you patch the file first.

## 2. Why the current Phase 6 notebook fails

The current Phase 6 cells are wrong for structural reasons, not just tuning reasons:

1. Stage A feeds `speech_encoder(...).last_hidden_state` directly into `t2u_model(...)`.
   - That is not how this model is wired.
   - The real path is:
     - `speech_encoder`
     - `text_decoder` forced on text tokens
     - text-decoder hidden states become `t2u_model.inputs_embeds`
     - `char_input_ids` and `char_count_per_id` must also be built

2. Stages B and C expect `outputs.t2u_loss` from the top-level S2S forward.
   - That loss is not produced by the source file.

3. The LoRA target names in the notebook do not match the saved file.
   - `encoder_attn` should be `cross_attention`
   - T2U decoder `fc1/fc2` do not exist

4. The notebook uses `torch.cuda.amp.autocast(dtype=torch.bfloat16)`.
   - Kaggle T4 should be run in `float16`, not `bfloat16`.

## 3. DoRA constraints that matter here

PEFT DoRA is still configured through `LoraConfig(..., use_dora=True)`.

Important limitation from the PEFT docs:

- DoRA supports linear layers (and newer PEFT versions add more, but we should not rely on Conv1d support here).
- In the `peft>=0.10.0` range used by the notebook, do **not** assume Conv1d DoRA support.
- Therefore:
  - target `nn.Linear` modules
  - do **not** target T2U `conv1/conv2`
  - do **not** target speech encoder `conv_module.*`

Because DoRA adds more overhead than plain LoRA, the plan below uses smaller ranks than the old LoRA plan.

## 4. Recommended recovery strategy

Do **not** use the old 3-stage LoRA plan as written.
It is partly redundant and partly impossible with this HF forward path.

Recommended structure:

1. Phase 6A: offline teacher cache
2. Phase 6B: DoRA text-path recovery (`speech_encoder + text_decoder`)
3. Phase 6C: composite recovery (`speech_encoder + text_decoder + T2U`) with custom T2U KD
4. Phase 6D: optional short low-LR polish

This is still staged, but it is staged for architectural reasons, not just habit.

### Is one-stage fine-tuning better?

Not as the first implementation.

A single composite stage is possible only **after**:

- the teacher cache is correct
- the T2U helper path is correct
- the exact target modules are verified

For the notebook, the best first pass is:

- a short text-path recovery stage
- then a custom composite stage that includes T2U KD

That is simpler and safer than a monolithic "train everything from step 1" run.

## 5. Exact DoRA target modules from `AAA/modeling_seamless_m4t_v2.py`

Build targets dynamically from the loaded student model.
Do not hardcode layer counts from memory.

### 5.1 Strict dotted-path resolver

```python
import torch.nn as nn

def get_submodule_strict(root, dotted_name: str):
    cur = root
    for part in dotted_name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            if not hasattr(cur, part):
                raise AttributeError(f"Missing submodule part '{part}' in '{dotted_name}'")
            cur = getattr(cur, part)
    return cur

def assert_linear_targets_exist(model, target_names):
    bad = []
    for name in target_names:
        mod = get_submodule_strict(model, name)
        if not isinstance(mod, nn.Linear):
            bad.append((name, type(mod).__name__))
    if bad:
        msg = "\n".join([f"{n} -> {t}" for n, t in bad])
        raise TypeError(f"Non-linear DoRA targets found:\n{msg}")
```

### 5.2 Speech encoder DoRA targets

Use these exact names:

```python
def build_speech_encoder_dora_targets(model_student):
    n = len(model_student.speech_encoder.encoder.layers)
    targets = [
        "feature_projection.projection",
        "intermediate_ffn.intermediate_dense",
        "intermediate_ffn.output_dense",
    ]
    for i in range(n):
        prefix = f"encoder.layers.{i}"
        targets += [
            f"{prefix}.self_attn.linear_q",
            f"{prefix}.self_attn.linear_k",
            f"{prefix}.self_attn.linear_v",
            f"{prefix}.self_attn.linear_out",
            f"{prefix}.ffn1.intermediate_dense",
            f"{prefix}.ffn1.output_dense",
            f"{prefix}.ffn2.intermediate_dense",
            f"{prefix}.ffn2.output_dense",
        ]
    assert_linear_targets_exist(model_student.speech_encoder, targets)
    return targets
```

Do **not** target:

- `encoder.layers.*.conv_module.*`
- `adapter.*` Conv1d paths

### 5.3 Text decoder DoRA targets

Use these exact names:

```python
def build_text_decoder_dora_targets(model_student):
    n = len(model_student.text_decoder.layers)
    targets = []
    for i in range(n):
        prefix = f"layers.{i}"
        targets += [
            f"{prefix}.self_attn.q_proj",
            f"{prefix}.self_attn.k_proj",
            f"{prefix}.self_attn.v_proj",
            f"{prefix}.self_attn.out_proj",
            f"{prefix}.cross_attention.q_proj",
            f"{prefix}.cross_attention.k_proj",
            f"{prefix}.cross_attention.v_proj",
            f"{prefix}.cross_attention.out_proj",
            f"{prefix}.ffn.fc1",
            f"{prefix}.ffn.fc2",
        ]
    assert_linear_targets_exist(model_student.text_decoder, targets)
    return targets
```

Do **not** use `encoder_attn.*`. This file uses `cross_attention.*`.

### 5.4 T2U encoder DoRA targets

The T2U encoder is a standard `SeamlessM4Tv2Encoder` without input embeddings.

```python
def build_t2u_encoder_dora_targets(model_student):
    enc = model_student.t2u_model.model.encoder
    n = len(enc.layers)
    targets = []
    for i in range(n):
        prefix = f"layers.{i}"
        targets += [
            f"{prefix}.self_attn.q_proj",
            f"{prefix}.self_attn.k_proj",
            f"{prefix}.self_attn.v_proj",
            f"{prefix}.self_attn.out_proj",
            f"{prefix}.ffn.fc1",
            f"{prefix}.ffn.fc2",
        ]
    assert_linear_targets_exist(enc, targets)
    return targets
```

### 5.5 T2U decoder DoRA targets

The T2U decoder layer is **not** a normal FFN decoder.
It has:

- `self_attn`
- `conv1`
- `conv2`
- `duration_predictor`

DoRA-safe targets:

```python
def build_t2u_decoder_dora_targets(model_student):
    dec = model_student.t2u_model.model.decoder
    n = len(dec.layers)
    targets = []
    for i in range(n):
        prefix = f"layers.{i}"
        targets += [
            f"{prefix}.self_attn.q_proj",
            f"{prefix}.self_attn.k_proj",
            f"{prefix}.self_attn.v_proj",
            f"{prefix}.self_attn.out_proj",
        ]
    # Small but useful linear duration head
    targets += ["duration_predictor.proj"]
    assert_linear_targets_exist(dec, targets)
    return targets
```

Do **not** target:

- `layers.*.fc1`
- `layers.*.fc2`
- `layers.*.conv1`
- `layers.*.conv2`

For the T2U decoder, also allow these native parameters to train directly:

```python
T2U_NATIVE_TRAINABLE = [
    "pos_emb_alpha_char",
    "pos_emb_alpha",
]
```

## 6. Recommended DoRA configs

Because DoRA helps more at low rank than LoRA, start smaller than the old LoRA plan.

```python
from peft import LoraConfig

def make_dora_config(target_modules, r, alpha):
    return LoraConfig(
        target_modules=target_modules,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,   # deliberate: lower overhead, PEFT DoRA fast path
        bias="none",
        use_dora=True,
        use_rslora=True,
    )

speech_dora_cfg = make_dora_config(
    build_speech_encoder_dora_targets(model_student), r=16, alpha=32
)
text_dora_cfg = make_dora_config(
    build_text_decoder_dora_targets(model_student), r=16, alpha=32
)
t2u_enc_dora_cfg = make_dora_config(
    build_t2u_encoder_dora_targets(model_student), r=16, alpha=32
)
t2u_dec_dora_cfg = make_dora_config(
    build_t2u_decoder_dora_targets(model_student), r=8, alpha=16
)
```

Notes:

- `r=16` is the recommended default for the text path.
- `r=8` is enough for the T2U decoder because only a small linear subset is DoRA-compatible.
- If GPU0 stays below ~14 GB after 100 steady-state steps, you can retry text/speech at `r=24`.
- Do **not** start at `r=32` or `r=64` on T4 for DoRA.

## 7. Kaggle device plan (2 x T4, 16 GB each)

Use both GPUs, but do not split the student across them.

Recommended placement:

```python
student_device = torch.device("cuda:0")
teacher_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
```

Rules:

1. Student training stays on `cuda:0`
2. Teacher cache generation or teacher T2U KD runs on `cuda:1`
3. Do **not** call the notebook helper that forces both models onto `cuda:0`

Also set:

```python
model_student.config.use_cache = False
model_student.text_decoder.config.use_cache = False
model_student.gradient_checkpointing_enable()
model_student.speech_encoder.gradient_checkpointing_enable()
model_student.text_decoder.gradient_checkpointing_enable()
model_student.t2u_model.model.encoder.gradient_checkpointing_enable()
model_student.t2u_model.model.decoder.gradient_checkpointing_enable()
```

Use:

```python
autocast_dtype = torch.float16
```

Do not use `bfloat16` on T4.

## 8. Phase 6A: offline teacher cache

Cache teacher outputs once.
This is the single best speed/stability improvement for Kaggle.

### Cache contents

For each sample, store:

- `id`
- `src_lang`
- `tgt_lang`
- `teacher_text_sequences` (raw generated token ids)
- `teacher_text_str` (decoded text, for S2TT KD labels)
- `teacher_unit_sequences` (raw generated unit ids)
- `audio_len_s`

### Cache code sketch

```python
def build_teacher_cache_entry(model_teacher, processor, sample, device):
    inputs = processor(
        audio=sample["wav"],
        sampling_rate=16000,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model_teacher.generate(
            **inputs,
            tgt_lang=sample["tgt_lang"],
            return_intermediate_token_ids=True,
            text_num_beams=4,
            text_max_new_tokens=256,
            speech_do_sample=False,
        )

    if out.unit_sequences is None:
        raise RuntimeError(f"Teacher returned no unit sequence for sample {sample['id']}")

    teacher_text_sequences = out.sequences[0].detach().cpu()
    teacher_unit_sequences = out.unit_sequences[0].detach().cpu()
    teacher_text_str = processor.decode(
        teacher_text_sequences,
        skip_special_tokens=True,
    )

    return {
        "id": sample["id"],
        "src_lang": sample["src_lang"],
        "tgt_lang": sample["tgt_lang"],
        "teacher_text_sequences": teacher_text_sequences,
        "teacher_text_str": teacher_text_str,
        "teacher_unit_sequences": teacher_unit_sequences,
        "audio_len_s": len(sample["wav"]) / 16000.0,
    }
```

Strict checks:

- log and skip samples with empty teacher text
- log and skip samples with `len(teacher_unit_sequences) < 3`
- log and skip samples with clearly broken unit lengths (for example `> 1024`)

## 9. Phase 6B: text-path DoRA recovery

Goal:

- recover S2TT first
- stabilize the upstream hidden states that T2U will consume

### Stage 6B.1 (short warmup, recommended)

- train only `text_decoder` DoRA
- 200 to 400 steps
- micro-batch 1
- grad accumulation 8
- LR `1.5e-4`

### Stage 6B.2 (main text recovery)

- train `speech_encoder` + `text_decoder` DoRA
- 800 to 1200 steps
- micro-batch 1
- grad accumulation 8
- LR groups:
  - speech encoder DoRA: `5e-5`
  - text decoder DoRA: `8e-5`

### Text labels

Use:

- 50% cached teacher text (`teacher_text_str`)
- 50% gold reference text (`sample["ref"]`)

But build labels through the tokenizer target path, not by guessing special-token layout:

```python
def build_target_labels(processor, text_list, tgt_lang, device):
    tok = processor.tokenizer(
        text_target=text_list,
        tgt_lang=tgt_lang,
        return_tensors="pt",
        padding=True,
    )
    labels = tok["input_ids"].to(device)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return labels
```

### Training step sketch

```python
def text_recovery_step(model_student, processor, sample, use_teacher_text, cache_entry, device):
    audio_inputs = processor(
        audio=sample["wav"],
        sampling_rate=16000,
        return_tensors="pt",
    )
    audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}

    target_text = cache_entry["teacher_text_str"] if use_teacher_text else sample["ref"]
    labels = build_target_labels(processor, [target_text], sample["tgt_lang"], device)

    outputs = model_student(
        **audio_inputs,
        labels=labels,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return outputs.loss
```

## 10. Exact helper for the T2U path

This helper must mirror the source file.
Use it for both teacher and student in Phase 6C.

```python
def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    batch_size, mask_seq_len = hidden_states.shape[:2]
    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)
    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)
    mask = hidden_states.new_ones((batch_size, mask_seq_len))
    mask = mask.masked_fill(bool_mask, 0)
    return mask

def build_t2u_conditioning_from_sequences(model, input_features, attention_mask, text_sequences):
    # 1) speech encoder
    enc = model.speech_encoder(
        input_features=input_features,
        attention_mask=attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ).last_hidden_state

    # 2) subsampled encoder mask, same logic as the model file
    encoder_attention_mask = None
    if attention_mask is not None:
        sub_lengths = model._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(enc.device)
        encoder_attention_mask = _compute_new_attention_mask(enc, sub_lengths)

    # 3) forced text-decoder hidden states
    t2u_input_embeds = model.text_decoder(
        input_ids=text_sequences[:, :-1],
        encoder_hidden_states=enc,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ).last_hidden_state

    # 4) build char inputs exactly like generate()
    pad_token_id = model.generation_config.pad_token_id
    eos_token_id = model.generation_config.eos_token_id

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

    # 5) T2U attention mask, same logic as generate()
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
```

## 11. Phase 6C: composite recovery with T2U KD

This is the core recovery stage.

Do **not** train T2U in isolation at first.
Train it in a composite loss with the upstream path still active.

### What stays trainable in Phase 6C

- speech encoder DoRA adapters
- text decoder DoRA adapters
- T2U encoder DoRA adapters
- T2U decoder DoRA adapters
- `t2u_model.model.decoder.pos_emb_alpha_char`
- `t2u_model.model.decoder.pos_emb_alpha`

Everything else stays frozen.

### Why this works better than T2U-only hard-label training

1. T2U gets the correct conditioning path
2. text decoder hidden states can keep adapting
3. speech encoder can still move slightly if needed
4. we avoid pretending that the top-level HF forward exposes a T2U loss

### Recommended losses

Use the same cached `teacher_text_sequences` for both teacher and student conditioning.

Compute:

- `L_text`: normal text CE on teacher-or-gold labels
- `L_t2u_kl`: KL between student and teacher T2U logits
- `L_t2u_ce`: hard CE to teacher argmax unit ids on the overlapping time region
- `L_len`: SmoothL1 on valid output lengths from `padding_mask.sum(1)`

### Important alignment rule

Teacher and student T2U lengths may differ.
Never assume equal lengths.

Align by overlap:

```python
def t2u_overlap_losses(student_out, teacher_out, temperature=2.0):
    student_logits = student_out.last_hidden_state
    teacher_logits = teacher_out.last_hidden_state

    student_mask = student_out.padding_mask.bool()
    teacher_mask = teacher_out.padding_mask.bool()

    student_len = student_mask.sum(1).long()
    teacher_len = teacher_mask.sum(1).long()
    common_len = torch.minimum(student_len, teacher_len)

    total_kl = student_logits.new_zeros(())
    total_ce = student_logits.new_zeros(())
    valid = 0

    for b in range(student_logits.size(0)):
        L = int(common_len[b].item())
        if L < 2:
            continue

        s = student_logits[b, :L]
        t = teacher_logits[b, :L]

        total_kl = total_kl + F.kl_div(
            F.log_softmax(s / temperature, dim=-1),
            F.softmax(t / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        teacher_hard = t.argmax(dim=-1)
        total_ce = total_ce + F.cross_entropy(s, teacher_hard)
        valid += 1

    if valid == 0:
        raise RuntimeError("No valid T2U overlap found in batch")

    total_kl = total_kl / valid
    total_ce = total_ce / valid
    total_len = F.smooth_l1_loss(student_len.float(), teacher_len.float())

    return total_kl, total_ce, total_len
```

### Composite loss

Start with:

```python
loss = (
    1.00 * text_loss +
    0.35 * t2u_kl +
    0.20 * t2u_ce +
    0.05 * t2u_len
)
```

Then watch the scales:

- if `t2u_kl` is >5x larger than `text_loss` for 100+ steps, reduce it
- if `t2u_len` does not move at all, increase it slightly to `0.10`
- if text metrics regress, increase `text_loss` weight or reduce T2U LR

### Phase 6C optimizer groups

Recommended starting LRs:

- speech encoder DoRA: `3e-5`
- text decoder DoRA: `5e-5`
- T2U encoder DoRA: `8e-5`
- T2U decoder DoRA + native scalar params: `1e-4`

Use:

- AdamW
- weight decay `0.01`
- cosine schedule
- warmup `10%`
- grad clip `1.0`

### Phase 6C training sketch

```python
def phase6c_step(sample, cache_entry):
    # audio on student GPU
    audio_inputs_student = processor(
        audio=sample["wav"],
        sampling_rate=16000,
        return_tensors="pt",
    )
    audio_inputs_student = {k: v.to(student_device) for k, v in audio_inputs_student.items()}

    # text CE target
    use_teacher_text = (random.random() < 0.5)
    target_text = cache_entry["teacher_text_str"] if use_teacher_text else sample["ref"]
    labels = build_target_labels(processor, [target_text], sample["tgt_lang"], student_device)

    text_out = model_student(
        **audio_inputs_student,
        labels=labels,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    text_loss = text_out.loss

    # same teacher-generated token sequence on both sides
    teacher_text_sequences = cache_entry["teacher_text_sequences"].unsqueeze(0)

    # teacher side on GPU1
    audio_inputs_teacher = {k: v.to(teacher_device) for k, v in audio_inputs_student.items()}
    teacher_text_sequences_gpu = teacher_text_sequences.to(teacher_device)
    with torch.no_grad():
        teacher_cond = build_t2u_conditioning_from_sequences(
            model_teacher,
            input_features=audio_inputs_teacher["input_features"],
            attention_mask=audio_inputs_teacher.get("attention_mask"),
            text_sequences=teacher_text_sequences_gpu,
        )
        teacher_t2u = model_teacher.t2u_model(
            inputs_embeds=teacher_cond["t2u_input_embeds"],
            attention_mask=teacher_cond["t2u_attention_mask"],
            char_input_ids=teacher_cond["t2u_char_input_ids"],
            char_count_per_id=teacher_cond["t2u_char_count_per_id"],
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

    # student side on GPU0
    student_text_sequences_gpu = teacher_text_sequences.to(student_device)
    student_cond = build_t2u_conditioning_from_sequences(
        model_student,
        input_features=audio_inputs_student["input_features"],
        attention_mask=audio_inputs_student.get("attention_mask"),
        text_sequences=student_text_sequences_gpu,
    )
    student_t2u = model_student.t2u_model(
        inputs_embeds=student_cond["t2u_input_embeds"],
        attention_mask=student_cond["t2u_attention_mask"],
        char_input_ids=student_cond["t2u_char_input_ids"],
        char_count_per_id=student_cond["t2u_char_count_per_id"],
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    # bring teacher outputs to GPU0 for the KD loss
    teacher_t2u.last_hidden_state = teacher_t2u.last_hidden_state.to(student_device)
    teacher_t2u.padding_mask = teacher_t2u.padding_mask.to(student_device)

    t2u_kl, t2u_ce, t2u_len = t2u_overlap_losses(student_t2u, teacher_t2u)
    loss = 1.00 * text_loss + 0.35 * t2u_kl + 0.20 * t2u_ce + 0.05 * t2u_len
    return loss, {
        "text": text_loss.item(),
        "t2u_kl": t2u_kl.item(),
        "t2u_ce": t2u_ce.item(),
        "t2u_len": t2u_len.item(),
    }
```

## 12. Phase 6D: optional short polish

Only run this if:

- Phase 6B improved text metrics
- Phase 6C improved ASR-BLEU / ASR-ChrF
- the model is stable

Config:

- 200 to 300 steps
- same composite loss as Phase 6C
- reduce all LRs by about `3x`

If Phase 6C is still unstable, skip Phase 6D.

## 13. Kaggle memory settings

Recommended safe defaults:

- micro-batch: `1`
- grad accumulation: `8`
- audio length cap for training: start with `<= 18s`
- after stable training, raise to `<= 22s` or `<= 25s`
- eval can still use longer clips

Additional rules:

1. Do not call `torch.cuda.empty_cache()` every optimizer step
2. Only clear cache:
   - after checkpoint save
   - after quick eval
   - after OOM recovery
3. Bucket batches by audio duration so padding is not wasting memory
4. Keep `output_attentions=False` and `output_hidden_states=False` unless needed

## 14. Merge/save plan

At the very end, merge DoRA adapters back into the base weights.

If you wrapped submodules separately:

```python
model_student.speech_encoder = model_student.speech_encoder.merge_and_unload()
model_student.text_decoder = model_student.text_decoder.merge_and_unload()
model_student.t2u_model.model.encoder = model_student.t2u_model.model.encoder.merge_and_unload()
model_student.t2u_model.model.decoder = model_student.t2u_model.model.decoder.merge_and_unload()
```

Then:

```python
model_student.eval()
sync_model_config(model_student)
save_model_to_drive(model_student, processor, "phase6_dora_merged")
```

Use a new stage name.
Do not overwrite the current broken Phase 6 artifact.

## 15. What I would actually implement in `pragmata-recovery.ipynb`

If this were my Kaggle run order, I would do exactly this:

1. Replace current Phase 6 cells completely
2. Add strict target-module verification before any DoRA injection
3. Cache teacher text + unit outputs once on GPU1
4. Run Phase 6B text recovery
5. Add T2U DoRA only after Phase 6B is stable
6. Run Phase 6C composite text + T2U KD
7. Benchmark every 200 steps on a small dev slice
8. Merge and save only the best checkpoint by ASR-ChrF / ASR-BLEU

## 16. Final recommendation

For your exact saved HF file and Kaggle setup, the best notebook-safe plan is:

- DoRA for the text path
- custom composite KD for T2U
- teacher cache offline
- teacher on GPU1, student on GPU0
- no silent fallbacks
- no fake `t2u_loss`
- no guessed module names

If you later want the absolute maximum T2U recovery beyond this notebook-safe plan, the next step is not "more LoRA/DoRA".
The next step is a source patch that exposes duration supervision explicitly in the T2U decoder path.

## References

- `AAA/modeling_seamless_m4t_v2.py`
- `AAA/pragmata-recovery.ipynb`
- Hugging Face PEFT LoRA/DoRA docs
- Hugging Face SeamlessM4T tokenizer docs
