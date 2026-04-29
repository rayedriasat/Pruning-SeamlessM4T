# Phase 6 Recovery Plan for `phase5_dec_14L`: LoRA-First + Native T2U Recovery

This plan is written against the actual module structure in `AAA/modeling_seamless_m4t_v2.py`.
It replaces the current DoRA-first idea with the most reliable recovery path for this architecture and Kaggle's 2xT4 limit.

Short answer:

- DoRA is **not** a bad method in general.
- But for **this** model and **this** bottleneck, DoRA is **not** the best primary strategy.
- The best fail-proof strategy here is:
  - **LoRA** on the standard linear-heavy text path (`speech_encoder + text_decoder`)
  - **native full fine-tuning of `t2u_model`** for audio recovery
  - **teacher caching**
  - **strict 2-GPU separation**
  - **duration-aware T2U KD**

That is the plan below.

## 1. Hard facts from the saved model file

These are the facts we must code around:

1. `SeamlessM4Tv2ForSpeechToSpeech.forward()` does **not** train `t2u_model`.
   - In the saved file, the top-level `forward()` computes the text loss from:
     - `speech_encoder -> text_decoder -> lm_head`
   - Therefore `outputs.t2u_loss` does not exist in the standard HF forward path.

2. T2U is explicitly treated as **non auto-regressive** in generation.
   - The source file says the text-to-unit model is non auto-regressive.
   - Therefore T2U should not be trained like a normal autoregressive decoder.

3. The current notebook targets wrong module names.
   - Wrong: `text_decoder.layers.*.encoder_attn.*`
   - Correct in this file: `text_decoder.layers.*.cross_attention.*`
   - Wrong: `t2u_model.model.decoder.layers.*.fc1/fc2`
   - Correct: the T2U decoder layer has:
     - `self_attn`
     - `conv1`
     - `conv2`
     - plus a decoder-level `duration_predictor`

4. The T2U decoder predicts output length before unit CE is computed.
   - It upsamples using `char_count_per_id`
   - predicts durations
   - upsamples again
   - then produces unit logits
   - So naive `labels=teacher_unit_ids` is unstable if student length and teacher length differ.

5. There is a source bug if `output_attentions=True` in the T2U decoder path.
   - `SeamlessM4Tv2TextToUnitDecoderLayer.forward()` returns `(hidden_states, self_attn_weights)`
   - `SeamlessM4Tv2TextToUnitDecoder.forward()` tries to read `layer_outputs[2]`
   - So keep `output_attentions=False` for T2U training unless you patch the source.

6. Kaggle T4 should be treated as an FP16 training target.
   - Do not build the plan around BF16.

## 2. Why the current Phase 6 notebook fails

The current Phase 6 cells are structurally wrong:

1. They feed `speech_encoder(...).last_hidden_state` directly into `t2u_model(...)`.
   - That is not the real T2U path in this model.
   - The real path is:
     - `speech_encoder`
     - `text_decoder` forced on text tokens
     - text-decoder hidden states become `t2u_model.inputs_embeds`
     - `char_input_ids` and `char_count_per_id` must also be built

2. They expect `outputs.t2u_loss` from the top-level S2S forward.
   - That loss is not produced by the source file.

3. They use adapter target names that do not match the saved file.

4. They use `bfloat16` autocast on T4.

## 3. Should we use DoRA or LoRA?

### Decision

Use **LoRA instead of DoRA** as the primary recovery method.

### Why

This is the architecture-aware reason:

1. DoRA is best when the important trainable surface is mostly supported linear layers.
2. Your hardest recovery problem is **T2U**, not just the text path.
3. The most important T2U-specific modules are not mainly the easy DoRA targets:
   - `conv1`
   - `conv2`
   - `duration_predictor.conv1`
   - `duration_predictor.conv2`
   - `duration_predictor.proj`
   - `pos_emb_alpha_char`
   - `pos_emb_alpha`
4. PEFT docs say DoRA supports embedding, linear, and Conv2d layers, and adds more overhead than plain LoRA.
5. Your T2U decoder uses **Conv1d**, not Conv2d.
6. Therefore a DoRA-first plan leaves the most T2U-specific recovery pieces outside its best coverage zone, while also spending more VRAM and runtime.

### Practical conclusion

For this model:

- **LoRA** is the better adapter method for the speech/text path.
- **Native full-parameter T2U tuning** is the better recovery method for the audio path.

### What DoRA is still good for

DoRA is still a reasonable **optional follow-up experiment** on the text path only, after a working LoRA pipeline exists.

It is just not the best **first** or **main** recovery strategy here.

## 4. Final recommended strategy

Use this 4-part recovery path:

1. **Phase 6A - Offline teacher cache**
   - Cache teacher text sequences
   - Cache teacher decoded text
   - Cache teacher unit sequences
   - Cache audio duration and sanity metadata

2. **Phase 6B - Text-path recovery with LoRA**
   - Train `text_decoder` first
   - Then train `speech_encoder + text_decoder`
   - Goal: restore S2TT quality and stabilize the hidden states that feed T2U

3. **Phase 6C - Native T2U recovery**
   - Freeze speech encoder and text path
   - Train the **entire `t2u_model` natively**
   - Use the correct forced text-decoder hidden-state path
   - Use teacher T2U KD and length supervision

4. **Phase 6D - Optional short joint polish**
   - Unfreeze only:
     - `text_decoder` LoRA
     - full `t2u_model`
   - Keep `speech_encoder` frozen
   - Very low LR

This is the best mix of:

- maximum recovery
- architectural correctness
- low OOM risk
- notebook implementation simplicity

## 5. What we are not doing

We are **not** using these as the main plan:

1. Pure DoRA everywhere
2. Pure end-to-end full-model KD from the start
3. T2U adapter-only recovery
4. A single giant monolithic stage

Reasons:

- too much VRAM pressure
- bad fit for Conv1d-heavy T2U recovery
- harder to debug
- too many failure points at once

## 6. Exact LoRA target modules from the saved file

Use explicit, source-validated targets.
Do not guess names and do not rely on broad wildcard strings without validation.

### 6.1 Strict dotted-path resolver

```python
import torch
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
        raise TypeError(f"Non-linear LoRA targets found:\n{msg}")
```

### 6.2 Speech encoder LoRA targets

From `AAA/modeling_seamless_m4t_v2.py`, the speech encoder is:

- `feature_projection.projection`
- `intermediate_ffn.intermediate_dense`
- `intermediate_ffn.output_dense`
- conformer layers with:
  - `self_attn.linear_q/k/v/out`
  - `ffn1.intermediate_dense/output_dense`
  - `ffn2.intermediate_dense/output_dense`

```python
def build_speech_encoder_lora_targets(model_student):
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

- `conv_module.*`
- adapter Conv1d blocks

### 6.3 Text decoder LoRA targets

From the saved file:

- `self_attn.q_proj/k_proj/v_proj/out_proj`
- `cross_attention.q_proj/k_proj/v_proj/out_proj`
- `ffn.fc1/fc2`

```python
def build_text_decoder_lora_targets(model_student):
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

Do **not** use `encoder_attn.*`.
This file uses `cross_attention.*`.

## 7. LoRA config that matches the use case

Use LoRA with rsLoRA.
Do not start at tiny rank if recovery is the goal and VRAM allows more.

Recommended starting configs:

```python
from peft import LoraConfig

def make_lora_config(target_modules, r, alpha, dropout=0.05):
    return LoraConfig(
        target_modules=target_modules,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        use_rslora=True,
    )

speech_lora_cfg = make_lora_config(
    build_speech_encoder_lora_targets(model_student),
    r=16,
    alpha=32,
)

text_lora_cfg = make_lora_config(
    build_text_decoder_lora_targets(model_student),
    r=32,
    alpha=64,
)
```

Why this split:

- speech encoder is large and expensive, so keep it lower-rank
- text decoder is the most important linear-heavy recovery path, so give it higher rank

If stable after 100-150 steps:

- you may raise speech rank to `24`
- keep text rank `32`

Do **not** start at:

- text rank `64`
- speech rank `32+`

on the first Kaggle run.

## 8. T2U strategy: full native fine-tuning

This is the key change.

Do **not** adapterize T2U as the main plan.
Train `model_student.t2u_model` directly.

### Why full native T2U is better here

Because the T2U recovery path depends on:

- Conv1d decoder blocks
- duration predictor
- output-length behavior
- decoder scalar position parameters

Those are exactly the parts you do not want to miss.

### Primary T2U trainable surface

Train the **entire** `t2u_model`:

- `t2u_model.model.encoder.*`
- `t2u_model.model.decoder.*`
- `t2u_model.lm_head`

### Freeze rule for Phase 6C

During T2U recovery:

- freeze `speech_encoder`
- freeze `text_decoder`
- freeze `lm_head` of the top-level S2S model
- train only `t2u_model`

That keeps memory under control and makes the T2U stage easier to stabilize.

### Fallback if T2U full tuning still OOMs

If full `t2u_model` fine-tuning still OOMs after:

- micro-batch `1`
- grad accumulation `8`
- short-audio cap
- gradient checkpointing

then fall back to this exact subset:

```python
def mark_t2u_selective_trainable(model_student):
    for p in model_student.parameters():
        p.requires_grad_(False)

    t2u = model_student.t2u_model

    for name, p in t2u.named_parameters():
        if (
            name.startswith("model.decoder.layers.") or
            name.startswith("model.decoder.duration_predictor.") or
            name in {
                "model.decoder.pos_emb_alpha_char",
                "model.decoder.pos_emb_alpha",
                "lm_head.weight",
            }
        ):
            p.requires_grad_(True)
```

That fallback still covers the most T2U-specific recovery parts.

## 9. Kaggle 2-GPU memory plan

Use both GPUs deliberately.
Do not let both heavy models drift onto `cuda:0`.

### Placement

```python
student_device = torch.device("cuda:0")
teacher_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
```

### Device policy

1. Student always on `cuda:0`
2. Teacher always on `cuda:1`
3. Teacher cache stored on CPU RAM or disk
4. MMS / Whisper / extra ASR models must be unloaded before finetuning

### Memory budget target

Aim for:

- GPU0 steady-state training memory: `<= 13.5 GB`
- GPU1 teacher memory: `<= 13.5 GB`

That leaves headroom for:

- fragmentation
- peak activation spikes
- checkpointing overhead

### Mixed precision

Use:

```python
autocast_dtype = torch.float16
```

Do not use BF16 on T4.

### Gradient checkpointing

```python
model_student.config.use_cache = False
model_student.text_decoder.config.use_cache = False
model_student.gradient_checkpointing_enable()
model_student.speech_encoder.gradient_checkpointing_enable()
model_student.text_decoder.gradient_checkpointing_enable()
model_student.t2u_model.model.encoder.gradient_checkpointing_enable()
model_student.t2u_model.model.decoder.gradient_checkpointing_enable()
```

### CPU and disk usage

Use CPU RAM and disk for:

- teacher cache
- sample metadata
- checkpoint shards
- any large dev metrics history

Do **not** keep:

- raw teacher outputs
- large eval objects
- old adapters

in GPU memory between stages.

## 10. Phase 6A: offline teacher cache

This is mandatory.
It is the biggest win for both stability and speed.

### Cache contents

For each training sample, store:

- `id`
- `src_lang`
- `tgt_lang`
- `teacher_text_sequences`
- `teacher_text_str`
- `teacher_unit_sequences`
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

### Cache sanity checks

Reject or skip entries with:

- empty teacher text
- `teacher_unit_sequences` length `< 3`
- obviously broken unit lengths, for example `> 1024`

### Cache storage recommendation

Store cache in shards, not one giant object:

```python
cache_shards/
  train_cache_000.pt
  train_cache_001.pt
  ...
  dev_cache_000.pt
```

Use sample-id lookup dictionaries in CPU memory.

### Unload teacher after cache if needed

After Phase 6A, if Stage 6B uses only cached text labels:

```python
del model_teacher
gc.collect()
torch.cuda.empty_cache()
```

Reload the teacher only when Phase 6C begins.

## 11. Text labels: correct tokenizer usage

Do not guess special-token layout.
The HF tokenizer docs say the target tokenization format is handled through `text_target=...` and `tgt_lang=...`.

Use:

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

## 12. Phase 6B: text-path recovery with LoRA

Goal:

- restore S2TT quality first
- stabilize hidden states before T2U training

### Phase 6B.1: text decoder warmup

Train:

- `text_decoder` LoRA only

Settings:

- micro-batch: `1`
- grad accumulation: `8`
- max audio length: `20s`
- steps: `300`
- LR:
  - text decoder LoRA: `1e-4`

Labels:

- 50% teacher text
- 50% gold reference text

### Phase 6B.2: speech + text LoRA

Train:

- `speech_encoder` LoRA
- `text_decoder` LoRA

Settings:

- micro-batch: `1`
- grad accumulation: `8`
- max audio length: `20s`
- steps: `800-1200`
- LR:
  - speech LoRA: `4e-5`
  - text LoRA: `8e-5`

### Stage 6B training step sketch

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

### Promotion rule to Phase 6C

Do not move to T2U recovery unless:

- text loss is clearly falling
- quick dev ASR-ChrF or S2TT proxy improves
- no recurrent OOMs in the last 150 steps

## 13. Correct T2U conditioning helper

This helper must mirror the source file.
This is the exact architectural bridge the broken notebook was missing.

```python
def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    batch_size, mask_seq_len = hidden_states.shape[:2]
    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)
    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)
    mask = hidden_states.new_ones((batch_size, mask_seq_len))
    mask = mask.masked_fill(bool_mask, 0)
    return mask

def build_t2u_conditioning_from_sequences(model, input_features, attention_mask, text_sequences):
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
        encoder_attention_mask = _compute_new_attention_mask(enc, sub_lengths)

    t2u_input_embeds = model.text_decoder(
        input_ids=text_sequences[:, :-1],
        encoder_hidden_states=enc,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ).last_hidden_state

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

## 14. Phase 6C: native T2U recovery

This is the core audio-recovery stage.

### Freeze policy

For Phase 6C:

- `speech_encoder`: frozen
- `text_decoder`: frozen
- top-level `lm_head`: frozen
- `t2u_model`: trainable

This is deliberate.
It lowers memory and isolates the actual broken path.

### Teacher setup for Phase 6C

Reload teacher on `cuda:1`.
Do **not** recompute full `generate()` each step.

Use cached `teacher_text_sequences` from Phase 6A and run only the teacher forward needed for T2U KD.

### Losses

Use:

1. `L_t2u_soft`
   - KL divergence between teacher and student T2U logits

2. `L_t2u_hard`
   - hard CE against teacher argmax units on overlapping time positions

3. `L_len`
   - length regression between teacher and student padding-mask lengths

### Overlap-based alignment

Teacher and student T2U lengths will not always match.
Never assume equal lengths.

```python
import torch.nn.functional as F

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

### T2U loss mix

Start with:

```python
loss = (
    0.60 * t2u_soft +
    0.30 * t2u_hard +
    0.10 * t2u_len
)
```

This stage does **not** need text CE because the text path is frozen on purpose.

### T2U optimizer groups

Use separate LR groups because duration is sensitive:

```python
def make_t2u_param_groups(model_student):
    enc_params = []
    dec_params = []
    dur_params = []
    scalar_params = []
    head_params = []

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

    return [
        {"params": enc_params, "lr": 8e-5, "weight_decay": 0.01},
        {"params": dec_params, "lr": 8e-5, "weight_decay": 0.01},
        {"params": dur_params, "lr": 1e-4, "weight_decay": 0.00},
        {"params": scalar_params, "lr": 1e-4, "weight_decay": 0.00},
        {"params": head_params, "lr": 8e-5, "weight_decay": 0.01},
    ]
```

### Phase 6C runtime settings

Start conservatively:

- micro-batch: `1`
- grad accumulation: `8`
- max audio length: `12s`
- steps: `600`

If stable after 150-200 steps:

- raise max audio length to `16s`
- continue to `1000-1200` steps if metrics improve

### Phase 6C step sketch

```python
def phase6c_t2u_step(sample, cache_entry):
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
        teacher_t2u = model_teacher.t2u_model(
            inputs_embeds=teacher_cond["t2u_input_embeds"],
            attention_mask=teacher_cond["t2u_attention_mask"],
            char_input_ids=teacher_cond["t2u_char_input_ids"],
            char_count_per_id=teacher_cond["t2u_char_count_per_id"],
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

    # Student path on GPU0
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

    teacher_t2u.last_hidden_state = teacher_t2u.last_hidden_state.to(student_device)
    teacher_t2u.padding_mask = teacher_t2u.padding_mask.to(student_device)

    t2u_soft, t2u_hard, t2u_len = t2u_overlap_losses(student_t2u, teacher_t2u)
    loss = 0.60 * t2u_soft + 0.30 * t2u_hard + 0.10 * t2u_len

    return loss, {
        "t2u_soft": t2u_soft.item(),
        "t2u_hard": t2u_hard.item(),
        "t2u_len": t2u_len.item(),
    }
```

## 15. Phase 6D: optional joint polish

Only do this if:

- Phase 6B improved text quality
- Phase 6C improved ASR-BLEU / ASR-ChrF
- the run is stable

### Unfreeze set

Unfreeze only:

- `text_decoder` LoRA
- full `t2u_model`

Keep:

- `speech_encoder` frozen

### Why this stage exists

It gives T2U one chance to co-adapt with the text decoder hidden states without reopening the whole model.

### Settings

- micro-batch: `1`
- grad accumulation: `8`
- max audio length: `14s`
- steps: `200-300`
- LR:
  - text decoder LoRA: `1e-5`
  - full `t2u_model`: `4e-5`

### Loss

Use:

```python
loss = (
    0.35 * text_loss +
    0.40 * t2u_soft +
    0.20 * t2u_hard +
    0.05 * t2u_len
)
```

Do not reopen `speech_encoder` in this stage unless you already have a stable working pipeline and spare VRAM.

## 16. Memory safety rules

These are mandatory for Kaggle stability:

1. Use audio-length bucketing
2. Start with short clips
3. Use micro-batch `1`
4. Use grad accumulation instead of bigger batch size
5. Do not keep teacher and student on the same GPU
6. Do not keep ASR benchmark models in memory during finetuning
7. Do not call `torch.cuda.empty_cache()` every step
8. Save checkpoints in shards
9. Unload teacher after Phase 6A and reload only for Phase 6C

### Safe training defaults

```python
MICRO_BATCH = 1
GRAD_ACCUM = 8
MAX_AUDIO_SEC_B = 20
MAX_AUDIO_SEC_C = 12
MAX_AUDIO_SEC_D = 14
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.10
```

### OOM fallback ladder

If OOM happens:

1. reduce max audio length
2. reduce eval frequency
3. unload teacher between eval windows
4. fall back from full `t2u_model` tuning to selective T2U subset
5. only then reduce LoRA rank

Do not immediately slash rank first.

## 17. Merge and save plan

At the end of the full run:

```python
model_student.speech_encoder = model_student.speech_encoder.merge_and_unload()
model_student.text_decoder = model_student.text_decoder.merge_and_unload()
```

There is no merge for native T2U tuning because those are real updated base weights.

Then:

```python
model_student.eval()
sync_model_config(model_student)
save_model_to_drive(model_student, processor, "phase6_lora_t2u_merged")
```

Use a new name.
Do not overwrite the current broken Phase 6 artifact.

## 18. What to remove from `pragmata-recovery.ipynb`

Delete or replace the current Phase 6 cells that do these things:

1. DoRA/LoRA target strings using nonexistent names
2. direct `speech_encoder -> t2u_model(labels=...)`
3. any expectation of `outputs.t2u_loss`
4. BF16 autocast on T4
5. one-stage joint training from step 1

## 19. Final recommendation

For your exact source file, your exact pruned model, and Kaggle 2xT4:

### Best primary strategy

1. **LoRA** on `text_decoder`
2. then **LoRA** on `speech_encoder + text_decoder`
3. then **full native T2U fine-tuning**
4. then optional short **text_decoder LoRA + T2U** polish

### What I would not use as the main plan

- DoRA-first
- adapter-only T2U
- full-model KD from step 1

### One-line decision

If you want the most fail-proof recovery strategy for this use case:

**Use LoRA instead of DoRA, and recover T2U with native full fine-tuning plus teacher KD.**

## References

- `AAA/modeling_seamless_m4t_v2.py`
- `AAA/pragmata-recovery.ipynb`
- Hugging Face PEFT LoRA / DoRA docs
- Hugging Face SeamlessM4T tokenizer docs
- NVIDIA T4 precision docs
