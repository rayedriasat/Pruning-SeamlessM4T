import gc
import glob
import math
import os
import random
import time
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

PHASE6_MODEL_NAME = 'phase6_lora_t2u_merged'
PHASE6_BENCHMARK_NAME = 'phase6_lora_t2u_benchmark'
PHASE6_CACHE_PREFIX = 'phase6_teacher_cache'

MICRO_BATCH    = 1
GRAD_ACCUM     = 8
MAX_AUDIO_SEC_B1 = 20
MAX_AUDIO_SEC_B2 = 20
MAX_AUDIO_SEC_C  = 12
MAX_AUDIO_SEC_D  = 14
MAX_GRAD_NORM    = 1.0
WARMUP_RATIO     = 0.10

# ── Logging / eval / checkpoint cadence ───────────────────────────────────────
# LOG_EVERY   : print a loss line every N optimizer steps (keep this low — you want feedback)
# EVAL_EVERY  : run phase6_quick_eval every N optimizer steps
# SAVE_EVERY  : save a checkpoint every N optimizer steps
# These are absolute step counts. Eval and Save are also clamped so they never
# fire more often than LOG_EVERY regardless of what you set.
LOG_EVERY  = 10    # print every 25 opt steps  → ~200 fwd passes between prints
EVAL_EVERY = 50   # quick eval every 100 steps → enough to track quality
SAVE_EVERY = 50   # checkpoint every 200 steps → you lose at most 200 steps on crash


# ── All STEPS values are OPTIMIZER STEPS (gradient updates), NOT micro-steps.
# ── Micro-steps per run = STEPS × GRAD_ACCUM  (e.g. 1200 × 8 = 9600 fwd passes)
# ── Warmup covers the first WARMUP_RATIO × STEPS optimizer steps as intended.
# ── To extend a run, increase the value here and re-run — auto-resume handles the rest.
STAGE6B1_STEPS   = 400    # ~3,200 fwd passes  — LoRA warmup, converges fast
STAGE6B2_STEPS   = 900    # ~7,200 fwd passes  — joint LoRA recovery
STAGE6C_STEPS    = 700    # ~5,600 fwd passes  — T2U distillation
STAGE6D_STEPS    = 350    # ~2,800 fwd passes  — polish, keep short
STAGE6D_ENABLED  = True

PHASE6_TEXT_KD_PROB   = 0.4
PHASE6_T2U_TRAIN_MODE = 'full'
autocast_dtype        = torch.float16

student_device = torch.device('cuda:0')
teacher_device = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')

phase6_logs = {'6b1': [], '6b2': [], '6c': [], '6d': []}
phase6_eval_history = []
phase6_cache_manifest = {}
phase6_cache_index = {}
phase6_cache_keys_by_pair = defaultdict(list)
phase6_sample_lookup = {}
phase6_shard_cache = OrderedDict()

PHASE6_CACHE_SHARD_SIZE = 2048
PHASE6_CACHE_SYNC_PARTS = 4
PHASE6_SHARD_LRU_LIMIT = 16
PHASE6_MAX_TEACHER_UNIT_TOKENS = 2048
PHASE6_MIN_TEACHER_UNIT_TOKENS = 3

phase6_cache_stats = {
    'cached_ok': 0,
    'skipped_empty_text': 0,
    'skipped_short_units': 0,
    'skipped_long_units': 0,
    'skipped_errors': 0,
}


class _NoopVocoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get('input_ids', args[0] if args else None)
        if input_ids is None:
            raise RuntimeError('Noop vocoder expected input_ids.')
        batch = input_ids.shape[0]
        waveform = torch.zeros(batch, 1, device=self.device)
        lengths = torch.ones(batch, dtype=torch.int32, device=self.device)
        return waveform, lengths


def safe_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_model_to_device(mdl, device):
    if not torch.cuda.is_available():
        return mdl
    try:
        from accelerate.hooks import remove_hook_from_submodules
        remove_hook_from_submodules(mdl)
    except Exception:
        pass
    return mdl.to(device)


def maybe_enable_gradient_checkpointing(module, label):
    fn = getattr(module, 'gradient_checkpointing_enable', None)
    if callable(fn):
        fn()
        print(f'  gradient checkpointing enabled: {label}')


def disable_generation_cache(mdl):
    if hasattr(mdl, 'config'):
        mdl.config.use_cache = False
    if hasattr(mdl, 'text_decoder') and hasattr(mdl.text_decoder, 'config'):
        mdl.text_decoder.config.use_cache = False


def phase6_sample_key(sample_or_entry):
    sample_id = sample_or_entry.get('sample_id', sample_or_entry.get('id'))
    return f"{sample_or_entry['src_lang']}__{sample_or_entry['tgt_lang']}__{sample_id}"


def build_sample_lookup(streaming_ds):
    lookup = {}
    for outer_idx, (ds_idx, sample_idx) in enumerate(streaming_ds.index):
        meta = streaming_ds.datasets[ds_idx].samples[sample_idx]
        key = f"{meta['src_lang']}__{meta['tgt_lang']}__{meta['id']}"
        if key in lookup:
            raise RuntimeError(f'Duplicate Phase 6 sample key found: {key}')
        lookup[key] = outer_idx
    return lookup


def get_submodule_strict(root, dotted_name: str):
    cur = root
    for part in dotted_name.split('.'):
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
        msg = '\n'.join([f'{name} -> {kind}' for name, kind in bad])
        raise TypeError(f'Non-linear LoRA targets found:\n{msg}')


def build_speech_encoder_lora_targets(model_student):
    n = len(model_student.speech_encoder.encoder.layers)
    targets = [
        'feature_projection.projection',
        'intermediate_ffn.intermediate_dense',
        'intermediate_ffn.output_dense',
    ]
    for i in range(n):
        prefix = f'encoder.layers.{i}'
        targets += [
            f'{prefix}.self_attn.linear_q',
            f'{prefix}.self_attn.linear_k',
            f'{prefix}.self_attn.linear_v',
            f'{prefix}.self_attn.linear_out',
            f'{prefix}.ffn1.intermediate_dense',
            f'{prefix}.ffn1.output_dense',
            f'{prefix}.ffn2.intermediate_dense',
            f'{prefix}.ffn2.output_dense',
        ]
    assert_linear_targets_exist(model_student.speech_encoder, targets)
    return targets


def build_text_decoder_lora_targets(model_student):
    n = len(model_student.text_decoder.layers)
    targets = []
    for i in range(n):
        prefix = f'layers.{i}'
        targets += [
            f'{prefix}.self_attn.q_proj',
            f'{prefix}.self_attn.k_proj',
            f'{prefix}.self_attn.v_proj',
            f'{prefix}.self_attn.out_proj',
            f'{prefix}.cross_attention.q_proj',
            f'{prefix}.cross_attention.k_proj',
            f'{prefix}.cross_attention.v_proj',
            f'{prefix}.cross_attention.out_proj',
            f'{prefix}.ffn.fc1',
            f'{prefix}.ffn.fc2',
        ]
    assert_linear_targets_exist(model_student.text_decoder, targets)
    return targets


def make_lora_config(target_modules, r, alpha, dropout=0.05):
    return LoraConfig(
        target_modules=target_modules,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias='none',
        use_rslora=True,
    )


def wrap_with_lora_if_needed(module, config, label):
    if hasattr(module, 'peft_config'):
        print(f'{label} already has LoRA adapters attached.')
        return module
    wrapped = get_peft_model(module, config)
    print(f'{label} LoRA attached.')
    wrapped.print_trainable_parameters()
    return wrapped


def freeze_all_student():
    for p in model_student.parameters():
        p.requires_grad_(False)


def enable_lora_params(module, label):
    found = 0
    for name, p in module.named_parameters():
        if 'lora_' in name:
            p.requires_grad_(True)
            found += p.numel()
    if found == 0:
        raise RuntimeError(f'No LoRA parameters found in {label}.')
    print(f'  enabled LoRA params for {label}: {found/1e6:.2f}M')


def mark_t2u_trainable_full():
    for p in model_student.parameters():
        p.requires_grad_(False)
    for p in model_student.t2u_model.parameters():
        p.requires_grad_(True)
    print('  T2U mode: full native training')


def mark_t2u_selective_trainable():
    for p in model_student.parameters():
        p.requires_grad_(False)
    t2u = model_student.t2u_model
    for name, p in t2u.named_parameters():
        if (
            name.startswith('model.decoder.layers.')
            or name.startswith('model.decoder.duration_predictor.')
            or name in {
                'model.decoder.pos_emb_alpha_char',
                'model.decoder.pos_emb_alpha',
                'lm_head.weight',
            }
        ):
            p.requires_grad_(True)
    print('  T2U mode: selective decoder + duration predictor')


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6


def trainable_named_params(module):
    return [p for p in module.parameters() if p.requires_grad]


def make_cosine_scheduler(optimizer, total_steps, warmup_ratio=WARMUP_RATIO, eta_min_ratio=0.05):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Floor at eta_min_ratio so LR never reaches zero
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

### ADD THIS: Cyclic LR scheduler for resume extensions
def make_cyclic_scheduler(optimizer, base_lr=1.5e-5, max_lr=8e-5,
                          cycle_steps=75, warmup_within_cycle=8):
    """
    Triangular cyclic LR. The optimizer's param_group['lr'] MUST already be
    set to base_lr before this scheduler is constructed — LambdaLR snapshots
    base_lrs at __init__ time and the lambda multiplies from that base.

    Lambda output range: 1.0 (base_lr) → ratio (max_lr) → 1.0, repeating.
    """
    if base_lr <= 0:
        raise ValueError(f'base_lr must be > 0, got {base_lr}')
    ratio = max_lr / base_lr  # e.g. 8e-5 / 1.5e-5 ≈ 5.33

    def lr_lambda(step):
        cycle_pos = step % cycle_steps
        if cycle_pos < warmup_within_cycle:
            # Ramp up: 1.0 → ratio
            progress = cycle_pos / max(1, warmup_within_cycle)
            return 1.0 + (ratio - 1.0) * progress
        else:
            # Ramp down: ratio → 1.0
            progress = (cycle_pos - warmup_within_cycle) / max(1, cycle_steps - warmup_within_cycle)
            return ratio - (ratio - 1.0) * progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def phase6_prepare_audio_inputs(sample, device):
    inputs = processor(
        audio=sample['wav'],
        sampling_rate=16000,
        return_tensors='pt',
    )
    return {k: v.to(device) for k, v in inputs.items()}


def phase6_quick_eval(tag, max_samples=16):
    model_student.eval()
    text_score, asr_score = quick_eval_chrf(model_student, eval_samples, max_samples=max_samples)
    phase6_eval_history.append({'tag': tag, 'text_chrf': text_score, 'asr_chrf': asr_score})
    print(f'  [{tag}] quick Text-ChrF: {text_score:.2f} | ASR-ChrF: {asr_score:.2f}')
    model_student.train()
    return text_score, asr_score


def phase6_raise_oom(stage_name, step_idx, max_audio_sec, extra=''):
    msg = (
        f'{stage_name} hit CUDA OOM at step {step_idx}. '
        f'Reduce max audio below {max_audio_sec}s'
    )
    if extra:
        msg += f' or {extra}'
    raise RuntimeError(msg)


def phase6_get_cache_entry(sample_key):
    if sample_key not in phase6_cache_index:
        raise KeyError(f'Missing cache index for {sample_key}')

    ref = phase6_cache_index[sample_key]
    shard_idx = ref['shard_idx']
    split_name = ref['split']
    shard_key = f'{split_name}:{shard_idx}'

    if shard_key in phase6_shard_cache:
        entries = phase6_shard_cache.pop(shard_key)
        phase6_shard_cache[shard_key] = entries
    else:
        shard_path = os.path.join(
            CKPT_DIR,
            f"{phase6_cache_checkpoint_name(split_name)}_step{shard_idx:06d}.pt",
        )
        if not os.path.exists(shard_path):
            raise RuntimeError(f'Cache shard missing locally: {shard_path}')
        shard_blob = torch.load(shard_path, map_location='cpu', weights_only=False)
        entries = shard_blob.get('entries', [])
        phase6_shard_cache[shard_key] = entries
        while len(phase6_shard_cache) > PHASE6_SHARD_LRU_LIMIT:
            phase6_shard_cache.popitem(last=False)

    return entries[ref['offset']]


def phase6_pick_training_pair(max_audio_sec, balanced=True):
    if not phase6_cache_keys_by_pair:
        raise RuntimeError('Teacher cache index is not loaded. Run the Phase 6A cache cell first.')

    pair_keys = list(phase6_cache_keys_by_pair.keys())
    all_keys = list(phase6_cache_index.keys())

    for _ in range(512):
        pair = random.choice(pair_keys) if balanced else None
        candidates = phase6_cache_keys_by_pair[pair] if balanced else all_keys
        cache_key = random.choice(candidates)
        ref = phase6_cache_index[cache_key]
        if ref['audio_len_s'] <= max_audio_sec:
            sample_idx = phase6_sample_lookup[cache_key]
            sample = ft_samples[sample_idx]
            entry = phase6_get_cache_entry(cache_key)
            return sample, entry

    raise RuntimeError(
        f'Could not find a training sample under {max_audio_sec}s. '
        'Lower the dataset cap or increase the per-stage audio budget.'
    )


def teacher_generate_tokens(model_teacher, inputs, tgt_lang):
    original_vocoder = model_teacher.vocoder
    teacher_dev = next(model_teacher.parameters()).device
    model_teacher.vocoder = _NoopVocoder(teacher_dev)
    try:
        with torch.no_grad():
            return model_teacher.generate(
                **inputs,
                tgt_lang=tgt_lang,
                return_intermediate_token_ids=True,
                text_num_beams=4,
                text_max_new_tokens=256,
                speech_do_sample=False,
            )
    finally:
        model_teacher.vocoder = original_vocoder


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
        't2u_input_embeds': t2u_input_embeds,
        't2u_attention_mask': t2u_attention_mask,
        't2u_char_input_ids': t2u_char_input_ids,
        't2u_char_count_per_id': t2u_char_count_per_id,
    }


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
        length = int(common_len[b].item())
        if length < 2:
            continue

        s = student_logits[b, :length]
        t = teacher_logits[b, :length]

        total_kl = total_kl + F.kl_div(
            F.log_softmax(s / temperature, dim=-1),
            F.softmax(t / temperature, dim=-1),
            reduction='batchmean',
        ) * (temperature ** 2)

        teacher_hard = t.argmax(dim=-1)
        total_ce = total_ce + F.cross_entropy(s, teacher_hard)
        valid += 1

    if valid == 0:
        raise RuntimeError('No valid T2U overlap found in batch.')

    total_kl = total_kl / valid
    total_ce = total_ce / valid
    total_len = F.smooth_l1_loss(student_len.float(), teacher_len.float())
    return total_kl, total_ce, total_len


def make_t2u_param_groups(model_student, base_lr=8e-5, dur_lr=1e-4, scalar_lr=1e-4, head_lr=8e-5):
    enc_params = []
    dec_params = []
    dur_params = []
    scalar_params = []
    head_params = []

    for name, p in model_student.t2u_model.named_parameters():
        if not p.requires_grad:
            continue
        if 'duration_predictor' in name:
            dur_params.append(p)
        elif 'pos_emb_alpha' in name:
            scalar_params.append(p)
        elif name == 'lm_head.weight':
            head_params.append(p)
        elif name.startswith('model.decoder.'):
            dec_params.append(p)
        else:
            enc_params.append(p)

    groups = []
    if enc_params:
        groups.append({'params': enc_params, 'lr': base_lr, 'weight_decay': 0.01})
    if dec_params:
        groups.append({'params': dec_params, 'lr': base_lr, 'weight_decay': 0.01})
    if dur_params:
        groups.append({'params': dur_params, 'lr': dur_lr, 'weight_decay': 0.00})
    if scalar_params:
        groups.append({'params': scalar_params, 'lr': scalar_lr, 'weight_decay': 0.00})
    if head_params:
        groups.append({'params': head_params, 'lr': head_lr, 'weight_decay': 0.01})
    return groups

def build_target_labels(processor, text_list, tgt_lang, device):
    tok = processor.tokenizer(
        text_target=text_list,
        tgt_lang=tgt_lang,
        return_tensors='pt',
        padding=True,
    )
    labels = tok['input_ids'].clone()  # still 256K IDs

    # Remap every ID through old->new map
    remapped = torch.full_like(labels, -100)  # default: ignore
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            old_id = int(labels[i, j].item())
            if old_id == processor.tokenizer.pad_token_id:
                remapped[i, j] = -100
            elif old_id in _old_to_new:
                remapped[i, j] = _old_to_new[old_id]
            # else: pruned token, stays -100 (ignored in loss)

    return remapped.to(device)

def text_recovery_step(sample, cache_entry, use_teacher_text):
    """
    Text recovery training step.
    Cache entries are already remapped to 22K vocab.
    """
    audio_inputs = phase6_prepare_audio_inputs(sample, student_device)
    
    if use_teacher_text:
        # Cache entries are already remapped - use directly
        labels = cache_entry['teacher_text_sequences'].unsqueeze(0).to(student_device)
        labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    else:
        labels = build_target_labels(processor, [sample['ref']], sample['tgt_lang'], student_device)

    outputs = model_student(
        **audio_inputs,
        labels=labels,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return outputs.loss

print("✓ Function updated")


def run_text_recovery_stage(
    stage_key,
    title,
    steps,
    max_audio_sec,
    text_lr,
    speech_lr=None,
    kd_prob=0.5,
    resume_from_step=0,
):
    freeze_all_student()
    enable_lora_params(model_student.text_decoder, 'text_decoder')
    if speech_lr is not None:
        enable_lora_params(model_student.speech_encoder, 'speech_encoder')

    optimizer_groups = [{'params': trainable_named_params(model_student.text_decoder),
                         'lr': text_lr, 'weight_decay': 0.01}]
    if speech_lr is not None:
        optimizer_groups.append({'params': trainable_named_params(model_student.speech_encoder),
                                 'lr': speech_lr, 'weight_decay': 0.01})

    optimizer = torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.98))
    scheduler = make_cosine_scheduler(optimizer, steps)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ── Derived cadences (clamped so eval/save never fire more often than log) ─
    log_every  = max(1,        LOG_EVERY)
    eval_every = max(log_every, EVAL_EVERY)
    save_every = max(log_every, SAVE_EVERY)

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume_from_step > 0:
        ckpt = phase6_load_latest_local_checkpoint(f'phase6_{stage_key}')
        if ckpt is None:
            raise RuntimeError(
                f'resume_from_step={resume_from_step} requested but no checkpoint found '
                f'for phase6_{stage_key}'
            )
        model_student.text_decoder.load_state_dict(ckpt['text_decoder'])
        if speech_lr is not None and 'speech_encoder' in ckpt:
            model_student.speech_encoder.load_state_dict(ckpt['speech_encoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('logs'):
            phase6_logs[stage_key] = ckpt['logs']
        for _ in range(resume_from_step):
            scheduler.step()
        print(f'  Resumed {stage_key} from optimizer step {resume_from_step}/{steps}')
        # checking if LR scheduler is working
        print(f'  Scheduler base_lrs: {scheduler.base_lrs}')   # must show [1.5e-05, ...]
        print(f'  Current param_group lrs: {[g["lr"] for g in optimizer.param_groups]}')

    model_student.train()
    optimizer.zero_grad(set_to_none=True)

    print(f'\n[{stage_key}] {title}')
    print(f'  optimizer steps : {resume_from_step} → {steps}')
    print(f'  fwd passes left : {(steps - resume_from_step) * GRAD_ACCUM}')
    print(f'  log/eval/save   : every {log_every}/{eval_every}/{save_every} opt steps')
    print(f'  max_audio={max_audio_sec}s | trainable={count_trainable_params(model_student):.2f}M')

    start_micro = resume_from_step * GRAD_ACCUM
    total_micro = steps            * GRAD_ACCUM

    for micro_step in range(start_micro, total_micro):
        sample, cache_entry  = phase6_pick_training_pair(max_audio_sec=max_audio_sec, balanced=True)
        use_teacher_text     = random.random() < kd_prob

        try:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                loss = text_recovery_step(sample, cache_entry, use_teacher_text=use_teacher_text)
            scaler.scale(loss / GRAD_ACCUM).backward()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                safe_gc()
                phase6_raise_oom(stage_key, micro_step + 1, max_audio_sec,
                                 extra='lower the stage audio cap')
            raise

        phase6_logs[stage_key].append({
            'micro_step':       micro_step + 1,
            'loss':             float(loss.detach().cpu()),
            'use_teacher_text': bool(use_teacher_text),
            'text_lr':          optimizer.param_groups[0]['lr'],
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

            # ── Log ───────────────────────────────────────────────────────────
            if opt_step % log_every == 0:
                recent   = phase6_logs[stage_key][-(log_every * GRAD_ACCUM):]
                avg_loss = np.mean([r['loss']             for r in recent])
                kd_ratio = np.mean([r['use_teacher_text'] for r in recent])
                print(
                    f"  [{stage_key}] opt {opt_step:>4}/{steps} | "
                    f"loss={avg_loss:.4f} | KD={kd_ratio:.0%} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            # ── Eval ──────────────────────────────────────────────────────────
            if opt_step % eval_every == 0:
                phase6_quick_eval(f'{stage_key}_step{opt_step}', max_samples=16)

            # ── Checkpoint ────────────────────────────────────────────────────
            if opt_step % save_every == 0:
                state = {
                    'stage':          stage_key,
                    'optimizer_step': opt_step,
                    'steps_total':    steps,
                    'logs':           phase6_logs[stage_key],
                    'text_decoder':   model_student.text_decoder.state_dict(),
                    'optimizer':      optimizer.state_dict(),
                }
                if speech_lr is not None:
                    state['speech_encoder'] = model_student.speech_encoder.state_dict()
                save_checkpoint(state, f'phase6_{stage_key}', opt_step)

    return phase6_logs[stage_key]


def ensure_teacher_loaded():
    global model_teacher
    if 'model_teacher' in globals() and model_teacher is not None:
        return model_teacher

    print('Reloading teacher model for Phase 6 KD...')
    try:
        model_teacher, _ = load_model_from_drive('phase1_vocab_5lang', device_map=None)
    except Exception:
        print('  Teacher checkpoint not found, loading HF base teacher.')
        model_teacher, _ = load_base_model()
    model_teacher = move_model_to_device(model_teacher, teacher_device)
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    disable_generation_cache(model_teacher)
    print(f'  Teacher device: {next(model_teacher.parameters()).device}')
    return model_teacher

def ensure_trainable_fp32():
    """Cast all trainable parameters to FP32 to avoid GradScaler FP16 grad errors."""
    count = 0
    for p in model_student.parameters():
        if p.requires_grad and p.dtype == torch.float16:
            p.data = p.data.float()
            count += 1
    print(f'  Cast {count} trainable FP16 params to FP32')

def run_t2u_recovery_stage(
    stage_key,
    title,
    steps,
    max_audio_sec,
    resume_from_step=0,
):
    ensure_teacher_loaded()
    if PHASE6_T2U_TRAIN_MODE == 'selective':
        mark_t2u_selective_trainable()
    else:
        mark_t2u_trainable_full()

    ensure_trainable_fp32()  # <-- ADD THIS

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

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume_from_step > 0:
        ckpt = phase6_load_latest_local_checkpoint(f'phase6_{stage_key}')
        if ckpt is None:
            raise RuntimeError(
                f'resume_from_step={resume_from_step} requested but no checkpoint found '
                f'for phase6_{stage_key}'
            )
        model_student.t2u_model.load_state_dict(ckpt['t2u_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('logs'):
            phase6_logs[stage_key] = ckpt['logs']
        for _ in range(resume_from_step):
            scheduler.step()
        print(f'  Resumed {stage_key} from optimizer step {resume_from_step}/{steps}')

    model_student.train()
    optimizer.zero_grad(set_to_none=True)

    print(f'\n[{stage_key}] {title}')
    print(f'  optimizer steps : {resume_from_step} → {steps}')
    print(f'  fwd passes left : {(steps - resume_from_step) * GRAD_ACCUM}')
    print(f'  log/eval/save   : every {log_every}/{eval_every}/{save_every} opt steps')
    print(f'  max_audio={max_audio_sec}s | trainable={count_trainable_params(model_student):.2f}M')

    start_micro = resume_from_step * GRAD_ACCUM
    total_micro = steps            * GRAD_ACCUM

    for micro_step in range(start_micro, total_micro):
        sample, cache_entry    = phase6_pick_training_pair(max_audio_sec=max_audio_sec, balanced=True)
        teacher_text_sequences = cache_entry['teacher_text_sequences'].unsqueeze(0)
        audio_inputs_student   = phase6_prepare_audio_inputs(sample, student_device)
        audio_inputs_teacher   = {k: v.to(teacher_device) for k, v in audio_inputs_student.items()}

        try:
            with torch.no_grad():
                teacher_cond = build_t2u_conditioning_from_sequences(
                    model_teacher,
                    input_features=audio_inputs_teacher['input_features'],
                    attention_mask=audio_inputs_teacher.get('attention_mask'),
                    text_sequences=teacher_text_sequences.to(teacher_device),
                )
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    teacher_t2u = model_teacher.t2u_model(
                        inputs_embeds=teacher_cond['t2u_input_embeds'],
                        attention_mask=teacher_cond['t2u_attention_mask'],
                        char_input_ids=teacher_cond['t2u_char_input_ids'],
                        char_count_per_id=teacher_cond['t2u_char_count_per_id'],
                        output_attentions=False, output_hidden_states=False, return_dict=True,
                    )

            student_cond = build_t2u_conditioning_from_sequences(
                model_student,
                input_features=audio_inputs_student['input_features'],
                attention_mask=audio_inputs_student.get('attention_mask'),
                text_sequences=teacher_text_sequences.to(student_device),
            )
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                student_t2u = model_student.t2u_model(
                    inputs_embeds=student_cond['t2u_input_embeds'],
                    attention_mask=student_cond['t2u_attention_mask'],
                    char_input_ids=student_cond['t2u_char_input_ids'],
                    char_count_per_id=student_cond['t2u_char_count_per_id'],
                    output_attentions=False, output_hidden_states=False, return_dict=True,
                )
                teacher_t2u.last_hidden_state = teacher_t2u.last_hidden_state.to(student_device)
                teacher_t2u.padding_mask      = teacher_t2u.padding_mask.to(student_device)
                t2u_soft, t2u_hard, t2u_len   = t2u_overlap_losses(student_t2u, teacher_t2u)
                loss = 0.60 * t2u_soft + 0.30 * t2u_hard + 0.10 * t2u_len
            scaler.scale(loss / GRAD_ACCUM).backward()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                safe_gc()
                phase6_raise_oom(stage_key, micro_step + 1, max_audio_sec,
                                 extra='set PHASE6_T2U_TRAIN_MODE="selective"')
            raise

        phase6_logs[stage_key].append({
            'micro_step': micro_step + 1,
            'loss':       float(loss.detach().cpu()),
            't2u_soft':   float(t2u_soft.detach().cpu()),
            't2u_hard':   float(t2u_hard.detach().cpu()),
            't2u_len':    float(t2u_len.detach().cpu()),
            'lr':         optimizer.param_groups[0]['lr'],
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

            # ── Log ───────────────────────────────────────────────────────────
            if opt_step % log_every == 0:
                recent   = phase6_logs[stage_key][-(log_every * GRAD_ACCUM):]
                avg_loss = np.mean([r['loss']     for r in recent])
                avg_soft = np.mean([r['t2u_soft'] for r in recent])
                avg_hard = np.mean([r['t2u_hard'] for r in recent])
                avg_len  = np.mean([r['t2u_len']  for r in recent])
                print(
                    f"  [{stage_key}] opt {opt_step:>4}/{steps} | "
                    f"loss={avg_loss:.4f} | soft={avg_soft:.4f} | "
                    f"hard={avg_hard:.4f} | len={avg_len:.4f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            # ── Eval ──────────────────────────────────────────────────────────
            if opt_step % eval_every == 0:
                phase6_quick_eval(f'{stage_key}_step{opt_step}', max_samples=16)

            # ── Checkpoint ────────────────────────────────────────────────────
            if opt_step % save_every == 0:
                save_checkpoint(
                    {
                        'stage':          stage_key,
                        'optimizer_step': opt_step,
                        'steps_total':    steps,
                        'mode':           PHASE6_T2U_TRAIN_MODE,
                        'logs':           phase6_logs[stage_key],
                        't2u_model':      model_student.t2u_model.state_dict(),
                        'optimizer':      optimizer.state_dict(),
                    },
                    f'phase6_{stage_key}', opt_step,
                )

    return phase6_logs[stage_key]


def run_joint_polish_stage(
    stage_key,
    title,
    steps,
    max_audio_sec,
    resume_from_step=0,
):
    ensure_teacher_loaded()
    freeze_all_student()
    enable_lora_params(model_student.text_decoder, 'text_decoder')
    if PHASE6_T2U_TRAIN_MODE == 'selective':
        mark_t2u_selective_trainable()
    else:
        mark_t2u_trainable_full()

    ensure_trainable_fp32()  # <-- ADD THIS
    
    enable_lora_params(model_student.text_decoder, 'text_decoder')

    text_params = trainable_named_params(model_student.text_decoder)
    t2u_groups  = make_t2u_param_groups(model_student, base_lr=4e-5, dur_lr=5e-5,
                                         scalar_lr=5e-5, head_lr=4e-5)
    optimizer   = torch.optim.AdamW(
        [{'params': text_params, 'lr': 1e-5, 'weight_decay': 0.01}] + t2u_groups,
        betas=(0.9, 0.98),
    )
    scheduler = make_cosine_scheduler(optimizer, steps)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    log_every  = max(1,        LOG_EVERY)
    eval_every = max(log_every, EVAL_EVERY)
    save_every = max(log_every, SAVE_EVERY)

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume_from_step > 0:
        ckpt = phase6_load_latest_local_checkpoint(f'phase6_{stage_key}')
        if ckpt is None:
            raise RuntimeError(
                f'resume_from_step={resume_from_step} requested but no checkpoint found '
                f'for phase6_{stage_key}'
            )
        model_student.text_decoder.load_state_dict(ckpt['text_decoder'])
        model_student.t2u_model.load_state_dict(ckpt['t2u_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('logs'):
            phase6_logs[stage_key] = ckpt['logs']
        for _ in range(resume_from_step):
            scheduler.step()
        print(f'  Resumed {stage_key} from optimizer step {resume_from_step}/{steps}')

    model_student.train()
    optimizer.zero_grad(set_to_none=True)

    print(f'\n[{stage_key}] {title}')
    print(f'  optimizer steps : {resume_from_step} → {steps}')
    print(f'  fwd passes left : {(steps - resume_from_step) * GRAD_ACCUM}')
    print(f'  log/eval/save   : every {log_every}/{eval_every}/{save_every} opt steps')
    print(f'  max_audio={max_audio_sec}s | trainable={count_trainable_params(model_student):.2f}M')

    start_micro = resume_from_step * GRAD_ACCUM
    total_micro = steps            * GRAD_ACCUM

    for micro_step in range(start_micro, total_micro):
        sample, cache_entry    = phase6_pick_training_pair(max_audio_sec=max_audio_sec, balanced=True)
        teacher_text_sequences = cache_entry['teacher_text_sequences'].unsqueeze(0)
        use_teacher_text       = random.random() < PHASE6_TEXT_KD_PROB
        target_text            = cache_entry['teacher_text_str'] if use_teacher_text else sample['ref']

        audio_inputs_student = phase6_prepare_audio_inputs(sample, student_device)
        audio_inputs_teacher = {k: v.to(teacher_device) for k, v in audio_inputs_student.items()}
        labels               = build_target_labels(processor, [target_text], sample['tgt_lang'],
                                                   student_device)

        try:
            with torch.no_grad():
                teacher_cond = build_t2u_conditioning_from_sequences(
                    model_teacher,
                    input_features=audio_inputs_teacher['input_features'],
                    attention_mask=audio_inputs_teacher.get('attention_mask'),
                    text_sequences=teacher_text_sequences.to(teacher_device),
                )
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    teacher_t2u = model_teacher.t2u_model(
                        inputs_embeds=teacher_cond['t2u_input_embeds'],
                        attention_mask=teacher_cond['t2u_attention_mask'],
                        char_input_ids=teacher_cond['t2u_char_input_ids'],
                        char_count_per_id=teacher_cond['t2u_char_count_per_id'],
                        output_attentions=False, output_hidden_states=False, return_dict=True,
                    )

            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                text_outputs = model_student(
                    **audio_inputs_student,
                    labels=labels,
                    use_cache=False,
                    output_attentions=False, output_hidden_states=False, return_dict=True,
                )
                student_cond = build_t2u_conditioning_from_sequences(
                    model_student,
                    input_features=audio_inputs_student['input_features'],
                    attention_mask=audio_inputs_student.get('attention_mask'),
                    text_sequences=teacher_text_sequences.to(student_device),
                )
                student_t2u = model_student.t2u_model(
                    inputs_embeds=student_cond['t2u_input_embeds'],
                    attention_mask=student_cond['t2u_attention_mask'],
                    char_input_ids=student_cond['t2u_char_input_ids'],
                    char_count_per_id=student_cond['t2u_char_count_per_id'],
                    output_attentions=False, output_hidden_states=False, return_dict=True,
                )
                teacher_t2u.last_hidden_state = teacher_t2u.last_hidden_state.to(student_device)
                teacher_t2u.padding_mask      = teacher_t2u.padding_mask.to(student_device)
                t2u_soft, t2u_hard, t2u_len   = t2u_overlap_losses(student_t2u, teacher_t2u)
                text_loss = text_outputs.loss
                loss = 0.35 * text_loss + 0.40 * t2u_soft + 0.20 * t2u_hard + 0.05 * t2u_len
            scaler.scale(loss / GRAD_ACCUM).backward()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                safe_gc()
                phase6_raise_oom(stage_key, micro_step + 1, max_audio_sec,
                                 extra='reduce Stage 6D audio cap')
            raise

        phase6_logs[stage_key].append({
            'micro_step':       micro_step + 1,
            'loss':             float(loss.detach().cpu()),
            'text_loss':        float(text_loss.detach().cpu()),
            't2u_soft':         float(t2u_soft.detach().cpu()),
            't2u_hard':         float(t2u_hard.detach().cpu()),
            't2u_len':          float(t2u_len.detach().cpu()),
            'use_teacher_text': bool(use_teacher_text),
            'lr':               optimizer.param_groups[0]['lr'],
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

            # ── Log ───────────────────────────────────────────────────────────
            if opt_step % log_every == 0:
                recent   = phase6_logs[stage_key][-(log_every * GRAD_ACCUM):]
                avg_loss = np.mean([r['loss']      for r in recent])
                avg_text = np.mean([r['text_loss'] for r in recent])
                avg_soft = np.mean([r['t2u_soft']  for r in recent])
                avg_hard = np.mean([r['t2u_hard']  for r in recent])
                print(
                    f"  [{stage_key}] opt {opt_step:>4}/{steps} | "
                    f"loss={avg_loss:.4f} | text={avg_text:.4f} | "
                    f"soft={avg_soft:.4f} | hard={avg_hard:.4f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            # ── Eval ──────────────────────────────────────────────────────────
            if opt_step % eval_every == 0:
                phase6_quick_eval(f'{stage_key}_step{opt_step}', max_samples=16)

            # ── Checkpoint ────────────────────────────────────────────────────
            if opt_step % save_every == 0:
                save_checkpoint(
                    {
                        'stage':          stage_key,
                        'optimizer_step': opt_step,
                        'steps_total':    steps,
                        'mode':           PHASE6_T2U_TRAIN_MODE,
                        'logs':           phase6_logs[stage_key],
                        'text_decoder':   model_student.text_decoder.state_dict(),
                        't2u_model':      model_student.t2u_model.state_dict(),
                        'optimizer':      optimizer.state_dict(),
                    },
                    f'phase6_{stage_key}', opt_step,
                )

    return phase6_logs[stage_key]


print('Loading Phase 5 student model...')
model_student, processor = load_model_from_drive('phase5_dec_14L', device_map=None)
disable_generation_cache(model_student)

print('Loading teacher model (vocab pruned) for Phase 6A cache build...')
try:
    model_teacher, _ = load_model_from_drive('phase1_vocab_5lang', device_map=None)
except Exception:
    print('  Teacher checkpoint not found, NOT loading HF base teacher.')
    # model_teacher, _ = load_base_model()
model_teacher = move_model_to_device(model_teacher, teacher_device)
model_teacher.eval()
for p in model_teacher.parameters():
    p.requires_grad_(False)
disable_generation_cache(model_teacher)

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

model_student.speech_encoder = wrap_with_lora_if_needed(
    model_student.speech_encoder,
    speech_lora_cfg,
    'speech_encoder',
)
model_student.text_decoder = wrap_with_lora_if_needed(
    model_student.text_decoder,
    text_lora_cfg,
    'text_decoder',
)
freeze_all_student()




# Build once at startup, reuse everywhere
_old_to_new = {
    int(old_id): new_id
    for new_id, old_id in enumerate(model_student._vocab_remap_to_old.tolist())
}
_student_vocab_size = model_student.text_decoder.get_base_model().embed_tokens.num_embeddings
print(f"Remap table built: {len(_old_to_new)} entries, student vocab={_student_vocab_size}")

model_student = move_model_to_device(model_student, student_device)

print(f'Student device: {next(model_student.parameters()).device}')
print(f'Teacher device: {next(model_teacher.parameters()).device}')
print_model_breakdown(model_student, 'Phase 6 student with LoRA wrappers')
gpu_mem()

Cell2:
def phase6_cache_checkpoint_name(split_name):
    return f'{PHASE6_CACHE_PREFIX}_{split_name}'


def phase6_cache_manifest_name(split_name):
    return f'{PHASE6_CACHE_PREFIX}_{split_name}_manifest'


def phase6_save_checkpoint_local(state, name, step=0, keep=3):
    fname = f'{name}_step{step:06d}.pt'
    path = f'{CKPT_DIR}/{fname}'
    torch.save(state, path)
    mb = os.path.getsize(path) / 1e6
    print(f'[ckpt-local] Saved {fname} ({mb:.1f} MB)')
    old = sorted(glob.glob(f'{CKPT_DIR}/{name}_step*.pt'))
    for f in old[:-keep]:
        if os.path.exists(f):
            os.remove(f)
    return path


def phase6_rclone_copy_checkpoint_family(prefixes, direction='push'):
    if not ON_KAGGLE:
        return

    if isinstance(prefixes, str):
        prefixes = [prefixes]

    include_args = ' '.join([f'--include "{prefix}_step*.pt"' for prefix in prefixes])
    if direction == 'push':
        src = f'{CKPT_DIR}/'
        dst = f'{GDRIVE_ROOT}/checkpoints/'
        verb = 'push'
    else:
        src = f'{GDRIVE_ROOT}/checkpoints/'
        dst = f'{CKPT_DIR}/'
        verb = 'pull'

    cmd = (
        f'rclone copy "{src}" "{dst}" {include_args} '
        f'--transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'[rclone] phase6 cache {verb} failed: {result.stderr[:300]}')
    print(f'[rclone] Phase 6 cache {verb} OK for: {prefixes}')


def phase6_load_latest_local_checkpoint(name):
    files = sorted(glob.glob(f'{CKPT_DIR}/{name}_step*.pt'))
    if not files:
        return None
    return torch.load(files[-1], map_location='cpu', weights_only=False)


def phase6_load_manifest(split_name):
    name = phase6_cache_manifest_name(split_name)
    manifest = phase6_load_latest_local_checkpoint(name)
    if manifest is None and ON_KAGGLE:
        print(f'No local manifest for {split_name}. Pulling Phase 6 cache family from Drive...')
        phase6_rclone_copy_checkpoint_family(
            [phase6_cache_checkpoint_name(split_name), phase6_cache_manifest_name(split_name)],
            direction='pull',
        )
        manifest = phase6_load_latest_local_checkpoint(name)

    if manifest is not None:
        return manifest

    return {
        'split': split_name,
        'total_cached': 0,
        'cache_index': {},
        'pair_to_keys': {},
        'num_shards': 0,
        'synced_until': 0,
    }


def phase6_save_manifest_local(split_name, manifest):
    return phase6_save_checkpoint_local(
        manifest,
        phase6_cache_manifest_name(split_name),
        step=manifest['num_shards'],
        keep=8,
    )


def phase6_save_cache_shard_local(split_name, shard_idx, entries, total_cached):
    return phase6_save_checkpoint_local(
        {
            'split': split_name,
            'shard_idx': shard_idx,
            'entries': entries,
            'total_cached': total_cached,
        },
        phase6_cache_checkpoint_name(split_name),
        step=shard_idx,
        keep=100000,
    )


def build_teacher_cache_entry(model_teacher, sample):
    teacher_inputs = phase6_prepare_audio_inputs(sample, teacher_device)
    out = teacher_generate_tokens(model_teacher, teacher_inputs, tgt_lang=sample['tgt_lang'])

    if out.unit_sequences is None:
        return None, 'teacher_returned_no_unit_sequence'

    teacher_text_sequences = out.sequences[0].detach().cpu()
    
    # Validate sequence length
    if teacher_text_sequences.numel() == 0:
        return None, 'empty_teacher_sequence'
    
    if teacher_text_sequences.numel() > 512:  # max position embeddings
        return None, f'teacher_sequence_too_long:{teacher_text_sequences.numel()}'

    teacher_unit_sequences = out.unit_sequences[0].detach().cpu()
    teacher_text_str = processor.batch_decode(
        teacher_text_sequences.unsqueeze(0),
        skip_special_tokens=True,
    )[0].strip()

    unit_len = int(teacher_unit_sequences.numel())

    if not teacher_text_str:
        return None, 'empty_teacher_text'
    if unit_len < PHASE6_MIN_TEACHER_UNIT_TOKENS:
        return None, f'unit_sequence_too_short:{unit_len}'
    if unit_len > PHASE6_MAX_TEACHER_UNIT_TOKENS:
        return None, f'unit_sequence_too_long:{unit_len}'

    return {
        'sample_key': phase6_sample_key(sample),
        'sample_id': sample['id'],
        'src_lang': sample['src_lang'],
        'tgt_lang': sample['tgt_lang'],
        'teacher_text_sequences': teacher_text_sequences,
        'teacher_text_str': teacher_text_str,
        'teacher_unit_sequences': teacher_unit_sequences,
        'audio_len_s': len(sample['wav']) / 16000.0,
    }, None


def build_or_load_phase6_cache(split_name, samples, shard_size=PHASE6_CACHE_SHARD_SIZE):
    manifest = phase6_load_manifest(split_name)
    cache_index = manifest.get('cache_index', {})
    pair_to_keys = defaultdict(list)
    for pair, keys in manifest.get('pair_to_keys', {}).items():
        pair_to_keys[pair].extend(keys)
    skipped_count = manifest.get('skipped_count', 0)
    total_samples = len(samples)
    sync_every = max(1, math.ceil(total_samples / PHASE6_CACHE_SYNC_PARTS))
    next_sync_target = max(sync_every, ((len(cache_index) // sync_every) + 1) * sync_every)
    next_sync_target = min(total_samples, next_sync_target)
    print(
        f'Existing {split_name} cache index: {len(cache_index)} samples | '
        f'shard_size={shard_size} | sync_every~{sync_every} samples | skipped={skipped_count}'
    )
    # If cache marked complete, skip entirely
    if manifest.get('cache_complete', False):
        print(f'✅ Cache already marked complete. Cached={len(cache_index)}')
        return manifest
    # If we already have all cacheable samples (cached + known skipped >= total), treat as done
    if len(cache_index) + skipped_count >= total_samples and len(cache_index) > 0:
        print(f'✅ Cache appears complete (cached {len(cache_index)} + skipped {skipped_count} >= {total_samples}). Done.')
        manifest['cache_complete'] = True
        phase6_save_manifest_local(split_name, manifest)
        return manifest
    buffer = []
    shard_idx = int(manifest.get('num_shards', 0))

    
    for idx in range(len(samples)):
        sample = samples[idx]
        sample_key = phase6_sample_key(sample)
        if sample_key in cache_index:
            print(f"already cached> {sample_key}")
            continue
        entry, err = build_teacher_cache_entry(model_teacher, sample)
        if entry is None:
            skipped_count += 1
            print(f'  skipped {skipped_count} samples (latest: {err}) | {sample_key}')
            continue
        offset = len(buffer)
        buffer.append(entry)
        pair = f"{entry['src_lang']}->{entry['tgt_lang']}"
        cache_index[sample_key] = {
            'split': split_name,
            'shard_idx': shard_idx,
            'offset': offset,
            'audio_len_s': entry['audio_len_s'],
            'src_lang': entry['src_lang'],
            'tgt_lang': entry['tgt_lang'],
        }
        pair_to_keys[pair].append(sample_key)
        if len(buffer) >= shard_size:
            phase6_save_cache_shard_local(
                split_name=split_name,
                shard_idx=shard_idx,
                entries=buffer,
                total_cached=len(cache_index),
            )
            shard_idx += 1
            manifest = {
                'split': split_name,
                'total_cached': len(cache_index),
                'cache_index': cache_index,
                'pair_to_keys': dict(pair_to_keys),
                'num_shards': shard_idx,
                'synced_until': manifest.get('synced_until', 0),
                'skipped_count': skipped_count,
            }
            phase6_save_manifest_local(split_name, manifest)
            buffer = []
            safe_gc()
        if len(cache_index) >= next_sync_target:
            manifest = {
                'split': split_name,
                'total_cached': len(cache_index),
                'cache_index': cache_index,
                'pair_to_keys': dict(pair_to_keys),
                'num_shards': shard_idx,
                'synced_until': len(cache_index),
                'skipped_count': skipped_count,
            }
            phase6_save_manifest_local(split_name, manifest)
            phase6_rclone_copy_checkpoint_family(
                [phase6_cache_checkpoint_name(split_name), phase6_cache_manifest_name(split_name)],
                direction='push',
            )
            print(f'  synced Phase 6A cache to Drive at {len(cache_index)}/{total_samples} samples')
            next_sync_target = min(total_samples, next_sync_target + sync_every)
        if (idx + 1) % 200 == 0:
            print(f'  cached {len(cache_index)}/{total_samples} {split_name} samples')
    if buffer:
        phase6_save_cache_shard_local(
            split_name=split_name,
            shard_idx=shard_idx,
            entries=buffer,
            total_cached=len(cache_index),
        )
        shard_idx += 1
    manifest = {
        'split': split_name,
        'total_cached': len(cache_index),
        'cache_index': cache_index,
        'pair_to_keys': dict(pair_to_keys),
        'num_shards': shard_idx,
        'synced_until': len(cache_index),
        'skipped_count': skipped_count,
        'cache_complete': True,
    }
    phase6_save_manifest_local(split_name, manifest)
    phase6_rclone_copy_checkpoint_family(
        [phase6_cache_checkpoint_name(split_name), phase6_cache_manifest_name(split_name)],
        direction='push',
    )
    print(f'✅ Done. Cached={len(cache_index)} | Skipped={skipped_count}')
    return manifest


print('Building or loading Phase 6A teacher cache...')
phase6_cache_manifest = build_or_load_phase6_cache('train', ft_samples, shard_size=PHASE6_CACHE_SHARD_SIZE)
phase6_cache_index = phase6_cache_manifest['cache_index']
phase6_cache_keys_by_pair = defaultdict(list)
for pair, keys in phase6_cache_manifest['pair_to_keys'].items():
    phase6_cache_keys_by_pair[pair].extend(keys)

phase6_sample_lookup = build_sample_lookup(ft_samples)
phase6_shard_cache = OrderedDict()

print(f"Phase 6A cache ready: {phase6_cache_manifest['total_cached']} entries")
print(f"  shards: {phase6_cache_manifest['num_shards']} | shard_size: {PHASE6_CACHE_SHARD_SIZE}")
print(f"  sync parts: {PHASE6_CACHE_SYNC_PARTS} | shard LRU limit: {PHASE6_SHARD_LRU_LIMIT}")
for pair, keys in sorted(phase6_cache_keys_by_pair.items()):
    print(f'  {pair:<12} {len(keys):>5} samples')

# print('Unloading teacher after cache build to free GPU1...')
del model_teacher
model_teacher = None
safe_gc()
gpu_mem()

Cell3:
# ── Helper: find the latest saved optimizer step for a stage ─────────────────
def phase6_get_resume_step(stage_key):
    """
    Returns the optimizer step stored in the latest checkpoint for this stage.
    Returns 0 if no checkpoint exists (fresh start).
    To extend a run: just increase the STAGE6XX_STEPS constant and re-run the cell.
    The function will pick up from the last saved checkpoint automatically.
    """
    ckpt = phase6_load_latest_local_checkpoint(f'phase6_{stage_key}')
    if ckpt is None:
        print(f'  [{stage_key}] No checkpoint found — starting fresh.')
        return 0
    saved_step   = ckpt.get('optimizer_step', 0)
    saved_total  = ckpt.get('steps_total',    '?')
    print(f'  [{stage_key}] Checkpoint found at optimizer step {saved_step}/{saved_total}')
    return saved_step

Cell4:
# ── Stage 6C ──────────────────────────────────────────────────────────────────
phase6_logs['6c'] = run_t2u_recovery_stage(
    stage_key        = '6c',
    title            = 'Native T2U recovery with teacher KD',
    steps            = STAGE6C_STEPS,
    max_audio_sec    = MAX_AUDIO_SEC_C,
    resume_from_step = phase6_get_resume_step('6c'),
)
phase6_quick_eval('stage6c_done', max_samples=16)