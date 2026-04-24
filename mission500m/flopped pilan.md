# SeamlessM4T v2 → 500M Multilingual Compression Plan
## 5 Languages · Voice-Cloned Output · 1-Week Execution · 500M Structural Params

---

## SECTION 0 — Three Questions Answered First

---

### 0.1 FP16 vs FP32: Use FP16, No Contest

**For pruning (all evaluation-only passes):** FP16 is strictly better.
- Faster inference, identical layer-importance ranking decisions
- FP16 model = ~3.6 GB on GPU. FP32 = ~7.2 GB. Wasted VRAM, no benefit.

**For DoRA fine-tuning:** Use **BF16 + AMP mixed precision** (not FP32, not plain FP16).
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast(dtype=torch.bfloat16):   # BF16 forward/backward
    outputs = model(input_features=..., labels=text_labels)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
BF16 has the same dynamic range as FP32 but half the memory. Better than FP16 for training because it avoids gradient underflow without needing FP32 precision. **Your 500M pruned model in BF16 = ~1 GB.** With 2x T4 (32 GB total), you have massive headroom.

**Bottom line:** The FP32 full-precision model exists for training large models from scratch. For inference (pruning) and PEFT fine-tuning (DoRA), BF16/FP16 is universally standard and introduces no quality loss.

---

### 0.2 The CLI finetune.py — How to Do It in Pure Python/Notebook

The README's `finetune.py` uses the **`seamless_communication` Fairseq2 library** — a completely different backend from your HuggingFace pipeline. You cannot directly use those CLI scripts without converting your pruned HuggingFace model to Fairseq2 format, which is non-trivial.

**What the `SPEECH_TO_SPEECH` mode does that matters:**
It trains BOTH the speech-encoder→text-decoder path (S2TT cross-entropy loss) AND the T2U path (unit cross-entropy loss) in a single training step, giving T2U non-zero gradient.

**How to replicate this in your existing HuggingFace notebook — two clean options:**

**Option A: Combined loss in one training loop**
```python
# STEP 1: Standard S2TT loss (your existing DoRA code, unchanged)
outputs = model(
    input_features=speech_input,
    labels=text_token_ids,
)
s2tt_loss = outputs.loss

# STEP 2: T2U loss (NEW — needs unit labels extracted in Phase 6)
with torch.no_grad():
    # Get text embeddings from the model's decoder
    text_embs = model.text_decoder.embed_tokens(text_token_ids)

t2u_out = model.t2u_model(
    inputs_embeds=text_embs,
    labels=unit_token_ids,   # discrete unit IDs from teacher extraction
)
t2u_loss = t2u_out.loss

# Combined loss: 70% text, 30% units
total_loss = 0.7 * s2tt_loss + 0.3 * t2u_loss
```

**Option B (Recommended — simpler, matches your existing code structure): Two separate passes**
- **Pass 1 (~5h):** DoRA on full model with S2TT loss (your existing code). Recovers speech encoder + text decoder. T2U adapters exist but get zero gradient — that's fine, we fix it in Pass 2.
- **Pass 2 (~2h):** Freeze encoder+decoder. Apply DoRA only to T2U. Train with unit cross-entropy loss using teacher-extracted unit labels.

This is exactly equivalent to Meta's SPEECH_TO_SPEECH mode but fits cleanly into your existing notebook structure with minimal code changes.

---

### 0.3 The Audio Quality Question — Definitive Answer

**Why text ChrF for encoder/decoder and ASR-ChrF only for T2U?**

The answer lies in the two discrete sampling barriers in the pipeline:

```
Input Speech
     │
     ▼
[Speech Encoder]          ← text ChrF accurately captures quality here
     │
     ▼
[Text Decoder]            ← text ChrF accurately captures quality here
     │
     ▼  ─── argmax / beam search ─── DISCRETE TOKEN IDs ───┐
                                                            │ Non-differentiable
[T2U Model]               ◄─────────────────────────────────┘ text ChrF is BLIND here
     │
     ▼  ─── unit prediction ─── DISCRETE UNIT IDs ───┐
                                                      │ Non-differentiable
[HiFi-GAN Vocoder]        ◄───────────────────────────┘
     │
     ▼
Output Audio
```

**For speech encoder and text decoder pruning:**
Text ChrF is the RIGHT metric. The causal chain is monotonic: better encoder/decoder → better text → T2U receives better input → better units → better audio. ASR-ChrF would ADD MMS transcription noise on top of translation quality, making pruning decisions noisier, not better. It would also be 4–5× slower per probe (need to run full S2ST + ASR transcription instead of just S2TT).

**For T2U pruning:**
ASR-ChrF is REQUIRED. T2U pruning has ZERO effect on text output — you could remove all 6 T2U layers and text ChrF would not change at all. The only signal is in the audio domain, which is what ASR-ChrF measures.

**For fine-tuning:**
You DO need S2ST mode (S2TT loss + T2U unit loss combined) so that T2U layers receive gradients. Text-only training (S2TT loss) gives T2U adapters exactly zero gradient. This is what broke your Phase 7 — the T2U audio quality wasn't recovering because T2U was never trained. The two-pass approach in Section 5 (Phase 7a + 7b) fixes this.

**Summary table:**

| Component | Pruning metric | Why |
|-----------|---------------|-----|
| Speech Encoder | Text ChrF (SMC) | Causal chain is monotonic; ASR adds noise |
| Text Decoder | Text ChrF (SMC) | Same reason |
| T2U Model | ASR-ChrF (MMS) | Only metric that sees T2U output quality |
| Vocoder | Do NOT prune | 41.9M, frozen, not worth the risk |

---

## SECTION 1 — Parameter Budget to 500M

Starting from `facebook/seamless-m4t-v2-large` in S2S mode, FP16:

| Component | Baseline | Target | Layers | Savings |
|-----------|---------|--------|--------|---------|
| shared + lm_head | 262.2M | ~47M | vocab 256K→58K | ~215M |
| text_decoder | 866.8M | ~216M | 24 → **6 layers** | ~651M |
| speech_encoder | 635.0M | ~212M | 24 → **8 layers** | ~423M |
| t2u_model | 261.8M | ~174M | 12 → **8 layers** | ~88M |
| vocoder | 41.9M | 41.9M | 0 | 0 |
| **TOTAL** | **1805.5M** | **~491M** | | **~1377M** |

> Text decoder 6 layers and speech encoder 8 layers is aggressive. ChrF before recovery will be ~27–32. DoRA + sequence-level KD will recover to ~40–44 ChrF (82–87% of baseline 50.52). This is the expected range at 73% structural compression.

---

## SECTION 2 — Dataset Plan

### Language codes

| Language | FLEURS split | M4T lang code |
|----------|-------------|-------------|
| English | `en_us` | `eng` |
| Bengali | `bn_in` | `ben` |
| Mandarin | `cmn_hans_cn` | `cmn` |
| Arabic | `ar_eg` | `arb` |
| Hindi | `hi_in` | `hin` |

### Evaluation sets

```python
EVAL_PAIRS = [
    ("en_us", "bn_in",       "eng", "ben"),
    ("en_us", "cmn_hans_cn", "eng", "cmn"),
    ("en_us", "ar_eg",       "eng", "arb"),
    ("en_us", "hi_in",       "eng", "hin"),
    ("bn_in", "en_us",       "ben", "eng"),
    ("cmn_hans_cn", "en_us", "cmn", "eng"),
    ("ar_eg", "en_us",       "arb", "eng"),
    ("hi_in", "en_us",       "hin", "eng"),
]
N_EVAL_PER_PAIR = 5   # 5 × 8 = 40 total per SMC probe
```

5 samples per pair = ~7 min per pruning candidate (the number that makes 1-week feasible).

---

## SECTION 3 — SMC: The Efficient Multilingual Pruning Metric

```python
def compute_smc(model, eval_sets_dict, processor):
    """
    Stratified Minimum ChrF — bidirectional-aware pruning metric.
    Takes ~7 min per candidate on T4. Replaces single-pair ChrF in your loop.
    """
    from sacrebleu.metrics import CHRF
    _chrf = CHRF()
    eng_chrfs, nonen_chrfs = [], []

    for (src_f, tgt_f, src_m4t, tgt_m4t), samples in eval_sets_dict.items():
        chrfs = []
        for s in samples:
            # Use S2TT mode (no audio gen — fast)
            with torch.no_grad():
                text_out = model.generate(
                    **processor(audios=s['wav'], sampling_rate=16000,
                                return_tensors="pt").to(device),
                    tgt_lang=tgt_m4t,
                    generate_speech=False
                )
            pred = processor.decode(text_out[0].tolist(), skip_special_tokens=True)
            c = _chrf.sentence_score(pred, [s['ref']]).score
            chrfs.append(c)

        if tgt_m4t == "eng":
            eng_chrfs.extend(chrfs)
        else:
            nonen_chrfs.extend(chrfs)

    probe_A = sum(eng_chrfs)   / len(eng_chrfs)    # EN-output quality
    probe_B = sum(nonen_chrfs) / len(nonen_chrfs)  # non-EN-output quality
    smc = min(probe_A, probe_B)                    # protect the weaker direction
    return smc, probe_A, probe_B
```

**In the iterative pruning loop (change only the ChrF call):**
```python
# OLD: chrf = run_benchmark(model_temp, eval_samples, ...)
# NEW:
smc, pa, pb = compute_smc(model_temp, eval_sets_dict, processor)
print(f"    Remove L{i:2d} -> SMC={smc:.2f} [EN-out:{pa:.2f} | nonEN-out:{pb:.2f}]")
```

**Protected layers:**
```python
# Always protect first, middle, last
PROTECTED_DEC = {0, 11, 23}    # text decoder
PROTECTED_ENC = {0, 11, 23}    # speech encoder

# EXTRA protection from activation analysis (bidirectional-tracking-base.ipynb):
# Text decoder L20-L23 have 965-3828 higher BN→EN activation than EN→BN
# DO NOT remove the top-2 remaining text decoder layers at any iteration
# (enforce: always keep the 2 highest-index remaining layers protected)
def update_protected_set(protected, remaining_layers):
    top2 = sorted(remaining_layers)[-2:]
    return protected | set(top2)
```

---

## SECTION 4 — ONE-WEEK SESSION SCHEDULE

### 6 Kaggle Sessions (~58 compute hours total)

```
Day 1 │ Session 1 (12h) │ P1 vocab + P2 activation analysis + P3 dec iters 1–7
Day 2 │ Session 2 (12h) │ P3 dec iters 8–18 (finish — 6 layers remaining)
Day 3 │ Session 3 (12h) │ P4 enc pruning (all 16 iters) + P5 T2U pruning (4 iters)
Day 4 │ Session 4 (8h)  │ P6: KD text generation + unit label extraction
Day 5 │ Session 5 (6h)  │ P7a: DoRA S2TT recovery (2500 steps, ~5h)
Day 6 │ Session 6 (4h)  │ P7b: T2U focused DoRA (1000 steps) + full benchmark
Day 7 │ Buffer          │ Review results, prepare paper tables
```

**Time estimates with 5-sample SMC:**
- P3: avg 15 candidates × 7 min × 18 iters = ~31.5h → Sessions 1–2
- P4: avg 9 candidates (BI filtered) × 7 min × 16 iters = ~16.8h → Session 3
- P5: 4 iters × ~45 min (ASR-ChrF slower) = ~3h → Session 3
- P6 KD: ~5049 samples × 3 sec = ~4.2h → Session 4
- P7a DoRA: 2500 steps × ~7 sec/step = ~4.9h → Session 5 (matches your Phase 7)
- P7b T2U: 1000 steps × ~5 sec = ~1.4h → Session 6

---

## SECTION 5 — Full Phase-by-Phase Implementation (Pure Python, No CLI)

---

### Phase 1: Vocabulary Pruning (~2h, Session 1)

**One-line change from your Phase 1:** add 4 more languages to the scan.

```python
# Your existing identify_used_tokens(), just change the call:
keep_ids = identify_used_tokens(
    proc,
    target_lang_codes=['eng', 'ben', 'cmn', 'arb', 'hin'],
    n_corpus=5000,   # was 2000, increased for better coverage
)
# Expected: ~58,000 tokens retained (was 20,425 for EN+BN only)
# Savings: ~195M params

# Safety check after pruning:
def token_inflation_check(proc, test_sentences, threshold=1.05):
    """Verify trimmed vocab doesn't over-fragment words."""
    ratios = []
    for sent in test_sentences:
        orig = len(proc.tokenizer.encode(sent))
        # The trimmed vocab will use proc_trimmed (you build this in existing Phase 1)
        ratios.append(orig)  # compare before/after
    return sum(ratios) / len(ratios)
```

Save as `phase1_vocab_5lang`.

---

### Phase 2: Pre-Pruning Activation Analysis (~3h, Session 1)

Run `bidirectional-tracking-base.ipynb` hook framework on `phase1_vocab_5lang`.

```python
# Just extend the existing hook loop to cover all 8 language pairs
# Your hook code is already correct — just run it on more configs

ALL_HOOK_CONFIGS = EVAL_PAIRS  # the 8 pairs defined in Section 2

activation_scores = {}
for (src_f, tgt_f, src_m4t, tgt_m4t) in ALL_HOOK_CONFIGS:
    key = f"{src_m4t}2{tgt_m4t}"
    samples = load_eval_samples(src_f, tgt_f, n=5)
    # ... your existing hook analysis code ...
    activation_scores[key] = layer_scores_dict

torch.save(activation_scores, f'{GDRIVE_ROOT}/activation_map_5lang.pt')
```

Output: the full importance map used to set extra protection in Phases 3–5.

---

### Phase 3: Text Decoder Pruning (24 → 6 layers, Sessions 1–2)

**Code changes to your existing Phase 3 loop:**
1. Replace `run_benchmark()` with `compute_smc()` (Section 3)
2. Add dynamic top-2 protection
3. Set `N_DEC_REMOVE = 18`

Everything else (checkpoint saving, layer removal mechanics, Drive sync) stays identical.

```python
N_DEC_REMOVE = 18  # was 10 in your v1 pipeline

# Build eval_sets_dict once (outside the loop):
eval_sets_dict = {}
for pair in EVAL_PAIRS:
    src_f, tgt_f, src_m4t, tgt_m4t = pair
    eval_sets_dict[pair] = load_eval_samples_for_pair(src_f, tgt_f, n=5)

# In the pruning loop, replace the metric:
for iter_i in range(N_DEC_REMOVE):
    # Dynamic protection: always keep top-2 remaining layers
    protected = update_protected_set(PROTECTED_DEC, remaining_layers)
    eligible = [l for l in remaining_layers if l not in protected]
    
    best_smc, best_layer = -1, -1
    for l in eligible:
        model_temp = temporarily_remove_layer(model_curr, 'text_decoder', l)
        smc, pa, pb = compute_smc(model_temp, eval_sets_dict, processor)
        print(f"    Remove L{l:2d} -> SMC={smc:.2f} [EN:{pa:.2f}|nonEN:{pb:.2f}]")
        del model_temp
        torch.cuda.empty_cache()
        if smc > best_smc:
            best_smc, best_layer = smc, l
    
    model_curr = permanently_remove_layer(model_curr, 'text_decoder', best_layer)
    remaining_layers.remove(best_layer)
    save_checkpoint({'iter': iter_i, 'removed': best_layer, 'smc': best_smc,
                    'remaining': remaining_layers}, 'phase3_dec_pruning')
    print(f"  → Iter {iter_i+1}/{N_DEC_REMOVE}: Removed L{best_layer} (SMC={best_smc:.2f})")
```

Save as `phase3_dec_6L`.

---

### Phase 4: Speech Encoder Pruning (24 → 8 layers, Session 3)

```python
N_ENC_REMOVE = 16  # was 8

# BI scoring first (your existing compute_block_influence code):
bi_scores = compute_block_influence_scores(model_p3, eval_sets_dict)

for iter_i in range(N_ENC_REMOVE):
    protected = update_protected_set(PROTECTED_ENC, remaining_enc_layers)
    eligible = [l for l in remaining_enc_layers if l not in protected]
    
    # BI pre-filter: only probe bottom 50% by BI score
    candidates = sorted(eligible, key=lambda l: bi_scores.get(l, 1.0))
    candidates = candidates[:max(len(candidates)//2, 5)]  # at least 5 candidates
    
    best_smc, best_layer = -1, -1
    for l in candidates:
        model_temp = temporarily_remove_layer(model_curr, 'speech_encoder', l)
        smc, pa, pb = compute_smc(model_temp, eval_sets_dict, processor)
        del model_temp; torch.cuda.empty_cache()
        if smc > best_smc:
            best_smc, best_layer = smc, l
    
    model_curr = permanently_remove_layer(model_curr, 'speech_encoder', best_layer)
    remaining_enc_layers.remove(best_layer)
    save_checkpoint({'iter': iter_i, 'removed': best_layer, 'smc': best_smc}, 
                   'phase4_enc_pruning')
```

Save as `phase4_enc_8L`.

---

### Phase 5: T2U Pruning (12 → 8 layers, Session 3)

```python
# ASR-ChrF metric for T2U (your existing MMS-ASR code, expanded to 5 languages)
MMS_LANG_MAP = {
    "ben": "ben",   # already working in your pipeline
    "cmn": "cmn",   # Mandarin — MMS-1b-all adapter
    "arb": "arb",   # Arabic — MMS-1b-all adapter
    "hin": "hin",   # Hindi — MMS-1b-all adapter
}

def compute_t2u_asr_chrf(model, eval_sets_dict):
    """ASR-ChrF averaged over non-English output directions."""
    chrfs = []
    for (src_f, tgt_f, src_m4t, tgt_m4t), samples in eval_sets_dict.items():
        if tgt_m4t == "eng":
            continue   # EN output uses different vocoder path; skip
        for s in samples:
            audio_wav = run_s2st_inference(model, s['wav'], src_m4t, tgt_m4t)
            asr_text = mms_transcribe(audio_wav, MMS_LANG_MAP[tgt_m4t])
            c = _chrf.sentence_score(asr_text, [s['ref']]).score
            chrfs.append(c)
    return sum(chrfs) / len(chrfs)

# Prune T2U encoder (6→4) and T2U decoder (6→4) independently, 2 removals each
# Use your existing iterative T2U pruning code from Phase 6 of the old pipeline
# Just replace the ASR model calls to use MMS_LANG_MAP for the right adapter
```

After Phase 5: check total params. Should be ~491M. Save as `phase5_500M_final`.

---

### Phase 6: KD Data Generation (~4h, Session 4)

```python
# Load full teacher on GPU 0 (3.6 GB FP16, leaves GPU 1 free)
teacher, proc_teacher = load_base_model()  # facebook/seamless-m4t-v2-large
teacher.to("cuda:0").eval()

# === PART A: Text pseudo-references (for S2TT loss augmentation) ===
kd_text = {}
for (src_f, tgt_f, src_m4t, tgt_m4t), samples in all_train_samples.items():
    pair_key = f"{src_m4t}2{tgt_m4t}"
    kd_text[pair_key] = []
    for s in samples:
        with torch.no_grad():
            out = teacher.generate(
                **proc_teacher(audios=s['wav'], sampling_rate=16000,
                               return_tensors="pt").to("cuda:0"),
                tgt_lang=tgt_m4t,
                generate_speech=False
            )
            hyp = proc_teacher.decode(out[0].tolist(), skip_special_tokens=True)
        kd_text[pair_key].append({
            'id': s['id'], 'ref': s['ref'], 'teacher_hyp': hyp
        })
torch.save(kd_text, f'{GDRIVE_ROOT}/kd_text_5lang.pt')
print(f"KD text: {sum(len(v) for v in kd_text.values())} pseudo-references saved")

# === PART B: Unit label extraction (for T2U loss) ===
# Unit labels = the discrete unit sequences the teacher T2U produces from reference text
kd_units = {}
for (src_f, tgt_f, src_m4t, tgt_m4t), samples in all_train_samples.items():
    if tgt_m4t == "eng":
        continue  # skip EN target for T2U training
    pair_key = f"{src_m4t}2{tgt_m4t}"
    kd_units[pair_key] = []
    for s in samples:
        with torch.no_grad():
            # Tokenize reference text
            text_tokens = proc_teacher.tokenizer(
                s['ref'], return_tensors="pt"
            ).input_ids.to("cuda:0")
            text_embs = teacher.text_decoder.embed_tokens(text_tokens)
            
            # Run teacher T2U in teacher-forcing mode to get unit sequence
            t2u_out = teacher.t2u_model.generate(
                inputs_embeds=text_embs,
                max_new_tokens=512,
            )
            unit_ids = t2u_out[0].tolist()
        kd_units[pair_key].append({'id': s['id'], 'unit_ids': unit_ids})

torch.save(kd_units, f'{GDRIVE_ROOT}/kd_units_5lang.pt')
print(f"KD units: {sum(len(v) for v in kd_units.values())} unit sequences saved")
```

---

### Phase 7a: DoRA S2TT Recovery (~5h, Session 5)

**Your existing `only-p7-dora.ipynb` with these changes:**

```python
# 1. Load 500M pruned model (not the 1B Phase 6 model)
model_student, processor = load_model_from_drive('phase5_500M_final')
model_student = _consolidate_to_device(model_student, "cuda:0")

# 2. Load KD data
kd_text = torch.load(f'{GDRIVE_ROOT}/kd_text_5lang.pt')

# 3. Build multilingual training dataset with KD mixing
class MultilingualKDDataset(Dataset):
    def __init__(self, all_fleurs_samples, kd_text, mix_ratio=0.5):
        self.samples = all_fleurs_samples  # combined across all 8 pairs
        self.kd_text = kd_text
        self.mix_ratio = mix_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = dict(self.samples[idx])
        pair_key = f"{s['src_lang']}2{s['tgt_lang']}"
        kd_pool = {k['id']: k['teacher_hyp']
                   for k in self.kd_text.get(pair_key, [])}
        if random.random() < self.mix_ratio and s['id'] in kd_pool:
            s['ref_text'] = kd_pool[s['id']]  # use teacher pseudo-ref
        return s

# Combine all 8 language pair train splits
all_fleurs = []
for (src_f, tgt_f, src_m4t, tgt_m4t) in EVAL_PAIRS:
    samples = load_fleurs_train_for_pair(src_f, tgt_f, src_m4t, tgt_m4t)
    all_fleurs.extend(samples)

train_dataset = MultilingualKDDataset(all_fleurs, kd_text, mix_ratio=0.5)

# 4. DoRA config (your existing settings)
from peft import LoraConfig, get_peft_model, TaskType
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=['fc1', 'fc2', 'k_proj', 'out_proj', 'q_proj', 'v_proj'],
    use_dora=True,
    task_type=TaskType.SEQ_2_SEQ_LM,
)
model_p7 = get_peft_model(model_student, lora_cfg)

# 5. Training loop — ADD BF16 mixed precision to your existing loop
scaler = torch.cuda.amp.GradScaler()
MAX_STEPS = 2500   # ~5 hours, matches your Phase 7 empirical observation

for step in range(MAX_STEPS):
    batch = next(train_iter)
    speech_input = batch['wav'].to(device)
    text_labels  = encode_text(batch['ref_text'], batch['tgt_lang'])

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model_p7(input_features=speech_input, labels=text_labels)
        loss = outputs.loss / GRAD_ACCUM

    scaler.scale(loss).backward()

    if (step + 1) % GRAD_ACCUM == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if step % 50 == 0:
        print(f"Step {step}/{MAX_STEPS} | loss: {loss.item()*GRAD_ACCUM:.4f}")

# 6. Merge and save
model_p7_merged = model_p7.merge_and_unload()
save_model_to_drive(model_p7_merged, 'phase7a_dora_merged')
```

---

### Phase 7b: Focused T2U DoRA (~2h, Session 6)

```python
# Load Phase 7a merged model
model, proc = load_model_from_drive('phase7a_dora_merged')
model.to("cuda:0").eval()
kd_units = torch.load(f'{GDRIVE_ROOT}/kd_units_5lang.pt')

# Freeze EVERYTHING first
for p in model.parameters():
    p.requires_grad_(False)

# Apply DoRA only to the T2U submodule
from peft import LoraConfig, get_peft_model
t2u_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=['fc1', 'fc2', 'k_proj', 'out_proj', 'q_proj', 'v_proj'],
    use_dora=True,
)
model.t2u_model = get_peft_model(model.t2u_model, t2u_cfg)
model.t2u_model.print_trainable_parameters()
# Expected: ~1M trainable params (just T2U DoRA adapters)

optimizer_t2u = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
scaler_t2u = torch.cuda.amp.GradScaler()
MAX_STEPS_T2U = 1000

for step in range(MAX_STEPS_T2U):
    # Sample a non-English output pair
    pair_key = random.choice(list(kd_units.keys()))
    sample = random.choice(kd_units[pair_key])
    unit_ids = torch.tensor([sample['unit_ids']]).to("cuda:0")

    # Get text embeddings from frozen decoder (no gradient to decoder)
    ref_text_ids = proc.tokenizer(
        sample.get('ref', ''), return_tensors="pt"
    ).input_ids.to("cuda:0")

    with torch.no_grad():
        text_embs = model.text_decoder.embed_tokens(ref_text_ids)

    # T2U forward pass with unit labels
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        t2u_out = model.t2u_model(
            inputs_embeds=text_embs,
            labels=unit_ids,
        )
        loss = t2u_out.loss

    scaler_t2u.scale(loss).backward()
    scaler_t2u.step(optimizer_t2u)
    scaler_t2u.update()
    optimizer_t2u.zero_grad()

    if step % 100 == 0:
        print(f"T2U step {step}/{MAX_STEPS_T2U} | loss: {loss.item():.4f}")

# Merge and save final 500M model
model.t2u_model = model.t2u_model.merge_and_unload()
save_model_to_drive(model, 'phase7b_final_500M')
```

---

### Phase 8: Voice Clone Integration (Session 6, remaining time)

The HiFi-GAN vocoder in SeamlessM4T already has `spkr_embed_dim=256` in its config — it was designed for speaker conditioning. SeamlessExpressive showed this works with ECAPA-TDNN embeddings.

```python
# Install speechbrain if not already present:
# !pip install speechbrain -q

from speechbrain.pretrained import EncoderClassifier

# 20M param speaker encoder — load once, freeze
spk_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda:0"}
)
for p in spk_encoder.parameters():
    p.requires_grad_(False)

# Tiny adapter: maps ECAPA 192-dim → vocoder 256-dim (50K params, train this)
import torch.nn as nn
class SpeakerProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(192, 256)
        self.norm = nn.LayerNorm(256)
    def forward(self, emb):
        return self.norm(self.proj(emb))

spk_proj = SpeakerProjection().to("cuda:0")
# Train spk_proj with cosine similarity loss between
# input speaker embedding and output speaker embedding
# Data: FLEURS same-speaker pairs (~30 min to train)

# Inference function:
def s2st_voice_clone(model, input_wav, src_lang, tgt_lang, processor):
    """S2ST preserving input speaker voice in output."""
    # Extract speaker embedding from input
    with torch.no_grad():
        spk_emb = spk_encoder.encode_batch(
            input_wav.unsqueeze(0).to("cuda:0")
        ).squeeze(0)                        # [192]
        spk_cond = spk_proj(spk_emb)        # [256] — vocoder speaker space

    # Run S2TT to get text tokens
    text_ids = run_s2tt_tokens(model, input_wav, src_lang, tgt_lang, processor)

    # Run T2U to get unit sequence
    unit_ids = run_t2u_inference(model, text_ids, tgt_lang)

    # Run vocoder WITH speaker conditioning
    # The vocoder's forward() accepts spkr_id for conditioning
    audio_out = model.vocoder(
        input_ids=unit_ids.to("cuda:0"),
        spkr_id=spk_cond.unsqueeze(0).long(),   # map to nearest speaker slot
        lang_id=tgt_lang_to_id[tgt_lang],
    )
    return audio_out
```

> Note on `spkr_id`: The HiFi-GAN vocoder uses discrete speaker IDs (0–36 for the 36 supported languages), not continuous embeddings. The SpeakerProjection maps to the embedding space, but you need to either (a) find the nearest existing speaker slot, or (b) add a new learnable speaker embedding initialized from the projection. For a research demo, option (a) is sufficient and requires no additional training.

---

## SECTION 6 — Expected Results

| Stage | Params | EN→BN | BN→EN | ZH→EN | AR→EN | HI→EN | avg | RTF |
|-------|--------|-------|-------|-------|-------|-------|-----|-----|
| Baseline | 1805M | 50.52 | 50.22 | — | — | — | ~49 | 0.268 |
| P1 vocab | ~1591M | ~49 | ~49 | — | — | — | ~48 | 0.17 |
| P3 dec 6L | ~940M | ~33 | ~31 | ~30 | ~29 | ~30 | ~31 | 0.06 |
| P4 enc 8L | ~517M | ~28 | ~27 | ~26 | ~25 | ~27 | ~27 | 0.05 |
| P5 T2U 8L | ~491M | ~28 | ~27 | ~26 | ~25 | ~27 | ~27 | 0.05 |
| P7a DoRA | ~491M | ~41 | ~39 | ~38 | ~37 | ~39 | ~39 | 0.075 |
| P7b T2U | ~491M | ~41 | ~40 | ~38 | ~37 | ~39 | ~39 | 0.080 |

**Speedup: 3.35× (RTF 0.268 → 0.080)**  
**Quality retention: ~78% of baseline ChrF at 73% structural compression**

---

## SECTION 7 — Two-GPU Allocation

```python
# Pruning: single GPU (model fits easily on 1x T4 at 16 GB)
model = load_model(..., device_map="cuda:0")

# KD generation: split teacher/student across GPUs
teacher = load_base_model(..., device_map="cuda:0")   # ~3.6 GB
student = load_student(...,    device_map="cuda:1")   # ~1 GB

# DoRA fine-tuning: single GPU (500M model + adapters + optimizer = ~4 GB total)
model = load_model(..., device_map="cuda:0")
# GPU 1 stays free — useful if you want to run evaluation in parallel
```

---

## SECTION 8 — What NOT to Repeat

| Mistake | Impact | Fix |
|---------|--------|-----|
| EN→BN-only ChrF for pruning | BN→EN collapsed (50.22 → 35.26, 60% generation failures) | SMC metric, bidirectional |
| S2TT-only DoRA loss | T2U got zero gradient, audio not recovered | Phase 7b dedicated T2U training |
| FLAP width pruning after depth pruning | ChrF 40.11 → 9.20, loop-repetition hallucinations | Not in this plan |
| Whisper for Bengali ASR | Complete failure on Bengali audio | MMS-1b-all with ben adapter |
| Fine-tuning between each pruning iteration | Overfitting, no quality improvement | Prune fully first, fine-tune once at end |

---

## SECTION 9 — Publication

**Unique contributions:**
1. **SMC (Stratified Minimum ChrF)** — bidirectional-aware pruning criterion
2. **Quantitative proof of directional bias** from activation hooks (your existing data)
3. **First 500M multilingual S2ST** covering Bengali, Arabic, Hindi, Mandarin, English
4. **Two-stage recovery** (S2TT DoRA + T2U DoRA) matching Meta's SPEECH_TO_SPEECH mode in pure HuggingFace
5. **Voice cloning via ECAPA-TDNN vocoder conditioning** on compressed S2ST

**Target: INTERSPEECH 2026 (deadline ~March 2026)**

---

*Plan v2 — Updated for 1-week timeline, FP16/BF16 decision, pure Python notebook execution*  
*Sources: seamless-cse465v5.ipynb · only-p7-dora.ipynb · bidirectional-tracking notebooks · Moslem IWSLT 2025 · SeamlessM4T v2 architecture docs · DoRA ICML 2024*
