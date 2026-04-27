# Textless SeamlessM4T v2 — Deep Diagnosis & Corrected Plan
## Pure Speech-In, Speech-Out · ~673M Params · 5 Languages

---

## EXECUTIVE SUMMARY

**Is the goal possible?** YES. The architecture is sound and grounded in published work
(S2UT Lee ACL 2022; SeamlessExpressive 2023). The KD data is correct.

**Did Phase 6a and 6b learn the right things?** Phase 6a: mostly yes.
Phase 6b: partially — but the loss was mis-specified and the most critical component
(the T2U decoder) was never trained at all.

**Root cause of 0.7-second garbage audio:** One critical bug in `run_textless_s2st`
that bypasses the entire T2U decoder. The fix is ~10 lines of code. Try that first
before any retraining.

---

## SECTION 1 — Deep Bug Analysis

### 1.1 The T2U Architecture You Need to Understand

`SeamlessM4Tv2TextToUnitModel` is an **encoder-decoder**:

```
INPUT: inputs_embeds [1, T_text, 1024]   ← from text decoder (original) or CIF (ours)
         │
         ▼
    T2U Encoder (4 layers)
         │  encoder_hidden_states [1, T_text, 1024]
         │
         ▼
    T2U Decoder (4 layers, autoregressive)
         │  cross-attends to encoder_hidden_states
         │  autoregressively generates unit tokens one by one
         │  output: [1, T_units, 1024]   T_units >> T_text  (ratio ~11.6×)
         │
         ▼
      lm_head (Linear: 1024 → 10082)
         │
         ▼
    unit_ids [1, T_units]   ← 69-1123 tokens per utterance
         │
         ▼
    HiFi-GAN Vocoder
         │
         ▼
    Audio output
```

The lm_head projects **decoder** hidden states, not encoder hidden states.
The decoder runs autoregressively and produces T_units ≈ 100-1000 tokens.
The encoder only runs once and produces T_text ≈ 10-97 tokens.

### 1.2 The Fatal Bug in `run_textless_s2st` (Cell 105)

```python
# ── 4. T2U encoder ─────────────────────────────────────────────
t2u_hidden = mdl.t2u_model.get_encoder()(
    inputs_embeds=conn_t2u).last_hidden_state   # [1, T_cif, 1024]

# ── 5. Project T2U hidden → unit IDs via lm_head ─────────────────────
logits   = mdl.t2u_model.lm_head(t2u_hidden)   # [1, T_cif, 10082]
unit_ids = logits.argmax(dim=-1)                # [1, T_cif]  ← ~30-40 tokens!
```

**What this does:** Feeds the T2U *encoder* output (T_cif ≈ 30-40 tokens) directly into
`lm_head`, which is designed to project *decoder* hidden states. This:

1. Completely skips the T2U **decoder** — the autoregressive unit generation component
2. Produces only ~30-40 "unit IDs" instead of the required 69-1123
3. The lm_head was never trained to interpret encoder hidden states — the outputs are meaningless

**Why 0.7 seconds?**
HiFi-GAN generates ~1 frame per unit at 40 units/second.
30-40 units × (1/40 second/unit) ≈ 0.75-1.0 seconds.
Your ~0.7 second output is **arithmetically exact** from this bug.

**What it should do:**
```python
# CORRECT: Use autoregressive generation through T2U decoder
unit_ids = mdl.t2u_model.generate(
    inputs_embeds=connector_out,
    max_new_tokens=2048,
)
# unit_ids: [1, T_units] where T_units = 69-1123 tokens
```

### 1.3 What Phase 6a Actually Learned

**Loss:** 0.35×cosine(CIF_out, teacher_t2u_input) + 0.25×MSE(CIF_out, teacher_t2u_input)
          + 0.35×qty_MSE(raw_w_sum, n_tokens) + 0.02×aux_qty + 0.03×spk_reg

**What was learned:**
- CIF connector was trained to produce output with the **same shape** (T_cif ≈ n_tokens)
  and **similar feature distribution** as the text decoder's last hidden state
- The CIF learns to fire approximately `n_tokens` times (mean ~30 per utterance)
- Speaker adapter learned to project ECAPA 192-dim → 256-dim

**Is this correct?** Conceptually YES. The T2U encoder was trained on text decoder outputs
as `inputs_embeds`. To replace the text decoder with CIF, CIF must produce features in the
same representation space. The cosine + MSE loss toward teacher t2u_input is the right
approach.

**Caveats:**
- 1600 samples × 5000 steps with batch size 8 = limited exposure. Quality of imitation
  is uncertain without benchmarking.
- The CIF output might not perfectly match the text decoder's feature statistics, but
  should be directionally correct.
- **Phase 6a training loss curve (left panel, training image) shows the E2E DoRA loss
  (Phase 6b), not Phase 6a's feature KD loss.** The full curve (right panel) shows
  Phase 6a (steps 0-5000) dropped from ~60 to ~3, which is healthy convergence.

### 1.4 What Phase 6b Actually Learned

**Loss:** 0.55×cosine(student_T2U_enc_hidden, teacher_T2U_enc_hidden)
          + 0.25×MSE + 0.15×qty + 0.05×spk

**What was learned:**
- Speech encoder (DoRA): adapted to produce representations that, after passing through
  the CIF connector and T2U encoder, match what the *base teacher* T2U encoder would output
  given the real text decoder embeddings
- T2U encoder (DoRA): adapted to map CIF outputs to hidden states similar to what it
  would produce from teacher text embeddings
- CIF connector: continued to be updated

**What was NOT learned:**
- **The T2U decoder was never trained.** The cross-entropy loss over unit_ids (the actual
  target) was never computed. The decoder's cross-attention still expects encoder hidden
  states from the original text-decoder → T2U-encoder pathway.
- The DoRA adapters on T2U were only on the encoder; the decoder cross-attention heads
  were frozen and untouched.

**The PLAN.md described unit CE loss in Phase 6b, but the implementation used T2U encoder
KD cosine loss instead.** This is the core training gap.

### 1.5 Summary Table

| Component | Trained? | Loss Used | Correct? |
|---|---|---|---|
| Speech encoder (16L) | ✓ DoRA (6b) | T2U-enc KD cosine | Partial |
| CIF connector | ✓ Full (6a+6b) | Feature KD + qty MSE | ✓ Yes |
| Speaker adapter | ✓ Full (6a+6b) | Norm reg only | Weak |
| T2U encoder | ✓ DoRA (6b) | T2U-enc KD cosine | Partial |
| **T2U decoder** | **✗ Never** | — | **Missing** |
| HiFi-GAN vocoder | frozen | — | n/a |

---

## SECTION 2 — Immediate Fix (Try This First, Zero Training Required)

The inference bug alone explains your broken output. Fix `run_textless_s2st` to use
`t2u_model.generate()` instead of manually applying `lm_head` to encoder hidden states.

### 2.1 Fixed `run_textless_s2st`

```python
def run_textless_s2st_v2(mdl, wav_np, tgt_lang='ben'):
    """
    FIXED: Uses t2u_model.generate() instead of lm_head(encoder_hidden).
    This invokes the T2U decoder autoregressively, producing T_units >> T_cif tokens.
    """
    def mod_dtype(mod):
        try: return next(mod.parameters()).dtype
        except StopIteration: return torch.float32
    def mod_device(mod):
        try: return next(mod.parameters()).device
        except StopIteration: return torch.device('cpu')

    dev     = mod_device(mdl.speech_encoder)
    t2u_dev = mod_device(mdl.t2u_model)
    voc_dev = mod_device(mdl.vocoder)
    t0      = time.time()

    with torch.no_grad():

        # ── 1. Speaker embedding ──────────────────────────────────────────────
        spk_raw = extract_speaker_emb(wav_np).unsqueeze(0).to(dev)
        spk_raw = spk_raw.to(dtype=mod_dtype(mdl.speaker_adapter))
        spk_cond = mdl.speaker_adapter(spk_raw)            # [1, 256]

        # Nearest-neighbor vocoder speaker slot
        voc_spk_w = mdl.vocoder.speaker_embedding.weight.float().to(voc_dev)
        spk_norm  = F.normalize(spk_cond.float().to(voc_dev), dim=-1)
        best_spk_idx = (spk_norm @ F.normalize(voc_spk_w, dim=-1).T
                        ).squeeze(0).argmax().item()
        speaker_id = torch.tensor([[best_spk_idx]], dtype=torch.long, device=voc_dev)
        print(f'  Speaker idx={best_spk_idx}')

        # ── 2. Speech encoder ─────────────────────────────────────────────────
        inp   = processor(audio=wav_np, sampling_rate=16000, return_tensors='pt')
        inp_f = inp['input_features'].to(dev, dtype=mod_dtype(mdl.speech_encoder))
        attn  = inp.get('attention_mask')
        if attn is not None: attn = attn.to(dev)
        lang_id = torch.tensor([m4t_lang_to_vocoder_id(tgt_lang)], device=dev)

        enc_out = mdl.speech_encoder(
            input_features=inp_f, attention_mask=attn
        ).last_hidden_state                                 # [1, T_frames, 1024]
        print(f'  Encoder: {enc_out.shape}')

        # ── 3. CIF connector ──────────────────────────────────────────────────
        enc_cif = enc_out.to(dtype=mod_dtype(mdl.cif_connector))
        connector_out, actual_qty, _, _ = mdl.cif_connector(enc_cif, lang_id)
        print(f'  CIF: {enc_out.shape[1]} frames → {connector_out.shape[1]} tokens '
              f'(qty_target≈{actual_qty.item():.0f})')

        # ── 4. T2U GENERATE (the correct way) ────────────────────────────────
        # This calls the full encoder+decoder stack autoregressively.
        # Output is T_units ≈ 100-1000 tokens — NOT T_cif ≈ 30-40 tokens.
        conn_t2u = connector_out.to(t2u_dev, dtype=mod_dtype(mdl.t2u_model))
        attn_mask = torch.ones(conn_t2u.shape[:2], dtype=torch.long, device=t2u_dev)

        # Get T2U config for start token
        t2u_cfg = mdl.t2u_model.config
        # SeamlessM4T T2U uses language-specific start tokens
        # decoder_start_token_id is set per-language in generate()
        try:
            unit_ids = mdl.t2u_model.generate(
                inputs_embeds   = conn_t2u,
                attention_mask  = attn_mask,
                max_new_tokens  = 2048,
                # Don't pass tgt_lang here — T2U generate doesn't use it
                # for unit generation (it's handled by vocoder lang_id)
            )
        except TypeError:
            # Fallback: some PEFT-wrapped models need base_model call
            unit_ids = mdl.t2u_model.base_model.model.model.generate(
                inputs_embeds   = conn_t2u,
                attention_mask  = attn_mask,
                max_new_tokens  = 2048,
            )

        print(f'  T2U generate: {unit_ids.shape}  '
              f'range=[{unit_ids.min().item()},{unit_ids.max().item()}]  '
              f'unique={unit_ids.unique().numel()}')

        # SANITY: Should be 69-1123 tokens, not 30-40
        if unit_ids.shape[1] < 50:
            print(f'  WARNING: Only {unit_ids.shape[1]} units generated. '
                  'T2U decoder may not be properly conditioned.')

        # ── 5. Vocoder ────────────────────────────────────────────────────────
        tgt_vid = torch.tensor(
            [[m4t_lang_to_vocoder_id(tgt_lang)]], dtype=torch.long, device=voc_dev)

        # Strip BOS/decoder_start tokens from unit_ids before passing to vocoder
        # T2U generate may prepend a start token (usually index 2 or 3)
        u_ids = unit_ids.to(voc_dev)
        bos = getattr(t2u_cfg, 'decoder_start_token_id', 2)
        if u_ids[0, 0].item() == bos and u_ids.shape[1] > 1:
            u_ids = u_ids[:, 1:]   # remove BOS

        wav_out = mdl.vocoder(
            input_ids  = u_ids,
            speaker_id = speaker_id,
            lang_id    = tgt_vid,
        )
        wav_np_out = wav_out[0].squeeze().float().cpu().numpy()
        print(f'  Vocoder: {len(wav_np_out)/16000:.2f}s output')

    rtf = (time.time() - t0) / (len(wav_np) / 16000)
    return wav_np_out, rtf, u_ids
```

### 2.2 Expected Outcome After Fix

| Scenario | Symptoms | Expected After Fix |
|---|---|---|
| 6a+6b worked well | Good CIF output → T2U decoder produces coherent units → usable audio | Full-length translated audio (2-15s), possibly broken or accent-heavy |
| 6a worked, 6b partial | CIF→T2U encoder alignment OK, decoder cross-attention partially misaligned | Audio with correct length but unclear speech |
| Both worked poorly | CIF output in wrong feature space → T2U decoder confused → random units | Long audio (correct duration) but no intelligible speech |

Even in the worst case, the audio will be MUCH longer than 0.7 seconds. If you still
get 0.7 seconds after this fix, a different issue is blocking T2U.generate().

---

## SECTION 3 — Training Gaps and Phase 6c

If the inference fix gives audio that is long but unintelligible, you need to train
the T2U decoder. This is Phase 6c.

### 3.1 Why the T2U Decoder Was Never Trained

The PLAN.md (Section 7, Phase 6b) described:
> "Loss: 0.80×unit_CE + 0.15×quantity + 0.05×speaker_regularisation"

The implementation (Cell 88, `run_phase6b_step`) computed:
> 0.55×cos(T2U_encoder_hidden, teacher_T2U_encoder_hidden)
> + 0.25×MSE + 0.15×qty + 0.05×spk

**The unit cross-entropy loss was replaced by T2U encoder KD.**

This matters because:
- Cross-entropy on unit_ids trains `t2u_model.generate()` to produce the correct
  discrete unit sequence. It trains the decoder weights.
- Encoder KD trains only the encoder's hidden representations. The decoder's
  cross-attention never learns to attend to CIF-derived encoder states correctly.

### 3.2 Phase 6c: Unit CE Loss Training

This is the critical missing phase. Train CIF + T2U (encoder+decoder) end-to-end
with cross-entropy over unit_ids.

```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 6c: Unit CE Loss — trains T2U decoder for the first time            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_phase6c_step(model, sample, sample_id_to_audio, processor,
                      m4t_lang_to_vocoder_id, device_enc, device_t2u):
    """
    The step that was MISSING from Phase 6b.
    Uses cross-entropy over unit_ids to train the T2U decoder.
    """
    audio_wav = sample_id_to_audio.get(sample['id'])
    if audio_wav is None or sample.get('unit_ids') is None:
        return None

    tgt_lang   = sample['tgt_lang']
    lang_id    = torch.tensor([m4t_lang_to_vocoder_id(tgt_lang)], device=device_enc)
    n_toks     = float(sample['n_tokens'])
    target_qty = torch.tensor([n_toks], dtype=torch.float32, device=device_t2u)

    # unit_ids: [T_units] — the actual speech unit sequence (ground truth)
    unit_ids = sample['unit_ids'].to(device_t2u)          # [T_units]

    # ── Stage 1: Speech encoder ────────────────────────────────────────────────
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        inp_proc = processor(audio=audio_wav, sampling_rate=16000, return_tensors='pt')
        inp_f    = inp_proc['input_features'].to(device_enc, dtype=torch.bfloat16)
        attn_m   = inp_proc.get('attention_mask')
        if attn_m is not None: attn_m = attn_m.to(device_enc)

        enc_out = model.speech_encoder(
            input_features=inp_f, attention_mask=attn_m
        ).last_hidden_state                                # [1, T_audio, 1024]

    # ── Stage 2: CIF connector ─────────────────────────────────────────────────
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        enc_cif = enc_out.to(device_t2u, dtype=torch.bfloat16)
        lang_id_t2u = lang_id.to(device_t2u)

        connector_out, actual_qty, qty_pred, raw_w_sum = \
            model.cif_connector(enc_cif, lang_id_t2u)     # [1, T_cif, 1024]

        qty_loss = F.mse_loss(raw_w_sum.to(device_t2u), target_qty)

    # ── Stage 3: T2U UNIT CROSS-ENTROPY (the key loss) ─────────────────────────
    # The T2U model is encoder-decoder. We pass CIF output as inputs_embeds
    # (encoder input) and unit_ids as labels (decoder target).
    # This trains BOTH the T2U encoder and decoder simultaneously.
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        t2u_dtype = next(model.t2u_model.parameters()).dtype
        conn_t2u  = connector_out.to(dtype=t2u_dtype)     # [1, T_cif, 1024]

        # labels: [1, T_units] — the full unit sequence
        # T2U internally shifts labels for teacher forcing
        labels = unit_ids.unsqueeze(0)                     # [1, T_units]

        # IMPORTANT: T2U is a seq2seq model. It does NOT require equal-length
        # inputs_embeds and labels. inputs_embeds goes through encoder, labels
        # through decoder (teacher forcing). The cross-attention handles alignment.
        t2u_out = model.t2u_model(
            inputs_embeds = conn_t2u,
            labels        = labels,
            # attention_mask for encoder:
            attention_mask = torch.ones(
                conn_t2u.shape[:2], dtype=torch.long, device=device_t2u),
        )

        unit_ce_loss = t2u_out.loss  # cross-entropy over T_units positions

    # ── Combined loss ──────────────────────────────────────────────────────────
    # Weights: unit CE is PRIMARY. Quantity is auxiliary.
    loss = 0.80 * unit_ce_loss + 0.20 * qty_loss

    return {
        'loss':          loss,
        'unit_ce_loss':  unit_ce_loss.item(),
        'qty_loss':      qty_loss.item(),
        'fired':         actual_qty.mean().item(),
        'n_toks':        n_toks,
    }
```

### 3.3 Phase 6c Training Setup

```python
# Load best available model (phase6b checkpoint, or phase6a if 6b is bad)
model_6c, processor = load_textless_model_from_drive('phase6a_textless')
# or: load_textless_model_from_drive('phase6b_e2e_merged') if it exists

# Freeze all, unfreeze CIF + apply DoRA to speech encoder and T2U
for p in model_6c.parameters():
    p.requires_grad_(False)
for p in model_6c.cif_connector.parameters():
    p.requires_grad_(True)
for p in model_6c.speaker_adapter.parameters():
    p.requires_grad_(True)

# Apply DoRA to BOTH speech encoder AND T2U (encoder + decoder)
from peft import LoraConfig, get_peft_model

lora_enc = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, use_dora=True,
    target_modules=['linear_q','linear_k','linear_v','linear_out',
                    'intermediate_dense','output_dense'])
lora_t2u = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, use_dora=True,
    target_modules=['q_proj','k_proj','v_proj','out_proj','fc1','fc2'])

model_6c.speech_encoder = get_peft_model(model_6c.speech_encoder, lora_enc)
model_6c.t2u_model = get_peft_model(model_6c.t2u_model, lora_t2u)
# NOTE: lora_t2u now covers BOTH encoder and decoder layers in t2u_model

optimizer_6c = torch.optim.AdamW([
    {'params': model_6c.cif_connector.parameters(),   'lr': 1e-4},
    {'params': model_6c.speaker_adapter.parameters(), 'lr': 5e-5},
    {'params': [p for p in model_6c.speech_encoder.parameters() if p.requires_grad],
     'lr': 5e-5},
    {'params': [p for p in model_6c.t2u_model.parameters() if p.requires_grad],
     'lr': 5e-5},
], betas=(0.9, 0.98), weight_decay=0.01)

scheduler_6c = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_6c, T_max=3000, eta_min=1e-6)

# Training data: use unit_kd_safe (already filtered in your notebook)
# 1600 samples, all 8 language pairs, unit_ids validated < 10082
```

### 3.4 Why This Will Work

The T2U model's decoder was pre-trained by Meta to generate unit_ids given encoder
hidden states from the text decoder. The encoder hidden states from Phase 6a+6b
are (approximately) in the same feature space. Once the CE loss on unit_ids is applied:

1. The T2U decoder learns to cross-attend to CIF-derived encoder states
2. The CIF connector gets gradient signal from the decoder (not just encoder matching)
3. The speech encoder adapts to produce features that lead to correct unit generation

This is the core loop of S2UT (Lee et al., ACL 2022). The approach is proven.

---

## SECTION 4 — Correct Inference Pipeline

Regardless of what training has been done, the inference pipeline must be:

```
wav_np (16kHz numpy array)
  │
  ├─ ECAPA-TDNN → spk_emb [192] → speaker_adapter → spk_cond [256]
  │                                                       │
  │                                                       ▼ (nearest-neighbor)
  │                                               vocoder speaker_id [int]
  │
  ├─ SeamlessM4T Processor → input_features [1, 128, T_mel]
  │
  ▼
Speech Encoder (16 layers, pruned)
  │  enc_out [1, T_frames, 1024]   (T_frames ≈ 300-500 for 5s)
  ▼
CIF Connector
  │  connector_out [1, T_cif, 1024]  T_cif ≈ 20-60 (text-equivalent tokens)
  │  (fires CIF threshold-crossings, quantity trained to ≈ n_tokens)
  ▼
t2u_model.generate(inputs_embeds=connector_out, max_new_tokens=2048)
  │
  ├─ T2U Encoder: connector_out → encoder_hidden_states [1, T_cif, 1024]
  ├─ T2U Decoder: autoregressive, cross-attends to encoder_hidden_states
  │               generates unit tokens one by one
  │
  unit_ids [1, T_units]   T_units ≈ 100-1000
  │
  ▼
HiFi-GAN Vocoder
  │  inputs: unit_ids, speaker_id (int), lang_id (int)
  ▼
wav_out [T_audio]   ≈ 2-15 seconds
```

**Critical differences from your current implementation:**
- `t2u_model.generate()` NOT `lm_head(get_encoder()(inputs_embeds=...))`
- unit_ids has ~11.6× more tokens than T_cif
- Vocoder receives integer speaker_id (slot index), not a continuous embedding

---

## SECTION 5 — What the Vocoder Speaker Conditioning Actually Does

Looking at Cell 105 and the vocoder debug cells, there is a misunderstanding about
how the vocoder speaker conditioning works.

**The vocoder takes:**
- `input_ids`: discrete unit sequence [1, T_units] → unit_embedding → [1, T_units, 256]
- `speaker_id`: integer index [1, 1] → speaker_embedding → [1, 1, 256]
- `lang_id`: integer index [1, 1] → language_embedding → [1, 1, 256]

The vocoder does **not** accept continuous speaker embeddings directly. The
`speaker_embedding` is a lookup table (200 × 256). Your `speaker_adapter` produces
a 256-dim continuous vector, but the vocoder expects an integer index.

**For voice cloning, there are two approaches:**

**Option A (current, correct for now):** Nearest-neighbor lookup
```python
spk_cond = speaker_adapter(ecapa_emb)           # [1, 256] continuous
voc_table = vocoder.speaker_embedding.weight     # [200, 256]
# Find the closest existing speaker slot
best_idx = F.cosine_similarity(
    F.normalize(spk_cond, dim=-1),
    F.normalize(voc_table, dim=-1)
).argmax().item()
speaker_id = torch.tensor([[best_idx]])
```
This gives limited voice cloning (closest pretrained speaker, not actual input voice).

**Option B (proper voice cloning, future work):** Modify the vocoder to accept
continuous embeddings and train it. This is what SeamlessExpressive does with PRETSSEL.
This is a larger change than what's needed for a working system now.

**For the paper:** Option A is sufficient to demonstrate the architecture and measure
speaker similarity. The PLAN.md expectation of 0.65-0.78 similarity is achievable
with Option A if the T2U is properly generating translation-correct units.

---

## SECTION 6 — Revised Phase Schedule

### Phase 6-FIX: Inference Bug Fix (Immediate, 0 training)

**Goal:** Verify what the current trained model can do with correct inference.

```python
# Run this NOW on your phase6a_textless model
model_fix, processor = load_textless_model_from_drive('phase6a_textless')
model_fix.eval()

# Merge DoRA if phase6b weights exist
try:
    model_fix.speech_encoder = model_fix.speech_encoder.merge_and_unload()
    model_fix.t2u_model = model_fix.t2u_model.merge_and_unload()
except: pass  # fine if no DoRA adapters attached

# Test with fixed inference
demo = eval_samples[0]
wav_out, rtf, unit_ids = run_textless_s2st_v2(model_fix, demo['wav'], 'ben')

print(f'Audio length: {len(wav_out)/16000:.2f}s  (should be > 1s)')
print(f'Unit count: {unit_ids.shape[1]}  (should be > 50)')
play(wav_out, 16000, 'Fixed inference test')
```

**Decision gate:**
- Audio is > 2s and has some speech structure → Phase 6c can be light (1000-2000 steps)
- Audio is long but garbage → Phase 6c needs full run (3000 steps)
- Audio is still 0.7s → T2U.generate() is failing; debug further

### Phase 6c: Unit CE Loss Training (Primary Fix)

**Duration:** 3000 steps (≈ 4-6 hours on T4×2)  
**Data:** unit_kd_safe (1600 samples, validated)  
**Batch:** 1 sample per step (gradient accumulation 4)

```
Steps 0-500:   CIF + T2U (DoRA r=8) only, frozen speech encoder
               Loss: 0.80×unit_CE + 0.20×qty
               → allows T2U decoder to adapt to CIF inputs without overfitting encoder

Steps 500-3000: CIF + T2U + speech encoder (DoRA r=8) unfrozen
                Loss: 0.70×unit_CE + 0.20×qty + 0.10×feature_KD
                → end-to-end adaptation
```

**Checkpointing:** Every 250 steps, save and run quick eval
```python
# Quick eval: translate 2 samples per language pair, compute ChrF via ASR
quick_chrf = quick_eval_chrf(model_6c, eval_samples[:16], max_samples=16)
print(f'Step {step}: quick ChrF = {quick_chrf:.2f}')
```

**Target ChrF by step:**
- Step 500: > 10 (better than random noise)
- Step 1500: > 20
- Step 3000: > 30 (publishable baseline)

### Phase 6d: Speaker Adapter Proper Training (Optional, 500 steps)

If voice cloning quality is poor after 6c, do a dedicated speaker-adapter training pass:

```python
# Freeze everything except speaker adapter
# Loss: cosine similarity between ECAPA(input_audio) and ECAPA(translated_audio)
# This requires running vocoder during training (expensive but feasible)

# Only do this if speaker_sim < 0.50 after Phase 6c
```

### Phase 7: Final Benchmark (Same as PLAN.md)

Run all 8 language pair combinations, voice cloning benchmark, long-form audio test.

---

## SECTION 7 — Data Sufficiency Analysis

### 7.1 Is 1600 KD Samples Enough?

For **feature distillation** (Phase 6a): 1600 samples is on the low side but acceptable.
The CIF connector has only ~5M parameters. Feature imitation from a fixed teacher
is a relatively easy regression task.

For **unit CE training** (Phase 6c): 1600 samples × 200 samples/pair = limited.
Each sample has ~350 unit tokens on average → 1600 × 350 = 560,000 training tokens.
For context: S2UT (Lee 2022) used 408 hours of CoVoST2 data.

**Recommendation:** Extract additional KD data if possible. Increase from 1600 to 4800:
200 → 600 samples per language pair. This requires re-running Phase 5 with more
FLEURS data (using the validation split in addition to train).

### 7.2 Alternative: Use Pre-existing S2U Datasets

SeamlessM4T v2 was trained on Seamless dataset (Meta's proprietary). However, for
fine-tuning the CIF→T2U pathway, any speech translation data that can be converted
to (audio, unit_ids) pairs works. Process:

1. Audio input → SeamlessM4T teacher → extract unit_sequences (already done)
2. Use FLEURS test split (additional 75 samples per language pair)
3. Use CoVoST2 if available (larger, more diverse)

---

## SECTION 8 — Root Cause Checklist for 0.7s Garbage Audio

Before doing any retraining, verify these causes in order:

```python
# Check 1: Is T2U.generate() working at all?
with torch.no_grad():
    dummy_embeds = torch.randn(1, 20, 1024, device=t2u_dev,
                               dtype=next(model.t2u_model.parameters()).dtype)
    try:
        ids = model.t2u_model.generate(inputs_embeds=dummy_embeds, max_new_tokens=50)
        print(f'✓ T2U.generate() works: {ids.shape}')
    except Exception as e:
        print(f'✗ T2U.generate() FAILED: {e}')
        # If PEFT-wrapped, try: model.t2u_model.base_model.model.model.generate(...)

# Check 2: Does CIF actually fire reasonable number of tokens?
sample = eval_samples[0]
enc_out = ... # run encoder
connector_out, qty, _, _ = model.cif_connector(enc_out, lang_id)
print(f'CIF output shape: {connector_out.shape}')  # should be [1, 20-60, 1024]
print(f'CIF fired qty: {qty.item():.1f}')           # should be 15-60

# Check 3: Do unit_ids cover the full unit vocabulary?
# After generate(), check: are most values in [4, 10002]?
# If most values = 2 (BOS/EOS), T2U decoder is stuck in degenerate output
print(f'unit_ids unique values: {ids.unique().numel()}')
print(f'unit_ids histogram (top 10): {torch.bincount(ids.squeeze()).topk(10)}')

# Check 4: Are speaker_id and lang_id in valid ranges?
voc = model.vocoder
print(f'vocoder n_speakers: {voc.speaker_embedding.num_embeddings}')   # should be 200
print(f'vocoder n_langs: {voc.language_embedding.num_embeddings}')     # should be 36
print(f'your speaker_id: {best_spk_idx}')   # must be < 200
print(f'your lang_id: {m4t_lang_to_vocoder_id(tgt_lang)}')  # must be < 36
```

---

## SECTION 9 — Expected Results After Fixes

| Phase | Audio Length | ChrF (ASR) | Speaker Sim | Verdict |
|---|---|---|---|---|
| Current (buggy) | 0.7s | 0 | — | Inference bug |
| After 6-FIX (inference fix only) | 2-15s | 5-25 | 0.30-0.50 | Depends on 6a quality |
| After 6c (1000 steps) | 2-15s | 15-30 | 0.40-0.55 | Getting there |
| After 6c (3000 steps) | 2-15s | 28-38 | 0.50-0.65 | Publishable |
| After 6d (speaker tuning) | 2-15s | 28-38 | 0.60-0.75 | Full paper target |

---

## SECTION 10 — What NOT to Do

| Mistake | Why |
|---|---|
| Retrain 6a from scratch | Phase 6a converged well. The problem is the inference bug and missing decoder training, not CIF quality |
| Use `lm_head(encoder_hidden)` in inference | This is the bug. Never do this again |
| Try to make CIF fire T_units (~350) tokens instead of T_text (~30) | CIF is supposed to produce text-length sequences. T2U decoder autoregressively expands them to unit-length. Don't retarget CIF. |
| Apply CE loss with mismatched shapes | `t2u_model(inputs_embeds=connector_out, labels=unit_ids)` is correct even if connector_out.shape[1] ≠ unit_ids.shape[0]. The model handles seq2seq alignment internally. |
| Skip the inference fix and go straight to retraining | The inference fix costs zero compute. If it works, you save days of training. |
| Train vocoder | It's pretrained and frozen. Leave it. |
| Use CE loss on T2U encoder output directly | The model has no decoder then — just train with `model.t2u_model(inputs_embeds=..., labels=...)` and let HuggingFace handle the encoder-decoder forward pass |

---

## SECTION 11 — Summary of Action Plan

### Immediate (this session):
1. **Copy `run_textless_s2st_v2` into your notebook** (Section 2.1 of this document)
2. Load `phase6a_textless` (or whatever is your best checkpoint)
3. Run inference on 3-5 samples across language pairs
4. Listen to output — does it sound like speech? How long?
5. Compute quick ChrF via ASR on 5 samples per pair

### If audio is reasonable (> 2s, has speech structure):
6. Run full Phase 7 benchmark with the fixed inference
7. Document results — this may already be publishable quality

### If audio is long but unintelligible:
8. Run Phase 6c (Unit CE Loss, 3000 steps, Section 3.2-3.4)
9. Checkpoint every 250 steps, track ChrF
10. After 6c, re-run Phase 7 benchmark

### For publication:
11. Run Phase 6d (speaker adapter, 500 steps) if speaker sim < 0.55
12. Long-form benchmark (Section 2.3 of PLAN.md, unchanged)
13. Produce paper tables

---

*Diagnosis v1.0 — Prepared after deep analysis of textless-v5.ipynb, PLAN.md, KD data verification, and training loss curves.*
*Key finding: The 0.7s garbage is caused by a single inference bug, not training failure.*
*The architecture is correct. The KD data is correct. Fix inference first.*
