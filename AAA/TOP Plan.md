# SeamlessLite: Compressing SeamlessM4T v2 to 850M for On-Device Multilingual Speech-to-Speech Translation

> **Blueprint Version:** 1.0 | **Target Platform:** Kaggle 2×T4 (15 GB VRAM each) | **Research Target:** ACL / Interspeech / EMNLP

---

## Overview & Design Rationale

| Property | Teacher (Baseline) | Target (SeamlessLite) |
|---|---|---|
| Model class | `SeamlessM4Tv2ForSpeechToSpeech` | Same class, surgically pruned |
| Parameters | ~1.8 B | **~850 M** |
| Languages supported | 35 speech output, 101 speech input | **5** (ben, eng, hin, tam, arb) |
| Max audio duration | ~30 s (practical) | **40–60 s** (chunked) |
| Vocoder | Multilingual HiFi-GAN | HiFi-GAN + speaker-conditioned extension |
| Voice cloning | ✗ | ✓ (optional, detachable plug-in) |
| Primary metrics | ASR-BLEU, ASR-ChrF | ASR-BLEU ↑, ASR-ChrF ↑ (vs. teacher) |
| ASR scorer | — | `facebook/mms-1b-all` |

### Language codes (SeamlessM4T convention)
```
Bengali  → ben  (script: Beng)
English  → eng  (script: Latn)
Hindi    → hin  (script: Deva)
Tamil    → tam  (script: Taml)
Arabic   → arb  (Modern Standard Arabic, script: Arab)
```

### Compression Budget (1.8 B → 850 M, ~950 M to remove)

| Sub-Module | Before (M params) | After (M params) | Delta | Technique |
|---|---|---|---|---|
| Speech encoder (w2v-BERT 2.0) | ~600 | **~280** | −320 | Layer pruning 24→12 + FLAP width |
| Text decoder (NLLB-based) | ~600 | **~300** | −300 | Layer pruning 24→12 + FLAP width |
| T2U encoder + decoder | ~150 | **~80** | −70 | Layer pruning 6+6→4+4 |
| Embedding table (256k vocab) | ~260 | **~65** | −195 | Vocab pruning 256k→64k |
| HiFi-GAN vocoder | ~50 | ~50 | 0 | Kept intact |
| Length adapter + misc | ~40 | ~35 | −5 | Minor |
| **Total** | **~1800** | **~810** | **−990** | |

> ±50M slack gives the ~850M target after optional LoRA adapter merging.

---

## Phase 0 — Environment Setup & Baseline Benchmark

**Goal:** Reproducible Kaggle session; establish teacher-quality floor for all 5×2 directions.

### 0.1 Environment

```
Kaggle T4 ×2 (each 15 GB VRAM, 16 GB RAM)
VRAM budget: Model in fp16 → 1.8B × 2 bytes ≈ 3.6 GB + activations ≈ 6–8 GB
Use device_map='auto' to split across both GPUs for the teacher
```

**Dependencies to install:**
```
transformers>=4.41  datasets  torchaudio  peft  bitsandbytes
sacrebleu  evaluate  jiwer  librosa  speechbrain  webrtcvad
silero-vad  resemblyzer  sentencepiece  safetensors  accelerate
```

### 0.2 Benchmark Datasets (5-lang, bidirectional)

| Dataset | Task | Languages | Usage |
|---|---|---|---|
| FLORES-200 | S2TT (for ASR-BLEU proxy) | all 5 | Primary eval |
| CVSS-C | S2ST (gold audio) | eng, arb, hin | Secondary eval |
| Common Voice 17 | ASR / S2ST | ben, tam | Dev-set |
| VoxPopuli | S2ST en→X | eng→hin, eng→arb | Training data |
| mExpresso | Expressive S2ST | eng subset | Optional future |

### 0.3 Benchmark Cell Template

```python
# ── Phase 0: Teacher Baseline ─────────────────────────────────────────────
LANG_PAIRS = [
    ("eng", "ben"), ("eng", "hin"), ("eng", "tam"), ("eng", "arb"),
    ("ben", "eng"), ("hin", "eng"), ("tam", "eng"), ("arb", "eng"),
]
# Load teacher model once in fp16
model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
    "facebook/seamless-m4t-v2-large",
    torch_dtype=torch.float16, device_map="auto"
)
# Run ASR-BLEU + ASR-ChrF via mms-1b-all
# Save to: results/phase0_teacher_scores.csv
```

### 0.4 Expected Baseline (FLORES-200, X→Eng avg from paper)

| Direction | ASR-BLEU (teacher, expected) |
|---|---|
| eng→ben | ~10–14 |
| eng→hin | ~18–22 |
| eng→tam | ~9–12 |
| eng→arb | ~14–18 |
| ben→eng | ~22–28 |
| hin→eng | ~28–34 |
| tam→eng | ~18–24 |
| arb→eng | ~26–32 |

> These are approximate extrapolations from SeamlessM4T v2 published results (Seamless Communication et al., 2023). Exact numbers must come from your Phase 0 run.

**Benchmark figure to generate:** Bar chart comparing all 8 directions (teacher), saved to `figures/phase0_teacher_asrbleu.png`.

---

## Phase 1 — Vocabulary / Embedding Pruning

**Paper:** Asahi et al., "Pruning Pre-trained Language Models Without Fine-Tuning" (EMNLP 2023); CULL-MT (Rostami & Dousti, 2024).

**Goal:** Reduce the shared embedding table from the full multilingual vocabulary (~256k tokens) to a 5-language vocabulary (~60–80k tokens). This is the **highest ROI** single step.

### 1.1 Technique

The NLLB tokenizer uses a SentencePiece model with a shared vocabulary. We:
1. Collect all token IDs that appear in the 5 target languages' training text.
2. Always keep: language-tag tokens for 5 langs, special tokens (`<s>`, `</s>`, `<pad>`, `<unk>`), numeral tokens.
3. Build a `old_id → new_id` remap dictionary.
4. Slice `model.shared.weight` and `model.text_decoder.embed_tokens.weight` to the new vocab.
5. Save remap as `_vocab_remap_to_old` attribute on model (used for decode remapping at inference).

### 1.2 Expected Savings

```
Original: 256,000 × 1,024 (fp16) ≈ 524 MB ≈ 256M params
After:     64,000 × 1,024 (fp16) ≈ 131 MB ≈  64M params
Savings: ~192M params, ~393 MB
```

### 1.3 Benchmark Cell

```python
# ── Phase 1 Benchmark ─────────────────────────────────────────────────────
# Load phase0 teacher; apply vocab pruning; measure quality
# Keys: model size (MB), vocab size, ASR-BLEU per direction
# Expected quality drop: < 0.5 BLEU (lossless for in-vocab tokens)
# Save to: results/phase1_vocab_scores.csv, models/phase1_vocab/
```

**Figure:** Side-by-side bar: vocab size (256k vs 64k) + ASR-BLEU per direction before/after.

---

## Phase 2 — Text Decoder Iterative Layer Pruning

**Paper:** Moslem, "Efficient Speech Translation through Model Compression and Knowledge Distillation" (IWSLT 2025, arXiv:2505.20237). Peer et al., "Greedy Layer Pruning" (2022). Sajjad et al., "On the Effect of Dropping Layers" (CSL 2023).

**Goal:** Prune the text decoder from 24 layers down to 12, saving ~150M parameters.

### 2.1 Technique: Greedy Layer Importance Evaluation

For each layer `l` in `{0..23}`:
1. Temporarily zero-out or skip layer `l`.
2. Compute ASR-BLEU on 100-sample dev set (fast proxy).
3. Record the `importance_score[l]` = quality drop when layer is removed.
4. Remove the layer with the **lowest importance score** permanently.
5. Repeat until target layer count reached.

```python
def layer_importance_scores(model, dev_loader, base_bleu, n_layers=24):
    scores = {}
    for l in range(n_layers):
        # Temporarily hook layer l to return identity
        with skip_layer(model.text_decoder, l):
            bleu = fast_eval_bleu(model, dev_loader)
        scores[l] = base_bleu - bleu  # higher = more important
    return scores

# Greedy prune: iteratively remove least important layer
# Target: 24 → 12 layers (remove 12 layers, one at a time)
```

### 2.2 Expected Savings

```
Each text decoder layer ≈ 4 × 1024² (QKV+O) + 2 × 1024 × 4096 (FFN) ≈ 12.6M params
12 layers removed = ~151M params
```

### 2.3 Benchmark Cell

```python
# ── Phase 2 Benchmark ─────────────────────────────────────────────────────
# Load phase1 model; apply text decoder pruning (24→12)
# Log importance scores per layer → figures/phase2_decoder_importance.png
# Before/after: param count, ASR-BLEU/ChrF per direction
# Expected: ≤8% ASR-BLEU degradation before recovery fine-tuning
# Save to: results/phase2_decoder_scores.csv, models/phase2_decoder/
```

**Figures:**
1. Heatmap of decoder layer importance scores (layers × directions).
2. ASR-BLEU bar chart: Teacher vs Phase1 vs Phase2.

---

## Phase 3 — Speech Encoder (w2v-BERT 2.0) Layer Pruning

**Paper:** Ma et al., "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect" (ACL 2025). Sajjad et al. (CSL 2023). CoLLD: Contrastive Layer-to-layer Distillation for compressing multilingual speech encoders (arXiv:2309.07707).

**Goal:** Prune the Conformer speech encoder from 24 layers to 12 layers, saving ~200M parameters.

### 3.1 Technique: Block Influence (BI) Score

ShortGPT's Block Influence metric measures how much each layer transforms the hidden representation:

```python
def block_influence(model, calibration_loader, layer_idx):
    """BI(l) = 1 - cosine_similarity(input_l, output_l)"""
    hidden_in, hidden_out = [], []
    # Register forward hooks on speech_encoder layer l
    # Compute mean BI over calibration set
    # Low BI → layer is near-identity → safe to remove
```

### 3.2 Kaggle VRAM Consideration

The w2v-BERT 2.0 encoder is 600M params. On 2×T4:
- Load model on GPU 0 only for importance scoring
- Use `torch.no_grad()` + `torch.cuda.empty_cache()` between iterations
- Process calibration in batches of 4, ~30s audio clips

### 3.3 Expected Savings

```
Each Conformer layer: Attention(1024, 16 heads) + Conv + FFN(1024→4096→1024) ≈ 16.7M params
12 layers removed = ~200M params
```

### 3.4 Benchmark Cell

```python
# ── Phase 3 Benchmark ─────────────────────────────────────────────────────
# Load phase2 model; apply speech encoder pruning (24→12)
# Log BI scores per conformer layer → figures/phase3_encoder_BI.png
# Before/after: ASR accuracy (WER on Common Voice), ASR-BLEU/ChrF
# Expected: ≤12% ASR-BLEU drop (pre-recovery). Layers 0-6 most critical.
# Save to: results/phase3_encoder_scores.csv, models/phase3_encoder/
```

**Figures:**
1. Block Influence bar chart per encoder layer.
2. Cumulative ASR-BLEU vs. number of encoder layers removed.

---

## Phase 4 — Width Pruning: FFN + Attention Head Pruning (FLAP)

**Paper:** An et al., "Fluctuation-based Adaptive Structured Pruning for Large Language Models" (AAAI 2024, arXiv:2312.11983). GitHub: `CASIA-IVA-Lab/FLAP`.

**Goal:** Reduce FFN intermediate size and attention heads in both the speech encoder and text decoder, saving ~100M parameters without removing entire layers.

### 4.1 Technique: FLAP WIFV Metric

FLAP computes the **W**eighted **I**nput **F**eature **V**ariance for each channel/head, standardizes scores globally across all modules, then applies adaptive pruning ratios per layer.

```python
# FLAP for transformer (adapted from CASIA-IVA-Lab/FLAP)
# Step 1: Run calibration data through model, collect activation statistics
# Step 2: Compute WIFV per column of weight matrices
# Step 3: Standardize scores globally (speech encoder + text decoder jointly)
# Step 4: Remove columns with lowest WIFV scores
# Step 5: Apply bias compensation to recover output feature maps

# Target pruning ratios:
#   Speech encoder FFN: 4096 → 2816 (-31%)
#   Speech encoder attn heads: 16 → 12 (-25%)
#   Text decoder FFN: 4096 → 2816 (-31%)
#   Text decoder attn heads: 16 → 12 (-25%)
```

### 4.2 Calibration Data

Use ~512 samples from the 5-language training set as FLAP calibration. Mix approximately equally across languages.

### 4.3 Expected Savings

```
Speech encoder (12 layers after Phase 3):
  FFN: 12 × 2 × (4096-2816) × 1024 × 2 bytes ≈ 63M params saved
  Heads: 12 × (16-12) × 64 × 1024 × 4 ≈ 15M params saved = ~78M

Text decoder (12 layers after Phase 2):
  FFN: 12 × 2 × (4096-2816) × 1024 × 2 bytes ≈ 63M params saved  
  Heads: 12 × (16-12) × 64 × 1024 × 4 ≈ 15M params saved = ~78M

Total FLAP savings: ~156M params (target: ~100M conservative)
```

### 4.4 Benchmark Cell

```python
# ── Phase 4 Benchmark ─────────────────────────────────────────────────────
# Load phase3 model; apply FLAP width pruning
# Calibration: 512 mixed-language samples, 5-10s audio
# Log per-layer pruning ratios → figures/phase4_flap_ratios.png
# Before/after: inference latency (ms/sample), memory (MB), ASR-BLEU
# Expected: ≤5% additional ASR-BLEU drop. Latency reduction: ~15%.
# Save to: results/phase4_flap_scores.csv, models/phase4_flap/
```

**Figures:**
1. Per-layer adaptive pruning ratio heatmap (encoder vs. decoder).
2. Memory vs. ASR-BLEU Pareto curve (all phases so far).

---

## Phase 5 — T2U Model Pruning

**Paper:** Greedy layer pruning (Peer et al., 2022); iterative layer removal (Rostami & Dousti, CULL-MT 2024).

**Goal:** Prune the non-autoregressive T2U encoder (6→4 layers) and T2U decoder (6→4 layers), saving ~70M parameters.

### 5.1 Technique

The T2U model in UnitY2 uses a NAR decoder (FastSpeech2-inspired). We apply the same greedy importance scoring as Phase 2, but evaluate on unit-level accuracy (via forced decoding) rather than ASR-BLEU, since ASR-BLEU requires full vocoder inference.

```python
def t2u_layer_importance(model, text_samples):
    """
    Evaluate T2U unit prediction accuracy with each layer skipped.
    Access via: model.t2u_model.model.encoder.layers[l]
                model.t2u_model.model.decoder.layers[l]
    """
    # Lower unit accuracy when layer skipped → layer is more important
    # Prune T2U encoder: 6→4 (remove 2 least important)
    # Prune T2U decoder: 6→4 (remove 2 least important)
```

### 5.2 Expected Savings

```
T2U encoder layer: ~12M params each × 2 removed = ~24M
T2U decoder layer: ~12M params each × 2 removed = ~24M  
Duration predictor + aligner: minimal (~3M)
Total T2U savings: ~50M params
```

### 5.3 Benchmark Cell

```python
# ── Phase 5 Benchmark ─────────────────────────────────────────────────────
# Load phase4 model; prune T2U encoder 6→4, decoder 6→4
# Before/after: unit prediction accuracy, full ASR-BLEU pipeline
# Expected: ≤3% additional ASR-BLEU drop
# Save to: results/phase5_t2u_scores.csv, models/phase5_t2u/
```

---

## Phase 6 — Recovery Fine-Tuning: LoRA + Sequence-Level Knowledge Distillation

**Papers:**
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
- Kim & Rush, "Sequence-Level Knowledge Distillation" (EMNLP 2016)
- Moslem (IWSLT 2025): Combined QLoRA + KD approach, 97–100% quality recovery at 50% model size
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (NeurIPS 2023)

**Goal:** Restore ASR-BLEU/ChrF to ≥95% of teacher quality using LoRA adapters on the pruned model, guided by sequence-level KD from the teacher.

### 6.1 Data Preparation for Fine-Tuning

**Primary data sources (use only what's available on Kaggle without download blocks):**

| Dataset | Hours | Languages | Use |
|---|---|---|---|
| Common Voice 17 (en, ar, hi) | ~50h each | eng, arb, hin | Supervised S2TT |
| IndicSUPERB | ~20h | ben, hin, tam | Supervised ASR + S2TT |
| FLEURS (subset) | ~10h/lang | all 5 | Eval + light train |
| Google Fleurs Arabic | ~10h | arb | Supervised |

**Sequence-Level KD pipeline:**
```python
# For each input audio in training set:
#   1. Run teacher model (in fp16, frozen) → get target text + target audio
#   2. Use teacher's text output as gold labels for student S2TT training
#   3. Use teacher's unit sequence as gold labels for student T2U training
# This is more efficient than online KD (teacher runs once, output cached)
```

### 6.2 LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,           # rank (Moslem 2025 used r=64 with rsLoRA)
    lora_alpha=128,
    target_modules=[
        # Text decoder attention projections
        "text_decoder.layers.*.self_attn.q_proj",
        "text_decoder.layers.*.self_attn.k_proj", 
        "text_decoder.layers.*.self_attn.v_proj",
        "text_decoder.layers.*.self_attn.out_proj",
        # T2U encoder/decoder
        "t2u_model.model.encoder.layers.*.self_attn.q_proj",
        "t2u_model.model.decoder.layers.*.self_attn.q_proj",
    ],
    lora_dropout=0.05,
    use_rslora=True,  # Rank-Stabilized LoRA (Kalajdzievski 2023)
)
```

### 6.3 Training Setup for 2×T4

```
Batch size: 4 (per GPU) × 2 GPUs = 8 effective
Gradient accumulation: 8 steps → effective batch = 64
Max input length: 30s audio (padded/chunked)
Max target length: 256 tokens
Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
Scheduler: cosine with 10% warmup
Mixed precision: fp16 (AMP)
Gradient checkpointing: ON (saves ~30% VRAM)
Epochs: 3–5 (monitor FLORES dev ASR-BLEU)
```

### 6.4 Multi-Stage Fine-Tuning

**Stage A (1 epoch, lr=2e-4):** Train only T2U model (frozen speech encoder + text decoder). Fast, restores unit prediction quality.

**Stage B (2–3 epochs, lr=1e-4):** LoRA on text decoder + T2U jointly. Sequence-level KD from teacher. Mixed data: 50% KD pseudo-labels + 50% authentic supervised data.

**Stage C (1 epoch, lr=5e-5):** Joint fine-tuning of all LoRA adapters. Language-balanced sampling (ensure ben+tam don't get crowded out by eng).

### 6.5 Benchmark Cell

```python
# ── Phase 6 Benchmark ─────────────────────────────────────────────────────
# After each training stage: eval on FLORES-200 dev, 5 directions × 2
# Metrics: ASR-BLEU (mms-1b-all), ASR-ChrF, RTF (real-time factor)
# Target: ≥95% of teacher ASR-BLEU on all 8 directions
# Plot: Training curve (steps vs dev ASR-BLEU) → figures/phase6_training.png
# Plot: Before/after recovery bar chart → figures/phase6_recovery.png
# Save: models/phase6_lora_merged/
```

**Figures:**
1. Training loss curve (stages A, B, C marked).
2. Dev ASR-BLEU vs. training steps, per language direction.
3. Final recovery table: Teacher vs. Phase5 (pruned) vs. Phase6 (recovered).

---

## Phase 7 — Long Audio Support (40–60 s)

**Papers:**
- Ma et al., "EMMA: Efficient Monotonic Multihead Attention" (SeamlessStreaming, 2023)
- Shen et al., "Chunk-based Speech Translation" — standard practice

**Goal:** Enable reliable translation of audio segments up to 60 seconds without quality degradation.

### 7.1 The Problem

SeamlessM4T's speech encoder uses a w2v-BERT 2.0 model with a length adapter (CNN with stride 8) followed by a max positional encoding of ~4000 frames (~40 s at 16kHz before striding). Practical quality degrades beyond 30 s due to attention memory limits.

### 7.2 Technique: VAD-Guided Overlapping Chunking

```python
import silero_vad  # lightweight VAD (Silero, 2021)

class LongAudioTranslator:
    """
    Strategy: 
    1. Run Silero VAD to find speech/silence boundaries
    2. Split at natural silence points, max chunk = 25s
    3. Overlap between chunks: 2s on each side (for context)
    4. Translate each chunk independently
    5. Stitch output audio: crossfade at boundaries (50ms fade)
    """
    
    def __init__(self, model, processor, max_chunk_s=25, overlap_s=2):
        self.model = model
        self.processor = processor
        self.max_chunk_s = max_chunk_s
        self.overlap_s = overlap_s
        self.vad = load_silero_vad()
    
    def translate_long(self, audio_array, src_lang, tgt_lang, sr=16000):
        # Step 1: VAD segmentation
        segments = self.get_vad_segments(audio_array, sr)
        # Step 2: Chunk segments to max_chunk_s with overlap
        chunks = self.build_chunks(segments, sr)
        # Step 3: Translate each chunk
        translated_chunks = [self.translate_chunk(c, src_lang, tgt_lang) for c in chunks]
        # Step 4: Crossfade stitch
        return self.stitch_audio(translated_chunks)
```

### 7.3 Benchmark Cell

```python
# ── Phase 7 Benchmark ─────────────────────────────────────────────────────
# Test audio lengths: 10s, 20s, 30s, 40s, 50s, 60s synthetic clips
# Metrics: ASR-BLEU, WER on stitched output, audio naturalness (MOS proxy)
# Plot: ASR-BLEU vs. audio duration → figures/phase7_duration_curve.png
# Plot: Comparison without/with chunking at 40s, 50s, 60s
# VRAM profile: peak memory vs. audio length
```

**Figure:** ASR-BLEU vs. input audio duration (10–60 s), with/without VAD chunking.

---

## Phase 8 — Final Comprehensive Benchmark

**Goal:** Paper-ready results table. Full evaluation on FLORES-200, CVSS-C (where available), and curated 5-language test sets.

### 8.1 Evaluation Protocol

```python
# ASR-BLEU scorer (your primary metric):
from transformers import AutoProcessor, AutoModelForCTC
asr_model = AutoModelForCTC.from_pretrained("facebook/mms-1b-all")
# Force-decode with target language

# Metrics computed:
# 1. ASR-BLEU (detokenized MMS transcript → sacrebleu)
# 2. ASR-ChrF (character-level F-score, better for ben/tam/arb)
# 3. RTF (real-time factor on T4 GPU, single stream)
# 4. Model size (MB on disk, fp16)
# 5. Peak VRAM (GB during inference, batch=1)
```

### 8.2 Comparison Table Template

| Model | Params | Size (MB) | eng→hin BLEU | eng→ben BLEU | eng→tam BLEU | eng→arb BLEU | avg X→eng BLEU | RTF |
|---|---|---|---|---|---|---|---|---|
| Teacher (v2-large, S2ST) | 1.8B | 3600 | — | — | — | — | — | — |
| Phase 1 (vocab prune) | 1.6B | 3200 | — | — | — | — | — | — |
| Phase 2 (+ dec prune) | 1.45B | 2900 | — | — | — | — | — | — |
| Phase 3 (+ enc prune) | 1.25B | 2500 | — | — | — | — | — | — |
| Phase 4 (+ FLAP) | 1.1B | 2200 | — | — | — | — | — | — |
| Phase 5 (+ T2U prune) | 1.05B | 2100 | — | — | — | — | — | — |
| **Phase 6 (+ LoRA recovery)** | **0.85B** | **1700** | **—** | **—** | **—** | **—** | **—** | **—** |

> Fill with actual numbers from your Kaggle runs. Target: Phase 6 ≥ 95% of Teacher on all columns.

### 8.3 Ablation Study (for paper)

Run the following ablations (each requires one Kaggle session):

| Ablation | Config | Purpose |
|---|---|---|
| A1 | Remove only vocab pruning, keep others | Quantify vocab pruning contribution |
| A2 | 24→16 dec layers (not 12) | Accuracy/compression tradeoff curve |
| A3 | 24→8 enc layers | Lower bound of encoder pruning |
| A4 | KD only (no LoRA) | Value of LoRA vs. pure KD |
| A5 | LoRA only (no KD) | Value of KD vs. pure LoRA |
| A6 | English-only fine-tuning | Catastrophic forgetting analysis |

---

## Phase 9 (Optional) — Voice Cloning Plug-in Layer

**Papers:**
- Casanova et al., "YourTTS: Towards Zero-Shot Multi-Speaker TTS" (ICML 2022)
- XTTS: "A Massively Multilingual Zero-Shot Text-to-Speech Model" (arXiv:2406.04904, 2024)
- Baumann et al., "ECAPA-TDNN: Emphasized Channel Attention" — speaker embedding extractor (Interspeech 2020)

**Goal:** Replace the standard HiFi-GAN vocoder with a speaker-conditioned version. The translated unit sequence remains identical; only the synthesis step changes. This module is **fully detachable** — setting `use_voice_cloning=False` reverts to standard synthesis.

> **Important:** Phase 9 is independent of Phases 0–8. Complete Phase 8 first. The voice cloning module does NOT modify the core translation pipeline.

### 9.1 Architecture: Speaker-Conditioned HiFi-GAN

```
Input audio (reference speaker, 3–10s)
        ↓
[Speaker Encoder: CAM++ or ECAPA-TDNN] → 192-dim speaker embedding
        ↓
[FiLM conditioning layer] → scale + shift ResBlocks of HiFi-GAN
        ↑
Unit sequence (from T2U model, language = TARGET)
        ↓
[HiFi-GAN Generator + FiLM conditioning] → Waveform in cloned voice
```

**FiLM (Feature-wise Linear Modulation):**
```python
class FiLMLayer(nn.Module):
    """Perez et al., AAAI 2018: Modulates HiFi-GAN ResBlock."""
    def __init__(self, speaker_dim=192, feature_dim=512):
        self.gamma = nn.Linear(speaker_dim, feature_dim)
        self.beta  = nn.Linear(speaker_dim, feature_dim)
    
    def forward(self, x, speaker_emb):
        γ = self.gamma(speaker_emb).unsqueeze(-1)
        β = self.beta(speaker_emb).unsqueeze(-1)
        return γ * x + β  # element-wise modulation
```

### 9.2 Training Strategy

**Phase 9A — Speaker encoder pre-training (freeze HiFi-GAN):**
- Use LibriSpeech + VoxCeleb2 (or mined from Common Voice) for speaker verification training
- Train ECAPA-TDNN with AAM-Softmax loss (speaker classification proxy)
- Target: EER < 5% on VoxCeleb1 test

**Phase 9B — FiLM adapter training (freeze speaker encoder, freeze unit vocoder):**
- Paired data: (unit_sequence, reference_audio, target_audio) from same speaker
- Loss: L1 mel + feature matching + discriminator adversarial
- Only FiLM γ/β projection layers are trainable (< 2M params per ResBlock × 3 = ~6M total)
- This keeps Kaggle VRAM feasible

**Phase 9C — Joint fine-tuning (low LR):**
- Unfreeze HiFi-GAN generator (not discriminator)
- LoRA-style: only adapt HiFi-GAN ResBlock weight matrices at r=16

### 9.3 Benchmark Cell

```python
# ── Phase 9 Benchmark ─────────────────────────────────────────────────────
# Speaker similarity: cosine similarity of ECAPA embeddings (reference vs. output)
# Naturalness: UTMOS score (automated MOS predictor)
# Translation quality: ASR-BLEU must stay within ±2 BLEU of Phase 6
# Test conditions:
#   - 3s reference audio (minimal cloning)
#   - 10s reference audio (standard cloning)
#   - Cross-lingual: English reference → Hindi output voice
# Figures:
#   figures/phase9_speaker_similarity.png (violin plot by language)
#   figures/phase9_sample_spectrograms.png (mel comparisons)
```

### 9.4 Dataset for Voice Cloning Training

| Dataset | Speakers | Languages | Hours |
|---|---|---|---|
| VCTK | 110 | English | 44h |
| Common Voice (multi) | 1000+ | all 5 | ~80h |
| Google TTS (synthetic) | synthetic | all 5 | ~20h |
| VoxCeleb2 (optional) | 5,994 | multilingual | large |

---

## Phase 10 — Research Paper Preparation

**Target venues:** Interspeech 2026, ACL 2026 Findings, EMNLP 2026

### 10.1 Paper Title Candidates

- *"SeamlessLite: Efficient On-Device Multilingual Speech-to-Speech Translation via Structured Pruning and Knowledge Distillation"*
- *"From 1.8B to 850M: Language-Targeted Compression of SeamlessM4T for 5-Language On-Device S2ST"*

### 10.2 Novel Contributions to Highlight

1. **Language-targeted vocabulary pruning** applied to SeamlessM4T's shared embedding — first study to do this for S2ST.
2. **Hierarchical importance scoring** across speech encoder (Block Influence) and text decoder (greedy BLEU drop) simultaneously, with joint FLAP width pruning.
3. **Combined pipeline achieving >95% ASR-BLEU retention at 53% parameter reduction** on 5 typologically diverse languages (Indic + Arabic).
4. **Speaker-conditioned HiFi-GAN plug-in** via lightweight FiLM adapters (<6M params) for zero-shot voice cloning in the translated language.
5. **VAD-guided chunked translation** enabling 40–60 s audio without architectural changes.

### 10.3 Comparison Baselines (for paper)

| System | Params | Method | Our Coverage |
|---|---|---|---|
| SeamlessM4T-v2-Large | 2.3B | Full training | Teacher |
| SeamlessM4T-Medium (v1) | 1.2B | Smaller arch | Comparable |
| **SeamlessLite (ours)** | **850M** | **Pruned + KD** | ✓ |
| CoLLD-0.3B encoder | 0.3B | Distillation only | Encoder only |

### 10.4 Key Citations

```bibtex
@article{seamless2023,
  title={Seamless: Multilingual Expressive and Streaming Speech Translation},
  author={Seamless Communication et al.},
  journal={arXiv:2312.05187},
  year={2023}
}

@inproceedings{moslem2025efficient,
  title={Efficient Speech Translation through Model Compression and Knowledge Distillation},
  author={Moslem, Yasmin},
  booktitle={Proceedings of IWSLT 2025},
  year={2025}
}

@inproceedings{an2024flap,
  title={Fluctuation-based Adaptive Structured Pruning for Large Language Models},
  author={An, Yongqi and Zhao, Xu and Yu, Tao and Tang, Ming and Wang, Jinqiao},
  booktitle={Proceedings of AAAI 2024},
  year={2024}
}

@article{ma2024shortgpt,
  title={ShortGPT: Layers in Large Language Models are More Redundant Than You Expect},
  author={Ma, Xin et al.},
  journal={arXiv:2403.03853},
  year={2024}
}

@inproceedings{peer2022greedy,
  title={Greedy-layer Pruning: Speeding up Transformer Models for NLP},
  author={Peer, David et al.},
  booktitle={EMNLP 2022},
  year={2022}
}

@article{sajjad2023effect,
  title={On the Effect of Dropping Layers of Pre-trained Transformer Models},
  author={Sajjad, Hassan et al.},
  journal={Computer Speech and Language},
  year={2023}
}

@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. et al.},
  booktitle={ICLR 2022},
  year={2022}
}

@inproceedings{kim2016sequence,
  title={Sequence-Level Knowledge Distillation},
  author={Kim, Yoon and Rush, Alexander M.},
  booktitle={EMNLP 2016},
  year={2016}
}

@inproceedings{casanova2022yourtts,
  title={YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for Everyone},
  author={Casanova, Edresson et al.},
  booktitle={ICML 2022},
  year={2022}
}

@article{xtts2024,
  title={XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model},
  author={Casanova et al.},
  journal={arXiv:2406.04904},
  year={2024}
}

@article{colld2023,
  title={CoLLD: Contrastive Layer-to-layer Distillation for Compressing Multilingual Pre-trained Speech Encoders},
  author={Chung et al.},
  journal={arXiv:2309.07707},
  year={2023}
}

@inproceedings{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim et al.},
  booktitle={NeurIPS 2023},
  year={2023}
}

@inproceedings{rostami2024cullmt,
  title={CULL-MT: Compression Using Language and Layer Pruning for Machine Translation},
  author={Rostami, Pedram and Dousti, Mohammad Javad},
  journal={arXiv},
  year={2024}
}
```

---

## Execution Checklist for Kaggle Sessions

Each Kaggle free session = ~9–12 hours. Plan accordingly:

| Session | Phases | Estimated Time | Output |
|---|---|---|---|
| S1 | 0 (setup + teacher baseline) | 6–8h | `phase0_teacher_scores.csv`, baseline figures |
| S2 | 1 (vocab pruning) | 2h | `models/phase1_vocab/` |
| S3 | 2 (text decoder pruning) | 4–6h | `models/phase2_decoder/` |
| S4 | 3 (speech encoder pruning) | 4–6h | `models/phase3_encoder/` |
| S5 | 4 (FLAP width pruning) | 2–3h | `models/phase4_flap/` |
| S6 | 5 (T2U pruning) | 2h | `models/phase5_t2u/` |
| S7 | 6A–6B (LoRA recovery, stages A+B) | 8–10h | `models/phase6_lora/` (checkpoint) |
| S8 | 6C + Phase 7 (long audio) | 6h | `models/phase6_lora_merged/`, chunker |
| S9 | 8 (full benchmark + ablation) | 8h | All result CSVs, paper tables |
| S10+ | 9 (voice cloning, optional) | 10–15h | `models/phase9_vc_hifigan/` |

**State persistence:** Use rclone to sync `WORK_DIR/models/` and `WORK_DIR/results/` to Google Drive at the end of each session. The setup cell infrastructure in your existing notebook (setup_cells_p7.ipynb) already handles this.

---

## Directory Structure

```
/kaggle/working/
├── models/
│   ├── phase0_teacher/          # HF save_pretrained of teacher (fp16)
│   ├── phase1_vocab/            # After vocab pruning
│   ├── phase2_decoder/          # After text decoder pruning
│   ├── phase3_encoder/          # After speech encoder pruning
│   ├── phase4_flap/             # After FLAP width pruning
│   ├── phase5_t2u/              # After T2U pruning
│   ├── phase6_lora_merged/      # Final 850M model (LoRA merged)
│   └── phase9_vc_hifigan/       # Voice cloning HiFi-GAN weights
├── checkpoints/                 # Training checkpoints (rclone synced)
├── results/
│   ├── phase0_teacher_scores.csv
│   ├── phase1_vocab_scores.csv
│   ├── ... (one CSV per phase)
│   └── phase8_final_table.csv   # Master comparison table
├── figures/
│   ├── phase0_teacher_asrbleu.png
│   ├── phase2_decoder_importance.png
│   ├── phase3_encoder_BI.png
│   ├── phase4_flap_ratios.png
│   ├── phase6_training.png
│   ├── phase6_recovery.png
│   ├── phase7_duration_curve.png
│   └── phase9_speaker_similarity.png
└── audio/                       # Sample audio for qualitative checks
```

---

## Risk Register & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bengali/Tamil ASR-BLEU collapses after pruning | Medium | High | Fine-tune Phase 6 with language-balanced sampling (30% each for ben+tam) |
| VRAM OOM during importance scoring | Medium | Medium | Use batch_size=1, gradient checkpointing, offload CPU |
| Vocabulary remap breaks tokenizer | Low | High | Always test decode/encode round-trip before saving |
| FLAP over-prunes speech encoder | Low | High | Set FLAP max_pruning_ratio=0.25 for encoder (vs 0.30 for decoder) |
| Voice cloning distorts translation quality | Low | Medium | VCLayer is detachable — keep as post-hoc replacement, don't retrain core model |
| Kaggle session timeout mid-training | High | Medium | Save checkpoint every 500 steps, rclone push immediately |
| Arabic ASR quality from mms-1b-all is unreliable | Low | Medium | Use additional ChrF metric; validate with Google FLEURS Arabic reference |

---

## Quick Reference: Key Parameters

```python
# Target languages
LANGS = ["ben", "eng", "hin", "tam", "arb"]
LANG_PAIRS = [(s, t) for s in LANGS for t in LANGS if s != t 
              and ("eng" in (s, t))]  # only X↔eng pairs

# Model loading (teacher)
TEACHER_MODEL = "facebook/seamless-m4t-v2-large"
STUDENT_CLASS = SeamlessM4Tv2ForSpeechToSpeech

# Target sizes
TARGET_ENC_LAYERS = 12      # was 24
TARGET_DEC_LAYERS = 12      # was 24
TARGET_T2U_LAYERS = 4       # was 6
TARGET_VOCAB_SIZE = 64_000  # was 256,206
TARGET_FFN_DIM    = 2816    # was 4096
TARGET_ATT_HEADS  = 12      # was 16
TARGET_PARAMS     = 850_000_000

# LoRA
LORA_RANK    = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05

# Training
BATCH_SIZE  = 4
GRAD_ACCUM  = 8
LR_STAGE_A  = 2e-4
LR_STAGE_B  = 1e-4
LR_STAGE_C  = 5e-5
MAX_EPOCHS  = 5

# ASR evaluation
ASR_MODEL = "facebook/mms-1b-all"
EVAL_DATASET = "facebook/flores"
EVAL_SPLIT = "devtest"
```

---

*End of Plan.md — Blueprint for SeamlessLite 850M*
*Next step: Generate the full Kaggle .ipynb from this plan.*
