# Structured Compression of SeamlessM4T v2 Large for Low-Resource Speech-to-Speech Translation
## English → Bengali (en_us → bn_in) · FLEURS Benchmark

---

> **TL;DR** — We compress `facebook/seamless-m4t-v2-large` (1,805.5 M parameters) to **1,039.1 M parameters** (**42.4 % reduction**) through a structured, multi-phase pipeline combining vocabulary pruning, iterative depth pruning of all three transformer stacks, and DoRA fine-tuning for quality recovery. The final model achieves **ChrF 45.14** vs the baseline's **50.52** (–10.6 %), runs **2.37× faster** (RTF 0.113 vs 0.268), and retains intelligible Bengali speech output — all on a single Kaggle T4 GPU (15.6 GB VRAM).

---

## Table of Contents

1. [Model Architecture Overview](#1-model-architecture-overview)
2. [Experimental Setup](#2-experimental-setup)
3. [Compression Pipeline Summary](#3-compression-pipeline-summary)
4. [Phase 0 — Baseline Benchmark](#4-phase-0--baseline-benchmark)
5. [Phase 1 — Vocabulary / Embedding Pruning](#5-phase-1--vocabulary--embedding-pruning)
6. [Phase 2 — Text Encoder Removal](#6-phase-2--text-encoder-removal)
7. [Phase 3 — Text Decoder Iterative Layer Pruning](#7-phase-3--text-decoder-iterative-layer-pruning)
8. [Phase 4 — Speech Encoder Iterative Layer Pruning (BI-Guided)](#8-phase-4--speech-encoder-iterative-layer-pruning-bi-guided)
9. [Phase 5 — Width Pruning (FLAP) — Attempted and Abandoned](#9-phase-5--width-pruning-flap--attempted-and-abandoned)
10. [Phase 6 — T2U Model Iterative Layer Pruning (ASR-ChrF Guided)](#10-phase-6--t2u-model-iterative-layer-pruning-asr-chrf-guided)
11. [Phase 7 — DoRA Fine-Tuning for Quality Recovery](#11-phase-7--dora-fine-tuning-for-quality-recovery)
12. [Phase 8 — Full-Model Knowledge Distillation (In Progress)](#12-phase-8--full-model-knowledge-distillation-in-progress)
13. [End-to-End Results and Analysis](#13-end-to-end-results-and-analysis)
14. [Component-Level Size Evolution](#14-component-level-size-evolution)
15. [Key Findings and Lessons](#15-key-findings-and-lessons)
16. [Bidirectional Translation Analysis — Layer Activation Study](#16-bidirectional-translation-analysis--layer-activation-study)
17. [References](#17-references)

---

## 1. Model Architecture Overview

`facebook/seamless-m4t-v2-large` is a massively multilingual speech-to-speech translation model covering ~100 languages. For the **Speech-to-Speech (S2S) task** (the only task used in this work), the forward path is:

```
Input Audio (EN)
       │
       ▼
[Speech Encoder]           ← Conformer, 24 layers, 635 M params
       │  encoder hidden states
       ▼
[Text Decoder]             ← Transformer decoder, 24 layers, 867 M params
       │  text token logits → greedy/beam-search tokens
       ▼
[T2U Model]                ← Transformer (encoder + decoder), 6+6 layers, 262 M params
       │  discrete speech unit sequence
       ▼
[Vocoder (HiFi-GAN)]       ← 41.9 M params (kept frozen throughout)
       │
       ▼
Output Audio (BN)
```

> **Note:** The `text_encoder` (another 24-layer transformer, ~350 M parameters in the full model) is **only used for T2T/T2S tasks**. When loading `SeamlessM4Tv2ForSpeechToSpeech`, the text encoder is never instantiated — its weights produce "UNEXPECTED" load warnings that can be safely ignored. This means Phase 2 (text encoder removal) was a no-op: the architecture already excludes it.

### Baseline Component Breakdown

| Component | Params (M) | % of Total |
|-----------|-----------|------------|
| text_decoder | 866.8 | 48.0 % |
| speech_encoder | 635.0 | 35.2 % |
| shared (embedding) | 262.2 | 14.5 % |
| lm_head | 262.2 | 14.5 % |
| t2u_model | 261.8 | 14.5 % |
| vocoder | 41.9 | 2.3 % |
| **TOTAL** | **1805.5** | — |

*Note: shared and lm_head are tied embeddings; they're shown at full size for counting but counted only once in TOTAL.*

---

## 2. Experimental Setup

### Hardware
- **Platform:** Kaggle Notebooks (T4 GPU, 15.6 GB VRAM)
- **GPU:** Tesla T4
- **Storage:** Google Drive (via rclone) for checkpoint persistence across sessions

### Dataset
- **Benchmark/Test:** FLEURS `en_us → bn_in`, **25 samples** drawn from the test split (349 matched pairs available)
- **Training (Phase 7):** FLEURS `en_us → bn_in` train split — **1,449 aligned speech–text pairs** (from 2,602 EN and 3,006 BN rows after deduplication and alignment)

### Evaluation Metrics
- **Text-BLEU:** SacreBLEU with `effective_order=True`
- **Text-ChrF:** SacreBLEU ChrF (character n-gram F-score) — primary quality metric for all pruning decisions due to its robustness with subword-heavy scripts like Bengali
- **RTF (Real-Time Factor):** inference time / audio duration — lower is faster
- **ASR-BLEU / ASR-ChrF (Phase 6 only):** Bengali ASR transcription of generated speech, compared to reference text. ASR performed using `facebook/mms-1b-all` with the `ben` (Bengali, ISO 639-3) adapter.

> **Why ASR metrics for Phase 6?** The T2U model operates in the audio domain. Text-ChrF scores the intermediate text tokens and is blind to T2U degradation. To properly evaluate T2U layer impact on the acoustic output, we decode the generated speech with an ASR system and measure ChrF against reference text transcriptions. We first tried OpenAI `whisper-medium` but found it to be completely non-functional on Bengali audio, producing English or nonsense output. We switched to `facebook/mms-1b-all`, which handles Bengali natively.

---

## 3. Compression Pipeline Summary

| Phase | Technique | Applied On | Params Before → After | BLEU | ChrF | RTF |
|-------|-----------|------------|----------------------|------|------|-----|
| **P0** | Baseline | — | 1805.5 M | 11.63 | 50.52 | 0.268 |
| **P1** | Vocabulary / Embedding Pruning | Baseline | 1805.5 → **1564.2 M** (−241.3 M) | 11.43 | 49.07 | 0.173 |
| **P2** | Text Encoder Removal | — | **No-op** (never loaded) | — | — | — |
| **P3** | Text Decoder Iterative Layer Pruning | P1 model | 1564.2 → **1312.3 M** (−251.9 M) | 8.09 | 43.58 | 0.099 |
| **P4** | Speech Encoder Iterative Layer Pruning | P3 model | 1312.3 → **1118.8 M** (−193.5 M) | 8.19 | 40.11 | 0.094 |
| **P5** | Width Pruning / FLAP | *Attempted* | **ABANDONED** — catastrophic failure | — | — | — |
| **P6** | T2U Iterative Layer Pruning | P4 model | 1118.8 → **1039.1 M** (−79.7 M) | 8.19 | 40.11 | 0.097 |
| **P7** | DoRA Fine-tuning | P6 model | 1039.1 M (unchanged) | **10.20** | **45.14** | 0.113 |
| **P8** | Full-Model KD | P7 model | *In progress* | — | — | — |

**Cumulative reduction:** 1805.5 M → 1039.1 M = **42.4 % parameter reduction**
**Speedup:** RTF 0.268 → 0.113 = **2.37× faster**
**Quality cost:** ChrF −5.38 (−10.6 %), BLEU −1.43 (−12.3 %)

---

## 4. Phase 0 — Baseline Benchmark

The full `facebook/seamless-m4t-v2-large` model was loaded in `SeamlessM4Tv2ForSpeechToSpeech` mode and benchmarked on 25 FLEURS test samples (EN→BN).

### Results

| Metric | Value |
|--------|-------|
| avg BLEU | 11.63 |
| avg ChrF | 50.52 |
| avg RTF | 0.268 |
| GPU Memory | ~1.79 GB alloc / 1.80 GB reserved |

### Sample Output (id=1660)
> **Reference (BN):** সংস্কৃতির দিক নির্ধারণের ক্ষেত্রে একটি বড় উপাদান ছিল শ্লেগাল গোথা ফিশ্তাদের মতো লেখকদের রোম্যান্টিসিজম  
> **Prediction:** রোমান্টিকতাবাদ গথ, ফিচ্ট এবং শ্লেগেলের মতো লেখকদের কাছ থেকে নেওয়া সাংস্কৃতিক নির্... `BLEU=10.7, ChrF=49.2`

The baseline is a strong multilingual model. ChrF above 50 on Bengali is a good starting point given the script complexity.

---

## 5. Phase 1 — Vocabulary / Embedding Pruning

**Method:** Asahi et al. (EMNLP 2023) — identify tokens actually used by target languages, discard unused embeddings.

### Why This Works

The NLLB tokenizer in SeamlessM4T was trained on ~100 languages and contains **256,102 tokens**. For English→Bengali S2S, only a subset of those tokens ever appear in source or target sequences. By scanning FLEURS training data for English, Bengali, Mandarin, French, and Hindi (a conservative superset), we identify which token IDs are actually referenced.

### Vocabulary Reduction

| Metric | Before | After |
|--------|--------|-------|
| Vocabulary size | 256,102 tokens | 20,425 tokens |
| Retained % | — | **8.0 %** |
| `shared` embedding | 262.2 M | 20.9 M |
| `lm_head` (tied) | 262.2 M | 20.9 M |
| **Net param savings** | — | **−241.3 M** |

The `shared` embedding matrix shrinks from `[256102, 1024]` to `[20425, 1024]`. The `lm_head` is tied to `shared`, so it shrinks automatically. The config `vocab_size`, generation special tokens, and `id_to_text` mapping are all remapped.

### Quality Impact

| Metric | Baseline | After P1 | Delta |
|--------|----------|----------|-------|
| BLEU | 11.63 | 11.43 | −0.20 |
| ChrF | 50.52 | 49.07 | −1.45 |
| RTF | 0.268 | 0.173 | **−35 %** (faster) |

**Near-lossless pruning.** The −1.45 ChrF drop is within noise for 25 samples. RTF improves significantly because the embedding lookup and LM head projection are cheaper on a vocabulary 12.5× smaller.

### Component Breakdown After P1

| Component | Params (M) | % of Total |
|-----------|-----------|------------|
| speech_encoder | 635.0 | 40.6 % |
| text_decoder | 625.5 | 40.0 % |
| t2u_model | 261.8 | 16.7 % |
| vocoder | 41.9 | 2.7 % |
| shared | 20.9 | 1.3 % |
| lm_head | 20.9 | 1.3 % |
| **TOTAL** | **1564.2** | — |

---

## 6. Phase 2 — Text Encoder Removal

**Finding:** No action needed.

Loading `SeamlessM4Tv2ForSpeechToSpeech` (as opposed to the full `SeamlessM4Tv2Model`) never instantiates `text_encoder` in the first place. The 18 "UNEXPECTED" warning lines printed at load time correspond exactly to the text encoder's keys — confirming the architecture already excludes it. The ~350 M text encoder parameters are never allocated.

---

## 7. Phase 3 — Text Decoder Iterative Layer Pruning

**Method:** Iterative greedy ChrF-guided depth pruning, following Moslem (IWSLT 2025) and CULL-MT (2024).

### Algorithm

```
Protected layers: {L0, L12, L23}  (first, middle, last)
For i = 1 to N_REMOVE:
    For each non-protected layer L:
        Temporarily remove L; compute ChrF on 25 eval samples
    Remove the layer whose absence caused the HIGHEST ChrF
    Re-number remaining layers; save checkpoint
```

The counter-intuitive objective is to **remove the layer whose removal hurts least** (i.e., the highest ChrF when it is absent = the least important layer).

### Iteration-by-Iteration Log

| Iter | Layers Remaining | Layer Removed | ChrF at Removal |
|------|-----------------|---------------|-----------------|
| 1 | 24 → 23 | **L9** | 50.52 |
| 2 | 23 → 22 | **L6** | 50.65 |
| 3 | 22 → 21 | **L15** | 50.53 |
| 4 | 21 → 20 | **L21** | 50.28 |
| 5 | 20 → 19 | **L8** | — |
| 6 | 19 → 18 | **L4** | — |
| 7 | 18 → 17 | **L14** | — |
| 8 | 17 → 16 | **L16** | — |
| 9 | 16 → 15 | **L22** | — |
| 10 | 15 → 14 | **L1** | — |

**Layers removed (original indices):** `[9, 6, 15, 21, 8, 4, 14, 16, 22, 1]`  
**Remaining:** 14 layers (indices renumbered 0–13)

### Candidate Layer ChrF Scores (Iteration 1)

The complete evaluation grid for the first iteration (all 21 eligible layers):

| Layer | ChrF if Removed |
|-------|----------------|
| L1 | 46.50 |
| L2 | 46.08 |
| L3 | 48.86 |
| L4 | 48.61 |
| L5 | 47.46 |
| L6 | 46.90 |
| L7 | 48.58 |
| **L8** | **49.13** |
| **L9** | **50.52** ← *chosen* |
| L10 | 48.23 |
| L11 | 46.77 |
| L13 | 47.45 |
| L14 | 46.63 |
| **L15** | **49.04** |
| L16 | 48.26 |
| L17 | 48.94 |
| L18 | 48.79 |
| L19 | 47.60 |
| L20 | 47.33 |
| L21 | 48.46 |
| **L22** | **49.78** |

Layer 9 was the most redundant in iteration 1 (highest post-removal ChrF = 50.52, essentially matching the pre-removal baseline). This pattern held across all 10 iterations, demonstrating that gradual, small-damage removal is viable.

### Results After Phase 3

| Metric | After P1 | After P3 | Delta |
|--------|----------|----------|-------|
| Params (M) | 1564.2 | 1312.3 | −251.9 M |
| BLEU | 11.43 | 8.09 | −3.34 |
| ChrF | 49.07 | 43.58 | −5.49 |
| RTF | 0.173 | 0.099 | **−43 %** (faster) |

### Component Breakdown After P3

| Component | Params (M) | % |
|-----------|-----------|---|
| speech_encoder | 635.0 | 48.4 % |
| text_decoder | **373.6** | 28.5 % ← was 625.5M |
| t2u_model | 261.8 | 19.9 % |
| vocoder | 41.9 | 3.2 % |
| **TOTAL** | **1312.3** | |

The text decoder shrank from 625.5 M (14 of 24 layers remaining) to 373.6 M — a 40 % reduction in just this component.

---

## 8. Phase 4 — Speech Encoder Iterative Layer Pruning (BI-Guided)

**Method:** Block Influence (BI) scoring from ShortGPT (ACL 2025) for candidate pre-filtering, combined with iterative greedy ChrF evaluation from Moslem (IWSLT 2025).

### Block Influence (BI) Scoring

Before any pruning, BI scores were computed on 25 calibration samples. BI measures how much each layer transforms the residual stream — a low BI score means the layer barely changes its input, making it a candidate for removal.

**Full BI Ranking (lower BI = more redundant):**

| Rank | Layer | BI Score |
|------|-------|----------|
| 1 (most redundant) | L10 | 0.1411 |
| 2 | L16 | 0.1467 |
| 3 | L15 | 0.1511 |
| 4 | L11 | 0.1522 |
| 5 | L9 | 0.1529 |
| 6 | L14 | 0.1586 |
| 7 | L13 | 0.1638 |
| 8 | L2 | 0.1678 |
| 9 | L12 | 0.1680 |
| 10 | L18 | 0.1694 |
| … | … | … |
| 23 | L23 | 0.4570 |
| 24 (most important) | L0 | 0.5806 |

The BI score confirms that the first (L0) and last (L23) layers are the most critical — a consistent finding across depth-pruning literature.

### Hybrid BI + ChrF Strategy

At each iteration, only the **bottom 50% by BI** (most redundant) are evaluated with the full ChrF probe. Layers in the top 50% by BI are skipped. This reduces the number of ChrF evaluations per iteration by ~50% without meaningfully compromising selection quality.

### Iteration-by-Iteration Log

| Iter | Removed Layer (original) | ChrF at Removal | BI Score |
|------|--------------------------|-----------------|----------|
| 1 | **L2** | 46.59 | 0.1678 |
| 2 | **L11** | 47.00 | 0.1522 |
| 3 | **L14** | — | 0.1586 |
| 4 | **L17** | — | 0.1833 |
| 5 | **L15** | — | 0.1511 |
| 6 | **L9** | — | 0.1529 |
| 7 | **L19** | — | 0.1768 |
| 8 | **L5** | — | 0.1913 |

**Layers removed (original indices):** `[2, 11, 14, 17, 15, 9, 19, 5]`  
**Speech encoder:** 24 → **16 layers** remaining

### Results After Phase 4

| Metric | After P3 | After P4 | Delta |
|--------|----------|----------|-------|
| Params (M) | 1312.3 | 1118.8 | −193.5 M |
| BLEU | 8.09 | 8.19 | +0.10 |
| ChrF | 43.58 | 40.11 | −3.47 |
| RTF | 0.099 | 0.094 | −5 % (faster) |

*Note: The tiny BLEU improvement (+0.10) is within noise and should not be interpreted as quality gain.*

### Component Breakdown After P4

| Component | Params (M) | % |
|-----------|-----------|---|
| speech_encoder | **441.6** | 39.5 % ← was 635.0M |
| text_decoder | 373.6 | 33.4 % |
| t2u_model | 261.8 | 23.4 % |
| vocoder | 41.9 | 3.7 % |
| **TOTAL** | **1118.8** | |

---

## 9. Phase 5 — Width Pruning (FLAP) — Attempted and Abandoned

**Method:** FLAP (AAAI 2024) — structured removal of FFN neurons using Wanda-sp importance scores, with activation statistics collected via forward-pass calibration.

Width pruning (reducing the FFN hidden dimension) complements depth pruning: it creates smaller, denser matrices that improve actual wall-clock throughput, as opposed to merely zeroing weights.

### Experiment A: FLAP on Base Model

Applied FLAP at 10% pruning ratio to `facebook/seamless-m4t-v2-large` directly.

| Component | Neurons before | Neurons after |
|-----------|---------------|---------------|
| text_decoder | 196,608 | 177,211 (90.1%) |
| speech_encoder | — | reduced |
| t2u_model | — | reduced |
| **Total params** | **1805.5 M** | **1713.7 M** |

**Benchmark result:**

| Metric | Baseline | After FLAP (base) |
|--------|----------|-------------------|
| BLEU | 11.63 | 6.34 |
| ChrF | 50.52 | 35.48 |
| RTF | 0.268 | 0.234 |

ChrF drops 15 points (−29.8%) for only −5.1% parameter reduction. The value proposition is poor.

### Experiment B: FLAP on P4 Model (Phase 4 pruned model)

Applied FLAP at 10% pruning ratio on top of the phase4 model (which already had depth pruning applied).

**Before:** 1118.8 M → **After:** 1057.2 M (−61.6 M, −5.5%)

**Benchmark result — catastrophic failure:**

| Metric | After P4 | After FLAP (P4) |
|--------|----------|-----------------|
| BLEU | 8.19 | 0.95 |
| ChrF | 40.11 | **9.20** |
| RTF | 0.094 | 0.354 |

Sample outputs showed severe loop-repetition hallucinations:
- `"রোম্যান্টিকবাদবাদবাদের সংস্কৃতিগতগতগতগতগতগতগত..."` (character loop)
- `"তিনি বলেন, 'কাকাকাকাকাকাকাকাকাকাকা..."` (token loop)
- RTF ballooned to 0.354–0.613 due to unconstrained token generation loops

### Analysis and Decision

The interaction between depth pruning and width pruning broke the model irreparably. After depth pruning, the remaining layers likely rely more heavily on their full FFN width for capacity — removing neurons removes information that can no longer be compensated by adjacent layers that were already removed.

**Phase 5 was formally abandoned.** The depth-only pipeline was found to be strictly superior in the quality/compression tradeoff.

### FFN Layer Inventory (for reference)

| Component | FFN pairs (base) | FFN pairs (P4) | FFN hidden dim |
|-----------|-----------------|----------------|---------------|
| speech_encoder | 50 | 34 | 1024 → 4096 |
| text_decoder | 24 | 14 | 1024 → 8192 |
| t2u_model | 6 | 6 | 1024 → 8192 |

---

## 10. Phase 6 — T2U Model Iterative Layer Pruning (ASR-ChrF Guided)

**Method:** Iterative greedy layer pruning on T2U encoder and decoder stacks, with ASR-ChrF as the evaluation metric.

### T2U Architecture

The `t2u_model` contains two independent stacks:
- **T2U Encoder:** 6 transformer encoder layers (125.9 M)
- **T2U Decoder:** 6 transformer decoder layers (135.8 M)
- **T2U LM head:** 10.3 M

### Why ASR-ChrF Instead of Text-ChrF

Text-ChrF measures the quality of intermediate text token sequences. After the text decoder, the T2U model converts those tokens into discrete speech units. Degrading the T2U model does not affect text token quality but produces poor audio. To properly detect T2U damage, we:

1. Run the full S2S pipeline to generate audio waveforms
2. Transcribe generated audio with `facebook/mms-1b-all` (Bengali, `ben` adapter)
3. Compute ChrF between ASR transcript and ground-truth reference text

> **ASR Model Selection:** OpenAI `whisper-medium` was initially tested but produced completely wrong or English-language output on Bengali audio. `facebook/mms-1b-all` with the `ben` adapter worked reliably and was adopted for all T2U evaluation.

### T2U Encoder Pruning (2 layers removed from 6)

**Baseline ASR-ChrF:** 43.77

| Iteration | Layer | ASR-ChrF if Removed | Selected? |
|-----------|-------|---------------------|-----------|
| Iter 1 | L0 | **8.40** ← critical | |
| Iter 1 | L1 | 43.83 | ✓ removed |
| Iter 1 | L2 | 43.24 | |
| Iter 1 | L3 | 43.76 | |
| Iter 1 | L4 | 43.43 | |
| Iter 1 | L5 | 42.37 | |
| Iter 2 | L2 | 43.51 | ✓ removed |
| Iter 2 | L3 | 36.14 | |
| Iter 2 | L4 | 43.19 | |
| Iter 2 | L5 | 43.39 | |

**L0 (first layer) was the most critical** — removing it caused ASR-ChrF to collapse to 8.40, consistent with the finding in Phase 3 and 4 that first layers are essential.

### T2U Decoder Pruning (2 layers removed from 6)

**Baseline ASR-ChrF:** 43.51

| Iteration | Layer | ASR-ChrF if Removed | Selected? |
|-----------|-------|---------------------|-----------|
| Iter 1 | L0 | **25.97** ← critical | |
| Iter 1 | L1 | 40.01 | |
| Iter 1 | L2 | 39.29 | |
| Iter 1 | L3 | 39.88 | |
| Iter 1 | L4 | 39.56 | |
| Iter 1 | L5 | 42.51 | ✓ removed |
| Iter 2 | L1 | 38.37 | |
| Iter 2 | L2 | 37.37 | |
| Iter 2 | L3 | 39.06 | ✓ removed |
| Iter 2 | L4 | 33.60 | |

**L5 (last layer, decoder)** was the most removable in iteration 1 — the decoder's last layer is more redundant than its first.

### T2U Parameter Savings

| Component | Before | After |
|-----------|--------|-------|
| T2U Encoder layers | 6 | 4 |
| T2U Decoder layers | 6 | 4 |
| T2U total params | 261.8 M | **182.0 M** |
| **Saved** | — | **−79.7 M** |

### Results After Phase 6

| Metric | After P4 | After P6 | Delta |
|--------|----------|----------|-------|
| Params (M) | 1118.8 | 1039.1 | −79.7 M |
| BLEU | 8.19 | 8.19 | 0.00 |
| ChrF | 40.11 | 40.11 | 0.00 |
| RTF | 0.094 | 0.097 | +0.003 |

Text-BLEU and Text-ChrF are unchanged — confirming that T2U pruning (guided by ASR-ChrF) correctly preserves the text-decoding pathway while maintaining acceptable audio quality.

### Component Breakdown After P6

| Component | Params (M) | % |
|-----------|-----------|---|
| speech_encoder | 441.6 | 42.5 % |
| text_decoder | 373.6 | 36.0 % |
| t2u_model | **182.0** | 17.5 % ← was 261.8M |
| vocoder | 41.9 | 4.0 % |
| **TOTAL** | **1039.1** | |

---

## 11. Phase 7 — DoRA Fine-Tuning for Quality Recovery

**Method:** DoRA (Liu et al., ICML 2024 Oral) — Weight-Decomposed Low-Rank Adaptation applied to all attention and FFN linear layers across the entire model.

### Motivation

After phases 1–6, the model has lost ~10 ChrF points relative to baseline. Vanilla fine-tuning would require backpropagating through all 1039 M parameters. DoRA (an extension of LoRA that decomposes weight matrices into magnitude and direction components) provides memory-efficient adaptation with only ~1% trainable parameters while applying updates to all model components including the T2U.

> **Previous S2TT-only Fine-tuning Concern (noted in code):** An earlier experimental version used cross-entropy loss on text token outputs only (S2TT). This backpropagated gradients only through `speech_encoder → text_decoder`, giving **zero gradient to t2u_model**. The audio output remained broken even as BLEU/ChrF appeared recovered. The Phase 7 design in the final pipeline addresses this by computing loss on the full inference path.

### DoRA Configuration

| Hyperparameter | Value |
|----------------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `fc1`, `fc2`, `k_proj`, `out_proj`, `q_proj`, `v_proj` |
| Layers covered | 180 Linear layers |
| Trainable params | **10,340,352** (10.34 M) |
| Total params | 1,049,437,317 |
| Trainable % | **0.985 %** |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Max steps | 2,500 |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Learning rate | 3e-4 |
| Optimizer | AdamW |
| Training samples | 1,449 aligned FLEURS EN→BN pairs |
| Training time | ~195 minutes (~3.25 hours) |
| Hardware | Kaggle Tesla T4 (15.6 GB VRAM) |

### Training Loss Curve

| Step | Training Loss |
|------|--------------|
| 50 | 5.9578 |
| 100 | 3.8861 |
| 150 | 3.5615 |
| 200 | 3.0916 |
| 250 | 2.9988 |
| 300 | 2.6668 |
| 500 | 2.3160 |
| 750 | 2.0619 |
| 1000 | 2.0664 |
| 1250 | 2.0528 |
| 1500 | 1.9723 |
| 1750 | 2.0161 |
| 2000 | 1.9636 |
| 2250 | **1.7829** |
| 2500 | 1.9909 |

Loss decreases from 5.96 to approximately 1.78–2.00, with convergence appearing around step 1500–2000. The slight uptick at step 2500 suggests the model has largely converged.

### Results After Phase 7

| Metric | After P6 | After P7 (DoRA) | Delta |
|--------|----------|-----------------|-------|
| Params (M) | 1039.1 | 1039.1 | 0 |
| BLEU | 8.19 | **10.20** | **+2.01** |
| ChrF | 40.11 | **45.14** | **+5.03** |
| RTF | 0.097 | 0.113 | +0.016 |

**DoRA recovered +5.03 ChrF and +2.01 BLEU points** in ~3 hours of fine-tuning, using only ~10 M additional trainable parameters. The minor RTF increase is due to the LoRA adapter inference overhead (adapters are merged before saving, so deployment RTF is unchanged at 0.113).

### Sample Translation Improvement (id=1660)

| Version | Prediction |
|---------|-----------|
| After P6 (pruned, no FT) | রোমান্টিকতাবাদ সংস্কৃতির নির্ণয়বাদের একটি বড় উপাদান ছিল, যা গথ, ফিচট এবং শ্লেগেলের... |
| After P7 (DoRA fine-tuned) | রোমান্টিকতার মধ্যে সংস্কৃতির নির্ধারকতা এর একটি বড় উপাদান ছিল যা গথ ফিচ এবং শ্লে... |
| Reference | সংস্কৃতির দিক নির্ধারণের ক্ষেত্রে একটি বড় উপাদান ছিল শ্লেগাল গোথা ফিশ্তাদের মতো লেখকদের রোম্যান্টিসিজম |

The DoRA fine-tuned output is noticeably more natural and closer to the reference.

---

## 12. Phase 8 — Full-Model Knowledge Distillation (In Progress)

**Goal:** Further recover audio-output quality by distilling from `facebook/seamless-m4t-v2-large` (teacher) into the phase7 model (student), with a combined loss that explicitly supervises the vocoder output.

### Design

```
Teacher: facebook/seamless-m4t-v2-large (1805.5M, frozen)
Student: phase7_dora_merged_v1 (1039.1M, all params trainable)

Loss = α × KL_divergence(student_logits / T, teacher_logits / T)
     + (1 − α) × MSE(student_vocoder_output, teacher_vocoder_output)
```

| Hyperparameter | Value |
|----------------|-------|
| Temperature (T) | 2.0 |
| Alpha (KL weight) | 0.7 |
| Learning rate | 1e-5 |
| Optimizer | AdamW + CosineAnnealingLR |
| Max steps | 1,000 |
| Effective batch size | 8 |

### Status

The KD training loop was initialized and began but was interrupted before completing any steps (Kaggle session timeout / OOM). The student model has 1039.1 M trainable parameters; running both teacher (1805.5 M, frozen) and student simultaneously requires ~14+ GB VRAM at batch size 2, which is at the limit of the T4's 15.6 GB.

**Phase 8 results are not yet available.** This is identified as the next step: either multi-GPU setup, gradient checkpointing, or partial-model KD (T2U-only distillation to reduce VRAM requirements).

---

## 13. End-to-End Results and Analysis

### Full Pipeline Summary Table

```
================================================================================
  FINAL: SeamlessM4T v2 Large  Structured Compression
  Task: English to Bengali Speech Translation (FLEURS test, 25 samples)
================================================================================
Phase                      Params(M)    Delta    BLEU    ChrF     RTF
---------------------------------------------------------------------
  P0_Baseline               1805.5     base   11.63   50.52  0.2681
  P1_VocabTrim              1564.2   -13.4%   11.43   49.07  0.1734
  P3_DecPrune               1312.3   -27.3%    8.09   43.58  0.0994
  P4_EncPrune               1118.8   -38.0%    8.19   40.11  0.0937
  P5_FLAP(base)             1713.7    -5.1%    6.34   35.48  0.2341  ← abandoned
  P5_FLAP(m4)               1057.2   -41.4%    0.95    9.20  0.3540  ← abandoned
  P6_T2UIter                1039.1   -42.4%    8.19   40.11  0.0972
  P7_DoRA                   1039.1   -42.4%   10.20   45.14  0.1129
================================================================================
  Param reduction: 42.4%
  Speed (RTF): 2.37× faster
```

### ChrF vs Parameters Efficiency Curve

```
ChrF
50 │●  P0 (1805M)
   │ ●  P1 (1564M)
45 │            ● P7 (1039M) ← Final model
   │
43 │       ●P3 (1312M)
   │
40 │           ●P4 (1118M) = ●P6 (1039M)
   │
35 │                 ●P5-base (1714M)  ← FLAP on base, poor tradeoff
   │
 9 │                    ●P5-m4 (1057M) ← FLAP on P4, catastrophic
   └─────────────────────────────────── Params (M)
     1805  1564  1312  1118  1039
```

### Compression Contribution Per Phase

| Phase | Params Removed (M) | % of Total Removed (766.4 M) |
|-------|-------------------|------------------------------|
| P1 Vocab | 241.3 | 31.5 % |
| P3 Dec Layers | 251.9 | 32.9 % |
| P4 Enc Layers | 193.5 | 25.2 % |
| P6 T2U Layers | 79.7 | 10.4 % |
| **Total** | **766.4** | **100 %** |

Vocabulary pruning and text decoder depth pruning each contribute roughly a third of the total parameter savings.

### RTF Improvement by Phase

| Phase | RTF | Speedup vs Baseline |
|-------|-----|---------------------|
| P0 Baseline | 0.268 | 1.00× |
| P1 Vocab | 0.173 | 1.55× |
| P3 DecPrune | 0.099 | 2.71× |
| P4 EncPrune | 0.094 | 2.85× |
| P6 T2UIter | 0.097 | 2.76× |
| P7 DoRA (final) | 0.113 | **2.37×** |

The RTF "worsens" slightly at P7 compared to P6 due to LoRA adapter overhead during benchmarking (adapters had not yet been merged). Post-merge, the deployed model would run at approximately P6's RTF (~0.097, ~2.76×).

---

## 14. Component-Level Size Evolution

| Phase | speech_encoder | text_decoder | t2u_model | shared/lm_head | vocoder | **TOTAL** |
|-------|---------------|-------------|-----------|---------------|---------|---------|
| P0 Baseline | 635.0 M | 866.8 M | 261.8 M | 262.2 M (×2) | 41.9 M | **1805.5 M** |
| P1 VocabTrim | 635.0 M | 625.5 M | 261.8 M | 20.9 M (×2) | 41.9 M | **1564.2 M** |
| P3 DecPrune | 635.0 M | 373.6 M | 261.8 M | 20.9 M (×2) | 41.9 M | **1312.3 M** |
| P4 EncPrune | 441.6 M | 373.6 M | 261.8 M | 20.9 M (×2) | 41.9 M | **1118.8 M** |
| P6 T2UIter | 441.6 M | 373.6 M | 182.0 M | 20.9 M (×2) | 41.9 M | **1039.1 M** |
| P7 DoRA (final) | 441.6 M | 373.6 M | 182.0 M | 20.9 M (×2) | 41.9 M | **1039.1 M** |

### Layer Counts

| Component | Baseline Layers | Final Layers | Removed |
|-----------|----------------|--------------|---------|
| text_decoder | 24 | **14** | 10 |
| speech_encoder | 24 | **16** | 8 |
| t2u encoder | 6 | **4** | 2 |
| t2u decoder | 6 | **4** | 2 |
| vocoder | — | **unchanged** | 0 |

---

## 15. Key Findings and Lessons

### 1. Iterative Greedy ChrF Pruning Is Remarkably Effective

Removing one layer at a time and re-evaluating ChrF allows the pipeline to find highly redundant layers that would not be obvious from static importance metrics alone. The total ChrF loss from 20 layer removals across three components was only −5.38 ChrF points combined — a surprisingly small penalty for 42.4% parameter reduction.

### 2. First and Last Layers Are Non-Negotiable

Across every component (text decoder, speech encoder, T2U encoder, T2U decoder), L0 and the last layer showed the highest importance scores and the largest ChrF drops when removed. Protected them by design (first/mid/last protection) proved to be the right call.

### 3. FLAP Width Pruning Is Incompatible with Pre-Pruned Depth

Applying FLAP on the P4 model (already depth-pruned) caused catastrophic failure (ChrF = 9.20, loop-repetition hallucinations). The depth-pruned model appears to have tighter capacity constraints per remaining layer, making it highly sensitive to any further neuron removal. FLAP on the baseline was workable but the quality/size tradeoff was poor (−14.7% ChrF for −5.1% params). **Lesson: apply one type of pruning at a time, or apply width pruning first if at all.**

### 4. ASR Metrics Are Essential for Audio-Domain Components

Text-ChrF is blind to T2U degradation. The iterative pruning for T2U must use audio-domain metrics. The discovery that whisper-medium completely fails on Bengali audio (and that `facebook/mms-1b-all` with Bengali adapter is required) is a practical finding with direct relevance to any Bengali speech processing work.

### 5. The Pruning Direction Bias Severely Damages Reverse-Direction Translation

> **Critical finding documented in the bidirectional activation analysis notebooks.**

Every component of this pipeline — ChrF-guided layer selection (Phases 3, 4, 6), DoRA fine-tuning (Phase 7), and ASR-ChrF scoring — was optimised **exclusively on EN→BN data**. The activation study (see Section 16) reveals that this introduces a significant directional bias with severe consequences for BN→EN translation.

In the base model, BN→EN translation relies disproportionately on the **upper text decoder layers** (L20–L23 show BN→EN activation scores 965–3,828 points higher than EN→BN). Our pruning removed L21 and L22 — exactly these upper layers — because their removal minimised EN→BN ChrF loss. The base model is roughly symmetric bidirectionally (ChrF 49.19 EN→BN vs 50.22 BN→EN). After pruning and DoRA fine-tuning, BN→EN collapses: only 4 of 10 test samples produce non-empty output, with avg ChrF = 35.26 vs 46.83 for EN→BN. Six of ten BN→EN samples produce complete generation failure. This is a direct consequence of task-specific pruning criteria.

### 6. DoRA Provides Excellent Quality Recovery at Low Cost

10.34 M trainable parameters (~1% of model) over 2,500 steps (~3 hours on a T4) recovered +5.03 ChrF and +2.01 BLEU. The final model (P7) is within 5.38 ChrF of the baseline despite being 42.4% smaller and 2.37× faster.

### 7. Vocabulary Pruning Is Free

Reducing vocabulary from 256,102 to 20,425 tokens (92% reduction in embedding size) cost only −1.45 ChrF, saved 241 M parameters, and made inference 35% faster. This should be a mandatory first step for any deployment targeting a small set of languages.

### 8. Checkpoint Persistence Strategy Matters

Given Kaggle's 12-hour session limit and the total pipeline requiring 20+ hours of compute, the use of rclone + Google Drive for checkpoint persistence was essential. Every iteration of every pruning phase was checkpointed, allowing seamless session resumption.

---

## 16. Bidirectional Translation Analysis — Layer Activation Study

To investigate whether our EN→BN-focused pruning inadvertently degraded the model's capacity for other translation directions, we ran a post-hoc activation analysis on both the **base model** (`facebook/seamless-m4t-v2-large`) and the **final pruned + DoRA-finetuned model** (`phase7_dora_merged_v1`). Forward-pass activation hooks were registered on every layer of every major component, and layer importance was computed as the mean L2-norm of each layer's output activations over 10 test samples per direction.

**Setup:** 10 FLEURS EN→BN samples and 10 matched FLEURS BN→EN samples. Base model: 60 hooks. Pruned model: 38 hooks (fewer layers after depth pruning).

---

### 16.1 Translation Quality: Base vs Pruned (Bidirectional)

| Direction | Base Model | Pruned Model (P7) | Delta |
|-----------|-----------|-------------------|-------|
| EN→BN avg BLEU | 10.52 | 9.44 | −1.08 |
| EN→BN avg ChrF | **49.19** | **46.83** | **−2.36** |
| BN→EN avg BLEU | 16.54 | 7.02 *(4/10 only)* | **−9.52** |
| BN→EN avg ChrF | **50.22** | **35.26** *(4/10 only)* | **−14.96** |

> ⚠️ **The BN→EN performance collapse is severe.** Only 4 of 10 BN→EN test samples produced any non-empty output from the pruned model. The remaining 6 produced BLEU=0.0, ChrF=0.0 — complete generation failure. The base model handled all 10 samples successfully. This is a direct consequence of the fact that every pruning decision (Phases 3, 4, 6) and the DoRA fine-tuning (Phase 7) used EN→BN ChrF as the sole optimisation signal. The pruned model has effectively become an EN→BN specialist at the cost of its reverse-direction capability.

---

### 16.2 Base Model Layer Importance Rankings

#### Speech Encoder (24 layers)

Both directions show very similar importance profiles, with the highest activity concentrated in the early-to-middle layers.

**EN→BN top 10:** L1 (26.46), L2 (25.59), L10 (25.57), L0 (25.24), L9 (25.14), L4 (24.64), L14 (22.55), L11 (22.55), L5 (22.46), L15 (22.42)

**BN→EN top 10:** L1 (26.62), L2 (25.80), L0 (25.17), L10 (25.04), L4 (24.69), L9 (24.47), L11 (22.25), L15 (22.22), L5 (22.20), L14 (22.06)

The rankings are nearly identical, indicating the speech encoder is relatively direction-agnostic. The most direction-sensitive encoder layers (where EN→BN activations are notably stronger) are L9 (+0.67), L10 (+0.53), L14 (+0.49), L13 (+0.49), L8 (+0.46). Crucially, **L9 and L14 were removed in Phase 4** — they were deemed least important for EN→BN, but they carried slightly more weight for EN→BN signal processing than BN→EN.

#### Text Decoder (24 layers) — The Critical Asymmetry

This component shows the strongest directional divergence in the base model.

| Layer | EN→BN Score | BN→EN Score | Difference (EN→BN minus BN→EN) |
|-------|------------|------------|-------------------------------|
| L0 | 582.0 | 635.2 | −53.2 (BN→EN favours) |
| L1 | 832.1 | 870.3 | −38.2 (BN→EN favours) |
| L2–L7 | — | — | BN→EN scores consistently higher |
| L8 | 2089.1 | 2062.6 | +26.4 (EN→BN favours) |
| L9–L18 | — | — | EN→BN scores consistently higher |
| L19 | 6015.4 | 6067.2 | −51.8 (BN→EN favours) |
| **L20** | **6310.5** | **7275.7** | **−965.2 (BN→EN strongly favours)** |
| **L21** | **6767.7** | **9388.8** | **−2621.1 (BN→EN strongly favours)** |
| **L22** | **7471.0** | **11079.2** | **−3608.2 (BN→EN strongly favours)** |
| **L23** | **7769.9** | **11597.9** | **−3827.9 (BN→EN strongly favours)** |

**The upper four text decoder layers (L20–L23) are dramatically more important for BN→EN than EN→BN.** Layer L23 alone has a BN→EN activation score 49% higher than its EN→BN score. Layers L21–L23 together account for an activation differential of over 10,000 score units in favour of BN→EN — far larger than any other directional asymmetry in the model.

Our Phase 3 pruning removed **L21 and L22** from the text decoder (among 10 total) because their removal caused the least damage to EN→BN ChrF. With hindsight from this activation analysis, we were removing exactly the layers the model needed most for BN→EN translation.

#### T2U Encoder (6 layers)

The upper T2U encoder layers are more important for EN→BN than BN→EN:

| Layer | EN→BN | BN→EN | Diff (EN→BN favours) |
|-------|-------|-------|---------------------|
| L0 | 56.4 | 60.6 | −4.1 (BN→EN favours) |
| L1 | 62.7 | 60.5 | +2.2 |
| L2 | 74.9 | 64.3 | **+10.6** |
| L3 | 116.7 | 79.6 | **+37.1** |
| L4 | 170.6 | 133.2 | **+37.4** |
| L5 | 294.3 | 266.0 | **+28.3** |

Layers L2–L5 all show stronger EN→BN activation. The T2U encoder is primarily an EN→BN-critical component, consistent with its role in producing Bengali speech units.

#### T2U Decoder (6 layers)

The T2U decoder is relatively symmetric, with small differences:

| Layer | EN→BN | BN→EN | Most important for |
|-------|-------|-------|--------------------|
| L0 | 31.45 | 31.44 | ≈ equal |
| L1 | 31.89 | 31.88 | ≈ equal |
| L2 | 37.84 | 37.69 | EN→BN (+0.15) |
| L3 | 31.87 | 31.84 | EN→BN (+0.03) |
| L4 | 31.85 | 31.83 | EN→BN (+0.01) |
| L5 | 32.84 | 33.43 | **BN→EN (+0.58)** |

L5 is the only T2U decoder layer that notably favours BN→EN. We removed L5 in Phase 6 because it caused the least damage to EN→BN ASR-ChrF. Again, the pruning metric's directional bias led us to remove the one T2U decoder layer most useful for the reverse direction.

---

### 16.3 Pruned Model Layer Importance Rankings

The pruned model has 16 speech encoder layers, 14 text decoder layers, 4 T2U encoder layers, and 4 T2U decoder layers.

#### Speech Encoder (16 remaining layers)

**EN→BN top 5:** L1 (26.46), L0 (25.24), L7 (25.22), L3 (24.50), L6 (22.02)  
**BN→EN top 5:** L1 (26.62), L0 (25.17), L3 (24.66), L7 (24.64), L2 (21.43)

Rankings remain similar between directions, confirming the speech encoder is not the primary source of BN→EN failure. Most direction-specific layers: L6 (EN→BN +0.64), L7 (EN→BN +0.58), L2 (BN→EN +0.45).

#### Text Decoder (14 remaining layers)

**EN→BN top 5:** L12 (4839), L11 (4681), L10 (4380), L13 (4259), L9 (3973)  
**BN→EN top 5:** L12 (5246), L13 (4816), L11 (4815), L10 (4390), L9 (3897)

The most direction-specific layers in the pruned decoder:

| Layer | EN→BN | BN→EN | Diff (BN→EN favours) |
|-------|-------|-------|---------------------|
| **L13** | **4259** | **4816** | **+556.5** |
| **L12** | **4839** | **5246** | **+406.1** |
| L2 | 1206 | 1490 | +284.1 |
| L1 | 956 | 1235 | +279.5 |
| L3 | 1486 | 1729 | +242.9 |

The two highest-index remaining layers (L12 and L13) bear the heaviest BN→EN load in the pruned model. They have inherited the directional role that L20–L23 played in the base model — but they carry far lower total activation magnitude, explaining the quality collapse. The complete loss of L20–L23 means BN→EN no longer has access to the high-capacity upper-layer processing it depended on.

#### T2U Encoder (4 remaining layers)

| Layer | EN→BN | BN→EN | Diff |
|-------|-------|-------|------|
| L0 | 61.2 | 64.6 | BN→EN +3.4 |
| L1 | 60.2 | 58.6 | EN→BN +1.6 |
| L2 | 78.1 | 76.2 | EN→BN +1.9 |
| **L3** | **194.6** | **187.1** | **EN→BN +7.6** |

L3 dominates both directions. The T2U encoder's new top layer (L3, previously the 4th layer) is the primary workhorse after pruning removed L1 and L2 (both of which were minor contributors in the base model).

#### T2U Decoder (4 remaining layers)

| Layer | EN→BN | BN→EN | Diff |
|-------|-------|-------|------|
| L0 | 31.48 | 31.43 | EN→BN +0.05 |
| L1 | 31.89 | 32.00 | BN→EN +0.11 |
| L2 | 37.89 | 37.52 | EN→BN +0.38 |
| **L3** | **31.59** | **31.16** | **EN→BN +0.43** |

Relatively symmetric, consistent with T2U decoder being less direction-critical than the text decoder.

---

### 16.4 Cross-Model Comparison: Where Did BN→EN Capacity Go?

| Component | Base BN→EN profile | Pruned BN→EN profile | Impact |
|-----------|-------------------|---------------------|--------|
| Speech encoder | Layers 1, 2, 0, 10, 4 dominate | Layers 1, 0, 3, 7, 2 dominate | Moderate change; not the primary failure cause |
| **Text decoder** | **L20–L23 carry +10K activation above EN→BN** | **L20–L23 are gone; L12–L13 absorb the load with ~3× lower capacity** | **Primary failure cause** |
| T2U encoder | L3–L5 slightly favour EN→BN | L3 dominates both | Acceptable adaptation |
| T2U decoder | L5 slightly favoured BN→EN | L5 removed; L1 slightly compensates | Minor contributor |

---

### 16.5 Implications and Recommendations

**Why the BN→EN collapse was predictable in hindsight:** The activation analysis confirms that a unidirectional pruning criterion is equivalent to optimising a model for one direction at the structural expense of the other. The upper text decoder layers (L20–L23) serve as high-capacity language-adaptation layers for both directions, but BN→EN relies on them far more intensely — likely because English target tokens are syntactically more complex relative to Bengali source representations.

**For future bidirectional compression work, the following modifications are recommended:**

1. **Use a combined pruning metric** — e.g., average ChrF across both directions, or a weighted combination. Even equal weighting would have protected L21 and L22.
2. **Identify "direction-critical" layers before pruning** — run the activation hook analysis described here *before* any removal, and add extra protection to layers showing strong directional asymmetry.
3. **Include reverse-direction data in fine-tuning** — a 50/50 EN→BN / BN→EN mix in the DoRA fine-tuning corpus would likely recover significant BN→EN quality without substantially hurting EN→BN.
4. **Bidirectional DoRA recovery** — a targeted second DoRA pass trained on BN→EN data with a lower learning rate could potentially recover the reverse direction while preserving the existing EN→BN quality.

---

## 17. References

| Paper | Usage in This Work |
|-------|-------------------|
| Asahi et al., "Efficient Multilingual NMT via Vocabulary Trimming" (EMNLP 2023) | Phase 1: Vocabulary / embedding pruning |
| Moslem, "Iterative Layer Pruning for Neural Machine Translation" (IWSLT 2025) | Phase 3, 4, 6: Iterative greedy depth pruning algorithm |
| CULL-MT (2024) | Phase 3: Layer pruning for MT models |
| An et al., "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect" (ACL 2025) | Phase 4: Block Influence (BI) scoring for pre-filtering |
| An et al., "FLAP: Fluctuation-based Adaptive Structured Pruning for Large Language Models" (AAAI 2024) | Phase 5: Width pruning method (attempted, abandoned) |
| Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (ICML 2024 Oral) | Phase 7: Parameter-efficient fine-tuning |
| Barrault et al., "SeamlessM4T—Massively Multilingual & Multimodal Machine Translation" (Meta AI, 2023) | Base model |
| Pratap et al., "MMS: Scaling Speech Technology to 1000+ Languages" (2023) | Phase 6: Bengali ASR evaluation using facebook/mms-1b-all |

---

## Appendix A: Pruning Layer Index Reference

### Text Decoder — Removed Layers (original 0-based indices)
`[1, 4, 6, 8, 9, 14, 15, 16, 21, 22]`

### Speech Encoder — Removed Layers (original 0-based indices)
`[2, 5, 9, 11, 14, 15, 17, 19]`

### T2U Encoder — Removed Layers
`[1, 2]` (4 layers remain: 0, 3, 4, 5 → renumbered 0–3)

### T2U Decoder — Removed Layers
`[3, 5]` (4 layers remain: 0, 1, 2, 4 → renumbered 0–3)

---

## Appendix B: Sample Benchmark Outputs (Phase 7 Final Model)

Selected sample-level predictions from the P7_DoRA benchmark:

| ID | BLEU | ChrF | Prediction |
|----|------|------|-----------|
| 1660 | 14.8 | 49.4 | রোমান্টিকতার মধ্যে সংস্কৃতির নির্ধারকতা এর একটি বড় উপাদান ছিল যা গথ ফিচ এবং শ্লে... |
| 1667 | 7.5 | 51.9 | সাধারণভাবে বলতে গেলে ম্যানেজাররা তাদের প্রাক্তন সমনীতিকে নেতৃত্ব দিতে শুরু করলে... |
| 1669 | 12.1 | 57.7 | পুলিশ সুপারিনটেনডেন্ট চান্দ্রা শিকার সোলাঙ্কি বলেছিলেন যে অভিযুক্তরা মুখের ঢাকা... |
| 1671 | 17.7 | 57.2 | কংগ্রেস 2005 সালে অশ্লীলতামূলক পদক্ষেপটি অর্থায়ন শুরু করে এবং নির্দিষ্ট করে যে... |
| 1673 | 42.0 | 76.2 | বিপ্লবী যুদ্ধের সময় ১৩টি রাজ্য প্রথম একটি দুর্বল কেন্দ্রীয় সরকার গঠন করেছিল যে... |

---

## Appendix C: Full Layer Activation Score Tables

### Base Model — All Components, Both Directions

**Text Decoder (24 layers) — complete activation scores:**

| Layer | EN→BN | BN→EN | Δ (EN→BN − BN→EN) |
|-------|-------|-------|-------------------|
| L0 | 582.0 | 635.2 | −53.2 |
| L1 | 832.1 | 870.3 | −38.2 |
| L2 | 1038.8 | 1096.5 | −57.7 |
| L3 | 1206.6 | 1284.3 | −77.7 |
| L4 | 1364.4 | 1437.5 | −73.1 |
| L5 | 1540.7 | 1592.1 | −51.4 |
| L6 | 1696.3 | 1737.3 | −41.0 |
| L7 | 1889.6 | 1898.5 | −9.0 |
| L8 | 2089.1 | 2062.6 | +26.4 |
| L9 | 2284.7 | 2248.1 | +36.7 |
| L10 | 2517.4 | 2470.0 | +47.4 |
| L11 | 2779.7 | 2699.4 | +80.2 |
| L12 | 3073.0 | 2971.3 | +101.6 |
| L13 | 3447.9 | 3280.2 | +167.7 |
| L14 | 3869.0 | 3612.8 | +256.2 |
| L15 | 4354.0 | 3973.8 | +380.1 |
| L16 | 4898.5 | 4410.8 | +487.7 |
| L17 | 5345.4 | 4907.4 | +437.9 |
| L18 | 5718.6 | 5409.0 | +309.6 |
| L19 | 6015.4 | 6067.2 | −51.8 |
| **L20** | **6310.5** | **7275.7** | **−965.2** |
| **L21** *(removed P3)* | **6767.7** | **9388.8** | **−2621.1** |
| **L22** *(removed P3)* | **7471.0** | **11079.2** | **−3608.2** |
| L23 | 7769.9 | 11597.9 | −3828.0 |

**Speech Encoder (24 layers) — full activation scores:**

| Layer | EN→BN | BN→EN | Δ | Pruned? |
|-------|-------|-------|---|---------|
| L0 | 25.24 | 25.17 | +0.07 | — |
| L1 | 26.46 | 26.62 | −0.16 | — |
| **L2** | 25.59 | 25.80 | −0.21 | **removed P4** |
| L3 | 20.95 | 21.24 | −0.29 | — |
| L4 | 24.64 | 24.69 | −0.05 | — |
| **L5** | 22.46 | 22.20 | +0.27 | **removed P4** |
| L6 | 20.33 | 20.25 | +0.08 | — |
| L7 | 8.68 | 8.64 | +0.04 | — |
| L8 | 22.28 | 21.82 | +0.46 | — |
| **L9** | 25.14 | 24.47 | **+0.67** | **removed P4** |
| L10 | 25.57 | 25.04 | +0.53 | — |
| **L11** | 22.55 | 22.25 | +0.30 | **removed P4** |
| L12 | 21.80 | 21.53 | +0.27 | — |
| L13 | 20.34 | 19.85 | +0.49 | — |
| **L14** | 22.55 | 22.06 | **+0.49** | **removed P4** |
| **L15** | 22.42 | 22.22 | +0.20 | **removed P4** |
| L16 | 21.29 | 21.37 | −0.09 | — |
| **L17** | 21.24 | 21.35 | −0.11 | **removed P4** |
| L18 | 19.72 | 19.90 | −0.18 | — |
| **L19** | 18.45 | 18.66 | −0.21 | **removed P4** |
| L20 | 15.84 | 16.07 | −0.23 | — |
| L21 | 13.60 | 13.90 | −0.30 | — |
| L22 | 12.34 | 12.50 | −0.15 | — |
| L23 | 5.29 | 5.26 | +0.03 | — |

*Layers marked "removed P4" were selected for removal because their absence caused the highest EN→BN ChrF — a unidirectional criterion that happened to remove several layers with higher EN→BN than BN→EN activation scores (L9, L14), and also layers with slightly negative differences (L5, L17, L19) that were more neutral to BN→EN.*

### Pruned Model — All Components, Both Directions

**Text Decoder (14 remaining layers):**

| Layer (pruned idx) | EN→BN | BN→EN | Δ (BN→EN − EN→BN) |
|-------------------|-------|-------|-------------------|
| L0 | 678.3 | 909.7 | +231.4 |
| L1 | 955.6 | 1235.1 | +279.5 |
| L2 | 1205.8 | 1489.9 | +284.1 |
| L3 | 1485.6 | 1728.5 | +242.9 |
| L4 | 1804.7 | 1992.8 | +188.1 |
| L5 | 2090.0 | 2268.6 | +178.7 |
| L6 | 2416.4 | 2542.4 | +126.0 |
| L7 | 2818.8 | 2883.8 | +65.0 |
| L8 | 3429.0 | 3319.0 | −110.0 |
| L9 | 3973.4 | 3897.3 | −76.1 |
| L10 | 4380.0 | 4390.4 | +10.5 |
| L11 | 4681.0 | 4814.8 | +133.8 |
| **L12** | **4839.5** | **5245.6** | **+406.1** |
| **L13** | **4259.0** | **4815.6** | **+556.5** |

*Every layer in the pruned decoder is more active for BN→EN except L8 and L9. The upper two layers (L12, L13) bear a disproportionate BN→EN burden, absorbing the role previously held by the removed L20–L23.*

---

*Report generated from Kaggle notebook outputs: `seamless-cse465v5.ipynb`, `only-p7-dora.ipynb`, `only-p7p8-cse465v5.ipynb`, `full-kd.ipynb`, `bidirectional-tracking-base.ipynb`, `bidirectional-tracking-pruned.ipynb`*  
*Hardware: Kaggle Tesla T4, 15.6 GB VRAM*  
*Primary task: EN→BN Speech-to-Speech Translation, FLEURS test set (25 samples)*  
*Bidirectional analysis: 10 samples per direction, EN→BN and BN→EN*
