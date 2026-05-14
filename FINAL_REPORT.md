# Structured Pruning, Recovery, and Deployment of SeamlessM4T v2 for Multilingual Speech-to-Speech Translation

## Title Page

**Project title:** Structured Pruning, Recovery, and Deployment of SeamlessM4T v2 for Multilingual Speech-to-Speech Translation  
**Course:** CSE465  
**Semester:** 261  
**Institution:** North South University  
**Repository:** https://github.com/rayedriasat/Pruning-SeamlessM4T  

**Group members:**

| Name | Student ID | Email | Primary contribution |
| --- | --- | --- | --- |
| Rayed Riasat | TODO | TODO | Compression pipeline, final multilingual notebook, local inference, duplex web app |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

## Abstract

This project studies practical compression of Meta's SeamlessM4T v2 Large speech-to-speech translation model for lower-resource multilingual deployment. SeamlessM4T v2 Large is a powerful multilingual speech translation system, but the speech-to-speech variant used in this project has approximately 1.805B parameters and multiple coupled components: a speech encoder, text decoder, text-to-unit (T2U) model, and vocoder. The project goal was to reduce parameter count and inference cost while preserving usable speech translation quality for English, Bengali, Mandarin, Arabic, and Hindi.

The final approach combines language-scoped vocabulary pruning, Block-Influence-guided depth pruning, conservative T2U layer merging, text decoder pruning, checkpointed recovery fine-tuning, sparse top-k teacher-student knowledge distillation, and LoRA/DoRA-style adapter recovery. The final selected multilingual model has approximately 1.0879B parameters, a 39.7% reduction from the baseline. On the saved eight-direction multilingual ASR benchmark, it achieves 33.73 ChrF, 8.16 BLEU, and 0.1336 real-time factor (RTF), compared with the 1.8055B baseline's 46.49 ChrF, 15.88 BLEU, and 0.2455 RTF. This gives a 1.84x RTF speedup and makes the model practical on a local RTX 3050 Laptop GPU with 4.3 GB VRAM.

The project also produced important negative findings. A width-pruning attempt based on FLAP collapsed generation quality after previous depth pruning, and an aggressive sub-500M branch showed that removing too much text decoder capacity destroys translation quality. These failures became part of the methodology: they defined a safe compression boundary and clarified which parts of SeamlessM4T can be pruned structurally. The final deliverables include the final training notebooks, a local inference notebook, a duplex web application with WebSocket audio streaming and boundary detection, this report, a mandatory appendix, and a from-scratch reproduction prompt for an automated coding agent.

## 1. Introduction

Speech-to-speech translation is valuable when users need direct spoken communication across languages without reading an intermediate transcript. However, modern multilingual speech translation systems are large. SeamlessM4T v2 Large is designed for broad language coverage and high quality, not for constrained deployment. Running it on a consumer laptop GPU, a classroom demo machine, or a lightweight web backend requires compression.

The project began with a broad question:

Can a large multilingual speech-to-speech model be structurally pruned enough to run in a practical local/web setting while still producing intelligible multilingual speech translation?

The project focused on five languages:

| Language | SeamlessM4T code | FLEURS code | Role in project |
| --- | --- | --- | --- |
| English | `eng` | `en_us` | Main pivot/source and target language |
| Bengali | `ben` | `bn_in` | Main low-resource target and local demo focus |
| Mandarin Chinese | `cmn` | `cmn_hans_cn` | Non-Latin multilingual stress test |
| Arabic | `arb` | `ar_eg` | Non-Latin multilingual stress test |
| Hindi | `hin` | `hi_in` | Indic language coverage |

The final model is not just a notebook result. The work also includes:

- a local inference notebook tested on an NVIDIA GeForce RTX 3050 Laptop GPU with 4.3 GB VRAM;
- a duplex browser web app using FastAPI, WebSockets, browser AudioWorklets, boundary detection, streamed audio playback, and barge-in cancellation;
- a documented failed sub-500M branch, useful because it reveals a real lower bound of structural pruning for this architecture;
- an ONNX conversion exploration notebook for future deployment work.

## 2. Problem Definition

### 2.1 Task

The task is multilingual speech-to-speech translation (S2ST):

1. Input: source-language speech waveform at 16 kHz.
2. Output: target-language speech waveform at 16 kHz.
3. Intermediate: generated target-language text tokens and discrete speech units.

The final benchmark evaluates eight directions:

| Direction | Description |
| --- | --- |
| `arb -> eng` | Arabic speech to English speech |
| `ben -> eng` | Bengali speech to English speech |
| `cmn -> eng` | Mandarin speech to English speech |
| `hin -> eng` | Hindi speech to English speech |
| `eng -> arb` | English speech to Arabic speech |
| `eng -> ben` | English speech to Bengali speech |
| `eng -> cmn` | English speech to Mandarin speech |
| `eng -> hin` | English speech to Hindi speech |

### 2.2 Constraints

The project had practical constraints:

- Training and pruning had to survive notebook session limits, so checkpointing and Drive/rclone persistence were mandatory.
- Pruning decisions had to be measurable with small evaluation subsets because full S2ST evaluation is slow.
- The final model had to be deployable locally and in a browser-backed app.
- The compression had to be structural, not just unstructured sparsity, because the goal was smaller model size and faster inference without requiring special sparse kernels.

### 2.3 Main contributions

The project contributes:

1. A complete structural pruning and recovery pipeline for SeamlessM4T v2 S2ST.
2. A final compressed multilingual model around 1.09B parameters.
3. A component-specific evaluation strategy: text/ASR metrics for encoder-decoder pruning and ASR metrics for T2U/audio-path pruning.
4. A detailed failure analysis of width pruning and sub-500M compression.
5. Local inference verification on a 4 GB-class laptop GPU.
6. A duplex web application demonstrating interactive speech translation.

## 3. Background: SeamlessM4T v2 Architecture

The HuggingFace `SeamlessM4Tv2ForSpeechToSpeech` path used in this project is organized as:

```text
Input audio
  -> Speech encoder
  -> Text decoder
  -> Target text tokens
  -> T2U model
  -> Discrete acoustic units
  -> Vocoder
  -> Output audio
```

A key early discovery was that the `text_encoder` weights in the full checkpoint are not used by the speech-to-speech model class. The notebooks showed `UNEXPECTED` text encoder key warnings during model load; this is not an error for S2ST. It means the text encoder is not instantiated and therefore cannot be counted as an active compression target in this setting.

The active baseline component structure was approximately:

| Component | Role | Approximate parameters | Compression status |
| --- | --- | ---: | --- |
| Speech encoder | Converts speech features to hidden states | about 635M | Pruned from 24 to 14 layers |
| Text decoder | Generates target text tokens | very large, dominant component | Pruned from 24 to 14 layers |
| Shared embedding / LM head | Multilingual token representation and logits | large because vocabulary covers many languages | Vocabulary-trimmed |
| T2U model | Converts text tokens to speech units | about 262M | Merged/pruned conservatively |
| Vocoder | Generates waveform from units | about 41.9M | Kept intact |

## 4. Literature Review

### 4.1 SeamlessM4T v2 and speech-to-speech translation

The Seamless paper introduced a family of multilingual expressive and streaming speech translation systems, including SeamlessM4T v2. The paper emphasizes multilingual S2ST, improved low-resource language coverage, expressive speech translation, and streaming translation. This project uses SeamlessM4T v2 Large as the base model because it already supports the required speech-to-speech path and the target languages.

The relevant insight from the Seamless architecture is that S2ST is not a single transformer. It is a pipeline with a speech encoder, text generation stage, unit generation stage, and vocoder. A compression strategy must therefore respect component boundaries. A layer removal that is safe in the text decoder may be unsafe in T2U, and a metric that sees text quality may not see audio damage.

### 4.2 Vocabulary trimming for multilingual models

Ushio, Zhou, and Camacho-Collados (Findings of EMNLP 2023) proposed vocabulary trimming for multilingual language model compression. Their core observation is that multilingual models have large embedding matrices because their vocabularies cover many languages, but a target use case usually needs only a subset. Removing irrelevant vocabulary items can reduce model size without retraining the whole model.

This project adapts that idea to SeamlessM4T. Since the final task uses only English, Bengali, Mandarin, Arabic, and Hindi, the tokenizer vocabulary can be scanned for those languages and reduced. The implementation had to be careful because SeamlessM4T ties shared embeddings and the LM head, uses language-token mappings in generation config, and needs intermediate token IDs to be decoded with the original tokenizer. The notebooks therefore preserve a `_vocab_remap_to_old` table.

### 4.3 Layer redundancy and Block Influence

ShortGPT studies the observation that large transformer models often contain redundant layers. It proposes Block Influence (BI), computed using the cosine similarity between a layer's input and output. If a layer changes its input very little, it may be less important for the model's behavior.

This project used the same intuition for speech encoder and text decoder pruning:

```text
BI(layer) = 1 - cosine_similarity(layer_input, layer_output)
```

Low-BI layers were treated as candidate redundant layers. However, the project did not blindly remove the lowest-BI layers. It used BI as a pre-filter, then evaluated candidate removals with translation quality metrics. This hybrid approach reduced the number of expensive candidate evaluations while still guarding against destructive pruning.

### 4.4 Iterative layer pruning for machine translation

Recent machine translation compression work, including CULL-MT and iterative layer pruning approaches, motivates structural pruning of selected language directions and transformer layers. The important practical idea is not to remove many layers in one step. Instead, remove one candidate, measure quality, keep the least damaging removal, then repeat.

This became the core pruning loop:

```text
for each pruning iteration:
    build candidate set
    temporarily remove each candidate layer
    run a small evaluation
    choose the candidate with the least quality loss
    permanently remove that layer
    sync configuration and layer indices
    save checkpoint
```

The checkpoints were important. The full pipeline required long sessions, and every pruning iteration could be expensive.

### 4.5 FLAP and structured width pruning

FLAP is a structured pruning method for LLMs that removes dimensions/neurons rather than individual unstructured weights. In principle, this is attractive because structured width pruning can create smaller dense matrices, which may produce real speedups on normal hardware.

The project tested this idea and found a negative result. On the already depth-pruned model, FLAP-style FFN width pruning caused catastrophic generation quality collapse. This matters because it shows that after depth pruning, the remaining layers likely depend strongly on their remaining FFN capacity. The project therefore abandoned width pruning in the final approach.

### 4.6 Wanda and activation-aware pruning

Wanda proposes pruning by combining weight magnitudes and activations. Although this project did not make Wanda the final pruning method, it influenced the broader thinking: pruning should be activation-aware, not just magnitude-based. This was consistent with the project's use of calibration samples, forward hooks, layer inputs/outputs, and candidate evaluation.

### 4.7 DPHuBERT and speech model distillation

DPHuBERT combines distillation and pruning for self-supervised speech models. Its relevance is that speech models are not simply text models with audio input; their intermediate representations carry acoustic information, and pruning should be paired with recovery. This project applies a similar high-level principle: structural compression is followed by teacher-student recovery, especially after aggressive decoder pruning.

### 4.8 Knowledge distillation and PEFT recovery

Knowledge distillation trains a smaller or pruned student to imitate a larger teacher. This project used a full SeamlessM4T v2 teacher and a compressed student. The final notebooks implemented sparse top-k teacher logit distillation because full teacher logits over the original vocabulary were too large and memory-expensive.

DoRA and LoRA-style parameter-efficient fine-tuning influenced the recovery stage. DoRA decomposes weight adaptation into magnitude and direction components and aims to approximate full fine-tuning behavior with fewer trainable parameters. In the project, adapter-based recovery was essential because full fine-tuning of a 1B+ parameter S2ST model would have been too expensive.

### 4.9 MMS ASR for audio-domain evaluation

The MMS project scales speech recognition and synthesis to over 1,000 languages. This mattered because the project needed an ASR system that could transcribe generated Bengali, Hindi, Arabic, and Mandarin speech. The older report records that Whisper-medium was tried for Bengali but produced unusable or English-biased output. MMS ASR was therefore used for ASR-ChrF/ASR-BLEU style evaluation where audio output quality mattered.

## 5. Experimental Setup

### 5.1 Hardware

Training and pruning were primarily performed in notebook environments with NVIDIA Tesla T4 GPUs. The final multilingual notebook detected:

| Hardware item | Value |
| --- | --- |
| GPU count | 2 |
| GPU model | Tesla T4 |
| VRAM per GPU | 15.6 GB |
| Framework | PyTorch CUDA |

Local inference was verified in `seamless_local/seamless_local_inference.ipynb`:

| Hardware item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4.3 GB |
| Model loaded | `phase7_dora_merged_v1` |
| Reported memory after load | about 2.10 GB allocated / 2.12 GB reserved |

### 5.2 Data

The main dataset was FLEURS. The final multilingual benchmark used 25 samples per language pair across eight pairs, for 200 samples total. For training and calibration, the notebooks used FLEURS train/test splits and cached Parquet shards.

The notebooks include several important data engineering choices:

- Use of HuggingFace/Kaggle Parquet paths instead of repeatedly downloading full datasets.
- Chunked streaming dataset loading to avoid keeping all audio in RAM.
- Audio loading from Parquet cells storing WAV bytes.
- Language-code mapping between FLEURS and SeamlessM4T.
- Saved checkpoints of benchmark summaries and detailed per-pair results.

### 5.3 Software

The project used:

- Python and Jupyter notebooks.
- PyTorch and CUDA.
- HuggingFace Transformers for SeamlessM4T v2.
- Datasets/Parquet tooling for FLEURS.
- SacreBLEU for BLEU and ChrF.
- PEFT-style LoRA/DoRA adapters.
- Torchaudio, Librosa, SoundFile, and NumPy for audio processing.
- FastAPI/WebSocket backend for the duplex app.
- Browser AudioWorklet frontend for microphone capture.
- rclone/Google Drive persistence for long-running notebook checkpoints.

## 6. Metrics and Justification

### 6.1 Parameter count

Parameter count is the main compression metric. It measures actual structural reduction and is more meaningful for this project than unstructured sparsity because the goal is to run a smaller dense model.

### 6.2 ChrF

ChrF is the primary translation quality metric because it is character-n-gram based and better suited to scripts and morphologies where tokenization can be brittle. This is important for Bengali, Arabic, Hindi, and Mandarin.

### 6.3 BLEU

BLEU is included because it is widely used in machine translation reporting. It is not the only metric because it can be harsh for morphologically rich and non-Latin languages, but it provides a standard reference.

### 6.4 ASR-ChrF and ASR-BLEU

For speech-to-speech output, the generated waveform is transcribed by ASR and compared to reference text. This is essential for T2U and vocoder-facing changes. Text-only metrics can completely miss audio-path failure because T2U sits after the text decoder.

### 6.5 RTF

Real-Time Factor (RTF) is inference time divided by audio duration. Lower is better. An RTF below 1.0 means faster-than-real-time generation. The final model's 0.1336 RTF shows practical interactive potential.

### 6.6 VRAM

VRAM was tracked because the project explicitly targets deployability. A model that scores well but cannot run on available hardware is not useful for the project goal.

## 7. Final Methodology: The v3 Multilingual Approach

This section describes the final successful approach. Earlier versions are discussed later as ablations and failure analysis.

### 7.1 Phase 0: Baseline capture

The baseline was `facebook/seamless-m4t-v2-large` loaded in HuggingFace speech-to-speech mode. The model was benchmarked over the multilingual FLEURS evaluation set.

Baseline output from the final notebook:

| Metric | Value |
| --- | ---: |
| Parameters | 1805.5M |
| ChrF | 46.49 |
| BLEU | 15.88 |
| RTF | 0.2455 |

This established the quality and speed target.

### 7.2 Phase 1: Five-language vocabulary pruning

The project trimmed the multilingual vocabulary to the five-language target set. The method:

1. Scan target-language corpora.
2. Keep observed token IDs.
3. Always keep special tokens and language tags.
4. Create a new shared embedding with only kept IDs.
5. Rebuild/tie text decoder embeddings and LM head.
6. Remap generation config language/token IDs.
7. Save `_vocab_remap_to_old` for decoding.

Final notebook result:

| Metric | Baseline | P1 |
| --- | ---: | ---: |
| Parameters | 1805.5M | 1566.6M |
| ChrF | 46.49 | 41.74 |
| BLEU | 15.88 | 13.65 |
| RTF | 0.2455 | 0.2435 |

This phase removed about 239M parameters. The quality drop was visible in the multilingual benchmark, especially for Mandarin target output, but vocabulary trimming was still the largest low-risk structural reduction.

### 7.3 Phase 2: Speech encoder pruning from 24 to 16 layers

The speech encoder was pruned with a BI-guided iterative loop:

1. Register forward hooks on each speech encoder layer.
2. Compute BI scores over calibration samples.
3. Protect first, middle, and last layers.
4. Evaluate only the bottom BI candidates.
5. Remove the candidate whose removal gives the best validation ChrF.
6. Sync model config and save a checkpoint.

Final notebook result:

| Metric | P1 | P2 |
| --- | ---: | ---: |
| Parameters | 1566.6M | 1373.1M |
| ChrF | 41.74 | 38.97 |
| BLEU | 13.65 | 11.13 |
| RTF | 0.2435 | 0.1617 |

This phase provided a large speed improvement. RTF improved from 0.2435 to 0.1617.

### 7.4 Phase 3: T2U LaCo/RDSC merge

The T2U module is smaller than the speech encoder and text decoder but very sensitive because it controls the audio path. The final notebook used a layer merging strategy rather than aggressive deletion. The method merged adjacent layers only when similarity was high enough, using a conservative threshold and re-indexing T2U layers afterward.

Final notebook result:

| Metric | P2 | P3 |
| --- | ---: | ---: |
| Parameters | 1373.1M | 1331.2M |
| ChrF | 38.97 | 38.47 |
| BLEU | 11.13 | 11.21 |
| RTF | 0.1617 | 0.1646 |

The small BLEU increase is within noise, but the important result is that T2U parameters were reduced with little additional quality loss.

### 7.5 Phase 4: Additional speech encoder pruning from 16 to 14 layers

After T2U merging, two more speech encoder layers were removed using the same BI-guided iterative strategy.

Final notebook result:

| Metric | P3 | P4 |
| --- | ---: | ---: |
| Parameters | 1331.2M | 1282.8M |
| ChrF | 38.47 | 35.74 |
| BLEU | 11.21 | 9.67 |
| RTF | 0.1646 | 0.1635 |

This phase was more costly in quality than Phase 2. It showed that the speech encoder still had some redundancy, but the easy savings had already been taken.

### 7.6 Phase 5: Text decoder pruning from 24 to 14 layers

The text decoder is one of the largest and most quality-critical components. The final pipeline pruned it aggressively but not to the extreme sub-500M target. The method:

1. Register hooks on decoder layers.
2. Compute decoder BI.
3. Protect first, middle, and last decoder layers.
4. Iteratively evaluate candidate removals.
5. Remove 10 layers.
6. Rebuild config and save as `phase5_dec_14L`.

Final notebook result:

| Metric | P4 | P5 |
| --- | ---: | ---: |
| Parameters | 1282.8M | 1030.9M |
| ChrF | 35.74 | 25.32 |
| BLEU | 9.67 | 5.83 |
| RTF | 0.1635 | 0.1881 |

This was the largest quality cliff in the final successful pipeline. It confirmed that the text decoder has limited redundancy compared with the vocabulary and first encoder pruning steps. The phase was still kept because it enabled the final near-1B model, but it required recovery.

### 7.7 Phase 6: Knowledge-distillation recovery

Phase 6 recovered the P5 model using teacher-student training:

- Teacher: full SeamlessM4T v2 model.
- Student: pruned `phase5_dec_14L`.
- Student adapters: LoRA/DoRA-style low-rank trainable modules.
- Loss: cross-entropy on target labels plus sparse KL divergence on teacher top-k logits.
- Important fix: teacher full-vocabulary logits had to be remapped into the student's pruned vocabulary.

The notebook records:

- student vocabulary size around 22,767 in the recovery run;
- teacher vocabulary size 256,102;
- trainable parameter count around 20.58M in one Phase 6 setup;
- sparse KL loss with sentinel handling for unmapped tokens;
- best intermediate multilingual ChrF noted as 38.17 around step 1120 in the Phase 6 line of work.

Final stored P6 benchmark:

| Metric | P5 | P6 |
| --- | ---: | ---: |
| Parameters | 1030.9M | about 1.03B |
| ChrF | 25.32 | 33.07 |
| BLEU | 5.83 | 7.95 |
| RTF | 0.1881 | 0.1484 |

Phase 6 recovered +7.75 ChrF over P5 and also improved RTF.

### 7.8 Phase 7: Final hybrid LoRA/DoRA recovery

Phase 7 further refined the recovered model. The notebook records:

- loading `phase6_kd_merged` directly to CUDA;
- disabling gradient checkpointing after inference load when needed;
- final trainable parameter count around 152.7M in the hybrid recovery setup;
- separate optimizer groups and fp32 trainable parameters for stable GradScaler behavior;
- best checkpoint reported as step 1300 with ASR-ChrF 41.08 in one training run;
- final merged deployable model saved and benchmarked.

Stored final benchmark:

| Metric | P6 | P7 final |
| --- | ---: | ---: |
| Parameters | about 1.03B | 1087.9M |
| ChrF | 33.07 | 33.73 |
| BLEU | 7.95 | 8.16 |
| RTF | 0.1484 | 0.1336 |

This is the final selected model.

## 8. Final Results

### 8.1 Phase-by-phase summary

| Phase | Method | Parameters | ChrF | BLEU | RTF |
| --- | --- | ---: | ---: | ---: | ---: |
| P0 | Baseline SeamlessM4T v2 S2ST | 1805.5M | 46.49 | 15.88 | 0.2455 |
| P1 | Five-language vocabulary pruning | 1566.6M | 41.74 | 13.65 | 0.2435 |
| P2 | Speech encoder pruning, 24 -> 16 layers | 1373.1M | 38.97 | 11.13 | 0.1617 |
| P3 | T2U LaCo/RDSC merge | 1331.2M | 38.47 | 11.21 | 0.1646 |
| P4 | Speech encoder pruning, 16 -> 14 layers | 1282.8M | 35.74 | 9.67 | 0.1635 |
| P5 | Text decoder pruning, 24 -> 14 layers | 1030.9M | 25.32 | 5.83 | 0.1881 |
| P6 | KD recovery | about 1.03B | 33.07 | 7.95 | 0.1484 |
| P7 | Final hybrid adapter recovery | 1087.9M | 33.73 | 8.16 | 0.1336 |

### 8.2 Final per-pair results

| Pair | Samples | ChrF | BLEU | RTF |
| --- | ---: | ---: | ---: | ---: |
| `arb -> eng` | 25 | 39.62 | 10.42 | 0.1241 |
| `ben -> eng` | 25 | 34.40 | 6.11 | 0.0969 |
| `cmn -> eng` | 25 | 35.62 | 7.33 | 0.1288 |
| `eng -> arb` | 25 | 32.96 | 6.51 | 0.1144 |
| `eng -> ben` | 25 | 39.32 | 8.49 | 0.1409 |
| `eng -> cmn` | 25 | 5.61 | 2.29 | 0.2529 |
| `eng -> hin` | 25 | 44.68 | 15.41 | 0.1014 |
| `hin -> eng` | 25 | 37.65 | 8.75 | 0.1094 |

The final model is strongest on `eng -> hin`, `arb -> eng`, and `eng -> ben`. The weakest direction is `eng -> cmn`, which remains a known limitation.

### 8.3 Compression and speed

| Quantity | Value |
| --- | ---: |
| Baseline parameters | 1805.5M |
| Final parameters | 1087.9M |
| Parameter reduction | 39.7% |
| Baseline RTF | 0.2455 |
| Final RTF | 0.1336 |
| RTF speedup | 1.84x |
| ChrF retention | 72.6% |
| BLEU retention | 51.4% |

The final model trades quality for size and speed. It does not preserve all baseline quality, but it crosses the practical deployment threshold on local hardware.

## 9. Ablation History and Failed Attempts

### 9.1 Older EN-BN v1 pipeline

The older `RESEARCH_REPORT.md` documents an EN-BN-focused pipeline with:

- baseline: 1805.5M parameters, 50.52 ChrF, 11.63 BLEU, 0.268 RTF;
- vocabulary pruning to about 1564.2M parameters with 49.07 ChrF;
- text decoder pruning to 14 layers;
- speech encoder pruning to 16 layers;
- T2U pruning;
- DoRA recovery to about 45.14 ChrF and 10.20 BLEU at 1039.1M parameters.

This earlier version showed that strong EN-BN recovery was possible, but it also exposed bidirectional problems: pruning only for EN-BN damaged BN-EN performance. The final multilingual work expanded the benchmark and training focus to five languages and eight directions.

### 9.2 FLAP width pruning failure

The v1 experiments attempted FLAP-style FFN width pruning. Two results were important:

1. FLAP on the base model gave only modest parameter reduction and large quality loss.
2. FLAP after depth pruning caused catastrophic generation collapse.

The older report records that FLAP on the P4 model reduced parameters from 1118.8M to 1057.2M but dropped ChrF from 40.11 to 9.20 and BLEU from 8.19 to 0.95. Outputs showed token repetition loops and unstable generation. The final methodology therefore avoids width pruning after depth pruning.

### 9.3 Sub-500M branch failure

The `mission500m/` branch attempted to compress under 500M parameters. The plan was ambitious:

- heavily reduce the text decoder;
- heavily reduce the speech encoder;
- prune T2U to 8 total layers;
- recover with DoRA and KD.

The project learned that this was too aggressive. One notebook output showed an intermediate model around 1213.9M parameters with only 0.73 ChrF and 0.04 BLEU after destructive pruning. The conceptual failure was that the large text decoder could not be replaced or bypassed with a simple CIF-style shortcut. The text decoder carries essential multilingual sequence modeling capacity.

This failure changed the final design: prune only 10 text decoder layers, keep a 14-layer text decoder, and accept a final size near 1.09B rather than forcing an unusable 500M model.

### 9.4 Directional bias and layer activation findings

The older report includes a valuable bidirectional activation analysis. It found that upper text decoder layers in the base model are disproportionately important for BN-EN. The EN-BN-only pruning removed some of those upper layers because they seemed safe for EN-BN, causing BN-EN generation failures.

This result motivated the final multilingual approach:

- evaluate multiple directions rather than only one pair;
- preserve first/middle/last layers;
- avoid over-aggressive text decoder pruning;
- use per-pair result tables rather than a single aggregate metric.

## 10. Local Inference System

The `seamless_local/` folder turns the notebook result into a local test workflow. It includes:

- `seamless_local_inference.ipynb`: local loading, EN-BN and BN-EN evaluation, saved audio samples;
- setup instructions using `uv`;
- model loading repairs for layer-count mismatches;
- local FLEURS Parquet loading;
- optional microphone translation cells.

Important local engineering details:

1. The model loader reads local model folders instead of pulling from Drive.
2. It repairs `decoder_layers` and `speech_encoder_layers` from actual weight keys if config metadata is stale.
3. It supports `_custom_state.pt` and vocabulary remapping.
4. It loads audio from Parquet bytes and resamples to 16 kHz.
5. It verifies that the final compressed model can run on an RTX 3050 Laptop GPU.

The notebook output shows:

```text
GPU  : NVIDIA GeForce RTX 3050 Laptop GPU
VRAM : 4.3 GB
GPU mem after model load: about 2.10 GB allocated / 2.12 GB reserved
```

This is one of the project's strongest practical results: the final model is not just theoretically smaller, it actually runs on a 4 GB-class laptop GPU.

## 11. Duplex Web Application

The `duplexWebApp/` folder implements an interactive speech translation application using the compressed model. It is not a thin wrapper; it required substantial engineering to make the model usable in a conversational browser setting.

### 11.1 Architecture

```text
Browser microphone
  -> AudioWorklet captures 16 kHz PCM
  -> WebSocket binary frames
  -> FastAPI backend session
  -> Boundary VAD over SeamlessM4T speech encoder
  -> Utterance boundary detection
  -> SeamlessM4T compressed model translation
  -> Output speech chunks
  -> Browser scheduled playback
```

### 11.2 Frontend engineering

The frontend in `frontend/app.js`, `frontend/index.html`, and `frontend/worklet.js` includes:

- browser microphone capture using `AudioContext` at 16 kHz;
- `AudioWorklet` conversion to int16 PCM frames;
- push-to-talk mode and always-listen mode;
- Ctrl+D/hold-to-talk interaction;
- microphone level metering;
- WebSocket binary streaming;
- scheduled output audio playback;
- playback queue flushing for barge-in;
- silence-tail sending after push-to-talk release so the server can close the turn.

The silence-tail design is important. If the browser simply stops streaming when the user releases push-to-talk, the server may not observe trailing silence and may wait too long to finish the utterance. Sending a short silence tail gives the boundary detector the evidence it needs to close the turn.

### 11.3 Backend engineering

The FastAPI backend includes:

- `backend/app.py`: app lifecycle, model loading, health route, WebSocket route;
- `backend/model.py`: model wrapper for compressed SeamlessM4T inference;
- `backend/boundary_vad.py`: CIF boundary detector adapter on encoder hidden states;
- `backend/session.py`: per-connection state machine.

Important backend engineering decisions:

1. **Shared speech encoder:** the boundary adapter uses the same loaded SeamlessM4T speech encoder instead of loading a second speech model.
2. **Inference lock:** VAD and translation share a lock to prevent overlapping GPU-heavy passes.
3. **Rolling buffer:** audio chunks are appended cheaply and only concatenated for boundary checks.
4. **Minimum new audio threshold:** avoids running the encoder too frequently.
5. **Sustained speech-start detection:** avoids false speech starts from noise.
6. **Stricter start threshold during bot speech:** reduces self-barge-in from echo-cancelled playback leakage.
7. **Barge-in cancellation:** if user speech starts while bot audio is streaming, generation/playback is cancelled and the browser receives a `stop_playback` event.
8. **Chunked output:** translated audio is streamed back in 100 ms chunks.
9. **CUDA cache hygiene:** stale reserved cache is compacted only after translation when it grows large enough, avoiding allocator churn.
10. **Intermediate text decode:** pruned vocabulary IDs are remapped back to original tokenizer IDs before decoding text.

### 11.4 Boundary detector

The boundary detector is a 3-layer MLP over SeamlessM4T speech encoder hidden states:

```text
1024 input
  -> Linear 512
  -> LayerNorm
  -> ReLU
  -> Dropout
  -> Linear 256
  -> LayerNorm
  -> ReLU
  -> Dropout
  -> Linear 1
  -> Sigmoid
```

A frame is treated as silence when the probability is below the configured threshold, 0.2 by default. The session requires enough trailing silence before dispatching an utterance. This creates a practical turn-taking layer on top of the translation model.

## 12. ONNX Conversion Exploration

`Final Notebooks/v3_WIP onnx-conversion.ipynb` explores export/deployment possibilities. It is not part of the final submitted model, but it shows the next engineering direction: making the model easier to package and run outside notebooks. SeamlessM4T's multi-stage generation path makes ONNX conversion non-trivial because speech encoding, text token generation, T2U unit prediction, and vocoder synthesis have different dynamic shapes and generation semantics.

## 13. Discussion

### 13.1 What worked

Vocabulary pruning was the cleanest win. It removed hundreds of millions of parameters with limited engineering risk. BI-guided encoder pruning also worked well, producing major speedups before quality loss became severe. Recovery fine-tuning was essential and proved that a pruned model can regain useful quality.

### 13.2 What did not work

Width pruning did not work in this setting after depth pruning. The sub-500M target was too aggressive. The project found that the text decoder cannot be treated as disposable. It is the model's main multilingual sequence modeling engine.

### 13.3 Why final model size is 1.09B, not 500M

The final model is larger than the original ambition, but it is the correct engineering decision. The failed 500M branch showed that smaller is not automatically better if quality collapses. A 1.09B model that runs locally and translates intelligibly is more valuable than a 500M model that cannot produce usable output.

### 13.4 Limitations

- English-to-Mandarin remains weak.
- The final model still loses significant BLEU/ChrF compared with baseline.
- The final pipeline is notebook-based rather than a clean one-command training script.
- Large model artifacts are external and not stored directly in Git.
- ONNX conversion is incomplete.
- The duplex web app currently assumes local model artifacts are present.

## 14. Conclusion

This project successfully compressed, recovered, and deployed SeamlessM4T v2 Large for multilingual speech-to-speech translation. The final selected model reduces parameters from 1.8055B to 1.0879B, improves RTF from 0.2455 to 0.1336, and runs on a 4 GB-class laptop GPU. It also powers a duplex browser application with streaming microphone input, boundary detection, translated audio playback, and barge-in handling.

The project's most important research findings are:

1. Language-scoped vocabulary pruning is highly effective for multilingual deployment.
2. BI-guided iterative depth pruning is useful but must be validated with task metrics.
3. T2U/audio components require ASR-based metrics.
4. Post-pruning KD/adapter recovery is necessary after aggressive pruning.
5. Width pruning after depth pruning can collapse generation.
6. The text decoder is the main limiting factor for extreme compression.
7. Deployment engineering matters: a compressed model is only useful if it can be loaded, streamed, and interacted with.

The final work is therefore both a model-compression study and a practical deployment project.

## 15. References

1. Barrault et al. Seamless: Multilingual Expressive and Streaming Speech Translation. arXiv:2312.05187.
2. Ushio, Zhou, and Camacho-Collados. Efficient Multilingual Language Model Compression through Vocabulary Trimming. Findings of EMNLP 2023.
3. An et al. ShortGPT: Layers in Large Language Models are More Redundant Than You Expect. Findings of ACL 2025.
4. An et al. FLAP: Fluctuation-based Adaptive Structured Pruning for Large Language Models. AAAI 2024.
5. Sun et al. A Simple and Effective Pruning Approach for Large Language Models (Wanda). ICLR 2024.
6. Peng et al. DPHuBERT: Joint Distillation and Pruning of Self-Supervised Speech Models. Interspeech 2023.
7. CULL-MT: Compression Using Language and Layer Pruning for Machine Translation. arXiv:2411.06506.
8. Moslem et al. Iterative Layer Pruning for Efficient Translation Inference. WMT 2025.
9. Liu et al. DoRA: Weight-Decomposed Low-Rank Adaptation. ICML 2024.
10. Pratap et al. Scaling Speech Technology to 1,000+ Languages. arXiv:2305.13516.

## Mandatory Appendix - 1

The mandatory appendix is provided separately in `APPENDIX_1.md`. It includes the GitHub repository link, final file inventory, and a detailed from-scratch reproduction prompt for an automated coding agent.

