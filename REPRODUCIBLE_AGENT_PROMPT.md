# From-Scratch Reproduction Prompt for an Agent Like OpenCode

You are an autonomous research/coding agent. Your task is to reproduce a final compressed SeamlessM4T v2 multilingual speech-to-speech translation system and its duplex web application from scratch.

Important assumption: you do not have access to the original project notebooks, saved notebook outputs, or hidden checkpoint files. You only know the target methodology and expected outcomes described in this prompt. You must implement the pipeline yourself, create clean scripts/notebooks, run experiments, save checkpoints, and report deviations.

## Goal

Build a structurally compressed and recovered version of `facebook/seamless-m4t-v2-large` for multilingual speech-to-speech translation over five languages:

- English: `eng`, FLEURS `en_us`
- Bengali: `ben`, FLEURS `bn_in`
- Mandarin Chinese: `cmn`, FLEURS `cmn_hans_cn`
- Arabic: `arb`, FLEURS `ar_eg`
- Hindi: `hin`, FLEURS `hi_in`

The final target model should be close to:

- 1.09B parameters
- 14 speech encoder layers
- 14 text decoder layers
- conservative T2U compression/merge
- pruned vocabulary for the five-language task
- recovered with teacher-student KD and LoRA/DoRA-style adapters
- capable of local inference and browser-based duplex speech translation

Expected final benchmark ballpark:

| Metric | Baseline target | Final target |
| --- | ---: | ---: |
| Parameters | about 1805.5M | about 1088M |
| ChrF | about 46.5 | about 33-35 |
| BLEU | about 15.9 | about 8-9 |
| RTF | about 0.245 | about 0.13-0.15 |

Small deviations are acceptable if you document them. Do not force the model below 500M if quality collapses.

## Deliverables

Create the following deliverables:

1. `reproduce/requirements.txt` or `pyproject.toml`
2. `reproduce/prepare_data.py`
3. `reproduce/model_utils.py`
4. `reproduce/metrics.py`
5. `reproduce/prune_vocab.py`
6. `reproduce/prune_layers.py`
7. `reproduce/recover_kd.py`
8. `reproduce/benchmark.py`
9. `reproduce/local_inference.py`
10. `webapp/` with a FastAPI/WebSocket backend and browser frontend
11. `artifacts/results/final_results.md`
12. `artifacts/results/phase_summary.csv`
13. Saved model folders for each phase, or checkpoint instructions if storage is external

You may implement as scripts, notebooks, or both. Prefer scripts for reproducibility.

## Hardware and Environment

Use a CUDA GPU environment. Recommended:

- Training/pruning: two NVIDIA T4 GPUs or better, 15 GB VRAM each.
- Local inference validation: any CUDA GPU with 4 GB or more VRAM.

Install:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft sentencepiece sacrebleu evaluate librosa soundfile pandas pyarrow numpy scipy matplotlib seaborn fastapi uvicorn websockets python-multipart safetensors
```

If your CUDA version differs, install the correct PyTorch wheel for your environment.

## Data Preparation

Use FLEURS from HuggingFace Datasets.

Create train/eval data for these language pairs:

```python
EVAL_PAIRS = [
    ("ar_eg", "en_us", "arb", "eng"),
    ("bn_in", "en_us", "ben", "eng"),
    ("cmn_hans_cn", "en_us", "cmn", "eng"),
    ("hi_in", "en_us", "hin", "eng"),
    ("en_us", "ar_eg", "eng", "arb"),
    ("en_us", "bn_in", "eng", "ben"),
    ("en_us", "cmn_hans_cn", "eng", "cmn"),
    ("en_us", "hi_in", "eng", "hin"),
]
```

Use 25 test samples per pair for full benchmarking. For pruning candidate probes, use a smaller subset such as 5 samples per pair to control runtime.

For each paired sample, store:

- `id`
- source waveform as float32 16 kHz mono
- source language code
- target language code
- target transcription/reference text
- optionally source transcription text

Cache the processed data as Parquet or `.pt` files. Do not repeatedly download or decode audio inside every pruning loop.

## Baseline Model

Load:

```python
from transformers import SeamlessM4TProcessor, SeamlessM4Tv2ForSpeechToSpeech

processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
    "facebook/seamless-m4t-v2-large",
    torch_dtype=torch.float16,
).cuda().eval()
```

Important: use `SeamlessM4Tv2ForSpeechToSpeech`. Do not count or prune the text encoder, because this class does not use it in the S2ST path.

Implement:

- `count_params(model)`
- `print_component_breakdown(model)`
- `sync_model_config(model)` to update layer counts after pruning
- `reindex_layer_idx(module)` for decoder/T2U attention layers after removing layers

Run a baseline benchmark over all 8 pairs:

- generate speech
- decode intermediate text if available
- transcribe generated speech with ASR when measuring ASR metrics
- compute ChrF/BLEU against reference text
- measure RTF

Save the baseline model summary.

## Metrics

Use SacreBLEU:

```python
from sacrebleu.metrics import BLEU, CHRF
bleu = BLEU(effective_order=True)
chrf = CHRF()
```

Implement:

- `compute_bleu(pred, ref)`
- `compute_chrf(pred, ref)`
- `run_s2st(model, processor, wav, tgt_lang)`
- `run_s2tt_text(model, processor, wav, tgt_lang)` if available/needed
- `benchmark_s2st_asr(model, samples, label)`

For T2U/audio-path decisions, use ASR-ChrF:

1. Generate output audio.
2. Transcribe it with a multilingual ASR model.
3. Compare ASR transcript to target reference with ChrF.

Use MMS ASR for multilingual support. If implementation time is limited, benchmark T2U on non-English target directions first, especially Bengali.

## Phase 1: Five-Language Vocabulary Pruning

Implement vocabulary trimming:

1. Define target languages: `eng`, `ben`, `cmn`, `arb`, `hin`.
2. Scan FLEURS train text for all target languages.
3. Collect token IDs used by the tokenizer.
4. Always keep:
   - special tokens
   - language tokens
   - padding/eos/bos/unk tokens
   - any generation config token IDs
5. Build `keep_ids`.
6. Rebuild the shared embedding:
   - new shape `[len(keep_ids), hidden_size]`
   - copy old rows by `keep_ids`
7. Rebuild/tie:
   - `model.shared`
   - `model.text_decoder.embed_tokens`
   - `model.lm_head`
8. Update `model.config.vocab_size`.
9. Update language token mappings in generation config.
10. Save a tensor named `_vocab_remap_to_old = torch.tensor(keep_ids)`.

Important decoding requirement:

When decoding intermediate IDs from the pruned model, remap them to original tokenizer IDs:

```python
ids_old = model._vocab_remap_to_old[ids_new]
text = processor.batch_decode(ids_old, skip_special_tokens=True)
```

Benchmark and save Phase 1.

Expected ballpark:

- parameters around 1.56B
- noticeable but acceptable quality drop

## Phase 2: Speech Encoder Pruning, 24 to 16 Layers

Implement Block Influence:

```python
BI(layer) = 1 - cosine_similarity(layer_input, layer_output).mean()
```

Procedure:

1. Find the speech encoder layer `ModuleList`.
2. Register forward hooks on every layer.
3. Run 50 calibration samples through the speech encoder.
4. Compute BI score per layer.
5. Protect layers `{0, n//2, n-1}`.
6. Iteratively remove 8 layers:
   - candidate set: non-protected layers
   - optionally pre-filter to lowest 50% BI scores
   - for each candidate, temporarily remove it
   - sync config
   - run small multilingual ChrF probe
   - restore model
   - choose the candidate with the highest probe score
   - permanently remove it
   - save checkpoint

After pruning:

- speech encoder should have 16 layers
- sync all config layer counts
- save model and benchmark

Expected ballpark:

- parameters around 1.37B
- RTF around 0.16

## Phase 3: Conservative T2U Layer Merge

The T2U path affects output audio and is sensitive. Do not aggressively delete T2U layers unless ASR metrics confirm it.

Implement a conservative merge:

1. Find T2U encoder and decoder layer stacks.
2. For adjacent candidate layers, create a merged candidate by averaging/interpolating compatible parameters:

```python
merged = alpha * layer_i + (1 - alpha) * layer_j
```

3. Evaluate output similarity on calibration hidden states.
4. Merge only if similarity is high enough, e.g. cosine similarity above 0.85.
5. Limit removals per T2U stack.
6. Re-index T2U attention `layer_idx`.
7. Sync T2U config fields.

Benchmark after merge.

Expected ballpark:

- parameters around 1.33B
- small additional quality loss

## Phase 4: Additional Speech Encoder Pruning, 16 to 14 Layers

Repeat the Phase 2 BI-guided method, but remove only 2 additional speech encoder layers.

Benchmark and save.

Expected ballpark:

- parameters around 1.28B
- quality drops more noticeably than Phase 2

## Phase 5: Text Decoder Pruning, 24 to 14 Layers

The text decoder is critical. Do not prune below 14 layers for the final target.

Procedure:

1. Find `model.text_decoder.layers`.
2. Compute decoder BI using hooks during generation.
3. Protect `{0, n//2, n-1}`.
4. Iteratively remove 10 layers:
   - use BI pre-filtering
   - evaluate small multilingual ChrF probe
   - remove least damaging layer
   - re-index decoder attention layer indices
   - sync config
   - checkpoint after every removal
5. Save model as the pruned student.

Expected ballpark:

- parameters around 1.03B
- quality cliff is expected
- ChrF may fall into the mid-20s

Do not run FLAP/width pruning in the final path. Previous evidence showed catastrophic collapse after depth pruning.

## Phase 6: Teacher-Student KD Recovery

Load:

- teacher: full `facebook/seamless-m4t-v2-large`, frozen
- student: Phase 5 pruned model, trainable through adapters

Recommended GPU placement:

- student on `cuda:0`
- teacher on `cuda:1`

Attach LoRA/DoRA-style adapters to key student modules:

- text decoder self-attention q/v/out projections
- text decoder cross-attention q/v/out projections
- text decoder FFN layers
- selected T2U layers if memory permits

Use mixed precision carefully:

- base weights fp16
- trainable adapter params fp32 if using GradScaler
- freeze all non-adapter parameters

Loss:

```python
loss = (1 - alpha) * CE(student_logits, labels) + alpha * KL(student_logits, teacher_topk_logits)
```

Use `alpha` around 0.15 as a starting point.

Sparse teacher KD:

1. Run teacher forward without gradients.
2. Take top-k teacher logits, e.g. k=128 or 256.
3. Remap teacher token IDs to student token IDs using the vocabulary map.
4. Drop unmapped teacher tokens.
5. Renormalize teacher probabilities over mapped tokens.
6. Gather student logits at mapped token positions.
7. Compute KL at temperature around 3.0.

Checkpoint:

- adapter weights
- optimizer state
- best validation ChrF
- step count

Train until validation improves and stabilizes. Save merged model.

Expected ballpark after Phase 6:

- ChrF around 33
- BLEU around 8
- RTF around 0.15

## Phase 7: Final Adapter Recovery and Merge

Run a final recovery pass if Phase 6 quality is still low.

Recommended:

- load Phase 6 merged model
- attach broader hybrid adapters
- keep speech encoder frozen unless evidence suggests otherwise
- prioritize text decoder and T2U/text-unit path
- train for several epochs over FLEURS multilingual data
- evaluate every fixed number of steps on the 8-pair validation subset
- save best checkpoint
- merge adapters into base weights

Final save format must be directly loadable with:

```python
SeamlessM4TProcessor.from_pretrained(final_model_dir)
SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(final_model_dir)
```

Also save:

- `_custom_state.pt` containing `_vocab_remap_to_old`
- generation config
- tokenizer/processor files
- model config with correct layer counts

## Final Benchmark

Benchmark the final merged model on:

- 8 language pairs
- 25 samples per pair
- 200 total samples

Report:

| Field | Required |
| --- | --- |
| parameters | yes |
| ChrF | yes |
| BLEU | yes |
| RTF | yes |
| per-pair ChrF/BLEU/RTF | yes |
| sample input/output audio | optional but recommended |
| GPU model and VRAM | yes |
| failure cases | yes |

Expected strongest directions:

- `eng -> hin`
- `arb -> eng`
- `eng -> ben`

Expected weakest direction:

- `eng -> cmn`

## Local Inference Tool

Create a local inference script or notebook that:

1. Loads the final model from a local folder.
2. Repairs stale config layer counts from actual weight keys if needed.
3. Loads `_custom_state.pt` and attaches `_vocab_remap_to_old`.
4. Accepts a WAV file or FLEURS sample.
5. Runs S2ST.
6. Saves output WAV.
7. Prints intermediate decoded text if available.
8. Prints GPU memory usage.

Target local validation:

- RTX 3050 Laptop GPU or similar 4 GB-class GPU
- model memory around 2-3 GB fp16

## Duplex Web App

Build a browser app that uses the final model interactively.

Backend:

- FastAPI
- WebSocket endpoint `/ws`
- health endpoint `/health`
- static frontend serving
- load final SeamlessM4T model once at startup
- load a boundary/VAD adapter if available
- use an inference lock so boundary checks and translation do not overlap on GPU

Frontend:

- `index.html`
- `app.js`
- `worklet.js`
- capture mic at 16 kHz
- use AudioWorklet to emit int16 PCM frames
- stream binary frames over WebSocket
- support push-to-talk and always-listen
- schedule translated output audio
- flush scheduled playback on barge-in

Session behavior:

1. Browser sends PCM frames.
2. Backend keeps a rolling audio buffer.
3. Boundary detector checks encoder-frame speech probabilities.
4. When trailing silence exceeds patience threshold, dispatch utterance.
5. Backend runs translation in executor/thread so WebSocket stays responsive.
6. Backend sends:
   - `user_speech_start`
   - `user_speech_end`
   - `thinking`
   - `speaking_start`
   - binary PCM chunks
   - `speaking_text`
   - `speaking_end`
7. If user starts speaking while bot is speaking, cancel generation/playback and send `stop_playback`.

Boundary detector:

Use a 3-layer MLP over SeamlessM4T speech encoder hidden states:

```text
Linear(1024, 512)
LayerNorm
ReLU
Dropout
Linear(512, 256)
LayerNorm
ReLU
Dropout
Linear(256, 1)
Sigmoid
```

If you do not have a trained boundary adapter, implement a fallback energy-based VAD but document that it is not equivalent.

## Failure Boundaries to Respect

Do not pursue these as the final path:

1. Do not force a sub-500M model by pruning the text decoder to 6 layers. It is likely to collapse.
2. Do not remove the text decoder entirely or replace it with a simple CIF shortcut.
3. Do not use text-only metrics to decide T2U pruning.
4. Do not apply FLAP width pruning after heavy depth pruning unless it is clearly an ablation and you expect collapse.
5. Do not forget to sync config layer counts and attention `layer_idx` after structural edits.

## Final Report Required From You

At the end, produce `artifacts/results/final_results.md` with:

1. Phase-by-phase result table.
2. Final per-pair table.
3. Model folder paths.
4. Exact hardware used.
5. Reproduction commands.
6. Deviations from expected values.
7. Local inference status.
8. Web app status.
9. Known failure cases.
10. Recommendations for future work.

