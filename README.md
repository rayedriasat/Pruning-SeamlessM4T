# Pruning SeamlessM4T

Structured compression, recovery, and deployment experiments for Meta SeamlessM4T v2 speech-to-speech translation.

This repository contains a semester-long CSE465 project that compresses SeamlessM4T v2 large for multilingual speech translation across English, Bengali, Mandarin, Arabic, and Hindi. The final selected model is a recovered compressed model of about 1.09B parameters, down from a 1.805B-parameter baseline, with a working local inference notebook and duplex browser demo.

## Headline Result

| Model | Params | ChrF | BLEU | RTF |
| --- | ---: | ---: | ---: | ---: |
| SeamlessM4T v2 baseline | 1805.5M | 46.49 | 15.88 | 0.2455 |
| Final compressed model | 1087.9M | 33.73 | 8.16 | 0.1336 |

The final model reduces parameters by about 39.7% and runs about 1.84x faster by RTF on the saved multilingual benchmark.

## What Is in This Repo

| Path | Description |
| --- | --- |
| `Final Notebooks/` | The important notebooks. Start here. |
| `Final Notebooks/v2 Multilingual full-final-notebook.ipynb` | Main final notebook and the source of the final model pipeline. |
| `Final Notebooks/v2_2 Multilingual Finetuning Phase_6.ipynb` | Recovery fine-tuning, KD, LoRA/DoRA, and final training refinements. |
| `seamless_local/` | Local inference notebook tested on an RTX 3050 Laptop GPU with 4.3 GB VRAM. |
| `duplexWebApp/` | FastAPI/WebSocket/browser app for duplex speech translation with the final model. |
| `mission500m/` | Failed sub-500M compression attempt. Useful as research evidence, not the final path. |
| `papers/` | Papers used to design pruning and recovery strategies. |
| `cluttered Experiments (includes every file we produced)/` | Full experiment history, notes, fixes, and abandoned branches. |
| `FINAL_REPORT.md` | Submission-ready final report draft. |
| `APPENDIX_1.md` | Mandatory appendix with code link, results, and notebook map. |
| `REPRODUCIBLE_AGENT_PROMPT.md` | Prompt for an agent like OpenCode to reproduce the project. |

## Final Pipeline

| Phase | Method | Params | ChrF | BLEU | RTF |
| --- | --- | ---: | ---: | ---: | ---: |
| P0 | Baseline SeamlessM4T v2 S2ST | 1805.5M | 46.49 | 15.88 | 0.2455 |
| P1 | Five-language vocabulary pruning | 1566.6M | 41.74 | 13.65 | 0.2435 |
| P2 | Speech encoder pruning, 24 -> 16 layers | 1373.1M | 38.97 | 11.13 | 0.1617 |
| P3 | T2U LaCo/RDSC merge | 1331.2M | 38.47 | 11.21 | 0.1646 |
| P4 | Speech encoder pruning, 16 -> 14 layers | 1282.8M | 35.74 | 9.67 | 0.1635 |
| P5 | Text decoder pruning, 24 -> 14 layers | 1030.9M | 25.32 | 5.83 | 0.1881 |
| P6 | KD/LoRA recovery | about 1.03B | 33.07 | 7.95 | 0.1484 |
| P7 | Final hybrid LoRA/DoRA recovery | 1087.9M | 33.73 | 8.16 | 0.1336 |

## Main Ideas

- Vocabulary pruning removes unused tokens for the five-language target set.
- Block Influence identifies lower-impact layers for structured pruning.
- Speech encoder and text decoder pruning use text/ASR translation quality checks.
- T2U pruning must use ASR-ChrF because text-only metrics cannot see audio-path damage.
- KD and LoRA/DoRA recovery are needed after aggressive structural pruning.
- The sub-500M branch showed a practical limit: too much text-decoder removal destroys the model.

## Local Inference

Use the notebook in `seamless_local/` when you already have the final model artifacts.

```powershell
cd seamless_local
uv venv --python 3.12
.venv\Scripts\activate
uv pip install torch --torch-backend=auto
uv pip install transformers datasets accelerate peft librosa soundfile sounddevice requests pandas sacrebleu evaluate sentencepiece safetensors matplotlib seaborn notebook huggingface_hub
jupyter notebook seamless_local_inference.ipynb
```

Expected local model folder:

```text
seamless_local/local_working/models/phase7_dora_merged_v1/
```

The saved notebook output verifies loading on an NVIDIA GeForce RTX 3050 Laptop GPU with 4.3 GB VRAM, using about 2.10 GB allocated VRAM after model load.

## Duplex Web App

The app in `duplexWebApp/` provides an interactive browser voice translation interface.

```bash
cd duplexWebApp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

Then open:

```text
http://localhost:8000
```

Expected model/artifact layout:

```text
duplexWebApp/models2/phase7_final_merged/
duplexWebApp/models2/boundary_adapter.pt
```

The browser records 16 kHz microphone audio, streams it over WebSocket, runs boundary detection, translates with the compressed model, and streams translated speech back to the browser.

## Report Files

The submission materials are:

- `FINAL_REPORT.md`
- `APPENDIX_1.md`
- `REPRODUCIBLE_AGENT_PROMPT.md`

Before final submission, fill in the group member names, IDs, and email addresses in `FINAL_REPORT.md`.

## Known Limitations

- Final model is about 1.09B parameters, not under 500M.
- English-to-Mandarin quality remains weak after compression.
- Large model artifacts are not meant to be committed directly.
- ONNX conversion is still work-in-progress.

## Citation Notes

The methods were influenced by the papers in `papers/`, especially SeamlessM4T, vocabulary pruning, Block Influence/layer redundancy, FLAP, Wanda, CULL-MT, DPHuBERT, and recent speech/translation distillation work.

