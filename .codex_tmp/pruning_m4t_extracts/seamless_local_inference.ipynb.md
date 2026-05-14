# seamless_local\seamless_local_inference.ipynb

Extracted notebook map containing markdown headings plus code/output cells likely to matter for reports, reproduction, or agent steering.

## Markdown headings
cell 1: # SeamlessM4T v2 Compressed — Local Inference & Benchmark ## 🛠️ One-time Setup (run in terminal before opening this notebook) ### 1. Install uv # Windows PowerShell # Linux / macOS ### 2. Create virtual environment # Windows: .venv\Scripts\activate # Linux/Mac: source .venv/bin/activate ### 3. Install PyTorch CUDA if available (for RTX 3050) ### 4. Install all other packages ### 5. Download your model from Google Drive # Place your rclone.conf at ~/.config/rclone/rclone.conf ### 6. FLEURS parquet files (optional — Cell 7 auto-downloads if missing) ### 7. Launch Jupyter

## Key cells

### Cell 1 (markdown, score=68)
```markdown
# SeamlessM4T v2 Compressed — Local Inference & Benchmark
Adapted from the Kaggle Phase 7/8 notebook for **local RTX 3050 4 GB** use.

**Same folder structure as Kaggle** — only `ON_KAGGLE=False` and the root path is `./local_working` instead of `/kaggle/working`.

---
## 🛠️ One-time Setup (run in terminal before opening this notebook)

### 1. Install uv
```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual environment
```bash
uv venv --python 3.12
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
```

### 3. Install PyTorch CUDA if available (for RTX 3050)
```bash
uv pip install torch --torch-backend=auto
```

### 4. Install all other packages
```bash
uv pip install transformers datasets accelerate peft librosa soundfile sounddevice requests pandas sacrebleu evaluate sentencepiece safetensors matplotlib seaborn notebook huggingface_hub
```

### 5. Download your model from Google Drive
Your model is at `cse465v5/models/phase7_dora_merged_v1` on Drive.

**Option A — rclone (same rclone.conf from Kaggle secret):**
```bash
# Place your rclone.conf at ~/.config/rclone/rclone.conf
mkdir -p local_working/models/phase7_dora_merged_v1
rclone copy gdrive:cse465v5/models/phase7_dora_merged_v1 ./local_working/models/phase7_dora_merged_v1 --progress
```

**Option B — Google Drive browser download:**
- Open drive.google.com → `cse465v5/models/phase7_dora_merged_v1/`
- Download all files and place them in `./local_working/models/phase7_dora_merged_v1/`

### 6. FLEURS parquet files (optional — Cell 7 auto-downloads if missing)
If you already synced them on Kaggle:
```bash
mkdir -p local_working/fleurs_parquet
rclone copy gdrive:cse465v5/fleurs_parquet ./local_working/fleurs_parquet --progress
```
Otherwise Cell 7 will download them fresh from HuggingFace (no `trust_remote_code` needed).

### 7. Launch Jupyter
```bash
jupyter notebook seamless_local_inference.ipynb
```

> **VRAM note:** 1095.9M params × float16 ≈ 2.2 GB VRAM. Your RTX 3050 4 GB handles it with ~1.8 GB to spare.
```

### Cell 2 (code, score=46)
```python
# ── Cell 0: Platform & paths  (mirrors original Cell 1) ─────────────────────
import os, sys, subprocess, pathlib, re, glob, json, gc, copy, time, math, shutil
import warnings; warnings.filterwarnings('ignore')

# Always local — no Kaggle / Colab
ON_KAGGLE = False
ON_COLAB  = False
PLATFORM  = 'local'

# Mirror the exact sub-folder names used on Kaggle (/kaggle/working → ./local_working)
KAGGLE_WORK  = os.path.join(os.getcwd(), 'local_working')
WORK_DIR  = KAGGLE_WORK
CKPT_DIR  = f'{WORK_DIR}/checkpoints'
AUDIO_DIR = f'{WORK_DIR}/audio'
FIG_DIR   = f'{WORK_DIR}/figures'
MODEL_DIR = f'{WORK_DIR}/models'

# Parquet cache — same sub-path the Kaggle notebook wrote
LOCAL_PARQUET_CACHE = f'{WORK_DIR}/fleurs_parquet'

for d in [WORK_DIR, CKPT_DIR, AUDIO_DIR, FIG_DIR, MODEL_DIR, LOCAL_PARQUET_CACHE]:
    os.makedirs(d, exist_ok=True)

print(f'Platform : {PLATFORM}')
print(f'Work dir : {WORK_DIR}')
```
OUTPUT:
```text
Platform : local
Work dir : e:\NSU\Semester 261\CSE465\CSE465_Project\Rayed\Pruning SeamlessM4T\seamless_local\local_working
```

### Cell 3 (code, score=6)
```python
# ── Cell 1: Quick package sanity-check (no pip here — done via uv) ───────────
import importlib
required = [
    'torch', 'torchaudio', 'transformers', 'datasets', 'peft',
    'librosa', 'soundfile', 'sacrebleu', 'evaluate', 'sentencepiece',
    'accelerate', 'matplotlib', 'seaborn', 'pandas', 'requests', 'sounddevice',
]
missing = [p for p in required if importlib.util.find_spec(p) is None]
if missing:
    print(f'[WARN] Missing: {missing}')
    print('Run the uv install commands in the Setup cell, then restart the kernel.')
else:
    print('All required packages present.')
```
OUTPUT:
```text
All required packages present.
```

### Cell 4 (code, score=66)
```python
# ── Cell 2: Core utilities  (mirrors original Cell 6) ────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt, matplotlib, seaborn as sns
matplotlib.rcParams.update({'font.size': 11, 'figure.dpi': 120, 'savefig.bbox': 'tight'})
sns.set_style('whitegrid')

def count_params(module):
    return sum(p.numel() for p in module.parameters()) / 1e6

def count_params_detailed(model):
    bd = {}
    for name, child in model.named_children():
        bd[name] = count_params(child)
    bd['TOTAL'] = count_params(model)
    return bd

def print_model_breakdown(model, title='Model Breakdown'):
    bd = count_params_detailed(model)
    print(f'\n--- {title} ---')
    total = bd.pop('TOTAL')
    for name, p in sorted(bd.items(), key=lambda x: -x[1]):
        pct = p / total * 100 if total > 0 else 0
        print(f'  {name:<35} {p:>8.1f}M  ({pct:>5.1f}%)')
    print(f'  {"TOTAL":<35} {total:>8.1f}M')
    print('---')
    return {**bd, 'TOTAL': total}

def gpu_mem():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        print(f'  GPU mem: {a:.2f} GB alloc / {r:.2f} GB reserved')

def save_figure(fig, name):
    fig.savefig(f'{FIG_DIR}/{name}', dpi=150, bbox_inches='tight')
    print(f'[fig] Saved {FIG_DIR}/{name}')

import torchaudio
from IPython.display import Audio as IPAudio, display

def play(audio, sr, label=''):
    if hasattr(audio, 'numpy'): audio = audio.squeeze().numpy()
    print(f'  {label}  ({len(audio)/sr:.1f}s | sr={sr})')
    display(IPAudio(audio, rate=int(sr)))

def save_audio(audio, sr, filename, label=''):
    import soundfile as _sf
    path = f'{AUDIO_DIR}/{filename}'
    # Flatten to 1-D numpy float32 — soundfile needs no torchcodec
    if hasattr(audio, 'numpy'):
        arr = audio.squeeze().float().numpy()
    elif isinstance(audio, np.ndarray):
        arr = audio.squeeze().astype(np.float32)
    else:
        arr = np.array(audio, dtype=np.float32).squeeze()
    _sf.write(path, arr, int(sr))
    mb = os.path.getsize(path) / 1e6
    print(f'[audio] Saved {filename} ({mb:.1f} MB)')

print('Core utilities ready.')
print(f'CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU  : {torch.cuda.get_device_name(0)}')
    print(f'VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
```
OUTPUT:
```text
Core utilities ready.
CUDA available : True
GPU  : NVIDIA GeForce RTX 3050 Laptop GPU
VRAM : 4.3 GB
```

### Cell 5 (code, score=70)
```python
# ── Cell 3: Benchmark functions  (mirrors original Cell 7) ───────────────────
from sacrebleu.metrics import BLEU, CHRF

_bleu = BLEU(effective_order=True)
_chrf = CHRF()

def find_layers_attr(component):
    for attr in ['layers', 'layer', 'inner_layers', 'encoder_layers', 'decoder_layers']:
        if hasattr(component, attr): return attr
    return None

def compute_bleu(hyp, ref):
    if not hyp.strip() or not ref.strip(): return 0.0
    return _bleu.sentence_score(hyp.strip(), [ref.strip()]).score

def compute_chrf(hyp, ref):
    if not hyp.strip() or not ref.strip(): return 0.0
    return _chrf.sentence_score(hyp.strip(), [ref.strip()]).score

def _remap_ids_for_decode(mdl, ids):
    if hasattr(mdl, '_vocab_remap_to_old'):
        remap = mdl._vocab_remap_to_old
        ids = ids.clone()
        mask = (ids >= 0) & (ids < len(remap))
        ids[mask] = remap[ids[mask]]
    return ids

def _model_input_device(mdl):
    if hasattr(mdl, 'speech_encoder'):
        return next(mdl.speech_encoder.parameters()).device
    return next(mdl.parameters()).device

def run_s2st(mdl, wav, tgt_lang='ben'):
    inputs = processor(audio=wav, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}
    with torch.no_grad():
        try:
            out = mdl.generate(**inputs, tgt_lang=tgt_lang,
                               return_intermediate_token_ids=True)
            text_ids = _remap_ids_for_decode(mdl, out.sequences.cpu())
            text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
            wav_out = out.waveform.cpu().float().numpy().squeeze() if out.waveform is not None else np.zeros(16000, dtype=np.float32)
            return text, wav_out
        except RuntimeError:
            text = run_s2t_only(mdl, wav, tgt_lang)
            return text, np.zeros(16000)

def run_s2t_only(mdl, wav, tgt_lang='ben'):
    inputs = processor(audio=wav, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}
    orig_voc = mdl.vocoder
    inp_device = next(iter(inputs.values())).device
    class _NoOpVocoder(nn.Module):
        def forward(self, *args, **kwargs):
            return torch.zeros(1, 1, device=inp_device), [1]
    mdl.vocoder = _NoOpVocoder()
    try:
        with torch.no_grad():
            out = mdl.generate(**inputs, tgt_lang=tgt_lang,
                               return_intermediate_token_ids=True)
    finally:
        mdl.vocoder = orig_voc
    text_ids = _remap_ids_for_decode(mdl, out.sequences.cpu())
    return processor.batch_decode(text_ids, skip_special_tokens=True)[0]

def run_benchmark(mdl, samples, label='model', tgt_lang='ben', save_n=4):
    print(f'\n{"="*60}\n  BENCHMARK: {label}\n  Samples: {len(samples)}  Target: {tgt_lang}\n{"="*60}\n')
    gpu_mem()
    results = []
    for i, s in enumerate(samples):
        try:
            dur = len(s['wav']) / 16000
            t0 = time.time()
            pred_text = run_s2t_only(mdl, s['wav'], tgt_lang=tgt_lang)
            elapsed = time.time() - t0
            rtf  = elapsed / dur
            bleu = compute_bleu(pred_text, s['ref'])
            chrf = compute_chrf(pred_text, s['ref'])
            print(f'  [{i+1:>2}/{len(samples)}] BLEU={bleu:5.1f} ChrF={chrf:5.1f} RTF={rtf:.3f}  id={s["id"]}')
            print(f'              pred: {pred_text[:80]}')
            if save_n > 0 and i < save_n:
                _, out_wav = run_s2st(mdl, s['wav'], tgt_lang=tgt_lang)
                save_audio(s['wav'], mdl.config.sampling_rate, f'{label}_s{i+1}in.wav')
                play(s['wav'], mdl.config.sampling_rate, f'{label}_s{i+1}in.wav')
                save_audio(out_wav, mdl.config.sampling_rate, f'{label}_s{i+1}out.wav')
                play(out_wav, mdl.config.sampling_rate, f'{label}_s{i+1}out.wav')
            results.append(dict(id=s['id'], bleu=bleu, chrf=chrf, rtf=rtf, pred=pred_text, ref=s['ref']))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [{i+1:>2}/{len(samples)}] ERROR: {e}')
            results.append(dict(id=s['id'], bleu=0, chrf=0, rtf=float('nan'), pred='', ref=s.get('ref','')))
    valid = [r for r in results if not math.isnan(r['rtf'])]
    summary = dict(label=label, n=len(valid),
        avg_bleu=float(np.mean([r['bleu'] for r in valid])) if valid else 0,
        avg_chrf=float(np.mean([r['chrf'] for r in valid])) if valid else 0,
        avg_rtf=float(np.mean([r['rtf'] for r in valid])) if valid else 0,
        params_M=count_params(mdl))
    print(f'\n  Summary: BLEU={summary["avg_bleu"]:.2f}  ChrF={summary["avg_chrf"]:.2f}'
          f'  RTF={summary["avg_rtf"]:.4f}  Params={summary["params_M"]:.1f}M\n')
    return results, summary

print('Benchmark functions ready.')
```
OUTPUT:
```text
Benchmark functions ready.
```

### Cell 6 (code, score=131)
```python
# ── Cell 4: Model I/O helpers  (mirrors original Cell 8) ─────────────────────
_CUSTOM_STATE_FILE = '_custom_state.pt'
_PRUNING_MANIFEST  = 'pruning_manifest.pt'

def _flat_sd_from_model_dir(model_dir):
    """Load checkpoint as a flat str->tensor dict (safetensors or pytorch .bin)."""
    safe = os.path.join(model_dir, 'model.safetensors')
    if os.path.isfile(safe):
        try:
            from safetensors.torch import load_file
            return load_file(safe)
        except ImportError:
            pass
    pt = os.path.join(model_dir, 'pytorch_model.bin')
    if os.path.isfile(pt):
        blob = torch.load(pt, map_location='cpu', weights_only=False)
        if isinstance(blob, dict) and 'model' in blob:
            inner = blob['model']
            return inner if isinstance(inner, dict) else blob
        return blob
    return None


def _infer_stack_depth_from_sd(sd, prefix):
    """Largest N such that some key starts with ``prefix`` + ``N`` + '.'; depth = N+1."""
    if not sd:
        return None
    idx = set()
    for k in sd:
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):].split('.', 1)[0]
        if rest.isdigit():
            idx.add(int(rest))
    return (max(idx) + 1) if idx else None


def _load_custom_state(mdl, path):
    fpath = os.path.join(path, _CUSTOM_STATE_FILE)
    if not os.path.exists(fpath): return
    state = torch.load(fpath, map_location='cpu', weights_only=False)
    for k, v in state.items():
        setattr(mdl, k, v)
    print(f'  Restored custom state: {list(state.keys())}')

def sync_model_config(mdl):
    if hasattr(mdl, 'speech_encoder'):
        enc = mdl.speech_encoder
        parent = enc.encoder if hasattr(enc, 'encoder') else enc
        if hasattr(parent, 'layers'):
            actual = len(parent.layers)
            if hasattr(mdl.config, 'speech_encoder_layers'):
                old = mdl.config.speech_encoder_layers
                if old != actual:
                    mdl.config.speech_encoder_layers = actual
                    print(f'  [config] speech_encoder_layers: {old} -> {actual}')
    if hasattr(mdl, 'text_decoder'):
        dec = mdl.text_decoder
        la = find_layers_attr(dec)
        if la:
            actual = len(getattr(dec, la))
            if hasattr(mdl.config, 'decoder_layers'):
                old = mdl.config.decoder_layers
                if old != actual:
                    mdl.config.decoder_layers = actual
    print('  [config] sync done.')

def load_model_from_drive(stage_name):
    """On local, reads directly from MODEL_DIR (no rclone pull needed)."""
    from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor, AutoConfig
    local = f'{MODEL_DIR}/{stage_name}'
    if not os.path.exists(local) or not os.listdir(local):
        raise RuntimeError(
            f'[model] Path not found or empty: {local}\n'
            'Please download the model from Drive first (see Setup instructions).'
        )
    weight_files = [f for f in os.listdir(local)
                    if f.endswith('.safetensors') or f.endswith('.bin')]
    if not weight_files:
        raise RuntimeError(f'[model] No weight files in {local}')
    print(f'[model] Loading {stage_name} from {local} ...')
    cfg = AutoConfig.from_pretrained(local)
    sd = _flat_sd_from_model_dir(local)

    # Text decoder depth (Phase 3 prune / stale config after DoRA merge — same class of bug as T2U)
    td_n = _infer_stack_depth_from_sd(sd, 'text_decoder.layers.') if sd else None
    if td_n is not None and getattr(cfg, 'decoder_layers', None) != td_n:
        print(f'  [model] Repair decoder_layers from weights: {cfg.decoder_layers} -> {td_n}')
        cfg.decoder_layers = td_n

    # Speech encoder Conformer stack: speech_encoder.encoder.layers.N.*
    se_n = _infer_stack_depth_from_sd(sd, 'speech_encoder.encoder.layers.') if sd else None
    if se_n is not None and getattr(cfg, 'speech_encoder_layers', None) != se_n:
        print(f'  [model] Repair speech_encoder_layers from weights: {cfg.speech_encoder_layers} -> {se_n}')
        cfg.speech_encoder_layers = se_n
        sec = getattr(cfg, 'speech_encoder_config', None)
        if sec is not None and hasattr(sec, 'num_hidden_layers'):
            sec.num_hidden_layers = se_n

    enc_n = _infer_stack_depth_from_sd(sd, 't2u_model.model.encoder.layers.') if sd else None
    dec_n = _infer_stack_depth_from_sd(sd, 't2u_model.model.decoder.layers.') if sd else None
    if enc_n is not None and getattr(cfg, 't2u_encoder_layers', None) != enc_n:
        print(f'  [model] Repair T2U encoder depth from weights: {cfg.t2u_encoder_layers} -> {enc_n}')
        cfg.t2u_encoder_layers = enc_n
    if dec_n is not None and getattr(cfg, 't2u_decoder_layers', None) != dec_n:
        print(f'  [model] Repair T2U decoder depth from weights: {cfg.t2u_decoder_layers} -> {dec_n}')
        cfg.t2u_decoder_layers = dec_n
    mdl = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
        local, config=cfg, torch_dtype=torch.float16, device_map='auto')
    _load_custom_state(mdl, local)
    proc = SeamlessM4TProcessor.from_pretrained(local)
    mdl.eval()
    return mdl, proc

print('Model I/O helpers ready.')
```
OUTPUT:
```text
Model I/O helpers ready.
```

### Cell 7 (code, score=76)
```python
# ── Cell 5: FLEURS data loaders  (mirrors original Cell 11) ──────────────────
# Downloads raw parquet shards directly from HuggingFace CDN.
# No load_dataset(), no trust_remote_code — same approach as the original notebook.

import concurrent.futures

BASE_PARQUET_URL = (
    "https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet"
)

def _list_parquet_urls(lang, split):
    """
    Discover all parquet shards for a given lang/split by probing sequentially
    until a 404 is hit. Falls back to at least returning shard 0000.
    """
    import requests
    urls = []
    i = 0
    while True:
        url = f"{BASE_PARQUET_URL}/{lang}/{split}/{i:04d}.parquet?download=true"
        try:
            r = requests.head(url, timeout=15, allow_redirects=True)
            if r.status_code == 200:
                urls.append(url)
                i += 1
            else:
                break  # 404 or anything else → no more shards
        except requests.RequestException:
            break
    if not urls:
        # fallback: blindly return shard 0 so downstream raises a clear error
        urls = [f"{BASE_PARQUET_URL}/{lang}/{split}/0000.parquet?download=true"]
        print(f"  [WARN] Could not probe shards for {lang}/{split}, falling back to 0000 only")
    print(f"  [shards] {lang}/{split}: {len(urls)} shard(s) found")
    return urls

def _download_shard(args):
    import requests
    url, dest = args
    dest = pathlib.Path(dest)
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        return url, True, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk: f.write(chunk)
            if dest.stat().st_size < 1024 * 1024:
                raise RuntimeError("Downloaded file too small")
            return url, True, "downloaded"
        except Exception as e:
            if dest.exists(): dest.unlink()
            if attempt == 2: return url, False, str(e)
    return url, False, "unknown error"

def load_fleurs_parallel(src_lang, tgt_lang, split="train", n_workers=4):
    import pandas as pd

    tasks = []
    shard_index = {}  # lang -> list of local dest paths

    for lang in [src_lang, tgt_lang]:
        urls = _list_parquet_urls(lang, split)
        shard_index[lang] = []
        for i, url in enumerate(urls):
            dest = f"{LOCAL_PARQUET_CACHE}/{lang}/{split}_{i:04d}.parquet"
            shard_index[lang].append(dest)
            tasks.append((url, dest))

    print(f"[Parallel] Downloading {len(tasks)} shards...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for url, ok, msg in pool.map(_download_shard, tasks):
            print(f"  {'OK' if ok else 'FAIL'}: {msg}")

    def _load_lang(lang):
        files = sorted(f for f in shard_index[lang] if os.path.exists(f))
        if not files:
            raise FileNotFoundError(f"No cached shards for {lang}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        print(f"  Loaded {lang}: {len(df)} rows from {len(files)} shard(s)")
        return df   # raw DataFrame — caller does the merge

    return _load_lang(src_lang), _load_lang(tgt_lang)

def load_fleurs_from_drive(src_lang, tgt_lang, split='train'):
    """
    Local equivalent of load_fleurs_from_drive().
    Returns raw pandas DataFrames (not HF Datasets) so the caller
    can do a proper ID-based merge without iterating every row.
    Returns (src_df, tgt_df) or (None, None) if cache is missing.
    """
    import pandas as pd

    def _load_lang(lang):
        files = sorted(glob.glob(f'{LOCAL_PARQUET_CACHE}/{lang}/{split}_*.parquet'))
        if not files:
            return None
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        print(f'  [local cache] {lang}/{split}: {len(df)} rows from {len(files)} shard(s)')
        return df

    src_df = _load_lang(src_lang)
    tgt_df = _load_lang(tgt_lang)
    return src_df, tgt_df

print('FLEURS data loaders ready.')
```
OUTPUT:
```text
FLEURS data loaders ready.
```

### Cell 8 (code, score=70)
```python
# ── Cell 6: Load the final merged model  (mirrors original Cell 15 load step) ─
from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor

model, processor = load_model_from_drive('phase7_dora_merged_v1')
print_model_breakdown(model, 'Phase 7 DoRA Merged (local)')
gpu_mem()
```
OUTPUT:
```text
[model] Loading phase7_dora_merged_v1 from e:\NSU\Semester 261\CSE465\CSE465_Project\Rayed\Pruning SeamlessM4T\seamless_local\local_working/models/phase7_dora_merged_v1 ...
  [model] Repair decoder_layers from weights: 16 -> 14

Instantiating a decoder SeamlessM4Tv2Attention without passing `layer_idx` is not recommended and will lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` when creating this class.
Loading weights: 100%|██████████| 1266/1266 [00:04<00:00, 289.28it/s]

  Restored custom state: ['_vocab_remap_to_old']

--- Phase 7 DoRA Merged (local) ---
  speech_encoder                         441.6M  ( 42.5%)
  text_decoder                           373.6M  ( 36.0%)
  t2u_model                              182.0M  ( 17.5%)
  vocoder                                 41.9M  (  4.0%)
  shared                                  20.9M  (  2.0%)
  lm_head                                 20.9M  (  2.0%)
  TOTAL                                 1039.1M
---
  GPU mem: 2.10 GB alloc / 2.12 GB reserved
```

### Cell 9 (code, score=69)
```python
# ── Cell 7: Load FLEURS test data  (fixed: pandas merge, no full iteration) ──
import numpy as np
import torch
import torchaudio
import io
import soundfile as sf
import pandas as pd

# ── GLOBAL: change this to run more/fewer benchmark samples ──────────────────
N_EVAL = 5
# ─────────────────────────────────────────────────────────────────────────────

TARGET_LANG = "ben"
FLEURS_SRC, FLEURS_TGT = "en_us", "bn_in"

print(f"Loading FLEURS {FLEURS_SRC}->{FLEURS_TGT} for benchmarking [test]")

df_src, df_tgt = load_fleurs_from_drive(FLEURS_SRC, FLEURS_TGT, split="test")

if df_src is None or df_tgt is None:
    print("\n[Cache miss] Downloading...")
    df_src, df_tgt = load_fleurs_parallel(FLEURS_SRC, FLEURS_TGT, split="test", n_workers=8)

# ── Robust audio loader — parquet stores bytes, not array dicts ───────────────
def _load_wav(audio_cell):
    """
    audio_cell: the value of row['audio'] from a pandas DataFrame.
    Parquet stores audio as a dict with 'bytes' key (raw WAV bytes).
    HF Dataset rows may have 'array'+'sampling_rate' instead.
    """
    audio = audio_cell
    if isinstance(audio, dict) and "array" in audio:
        arr, sr = audio["array"], audio["sampling_rate"]
    elif isinstance(audio, dict) and "bytes" in audio:
        wav, sr = sf.read(io.BytesIO(audio["bytes"]))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        arr = wav
    else:
        raise RuntimeError(f"Unsupported audio format: {list(audio.keys()) if isinstance(audio, dict) else type(audio)}")
    arr = np.array(arr, dtype=np.float32)
    if sr != 16000:
        arr = torchaudio.functional.resample(
            torch.tensor(arr), sr, 16000
        ).numpy()
    return arr

# ── ID-based inner merge — deduplicated, guaranteed unique pairs ─────────────
# The parquet shards can contain duplicate id rows (same sentence split across
# multiple shards). Drop duplicates on 'id' first — keep the first occurrence —
# so the merge produces exactly one row per unique sentence ID.
src_dedup = (
    df_src[['id', 'transcription', 'audio']]
    .drop_duplicates(subset='id', keep='first')
    .rename(columns={'transcription': 'en_text', 'audio': 'en_audio'})
)
tgt_dedup = (
    df_tgt[['id', 'transcription', 'audio']]
    .drop_duplicates(subset='id', keep='first')
    .rename(columns={'transcription': 'bn_text', 'audio': 'bn_audio'})
)

print(f"  Unique IDs — EN: {len(src_dedup)}, BN: {len(tgt_dedup)}")

merged = (
    pd.merge(src_dedup, tgt_dedup, on='id', how='inner')
    .sort_values('id')
    .reset_index(drop=True)
)

print(f"  Matched unique pairs available: {len(merged)}")
assert len(merged) >= N_EVAL, (
    f"Only {len(merged)} matched pairs found — lower N_EVAL or check parquet files."
)

# Take exactly N_EVAL rows
merged = merged.head(N_EVAL)

# Build eval_samples in the same structure as the original notebook
# (wav = English audio, ref = Bengali transcription, en_text = English text)
eval_samples = []
for _, row in merged.iterrows():
    eval_samples.append(dict(
        id=row['id'],
        wav=_load_wav(row['en_audio']),
        ref=row['bn_text'],
        en_text=row['en_text'],
    ))

# Keep src_by_id / tgt_by_id for Cell 9 (BN→EN samples) — keyed by id
common_ids = list(merged['id'])
src_by_id  = {row['id']: row for _, row in merged.iterrows()}
tgt_by_id  = {row['id']: row for _, row in merged.iterrows()}

print(f"Loaded {len(eval_samples)} eval samples.")
for s in eval_samples:
    dur = len(s['wav']) / 16000
    print(f"  id={s['id']}  {dur:.1f}s  EN: {s['en_text'][:60]}")
```
OUTPUT:
```text
Loading FLEURS en_us->bn_in for benchmarking [test]
  [local cache] en_us/test: 647 rows from 1 shard(s)
  [local cache] bn_in/test: 920 rows from 2 shard(s)
  Unique IDs — EN: 350, BN: 349
  Matched unique pairs available: 349
Loaded 5 eval samples.
  id=1660  10.7s  EN: romanticism had a large element of cultural determinism draw
  id=1661  6.4s  EN: he did not set a figure for the cuts saying they will be mad
  id=1662  8.4s  EN: alloys are basically a mixture of two or more metals don't f
  id=1663  12.1s  EN: cochamó valley - chile's premier climbing destination known 
  id=1664  8.4s  EN: swirl the two dry powders together and then with clean wet h
```

### Cell 10 (code, score=75)
```python
# ── Cell 8: Benchmark — English → Bengali ─────────────────────────────────────
# save_n=N_EVAL → input + output audio shown for every sample back-to-back.

en2bn_results, en2bn_summary = run_benchmark(
    model, eval_samples,
    label='EN2BN',
    tgt_lang='ben',
    save_n=N_EVAL
)
```
OUTPUT:
```text
============================================================
  BENCHMARK: EN2BN
  Samples: 5  Target: ben
============================================================

  GPU mem: 2.10 GB alloc / 2.12 GB reserved
  [ 1/5] BLEU= 14.8 ChrF= 49.4 RTF=0.322  id=1660
              pred: রোমান্টিকতার মধ্যে সংস্কৃতির নির্ধারকতা এর একটি বড় উপাদান ছিল যা গথ ফিচ এবং শ্ল
[audio] Saved EN2BN_s1in.wav (0.3 MB)
  EN2BN_s1in.wav  (10.7s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved EN2BN_s1out.wav (0.2 MB)
  EN2BN_s1out.wav  (6.4s | sr=16000)

<IPython.lib.display.Audio object>
  [ 2/5] BLEU=  6.2 ChrF= 39.5 RTF=0.120  id=1661
              pred: তিনি চীনের অর্থনৈতিক উৎপাদনের উপর ভিত্তি করে কাট করার জন্য কোনও সংখ্যা নির্ধারণ 
[audio] Saved EN2BN_s2in.wav (0.2 MB)
  EN2BN_s2in.wav  (6.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved EN2BN_s2out.wav (0.2 MB)
  EN2BN_s2out.wav  (5.0s | sr=16000)

<IPython.lib.display.Audio object>
  [ 3/5] BLEU= 10.8 ChrF= 45.4 RTF=0.110  id=1662
              pred: অ্যালোয়ি মূলত 2 বা একাধিক ধাতুর মিশ্রণ মনে রাখবেন না যে পিআইআর তে অনেক উপাদান র
[audio] Saved EN2BN_s3in.wav (0.3 MB)
  EN2BN_s3in.wav  (8.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved EN2BN_s3out.wav (0.1 MB)
  EN2BN_s3out.wav  (4.5s | sr=16000)

<IPython.lib.display.Audio object>
  [ 4/5] BLEU=  6.2 ChrF= 46.2 RTF=0.113  id=1663
              pred: চোকামো উপত্যকা চিলির শীর্ষস্থানীয় পর্বতারোহণের গন্তব্য যা দক্ষিণ আমেরিকার য়োসি
[audio] Saved EN2BN_s4in.wav (0.4 MB)
  EN2BN_s4in.wav  (12.1s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved EN2BN_s4out.wav (0.3 MB)
  EN2BN_s4out.wav  (8.0s | sr=16000)

<IPython.lib.display.Audio object>
  [ 5/5] BLEU=  8.8 ChrF= 48.0 RTF=0.104  id=1664
              pred: দুটি শুকনো শক্তি একসাথে ঘূর্ণিয়ে তারপর পরিষ্কার পাতলা হাত দিয়ে তাদের একটি বলে 
[audio] Saved EN2BN_s5in.wav (0.3 MB)
  EN2BN_s5in.wav  (8.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved EN2BN_s5out.wav (0.2 MB)
  EN2BN_s5out.wav  (5.4s | sr=16000)

<IPython.lib.display.Audio object>

  Summary: BLEU=9.38  ChrF=45.72  RTF=0.1539  Params=1039.1M
```

### Cell 11 (code, score=73)
```python
# ── Cell 9: Benchmark — Bengali → English ─────────────────────────────────────
# Swap: wav = Bengali audio, ref = English transcription.
# src_by_id / tgt_by_id are keyed merged rows with 'bn_audio', 'en_text' etc.

bn2en_samples = []
for sid in common_ids:
    row = tgt_by_id[sid]   # same merged row — has both bn_audio and en_text
    bn2en_samples.append(
        dict(
            id=sid,
            wav=_load_wav(row['bn_audio']),   # Bengali audio as input
            ref=row['en_text'],               # English text as reference
            bn_text=row['bn_text']
        )
    )

bn2en_results, bn2en_summary = run_benchmark(
    model, bn2en_samples,
    label='BN2EN',
    tgt_lang='eng',
    save_n=N_EVAL
)
```
OUTPUT:
```text
============================================================
  BENCHMARK: BN2EN
  Samples: 5  Target: eng
============================================================

  GPU mem: 2.11 GB alloc / 2.32 GB reserved
  [ 1/5] BLEU=  4.5 ChrF= 35.3 RTF=0.098  id=1660
              pred: "in the field of culture, there was a large increase from the slick to the slick
[audio] Saved BN2EN_s1in.wav (0.4 MB)
  BN2EN_s1in.wav  (11.6s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved BN2EN_s1out.wav (0.2 MB)
  BN2EN_s1out.wav  (6.1s | sr=16000)

<IPython.lib.display.Audio object>
  [ 2/5] BLEU=  0.0 ChrF=  0.0 RTF=0.049  id=1661
              pred: "
[audio] Saved BN2EN_s2in.wav (0.4 MB)
  BN2EN_s2in.wav  (11.7s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved BN2EN_s2out.wav (0.0 MB)
  BN2EN_s2out.wav  (0.4s | sr=16000)

<IPython.lib.display.Audio object>
  [ 3/5] BLEU=  6.4 ChrF= 29.4 RTF=0.062  id=1662
              pred: "concrete" is basically two or more metal mixture.
[audio] Saved BN2EN_s3in.wav (0.4 MB)
  BN2EN_s3in.wav  (11.0s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved BN2EN_s3out.wav (0.1 MB)
  BN2EN_s3out.wav  (3.3s | sr=16000)

<IPython.lib.display.Audio object>
  [ 4/5] BLEU= 11.4 ChrF= 41.5 RTF=0.066  id=1663
              pred: cocamau plantation is known as the yossemite of south america granite rock and d
[audio] Saved BN2EN_s4in.wav (0.6 MB)
  BN2EN_s4in.wav  (17.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved BN2EN_s4out.wav (0.2 MB)
  BN2EN_s4out.wav  (6.7s | sr=16000)

<IPython.lib.display.Audio object>
  [ 5/5] BLEU=  0.0 ChrF=  0.0 RTF=0.048  id=1664
              pred: "
[audio] Saved BN2EN_s5in.wav (0.4 MB)
  BN2EN_s5in.wav  (11.8s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved BN2EN_s5out.wav (0.0 MB)
  BN2EN_s5out.wav  (0.3s | sr=16000)

<IPython.lib.display.Audio object>

  Summary: BLEU=4.47  ChrF=21.24  RTF=0.0645  Params=1039.1M
```

### Cell 12 (code, score=85)
```python
# ── Cell 10: Comparison plot (EN↔BN) ──────────────────────────────────────────

sample_labels = [f"S{i+1}" for i in range(N_EVAL)]
x     = np.arange(N_EVAL)
bar_w = 0.35

bleu_en2bn = [r['bleu'] for r in en2bn_results]
bleu_bn2en = [r['bleu'] for r in bn2en_results]
chrf_en2bn = [r['chrf'] for r in en2bn_results]
chrf_bn2en = [r['chrf'] for r in bn2en_results]
rtf_en2bn  = [r['rtf']  for r in en2bn_results]
rtf_bn2en  = [r['rtf']  for r in bn2en_results]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f'SeamlessM4T v2 Compressed (Phase 7 DoRA, {en2bn_summary["params_M"]:.0f}M params, −39.3%)\n'
    f'EN↔BN Benchmark  |  N={N_EVAL} test samples  |  FLEURS en_us / bn_in',
    fontsize=12, fontweight='bold'
)

C_EN2BN = '#2196F3'
C_BN2EN = '#FF7043'

def _annotate(ax, bars, fmt='{:.1f}'):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                fmt.format(bar.get_height()), ha='center', va='bottom', fontsize=8)

# BLEU
ax = axes[0]
b1 = ax.bar(x - bar_w/2, bleu_en2bn, bar_w, label='EN→BN', color=C_EN2BN, alpha=0.85)
b2 = ax.bar(x + bar_w/2, bleu_bn2en, bar_w, label='BN→EN', color=C_BN2EN, alpha=0.85)
ax.axhline(en2bn_summary['avg_bleu'], color=C_EN2BN, ls='--', lw=1.5,
           label=f'EN→BN avg={en2bn_summary["avg_bleu"]:.1f}')
ax.axhline(bn2en_summary['avg_bleu'], color=C_BN2EN, ls='--', lw=1.5,
           label=f'BN→EN avg={bn2en_summary["avg_bleu"]:.1f}')
_annotate(ax, list(b1)+list(b2))
ax.set_title('BLEU (higher = better)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(sample_labels)
ax.set_ylabel('BLEU'); ax.legend(fontsize=8)

# ChrF
ax = axes[1]
b1 = ax.bar(x - bar_w/2, chrf_en2bn, bar_w, label='EN→BN', color=C_EN2BN, alpha=0.85)
b2 = ax.bar(x + bar_w/2, chrf_bn2en, bar_w, label='BN→EN', color=C_BN2EN, alpha=0.85)
ax.axhline(en2bn_summary['avg_chrf'], color=C_EN2BN, ls='--', lw=1.5,
           label=f'EN→BN avg={en2bn_summary["avg_chrf"]:.1f}')
ax.axhline(bn2en_summary['avg_chrf'], color=C_BN2EN, ls='--', lw=1.5,
           label=f'BN→EN avg={bn2en_summary["avg_chrf"]:.1f}')
_annotate(ax, list(b1)+list(b2))
ax.set_title('ChrF (higher = better)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(sample_labels)
ax.set_ylabel('ChrF'); ax.legend(fontsize=8)

# RTF
ax = axes[2]
b1 = ax.bar(x - bar_w/2, rtf_en2bn, bar_w, label='EN→BN', color=C_EN2BN, alpha=0.85)
b2 = ax.bar(x + bar_w/2, rtf_bn2en, bar_w, label='BN→EN', color=C_BN2EN, alpha=0.85)
ax.axhline(en2bn_summary['avg_rtf'], color=C_EN2BN, ls='--', lw=1.5,
           label=f'EN→BN avg={en2bn_summary["avg_rtf"]:.3f}')
ax.axhline(bn2en_summary['avg_rtf'], color=C_BN2EN, ls='--', lw=1.5,
           label=f'BN→EN avg={bn2en_summary["avg_rtf"]:.3f}')
_annotate(ax, list(b1)+list(b2), fmt='{:.3f}')
ax.set_title('RTF (lower = faster)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(sample_labels)
ax.set_ylabel('Real-Time Factor'); ax.legend(fontsize=8)

plt.tight_layout()
save_figure(fig, 'en_bn_benchmark_comparison.png')
plt.show()

print('\n📊 Comparison Summary')
print(f'{"Direction":<12} {"BLEU":>8} {"ChrF":>8} {"RTF":>8}')
print('-' * 40)
print(f'{"EN→BN":<12} {en2bn_summary["avg_bleu"]:>8.2f} {en2bn_summary["avg_chrf"]:>8.2f} {en2bn_summary["avg_rtf"]:>8.4f}')
print(f'{"BN→EN":<12} {bn2en_summary["avg_bleu"]:>8.2f} {bn2en_summary["avg_chrf"]:>8.2f} {bn2en_summary["avg_rtf"]:>8.4f}')
```
OUTPUT:
```text
[fig] Saved e:\NSU\Semester 261\CSE465\CSE465_Project\Rayed\Pruning SeamlessM4T\seamless_local\local_working/figures/en_bn_benchmark_comparison.png

<Figure size 1920x600 with 3 Axes>
[image/png output omitted]

📊 Comparison Summary
Direction        BLEU     ChrF      RTF
----------------------------------------
EN→BN            9.38    45.72   0.1539
BN→EN            4.47    21.24   0.0645
```

### Cell 13 (code, score=16)
```python
# # ── Cell 11: 🎤 Live mic input → translated speech output ─────────────────────
# #
# # HOW TO USE:
# #   1. Set DIRECTION below: 'en2bn'  or  'bn2en'
# #   2. Set RECORD_SECONDS to how long you want to speak
# #   3. Run the cell — speak when you see "Recording... Speak now!"
# #   4. Both input and translated audio play back in the cell output
# #
# # NOTE — sounddevice on Linux may need:
# #   sudo apt install portaudio19-dev
# #   (already installed if you ran the uv install block above)

# import sounddevice as sd

# DIRECTION      = 'en2bn'   # 'en2bn'  or  'bn2en'
# RECORD_SECONDS = 6
# SR_MIC         = 16000

# if DIRECTION == 'en2bn':
#     mic_src_lang, mic_tgt_lang = 'eng', 'ben'
#     direction_label = 'English → Bengali'
# elif DIRECTION == 'bn2en':
#     mic_src_lang, mic_tgt_lang = 'ben', 'eng'
#     direction_label = 'Bengali → English'
# else:
#     raise ValueError("DIRECTION must be 'en2bn' or 'bn2en'")

# print(f'Mode: {direction_label}')
# print(f'Recording {RECORD_SECONDS}s from microphone... Speak now!')

# recording = sd.rec(
#     int(RECORD_SECONDS * SR_MIC),
#     samplerate=SR_MIC,
#     channels=1,
#     dtype='float32'
# )
# sd.wait()  # blocks until done
# mic_wav = recording.squeeze()

# # Normalise
# mic_wav -= mic_wav.mean()
# peak = np.abs(mic_wav).max()
# if peak > 0:
#     mic_wav /= peak

# print(f'Recorded {len(mic_wav)/SR_MIC:.1f}s')

# # Save & play input (same helpers as run_benchmark)
# save_audio(mic_wav, SR_MIC, 'mic_input.wav')
# play(mic_wav, SR_MIC, label='🎙️ Your input:')

# # Full S2ST: speech → translated speech
# print(f'\nTranslating ({direction_label})...')
# pred_text, out_wav = run_s2st(model, mic_wav, tgt_lang=mic_tgt_lang)

# print(f'Predicted transcript: {pred_text}')

# # Cast to float32 — sounddevice & soundfile don't accept float16
# out_wav = np.array(out_wav, dtype=np.float32)

# # Save & play output
# save_audio(out_wav, model.config.sampling_rate, 'mic_translated_output.wav')
# play(out_wav, model.config.sampling_rate, label='🔊 Translated output:')

# # Also play through speakers
# print('Playing back through speakers...')
# sd.play(out_wav, samplerate=model.config.sampling_rate)
# sd.wait()

# print(f'\nFiles saved to {AUDIO_DIR}/')
```

### Cell 14 (code, score=63)
```python
stopehere
# from modelscope import snapshot_download
# snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
```
OUTPUT:
```text
ERROR: NameError name 'stopehere' is not defined
[31m---------------------------------------------------------------------------[39m
[31mNameError[39m                                 Traceback (most recent call last)
[36mCell[39m[36m [39m[32mIn[13][39m[32m, line 1[39m
[32m----> [39m[32m1[39m stopehere
[32m      2[39m [38;5;66;03m# from modelscope import snapshot_download[39;00m
[32m      3[39m [38;5;66;03m# snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')[39;00m

[31mNameError[39m: name 'stopehere' is not defined
```

### Cell 15 (code, score=12)
```python
from voice_cloning_module import SeamlessVoiceCloningPipeline
import soundfile as sf
import os

pipeline = SeamlessVoiceCloningPipeline(
    seamless_model=pruned_model,
    seamless_processor=processor,
    cosyvoice_model_dir='pretrained_models/CosyVoice-300M',
    cosyvoice_repo_dir='CosyVoice',   # the folder you just git-cloned
    tgt_lang='ben',                    # Bengali — was 'cmn' (Mandarin) in the template
)
en_audio = eval_samples[0]['wav']
cloned_bn = pipeline.translate_and_clone(en_audio, sr=16_000)

# Save & play
os.makedirs('audio_out', exist_ok=True)
sf.write('audio_out/cloned_bengali.wav', cloned_bn, 16_000)

from IPython.display import Audio, display
display(Audio('audio_out/cloned_bengali.wav'))
print("Done — saved to audio_out/cloned_bengali.wav")
```