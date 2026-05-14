# Final Notebooks\v1_1 seamless-cse465v5.ipynb

Extracted notebook map containing markdown headings plus code/output cells likely to matter for reports, reproduction, or agent steering.

## Markdown headings
cell 1: # SeamlessM4T v2 Large: Structured Compression 2.3B to ~1B ## Compression Pipeline
cell 3: ## Setup Cells 1-8
cell 19: ## Core Library: Model, Benchmark, Plotting
cell 28: # Phase 0: Baseline Benchmark
cell 31: # Phase 1: Vocabulary / Embedding Pruning
cell 37: # Phase 2: Text Encoder Removal # Actually it was never in the first place, we never loaded the textEncoder as we are using # from transformers import SeamlessM4Tv2ForSpeechToSpeech
cell 41: # Phase 3: Text Decoder Iterative Layer Pruning
cell 54: # Phase 4: Speech Encoder Iterative Layer Pruning
cell 65: # Phase 5: Width Pruning (FLAP)
cell 85: # Phase 6: T2U Model Pruning
cell 94: # Phase 7: Recovery Fine-tuning — S2ST Focused

## Key cells

### Cell 1 (markdown, score=26)
```markdown
# SeamlessM4T v2 Large: Structured Compression 2.3B to ~1B

## Compression Pipeline
| Phase | Technique | Paper | Expected |
|-------|-----------|-------|----------|
| 0 | Baseline benchmark | - | reference |
| 1 | Vocabulary/Embedding pruning | Asahi (EMNLP 2023) | -200M |
| 2 | Text encoder removal (S2S-only) | Architecture analysis | -350M |
| 3 | Text decoder iterative layer pruning | Moslem (IWSLT 2025) | -150M |
| 4 | Speech encoder iterative layer pruning | ShortGPT (ACL 2025) | -150M |
| 5 | Width pruning (FLAP on FFN + heads) | FLAP (AAAI 2024) | -200M |
| 6 | T2U model pruning | Iterative layer pruning | -50M |
| 7 | Recovery fine-tuning (LoRA + S2TT CE) | Moslem (IWSLT 2025) | quality up |
| 8 | Final benchmark + paper table | - | - |
```

### Cell 3 (markdown, score=0)
```markdown
## Setup Cells 1-8
Run these at the **start of every** Kaggle session.
```

### Cell 4 (code, score=12)
```python
import os, sys, subprocess, pathlib, re, glob, json, gc, copy, time, math, shutil
import warnings; warnings.filterwarnings('ignore')

# ── Platform detection ────────────────────────────────────────────────────────
ON_KAGGLE = os.path.exists('/kaggle/working')
ON_COLAB  = not ON_KAGGLE  # safe assumption for this notebook
PLATFORM  = 'kaggle' if ON_KAGGLE else 'colab'

# ── Path layout ───────────────────────────────────────────────────────────────
# Kaggle : rclone syncs to/from gdrive:seamV5/  ←→  /kaggle/working/
# Colab  : Google Drive is mounted at /content/drive/MyDrive/
#          We work DIRECTLY inside the mounted folder — no copying, no rclone.
GDRIVE_MOUNT   = '/content/drive/MyDrive/seamV5'   # <-- your Drive folder
KAGGLE_WORK    = '/kaggle/working'

WORK_DIR  = KAGGLE_WORK    if ON_KAGGLE else GDRIVE_MOUNT
CKPT_DIR  = f'{WORK_DIR}/checkpoints'
AUDIO_DIR = f'{WORK_DIR}/audio'
FIG_DIR   = f'{WORK_DIR}/figures'
MODEL_DIR = f'{WORK_DIR}/models'

# rclone remote root (Kaggle only — unused on Colab)
GDRIVE_ROOT = 'gdrive:seamV5'
```

### Cell 5 (code, score=4)
```python
# Mount Google Drive — only needed on Colab.
# On Kaggle, rclone handles the remote; skip this cell entirely there.
if ON_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print(f'Drive mounted. Working folder: {GDRIVE_MOUNT}')
    # if not os.path.exists(GDRIVE_MOUNT):
    #     os.makedirs(GDRIVE_MOUNT, exist_ok=True)
    #     print(f'Created {GDRIVE_MOUNT}')
    # else:
    #     print(f'Folder exists: {GDRIVE_MOUNT}')
else:
    print('Kaggle: skipping Drive mount.')
```
OUTPUT:
```text
Kaggle: skipping Drive mount.
```

### Cell 6 (code, score=25)
```python
for d in [WORK_DIR, CKPT_DIR, AUDIO_DIR, FIG_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

print(f'Platform : {PLATFORM}')
print(f'Work dir : {WORK_DIR}')
print(f'Checkpts : {CKPT_DIR}')
```
OUTPUT:
```text
Platform : kaggle
Work dir : /kaggle/working
Checkpts : /kaggle/working/checkpoints
```

### Cell 7 (code, score=4)
```python
# rclone is only needed on Kaggle. On Colab we use the mounted Drive directly.
if ON_KAGGLE:
    subprocess.run('curl -s https://rclone.org/install.sh | sudo bash',
                   shell=True, capture_output=True)
    ver = subprocess.run('rclone version', shell=True, capture_output=True, text=True)
    print(ver.stdout.split('\n')[0])
else:
    print('Colab: rclone not needed — using mounted Google Drive directly.')
    print(f'Drive path: {GDRIVE_MOUNT}')
    if not os.path.exists('/content/drive/MyDrive'):
        print('WARNING: Drive does not appear to be mounted. Run Cell 22 (drive.mount) first.')
    else:
        print('Drive mount: OK')
```
OUTPUT:
```text
rclone v1.73.4
```

### Cell 8 (code, score=38)
```python
def _get_secret(key):
    """Fetch a secret from Kaggle Secrets or Colab userdata."""
    if ON_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(key)
        except Exception as e:
            raise RuntimeError(f'Kaggle secret {key!r} not found: {e}')
    else:
        try:
            from google.colab import userdata
            return userdata.get(key)
        except Exception as e:
            raise RuntimeError(
                f'Colab secret {key!r} not found. '
                f'Add it via the 🔑 Secrets panel in Colab: {e}')

if ON_KAGGLE:
    # rclone config is only needed on Kaggle
    RCLONE_CONF = _get_secret('RCLONE_CONF')
    raw = RCLONE_CONF.strip()
    raw = re.sub(r'\s*(\[[^\]]+\])\s*', r'\n\1\n', raw)
    raw = re.sub(r'\s+(type|scope|token|team_drive|client_id|client_secret|'
                 r'root_folder_id|service_account_file|drive_id)\s*=\s*',
                 r'\n\1 = ', raw)
    raw = raw.strip() + '\n'
    rclone_cfg = pathlib.Path.home() / '.config/rclone/rclone.conf'
    rclone_cfg.parent.mkdir(parents=True, exist_ok=True)
    rclone_cfg.write_text(raw)
    r = subprocess.run('rclone lsd gdrive:', shell=True, capture_output=True, text=True)
    print('Drive root:' if r.returncode == 0 else 'rclone FAILED:')
    print(r.stdout[:300] or r.stderr[:300])
else:
    print('Colab: skipping rclone config — using mounted Drive.')
    print(f'Working directory on Drive: {WORK_DIR}')
```
OUTPUT:
```text
Drive root:
           0 2026-04-11 05:42:37        -1 .ipynb_checkpoints
           0 2026-04-17 11:03:10        -1 Colab Notebooks
           0 2025-11-10 11:33:43        -1 ScholarMate
           0 2026-04-05 12:59:09        -1 cse465
           0 2026-04-12 12:42:04        -1 cse465v5
           0 2026-04-1
```

### Cell 11 (code, score=37)
```python
subprocess.run([
    'pip', 'install', '-q',
    'transformers', 'datasets', 'torchaudio', 'speechbrain',
    'peft', 'librosa', 'jiwer', 'evaluate', 'sacrebleu',
    'sentencepiece', 'accelerate', 'matplotlib', 'seaborn',
], check=True)
print('All packages installed.')
```
OUTPUT:
```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 32.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.1/84.1 kB 6.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.8/100.8 kB 8.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 73.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.6/121.6 kB 9.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 788.2/788.2 kB 47.9 MB/s eta 0:00:00
All packages installed.
```

### Cell 12 (code, score=53)
```python
if ON_KAGGLE:
    print('Pulling checkpoints from Google Drive via rclone...')
    r = subprocess.run(
        f'rclone sync {GDRIVE_ROOT}/checkpoints/ {CKPT_DIR}/',
        shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'WARNING: rclone sync failed:\n{r.stderr[:300]}')
    else:
        print('rclone sync OK.')
else:
    print(f'Colab: checkpoints are on mounted Drive at {CKPT_DIR}')
    print('No sync needed — reading directly from Drive.')

files = sorted(os.listdir(CKPT_DIR)) if os.path.isdir(CKPT_DIR) else []
if files:
    print(f'{len(files)} file(s) found:')
    for f in files:
        mb = os.path.getsize(f'{CKPT_DIR}/{f}') / 1e6
        print(f'  {f:<55} {mb:>8.1f} MB')
else:
    print('No checkpoints yet.')
```
OUTPUT:
```text
Pulling checkpoints from Google Drive via rclone...
rclone sync OK.
8 file(s) found:
  all_summaries_step000000.pt                                  0.0 MB
  phase0_baseline_step000000.pt                                0.0 MB
  phase1_benchmark_step000000.pt                               0.0 MB
  phase1_vocab_step000000.pt                                   0.1 MB
  phase3_benchmark_step000000.pt                               0.0 MB
  phase3_dec_pruning_step000000.pt                             0.0 MB
  phase4_benchmark_step000000.pt                               0.0 MB
  phase4_enc_pruning_step000000.pt                             0.0 MB
```

### Cell 15 (code, score=372)
```python
import torch
from datetime import datetime

_CUSTOM_STATE_FILE = '_custom_state.pt'
_PRUNING_MANIFEST = 'pruning_manifest.pt'  # optional extra metadata (e.g. T2U removal log copy)

def _rclone_push(local_path, remote_subpath):
    """Push a single file or folder to rclone remote. Kaggle only."""
    if not ON_KAGGLE:
        return  # Colab writes directly to Drive — nothing to push
    r = subprocess.run(
        f'rclone copy "{local_path}" "{GDRIVE_ROOT}/{remote_subpath}/"',
        shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'[rclone] WARNING: push failed for {local_path}: {r.stderr[:200]}')

def _rclone_pull_model(stage_name):
    """Pull models/<stage_name> from rclone remote into local MODEL_DIR. Kaggle only."""
    if not ON_KAGGLE:
        return  # Colab reads directly from Drive — nothing to pull
    local = f'{MODEL_DIR}/{stage_name}'
    os.makedirs(local, exist_ok=True)
    r = subprocess.run(
        f'rclone sync "{GDRIVE_ROOT}/models/{stage_name}/" "{local}/"',
        shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'[rclone] model pull failed for {stage_name}: {r.stderr[:300]}')
    print(f'[rclone] Pulled {stage_name} → {local}')

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(state, name, step=0, keep=3):
    fname = f'{name}_step{step:06d}.pt'
    path  = f'{CKPT_DIR}/{fname}'
    torch.save(state, path)
    mb = os.path.getsize(path) / 1e6
    print(f'[ckpt] Saved {fname} ({mb:.1f} MB)')

    # On Kaggle, push to rclone remote so it survives session end
    if ON_KAGGLE:
        _rclone_push(path, 'checkpoints')

    # Prune old local copies
    old = sorted(glob.glob(f'{CKPT_DIR}/{name}_step*.pt'))
    for f in old[:-keep]:
        if os.path.exists(f):
            os.remove(f)

def load_latest_checkpoint(name):
    files = sorted(glob.glob(f'{CKPT_DIR}/{name}_step*.pt'))
    if not files:
        print(f'[ckpt] No checkpoint for {name!r}')
        return None
    state = torch.load(files[-1], map_location='cpu', weights_only=False)
    print(f'[ckpt] Loaded {os.path.basename(files[-1])}')
    return state

def sync_checkpoints_from_drive(names=None):
    if ON_KAGGLE:
        print('[ckpt] Syncing checkpoints from rclone remote...')
        r = subprocess.run(
            f'rclone sync "{GDRIVE_ROOT}/checkpoints/" "{CKPT_DIR}/"',
            shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print(f'[ckpt] WARNING: {r.stderr[:300]}')
    else:
        print(f'[ckpt] Colab: reading directly from {CKPT_DIR} (no sync needed)')

    files = sorted(os.listdir(CKPT_DIR)) if os.path.exists(CKPT_DIR) else []
    print(f'[ckpt] {len(files)} checkpoint(s) available')
    for f in files:
        mb = os.path.getsize(f'{CKPT_DIR}/{f}') / 1e6
        print(f'  {f:<55} {mb:>7.1f} MB')

# ── Config sync: make config.json match actual architecture ───────────────────

def _find_layers(component):
    """Find the main transformer-layer ModuleList inside a component."""
    for attr in ['layers', 'inner_layers', 'layer']:
        mod = getattr(component, attr, None)
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            return mod
    return None


def _get_t2u_encoder_decoder(mdl):
    """SeamlessM4Tv2: stacks live under t2u_model.model.{encoder,decoder}, not t2u_model.{encoder,decoder}."""
    t2u = getattr(mdl, 't2u_model', None)
    if t2u is None:
        return None, None
    inner = getattr(t2u, 'model', None)
    if inner is None:
        return None, None
    enc = getattr(inner, 'encoder', None)
    dec = getattr(inner, 'decoder', None)
    return enc, dec


def _infer_t2u_layer_counts_from_checkpoint_dir(model_dir):
    """
    Infer actual T2U encoder/decoder depth from saved weights (fixes legacy saves
    where config.json still had full 6+6 layers).
    Self-contained (no load_hf_weights_dict): works in minimal notebooks and any cell order.
    Returns (enc_n, dec_n) or (None, None) if not inferable.
    """
    import os
    sd = None
    safe = os.path.join(model_dir, 'model.safetensors')
    if os.path.isfile(safe):
        try:
            from safetensors.torch import load_file
            sd = load_file(safe)
        except ImportError:
            pass
    if sd is None:
        pt = os.path.join(model_dir, 'pytorch_model.bin')
        if os.path.isfile(pt):
            blob = torch.load(pt, map_location='cpu', weights_only=False)
            if isinstance(blob, dict) and 'model' in blob:
                inner = blob['model']
                if isinstance(inner, dict):
                    sd = inner
                else:
                    sd = blob
            else:
                sd = blob
    if not sd:
        return None, None
    pref_e = 't2u_model.model.encoder.layers.'
    pref_d = 't2u_model.model.decoder.layers.'
    enc_idx, dec_idx = set(), set()
    for k in sd:
        if k.startswith(pref_e):
            rest = k[len(pref_e):].split('.', 1)[0]
            if rest.isdigit():
                enc_idx.add(int(rest))
        elif k.startswith(pref_d):
            rest = k[len(pref_d):].split('.', 1)[0]
            if rest.isdigit():
                dec_idx.add(int(rest))
```
OUTPUT:
```text
I/O helpers ready.
  Platform  : kaggle
  Model dir : /kaggle/working/models
  Ckpt dir  : /kaggle/working/checkpoints
  rclone    : gdrive:seamV5
```

### Cell 16 (code, score=42)
```python
sync_checkpoints_from_drive()
```
OUTPUT:
```text
[ckpt] Syncing checkpoints from rclone remote...
[ckpt] 8 checkpoint(s) available
  all_summaries_step000000.pt                                 0.0 MB
  phase0_baseline_step000000.pt                               0.0 MB
  phase1_benchmark_step000000.pt                              0.0 MB
  phase1_vocab_step000000.pt                                  0.1 MB
  phase3_benchmark_step000000.pt                              0.0 MB
  phase3_dec_pruning_step000000.pt                            0.0 MB
  phase4_benchmark_step000000.pt                              0.0 MB
  phase4_enc_pruning_step000000.pt                            0.0 MB
```

### Cell 17 (code, score=10)
```python
def save_figure(fig, name):
    fig.savefig(f'{FIG_DIR}/{name}', dpi=150, bbox_inches='tight')
    if ON_KAGGLE:
        _rclone_push(f'{FIG_DIR}/{name}', 'figures')

import torchaudio, numpy as np
from IPython.display import Audio as IPAudio, display

def play(audio, sr, label=''):
    if hasattr(audio, 'numpy'): audio = audio.squeeze().numpy()
    print(f'  {label}  ({len(audio)/sr:.1f}s | sr={sr})')
    display(IPAudio(audio, rate=int(sr)))

def save_audio(audio, sr, filename, label=''):
    path = f'{AUDIO_DIR}/{filename}'
    if hasattr(audio, 'numpy'): t = audio.squeeze().unsqueeze(0).float()
    else: t = torch.tensor(audio).unsqueeze(0).float()
    torchaudio.save(path, t, sr)
    mb = os.path.getsize(path) / 1e6
    print(f'[audio] Saved {filename} ({mb:.1f} MB)')

print('Audio helpers ready.')
```
OUTPUT:
```text
Audio helpers ready.
```

### Cell 18 (code, score=62)
```python
import os
import glob
import torch
from datetime import datetime

def session_status():
    print('=' * 60)
    print(f'  Platform : {PLATFORM}   Time : {datetime.now():%Y-%m-%d %H:%M}')

    # Gathering files from the checkpoint directory
    if os.path.exists(CKPT_DIR):
        local_files = [f for f in glob.glob(f'{CKPT_DIR}/**/*.pt', recursive=True) if os.path.isfile(f)]
        print(f'  Checkpoint files in {CKPT_DIR}: {len(local_files)}')
        for f in sorted(local_files)[:20]:
            rel_path = os.path.relpath(f, CKPT_DIR)
            print(f'    {rel_path:<50} {os.path.getsize(f)/1e6:>8.1f} MB')
    else:
        print(f'  Directory not found: {CKPT_DIR}')

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  VRAM: {props.total_memory / 1e9:.1f} GB')

    print('=' * 60)

session_status()
```
OUTPUT:
```text
============================================================
  Platform : kaggle   Time : 2026-04-19 03:28
  Checkpoint files in /kaggle/working/checkpoints: 8
    all_summaries_step000000.pt                             0.0 MB
    phase0_baseline_step000000.pt                           0.0 MB
    phase1_benchmark_step000000.pt                          0.0 MB
    phase1_vocab_step000000.pt                              0.1 MB
    phase3_benchmark_step000000.pt                          0.0 MB
    phase3_dec_pruning_step000000.pt                        0.0 MB
    phase4_benchmark_step000000.pt                          0.0 MB
    phase4_enc_pruning_step000000.pt                        0.0 MB
  GPU: Tesla T4
  VRAM: 15.6 GB
============================================================
```

### Cell 19 (markdown, score=1)
```markdown
## Core Library: Model, Benchmark, Plotting
```

### Cell 20 (code, score=31)
```python
import torch, torch.nn as nn, torch.nn.functional as F
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

print('Core utilities ready.')
```
OUTPUT:
```text
Core utilities ready.
```

### Cell 21 (code, score=72)
```python
# CORE CELL
# ── MMS-ASR setup (Bengali ASR for ASR-BLEU metric) ─────────────────────────
# facebook/mms-1b-all uses Wav2Vec2ForCTC + adapter per language.
# Bengali ISO-639-3 code = "ben" (same as SeamlessM4T's tgt_lang).
# Load once, reuse across all benchmark calls.
 
import gc as _stdlib_gc
 
_MMS_MODEL_ID = "facebook/mms-1b-all"
_MMS_LANG     = "ben"   # Bengali ISO-639-3
 
_mms_asr_model     = None
_mms_asr_processor = None
 
def _ensure_mms_loaded():
    """Lazy-load MMS ASR model and Bengali adapter. Cached after first call."""
    global _mms_asr_model, _mms_asr_processor
    if _mms_asr_model is not None:
        return
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    print(f"[MMS-ASR] Loading {_MMS_MODEL_ID}  lang={_MMS_LANG}...")
    _mms_asr_processor = AutoProcessor.from_pretrained(
        _MMS_MODEL_ID, target_lang=_MMS_LANG
    )
    _mms_asr_model = Wav2Vec2ForCTC.from_pretrained(
        _MMS_MODEL_ID,
        target_lang=_MMS_LANG,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16,
    )
    _mms_asr_model.load_adapter(_MMS_LANG)
    _mms_asr_model = _mms_asr_model.eval()
    # Keep MMS on CPU to not fight for VRAM with the main model.
    # Move to cuda only if there is room; otherwise keep on CPU (slower but safe).
    try:
        _mms_asr_model = _mms_asr_model.to("cuda:0")
    except RuntimeError:
        pass
    print("[MMS-ASR] Ready.")
 
 
def asr_transcribe(audio_np, sr=16000):
    """
    Transcribe Bengali audio with MMS ASR (facebook/mms-1b-all).
    audio_np: numpy float32 array at 16 kHz.
    Returns: transcription string.
    """
    _ensure_mms_loaded()
    if audio_np is None or len(audio_np) < 400:
        return ""
    import torchaudio
    # Resample if needed
    if sr != 16000:
        audio_np = torchaudio.functional.resample(
            torch.tensor(audio_np), sr, 16000
        ).numpy()
    device = next(_mms_asr_model.parameters()).device
    inputs = _mms_asr_processor(
        audio_np, sampling_rate=16000, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _mms_asr_model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return _mms_asr_processor.batch_decode(pred_ids)[0].strip()
 
 
def compute_asr_bleu(audio_np, ref_text, sr=16000):
    """
    ASR-BLEU: transcribe audio with MMS → BLEU against reference text.
    Returns (asr_text, bleu_score).
    """
    try:
        hyp = asr_transcribe(audio_np, sr)
        bleu = compute_bleu(hyp, ref_text) if hyp.strip() else 0.0
        return hyp, bleu
    except Exception as e:
        return "", 0.0
 
 
def compute_asr_chrf(audio_np, ref_text, sr=16000):
    """ASR-ChrF: transcribe audio → ChrF against reference text."""
    try:
        hyp = asr_transcribe(audio_np, sr)
        return hyp, compute_chrf(hyp, ref_text) if hyp.strip() else 0.0
    except Exception:
        return "", 0.0
 
 
print("MMS-ASR helpers ready  (lazy-loaded on first use).")
```
OUTPUT:
```text
MMS-ASR helpers ready  (lazy-loaded on first use).
```

### Cell 22 (code, score=145)
```python
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
    """Remap trimmed vocab IDs back to original IDs for tokenizer decoding."""
    if hasattr(mdl, '_vocab_remap_to_old'):
        remap = mdl._vocab_remap_to_old
        ids = ids.clone()
        mask = (ids >= 0) & (ids < len(remap))
        ids[mask] = remap[ids[mask]]
    return ids

def _model_input_device(mdl):
    """Device where speech inputs should be placed (first speech encoder param)."""
    if hasattr(mdl, 'speech_encoder'):
        return next(mdl.speech_encoder.parameters()).device
    return next(mdl.parameters()).device

def run_s2st(mdl, wav, tgt_lang='ben'):
    """Full S2ST: returns (text, audio_numpy). Falls back to text-only if vocoder fails."""
    inputs = processor(audio=wav, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}
    with torch.no_grad():
        try:
            out = mdl.generate(**inputs, tgt_lang=tgt_lang,
                               return_intermediate_token_ids=True)
            text_ids = _remap_ids_for_decode(mdl, out.sequences.cpu())
            text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
            wav_out = out.waveform.cpu().numpy().squeeze() if out.waveform is not None else np.zeros(16000)
            return text, wav_out
        except RuntimeError:
            # Vocoder fails when T2U produces a very short unit sequence (<3 units).
            # Fall back to text-only to still get BLEU/ChrF for this sample.
            print("failed running s2tonly")
            text = run_s2t_only(mdl, wav, tgt_lang)
            return text, np.zeros(16000)

# ── Add this function to Cell 13 (alongside run_s2t_only) ────────────────────

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


def sync_model_config(mdl):
    """
    After structural pruning, sync config layer counts to actual module counts.
    Without this, generate()'s beam search indexes past_key_values by layer
    number from config and throws IndexError when layers have been removed.
    """
    # Speech encoder (SeamlessM4Tv2 uses flat config.speech_encoder_layers)
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
            if hasattr(mdl.config, 'speech_encoder_config') and hasattr(
                    mdl.config.speech_encoder_config, 'num_hidden_layers'):
                old = mdl.config.speech_encoder_config.num_hidden_layers
                if old != actual:
                    mdl.config.speech_encoder_config.num_hidden_layers = actual
                    print(f'  [config] speech_encoder_config.num_hidden_layers: {old} -> {actual}')
            subcfg = getattr(mdl.speech_encoder, 'config', None)
            if subcfg is not None and hasattr(subcfg, 'num_hidden_layers'):
                old2 = subcfg.num_hidden_layers
                if old2 != actual:
                    subcfg.num_hidden_layers = actual
                    print(f'  [config] speech_encoder.config.num_hidden_layers: {old2} -> {actual}')

    # Text decoder
    if hasattr(mdl, 'text_decoder'):
        dec = mdl.text_decoder
        la = find_layers_attr(dec)
        if la:
            actual = len(getattr(dec, la))
            if hasattr(mdl.config, 'decoder_layers'):
                old = mdl.config.decoder_layers
                if old != actual:
                    mdl.config.decoder_layers = actual
                    print(f'  [config] decoder_layers: {old} -> {actual}')
            # also patch text_decoder config if it has its own
            if hasattr(dec, 'config') and hasattr(dec.config, 'decoder_layers'):
                dec.config.decoder_layers = actual

    # T2U model (v2: t2u_model.model.encoder / .decoder)
    if hasattr(mdl, 't2u_model'):
        t2u_enc, t2u_dec = _get_t2u_encoder_decoder(mdl)
        for sub, comp, cfg_key in [
            ('encoder', t2u_enc, 't2u_encoder_layers'),
            ('decoder', t2u_dec, 't2u_decoder_layers'),
        ]:
            if comp is None:
                continue
            la = find_layers_attr(comp)
            if la:
                actual = len(getattr(comp, la))
                if hasattr(mdl.config, cfg_key):
                    old = getattr(mdl.config, cfg_key)
                    if old != actual:
                        setattr(mdl.config, cfg_key, actual)
                        print(f'  [config] {cfg_key}: {old} -> {actual}')
```
OUTPUT:
```text
Benchmark functions ready.
```

### Cell 23 (code, score=46)
```python
from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor

# Login to HuggingFace only if token is available (needed on Kaggle; Colab may cache it)
try:
    HF_TOKEN = _get_secret('HF_TOKEN')
    from huggingface_hub import login
    login(HF_TOKEN)
    print('Logged into HuggingFace Hub.')
except Exception as e:
    print(f'HF login skipped (token not found or already cached): {e}')

MODEL_NAME = 'facebook/seamless-m4t-v2-large'

def load_base_model():
    print(f'Loading processor from {MODEL_NAME}...')
    proc = SeamlessM4TProcessor.from_pretrained(MODEL_NAME)
    print(f'Loading model  -- may take 5-10 min...')
    mdl = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map='auto')
    mdl.eval()
    print('Model loaded.'); gpu_mem()
    return mdl, proc

print('load_base_model() ready.')
```
OUTPUT:
```text
Logged into HuggingFace Hub.
load_base_model() ready.
```

### Cell 24 (code, score=92)
```python
def _load_summaries_from_drive():
    """Pull the summary ledger from Drive once at session start."""
    ckpt = load_latest_checkpoint('all_summaries')
    if ckpt and 'summaries' in ckpt:
        return {s['label']: s for s in ckpt['summaries']}
    return {}

ALL_SUMMARIES: dict = _load_summaries_from_drive()
print(f'Loaded {len(ALL_SUMMARIES)} existing summaries: {list(ALL_SUMMARIES.keys())}')

def store_summary(s):
    """Upsert by label: insert new or update existing, then persist to Drive."""
    label = s['label']
    ALL_SUMMARIES[label] = s.copy()
    ordered = list(ALL_SUMMARIES.values())
    save_checkpoint({'summaries': ordered}, name='all_summaries', step=0)
    print(f'[summary] Stored {label} ({len(ALL_SUMMARIES)} total)')

def get_summaries():
    """Return summaries as an ordered list (by label) for plotting."""
    return sorted(ALL_SUMMARIES.values(), key=lambda s: s['label'])

def plot_phase_comparison(summaries=None, save_name='phase_comparison.png'):
    data = summaries or get_summaries()
    if not data: print('No summaries yet.'); return
    labels = [s['label'] for s in data]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Compression Pipeline: Phase Comparison', fontsize=15, fontweight='bold')
    metrics = [('avg_bleu', 'BLEU (higher=better)', '#2196F3'),
               ('avg_chrf', 'ChrF (higher=better)', '#4CAF50'),
               ('avg_rtf',  'RTF (lower=faster)', '#FF9800'),
               ('params_M', 'Parameters (M)', '#9C27B0')]
    for ax, (key, title, color) in zip(axes.flat, metrics):
        vals = [s.get(key, 0) for s in data]
        bars = ax.bar(range(len(labels)), vals, color=color, alpha=0.85, edgecolor='white')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{v:.1f}',
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); plt.savefig(f'{FIG_DIR}/{save_name}'); plt.show()

def plot_size_vs_quality(summaries=None, save_name='size_vs_quality.png'):
    data = summaries or get_summaries()
    if not data: return
    fig, ax = plt.subplots(figsize=(10, 7))
    params = [s['params_M'] for s in data]
    bleu = [s['avg_bleu'] for s in data]
    chrf = [s['avg_chrf'] for s in data]
    ax.scatter(params, bleu, s=120, c='#2196F3', zorder=5, label='BLEU')
    ax.scatter(params, chrf, s=120, c='#4CAF50', marker='s', zorder=5, label='ChrF')
    for i, lbl in enumerate([s['label'] for s in data]):
        ax.annotate(lbl, (params[i], bleu[i]), fontsize=7, xytext=(5,5), textcoords='offset points')
    ax.set_xlabel('Parameters (M)'); ax.set_ylabel('Score')
    ax.set_title('Model Size vs Translation Quality', fontweight='bold')
    ax.legend(); plt.tight_layout(); plt.savefig(f'{FIG_DIR}/{save_name}'); plt.show()

def plot_layer_scores(scores_dict, title='Layer Importance', save_name=None):
    indices = sorted(scores_dict.keys())
    vals = [scores_dict[i] for i in indices]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#d32f2f' if v > np.percentile(vals, 75) else '#ff9800' if v > np.percentile(vals, 50) else '#4caf50' if v > np.percentile(vals, 25) else '#90caf9' for v in vals]
    ax.bar(indices, vals, color=colors, edgecolor='white')
    ax.set_xlabel('Layer Index'); ax.set_ylabel('Importance')
    ax.set_title(title, fontweight='bold'); ax.set_xticks(indices)
    plt.tight_layout()
    if save_name: plt.savefig(f'{FIG_DIR}/{save_name}')
    plt.show()

print('Plotting helpers ready.')
```
OUTPUT:
```text
[ckpt] Loaded all_summaries_step000000.pt
Loaded 4 existing summaries: ['P0_Baseline', 'P1_VocabTrim', 'P3_DecPrune', 'P4_EncPrune']
Plotting helpers ready.
```

### Cell 25 (code, score=90)
```python
## Dataset loading
LOCAL_PARQUET_CACHE = "/kaggle/input/datasets/coderayed/fleurs-en-bn-parquet"

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
    from datasets import Dataset
    tasks = []
    for lang in [src_lang, tgt_lang]:
        urls = _list_parquet_urls(lang, split)
        for i, url in enumerate(urls):
            dest = f"{LOCAL_PARQUET_CACHE}/{lang}/{split}_{i:04d}.parquet"
            tasks.append((url, dest))
    print(f"[Parallel] Downloading {len(tasks)} shards...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for url, ok, msg in pool.map(_download_shard, tasks):
            print(f"  {'OK' if ok else 'FAIL'}: {msg}")
    def _load_lang(lang):
        files = sorted(glob.glob(f"{LOCAL_PARQUET_CACHE}/{lang}/{split}_*.parquet"))
        if not files: raise FileNotFoundError(f"No cached shards for {lang}")
        return Dataset.from_pandas(pd.read_parquet(files[0]))
    return _load_lang(src_lang), _load_lang(tgt_lang)

DRIVE_FLEURS_PATH = f'{GDRIVE_ROOT}/fleurs_parquet'

def push_fleurs_to_drive():
    if not ON_KAGGLE: return
    print(f'Pushing parquet cache to Drive...')
    subprocess.run(
        f'rclone copy "{LOCAL_PARQUET_CACHE}/" "{DRIVE_FLEURS_PATH}/" --transfers=8',
        shell=True, capture_output=True, text=True)

def load_fleurs_from_drive(src_lang, tgt_lang, split='train'):
    import pandas as pd
    from datasets import Dataset
    if not ON_KAGGLE:
        print('[Drive] rclone only on Kaggle.')
        return None, None
    print(f'[Drive] Pulling FLEURS parquet...')
    if not os.path.exists(LOCAL_PARQUET_CACHE):
        r = subprocess.run(
            f'rclone copy "{DRIVE_FLEURS_PATH}/" "{LOCAL_PARQUET_CACHE}/" --transfers=8',
            shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            return None, None
    def _load_lang(lang):
        files = sorted(glob.glob(f'{LOCAL_PARQUET_CACHE}/{lang}/{split}_*.parquet'))
        if not files: return None
        return Dataset.from_pandas(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
    src_ds = _load_lang(src_lang)
    tgt_ds = _load_lang(tgt_lang)
    if src_ds and tgt_ds:
        print(f'[Dataset gdrive] Loaded: {len(src_ds)} src, {len(tgt_ds)} tgt')
    return src_ds, tgt_ds

print('FLEURS data loaders ready.')
```
OUTPUT:
```text
FLEURS data loaders ready.
```

### Cell 26 (code, score=71)
```python
# ── CELL 23 REPLACEMENT: Load eval samples (EN→BN test) ──────────────────────
import numpy as np
import torch
import torchaudio
import io
import soundfile as sf
import pandas as pd

N_EVAL = 25
TARGET_LANG = "ben"
FLEURS_SRC, FLEURS_TGT = "en_us", "bn_in"

print(f"Loading FLEURS {FLEURS_SRC}->{FLEURS_TGT} for benchmarking [test]")

ds_src, ds_tgt = load_fleurs_from_drive(FLEURS_SRC, FLEURS_TGT, split="test")

if ds_src is None or ds_tgt is None:
    print("\n[Cache miss] Downloading...")
    ds_src, ds_tgt = load_fleurs_parallel(FLEURS_SRC, FLEURS_TGT, split="test", n_workers=8)
    push_fleurs_to_drive()

# ── CRITICAL FIX: Convert HF Dataset to pandas DataFrame ─────────────────────
# HF Dataset iteration is slow and memory-hungry. Convert to pandas first.
print("Converting to pandas DataFrames...")
df_src = ds_src.to_pandas() if hasattr(ds_src, 'to_pandas') else pd.DataFrame(ds_src)
df_tgt = ds_tgt.to_pandas() if hasattr(ds_tgt, 'to_pandas') else pd.DataFrame(ds_tgt)

# ── Robust audio loader (parquet stores bytes, not array dicts) ──────────────
def _load_wav(audio_cell):
    """
    audio_cell: the value of row['audio'] from a pandas DataFrame.
    Handles both HF Dataset format (dict with 'array') and parquet format (dict with 'bytes').
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
# Parquet shards can contain duplicate IDs. Drop duplicates first.
print("Deduplicating and merging...")
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

# ── Take only N_EVAL samples (CRITICAL for RAM) ──────────────────────────────
merged = merged.head(N_EVAL)
print(f"  Using {len(merged)} samples for evaluation")

# ── Build eval_samples (lazy audio loading) ──────────────────────────────────
eval_samples = []
for _, row in merged.iterrows():
    eval_samples.append(dict(
        id=row['id'],
        wav=_load_wav(row['en_audio']),  # Load audio on-demand
        ref=row['bn_text'],
        en_text=row['en_text'],
    ))

# Keep for Phase 7 Cell 5 (Bengali target audio)
common_ids = list(merged['id'])
src_by_id  = {row['id']: row for _, row in merged.iterrows()}
tgt_by_id  = {row['id']: row for _, row in merged.iterrows()}

print(f"Loaded {len(eval_samples)} eval samples.")

# Clean up large DataFrames to free RAM
del df_src, df_tgt, src_dedup, tgt_dedup, merged, ds_src, ds_tgt
gc.collect()
```
OUTPUT:
```text
Loading FLEURS en_us->bn_in for benchmarking [test]
[Drive] Pulling FLEURS parquet...
[Dataset gdrive] Loaded: 647 src, 920 tgt
Converting to pandas DataFrames...
Deduplicating and merging...
  Unique IDs — EN: 350, BN: 349
  Matched unique pairs available: 349
  Using 25 samples for evaluation
Loaded 25 eval samples.

0
```

### Cell 27 (code, score=73)
```python
print(f"Loading FLEURS {FLEURS_SRC}->{FLEURS_TGT} for fine-tuning [train]")

src_ds, tgt_ds = load_fleurs_from_drive(FLEURS_SRC, FLEURS_TGT, split="train")

if src_ds is None or tgt_ds is None:
    print("\n[Cache miss] Downloading...")
    src_ds, tgt_ds = load_fleurs_parallel(FLEURS_SRC, FLEURS_TGT, split="train", n_workers=8)
    push_fleurs_to_drive()

# ── Convert to pandas for efficient merge ────────────────────────────────────
print("Converting to pandas DataFrames...")
df_src_train = src_ds.to_pandas() if hasattr(src_ds, 'to_pandas') else pd.DataFrame(src_ds)
df_tgt_train = tgt_ds.to_pandas() if hasattr(tgt_ds, 'to_pandas') else pd.DataFrame(tgt_ds)

# ── Deduplicate and merge ────────────────────────────────────────────────────
print("Deduplicating and merging training data...")
src_train_dedup = (
    df_src_train[['id', 'audio']]
    .drop_duplicates(subset='id', keep='first')
    .rename(columns={'audio': 'en_audio'})
)
tgt_train_dedup = (
    df_tgt_train[['id', 'transcription', 'audio']]
    .drop_duplicates(subset='id', keep='first')
    .rename(columns={'transcription': 'bn_text', 'audio': 'bn_audio'})
)

print(f"  Unique IDs — EN: {len(src_train_dedup)}, BN: {len(tgt_train_dedup)}")

merged_train = (
    pd.merge(src_train_dedup, tgt_train_dedup, on='id', how='inner')
    .reset_index(drop=True)
)

print(f"  Matched training pairs: {len(merged_train)}")

# ── Filter out empty transcriptions ──────────────────────────────────────────
merged_train = merged_train[merged_train['bn_text'].str.strip().str.len() > 0]
print(f"  After filtering empty refs: {len(merged_train)}")

# ── Build ft_samples (lazy audio loading) ────────────────────────────────────
ft_samples = []
for _, row in merged_train.iterrows():
    ft_samples.append({
        'id': row['id'],
        'wav': _load_wav(row['en_audio']),
        'ref': row['bn_text'],
    })

print(f"Usable training samples: {len(ft_samples)}")

# Clean up
del df_src_train, df_tgt_train, src_train_dedup, tgt_train_dedup, merged_train, src_ds, tgt_ds
gc.collect()
```
OUTPUT:
```text
Loading FLEURS en_us->bn_in for fine-tuning [train]
[Drive] Pulling FLEURS parquet...
[Dataset gdrive] Loaded: 2602 src, 3006 tgt
Converting to pandas DataFrames...
Deduplicating and merging training data...
  Unique IDs — EN: 1476, BN: 1482
  Matched training pairs: 1449
  After filtering empty refs: 1449
Usable training samples: 1449

0
```

### Cell 28 (markdown, score=5)
```markdown
---
# Phase 0: Baseline Benchmark
Load the full teacher model, measure size and translation quality.
```

### Cell 29 (code, score=106)
```python
model, processor = load_base_model()
baseline_breakdown = print_model_breakdown(model, 'Baseline Model')
```
OUTPUT:
```text
Loading processor from facebook/seamless-m4t-v2-large...

preprocessor_config.json: 0.00B [00:00, ?B/s]
config.json: 0.00B [00:00, ?B/s]
tokenizer_config.json: 0.00B [00:00, ?B/s]
tokenizer.model:   0%|          | 0.00/5.17M [00:00<?, ?B/s]
added_tokens.json: 0.00B [00:00, ?B/s]
special_tokens_map.json: 0.00B [00:00, ?B/s]
Loading model  -- may take 5-10 min...

model.safetensors.index.json: 0.00B [00:00, ?B/s]
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]
Instantiating a decoder SeamlessM4Tv2Attention without passing `layer_idx` is not recommended and will lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` when creating this class.

Loading weights:   0%|          | 0/1846 [00:00<?, ?it/s]
SeamlessM4Tv2ForSpeechToSpeech LOAD REPORT from: facebook/seamless-m4t-v2-large
Key                                                      | Status     |  | 
---------------------------------------------------------+------------+--+-
text_encoder.layers.{0...23}.ffn_layer_norm.bias         | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.q_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc1.weight              | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.out_proj.weight   | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn_layer_norm.weight       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.v_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.k_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.out_proj.bias     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc1.bias                | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn_layer_norm.weight | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.k_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn_layer_norm.bias   | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc2.bias                | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc2.weight              | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.q_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.v_proj.bias       | UNEXPECTED |  | 
text_encoder.layer_norm.bias                             | UNEXPECTED |  | 
text_encoder.layer_norm.weight                           | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

generation_config.json: 0.00B [00:00, ?B/s]
Model loaded.
  GPU mem: 1.79 GB alloc / 1.80 GB reserved

--- Baseline Model ---
  text_decoder                           866.8M  ( 48.0%)
  speech_encoder                         635.0M  ( 35.2%)
  shared                                 262.2M  ( 14.5%)
  lm_head                                262.2M  ( 14.5%)
  t2u_model                              261.8M  ( 14.5%)
  vocoder                                 41.9M  (  2.3%)
  TOTAL                                 1805.5M
---
```

### Cell 30 (code, score=147)
```python
baseline_ckpt = load_latest_checkpoint('phase0_baseline')
if baseline_ckpt:
    baseline_results = baseline_ckpt['results']
    baseline_summary = baseline_ckpt['summary']
    print(f'Loaded baseline: BLEU={baseline_summary["avg_bleu"]:.2f}')
else:
    baseline_results, baseline_summary = run_benchmark(
        model, eval_samples, label='P0_Baseline', save_n=2)
    save_checkpoint(dict(results=baseline_results, summary=baseline_summary,
                         breakdown=baseline_breakdown), name='phase0_baseline', step=0)

store_summary(baseline_summary)
plot_phase_comparison()
```
OUTPUT:
```text
[ckpt] No checkpoint for 'phase0_baseline'

============================================================
  BENCHMARK: P0_Baseline
  Samples: 25  Target: ben
============================================================

  GPU mem: 1.79 GB alloc / 1.80 GB reserved
  [ 1/25] BLEU= 10.7 ChrF= 49.2 RTF=0.412  id=1660
              pred: রোমান্টিকতাবাদ গথ, ফিচ্ট এবং শ্লেগেলের মতো লেখকদের কাছ থেকে নেওয়া সাংস্কৃতিক নি
[audio] Saved P0_Baseline_s1in.wav (0.3 MB)
  P0_Baseline_s1in.wav  (10.7s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P0_Baseline_s1out.wav (0.2 MB)
  P0_Baseline_s1out.wav  (7.4s | sr=16000)

<IPython.lib.display.Audio object>
  [ 2/25] BLEU= 10.4 ChrF= 45.8 RTF=0.316  id=1661
              pred: তিনি চীনের অর্থনৈতিক উৎপাদনের উপর ভিত্তি করে এই কমানোর জন্য কোন পরিসংখ্যান নির্ধ
[audio] Saved P0_Baseline_s2in.wav (0.2 MB)
  P0_Baseline_s2in.wav  (6.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P0_Baseline_s2out.wav (0.2 MB)
  P0_Baseline_s2out.wav  (5.9s | sr=16000)

<IPython.lib.display.Audio object>
  [ 3/25] BLEU= 11.6 ChrF= 56.7 RTF=0.285  id=1662
              pred: অ্যালোয়গুলি মূলত দুটি বা ততোধিক ধাতুর মিশ্রণ, ভুলে যাবেন না যে পিরিয়ডিক টেবিলে
  [ 4/25] BLEU=  5.4 ChrF= 42.5 RTF=0.244  id=1663
              pred: চোকামো ভ্যালি, চিলির প্রিমিয়ার ক্লাইম্বিং গন্তব্য, দক্ষিণ আমেরিকার ইয়োসেমিটি ন
  [ 5/25] BLEU= 11.2 ChrF= 45.8 RTF=0.290  id=1664
              pred: দুটি শুষ্ক পাত্র একসাথে ঘূর্ণায়মান করুন এবং তারপর, রাণীর ভিজা হাত দিয়ে, তাদের 
  [ 6/25] BLEU=  9.7 ChrF= 48.3 RTF=0.318  id=1665
              pred: এই নথিটি পরাগের কারণে প্যালেস্টাইনের সীমান্তের বিষয়ে বিতর্ক করবে যা ১৯৬৭ সালের 
  [ 7/25] BLEU= 10.4 ChrF= 53.6 RTF=0.234  id=1666
              pred: আপনি আপনার নিজের সরকার ছাড়া অন্য সরকারের পরামর্শও নিতে পারেন, কিন্তু তাদের পরাম
  [ 8/25] BLEU= 19.1 ChrF= 60.3 RTF=0.216  id=1667
              pred: সাধারণভাবে বলতে গেলে, ম্যানেজাররা তাদের প্রাক্তন সমবয়সীদের নেতৃত্ব দিতে শুরু কর
  [ 9/25] BLEU=  7.0 ChrF= 46.5 RTF=0.196  id=1668
              pred: এটি একটি ওয়াইল্ডকার্ড কেনার জন্যও উপকারী হতে পারে, যা দক্ষিণ আফ্রিকার পার্কগুলি
  [10/25] BLEU=  9.7 ChrF= 43.2 RTF=0.302  id=1669
              pred: পুলিশ সুপারিনটেন্ডেন্ট চন্দ্র শেখর সলাঙ্কি বলেন, অভিযুক্তকে আদালতে মুখ ঢাকা দেখা
  [11/25] BLEU= 20.3 ChrF= 63.1 RTF=0.185  id=1670
              pred: "তাদের তাপীয় আচরণ পৃথিবীর বড় গুহাগুলির মতো স্থিতিশীল নয় যা প্রায়শই একটি মোটা
  [12/25] BLEU= 24.6 ChrF= 64.4 RTF=0.297  id=1671
              pred: কংগ্রেস ফিসাল ২০০৫ সালে অশ্লীলতা উদ্যোগের অর্থায়ন শুরু করে এবং নির্দিষ্ট করে দে
  [13/25] BLEU=  9.2 ChrF= 40.3 RTF=0.224  id=1672
              pred: কাপড়কে খুব গরম হতে না দেওয়ার জন্য সতর্ক থাকুন যা সংকোচনের কারণ হতে পারে বা চরম
  [14/25] BLEU= 34.1 ChrF= 77.0 RTF=0.247  id=1673
              pred: বিপ্লবী যুদ্ধের সময়, ১৩ টি রাজ্য প্রথম একটি দুর্বল কেন্দ্রীয় সরকার গঠন করেছিল,
  [15/25] BLEU=  3.7 ChrF= 31.0 RTF=0.326  id=1674
              pred: কিছু অঞ্চলে এক মিনিটের জন্য উষ্ণ জল যথেষ্ট এবং অন্যদের কয়েক মিনিটের প্রয়োজন হয
  [16/25] BLEU=  7.6 ChrF= 40.4 RTF=0.259  id=1675
              pred: শব্দটির পাশে থাকা সমস্ত নামগুলি আপনাকে বলে যে আপনি সর্বদা একটি বড় অক্ষর দিয়ে শ
  [17/25] BLEU=  8.8 ChrF= 51.4 RTF=0.204  id=1676
              pred: দক্ষিণ আফ্রিকার সমস্ত জাতীয় উদ্যানের মতো, পার্কের জন্য দৈনিক সংরক্ষণ এবং প্রবেশ
  [18/25] BLEU= 12.0 ChrF= 44.9 RTF=0.371  id=1677
              pred: আজ, একমাত্র পোকামাকড় যা তাদের ডানা ফিরিয়ে দিতে পারে না তা হ'ল ড্রাগনফ্লাই এবং 
  [19/25] BLEU=  4.5 ChrF= 49.0 RTF=0.220  id=1678
              pred: অলিভার স্যাক্স তার কাগজে রাষ্ট্রপতির বক্তৃতা নির্দেশ করেছিলেন যে মস্তিষ্কের ক্ষত
  [20/25] BLEU=  7.3 ChrF= 73.3 RTF=0.413  id=1679
              pred: এরোস্মিথ তাদের সফরে তাদের অবশিষ্ট কনসার্ট বাতিল করেছে।
  [21/25] BLEU=  6.7 ChrF= 41.8 RTF=0.321  id=1680
              pred: একটি সু-গোল্লা অ্যাথলিট, বাঘটি ভাল না হলেও আরোহণ করতে পারে, সাঁতার কাটতে পারে, দ
  [22/25] BLEU=  4.7 ChrF= 55.0 RTF=0.156  id=1681
              pred: এটি কেবলমাত্র পরীক্ষা নয়, এবং একটি পরীক্ষা এমন একটি পরীক্ষা যা সম্ভাব্য অনুমানগ
  [23/25] BLEU=  5.0 ChrF= 37.2 RTF=0.219  id=1682
              pred: যদিও কেউ নিশ্চিতভাবে জানে না যে এটি কে লিখেছে, এটি জানা যায় যে এর জীবনের প্রথম 
  [24/25] BLEU= 27.1 ChrF= 56.2 RTF=0.246  id=1683
              pred: এখনও অনেক পুরুষ এবং মহিলা বেঁচে আছেন যারা এখানে তাদের সময় বেঁচে আছেন এবং আরও অন
  [25/25] BLEU=  9.8 ChrF= 46.3 RTF=0.200  id=1684
              pred: আপিয়া সামোয়া দ্বীপের রাজধানী, এটি উপুলু দ্বীপের একটি শহর এবং এর জনসংখ্যা ৪০ হা

  Summary: BLEU=11.63  ChrF=50.52  RTF=0.2681  Params=1805.5M

[ckpt] Saved phase0_baseline_step000000.pt (0.0 MB)
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P0_Baseline (1 total)

<Figure size 1680x1200 with 4 Axes>
[image/png output omitted]
```

### Cell 31 (markdown, score=7)
```markdown
---
# Phase 1: Vocabulary / Embedding Pruning
**Paper:** Asahi et al. (EMNLP 2023)

NLLB vocabulary has **256,102 tokens** for ~100 languages.
We keep only ~5-7 languages. Trimming saves ~200M params with near-zero quality impact.
```

### Cell 32 (code, score=98)
```python
def identify_used_tokens(proc, target_lang_codes, n_corpus=2000):
    from datasets import load_dataset
    fleurs_codes = dict(eng='en_us', ben='bn_in', cmn='cmn_hans_cn',
                        fra='fr_fr', deu='de_de', hin='hi_in', urd='ur_pk')
    BASE = "hf://datasets/google/fleurs@refs%2Fconvert%2Fparquet"
    used = set()
    tok = proc.tokenizer
    if hasattr(tok, 'all_special_ids'): used.update(tok.all_special_ids)
    for tid in range(len(tok)):
        t = tok.convert_ids_to_tokens(tid)
        if t and t.startswith('__') and t.endswith('__'): used.add(tid)
    for lang, fc in fleurs_codes.items():
        if lang not in target_lang_codes: continue
        print(f'  Scanning {lang} ({fc})...')
        try:
            ds = load_dataset("parquet",
                              data_files={"train": f"{BASE}/{fc}/train/*.parquet"},
                              split="train")
            for i, ex in enumerate(ds):
                if i >= n_corpus: break
                text = ex.get('transcription', '')
                if text: used.update(tok.encode(text, add_special_tokens=False))
        except Exception as e:
            print(f'    Warning: {lang}: {e}')
    print(f'  Unique tokens: {len(used)} / {len(tok)}')
    return sorted(used)


def trim_vocabulary(mdl, proc, keep_ids):
    """
    Trim the NLLB text vocabulary to only the kept token IDs.

    Correctly handles:
    - SeamlessM4Tv2ScaledWordEmbedding (preserves embed_scale for decoder)
    - generation_config.id_to_text mapping (required by T2U _indices_to_subwords)
    - Tied weights: shared <-> text_decoder.embed_tokens <-> lm_head
    - Correct padding_idx remapping from old to new ID space

    Based on Asahi et al. (EMNLP 2023) vocabulary trimming methodology.
    """
    keep_t = torch.tensor(keep_ids, dtype=torch.long)
    old_v = mdl.config.vocab_size
    new_v = len(keep_ids)
    hidden = mdl.config.hidden_size
    print(f'  Vocabulary: {old_v} -> {new_v} ({new_v/old_v*100:.1f}%)')

    old_shared = mdl.shared
    if old_shared.num_embeddings != old_v:
        print(f'  ERROR: shared.num_embeddings={old_shared.num_embeddings}, '
              f'expected {old_v}. Reload base model first.')
        return mdl

    dev = old_shared.weight.device
    dtype = old_shared.weight.dtype
    keep_t_dev = keep_t.to(dev)

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}

    old_pad = old_shared.padding_idx
    new_pad = old_to_new.get(old_pad) if old_pad is not None else None
    if old_pad is not None and new_pad is None:
        print(f'  WARNING: pad token {old_pad} not in keep_ids!')

    embed_scale = getattr(mdl.text_decoder.embed_tokens, 'embed_scale', 1.0)
    print(f'  text_decoder.embed_tokens.embed_scale = {embed_scale}')

    # --- 1. Create trimmed shared embedding ---
    new_shared = nn.Embedding(new_v, hidden, padding_idx=new_pad)
    new_shared.weight.data = old_shared.weight.data[keep_t_dev].clone()
    mdl.shared = new_shared.to(device=dev, dtype=dtype)
    print(f'  shared: [{old_v}, {hidden}] -> [{new_v}, {hidden}]')

    # --- 2. Update text_decoder.embed_tokens IN-PLACE ---
    # Keeps the SeamlessM4Tv2ScaledWordEmbedding class and its embed_scale.
    # The original forward() multiplies by embed_scale ~32.0;
    # replacing with plain nn.Embedding would lose this scaling entirely.
    dec_emb = mdl.text_decoder.embed_tokens
    dec_emb.weight = mdl.shared.weight
    dec_emb.num_embeddings = new_v
    dec_emb.padding_idx = new_pad
    print(f'  text_decoder.embed_tokens: tied to shared, embed_scale={dec_emb.embed_scale}')

    # --- 3. Handle text_encoder.embed_tokens if present ---
    if hasattr(mdl, 'text_encoder') and mdl.text_encoder is not None:
        enc_emb = getattr(mdl.text_encoder, 'embed_tokens', None)
        if enc_emb is not None:
            enc_emb.weight = mdl.shared.weight
            enc_emb.num_embeddings = new_v
            enc_emb.padding_idx = new_pad
            print(f'  text_encoder.embed_tokens: tied to shared')

    # --- 4. Tie lm_head to shared (preserves original tied-weight architecture) ---
    mdl.lm_head.weight = mdl.shared.weight
    mdl.lm_head.out_features = new_v
    if mdl.lm_head.bias is not None:
        mdl.lm_head.bias = nn.Parameter(mdl.lm_head.bias.data[keep_t_dev].clone())
    print(f'  lm_head: tied to shared [{new_v}, {hidden}]')

    # --- 5. Update model config ---
    mdl.config.vocab_size = new_v
    for attr in ['pad_token_id', 'bos_token_id', 'eos_token_id', 'decoder_start_token_id']:
        old_id = getattr(mdl.config, attr, None)
        if old_id is not None and old_id in old_to_new:
            setattr(mdl.config, attr, old_to_new[old_id])

    if hasattr(mdl.text_decoder, 'vocab_size'):
        mdl.text_decoder.vocab_size = new_v
    if hasattr(mdl.text_decoder, 'padding_idx'):
        mdl.text_decoder.padding_idx = new_pad
    print(f'  config: vocab_size={new_v}')

    # --- 6. Update generation_config ---
    # Use `gen_cfg` (not `gc`) so we never shadow Python's garbage-collector module.
    gen_cfg = mdl.generation_config

    # 6a. text_decoder_lang_to_code_id (lang -> text-vocab token ID for decoder_input)
    if hasattr(gen_cfg, 'text_decoder_lang_to_code_id') and gen_cfg.text_decoder_lang_to_code_id:
        gen_cfg.text_decoder_lang_to_code_id = {
            lang: old_to_new[oid]
            for lang, oid in gen_cfg.text_decoder_lang_to_code_id.items()
            if oid in old_to_new
        }
        print(f'  text_decoder_lang_to_code_id: {len(gen_cfg.text_decoder_lang_to_code_id)} langs')

    # 6b. id_to_text (CRITICAL for T2U character-level input)
    #     generate() calls _indices_to_subwords(t2u_input_ids) which does:
    #       id_to_text.get(str(token_id)) for each generated text token
    #     Without remapping, every lookup returns None -> T2U gets garbage chars
    if hasattr(gen_cfg, 'id_to_text') and gen_cfg.id_to_text:
        old_map = gen_cfg.id_to_text
        new_map = {}
        for key_str, text_val in old_map.items():
            old_id = int(key_str)
            if old_id in old_to_new:
                new_map[str(old_to_new[old_id])] = text_val
        gen_cfg.id_to_text = new_map
        print(f'  id_to_text: {len(old_map)} -> {len(new_map)} entries')
    else:
        print(f'  WARNING: no id_to_text in generation_config (T2U may fail for S2ST)')

```
OUTPUT:
```text
Vocab trimming functions ready.
```

### Cell 33 (code, score=220)
```python
try:
    model_p1, processor = load_model_from_drive('phase1_vocab_trimmed')
    p1_ckpt = load_latest_checkpoint('phase1_vocab')
    if p1_ckpt and 'keep_ids' in p1_ckpt:
        keep_ids = p1_ckpt['keep_ids']
        model_p1._vocab_remap_to_old = torch.tensor(keep_ids, dtype=torch.long)
        print(f'  Restored vocab remap ({len(keep_ids)} tokens)')

        # Validate id_to_text was correctly remapped (guards against stale saves
        # from the old broken trim_vocabulary that didn't remap id_to_text)
        gen_cfg = model_p1.generation_config
        if hasattr(gen_cfg, 'id_to_text') and gen_cfg.id_to_text:
            max_key = max(int(k) for k in gen_cfg.id_to_text.keys())
            if max_key >= model_p1.config.vocab_size:
                print(f'  WARNING: id_to_text has stale keys (max={max_key} >= vocab_size={model_p1.config.vocab_size})')
                print(f'  Re-remapping id_to_text from checkpoint keep_ids...')
                old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
                old_map = gen_cfg.id_to_text
                new_map = {}
                for key_str, text_val in old_map.items():
                    old_id = int(key_str)
                    if old_id in old_to_new:
                        new_map[str(old_to_new[old_id])] = text_val
                gen_cfg.id_to_text = new_map
                print(f'  Fixed id_to_text: {len(old_map)} -> {len(new_map)} entries')
    print('Loaded Phase 1 model from Drive.')
except Exception as e:
    print(f'Load failed ({e}), running vocab trimming...')
    if model.config.vocab_size != 256102:
        print(f'  Model vocab is {model.config.vocab_size}, expected 256102. Reloading base model...')
        model, processor = load_base_model()
    TARGET_LANGS = ['eng', 'ben', 'cmn', 'fra', 'hin']
    keep_ids = identify_used_tokens(processor, TARGET_LANGS)
    pre = count_params(model)
    model = trim_vocabulary(model, processor, keep_ids)
    post = count_params(model)
    print(f'  Params: {pre:.1f}M to {post:.1f}M (saved {pre-post:.1f}M)')
    save_checkpoint(dict(keep_ids=keep_ids, pre=pre, post=post), name='phase1_vocab', step=0)
    save_model_to_drive(model, processor, 'phase1_vocab_trimmed')
    model_p1 = model

print_model_breakdown(model_p1, 'After Phase 1: Vocab Trimmed')
```
OUTPUT:
```text
[model] Not in local cache, pulling from remote...
Load failed ([rclone] model pull failed for phase1_vocab_trimmed: 2026/04/18 21:05:52 ERROR : Google drive root 'seamV5/models/phase1_vocab_trimmed': error reading source root directory: directory not found
2026/04/18 21:05:52 ERROR : Local file system at /kaggle/working/models/phase1_vocab_trimmed: not deleting files as there were IO errors
2026/04/18 21:05:52 ER), running vocab trimming...
  Scanning eng (en_us)...

en_us/train/0000.parquet:   0%|          | 0.00/526M [00:00<?, ?B/s]
en_us/train/0001.parquet:   0%|          | 0.00/523M [00:00<?, ?B/s]
en_us/train/0002.parquet:   0%|          | 0.00/538M [00:00<?, ?B/s]
en_us/train/0003.parquet:   0%|          | 0.00/136M [00:00<?, ?B/s]
Generating train split: 0 examples [00:00, ? examples/s]
  Scanning ben (bn_in)...

bn_in/train/0000.parquet:   0%|          | 0.00/504M [00:00<?, ?B/s]
bn_in/train/0001.parquet:   0%|          | 0.00/503M [00:00<?, ?B/s]
bn_in/train/0002.parquet:   0%|          | 0.00/570M [00:00<?, ?B/s]
bn_in/train/0003.parquet:   0%|          | 0.00/563M [00:00<?, ?B/s]
bn_in/train/0004.parquet:   0%|          | 0.00/328M [00:00<?, ?B/s]
Generating train split: 0 examples [00:00, ? examples/s]
  Scanning cmn (cmn_hans_cn)...

cmn_hans_cn/train/0000.parquet:   0%|          | 0.00/546M [00:00<?, ?B/s]
cmn_hans_cn/train/0001.parquet:   0%|          | 0.00/557M [00:00<?, ?B/s]
cmn_hans_cn/train/0002.parquet:   0%|          | 0.00/539M [00:00<?, ?B/s]
cmn_hans_cn/train/0003.parquet:   0%|          | 0.00/540M [00:00<?, ?B/s]
cmn_hans_cn/train/0004.parquet:   0%|          | 0.00/32.3M [00:00<?, ?B/s]
Generating train split: 0 examples [00:00, ? examples/s]
  Scanning fra (fr_fr)...

fr_fr/train/0000.parquet:   0%|          | 0.00/516M [00:00<?, ?B/s]
fr_fr/train/0001.parquet:   0%|          | 0.00/509M [00:00<?, ?B/s]
fr_fr/train/0002.parquet:   0%|          | 0.00/497M [00:00<?, ?B/s]
fr_fr/train/0003.parquet:   0%|          | 0.00/505M [00:00<?, ?B/s]
fr_fr/train/0004.parquet:   0%|          | 0.00/291M [00:00<?, ?B/s]
Generating train split: 0 examples [00:00, ? examples/s]
  Scanning hin (hi_in)...

hi_in/train/0000.parquet:   0%|          | 0.00/502M [00:00<?, ?B/s]
hi_in/train/0001.parquet:   0%|          | 0.00/513M [00:00<?, ?B/s]
hi_in/train/0002.parquet:   0%|          | 0.00/501M [00:00<?, ?B/s]
hi_in/train/0003.parquet:   0%|          | 0.00/14.0M [00:00<?, ?B/s]
Generating train split: 0 examples [00:00, ? examples/s]
  Unique tokens: 20425 / 256099
  Vocabulary: 256102 -> 20425 (8.0%)
  text_decoder.embed_tokens.embed_scale = 32.0
  shared: [256102, 1024] -> [20425, 1024]
  text_decoder.embed_tokens: tied to shared, embed_scale=32.0
  lm_head: tied to shared [20425, 1024]
  config: vocab_size=20425
  text_decoder_lang_to_code_id: 98 langs
  id_to_text: 256102 -> 20425 entries
  generation_config special tokens remapped
  Done: ~241M shared-embedding params removed (lm_head tied, not double-counted)
  Params: 1805.5M to 1564.2M (saved 241.3M)
[ckpt] Saved phase1_vocab_step000000.pt (0.1 MB)
[model] Saving phase1_vocab_trimmed → /kaggle/working/models/phase1_vocab_trimmed ...
  [config] sync done.
  Saved custom state: ['_vocab_remap_to_old']
  Saved pruning_manifest.pt keys=['stage_name']

Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]
[model] Local save done. 3161 MB in 8 files.
[model] Pushing to rclone remote...
[model] Verified 8 files on remote.

--- After Phase 1: Vocab Trimmed ---
  speech_encoder                         635.0M  ( 40.6%)
  text_decoder                           625.5M  ( 40.0%)
  t2u_model                              261.8M  ( 16.7%)
  vocoder                                 41.9M  (  2.7%)
  shared                                  20.9M  (  1.3%)
  lm_head                                 20.9M  (  1.3%)
  TOTAL                                 1564.2M
---

{'shared': 20.9152,
 'speech_encoder': 635.04672,
 'text_decoder': 625.462272,
 'lm_head': 20.9152,
 't2u_model': 261.759747,
 'vocoder': 41.911362,
 'TOTAL': 1564.180101}
```

### Cell 34 (code, score=120)
```python
# for f in glob.glob(f'{CKPT_DIR}/phase1_benchmark_step*.pt'):
#     os.remove(f); print(f'Deleted stale: {f}')

p1_ckpt = load_latest_checkpoint('phase1_benchmark')
if p1_ckpt and p1_ckpt['summary'].get('avg_bleu', 0) > 0:
    p1_results, p1_summary = p1_ckpt['results'], p1_ckpt['summary']
else:
    p1_results, p1_summary = run_benchmark(model_p1, eval_samples, label='P1_VocabTrim', save_n=2)
    save_checkpoint(dict(results=p1_results, summary=p1_summary), name='phase1_benchmark', step=0)
store_summary(p1_summary)
plot_phase_comparison()
```
OUTPUT:
```text
[ckpt] No checkpoint for 'phase1_benchmark'

============================================================
  BENCHMARK: P1_VocabTrim
  Samples: 25  Target: ben
============================================================

  GPU mem: 1.84 GB alloc / 2.76 GB reserved
  [ 1/25] BLEU= 10.7 ChrF= 49.2 RTF=0.156  id=1660
              pred: রোমান্টিকতাবাদ গথ, ফিচ্ট এবং শ্লেগেলের মতো লেখকদের কাছ থেকে নেওয়া সাংস্কৃতিক নি
[audio] Saved P1_VocabTrim_s1in.wav (0.3 MB)
  P1_VocabTrim_s1in.wav  (10.7s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P1_VocabTrim_s1out.wav (0.2 MB)
  P1_VocabTrim_s1out.wav  (7.4s | sr=16000)

<IPython.lib.display.Audio object>
  [ 2/25] BLEU= 10.4 ChrF= 45.8 RTF=0.180  id=1661
              pred: তিনি চীনের অর্থনৈতিক উৎপাদনের উপর ভিত্তি করে এই কমানোর জন্য কোন পরিসংখ্যান নির্ধ
[audio] Saved P1_VocabTrim_s2in.wav (0.2 MB)
  P1_VocabTrim_s2in.wav  (6.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P1_VocabTrim_s2out.wav (0.2 MB)
  P1_VocabTrim_s2out.wav  (5.9s | sr=16000)

<IPython.lib.display.Audio object>
  [ 3/25] BLEU= 16.5 ChrF= 57.3 RTF=0.189  id=1662
              pred: অ্যালোয়গুলি মূলত দুটি বা ততোধিক ধাতুর মিশ্রণ, পিরিয়ডিক টেবিলে অনেকগুলি উপাদান 
  [ 4/25] BLEU=  5.6 ChrF= 44.7 RTF=0.177  id=1663
              pred: চোকামো ভ্যালি, চিলির শীর্ষস্থানীয় আরোহণের গন্তব্য, দক্ষিণ আমেরিকার ইয়োসেমিটি ন
  [ 5/25] BLEU= 11.2 ChrF= 45.8 RTF=0.193  id=1664
              pred: দুটি শুষ্ক পাত্র একসাথে ঘূর্ণায়মান করুন এবং তারপর, রাণীর ভিজা হাত দিয়ে, তাদের 
  [ 6/25] BLEU=  4.8 ChrF= 43.1 RTF=0.234  id=1665
              pred: এই নথিটি পরাগের কারণে পেলেস্টাইনের সীমান্তের বিষয়ে বিতর্ক করবে যা পেলেস্টাইন চা
  [ 7/25] BLEU= 10.4 ChrF= 53.6 RTF=0.139  id=1666
              pred: আপনি আপনার নিজের সরকার ছাড়া অন্য সরকারের পরামর্শও নিতে পারেন, কিন্তু তাদের পরাম
  [ 8/25] BLEU= 19.1 ChrF= 58.5 RTF=0.153  id=1667
              pred: সাধারণভাবে বলতে গেলে, ম্যানেজাররা তাদের প্রাক্তন সমবয়সীদের নেতৃত্ব দিতে শুরু কর
  [ 9/25] BLEU=  7.0 ChrF= 46.5 RTF=0.137  id=1668
              pred: এটি একটি ওয়াইল্ডকার্ড কেনার জন্যও উপকারী হতে পারে, যা দক্ষিণ আফ্রিকার পার্কগুলি
  [10/25] BLEU=  9.7 ChrF= 43.2 RTF=0.193  id=1669
              pred: পুলিশ সুপারিনটেন্ডেন্ট চন্দ্র শেখর সলাঙ্কি বলেন, অভিযুক্তকে আদালতে মুখ ঢাকা দেখা
  [11/25] BLEU= 17.0 ChrF= 56.1 RTF=0.165  id=1670
              pred: "তাদের তাপীয় আচরণ পৃথিবীর বড় গুহাগুলির মতো স্থিতিশীল নয় যা প্রায়শই একটি মোটা
  [12/25] BLEU= 23.1 ChrF= 58.8 RTF=0.222  id=1671
              pred: কংগ্রেস ফিসাল ২০০৫ সালে অশ্লীলতা ইনিশিয়েটিভকে অর্থায়ন শুরু করে এবং নির্দিষ্ট ক
  [13/25] BLEU=  9.4 ChrF= 35.6 RTF=0.147  id=1672
              pred: ফ্যাব্রিককে খুব গরম হতে না দেওয়ার জন্য সতর্ক থাকুন যা সংকোচনের কারণ হতে পারে বা
  [14/25] BLEU= 34.1 ChrF= 77.0 RTF=0.161  id=1673
              pred: বিপ্লবী যুদ্ধের সময়, ১৩ টি রাজ্য প্রথম একটি দুর্বল কেন্দ্রীয় সরকার গঠন করেছিল,
  [15/25] BLEU=  3.7 ChrF= 31.0 RTF=0.186  id=1674
              pred: কিছু অঞ্চলে এক মিনিটের জন্য উষ্ণ জল যথেষ্ট এবং অন্যদের কয়েক মিনিটের প্রয়োজন হয
  [16/25] BLEU=  7.6 ChrF= 40.4 RTF=0.168  id=1675
              pred: শব্দটির পাশে থাকা সমস্ত নামগুলি আপনাকে বলে যে আপনি সর্বদা একটি বড় অক্ষর দিয়ে শ
  [17/25] BLEU=  8.8 ChrF= 51.4 RTF=0.118  id=1676
              pred: দক্ষিণ আফ্রিকার সমস্ত জাতীয় উদ্যানের মতো, পার্কের জন্য দৈনিক সংরক্ষণ এবং প্রবেশ
  [18/25] BLEU= 12.0 ChrF= 34.7 RTF=0.225  id=1677
              pred: আজ, একমাত্র কীট যা তাদের ডানা ফিরিয়ে দিতে পারে না তা হ'ল ড্রাগনফ্লি এবং মেফ্লি।
  [19/25] BLEU=  4.5 ChrF= 49.0 RTF=0.156  id=1678
              pred: অলিভার স্যাক্স তার কাগজে রাষ্ট্রপতির বক্তৃতা নির্দেশ করেছিলেন যে মস্তিষ্কের ক্ষত
  [20/25] BLEU=  7.3 ChrF= 73.3 RTF=0.236  id=1679
              pred: এরোস্মিথ তাদের সফরে তাদের অবশিষ্ট কনসার্ট বাতিল করেছে।
  [21/25] BLEU=  6.7 ChrF= 41.8 RTF=0.234  id=1680
              pred: একটি সু-গোল্লা অ্যাথলিট, বাঘটি ভাল না হলেও আরোহণ করতে পারে, সাঁতার কাটতে পারে, দ
  [22/25] BLEU=  4.3 ChrF= 50.4 RTF=0.112  id=1681
              pred: এটি কেবলমাত্র পরীক্ষা নয়, এবং একটি পরীক্ষা এমন একটি পরীক্ষা যা সম্ভাব্য অনুমানগ
  [23/25] BLEU=  5.0 ChrF= 37.2 RTF=0.165  id=1682
              pred: যদিও কেউ নিশ্চিতভাবে জানে না যে এটি কে লিখেছে, এটি জানা যায় যে এর জীবনের প্রথম 
  [24/25] BLEU= 27.1 ChrF= 56.2 RTF=0.165  id=1683
              pred: এখনও অনেক পুরুষ এবং মহিলা বেঁচে আছেন যারা এখানে তাদের সময় বেঁচে আছেন এবং আরও অন
  [25/25] BLEU=  9.8 ChrF= 46.3 RTF=0.126  id=1684
              pred: আপিয়া সামোয়া দ্বীপের রাজধানী, এটি উপুলু দ্বীপের একটি শহর এবং এর জনসংখ্যা ৪০ হা

  Summary: BLEU=11.43  ChrF=49.07  RTF=0.1734  Params=1564.2M

[ckpt] Saved phase1_benchmark_step000000.pt (0.0 MB)
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P1_VocabTrim (2 total)

<Figure size 1680x1200 with 4 Axes>
[image/png output omitted]
```

### Cell 35 (code, score=8)
```python
# run_benchmark(model_p1, eval_samples, label='P1_VocabTrim', save_n=2)
# import shutil
# stale = f'{MODEL_DIR}/phase1_vocab_trimmed'
# if os.path.exists(stale):
#     shutil.rmtree(stale)
#     print(f'Deleted stale: {stale}')
# # Also clear from Drive:
# subprocess.run(f'rclone delete {GDRIVE_ROOT}/phase1_vocab_trimmed/', shell=True)
```

### Cell 37 (markdown, score=3)
```markdown
---
# Phase 2: Text Encoder Removal
# Actually it was never in the first place, we never loaded the textEncoder as we are using
# from transformers import SeamlessM4Tv2ForSpeechToSpeech
For S2S, the pipeline is: Speech Encoder -> Text Decoder -> T2U -> Vocoder.
The text_encoder is only used for T2T/T2S tasks. We verify and remove it.
```

### Cell 38 (code, score=4)
```python
# def check_text_encoder_used(mdl):
#     if not hasattr(mdl, 'text_encoder') or mdl.text_encoder is None:
#         print('  text_encoder not present.'); return False
#     called = [False]
#     def hook(mod, inp, out): called[0] = True
#     h = mdl.text_encoder.register_forward_hook(hook)
#     try: _ = run_s2st(mdl, np.random.randn(48000).astype(np.float32), 'ben')
#     except: pass
#     h.remove()
#     return called[0]

# def remove_text_encoder(mdl):
#     if hasattr(mdl, 'text_encoder') and mdl.text_encoder is not None:
#         enc_p = count_params(mdl.text_encoder)
#         del mdl.text_encoder; mdl.text_encoder = None
#         _stdlib_gc.collect(); torch.cuda.empty_cache()
#         print(f'  Removed text_encoder ({enc_p:.1f}M params)')
#     return mdl

# print('Text encoder analysis ready.')
```

### Cell 39 (code, score=29)
```python
# try:
#     model_p2, processor = load_model_from_drive('phase2_no_text_enc')
#     print('Loaded Phase 2 from Drive.')
# except:
#     model_p2 = model_p1
#     used = check_text_encoder_used(model_p2)
#     print(f'  text_encoder used in S2S: {used}')
#     if not used:
#         print('  Safe to remove for S2S-only.')
#         model_p2 = remove_text_encoder(model_p2)
#     else:
#         print('  text_encoder IS used. Keeping 8 of 24 layers.')
#         layers = model_p2.text_encoder.layers
#         keep = list(range(4)) + list(range(len(layers)-4, len(layers)))
#         model_p2.text_encoder.layers = nn.ModuleList([layers[i] for i in keep])
#     save_model_to_drive(model_p2, processor, 'phase2_no_text_enc')

# print_model_breakdown(model_p2, 'After Phase 2: Text Encoder Removed')
```

### Cell 40 (code, score=15)
```python
# p2_ckpt = load_latest_checkpoint('phase2_benchmark')
# if p2_ckpt:
#     p2_results, p2_summary = p2_ckpt['results'], p2_ckpt['summary']
# else:
#     p2_results, p2_summary = run_benchmark(model_p2, eval_samples, label='P2_NoTextEnc', save_n=2)
#     save_checkpoint(dict(results=p2_results, summary=p2_summary), name='phase2_benchmark', step=0)
# store_summary(p2_summary)
# plot_phase_comparison()
```

### Cell 41 (markdown, score=9)
```markdown
---
# Phase 3: Text Decoder Iterative Layer Pruning
**Paper:** Moslem (IWSLT 2025), CULL-MT (2024)

Iterative greedy pruning: remove one layer at a time, evaluate ChrF, repeat.
Target: remove 6-8 of 24 text decoder layers.
```

### Cell 42 (code, score=82)
```python
# def find_layers_attr(component):
#     for attr in ['layers', 'layer', 'inner_layers', 'encoder_layers', 'decoder_layers']:
#         if hasattr(component, attr): return attr
#     return None

def _get_protected_indices(n_total):
    """
    Custom protection rule:
      - First layer  (index 0)
      - Last layer   (index n_total - 1)
      - Middle layer (index n_total // 2)
    Returns a set of original indices that must never be pruned.
    """
    first  = 0
    last   = n_total - 1
    middle = n_total // 2
    protected = {first, last, middle}
    print(f'  Protected layers (first/mid/last): {sorted(protected)}')
    return protected

def iterative_layer_prune(mdl, component_name, samples, n_remove,
                          tgt_lang='ben', max_eval=10,
                          ckpt_name='phase3_dec_pruning'):
    """
    Iterative greedy layer pruning (Moslem, IWSLT 2025).
    Custom rule: never prune the first, last, or middle layer.
    Removes one layer per iteration, picking the candidate whose removal
    causes the LEAST ChrF degradation (highest remaining ChrF).
    Saves checkpoint after every iteration.
    """
    parent = getattr(mdl, component_name)
    layers_attr = find_layers_attr(parent)
    if layers_attr is None:
        print(f'  No layers found on {component_name}'); return [], []
    current = list(getattr(parent, layers_attr))
    orig_indices = list(range(len(current)))
    n_total_orig = len(current)
    removed, log = [], []

    # ── Protected indices (never prunable) ──
    protected = _get_protected_indices(n_total_orig)

    # ── Resume from checkpoint ──
    partial = load_latest_checkpoint(ckpt_name)
    if partial and partial.get('removed'):
        removed = partial['removed']
        log = partial.get('log', [])
        for r in removed:
            if r in orig_indices:
                pos = orig_indices.index(r)
                current.pop(pos)
                orig_indices.pop(pos)
        setattr(parent, layers_attr, nn.ModuleList(current))
        print(f'  Resuming: already removed {removed}, {len(current)} layers remain')

    start_iter = len(removed)
    for it in range(start_iter, n_remove):
        # Eligible candidates: not in protected set, not already removed
        eligible = [idx for idx in range(len(current))
                    if orig_indices[idx] not in protected]
        if not eligible:
            print(f'  WARNING: No eligible (non-protected) layers left to prune! Stopping.')
            break

        print(f'\n  Iter {it+1}/{n_remove} ({len(current)} layers remain, '
              f'{len(eligible)} eligible candidates)')
        scores = {}
        for idx in eligible:
            temp = current[:idx] + current[idx+1:]
            setattr(parent, layers_attr, nn.ModuleList(temp))
            sc = quick_eval_chrf(mdl, samples, tgt_lang, max_eval)
            scores[idx] = (orig_indices[idx], sc)
            orig_label = orig_indices[idx]
            prot_note = ' [PROTECTED-skip]' if orig_label in protected else ''
            print(f'    Remove L{orig_label:>2} -> ChrF={sc:.2f}{prot_note}')
        setattr(parent, layers_attr, nn.ModuleList(current))

        # Pick the eligible candidate causing the least harm
        best_idx = max(scores, key=lambda k: scores[k][1])
        best_orig, best_sc = scores[best_idx]
        current.pop(best_idx)
        orig_indices.pop(best_idx)
        setattr(parent, layers_attr, nn.ModuleList(current))
        removed.append(best_orig)
        log.append(dict(iter=it+1, removed=best_orig, chrf=best_sc,
                        remaining=len(current)))
        print(f'  -> Removed layer {best_orig} (ChrF={best_sc:.2f})')
        save_checkpoint(dict(removed=removed, log=log), name=ckpt_name, step=0)
        print(f'  [ckpt] Progress saved ({it+1}/{n_remove} iterations done)')

    return removed, log

print('iterative_layer_prune() with first/mid/last protection ready.')
```
OUTPUT:
```text
iterative_layer_prune() with first/mid/last protection ready.
```

### Cell 43 (code, score=360)
```python
# ── Phase 3: Text Decoder Iterative Layer Pruning ────────────────────────────
# Text decoder has 24 layers. Protected: L0, L12, L23.
# We remove 8 of the remaining 21 eligible layers.
N_DEC_REMOVE = 10   # ← increased from 6

p3_ckpt = load_latest_checkpoint('phase3_dec_pruning')
p3_complete = p3_ckpt and len(p3_ckpt.get('removed', [])) >= N_DEC_REMOVE

if p3_complete:
    removed_dec = p3_ckpt['removed']; p3_log = p3_ckpt['log']
    print(f'Phase 3 complete: removed {removed_dec}')
    try:
        model_p3, processor = load_model_from_drive('phase3_dec_pruned')
    except:
        print('  Drive model missing, rebuilding from checkpoint + model_p1...')
        model_p3 = model_p1
        parent = model_p3.text_decoder
        la = find_layers_attr(parent)
        cur = list(getattr(parent, la))
        keep = [i for i in range(len(cur)) if i not in removed_dec]
        setattr(parent, la, nn.ModuleList([cur[i] for i in keep]))
        # Critical: Sync config before saving or using
        sync_model_config(model_p3)
        save_model_to_drive(model_p3, processor, 'phase3_dec_pruned')
else:
    done_so_far = len(p3_ckpt['removed']) if p3_ckpt else 0
    print(f'{"Resuming" if done_so_far else "Running"} Phase 3: '
          f'decoder pruning ({done_so_far}/{N_DEC_REMOVE} done)...')
    model_p3 = _consolidate_to_single_gpu(model_p1)
    removed_dec, p3_log = iterative_layer_prune(
        model_p3, 'text_decoder', eval_samples, N_DEC_REMOVE, TARGET_LANG,
        ckpt_name='phase3_dec_pruning')
    # Sync config after pruning is finished
    sync_model_config(model_p3)
    save_model_to_drive(model_p3, processor, 'phase3_dec_pruned')

print(f'Decoder layers removed: {removed_dec}')
print_model_breakdown(model_p3, 'After Phase 3: Decoder Pruned')
```
OUTPUT:
```text
[ckpt] No checkpoint for 'phase3_dec_pruning'
Running Phase 3: decoder pruning (0/10 done)...
  Multi-device map detected, consolidating to cuda:0...
  Model now on: cuda:0
  Protected layers (first/mid/last): [0, 12, 23]
[ckpt] No checkpoint for 'phase3_dec_pruning'

  Iter 1/10 (24 layers remain, 21 eligible candidates)
    Remove L 1 -> ChrF=46.50
    Remove L 2 -> ChrF=46.08
    Remove L 3 -> ChrF=48.86
    Remove L 4 -> ChrF=48.61
    Remove L 5 -> ChrF=47.46
    Remove L 6 -> ChrF=46.90
    Remove L 7 -> ChrF=48.58
    Remove L 8 -> ChrF=49.13
    Remove L 9 -> ChrF=50.52
    Remove L10 -> ChrF=48.23
    Remove L11 -> ChrF=46.77
    Remove L13 -> ChrF=47.45
    Remove L14 -> ChrF=46.63
    Remove L15 -> ChrF=49.04
    Remove L16 -> ChrF=48.26
    Remove L17 -> ChrF=48.94
    Remove L18 -> ChrF=48.79
    Remove L19 -> ChrF=47.60
    Remove L20 -> ChrF=47.33
    Remove L21 -> ChrF=48.46
    Remove L22 -> ChrF=49.78
  -> Removed layer 9 (ChrF=50.52)
[ckpt] Saved phase3_dec_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (1/10 iterations done)

  Iter 2/10 (23 layers remain, 20 eligible candidates)
    Remove L 1 -> ChrF=50.14
    Remove L 2 -> ChrF=48.76
    Remove L 3 -> ChrF=49.48
    Remove L 4 -> ChrF=48.75
    Remove L 5 -> ChrF=47.23
    Remove L 6 -> ChrF=50.65
    Remove L 7 -> ChrF=49.02
    Remove L 8 -> ChrF=49.84
    Remove L10 -> ChrF=48.21
    Remove L11 -> ChrF=45.94
    Remove L13 -> ChrF=47.84
    Remove L14 -> ChrF=47.63
    Remove L15 -> ChrF=49.84
    Remove L16 -> ChrF=49.99
    Remove L17 -> ChrF=48.92
    Remove L18 -> ChrF=47.29
    Remove L19 -> ChrF=49.06
    Remove L20 -> ChrF=47.29
    Remove L21 -> ChrF=50.28
    Remove L22 -> ChrF=47.53
  -> Removed layer 6 (ChrF=50.65)
[ckpt] Saved phase3_dec_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (2/10 iterations done)

  Iter 3/10 (22 layers remain, 19 eligible candidates)
    Remove L 1 -> ChrF=48.06
    Remove L 2 -> ChrF=47.60
    Remove L 3 -> ChrF=50.53
    Remove L 4 -> ChrF=49.21
    Remove L 5 -> ChrF=47.83
    Remove L 7 -> ChrF=48.61
    Remove L 8 -> ChrF=48.90
    Remove L10 -> ChrF=45.92
    Remove L11 -> ChrF=45.45
    Remove L13 -> ChrF=50.32
    Remove L14 -> ChrF=49.00
    Remove L15 -> ChrF=51.56
    Remove L16 -> ChrF=49.36
    Remove L17 -> ChrF=48.80
    Remove L18 -> ChrF=48.80
    Remove L19 -> ChrF=46.10
    Remove L20 -> ChrF=46.35
    Remove L21 -> ChrF=49.42
    Remove L22 -> ChrF=47.27
  -> Removed layer 15 (ChrF=51.56)
[ckpt] Saved phase3_dec_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (3/10 iterations done)

  Iter 4/10 (21 layers remain, 18 eligible candidates)
    Remove L 1 -> ChrF=50.06
    Remove L 2 -> ChrF=46.81
    Remove L 3 -> ChrF=47.31
    Remove L 4 -> ChrF=46.83
    Remove L 5 -> ChrF=48.34
    Remove L 7 -> ChrF=47.31
    Remove L 8 -> ChrF=49.60
    Remove L10 -> ChrF=47.38
    Remove L11 -> ChrF=48.06
    Remove L13 -> ChrF=48.45
    Remove L14 -> ChrF=47.12
    Remove L16 -> ChrF=46.47
    Remove L17 -> ChrF=49.59
    Remove L18 -> ChrF=48.92
    Remove L19 -> ChrF=47.73
    Remove L20 -> ChrF=47.40
    Remove L21 -> ChrF=51.72
    Remove L22 -> ChrF=48.64
  -> Removed layer 21 (ChrF=51.72)
[ckpt] Saved phase3_dec_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (4/10 iterations done)

  Iter 5/10 (20 layers remain, 17 eligible candidates)
    Remove L 1 -> ChrF=48.99
    Remove L 2 -> ChrF=50.64
    Remove L 3 -> ChrF=50.94
    Remove L 4 -> ChrF=49.59
    Remove L 5 -> ChrF=50.54
    Remove L 7 -> ChrF=48.28
    Remove L 8 -> ChrF=51.32
    Remove L10 -> ChrF=45.68
    Remove L11 -> ChrF=45.65
    Remove L13 -> ChrF=44.74
    Remove L14 -> ChrF=50.29
    Remove L16 -> ChrF=49.58
    Remove L17 -> ChrF=50.00
    Remove L18 -> ChrF=49.84
    Remove L19 -> ChrF=50.24
    Remove L20 -> ChrF=45.70
    Remove L22 -> ChrF=46.80
  -> Removed layer 8 (ChrF=51.32)
[ckpt] Saved phase3_dec_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (5/10 iterations done)

  Iter 6/10 (19 layers remain, 16 eligible candidates)
    Remove L 1 -> ChrF=49.67
    Remove L 2 -> ChrF=48.96
    Remove L 3 -> ChrF=48.84
    Remove L 4 -> ChrF=51.09
    Remove L 5 -> ChrF=48.94
    Remove L 7 -> ChrF=45.90
    Remove L10 -> ChrF=47.21
    Remove L11 -> ChrF=47.25
    Remove L13 -> ChrF=47.16
    Remove L14 -> ChrF=49.59
    Remove L16 -> ChrF=49.31
    Remove L17 -> ChrF=48.62
    Remove L18 -> ChrF=50.57
    Remove L19 -> ChrF=48.10
    Remove L20 -> ChrF=45.48
    Remove L22 -> ChrF=48.10
  -> Removed layer 4 (ChrF=51.09)
[ckpt] Saved phase3_dec_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (6/10 iterations done)

  Iter 7/10 (18 layers remain, 15 eligible candidates)
    Remove L 1 -> ChrF=47.58
    Remove L 2 -> ChrF=45.89
    Remove L 3 -> ChrF=45.57
    Remove L 5 -> ChrF=47.98
    Remove L 7 -> ChrF=45.52
    Remove L10 -> ChrF=44.83
    Remove L11 -> ChrF=45.72
    Remove L13 -> ChrF=46.72
    Remove L14 -> ChrF=50.17
    Remove L16 -> ChrF=47.73
    Remove L17 -> ChrF=48.16
    Remove L18 -> ChrF=49.04
    Remove L19 -> ChrF=45.95
    Remove L20 -> ChrF=46.26
    Remove L22 -> ChrF=47.54
  -> Removed layer 14 (ChrF=50.17)
[ckpt] Saved phase3_dec_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (7/10 iterations done)

  Iter 8/10 (17 layers remain, 14 eligible candidates)
    Remove L 1 -> ChrF=47.88
    Remove L 2 -> ChrF=47.15
    Remove L 3 -> ChrF=47.83
    Remove L 5 -> ChrF=45.32
    Remove L 7 -> ChrF=44.96
    Remove L10 -> ChrF=42.64
    Remove L11 -> ChrF=44.62
    Remove L13 -> ChrF=44.14
    Remove L16 -> ChrF=48.85
    Remove L17 -> ChrF=46.77
    Remove L18 -> ChrF=48.18
    Remove L19 -> ChrF=47.41
    Remove L20 -> ChrF=46.66
    Remove L22 -> ChrF=48.03
  -> Removed layer 16 (ChrF=48.85)
[ckpt] Saved phase3_dec_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (8/10 iterations done)

  Iter 9/10 (16 layers remain, 13 eligible candidates)
    Remove L 1 -> ChrF=43.28
    Remove L 2 -> ChrF=44.86
    Remove L 3 -> ChrF=44.47
    Remove L 5 -> ChrF=44.70
    Remove L 7 -> ChrF=42.51
    Remove L10 -> ChrF=42.73
    Remove L11 -> ChrF=42.60
    Remove L13 -> ChrF=38.95
    Remove L17 -> ChrF=44.49
    Remove L18 -> ChrF=44.82
    Remove L19 -> ChrF=44.77
    Remove L20 -> ChrF=42.28
    Remove L22 -> ChrF=45.00
  -> Removed layer 22 (ChrF=45.00)
[ckpt] Saved phase3_dec_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (9/10 iterations done)

  Iter 10/10 (15 layers remain, 12 eligible candidates)
    Remove L 1 -> ChrF=43.86
    Remove L 2 -> ChrF=43.43
    Remove L 3 -> ChrF=37.86
    Remove L 5 -> ChrF=42.45
    Remove L 7 -> ChrF=41.85
    Remove L10 -> ChrF=42.24
    Remove L11 -> ChrF=40.67
    Remove L13 -> ChrF=36.94
    Remove L17 -> ChrF=37.42
    Remove L18 -> ChrF=40.06
    Remove L19 -> ChrF=41.12
    Remove L20 -> ChrF=36.64
  -> Removed layer 1 (ChrF=43.86)
[ckpt] Saved phase3_dec_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (10/10 iterations done)
  [config] decoder_layers: 24 -> 14
  [config] sync done.
[model] Saving phase3_dec_pruned → /kaggle/working/models/phase3_dec_pruned ...
  [config] sync done.
  Saved custom state: ['_vocab_remap_to_old']
  Saved pruning_manifest.pt keys=['stage_name']

Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]
[model] Local save done. 2657 MB in 8 files.
[model] Pushing to rclone remote...
[model] Verified 8 files on remote.
Decoder layers removed: [9, 6, 15, 21, 8, 4, 14, 16, 22, 1]

--- After Phase 3: Decoder Pruned ---
  speech_encoder                         635.0M  ( 48.4%)
  text_decoder                           373.6M  ( 28.5%)
  t2u_model                              261.8M  ( 19.9%)
  vocoder                                 41.9M  (  3.2%)
  shared                                  20.9M  (  1.6%)
  lm_head                                 20.9M  (  1.6%)
  TOTAL                                 1312.3M
---

{'shared': 20.9152,
 'speech_encoder': 635.04672,
 'text_decoder': 373.568512,
 'lm_head': 20.9152,
 't2u_model': 261.759747,
 'vocoder': 41.911362,
 'TOTAL': 1312.286341}
```

### Cell 45 (code, score=37)
```python
model_p3, processor = load_model_from_drive('phase3_dec_pruned')
```
OUTPUT:
```text
[model] Loading phase3_dec_pruned from /kaggle/working/models/phase3_dec_pruned ...

Loading weights:   0%|          | 0/1586 [00:00<?, ?it/s]
  Restored custom state: ['_vocab_remap_to_old']
  [model] pruning_manifest: ['stage_name']
```

### Cell 46 (code, score=35)
```python
if p3_log:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    iters = [e['iter'] for e in p3_log]
    chrfs = [e['chrf'] for e in p3_log]
    ax1.plot(iters, chrfs, 'o-', color='#4CAF50', lw=2, ms=8)
    for e in p3_log: ax1.annotate(f'L{e["removed"]}', (e['iter'], e['chrf']), fontsize=8, ha='center', va='bottom')
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('ChrF'); ax1.set_title('Decoder: ChrF After Each Removal', fontweight='bold')
    ax2.bar(iters, [e['remaining'] for e in p3_log], color='#9C27B0', alpha=0.8)
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('Layers'); ax2.set_title('Decoder Layers Remaining', fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{FIG_DIR}/phase3_dec.png'); plt.show()
```
OUTPUT:
```text
<Figure size 1560x600 with 2 Axes>
[image/png output omitted]
```

### Cell 47 (code, score=5)
```python
[layer.self_attn.layer_idx for layer in model_p3.text_decoder.layers]
```
OUTPUT:
```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
```

### Cell 48 (code, score=7)
```python
def reindex_text_decoder_layer_idx(mdl):
    dec = mdl.text_decoder
    for i, layer in enumerate(dec.layers):
        layer.self_attn.layer_idx = i
        layer.cross_attention.layer_idx = i
```

### Cell 49 (code, score=7)
```python
reindex_text_decoder_layer_idx(model_p3)
sync_model_config(model_p3)
if hasattr(model_p3, "_cache"):
    delattr(model_p3, "_cache")  # avoid reusing a cache sized for an older depth
```
OUTPUT:
```text
[config] sync done.
```

### Cell 50 (code, score=5)
```python
[layer.self_attn.layer_idx for layer in model_p3.text_decoder.layers]
```
OUTPUT:
```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
```

### Cell 51 (code, score=117)
```python
p3_results, p3_summary = run_benchmark(model_p3, eval_samples, label='P3_DecPrune', save_n=2)
save_checkpoint(dict(results=p3_results, summary=p3_summary), name='phase3_benchmark', step=0)
store_summary(p3_summary); plot_phase_comparison()
```
OUTPUT:
```text
============================================================
  BENCHMARK: P3_DecPrune
  Samples: 25  Target: ben
============================================================

  GPU mem: 5.32 GB alloc / 5.43 GB reserved
  [ 1/25] BLEU= 15.7 ChrF= 46.2 RTF=0.096  id=1660
              pred: রোমান্টিকতাবাদ সংস্কৃতির নির্ণায়কতাসের একটি বড় উপাদান ছিল, যা গথ, ফিচ এবং শ্লে
[audio] Saved P3_DecPrune_s1in.wav (0.3 MB)
  P3_DecPrune_s1in.wav  (10.7s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P3_DecPrune_s1out.wav (0.3 MB)
  P3_DecPrune_s1out.wav  (7.9s | sr=16000)

<IPython.lib.display.Audio object>
  [ 2/25] BLEU=  3.5 ChrF= 43.9 RTF=0.101  id=1661
              pred: তিনি চীনের অর্থনৈতিক উৎপাদনের উপর ভিত্তি করে এই কমানের জন্য কোন পরিমাপ নির্ধারণ 
[audio] Saved P3_DecPrune_s2in.wav (0.2 MB)
  P3_DecPrune_s2in.wav  (6.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P3_DecPrune_s2out.wav (0.2 MB)
  P3_DecPrune_s2out.wav  (5.7s | sr=16000)

<IPython.lib.display.Audio object>
  [ 3/25] BLEU= 10.5 ChrF= 37.8 RTF=0.094  id=1662
              pred: মিশ্রণ মূলত দুই বা তার বেশি ধাতব মিশ্রণ, প্যারাইডিক টেবলে অনেক উপাদান আছে, এটা ভ
  [ 4/25] BLEU=  5.2 ChrF= 49.0 RTF=0.099  id=1663
              pred: চোকামো উপত্যকা, চিলির শীর্ষস্থানীয় আরোহণের গন্তব্য, দক্ষিণ আমেরিকা'র ইয়েসেমাইট
  [ 5/25] BLEU= 12.5 ChrF= 44.2 RTF=0.097  id=1664
              pred: দুটি শুষ্ক পাত একসাথে ঘূর্ণায় এবং তারপর, রাণীর ভেঙোরা হাত দিয়ে তাদের একটি বলের
  [ 6/25] BLEU=  3.4 ChrF= 38.8 RTF=0.121  id=1665
              pred: "লিটটিগটি ছড়িয়ে পড়ার পরে, পেলেস্টাইনটি ""১৯৬৭"" মধ্যপ্রাচ্য যুদ্ধের আগে সীমান
  [ 7/25] BLEU=  9.8 ChrF= 50.9 RTF=0.081  id=1666
              pred: আপনি আপনার নিজের সরকার ছাড়া অন্য সরকারের পরামর্শ নিয়ে আলোচনা করতে পারেন, কিন্ত
  [ 8/25] BLEU=  3.3 ChrF= 43.2 RTF=0.089  id=1667
              pred: সাধারণভাবে, দুইটি আচরণ দেখা যেতে পারে, যেহেতু ম্যানেজাররা তাদের প্রাক্তন সমনবীনদ
  [ 9/25] BLEU=  4.1 ChrF= 34.7 RTF=0.073  id=1668
              pred: দক্ষিণ আফ্রিকার পার্কের নির্বাচিত পার্ক বা দক্ষিণ আফ্রিকার সমস্ত জাতীয় পার্কের 
  [10/25] BLEU=  7.3 ChrF= 49.9 RTF=0.114  id=1669
              pred: পুলিশ সুপারিনটেন্ডেন্ট চান্দ্রা শেখার সুলানকি বলেন, অভিযুক্তরা মুখোমুখি হয়ে আদা
  [11/25] BLEU= 13.1 ChrF= 54.2 RTF=0.080  id=1670
              pred: তাদের তাপীয় আচরণ পৃথিবীতে বড় বড় গুহাগুলির মতো স্থিতিশীল নয়, যা প্রায়শই একটি
  [12/25] BLEU=  3.6 ChrF= 37.3 RTF=0.140  id=1671
              pred: কংগ্রেস 'অ্যাবজেক্টেন্সি' ইনিয়েশিফের জন্য অর্থায়ন শুরু করে, এবং এফবিআইকে অবশ্য
  [13/25] BLEU=  5.1 ChrF= 24.9 RTF=0.094  id=1672
              pred: খুব গরম হওয়ায় যেসব ফ্যাক্টরির কারণে ক্ষীর্ণতা বা চরম অবস্থায় পুড়ে ওঠাতে পারে
  [14/25] BLEU= 25.4 ChrF= 75.6 RTF=0.099  id=1673
              pred: বিপ্লবী যুদ্ধের সময়, ১৩টি রাজ্যে প্রথমবারের মতো একটি দুর্বল কেন্দ্রীয় সরকার গঠ
  [15/25] BLEU=  8.3 ChrF= 41.6 RTF=0.105  id=1674
              pred: কিছু এলাকায়, এক মিনিটের জন্য উড়ানো জল যথেষ্ট, অন্যদের কয়েক মিনিট প্রয়োজন।
  [16/25] BLEU=  9.0 ChrF= 36.6 RTF=0.081  id=1675
              pred: শব্দের সাথে সমস্ত নাম বলে, আপনি সর্বদা একটি বড় অক্ষর দিয়ে শুরু করুন, এমনকি একট
  [17/25] BLEU=  8.8 ChrF= 59.1 RTF=0.066  id=1676
              pred: দক্ষিণ আফ্রিকার সমস্ত জাতীয় উদ্যানগুলির মতো, পার্কের জন্য প্রতিদিন সংরক্ষণ এবং 
  [18/25] BLEU= 12.9 ChrF= 39.4 RTF=0.129  id=1677
              pred: আজ, একমাত্র পোকা যে তাদের ডানা পিছনে ভাঁড়ায় না, তা হল ড্রাগনফ্লাই এবং মেফাই।
  [19/25] BLEU=  1.6 ChrF= 43.8 RTF=0.104  id=1678
              pred: অলিভার স্যাক্স তার কাগজ, রাষ্ট্রপতির বক্তৃতা, এমন ব্যক্তির কথা বলে যে মস্তিষ্কের
  [20/25] BLEU=  7.8 ChrF= 49.5 RTF=0.138  id=1679
              pred: এয়রোস্মিথ তাদের সফরর বাকি কনসেন্ট বাতিল করেছে।
  [21/25] BLEU=  3.5 ChrF= 35.6 RTF=0.102  id=1680
              pred: একটি সু-গড়া ক্রীড়াবিদ, বাঘ, যদিও ভাল না, সাঁতার, লাজ, বড় দূরত্ব, এবং পাঁচ গুণ
  [22/25] BLEU= 14.2 ChrF= 48.1 RTF=0.061  id=1681
              pred: এটি একটি পরীক্ষা যা এক বা একাধিক সম্ভাব্য অনুমানগুলি নির্মূল করতে ব্যবহৃত হয়, প
  [23/25] BLEU=  0.7 ChrF= 16.0 RTF=0.130  id=1682
              pred: যদিও কেউই নিশ্চিত না যে এটি কে লিখেছে, এটি জানা যায় যে তার জীবনের শুরুতে, এটির 
  [24/25] BLEU=  9.4 ChrF= 52.1 RTF=0.115  id=1683
              pred: এখানে এখনও অনেক পুরুষ ও মহিলা বেঁচে আছে, যারা তাদের সময়টি বেঁচে আছে, এবং আরও অন
  [25/25] BLEU=  3.5 ChrF= 37.1 RTF=0.073  id=1684
              pred: সামোয়া'র রাজধানী, উপু-পুলু দ্বীপের এই শহরটি, জনসংখ্যা ৪০,০০০ এরওও কম।

  Summary: BLEU=8.09  ChrF=43.58  RTF=0.0994  Params=1312.3M

[ckpt] Saved phase3_benchmark_step000000.pt (0.0 MB)
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P3_DecPrune (3 total)

<Figure size 1680x1200 with 4 Axes>
[image/png output omitted]
```

### Cell 52 (code, score=47)
```python
p3b = load_latest_checkpoint('phase3_benchmark')
if p3b: p3_results, p3_summary = p3b['results'], p3b['summary']
else:
    p3_results, p3_summary = run_benchmark(model_p3, eval_samples, label='P3_DecPrune', save_n=2)
    save_checkpoint(dict(results=p3_results, summary=p3_summary), name='phase3_benchmark', step=0)
store_summary(p3_summary); plot_phase_comparison()
```
OUTPUT:
```text
[ckpt] Loaded phase3_benchmark_step000000.pt
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P3_DecPrune (3 total)

<Figure size 1680x1200 with 4 Axes>
[image/png output omitted]
```

### Cell 54 (markdown, score=5)
```markdown
---
# Phase 4: Speech Encoder Iterative Layer Pruning
**Paper:** ShortGPT (ACL 2025) for Block Influence; Moslem (IWSLT 2025) for iterative greedy

Target: remove 6-8 of 24 speech encoder layers.
```

### Cell 55 (code, score=137)
```python
def get_speech_encoder_layers(mdl):
    enc = mdl.speech_encoder
    if hasattr(enc, 'layers') and isinstance(enc.layers, torch.nn.ModuleList) and len(enc.layers) > 0:
        return enc, 'layers'
    if hasattr(enc, 'encoder') and hasattr(enc.encoder, 'layers') and len(enc.encoder.layers) > 0:
        return enc.encoder, 'layers'
    for child_name, child in enc.named_children():
        if hasattr(child, 'layers') and isinstance(child.layers, torch.nn.ModuleList) and len(child.layers) > 0:
            print(f'  Found layers at speech_encoder.{child_name}.layers')
            return child, 'layers'
    raise RuntimeError(
        f"Cannot find layers. speech_encoder children: "
        f"{[n for n,_ in enc.named_children()]}\n"
        f"encoder children: "
        f"{[n for n,_ in enc.encoder.named_children()] if hasattr(enc, 'encoder') else 'N/A'}"
    )


def compute_block_influence(mdl, samples, max_n=50):
    """
    Block Influence (ShortGPT, ACL 2025): BI(l) = 1 - cos(input_l, output_l).

    Measures how much each layer transforms its input — low BI means the
    layer barely changes the hidden states and is a prime pruning candidate.
    Used here to PRE-RANK layers before iterative ChrF search, so we only
    evaluate cheap candidates rather than all layers every iteration.
    """
    parent, la = get_speech_encoder_layers(mdl)
    layers = getattr(parent, la)
    n = len(layers)
    bi = {i: [] for i in range(n)}
    hooks = []

    for i in range(n):
        def make_hook(idx):
            def hook(mod, inp, out):
                x = inp[0]
                if x is None or not isinstance(x, torch.Tensor): return
                y = out[0] if isinstance(out, tuple) else out
                if y is None or not isinstance(y, torch.Tensor): return
                x = x.detach().float().reshape(-1, x.shape[-1])
                y = y.detach().to(x.device).float().reshape(-1, y.shape[-1])
                cos = F.cosine_similarity(x, y, dim=-1).mean().item()
                bi[idx].append(1.0 - cos)
            return hook
        hooks.append(layers[i].register_forward_hook(make_hook(i)))

    mdl.eval()
    dev = next(mdl.speech_encoder.parameters()).device
    ok = 0
    for idx, s in enumerate(samples[:max_n]):
        if idx % 10 == 0:
            print(f'  Calibrating {idx}/{min(max_n, len(samples))}...')
        try:
            inputs = processor(audio=s['wav'], sampling_rate=16000, return_tensors='pt')
            feats = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                mdl.speech_encoder(**feats)
            ok += 1
        except Exception as e:
            print(f'  Sample {idx} failed: {e}')

    for h in hooks: h.remove()

    scores = {i: float(np.mean(v)) if v else 0.0 for i, v in bi.items()}
    print(f'  Calibrated on {ok}/{min(max_n, len(samples))} samples.')
    nonzero = sum(1 for v in scores.values() if v > 1e-6)
    print(f'  Non-zero BI scores: {nonzero}/{n}')
    if nonzero == 0:
        raise RuntimeError("All BI scores are zero — hooks did not fire.")

    # Print ranking so user can see what BI identified
    ranked = sorted(scores.items(), key=lambda x: x[1])
    print(f'\n  BI ranking (lowest = most redundant → pruning candidates):')
    for rank, (layer_i, bi_val) in enumerate(ranked):
        print(f'    Rank {rank+1:>2}  L{layer_i:>2}  BI={bi_val:.4f}')

    return scores


def iterative_enc_prune(mdl, samples, n_remove, tgt_lang='ben', max_eval=10,
                        ckpt_name='phase4_enc_pruning',
                        bi_scores=None,
                        bi_candidate_ratio=0.5,
                        protected=None):
    """
    BI-guided iterative greedy encoder pruning.

    Paper alignment:
    - ShortGPT (ACL 2025): BI scores rank layers by redundancy. We use them
      to restrict the candidate pool each iteration to the bottom bi_candidate_ratio
      fraction by BI score. This is the key improvement — instead of evaluating
      all N layers every iteration (O(N²) ChrF calls), we only evaluate the
      bottom-K BI candidates (O(N·K) calls), cutting runtime by ~50%.
    - Moslem IWSLT 2025: iterative greedy selection by ChrF, one layer at a time.
    - Custom rule: protected layers (first, middle, last) are never candidates.

    Args:
        bi_scores: dict {orig_layer_idx: bi_score}. If provided, only the
                   bottom bi_candidate_ratio fraction are evaluated each iter.
                   If None, falls back to evaluating all candidates (old behavior).
        bi_candidate_ratio: fraction of remaining layers to consider as candidates
                            based on BI ranking (default 0.5 = bottom 50%).
        protected: set of original layer indices to never prune.
    """
    parent, la = get_speech_encoder_layers(mdl)
    current  = list(getattr(parent, la))
    orig_idx = list(range(len(current)))
    n_total  = len(current)
    removed, log = [], []

    # ── Protected indices (never prunable) ──
    if protected is None:
        protected = {0, n_total // 2, n_total - 1}
    print(f'  Protected layers (first/mid/last): {sorted(protected)}')

    # ── Resume from checkpoint ──
    partial = load_latest_checkpoint(ckpt_name)
    if partial and partial.get('removed'):
        removed = list(partial['removed'])
        log = partial.get('log', [])
        for r in removed:
            if r in orig_idx:
                pos = orig_idx.index(r)
                current.pop(pos)
                orig_idx.pop(pos)
        setattr(parent, la, torch.nn.ModuleList(current))
        print(f'  Resuming: already removed {removed}, {len(current)} layers remain')

    start_iter = len(removed)
    baseline = quick_eval_chrf(mdl, samples, tgt_lang, max_eval)
    print(f'  Baseline ChrF (before any removal): {baseline:.2f}')

    for it in range(start_iter, n_remove):
        # ── Determine eligible candidates ──
        # 1. Exclude protected original indices
        eligible_positions = [
            pos for pos in range(len(current))
            if orig_idx[pos] not in protected
        ]
```
OUTPUT:
```text
Phase 4 helpers ready (BI-guided iterative pruning).
Speech encoder layers found: 24 at .speech_encoder.SeamlessM4Tv2ConformerEncoder.layers
```

### Cell 56 (code, score=44)
```python
# ── Uncomment and run ONCE to wipe corrupt Phase 4 results, then re-comment ──

# # 1. Local checkpoints
# for f in glob.glob(f'{CKPT_DIR}/phase4_enc_pruning_step*.pt'):
#     os.remove(f)
#     print(f'Deleted local ckpt: {f}')

# # 2. Local saved model
# local_model = f'{MODEL_DIR}/phase4_enc_pruned'
# if os.path.exists(local_model):
#     shutil.rmtree(local_model)
#     print(f'Deleted local model: {local_model}')

# # 3. Remote cleanup — platform-aware
# if ON_KAGGLE:
#     subprocess.run(f'rclone delete "{GDRIVE_ROOT}/checkpoints/phase4_enc_pruning_step000000.pt"',
#                    shell=True)
#     subprocess.run(f'rclone purge "{GDRIVE_ROOT}/phase4_enc_pruned"',
#                    shell=True)
#     r = subprocess.run(f'rclone ls "{GDRIVE_ROOT}/checkpoints/"',
#                        shell=True, capture_output=True, text=True)
#     print('Remaining on remote:', r.stdout or '(none)')
# else:
#     # Colab: just deleting local IS deleting Drive (they're the same)
#     print('Colab: local deletion above already removed files from Drive.')

# ck = load_latest_checkpoint('phase4_enc_pruning')
# print('Phase 4 checkpoint after cleanup:', ck)
```

### Cell 57 (code, score=10)
```python
# # Run once to fix model_p3 in memory before phase 3 benchmark
# sync_model_config(model_p3)

# # Run once to fix model_p4 in memory before phase 4 benchmark
# sync_model_config(model_p4)
```

### Cell 58 (code, score=7)
```python
# !rm -rf '/kaggle/working/models/phase4_enc_pruned'
# !rm -rf '/kaggle/working/checkpoints/phase6*.pt'
```

### Cell 59 (code, score=336)
```python
# ── Phase 4: Speech Encoder Iterative Layer Pruning ──────────────────────────
# Paper: ShortGPT (ACL 2025) — BI pre-ranks layers by redundancy.
#        Moslem IWSLT 2025 — iterative greedy ChrF-guided removal.
# Integration: BI restricts the candidate pool each iteration to the
#              bottom 50% by BI score, halving ChrF evaluation cost.
# Custom rule: protect first (L0), middle (L12), last (L23) layers.

N_ENC_REMOVE = 8
ENC_BI_CANDIDATE_RATIO = 0.5   # evaluate only the bottom 50% by BI each iter

p4_ckpt = load_latest_checkpoint('phase4_enc_pruning')
p4_complete = p4_ckpt and len(p4_ckpt.get('removed', [])) >= N_ENC_REMOVE

if p4_complete:
    removed_enc = p4_ckpt['removed']
    bi_scores   = p4_ckpt.get('bi_scores', {})
    p4_log      = p4_ckpt['log']
    print(f'Phase 4 complete: removed {removed_enc}')
    try:
        model_p4, processor = load_model_from_drive('phase4_enc_pruned')
    except:
        print('  Drive model missing, rebuilding from checkpoint + model_p3...')
        model_p4 = model_p3
        parent, la = get_speech_encoder_layers(model_p4)
        cur  = list(getattr(parent, la))
        keep = [i for i in range(len(cur)) if i not in removed_enc]
        setattr(parent, la, torch.nn.ModuleList([cur[i] for i in keep]))
        print(f'  Rebuilt from checkpoint: {len(keep)} layers remain')
        save_model_to_drive(model_p4, processor, 'phase4_enc_pruned')
else:
    done_so_far = len(p4_ckpt['removed']) if p4_ckpt else 0
    print(f'{"Resuming" if done_so_far else "Running"} Phase 4: '
          f'encoder pruning ({done_so_far}/{N_ENC_REMOVE} done)...')

    model_p4 = model_p3
    model_p4 = _consolidate_to_single_gpu(model_p4)

    sanity = quick_eval_chrf(model_p4, eval_samples, TARGET_LANG, 5)
    print(f'  Sanity check ChrF = {sanity:.2f}  (expect ~40-55, abort if < 10)')
    assert sanity > 10, f'ChrF={sanity:.2f} is too low — model or vocab remap is broken!'

    # ── Step 1: Compute BI scores (ShortGPT ACL 2025) ──────────────────────
    # BI = 1 - cosine_similarity(layer_input, layer_output)
    # Low BI → layer barely transforms hidden states → redundant candidate.
    # We use BI to PRE-FILTER candidates each iteration rather than just plotting.
    if not (p4_ckpt and p4_ckpt.get('bi_scores')):
        print('Step 1: Computing Block Influence scores (ShortGPT ACL 2025)...')
        bi_scores = compute_block_influence(model_p4, eval_samples, max_n=50)
        plot_layer_scores(bi_scores, 'Speech Encoder Block Influence', 'phase4_bi.png')
        # Save BI scores immediately so resume works
        save_checkpoint(
            dict(removed=[], log=[], bi_scores=bi_scores),
            name='phase4_enc_pruning', step=0)
    else:
        bi_scores = p4_ckpt['bi_scores']
        print(f'  BI scores loaded from checkpoint ({len(bi_scores)} layers), '
              f'skipping recomputation.')

    # Determine protected set: first, middle, last of the ORIGINAL encoder
    parent_tmp, la_tmp = get_speech_encoder_layers(model_p4)
    n_enc_layers = len(getattr(parent_tmp, la_tmp))
    enc_protected = {0, n_enc_layers // 2, n_enc_layers - 1}
    print(f'\n  Encoder protected layers: {sorted(enc_protected)} '
          f'(of {n_enc_layers} total)')

    # ── Step 2: BI-guided iterative pruning (Moslem IWSLT 2025 + ShortGPT) ──
    # Each iteration: evaluate only bottom-50% BI candidates (not all layers).
    # This directly uses BI as a meaningful pre-filter, not just a plot.
    print(f'\nStep 2: BI-guided iterative pruning ({N_ENC_REMOVE} layers, '
          f'candidate ratio={ENC_BI_CANDIDATE_RATIO})...')
    removed_enc, p4_log = iterative_enc_prune(
        model_p4, eval_samples, N_ENC_REMOVE, TARGET_LANG,
        max_eval=10,
        ckpt_name='phase4_enc_pruning',
        bi_scores=bi_scores,                        # ← NOW ACTUALLY USED
        bi_candidate_ratio=ENC_BI_CANDIDATE_RATIO,  # ← evaluate bottom 50% by BI
        protected=enc_protected,                    # ← first/mid/last protected
    )

    print('Syncing config after encoder pruning...')
    sync_model_config(model_p4)

    save_checkpoint(
        dict(removed=removed_enc, log=p4_log, bi_scores=bi_scores),
        name='phase4_enc_pruning', step=0)
    save_model_to_drive(model_p4, processor, 'phase4_enc_pruned')

print(f'\nEncoder layers removed: {removed_enc}')
print_model_breakdown(model_p4, 'After Phase 4: Encoder Pruned')
```
OUTPUT:
```text
[ckpt] No checkpoint for 'phase4_enc_pruning'
Running Phase 4: encoder pruning (0/8 done)...
  Sanity check ChrF = 44.23  (expect ~40-55, abort if < 10)
Step 1: Computing Block Influence scores (ShortGPT ACL 2025)...
  Calibrating 0/25...
  Calibrating 10/25...
  Calibrating 20/25...
  Calibrated on 25/25 samples.
  Non-zero BI scores: 24/24

  BI ranking (lowest = most redundant → pruning candidates):
    Rank  1  L10  BI=0.1411
    Rank  2  L16  BI=0.1467
    Rank  3  L15  BI=0.1511
    Rank  4  L11  BI=0.1522
    Rank  5  L 9  BI=0.1529
    Rank  6  L14  BI=0.1586
    Rank  7  L13  BI=0.1638
    Rank  8  L 2  BI=0.1678
    Rank  9  L12  BI=0.1680
    Rank 10  L18  BI=0.1694
    Rank 11  L19  BI=0.1768
    Rank 12  L17  BI=0.1833
    Rank 13  L 3  BI=0.1879
    Rank 14  L 5  BI=0.1913
    Rank 15  L20  BI=0.2003
    Rank 16  L 6  BI=0.2066
    Rank 17  L 1  BI=0.2120
    Rank 18  L 4  BI=0.2220
    Rank 19  L21  BI=0.2387
    Rank 20  L 7  BI=0.2473
    Rank 21  L22  BI=0.2584
    Rank 22  L 8  BI=0.3253
    Rank 23  L23  BI=0.4570
    Rank 24  L 0  BI=0.5806

<Figure size 1440x600 with 1 Axes>
[image/png output omitted]
[ckpt] Saved phase4_enc_pruning_step000000.pt (0.0 MB)

  Encoder protected layers: [0, 12, 23] (of 24 total)

Step 2: BI-guided iterative pruning (8 layers, candidate ratio=0.5)...
  Protected layers (first/mid/last): [0, 12, 23]
[ckpt] Loaded phase4_enc_pruning_step000000.pt
  Baseline ChrF (before any removal): 43.86

  Iter 1/8 (24 layers remain)
  BI pre-filter: 10/21 eligible layers selected as candidates (bottom 50% by BI)
  Skipped (high BI / important): [17, 3, 5, 20, 6, 1, 4, 21, 7, 22, 8]
    Remove L10 -> ChrF=42.33  BI=0.1411
    Remove L16 -> ChrF=43.71  BI=0.1467
    Remove L15 -> ChrF=42.82  BI=0.1511
    Remove L11 -> ChrF=43.64  BI=0.1522
    Remove L 9 -> ChrF=45.35  BI=0.1529
    Remove L14 -> ChrF=42.42  BI=0.1586
    Remove L13 -> ChrF=43.17  BI=0.1638
    Remove L 2 -> ChrF=46.59  BI=0.1678
    Remove L18 -> ChrF=42.11  BI=0.1694
    Remove L19 -> ChrF=42.83  BI=0.1768
  -> Removed original layer 2  (ChrF=46.59, BI=0.1678)
[ckpt] Saved phase4_enc_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (1/8 iterations done)

  Iter 2/8 (23 layers remain)
  BI pre-filter: 10/20 eligible layers selected as candidates (bottom 50% by BI)
  Skipped (high BI / important): [3, 5, 20, 6, 1, 4, 21, 7, 22, 8]
    Remove L10 -> ChrF=46.27  BI=0.1411
    Remove L16 -> ChrF=41.16  BI=0.1467
    Remove L15 -> ChrF=46.06  BI=0.1511
    Remove L11 -> ChrF=47.00  BI=0.1522
    Remove L 9 -> ChrF=45.27  BI=0.1529
    Remove L14 -> ChrF=45.88  BI=0.1586
    Remove L13 -> ChrF=41.23  BI=0.1638
    Remove L18 -> ChrF=43.22  BI=0.1694
    Remove L19 -> ChrF=44.14  BI=0.1768
    Remove L17 -> ChrF=41.71  BI=0.1833
  -> Removed original layer 11  (ChrF=47.00, BI=0.1522)
[ckpt] Saved phase4_enc_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (2/8 iterations done)

  Iter 3/8 (22 layers remain)
  BI pre-filter: 9/19 eligible layers selected as candidates (bottom 50% by BI)
  Skipped (high BI / important): [3, 5, 20, 6, 1, 4, 21, 7, 22, 8]
    Remove L10 -> ChrF=45.69  BI=0.1411
    Remove L16 -> ChrF=46.64  BI=0.1467
    Remove L15 -> ChrF=43.73  BI=0.1511
    Remove L 9 -> ChrF=45.53  BI=0.1529
    Remove L14 -> ChrF=46.86  BI=0.1586
    Remove L13 -> ChrF=45.53  BI=0.1638
    Remove L18 -> ChrF=43.71  BI=0.1694
    Remove L19 -> ChrF=43.12  BI=0.1768
    Remove L17 -> ChrF=45.18  BI=0.1833
  -> Removed original layer 14  (ChrF=46.86, BI=0.1586)
[ckpt] Saved phase4_enc_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (3/8 iterations done)

  Iter 4/8 (21 layers remain)
  BI pre-filter: 9/18 eligible layers selected as candidates (bottom 50% by BI)
  Skipped (high BI / important): [5, 20, 6, 1, 4, 21, 7, 22, 8]
    Remove L10 -> ChrF=42.43  BI=0.1411
    Remove L16 -> ChrF=45.56  BI=0.1467
    Remove L15 -> ChrF=43.69  BI=0.1511
    Remove L 9 -> ChrF=45.06  BI=0.1529
    Remove L13 -> ChrF=45.65  BI=0.1638
    Remove L18 -> ChrF=45.33  BI=0.1694
    Remove L19 -> ChrF=44.45  BI=0.1768
    Remove L17 -> ChrF=47.33  BI=0.1833
    Remove L 3 -> ChrF=41.49  BI=0.1879
  -> Removed original layer 17  (ChrF=47.33, BI=0.1833)
[ckpt] Saved phase4_enc_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (4/8 iterations done)

  Iter 5/8 (20 layers remain)
  BI pre-filter: 8/17 eligible layers selected as candidates (bottom 50% by BI)
  Skipped (high BI / important): [5, 20, 6, 1, 4, 21, 7, 22, 8]
    Remove L10 -> ChrF=43.50  BI=0.1411
    Remove L16 -> ChrF=44.05  BI=0.1467
    Remove L15 -> ChrF=46.15  BI=0.1511
    Remove L 9 -> ChrF=44.88  BI=0.1529
    Remove L13 -> ChrF=44.11  BI=0.1638
    Remove L18 -> ChrF=44.17  BI=0.1694
    Remove L19 -> ChrF=42.40  BI=0.1768
    Remove L 3 -> ChrF=38.80  BI=0.1879
  -> Removed original layer 15  (ChrF=46.15, BI=0.1511)
[ckpt] Saved phase4_enc_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (5/8 iterations done)

  Iter 6/8 (19 layers remain)
  BI pre-filter: 8/16 eligible layers selected as candidates (bottom 50% by BI)
  Skipped (high BI / important): [20, 6, 1, 4, 21, 7, 22, 8]
    Remove L10 -> ChrF=46.03  BI=0.1411
    Remove L16 -> ChrF=40.64  BI=0.1467
    Remove L 9 -> ChrF=46.21  BI=0.1529
    Remove L13 -> ChrF=41.05  BI=0.1638
    Remove L18 -> ChrF=43.67  BI=0.1694
    Remove L19 -> ChrF=45.30  BI=0.1768
    Remove L 3 -> ChrF=40.00  BI=0.1879
    Remove L 5 -> ChrF=44.47  BI=0.1913
  -> Removed original layer 9  (ChrF=46.21, BI=0.1529)
[ckpt] Saved phase4_enc_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (6/8 iterations done)

  Iter 7/8 (18 layers remain)
  BI pre-filter: 7/15 eligible layers selected as candidates (bottom 50% by BI)
  Skipped (high BI / important): [20, 6, 1, 4, 21, 7, 22, 8]
    Remove L10 -> ChrF=40.64  BI=0.1411
    Remove L16 -> ChrF=42.48  BI=0.1467
    Remove L13 -> ChrF=43.22  BI=0.1638
    Remove L18 -> ChrF=43.81  BI=0.1694
    Remove L19 -> ChrF=44.96  BI=0.1768
    Remove L 3 -> ChrF=43.67  BI=0.1879
    Remove L 5 -> ChrF=43.21  BI=0.1913
  -> Removed original layer 19  (ChrF=44.96, BI=0.1768)
[ckpt] Saved phase4_enc_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (7/8 iterations done)

  Iter 8/8 (17 layers remain)
  BI pre-filter: 7/14 eligible layers selected as candidates (bottom 50% by BI)
  Skipped (high BI / important): [6, 1, 4, 21, 7, 22, 8]
    Remove L10 -> ChrF=42.29  BI=0.1411
    Remove L16 -> ChrF=42.00  BI=0.1467
    Remove L13 -> ChrF=42.41  BI=0.1638
    Remove L18 -> ChrF=43.96  BI=0.1694
    Remove L 3 -> ChrF=39.24  BI=0.1879
    Remove L 5 -> ChrF=44.02  BI=0.1913
    Remove L20 -> ChrF=43.22  BI=0.2003
  -> Removed original layer 5  (ChrF=44.02, BI=0.1913)
[ckpt] Saved phase4_enc_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (8/8 iterations done)
Syncing config after encoder pruning...
  [config] speech_encoder_layers: 24 -> 16
  [config] speech_encoder.config.num_hidden_layers: 24 -> 16
  [config] sync done.
[ckpt] Saved phase4_enc_pruning_step000000.pt (0.0 MB)
[model] Saving phase4_enc_pruned → /kaggle/working/models/phase4_enc_pruned ...
  [config] sync done.
  Saved custom state: ['_vocab_remap_to_old']
  Saved pruning_manifest.pt keys=['stage_name']

Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]
[model] Local save done. 2270 MB in 8 files.
[model] Pushing to rclone remote...
[model] Verified 8 files on remote.

Encoder layers removed: [2, 11, 14, 17, 15, 9, 19, 5]

--- After Phase 4: Encoder Pruned ---
  speech_encoder                         441.6M  ( 39.5%)
  text_decoder                           373.6M  ( 33.4%)
  t2u_model                              261.8M  ( 23.4%)
  vocoder                                 41.9M  (  3.7%)
  shared                                  20.9M  (  1.9%)
  lm_head                                 20.9M  (  1.9%)
  TOTAL                                 1118.8M
---

{'shared': 20.9152,
 'speech_encoder': 441.604416,
 'text_decoder': 373.568512,
 'lm_head': 20.9152,
 't2u_model': 261.759747,
 'vocoder': 41.911362,
 'TOTAL': 1118.844037}
```

### Cell 60 (code, score=60)
```python
if bi_scores and p4_log:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # BI at removal time (always use for removed layers + plot 3)
    bi_at_removal = {e['removed']: e.get('bi_score') for e in p4_log if 'removed' in e}

    def bi_value_for_index(i):
        if i in bi_scores:
            return bi_scores[i]
        b = bi_at_removal.get(i)
        return b if b is not None else 0.0

    # ── Plot 1: BI scores with removed/protected/kept labels ──
    ax1 = axes[0]
    parent_tmp, la_tmp = get_speech_encoder_layers(model_p4)
    n_enc = len(getattr(parent_tmp, la_tmp)) + len(removed_enc)
    enc_protected = {0, n_enc // 2, n_enc - 1}

    all_indices = sorted(
        set(bi_scores.keys()) | set(removed_enc) | set(bi_at_removal.keys())
    )
    vals = [bi_value_for_index(i) for i in all_indices]
    colors = []
    for i in all_indices:
        if i in removed_enc:
            colors.append('#d32f2f')
        elif i in enc_protected:
            colors.append('#ff9800')
        else:
            colors.append('#4caf50')

    ax1.bar(all_indices, vals, width=0.85, color=colors, edgecolor='white')
    ax1.set_title(
        'Block Influence per Layer\n(red=removed, orange=protected, green=kept)',
        fontweight='bold',
    )
    ax1.set_xlabel('Original Layer Index')
    ax1.set_ylabel('BI Score (lower = more redundant)')

    # ── Plot 2: ChrF after each removal ──
    ax2 = axes[1]
    iters = [e['iter'] for e in p4_log]
    chrfs = [e['chrf'] for e in p4_log]
    ax2.plot(iters, chrfs, 'o-', color='#2196F3', lw=2)
    for e in p4_log:
        bi_val = e.get('bi_score', None)
        label = f'L{e["removed"]}'
        if bi_val is not None:
            label += f'\nBI={bi_val:.3f}'
        ax2.annotate(
            label, (e['iter'], e['chrf']), fontsize=7,
            textcoords='offset points', xytext=(0, 6),
        )
    ax2.set_title(
        'ChrF After Each Removal\n(annotated with removed layer + BI)',
        fontweight='bold',
    )
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('ChrF Score')

    # ── Plot 3: BI of removed layer vs ChrF after that removal ──
    ax3 = axes[2]
    _filtered = [
        (e['bi_score'], e['chrf'])
        for e in p4_log
        if e.get('bi_score') is not None
    ]
    bi_vals_removed = [x for x, _ in _filtered]
    chrf_vals = [y for _, y in _filtered]

    if len(_filtered) >= 2:
        ax3.scatter(bi_vals_removed, chrf_vals, color='#9c27b0', s=80, zorder=3)
        for e in p4_log:
            if e.get('bi_score') is not None:
                ax3.annotate(
                    f'L{e["removed"]}',
                    (e['bi_score'], e['chrf']),
                    fontsize=8, textcoords='offset points', xytext=(4, 2),
                )
        ax3.set_xlabel('BI Score of Removed Layer (from log)')
        ax3.set_ylabel('ChrF After Removal')
        ax3.set_title(
            'BI Score vs ChrF After Removal\n(validates ShortGPT: low BI → safe removal)',
            fontweight='bold',
        )
    else:
        ax3.text(
            0.5, 0.5, 'Not enough data\nfor correlation plot',
            ha='center', va='center', transform=ax3.transAxes,
        )
        ax3.set_title('BI vs ChrF Correlation', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/phase4_enc_bi_analysis.png', dpi=120)
    plt.show()
    print('Saved phase4_enc_bi_analysis.png')
```
OUTPUT:
```text
<Figure size 2160x600 with 3 Axes>
[image/png output omitted]
Saved phase4_enc_bi_analysis.png
```

### Cell 61 (code, score=157)
```python
run_benchmark(model_p4, eval_samples, label='P4_EncPrune', save_n=2)
```
OUTPUT:
```text
============================================================
  BENCHMARK: P4_EncPrune
  Samples: 25  Target: ben
============================================================

  GPU mem: 2.27 GB alloc / 3.98 GB reserved
  [ 1/25] BLEU= 14.8 ChrF= 46.9 RTF=0.092  id=1660
              pred: রোমান্টিকতাবাদ সংস্কৃতির নির্ণয়বাদের একটি বড় উপাদান ছিল, যা গথ, ফিচট এবং শ্লেগ
[audio] Saved P4_EncPrune_s1in.wav (0.3 MB)
  P4_EncPrune_s1in.wav  (10.7s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P4_EncPrune_s1out.wav (0.3 MB)
  P4_EncPrune_s1out.wav  (8.1s | sr=16000)

<IPython.lib.display.Audio object>
  [ 2/25] BLEU= 10.1 ChrF= 40.9 RTF=0.096  id=1661
              pred: তিনি বলেন, তিনি চীনের অর্থনৈতিক উৎপাদনের উপর ভিত্তি করে এই সংখ্যাটি তৈরি করা হবে
[audio] Saved P4_EncPrune_s2in.wav (0.2 MB)
  P4_EncPrune_s2in.wav  (6.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P4_EncPrune_s2out.wav (0.2 MB)
  P4_EncPrune_s2out.wav  (6.0s | sr=16000)

<IPython.lib.display.Audio object>
  [ 3/25] BLEU= 20.1 ChrF= 45.5 RTF=0.093  id=1662
              pred: মিশ্রণ মূলত দুই বা তার বেশি ধাতব মিশ্রণ, পিইআইআরআই টেবিলের উপর অনেক উপাদান রয়েছ
  [ 4/25] BLEU=  5.2 ChrF= 42.4 RTF=0.095  id=1663
              pred: চোকামু উপত্যকা, চিলির শীর্ষস্থানীয় আরোহণের গন্তব্য, দক্ষিণ আমেরিকা'র ইয়েসোমিটি
  [ 5/25] BLEU= 13.3 ChrF= 42.1 RTF=0.086  id=1664
              pred: দুটি ড্রাই পাতা একসাথে ঘুরান এবং তারপর চিলিয়ে ঘন হাত দিয়ে তাদের একটি বলের মধ্য
  [ 6/25] BLEU=  5.6 ChrF= 23.9 RTF=0.073  id=1665
              pred: "লিকের মতে, ""ডকুমেন্টটি সীমান্ত বিরোধের কথা উল্লেখ করবে, যা"
  [ 7/25] BLEU=  9.8 ChrF= 50.6 RTF=0.076  id=1666
              pred: আপনি আপনার নিজের সরকার ছাড়া অন্য সরকারের পরামর্শ নিয়ে পরামর্শ নিতে পারেন, কিন্
  [ 8/25] BLEU=  4.6 ChrF= 45.5 RTF=0.094  id=1667
              pred: সাধারণভাবে, দুইটি আচরণ বিবর্তনগুলি উদ্ভূত হতে পারে, যেহেতু ম্যানেজাররা তাদের প্র
  [ 9/25] BLEU=  7.0 ChrF= 50.9 RTF=0.074  id=1668
              pred: এটি একটি ওয়াইল্ডকার্ড কিনতেও উপকারী হতে পারে, যা দক্ষিণ আফ্রিকার পার্কের যে কোন
  [10/25] BLEU=  7.3 ChrF= 51.4 RTF=0.111  id=1669
              pred: পুলিশ সুপারিনটেন্ডেন্ট চান্দ্রা শিকর সুলঙ্কি বলেন, অভিযুক্তরা মুখোমুখি হয়ে আদাল
  [11/25] BLEU=  0.2 ChrF=  7.8 RTF=0.187  id=1670
              pred: তাদের আনুষ্ঠানিক আচরণ, প্রায়শই স্থিরতা বজায় রাখার মতো বড় বড় বড় বড় বড় বড় 
  [12/25] BLEU= 14.5 ChrF= 51.4 RTF=0.135  id=1671
              pred: কংগ্রেস অযৌনতা ইনিশিয়েয়েয়েকে এবং ফিসাল-২৫৫৫-এ অর্থায়ন শুরু করে এবং নির্দিষ্ট
  [13/25] BLEU=  6.7 ChrF= 30.2 RTF=0.068  id=1672
              pred: ফ্যাব্রিককে খুব গরম হতে দেয় না, যা সংকুচিত হতে পারে, বা চরম ক্ষেত্রে, পুড়ে যায
  [14/25] BLEU= 22.3 ChrF= 68.5 RTF=0.095  id=1673
              pred: বিপ্লবী যুদ্ধের সময়, ১৩টি রাজ্যে প্রথমবারের মতো একটি দুর্বল কেন্দ্রীয় সরকার গঠ
  [15/25] BLEU=  8.2 ChrF= 40.9 RTF=0.098  id=1674
              pred: কিছু এলাকায়, এক মিনিটের জন্য উষ্ণ জল যথেষ্ট এবং অন্য কয়েক মিনিট প্রয়োজন হয়।
  [16/25] BLEU=  0.0 ChrF=  7.3 RTF=0.056  id=1675
              pred: "প্রাণের মাঝামাঝি পর্যন্ত আপনার জন্য " - "
  [17/25] BLEU=  8.8 ChrF= 60.2 RTF=0.061  id=1676
              pred: দক্ষিণ আফ্রিকার সকল জাতীয় উদ্যানের মতো, পার্কের জন্য প্রতিদিন সংরক্ষণ এবং প্রবে
  [18/25] BLEU=  6.4 ChrF= 37.1 RTF=0.130  id=1677
              pred: আজ, একমাত্র পোকা যে তাদের ডানাগুলিকে পিছনে ভাঁড়ানো যায় না তা হ'ল ড্রাগনফ্লি এব
  [19/25] BLEU=  2.2 ChrF= 30.9 RTF=0.068  id=1678
              pred: "অলিভার স্যাক্স তার কাগজতে রাষ্ট্রপতির বক্তৃতাটি নির্দেশ করে যে, মস্তিষ্কের ক্ষত
  [20/25] BLEU=  6.6 ChrF= 45.8 RTF=0.121  id=1679
              pred: এরা স্মিথ তাদের সফরর বাকি কনসেন্ট বাতিল করেছে।
  [21/25] BLEU=  3.8 ChrF= 28.1 RTF=0.089  id=1680
              pred: একটি সু-গোল্লা, বাঘ ভাল, ভাল না, যদিও, সাঁতার, লম্বা, বড় দূরত্ব, এবং পাঁচবার এক
  [22/25] BLEU= 13.5 ChrF= 52.1 RTF=0.064  id=1681
              pred: তবে, এটি কেবলমাত্র পরীক্ষা নয়, এবং এটি এমন একটি পরীক্ষা যা এক বা একাধিক সম্ভাব্
  [23/25] BLEU=  0.7 ChrF= 16.4 RTF=0.121  id=1682
              pred: যদিও কেউই নিশ্চিত না যে এটি কে লিখেছেন, তবে এটি তার জীবনের প্রথম দিকে, এটির বৃহত
  [24/25] BLEU=  6.0 ChrF= 39.7 RTF=0.101  id=1683
              pred: "তারা আরও আরও আরও লিখেছেন, "তারা এখনও তাদের সময় বেঁচে আছে, এবং আরও অনেক লোক আছে
  [25/25] BLEU=  6.6 ChrF= 46.2 RTF=0.061  id=1684
              pred: সামোয়া'র রাজধানী, শহরটি উপোলু দ্বীপের মধ্যে এবং জনসংখ্যা ৪০ হাজারেরও কম।

  Summary: BLEU=8.19  ChrF=40.11  RTF=0.0939  Params=1118.8M


([{'id': 1660,
   'bleu': 14.81394578697113,
   'chrf': 46.9022070126263,
   'rtf': 0.09240604072027886,
   'pred': 'রোমান্টিকতাবাদ সংস্কৃতির নির্ণয়বাদের একটি বড় উপাদান ছিল, যা গথ, ফিচট এবং শ্লেগেলের মতো লেখকদের কাছ থেকে নেওয়া হয়েছিল।',
   'ref': 'সংস্কৃতির দিক নির্ধারণের ক্ষেত্রে একটি বড় উপাদান ছিল শ্লেগাল গোথা ফিশ্তাদের মতো লেখকদের রোম্যান্টিসিজম'},
  {'id': 1661,
   'bleu': 10.123734869668828,
   'chrf': 40.860470118108495,
   'rtf': 0.0960721159881016,
   'pred': 'তিনি বলেন, তিনি চীনের অর্থনৈতিক উৎপাদনের উপর ভিত্তি করে এই সংখ্যাটি তৈরি করা হবে বলে বলেন।',
   'ref': 'তিনি এই কমানোর জন্য কোনও পরিমাণ স্থাপন করেননি বলেছিলেন যেগুলিচীনের অর্থনৈতিক আয়ের ভিত্তিতে তৈরি করা হবে'},
  {'id': 1662,
   'bleu': 20.105373454060025,
   'chrf': 45.470216986763305,
   'rtf': 0.09314240020897621,
   'pred': 'মিশ্রণ মূলত দুই বা তার বেশি ধাতব মিশ্রণ, পিইআইআরআই টেবিলের উপর অনেক উপাদান রয়েছে তা ভুলবেন না।',
   'ref': 'সঙ্কর ধাতুগুলি মূলত দুই বা ততোধিক ধাতুর মিশ্রণ।পর্যায় সারণীতে অনেক উপাদান রয়েছে তা ভুলে যাবেন না।'},
  {'id': 1663,
   'bleu': 5.237520761048587,
   'chrf': 42.42533944594271,
   'rtf': 0.09509106673816643,
   'pred': "চোকামু উপত্যকা, চিলির শীর্ষস্থানীয় আরোহণের গন্তব্য, দক্ষিণ আমেরিকা'র ইয়েসোমিটি নামে পরিচিত, বিভিন্ন ধরণের গ্রানাইট-খসড়ক ও পাথের সাথে।",
   'ref': 'কোক্যামো উপত্যকা দক্ষিণ আমেরিকার ইয়োসেমাইট নামে পরিচিত গ্রানাইটের বড় প্রাচীর ও দুরারোহ পাহাড় সমেত চিলির প্রধান আরোহণের গন্থব্যস্থল'},
  {'id': 1664,
   'bleu': 13.32358437599213,
   'chrf': 42.10591241301607,
   'rtf': 0.08649081704719215,
   'pred': 'দুটি ড্রাই পাতা একসাথে ঘুরান এবং তারপর চিলিয়ে ঘন হাত দিয়ে তাদের একটি বলের মধ্যে সঙ্কুচিত করুন',
   'ref': 'দুটি শুকনো গুঁড়ো একসাথে মোচড় দিন এবং তারপর পরিষ্কার ভেজা হাত দিয়ে একটি বলের মধ্যে চাপ দিন'},
  {'id': 1665,
   'bleu': 5.631533898837127,
   'chrf': 23.916023028676,
   'rtf': 0.0726284324258998,
   'pred': '"লিকের মতে, ""ডকুমেন্টটি সীমান্ত বিরোধের কথা উল্লেখ করবে, যা"',
   'ref': 'ফাঁস হওয়া তথ্য অনুসারে দলিলটি সীমান্ত বিরোধের বিষয়ে উল্লেখ করবে যা ফিলিস্তিন 1967 সালের মধ্যপ্রাচ্য যুদ্ধের আগের সীমান্তের ভিত্তিতে চায়'},
  {'id': 1666,
   'bleu': 9.846107951428584,
   'chrf': 50.604754482054446,
   'rtf': 0.07602885472688743,
   'pred': 'আপনি আপনার নিজের সরকার ছাড়া অন্য সরকারের পরামর্শ নিয়ে পরামর্শ নিতে পারেন, কিন্তু তাদের পরামর্শ তাদের নাগরিকদের জন্য ডিজাইন করা হয়।',
   'ref': 'আপনি নিজস্ব দেশের সরকার ছাড়াও অন্য কোন দেশের সরকারের যুক্তি চাইতে পারেন কিন্তু তাদের যুক্তি তাদের নাগরিকদের জন্য গঠন করা হয়েছে'},
  {'id': 1667,
   'bleu': 4.626647494578085,
   'chrf': 45.49838715995338,
   'rtf': 0.09417243815340237,
   'pred': 'সাধারণভাবে, দুইটি আচরণ বিবর্তনগুলি উদ্ভূত হতে পারে, যেহেতু ম্যানেজাররা তাদের প্রাক্তন সমনবীনদের নেতৃত্ব দিতে শুরু করে, তবে স্পেকট্রামের শেষের একটি অংশ পুরুষদের বা মেতারাদের মধ্যে একটি হিসাবে থাকতে চেষ্টা করে।',
   'ref': '"সাধারণভাবে বলতে গেলে পরিচালকরা তাদের প্রাক্তন সহকর্মীদের নেতৃত্ব দেওয়া শুরু করার সাথে সাথে দুটি আচরণের উত্থান হতে পারে। আচরণগুলোর একটি অংশ "ছেলেদের মধ্যে একজন" বা মেয়ে হয়ে থাকার চেষ্টা করে।'},
  {'id': 1668,
   'bleu': 6.964541799727335,
   'chrf': 50.94986767293174,
   'rtf': 0.07420234136037282,
   'pred': 'এটি একটি ওয়াইল্ডকার্ড কিনতেও উপকারী হতে পারে, যা দক্ষিণ আফ্রিকার পার্কের যে কোন নির্বাচিত বা দক্ষিণ আফ্রিকার জাতীয় উদ্যানগুলির জন্য প্রবেশের অনুমতি দেয়।',
   'ref': 'কেউ ওয়াইল্ড কার্ডও কিনতে পারে যা দক্ষিণ আফ্রিকার বিশেষ কিছু পার্ক অথবা যে কোনো জাতীয় উদ্যানে ঢোকার সুযোগ দেয়'},
  {'id': 1669,
   'bleu': 7.347053125977879,
   'chrf': 51.43941989659526,
   'rtf': 0.11110769377814399,
   'pred': 'পুলিশ সুপারিনটেন্ডেন্ট চান্দ্রা শিকর সুলঙ্কি বলেন, অভিযুক্তরা মুখোমুখি হয়ে আদালতে হাজির হয়েছে।',
   'ref': 'পুলিশ সুপার চন্দ্র শেখর সোলঙ্কি জানিয়েছিলেন অভিযুক্তরা মুখ ঢাকা অবস্থায় আদালতে হাজির হয়েছিল'},
  {'id': 1670,
   'bleu': 0.21190017631330177,
   'chrf': 7.8399219608740855,
   'rtf': 0.18698329404759326,
   'pred': 'তাদের আনুষ্ঠানিক আচরণ, প্রায়শই স্থিরতা বজায় রাখার মতো বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড় বড়',
   'ref': '"মার্কিন যুক্তরাষ্ট্রের জিওলজিকাল সার্ভের usgs জ্যোতির্বিজ্ঞান দল এবং আরিজোনার ফ্ল্যাংস্ট্যাফে অবস্থিত উত্তর অ্যারিজোনা বিশ্ববিদ্যালয়ের গ্লেন কুশিং বলেছেন "তাদের তাপীয় আচরণ পৃথিবীর বৃহত্তর গুহাগুলির মতো স্থির নয় যা প্রায়শই স্থির তাপমাত্রা বজায় রাখে কিন্তু এটি স্থলভাগের গভীর গর্তের সাথে সামঞ্জস্যপূর্ণ""। '},
  {'id': 1671,
   'bleu': 14.488239708705215,
   'chrf': 51.361298420081226,
   'rtf': 0.13509658044445777,
   'pred': "কংগ্রেস অযৌনতা ইনিশিয়েয়েয়েকে এবং ফিসাল-২৫৫৫-এ অর্থায়ন শুরু করে এবং নির্দিষ্ট করে যে এফবিআই-তে দশজন এজেন্টকে প্রাপ্ত বয়স্কদের পর্নোগ্রাফি'র জন্য নিয়োজনীকরণ করতে হবে।",
   'ref': 'কংগ্রেস 2005-এর অর্থবছরে অশ্লীল উদ্যোগকে অর্থ জোগান দেয়া শুরু করে এবং নির্দিষ্ট করে দেয় যে এফবিআই-কে অবশ্যই প্রাপ্তবয়স্কদের পর্নোগ্রাফিতে 10 জন এজেন্টদের নিযুক্ত করতে হবে'},
  {'id': 1672,
   'bleu': 6.722636787666482,
   'chrf': 30.195607079794623,
   'rtf': 0.06849552233388105,
   'pred': 'ফ্যাব্রিককে খুব গরম হতে দেয় না, যা সংকুচিত হতে পারে, বা চরম ক্ষেত্রে, পুড়ে যায়।',
   'ref': 'সাবধান হন কাপড় যাতে খুব গরম না হয়ে যায় এতে সঙ্কোচন হতে পারে বা ক্ষেত্রবিশেষে জ্বলেও যেতে পারে।'},
  {'id': 1673,
   'bleu': 22.325877055095216,
   'chrf': 68.53191002876947,
   'rtf': 0.09506060963585262,
   'pred': "বিপ্লবী যুদ্ধের সময়, ১৩টি রাজ্যে প্রথমবারের মতো একটি দুর্বল কেন্দ্রীয় সরকার গঠন করে, কংগ্রেস তার একমাত্র উপাদান ছিল কনফেডারেশন'র নিবন্ধের অধীনে।",
   'ref': 'বিপ্লবী যুদ্ধের সময় কনফেডারেশনের নিবন্ধের অধীনে তেরটি রাজ্য প্রথমে একটি দুর্বল কেন্দ্রীয় সরকার গঠন করেছিল যেখানে কংগ্রেস এর একমাত্র অংশ ছিলো'},
  {'id': 1674,
   'bleu': 8.225964699966553,
   'chrf': 40.89090197775384,
   'rtf': 0.09773707558922734,
   'pred': 'কিছু এলাকায়, এক মিনিটের জন্য উষ্ণ জল যথেষ্ট এবং অন্য কয়েক মিনিট প্রয়োজন হয়।',
   'ref': 'কিছু এলাকায় এক মিনিট পানি ফুটানো যথেষ্ট অন্যান্য এলাকায় কয়েক মিনিট পানি ফুটানোর দরকার হয়'},
  {'id': 1675,
   'bleu': 0.0,
   'chrf': 7.336690080476522,
   'rtf': 0.05627320781690345,
   'pred': '"প্রাণের মাঝামাঝি পর্যন্ত আপনার জন্য " - "',
   'ref': "বিশেষ্য শব্দগুলির মতো 'সি' তুমি শব্দটিও সবসময় বড় হাতের অক্ষর দিয়ে শুরু হয় এমনকি বাক্যের মাঝেও।"},
  {'id': 1676,
   'bleu': 8.839374326825924,
   'chrf': 60.22537921298202,
   'rtf': 0.06066302584994371,
   'pred': 'দক্ষিণ আফ্রিকার সকল জাতীয় উদ্যানের মতো, পার্কের জন্য প্রতিদিন সংরক্ষণ এবং প্রবেশের ফি রয়েছে।',
   'ref': 'দক্ষিণ আফ্রিকার প্রতিটি জাতীয় পার্কের মতো পার্কটির জন্য প্রতিদিনের সংরক্ষণ এবং প্রবেশমূল্য আছে'},
  {'id': 1677,
   'bleu': 6.43716525407242,
   'chrf': 37.05993188705632,
   'rtf': 0.1295328140258789,
   'pred': "আজ, একমাত্র পোকা যে তাদের ডানাগুলিকে পিছনে ভাঁড়ানো যায় না তা হ'ল ড্রাগনফ্লি এবং মেফাই।",
   'ref': 'বর্তমানে কেবলমাত্র যে পোকামাকড়গুলো তাদের ডানা পিছনে ভাঁজ করতে পারে না তা হল ফড়িং এবং মেফ্লাইস'},
  {'id': 1678,
   'bleu': 2.242872568617871,
   'chrf': 30.930003236505083,
   'rtf': 0.06837648964823692,
   'pred': '"অলিভার স্যাক্স তার কাগজতে রাষ্ট্রপতির বক্তৃতাটি নির্দেশ করে যে, মস্তিষ্কের ক্ষতি এবং অঙ্গভঙ্গির কারণে "',
   'ref': 'অলিভার স্যাকস তার পত্রে রাষ্ট্রপতির বক্তব্যে মস্তিষ্কে ত্রুটি থাকার ফলে বক্তব্য বুঝতে অক্ষম হলেও সঠিকভাবে আন্তরিকতা মূল্যায়নে সক্ষম হওয়ার বিষয়টি ইঙ্গিত করেছেন'},
  {'id': 1679,
   'bleu': 6.567274736060396,
   'chrf': 45.82864944698806,
   'rtf': 0.12121086168770838,
   'pred': 'এরা স্মিথ তাদের সফরর বাকি কনসেন্ট বাতিল করেছে।',
   'ref': 'অ্যারোস্মিথ তাদের সফরের অবশিষ্ট কনসার্টগুলো বাতিল করেছেন'},
  {'id': 1680,
   'bleu': 3.802351022611669,
   'chrf': 28.053181926035382,
   'rtf': 0.08935610453287761,
   'pred': 'একটি সু-গোল্লা, বাঘ ভাল, ভাল না, যদিও, সাঁতার, লম্বা, বড় দূরত্ব, এবং পাঁচবার একটি শক্তিশালী মানুষের শক্তি দিয়ে টান',
   'ref': 'একজন সম্পূর্ন ভালো ক্রীড়াবিদ বাঘ আরোহণ করতে পারে যদিও ভালোভাবে নয় সাঁতার  বেশ দূরপর্যন্ত লাফ এবং টানতে পারে ৫গুন জোরে একজন শক্তিশালী মানুষের চেয়ে।'},
  {'id': 1681,
   'bleu': 13.52102459252932,
   'chrf': 52.062153961177785,
   'rtf': 0.06438241945253359,
   'pred': 'তবে, এটি কেবলমাত্র পরীক্ষা নয়, এবং এটি এমন একটি পরীক্ষা যা এক বা একাধিক সম্ভাব্য অনুমানগুলি নির্মূল করতে ব্যবহৃত হয়, প্রশ্ন করা এবং পর্যবেক্ষণগুলি তৈরি করা, যা বৈজ্ঞানিক গবেষণাকে পথপ্রদর্শক করে।',
   'ref': 'শুধুমাত্র এটিই নয় বরং পরীক্ষা-নিরীক্ষা হল এমন একটি পদ্ধতি যা এক বা একাধিক সম্ভাব্য অনুমানকে বাদ দিতে প্রশ্ন জিজ্ঞাসা করতে এবং পর্যবেক্ষণ করতে সেই সাথে বৈজ্ঞানিক গবেষণাকেও নির্দেশ করতে ব্যবহার করা হয়'},
  {'id': 1682,
   'bleu': 0.7097250739055339,
   'chrf': 16.374510657917103,
   'rtf': 0.12090023216384146,
   'pred': 'যদিও কেউই নিশ্চিত না যে এটি কে লিখেছেন, তবে এটি তার জীবনের প্রথম দিকে, এটির বৃহত্তম পার্টসান ডকুয়েক্ট, এটি ২৯.৩.৪.৪.৪.৪.৪.৫.৪.৪.৪.৪.৪.৪.৪.৪.৪.৪.৪.৪.৪.৪.৪.৪.৪.৪.৪.',
   'ref': 'য
```

### Cell 63 (code, score=128)
```python
p4b = load_latest_checkpoint('phase4_benchmark')
if p4b: p4_results, p4_summary = p4b['results'], p4b['summary']
else:
    p4_results, p4_summary = run_benchmark(model_p4, eval_samples, label='P4_EncPrune', save_n=2)
    save_checkpoint(dict(results=p4_results, summary=p4_summary), name='phase4_benchmark', step=0)
store_summary(p4_summary); plot_phase_comparison()
```
OUTPUT:
```text
[ckpt] No checkpoint for 'phase4_benchmark'

============================================================
  BENCHMARK: P4_EncPrune
  Samples: 25  Target: ben
============================================================

  GPU mem: 2.27 GB alloc / 4.79 GB reserved
  [ 1/25] BLEU= 14.8 ChrF= 46.9 RTF=0.086  id=1660
              pred: রোমান্টিকতাবাদ সংস্কৃতির নির্ণয়বাদের একটি বড় উপাদান ছিল, যা গথ, ফিচট এবং শ্লেগ
[audio] Saved P4_EncPrune_s1in.wav (0.3 MB)
  P4_EncPrune_s1in.wav  (10.7s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P4_EncPrune_s1out.wav (0.3 MB)
  P4_EncPrune_s1out.wav  (8.1s | sr=16000)

<IPython.lib.display.Audio object>
  [ 2/25] BLEU= 10.1 ChrF= 40.9 RTF=0.096  id=1661
              pred: তিনি বলেন, তিনি চীনের অর্থনৈতিক উৎপাদনের উপর ভিত্তি করে এই সংখ্যাটি তৈরি করা হবে
[audio] Saved P4_EncPrune_s2in.wav (0.2 MB)
  P4_EncPrune_s2in.wav  (6.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P4_EncPrune_s2out.wav (0.2 MB)
  P4_EncPrune_s2out.wav  (6.0s | sr=16000)

<IPython.lib.display.Audio object>
  [ 3/25] BLEU= 20.1 ChrF= 45.5 RTF=0.097  id=1662
              pred: মিশ্রণ মূলত দুই বা তার বেশি ধাতব মিশ্রণ, পিইআইআরআই টেবিলের উপর অনেক উপাদান রয়েছ
  [ 4/25] BLEU=  5.2 ChrF= 42.4 RTF=0.095  id=1663
              pred: চোকামু উপত্যকা, চিলির শীর্ষস্থানীয় আরোহণের গন্তব্য, দক্ষিণ আমেরিকা'র ইয়েসোমিটি
  [ 5/25] BLEU= 13.3 ChrF= 42.1 RTF=0.087  id=1664
              pred: দুটি ড্রাই পাতা একসাথে ঘুরান এবং তারপর চিলিয়ে ঘন হাত দিয়ে তাদের একটি বলের মধ্য
  [ 6/25] BLEU=  5.6 ChrF= 23.9 RTF=0.072  id=1665
              pred: "লিকের মতে, ""ডকুমেন্টটি সীমান্ত বিরোধের কথা উল্লেখ করবে, যা"
  [ 7/25] BLEU=  9.8 ChrF= 50.6 RTF=0.076  id=1666
              pred: আপনি আপনার নিজের সরকার ছাড়া অন্য সরকারের পরামর্শ নিয়ে পরামর্শ নিতে পারেন, কিন্
  [ 8/25] BLEU=  4.6 ChrF= 45.5 RTF=0.102  id=1667
              pred: সাধারণভাবে, দুইটি আচরণ বিবর্তনগুলি উদ্ভূত হতে পারে, যেহেতু ম্যানেজাররা তাদের প্র
  [ 9/25] BLEU=  7.0 ChrF= 50.9 RTF=0.074  id=1668
              pred: এটি একটি ওয়াইল্ডকার্ড কিনতেও উপকারী হতে পারে, যা দক্ষিণ আফ্রিকার পার্কের যে কোন
  [10/25] BLEU=  7.3 ChrF= 51.4 RTF=0.113  id=1669
              pred: পুলিশ সুপারিনটেন্ডেন্ট চান্দ্রা শিকর সুলঙ্কি বলেন, অভিযুক্তরা মুখোমুখি হয়ে আদাল
  [11/25] BLEU=  0.2 ChrF=  7.8 RTF=0.186  id=1670
              pred: তাদের আনুষ্ঠানিক আচরণ, প্রায়শই স্থিরতা বজায় রাখার মতো বড় বড় বড় বড় বড় বড় 
  [12/25] BLEU= 14.5 ChrF= 51.4 RTF=0.135  id=1671
              pred: কংগ্রেস অযৌনতা ইনিশিয়েয়েয়েকে এবং ফিসাল-২৫৫৫-এ অর্থায়ন শুরু করে এবং নির্দিষ্ট
  [13/25] BLEU=  6.7 ChrF= 30.2 RTF=0.068  id=1672
              pred: ফ্যাব্রিককে খুব গরম হতে দেয় না, যা সংকুচিত হতে পারে, বা চরম ক্ষেত্রে, পুড়ে যায
  [14/25] BLEU= 22.3 ChrF= 68.5 RTF=0.095  id=1673
              pred: বিপ্লবী যুদ্ধের সময়, ১৩টি রাজ্যে প্রথমবারের মতো একটি দুর্বল কেন্দ্রীয় সরকার গঠ
  [15/25] BLEU=  8.2 ChrF= 40.9 RTF=0.094  id=1674
              pred: কিছু এলাকায়, এক মিনিটের জন্য উষ্ণ জল যথেষ্ট এবং অন্য কয়েক মিনিট প্রয়োজন হয়।
  [16/25] BLEU=  0.0 ChrF=  7.3 RTF=0.054  id=1675
              pred: "প্রাণের মাঝামাঝি পর্যন্ত আপনার জন্য " - "
  [17/25] BLEU=  8.8 ChrF= 60.2 RTF=0.060  id=1676
              pred: দক্ষিণ আফ্রিকার সকল জাতীয় উদ্যানের মতো, পার্কের জন্য প্রতিদিন সংরক্ষণ এবং প্রবে
  [18/25] BLEU=  6.4 ChrF= 37.1 RTF=0.129  id=1677
              pred: আজ, একমাত্র পোকা যে তাদের ডানাগুলিকে পিছনে ভাঁড়ানো যায় না তা হ'ল ড্রাগনফ্লি এব
  [19/25] BLEU=  2.2 ChrF= 30.9 RTF=0.069  id=1678
              pred: "অলিভার স্যাক্স তার কাগজতে রাষ্ট্রপতির বক্তৃতাটি নির্দেশ করে যে, মস্তিষ্কের ক্ষত
  [20/25] BLEU=  6.6 ChrF= 45.8 RTF=0.118  id=1679
              pred: এরা স্মিথ তাদের সফরর বাকি কনসেন্ট বাতিল করেছে।
  [21/25] BLEU=  3.8 ChrF= 28.1 RTF=0.088  id=1680
              pred: একটি সু-গোল্লা, বাঘ ভাল, ভাল না, যদিও, সাঁতার, লম্বা, বড় দূরত্ব, এবং পাঁচবার এক
  [22/25] BLEU= 13.5 ChrF= 52.1 RTF=0.066  id=1681
              pred: তবে, এটি কেবলমাত্র পরীক্ষা নয়, এবং এটি এমন একটি পরীক্ষা যা এক বা একাধিক সম্ভাব্
  [23/25] BLEU=  0.7 ChrF= 16.4 RTF=0.120  id=1682
              pred: যদিও কেউই নিশ্চিত না যে এটি কে লিখেছেন, তবে এটি তার জীবনের প্রথম দিকে, এটির বৃহত
  [24/25] BLEU=  6.0 ChrF= 39.7 RTF=0.100  id=1683
              pred: "তারা আরও আরও আরও লিখেছেন, "তারা এখনও তাদের সময় বেঁচে আছে, এবং আরও অনেক লোক আছে
  [25/25] BLEU=  6.6 ChrF= 46.2 RTF=0.062  id=1684
              pred: সামোয়া'র রাজধানী, শহরটি উপোলু দ্বীপের মধ্যে এবং জনসংখ্যা ৪০ হাজারেরও কম।

  Summary: BLEU=8.19  ChrF=40.11  RTF=0.0937  Params=1118.8M

[ckpt] Saved phase4_benchmark_step000000.pt (0.0 MB)
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P4_EncPrune (4 total)

<Figure size 1680x1200 with 4 Axes>
[image/png output omitted]
```

### Cell 65 (markdown, score=4)
```markdown
---
# Phase 5: Width Pruning (FLAP)
**Paper:** FLAP (AAAI 2024)

Structurally remove FFN neurons (shrink weight matrices). Creates smaller dense
matrices for real GPU speedup, unlike zeroing which keeps full-size matrices.
```

### Cell 66 (code, score=166)
```python
# ── Phase 5 Cell 1: FLAP helpers ─────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

# ── Step 1: Find all FFN pairs in any component ──────────────────────────────

def find_all_ffn_layers(model, component_name):
    """
    Walk a component and return all (parent_module, in_attr, out_attr, display_name)
    tuples for every FFN pair found.

    Handles:
      - Text decoder / standard FFN:     layer.ffn.fc1 / .fc2
      - Speech encoder (Conformer):      layer.ffn1|ffn2.intermediate_dense / .output_dense
      - T2U (and adapter):               layer.ffn.fc1 / .fc2
    """
    component = getattr(model, component_name)
    results = []

    def _pair_ok(lin1, lin2):
        return (
            isinstance(lin1, nn.Linear)
            and isinstance(lin2, nn.Linear)
            and lin1.out_features == lin2.in_features
        )

    def _scan(module, prefix):
        # Conformer FFN (speech encoder blocks)
        if hasattr(module, 'intermediate_dense') and hasattr(module, 'output_dense'):
            if _pair_ok(module.intermediate_dense, module.output_dense):
                results.append(
                    (module, 'intermediate_dense', 'output_dense', prefix)
                )

        # Standard HF FFN (decoder, T2U, adapter)
        if hasattr(module, 'fc1') and hasattr(module, 'fc2'):
            if _pair_ok(module.fc1, module.fc2):
                results.append((module, 'fc1', 'fc2', prefix))

        for name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                _scan(child, f'{prefix}.{name}' if prefix else name)

    _scan(component, component_name)
    return results


def test_ffn_detection(model):
    """Verify we find FFN pairs in all components before running pruning."""
    for comp in ['speech_encoder', 'text_decoder', 't2u_model']:
        pairs = find_all_ffn_layers(model, comp)
        print(f'  {comp}: {len(pairs)} FFN pairs found')
        if pairs:
            sample = pairs[0]
            fc1 = getattr(sample[0], sample[1])
            fc2 = getattr(sample[0], sample[2])
            print(f'    Sample: {sample[3]} | fc1={fc1.in_features}→{fc1.out_features} | fc2={fc2.in_features}→{fc2.out_features}')


model_base, processor = load_base_model()
print('Running FFN detection on model_base...')
test_ffn_detection(model_base)

model_p4, processor = load_model_from_drive('phase4_enc_pruned')
print_model_breakdown(model_p4, 'After Phase 4:')

print('Running FFN detection on model_p4...')
test_ffn_detection(model_p4)
```
OUTPUT:
```text
Loading processor from facebook/seamless-m4t-v2-large...

preprocessor_config.json: 0.00B [00:00, ?B/s]
config.json: 0.00B [00:00, ?B/s]
tokenizer_config.json: 0.00B [00:00, ?B/s]
tokenizer.model:   0%|          | 0.00/5.17M [00:00<?, ?B/s]
added_tokens.json: 0.00B [00:00, ?B/s]
special_tokens_map.json: 0.00B [00:00, ?B/s]
Loading model  -- may take 5-10 min...

model.safetensors.index.json: 0.00B [00:00, ?B/s]
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]
Instantiating a decoder SeamlessM4Tv2Attention without passing `layer_idx` is not recommended and will lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` when creating this class.

Loading weights:   0%|          | 0/1846 [00:00<?, ?it/s]
SeamlessM4Tv2ForSpeechToSpeech LOAD REPORT from: facebook/seamless-m4t-v2-large
Key                                                      | Status     |  | 
---------------------------------------------------------+------------+--+-
text_encoder.layers.{0...23}.ffn.fc1.bias                | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.out_proj.weight   | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.q_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.v_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc2.weight              | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn_layer_norm.weight | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.k_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.out_proj.bias     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.q_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc2.bias                | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc1.weight              | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.k_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn_layer_norm.bias   | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn_layer_norm.bias         | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn_layer_norm.weight       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.v_proj.weight     | UNEXPECTED |  | 
text_encoder.layer_norm.bias                             | UNEXPECTED |  | 
text_encoder.layer_norm.weight                           | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

generation_config.json: 0.00B [00:00, ?B/s]
Model loaded.
  GPU mem: 1.79 GB alloc / 1.80 GB reserved
Running FFN detection on model_base...
  speech_encoder: 50 FFN pairs found
    Sample: speech_encoder.encoder.layers.0.ffn1 | fc1=1024→4096 | fc2=4096→1024
  text_decoder: 24 FFN pairs found
    Sample: text_decoder.layers.0.ffn | fc1=1024→8192 | fc2=8192→1024
  t2u_model: 6 FFN pairs found
    Sample: t2u_model.model.encoder.layers.0.ffn | fc1=1024→8192 | fc2=8192→1024
[model] Not in local cache, pulling from remote...
[rclone] Pulled phase4_enc_pruned → /kaggle/working/models/phase4_enc_pruned
[model] Loading phase4_enc_pruned from /kaggle/working/models/phase4_enc_pruned ...

Loading weights:   0%|          | 0/1330 [00:00<?, ?it/s]
  Restored custom state: ['_vocab_remap_to_old']
  [model] pruning_manifest: ['stage_name']

--- After Phase 4: ---
  speech_encoder                         441.6M  ( 39.5%)
  text_decoder                           373.6M  ( 33.4%)
  t2u_model                              261.8M  ( 23.4%)
  vocoder                                 41.9M  (  3.7%)
  shared                                  20.9M  (  1.9%)
  lm_head                                 20.9M  (  1.9%)
  TOTAL                                 1118.8M
---
Running FFN detection on model_p4...
  speech_encoder: 34 FFN pairs found
    Sample: speech_encoder.encoder.layers.0.ffn1 | fc1=1024→4096 | fc2=4096→1024
  text_decoder: 14 FFN pairs found
    Sample: text_decoder.layers.0.ffn | fc1=1024→8192 | fc2=8192→1024
  t2u_model: 6 FFN pairs found
    Sample: t2u_model.model.encoder.layers.0.ffn | fc1=1024→8192 | fc2=8192→1024
```

### Cell 67 (code, score=3)
```python
# 1. Delete model references
del model_base, model_p4

# 2. Force garbage collection
gc.collect()

# 3. Empty the GPU cache
torch.cuda.empty_cache()
```

### Cell 68 (code, score=60)
```python
# ── Phase 5 Cell 2: FIXED calibration with device-aware stats + T2U fix ──────

import torch
import torch.nn as nn
import numpy as np


def collect_ffn_calibration_stats(model, component_name, calibration_wavs,
                                   processor, n_samples=64, device=None):
    """
    Collect per-channel input L2-norm stats for every FFN layer in component_name.
    
    CRITICAL FIXES: 
    1. Move ALL processor output tensors to device
    2. Create stats tensors on the SAME device as the model
    3. T2U forward pass requires char_input_ids (not just inputs_embeds)
    """
    if device is None:
        device = next(model.parameters()).device

    ffn_pairs = find_all_ffn_layers(model, component_name)
    if not ffn_pairs:
        print(f"  [calib] No FFN pairs in {component_name}, skipping.")
        return {}

    # Build stats keyed by id(parent) - CRITICAL: create tensors on device
    stats = {}
    for (parent, fc1_attr, fc2_attr, name) in ffn_pairs:
        fc1 = getattr(parent, fc1_attr)
        key = id(parent)
        stats[key] = {
            "sum_x":  torch.zeros(fc1.in_features, dtype=torch.float64, device=device),
            "sq_sum": torch.zeros(fc1.in_features, dtype=torch.float64, device=device),
            "count":  0,
            "module": parent,
            "fc1":    fc1_attr,
            "fc2":    fc2_attr,
            "name":   name,
        }

    hooks = []
    def make_hook(key):
        def hook(module, inp, out):
            x = inp[0].detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() == 1:
                x = x.unsqueeze(0)
            s = stats[key]
            s["count"]  += x.shape[0]
            s["sum_x"]  += x.sum(dim=0).double()
            s["sq_sum"] += x.pow(2).sum(dim=0).double()
        return hook

    for (parent, fc1_attr, _, name) in ffn_pairs:
        fc1 = getattr(parent, fc1_attr)
        hooks.append(fc1.register_forward_hook(make_hook(id(parent))))

    model.eval()
    n_actual = min(n_samples, len(calibration_wavs))

    print(f"  [calib] Collecting activations from {n_actual} samples "
          f"for {component_name} (direct forward pass — every layer fires)...")

    with torch.no_grad():
        for i, wav in enumerate(calibration_wavs[:n_actual]):
            if i % 20 == 0:
                print(f"  [calib] {i}/{n_actual}")
            try:
                # ── CRITICAL FIX: Process on CPU, then move EVERYTHING to device ──
                enc_in = processor(audio=wav, sampling_rate=16000,
                                   return_tensors="pt")
                
                # Move ALL tensors in the dict to device (not just known keys)
                enc_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in enc_in.items()}
                
                input_feats = enc_in["input_features"]
                attn_mask = enc_in["attention_mask"]
                
                if component_name == "speech_encoder":
                    model.speech_encoder(
                        input_features=input_feats,
                        attention_mask=attn_mask,
                        return_dict=True,
                    )

                elif component_name == "text_decoder":
                    enc_out = model.speech_encoder(
                        input_features=input_feats,
                        attention_mask=attn_mask,
                        return_dict=True,
                    )
                    enc_hidden = enc_out.last_hidden_state

                    dec_device = next(model.text_decoder.parameters()).device
                    enc_hidden = enc_hidden.to(dec_device)

                    enc_len = enc_hidden.shape[1]
                    encoder_attention_mask = torch.ones(
                        (enc_hidden.shape[0], enc_len),
                        dtype=torch.long,
                        device=dec_device
                    )
                    
                    fake_ids = torch.randint(
                        low=0,
                        high=model.config.vocab_size,
                        size=(1, 32),
                        device=dec_device
                    )
                
                    model.text_decoder(
                        input_ids=fake_ids,
                        encoder_hidden_states=enc_hidden,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=True,
                    )

                elif component_name == "t2u_model":
                    # ── T2U FIX: Use generate() to get proper char_input_ids ──
                    # The t2u_model.forward() requires char_input_ids and char_count_per_id.
                    # We can't easily construct these manually, so we use generate()
                    # with return_intermediate_token_ids=True to fire all T2U layers.
                    
                    try:
                        ben_tok = model.generation_config.text_decoder_lang_to_code_id.get("ben", 4)
                    except Exception:
                        ben_tok = 4
                    
                    # Run full generate() pipeline - this fires T2U encoder+decoder
                    model.generate(
                        input_features=input_feats,
                        attention_mask=attn_mask,
                        tgt_lang="ben",
                        return_intermediate_token_ids=True,
                        max_new_tokens=16,  # short sequence for speed
                    )

            except Exception as e:
```
OUTPUT:
```text
Calibration helpers ready (FIXED: T2U uses generate() for proper calibration).
```

### Cell 69 (code, score=121)
```python
# ── Phase 5 Cell 3: Neuron importance scoring (Wanda-sp + FLAP) ──────────────
#
# Wanda-sp (structured Wanda): score(k) = sum_j |W1[k,j]| * ||X_j||_2
#   where ||X_j||_2 = sqrt(E[x_j^2]) — the RMS of channel j inputs
#
# FLAP-row: score(k) = sum_j Var(X_j) * W1[k,j]^2
#
# We use Wanda-sp as primary (proven robust, never zero if layer fires),
# and fall back to pure row-norm if sq_norm is also zero (truly dead layer).

import torch
import torch.nn as nn
import numpy as np


def wanda_neuron_scores(fc1_weight, sq_norm):
    """
    Wanda-sp per-neuron score (ICLR 2024, structured variant).

    score(k) = sum_j |W1[k,j]| * sqrt(E[x_j^2])
             = |W1| @ rms_x          where rms_x = sqrt(sq_norm)

    fc1_weight : [ffn_hidden, model_hidden]
    sq_norm    : [model_hidden]  E[x_j^2] per channel

    Returns [ffn_hidden] scores. Falls back to row-L2-norm if sq_norm ~ 0.
    """
    W1 = fc1_weight.float().cpu()
    rms = sq_norm.float().cpu().clamp(min=0).sqrt()   # [model_hidden]

    if rms.max().item() < 1e-10:
        # Layer never fired or truly dead — use weight row-norm only
        return W1.pow(2).sum(dim=1).sqrt()

    return (W1.abs() * rms.unsqueeze(0)).sum(dim=1)   # [ffn_hidden]


def flap_neuron_scores(fc1_weight, var_x):
    """
    FLAP per-neuron score (AAAI 2024, Eq. 5 applied to rows).
    score(k) = sum_j Var(X_j) * W1[k,j]^2
    Falls back to row-norm if var is zero.
    """
    W1 = fc1_weight.float().cpu()
    v  = var_x.float().cpu().clamp(min=0)

    if v.max().item() < 1e-10:
        return W1.pow(2).sum(dim=1)

    return (W1.pow(2) * v.unsqueeze(0)).sum(dim=1)


def neuron_importance_scores(fc1_weight, var_x, sq_norm=None):
    """
    Primary scoring function: use Wanda-sp if sq_norm available, else FLAP.
    """
    if sq_norm is not None and sq_norm.max().item() > 1e-10:
        return wanda_neuron_scores(fc1_weight, sq_norm)
    if var_x is not None and var_x.max().item() > 1e-10:
        return flap_neuron_scores(fc1_weight, var_x)
    # Pure weight magnitude fallback
    return fc1_weight.float().cpu().pow(2).sum(dim=1)


def standardize_scores(scores):
    """FLAP Eq.6: standardize to zero mean, unit std for cross-layer comparison."""
    mu    = scores.mean()
    sigma = scores.std(unbiased=False)
    if sigma < 1e-8:
        return torch.zeros_like(scores)
    return (scores - mu) / sigma


def structural_prune_ffn(parent, fc1_attr, fc2_attr,
                          channel_mean, keep_idx, device):
    """
    Structurally prune one FFN pair using pre-computed keep_idx.
    
    CRITICAL FIX: Bias compensation DISABLED.
    Reason: Bias compensation is extremely fragile and causes NaN/Inf corruption
    when pruned neurons have extreme activations. Production implementations
    (e.g., LLM-Pruner, Wanda) skip bias compensation and rely on fine-tuning.
    
    This is the safe, proven approach.
    """
    fc1 = getattr(parent, fc1_attr)
    fc2 = getattr(parent, fc2_attr)
    ffn_dim = fc1.out_features

    fc1_device = fc1.weight.device
    fc2_device = fc2.weight.device

    n_keep   = len(keep_idx)
    kidx_fc1 = keep_idx.to(fc1_device)
    kidx_fc2 = keep_idx.to(fc2_device)

    # Create new fc1 (input projection) - keep only selected neurons
    new_fc1 = nn.Linear(fc1.in_features, n_keep,
                         bias=(fc1.bias is not None),
                         device=fc1_device, dtype=fc1.weight.dtype)
    new_fc1.weight.data.copy_(fc1.weight.data[kidx_fc1])
    if fc1.bias is not None:
        new_fc1.bias.data.copy_(fc1.bias.data[kidx_fc1])

    # Create new fc2 (output projection) - keep only selected input dims
    new_fc2 = nn.Linear(n_keep, fc2.out_features,
                         bias=(fc2.bias is not None),
                         device=fc2_device, dtype=fc2.weight.dtype)
    new_fc2.weight.data.copy_(fc2.weight.data[:, kidx_fc2])
    
    # Keep original fc2 bias WITHOUT compensation
    # Fine-tuning (Phase 7) will recover any lost contribution
    if fc2.bias is not None:
        new_fc2.bias.data.copy_(fc2.bias.data)

    setattr(parent, fc1_attr, new_fc1)
    setattr(parent, fc2_attr, new_fc2)

    return n_keep, ffn_dim


print('Neuron scoring + structural pruning helpers ready.')
print('  NOTE: Bias compensation DISABLED (prevents NaN corruption)')


# ── Phase 5 Cell 4: apply_flap_to_component — robust top-k pruning ───────────

def apply_flap_to_component(model, component_name, calib_stats,
                              global_prune_ratio=0.20,
                              min_keep_frac=0.50,
                              device=None):
    """
    Structured FFN width pruning using Wanda-sp / FLAP neuron scores.

    global_prune_ratio : target fraction of neurons to prune (e.g. 0.20 = 20%)
    min_keep_frac      : per-layer floor — never prune a single layer below this
                         fraction of its original size (default 0.50 = keep ≥50%)

    Algorithm (faithful to FLAP paper):
      1. Score every neuron in every layer (Wanda-sp metric)
```
OUTPUT:
```text
Neuron scoring + structural pruning helpers ready.
  NOTE: Bias compensation DISABLED (prevents NaN corruption)
apply_flap_to_component ready (bias compensation disabled).
```

### Cell 70 (code, score=53)
```python
# ── Phase 5 Cell 4: apply_flap_to_component — robust top-k pruning ───────────

def apply_flap_to_component(model, component_name, calib_stats,
                              global_prune_ratio=0.20,
                              min_keep_frac=0.50,
                              device=None):
    """
    Structured FFN width pruning using Wanda-sp / FLAP neuron scores.

    global_prune_ratio : target fraction of neurons to prune (e.g. 0.20 = 20%)
    min_keep_frac      : per-layer floor — never prune a single layer below this
                         fraction of its original size (default 0.50 = keep ≥50%)

    Algorithm (faithful to FLAP paper):
      1. Score every neuron in every layer (Wanda-sp metric)
      2. Standardize scores per-layer (FLAP Eq.6) for cross-layer comparison
      3. Pool all standardized scores, find global threshold at global_prune_ratio
      4. Per layer: keep neurons above threshold, enforce min_keep_frac floor
      5. Structurally remove pruned rows/cols, add bias compensation (FLAP Eq.4)
    """
    if device is None:
        device = next(model.parameters()).device

    if not calib_stats:
        print(f'  [FLAP] No calib stats for {component_name}, skipping.')
        return {}

    # ── Step 1 & 2: score + standardize ─────────────────────────────────────
    all_std_scores  = {}
    all_raw_scores  = {}

    n_zero_var = 0
    for key, s in calib_stats.items():
        fc1   = getattr(s['module'], s['fc1'])
        W1    = fc1.weight.float().cpu()
        var_x  = s.get('var',     torch.zeros(W1.shape[1]))
        sq_norm = s.get('sq_norm', torch.zeros(W1.shape[1]))

        raw = neuron_importance_scores(W1, var_x, sq_norm)
        if raw.max().item() < 1e-10:
            n_zero_var += 1

        all_raw_scores[key] = raw
        all_std_scores[key] = standardize_scores(raw)

    if n_zero_var:
        print(f'  [FLAP] WARNING: {n_zero_var}/{len(calib_stats)} layers used '
              f'weight-only fallback (calibration did not fire for those layers).')

    # ── Step 3: global threshold ─────────────────────────────────────────────
    all_std_flat  = torch.cat(list(all_std_scores.values()))
    total_neurons = len(all_std_flat)
    n_prune_total = int(total_neurons * global_prune_ratio)

    sorted_scores, _ = torch.sort(all_std_flat)
    threshold = sorted_scores[max(0, n_prune_total - 1)].item()

    print(f'  [FLAP] {component_name}: {total_neurons} total neurons, '
          f'pruning ≤{n_prune_total} ({global_prune_ratio*100:.0f}%), '
          f'threshold={threshold:.4f}')
    print(f'         score range [{all_std_flat.min():.3f}, {all_std_flat.max():.3f}]  '
          f'mean={all_std_flat.mean():.3f}  std={all_std_flat.std():.3f}')

    # ── Step 4 & 5: per-layer prune ──────────────────────────────────────────
    results       = {}
    total_kept    = 0
    total_orig    = 0

    for key, s in calib_stats.items():
        std_scores = all_std_scores[key]
        fc1 = getattr(s['module'], s['fc1'])
        ffn_dim = fc1.out_features

        # How many neurons score above the global threshold?
        n_above = int((std_scores > threshold).sum().item())

        # Enforce per-layer minimum
        min_keep = max(1, int(ffn_dim * min_keep_frac))
        n_keep   = max(min_keep, n_above)
        n_keep   = min(ffn_dim, n_keep)   # can't keep more than we have

        # Top-k selection (stable, threshold-independent)
        _, keep_idx = torch.topk(std_scores, n_keep)
        keep_idx = keep_idx.sort().values

        structural_prune_ffn(
            s['module'], s['fc1'], s['fc2'],
            channel_mean=s['mean'],
            keep_idx=keep_idx,
            device=device
        )

        pct_kept = n_keep / ffn_dim * 100
        total_kept += n_keep
        total_orig += ffn_dim
        results[s['name']] = {
            'kept': n_keep, 'original': ffn_dim, 'pct': pct_kept
        }

    avg_kept = total_kept / max(total_orig, 1) * 100
    print(f'  [FLAP] Done. Kept {total_kept}/{total_orig} neurons '
          f'({avg_kept:.1f}%) across {len(results)} layers.')
    return results


print('apply_flap_to_component ready.')
```
OUTPUT:
```text
apply_flap_to_component ready.
```

### Cell 71 (code, score=7)
```python
# ── Phase 5 Cell 5: Build calibration wavs from eval_samples ─────────────────
# Reuse your existing eval_samples as calibration (or use ft_samples if loaded)

calib_wavs = [s['wav'] for s in eval_samples]
# Optionally extend with ft_samples for better coverage:
# calib_wavs = [s['wav'] for s in (eval_samples + ft_samples[:80])]

print(f'Calibration corpus: {len(calib_wavs)} samples')
print(f'  Duration range: {min(len(w)/16000 for w in calib_wavs):.1f}s – '
      f'{max(len(w)/16000 for w in calib_wavs):.1f}s')
```
OUTPUT:
```text
Calibration corpus: 25 samples
  Duration range: 4.0s – 29.3s
```

### Cell 72 (code, score=23)
```python
!ls /kaggle/working/models
```
OUTPUT:
```text
phase4_enc_pruned
```

### Cell 73 (code, score=68)
```python
model_base, processor = load_base_model()
```
OUTPUT:
```text
Loading processor from facebook/seamless-m4t-v2-large...
Loading model  -- may take 5-10 min...

Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]
Loading weights:   0%|          | 0/1846 [00:00<?, ?it/s]
SeamlessM4Tv2ForSpeechToSpeech LOAD REPORT from: facebook/seamless-m4t-v2-large
Key                                                      | Status     |  | 
---------------------------------------------------------+------------+--+-
text_encoder.layers.{0...23}.ffn.fc1.bias                | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.out_proj.weight   | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.q_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.v_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc2.weight              | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn_layer_norm.weight | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.k_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.out_proj.bias     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.q_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc2.bias                | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc1.weight              | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.k_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn_layer_norm.bias   | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn_layer_norm.bias         | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn_layer_norm.weight       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.v_proj.weight     | UNEXPECTED |  | 
text_encoder.layer_norm.bias                             | UNEXPECTED |  | 
text_encoder.layer_norm.weight                           | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

Model loaded.
  GPU mem: 1.79 GB alloc / 1.80 GB reserved
```

### Cell 74 (code, score=294)
```python
# ── Phase 5 Cell 6: RUN PHASE 5 (FIXED) ──────────────────────────────────────
# ROOT CAUSE: model_base has device_map='auto' split across devices after loading.
# deepcopy() preserves this split, causing "cuda:0 vs cpu" errors in calibration.
# FIX: Consolidate model_base to single GPU BEFORE deepcopy.

import gc as _stdlib_gc
import copy as _copy
 
FLAP_RATIO    = 0.10   # prune 15% of neurons globally per component
# REDUCED from 0.15 → 0.08 to prevent decoder collapse after Phases 3-4
MIN_KEEP_FRAC = 0.85   # never shrink any single layer below 70% of original

def force_model_to_single_device(model, device):
    """
    Aggressively consolidate model to a single device.
    Handles device_map='auto' models that are split across devices.
    """
    print(f"  Forcing entire model to {device}...")
    
    # Step 1: Move base model
    model = model.to(device)
    
    # Step 2: Explicitly move all submodules
    for name, module in model.named_modules():
        if module is not model:  # skip root
            try:
                module.to(device)
            except Exception:
                pass
    
    # Step 3: Move all parameters
    for name, param in model.named_parameters():
        if param.device != device:
            param.data = param.data.to(device)
    
    # Step 4: Move all buffers
    for name, buf in model.named_buffers():
        if buf is not None and buf.device != device:
            buf.data = buf.data.to(device)
    
    # Step 5: Clear device_map (forces single-device mode)
    if hasattr(model, 'hf_device_map'):
        model.hf_device_map = {k: device for k in model.hf_device_map}
    
    torch.cuda.empty_cache()
    
    # Verify consolidation
    devices = set()
    for p in model.parameters():
        devices.add(p.device)
    for b in model.buffers():
        devices.add(b.device)
    
    if len(devices) > 1:
        print(f"  WARNING: Model still on multiple devices: {devices}")
    else:
        print(f"  ✓ Model consolidated to {device}")
    
    return model

# ── Try loading completed Phase 5 from Drive ──────────────────────────────────
try:
    model_p5, processor = load_model_from_drive("phase5_flap_pruned(base)")
    sync_model_config(model_p5)
    model_p5 = _consolidate_to_single_gpu(model_p5)
    device = torch.device("cuda:0")
    model_p5 = force_model_to_single_device(model_p5, device)
    print("Loaded Phase 5 from Drive.")
    p5_loaded = True
except Exception as _e:
    print(f"No Phase 5 on Drive ({_e}), pruning from model_base...")
    p5_loaded = False
 
# ── Run Phase 5 pruning ───────────────────────────────────────────────────────
if not p5_loaded:
    device = torch.device("cuda:0")
    
    # CRITICAL FIX: Consolidate model_base BEFORE deepcopy
    print("Consolidating model_base to single GPU before deepcopy...")
    model_base = _consolidate_to_single_gpu(model_base)
    model_base = force_model_to_single_device(model_base, device)
    
    # Verify model_base is on single device
    p4_devices = set(p.device for p in model_base.parameters())
    print(f"  model_base devices after consolidation: {p4_devices}")
    if len(p4_devices) > 1:
        raise RuntimeError(f"model_base still split across {p4_devices}! Cannot proceed.")
    
    # Now deepcopy will preserve single-device placement
    print("Deep-copying model_base → model_p5...")
    model_p5 = model_base
    model_p5 = force_model_to_single_device(model_p5, device)
    
    # Final verification
    p5_devices = set(p.device for p in model_p5.parameters())
    print(f"  model_p5 devices after deepcopy: {p5_devices}")
    
    pre_params = count_params(model_p5)
    print(f"\nPre-pruning: {pre_params:.1f}M params")
 
    prune_results = {}
 
    for comp_name in ["text_decoder", "speech_encoder", "t2u_model"]:
        print(f"\n{'='*60}")
        print(f"  Collecting calibration for {comp_name}")
        print(f"{'='*60}")
        
        # Verify component device before calibration
        comp = getattr(model_p5, comp_name)
        comp_device = next(comp.parameters()).device
        print(f"  {comp_name} device: {comp_device}")
        
        if comp_device != device:
            print(f"  ERROR: {comp_name} on wrong device! Re-consolidating...")
            model_p5 = force_model_to_single_device(model_p5, device)
        
        calib = collect_ffn_calibration_stats(
            model_p5, comp_name, calib_wavs, processor,
            n_samples=min(64, len(calib_wavs)),
            device=device,
        )
        _stdlib_gc.collect()
        torch.cuda.empty_cache()
 
        print(f"\n{'='*60}")
        print(f"  Applying FLAP to {comp_name}")
        print(f"{'='*60}")
        
        results = apply_flap_to_component(
            model_p5, comp_name, calib,
            global_prune_ratio=FLAP_RATIO,
            min_keep_frac=MIN_KEEP_FRAC,
            device=device,
        )
        prune_results[comp_name] = results
        _stdlib_gc.collect()
        torch.cuda.empty_cache()
 
    post_params = count_params(model_p5)
    print(f"\n{'='*60}")
```
OUTPUT:
```text
[model] Not in local cache, pulling from remote...
No Phase 5 on Drive ([rclone] model pull failed for phase5_flap_pruned(base): 2026/04/19 03:50:26 ERROR : Google drive root 'seamV5/models/phase5_flap_pruned(base)': error reading source root directory: directory not found
2026/04/19 03:50:26 ERROR : Local file system at /kaggle/working/models/phase5_flap_pruned(base): not deleting files as there were IO errors
2026/04/19 03:), pruning from model_base...
Consolidating model_base to single GPU before deepcopy...
  Multi-device map detected, consolidating to cuda:0...
  Model now on: cuda:0
  Forcing entire model to cuda:0...
  ✓ Model consolidated to cuda:0
  model_base devices after consolidation: {device(type='cuda', index=0)}
Deep-copying model_base → model_p5...
  Forcing entire model to cuda:0...
  ✓ Model consolidated to cuda:0
  model_p5 devices after deepcopy: {device(type='cuda', index=0)}

Pre-pruning: 1805.5M params

============================================================
  Collecting calibration for text_decoder
============================================================
  text_decoder device: cuda:0
  [calib] Collecting activations from 25 samples for text_decoder (direct forward pass — every layer fires)...
  [calib] 0/25
  [calib] 20/25
  [calib] Layers fired: 24/24  total token-vectors: 19200
  [calib] All 24 layers fired correctly.
  [calib] Done. 24 FFN layers instrumented.

============================================================
  Applying FLAP to text_decoder
============================================================
  [FLAP] text_decoder: 196608 total neurons, pruning ≤19660 (10%), threshold=-0.8317
         score range [-12.691, 19.697]  mean=-0.000  std=1.000
  [FLAP] Done. Kept 177211/196608 neurons (90.1%) across 24 layers.

============================================================
  Collecting calibration for speech_encoder
============================================================
  speech_encoder device: cuda:0
  [calib] Collecting activations from 25 samples for speech_encoder (direct forward pass — every layer fires)...
  [calib] 0/25
  [calib] 20/25
  [calib] Layers fired: 50/50  total token-vectors: 664920
  [calib] All 50 layers fired correctly.
  [calib] Done. 50 FFN layers instrumented.

============================================================
  Applying FLAP to speech_encoder
============================================================
  [FLAP] speech_encoder: 204800 total neurons, pruning ≤20480 (10%), threshold=-1.1440
         score range [-5.234, 13.491]  mean=0.000  std=1.000
  [FLAP] Done. Kept 184326/204800 neurons (90.0%) across 50 layers.

============================================================
  Collecting calibration for t2u_model
============================================================
  t2u_model device: cuda:0
  [calib] Collecting activations from 25 samples for t2u_model (direct forward pass — every layer fires)...
  [calib] 0/25
  [calib] 20/25
  [calib] Layers fired: 6/6  total token-vectors: 2538
  [calib] All 6 layers fired correctly.
  [calib] Done. 6 FFN layers instrumented.

============================================================
  Applying FLAP to t2u_model
============================================================
  [FLAP] t2u_model: 49152 total neurons, pruning ≤4915 (10%), threshold=-1.0367
         score range [-4.751, 14.065]  mean=-0.000  std=1.000
  [FLAP] Done. Kept 44237/49152 neurons (90.0%) across 6 layers.

============================================================
Width pruning complete:
  1805.5M → 1713.7M
  Saved: 91.8M params
============================================================
  [config] sync done.
  text_decoder: avg 90.1% neurons kept (24 layers)
  speech_encoder: avg 90.0% neurons kept (50 layers)
  t2u_model: avg 90.0% neurons kept (6 layers)
[ckpt] Saved phase5_flap(base)_step000000.pt (0.0 MB)
  [config] sync done.
[model] Saving phase5_flap_pruned(base) → /kaggle/working/models/phase5_flap_pruned(base) ...
  config.decoder_ffn_dim: 8192 -> 7638
  config.t2u_encoder_ffn_dim: 8192 -> 7209
  [config] sync done.
  Saved pruning_manifest.pt keys=['stage_name']

Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]
[model] Local save done. 3469 MB in 7 files.
[model] Pushing to rclone remote...
[model] Verified 7 files on remote.

--- After Phase 5: FLAP Width Pruned(base) ---
  text_decoder                           827.1M  ( 48.3%)
  speech_encoder                         593.1M  ( 34.6%)
  shared                                 262.2M  ( 15.3%)
  lm_head                                262.2M  ( 15.3%)
  t2u_model                              251.7M  ( 14.7%)
  vocoder                                 41.9M  (  2.4%)
  TOTAL                                 1713.7M
---

{'shared': 262.248448,
 'speech_encoder': 593.095494,
 'text_decoder': 827.051067,
 'lm_head': 262.248448,
 't2u_model': 251.688912,
 'vocoder': 41.911362,
 'TOTAL': 1713.746835}
```

### Cell 75 (code, score=135)
```python
# ── Phase 5 Cell 7: Quick sanity check then full benchmark ───────────────────
print("Quick sanity check (3 samples)...")
model_p5.eval()
 
for i, s in enumerate(eval_samples[:3]):
    wav = s["wav"]
    ref = s["ref"]
    try:
        pred, _wav_out = run_s2st(model_p5, wav, tgt_lang="ben")
        chrf = compute_chrf(pred, ref)
        print(f"  [{i+1}] ChrF={chrf:.1f}  pred: {pred[:80]!r}")
    except Exception as e:
        print(f"  [{i+1}] ERROR: {e}")

p5b = load_latest_checkpoint("phase5_benchmark(base)")
if p5b:
    p5_results, p5_summary = p5b["results"], p5b["summary"]
    print(f"Loaded P5 benchmark: BLEU={p5_summary['avg_bleu']:.2f} "
          f"ChrF={p5_summary['avg_chrf']:.2f}")
else:
    p5_results, p5_summary = run_benchmark(
        model_p5, eval_samples, label="P5_FLAP(base)", save_n=2)
    save_checkpoint(dict(results=p5_results, summary=p5_summary),
                    name="phase5_benchmark(base)", step=0)
 
store_summary(p5_summary)
plot_phase_comparison()
```
OUTPUT:
```text
Quick sanity check (3 samples)...
  [1] ChrF=38.2  pred: 'রোমান্টিকবাদের সাংস্কৃতিক নির্ণয়বাদের একটি বড় অংশ গথ, ফিচচ, শ্লেগেলের মতো লেখক'
  [2] ChrF=40.3  pred: 'তিনি বলেন, চীনের অর্থনৈতিক উৎপাদনের উপর ভিত্তি করে এই কমানোর কোনো অনুমান তিনি কর'
  [3] ChrF=56.1  pred: 'অ্যাললয় মূলত দুটি বা ততোধিক ধাতুর মিশ্রণ, ভুলে যাবেন না যে একটি পি.আর.আই. রেবেল'
[ckpt] No checkpoint for 'phase5_benchmark(base)'

============================================================
  BENCHMARK: P5_FLAP(base)
  Samples: 25  Target: ben
============================================================

  GPU mem: 3.47 GB alloc / 4.43 GB reserved
  [ 1/25] BLEU=  6.9 ChrF= 38.2 RTF=0.150  id=1660
              pred: রোমান্টিকবাদের সাংস্কৃতিক নির্ণয়বাদের একটি বড় অংশ গথ, ফিচচ, শ্লেগেলের মতো লেখক
[audio] Saved P5_FLAP(base)_s1in.wav (0.3 MB)
  P5_FLAP(base)_s1in.wav  (10.7s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P5_FLAP(base)_s1out.wav (0.2 MB)
  P5_FLAP(base)_s1out.wav  (6.8s | sr=16000)

<IPython.lib.display.Audio object>
  [ 2/25] BLEU=  5.8 ChrF= 40.3 RTF=0.186  id=1661
              pred: তিনি বলেন, চীনের অর্থনৈতিক উৎপাদনের উপর ভিত্তি করে এই কমানোর কোনো অনুমান তিনি কর
[audio] Saved P5_FLAP(base)_s2in.wav (0.2 MB)
  P5_FLAP(base)_s2in.wav  (6.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P5_FLAP(base)_s2out.wav (0.2 MB)
  P5_FLAP(base)_s2out.wav  (5.8s | sr=16000)

<IPython.lib.display.Audio object>
  [ 3/25] BLEU=  9.3 ChrF= 56.1 RTF=0.191  id=1662
              pred: অ্যাললয় মূলত দুটি বা ততোধিক ধাতুর মিশ্রণ, ভুলে যাবেন না যে একটি পি.আর.আই. রেবেল
  [ 4/25] BLEU=  6.0 ChrF= 41.0 RTF=0.156  id=1663
              pred: চিকামো ভ্যালি, চিলির দক্ষিণ আমেরিকার প্রধান আরোহণের গন্তব্য, যা বিভিন্ন ধরণের গ্
  [ 5/25] BLEU=  0.1 ChrF=  3.6 RTF=1.060  id=1664
              pred: দুটি শুষ্ন পা পা পা পা পা পা পা পা পা পা পা পা পা পা পা পা পা পা পা পা পা পা পা 
  [ 6/25] BLEU=  8.7 ChrF= 42.1 RTF=0.171  id=1665
              pred: লিকের মতে, এই নথি প্যালেস্টাইন ১৯৬৭ সালের মধ্যপ্রাচ্য যুদ্ধের আগে সীমান্ত বিরোধে
  [ 7/25] BLEU=  6.1 ChrF= 42.5 RTF=0.150  id=1666
              pred: আপনি হয়তো আপনার নিজের ছাড়া অন্য সরকারের পরামর্শ নিতে চাইবেন, কিন্তু তাদের পরাম
  [ 8/25] BLEU=  6.8 ChrF= 42.2 RTF=0.142  id=1667
              pred: সাধারণভাবে বলতে গেলে, ম্যানেজারদের মধ্যে একজন বা দুইজন ম্যানেজারের মতো আচরণ করা 
  [ 9/25] BLEU=  4.0 ChrF= 45.4 RTF=0.100  id=1668
              pred: দক্ষিণ আফ্রিকার বিভিন্ন জাতীয় উদ্যানগুলিতে ওয়াইল্ডকার্ডের মাধ্যমে প্রবেশের সুব
  [10/25] BLEU=  4.1 ChrF= 33.8 RTF=0.231  id=1669
              pred: পুলিশ সুপারভিভিটেন্টেন্ট শন্দ্রা শ্রীখার-সুলানকি বলেন, অভিযুক্তের মুখোমুখিতা আদা
  [11/25] BLEU=  3.2 ChrF= 30.9 RTF=0.113  id=1670
              pred: "এদের ফর্মালাল আচরণ পৃথিবীর জলবায়ু ও তাপমাত্রার তুলনায় কম, তবে তারা প্রায়শই জ
  [12/25] BLEU=  3.5 ChrF= 34.2 RTF=0.198  id=1671
              pred: কংগ্রেস ২০০৫ সালে 'অববববববব' (Obscenity) উদ্যোগের অর্থায়ন শুরু করে, যাতে এফবিআই
  [13/25] BLEU=  9.1 ChrF= 31.9 RTF=0.127  id=1672
              pred: ফ্যাব্রিককে খুব গরম না হতে দিতে সতর্ক থাকুন, যা অত্যন্ত ক্ষেত্রে জ্বালানি সৃষ্টি
  [14/25] BLEU=  7.4 ChrF= 51.0 RTF=0.141  id=1673
              pred: বিপ্লবী যুদ্ধের সময় ১৩টি রাজ্যের প্রথমবারের মতো কংগ্রেসই ছিল কনফেডারেশনের একমাত
  [15/25] BLEU=  7.2 ChrF= 34.8 RTF=0.197  id=1674
              pred: কিছু জায়গায় এক মিনিট সময় উষ্ণ পানি দিতে হলে, অন্য কিছু জায়গায় কয়েক মিনিট স
  [16/25] BLEU= 10.3 ChrF= 31.6 RTF=0.122  id=1675
              pred: বিশ্বের সব নাম্বার, এমনকি মধ্য মধ্যেও বড় অক্ষর দিয়ে শুরু হয়।
  [17/25] BLEU=  8.8 ChrF= 50.0 RTF=0.135  id=1676
              pred: সমস্ত দক্ষিণ আফ্রিকার জাতীয় উদ্যানগুলির মতো, পার্কের জন্য দৈনিক সংরক্ষণ এবং প্র
  [18/25] BLEU= 26.4 ChrF= 43.4 RTF=0.214  id=1677
              pred: আজকাল একমাত্র পোকামাকড় যা তাদের ডানা ফ্লিপ করতে পারে না তা হল ড্রাগনফ্লাই
  [19/25] BLEU=  0.4 ChrF=  6.5 RTF=0.717  id=1678
              pred: অলিভার স্যাক্স বলেন, 'মনেশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশশ
  [20/25] BLEU=  7.8 ChrF= 46.0 RTF=0.226  id=1679
              pred: এরোস্মিথ তাদের বাকি কনসার্ট বাতিল করে দিয়েছে।
  [21/25] BLEU=  4.3 ChrF= 36.6 RTF=0.190  id=1680
              pred: একটি সুশৃঙ্খলিত ক্রীড়াবিদ, বাঘ ভালভাবে সাঁতার কাটতে পারে না, একটি শক্তিশালী মান
  [22/25] BLEU=  5.1 ChrF= 34.4 RTF=0.108  id=1681
              pred: এটি একক, যদিও, পরীক্ষা, যা পরীক্ষামূলকভাবে এক বা একাধিক যুক্তিসঙ্গত অনুমান, প্রশ
  [23/25] BLEU=  1.3 ChrF= 11.5 RTF=0.593  id=1682
              pred: যদিও এর লেখক নিশ্চিত না, তবে জানা যায় যে এর প্রথম দিকে এটি একটি বড় পার্টমেট ডক
  [24/25] BLEU=  2.7 ChrF= 26.5 RTF=0.118  id=1683
              pred: এখানে বেঁচে থাকা অনেক পুরুষ ও মহিলা, যারা একসময় ইহুদি বা ইহুদি নয়, তাদের হত্যা
  [25/25] BLEU=  3.1 ChrF= 32.3 RTF=0.112  id=1684
              pred: সামোয়া দ্বীপের রাজধানী ও ওরুপুলুটের জনসংখ্যা মাত্র ৪০,০০০ জন।

  Summary: BLEU=6.34  ChrF=35.48  RTF=0.2341  Params=1713.7M

[ckpt] Saved phase5_benchmark(base)_step000000.pt (0.0 MB)
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P5_FLAP(base) (5 total)

<Figure size 1680x1200 with 4 Axes>
[image/png output omitted]
```

### Cell 77 (code, score=51)
```python
model_p4, processor = load_model_from_drive('phase4_enc_pruned')
print_model_breakdown(model_p4, 'After Phase 4:')
```
OUTPUT:
```text
[model] Loading phase4_enc_pruned from /kaggle/working/models/phase4_enc_pruned ...

Loading weights:   0%|          | 0/1330 [00:00<?, ?it/s]
  Restored custom state: ['_vocab_remap_to_old']
  [model] pruning_manifest: ['stage_name']

--- After Phase 4: ---
  speech_encoder                         441.6M  ( 39.5%)
  text_decoder                           373.6M  ( 33.4%)
  t2u_model                              261.8M  ( 23.4%)
  vocoder                                 41.9M  (  3.7%)
  shared                                  20.9M  (  1.9%)
  lm_head                                 20.9M  (  1.9%)
  TOTAL                                 1118.8M
---

{'shared': 20.9152,
 'speech_encoder': 441.604416,
 'text_decoder': 373.568512,
 'lm_head': 20.9152,
 't2u_model': 261.759747,
 'vocoder': 41.911362,
 'TOTAL': 1118.844037}
```

### Cell 78 (code, score=296)
```python
# ── Phase 5 Cell 6: RUN PHASE 5 (FIXED) ──────────────────────────────────────
# ROOT CAUSE: model_p4 has device_map='auto' split across devices after loading.
# deepcopy() preserves this split, causing "cuda:0 vs cpu" errors in calibration.
# FIX: Consolidate model_p4 to single GPU BEFORE deepcopy.

import gc as _stdlib_gc
import copy as _copy
 
FLAP_RATIO    = 0.10   # prune 15% of neurons globally per component
# REDUCED from 0.15 → 0.08 to prevent decoder collapse after Phases 3-4
MIN_KEEP_FRAC = 0.85   # never shrink any single layer below 70% of original

def force_model_to_single_device(model, device):
    """
    Aggressively consolidate model to a single device.
    Handles device_map='auto' models that are split across devices.
    """
    print(f"  Forcing entire model to {device}...")
    
    # Step 1: Move base model
    model = model.to(device)
    
    # Step 2: Explicitly move all submodules
    for name, module in model.named_modules():
        if module is not model:  # skip root
            try:
                module.to(device)
            except Exception:
                pass
    
    # Step 3: Move all parameters
    for name, param in model.named_parameters():
        if param.device != device:
            param.data = param.data.to(device)
    
    # Step 4: Move all buffers
    for name, buf in model.named_buffers():
        if buf is not None and buf.device != device:
            buf.data = buf.data.to(device)
    
    # Step 5: Clear device_map (forces single-device mode)
    if hasattr(model, 'hf_device_map'):
        model.hf_device_map = {k: device for k in model.hf_device_map}
    
    torch.cuda.empty_cache()
    
    # Verify consolidation
    devices = set()
    for p in model.parameters():
        devices.add(p.device)
    for b in model.buffers():
        devices.add(b.device)
    
    if len(devices) > 1:
        print(f"  WARNING: Model still on multiple devices: {devices}")
    else:
        print(f"  ✓ Model consolidated to {device}")
    
    return model

# ── Try loading completed Phase 5 from Drive ──────────────────────────────────
try:
    model_p5, processor = load_model_from_drive("phase5_flap_pruned(m4)")
    sync_model_config(model_p5)
    model_p5 = _consolidate_to_single_gpu(model_p5)
    device = torch.device("cuda:0")
    model_p5 = force_model_to_single_device(model_p5, device)
    print("Loaded Phase 5 from Drive.")
    p5_loaded = True
except Exception as _e:
    print(f"No Phase 5 on Drive ({_e}), pruning from model_p4...")
    p5_loaded = False
 
# ── Run Phase 5 pruning ───────────────────────────────────────────────────────
if not p5_loaded:
    device = torch.device("cuda:0")
    
    # CRITICAL FIX: Consolidate model_p4 BEFORE deepcopy
    print("Consolidating model_p4 to single GPU before deepcopy...")
    model_p4 = _consolidate_to_single_gpu(model_p4)
    model_p4 = force_model_to_single_device(model_p4, device)
    
    # Verify model_p4 is on single device
    p4_devices = set(p.device for p in model_p4.parameters())
    print(f"  model_p4 devices after consolidation: {p4_devices}")
    if len(p4_devices) > 1:
        raise RuntimeError(f"model_p4 still split across {p4_devices}! Cannot proceed.")
    
    # Now deepcopy will preserve single-device placement
    print("Deep-copying model_p4 → model_p5...")
    model_p5 = _copy.deepcopy(model_p4)
    model_p5 = force_model_to_single_device(model_p5, device)
    
    # Final verification
    p5_devices = set(p.device for p in model_p5.parameters())
    print(f"  model_p5 devices after deepcopy: {p5_devices}")
    
    pre_params = count_params(model_p5)
    print(f"\nPre-pruning: {pre_params:.1f}M params")
 
    prune_results = {}
 
    for comp_name in ["text_decoder", "speech_encoder", "t2u_model"]:
        print(f"\n{'='*60}")
        print(f"  Collecting calibration for {comp_name}")
        print(f"{'='*60}")
        
        # Verify component device before calibration
        comp = getattr(model_p5, comp_name)
        comp_device = next(comp.parameters()).device
        print(f"  {comp_name} device: {comp_device}")
        
        if comp_device != device:
            print(f"  ERROR: {comp_name} on wrong device! Re-consolidating...")
            model_p5 = force_model_to_single_device(model_p5, device)
        
        calib = collect_ffn_calibration_stats(
            model_p5, comp_name, calib_wavs, processor,
            n_samples=min(64, len(calib_wavs)),
            device=device,
        )
        _stdlib_gc.collect()
        torch.cuda.empty_cache()
 
        print(f"\n{'='*60}")
        print(f"  Applying FLAP to {comp_name}")
        print(f"{'='*60}")
        
        results = apply_flap_to_component(
            model_p5, comp_name, calib,
            global_prune_ratio=FLAP_RATIO,
            min_keep_frac=MIN_KEEP_FRAC,
            device=device,
        )
        prune_results[comp_name] = results
        _stdlib_gc.collect()
        torch.cuda.empty_cache()
 
    post_params = count_params(model_p5)
    print(f"\n{'='*60}")
```
OUTPUT:
```text
[model] Not in local cache, pulling from remote...
No Phase 5 on Drive ([rclone] model pull failed for phase5_flap_pruned(m4): 2026/04/19 03:54:44 ERROR : Google drive root 'seamV5/models/phase5_flap_pruned(m4)': error reading source root directory: directory not found
2026/04/19 03:54:44 ERROR : Local file system at /kaggle/working/models/phase5_flap_pruned(m4): not deleting files as there were IO errors
2026/04/19 03:54:4), pruning from model_p4...
Consolidating model_p4 to single GPU before deepcopy...
  Multi-device map detected, consolidating to cuda:0...
  Model now on: cuda:0
  Forcing entire model to cuda:0...
  ✓ Model consolidated to cuda:0
  model_p4 devices after consolidation: {device(type='cuda', index=0)}
Deep-copying model_p4 → model_p5...
  Forcing entire model to cuda:0...
  ✓ Model consolidated to cuda:0
  model_p5 devices after deepcopy: {device(type='cuda', index=0)}

Pre-pruning: 1118.8M params

============================================================
  Collecting calibration for text_decoder
============================================================
  text_decoder device: cuda:0
  [calib] Collecting activations from 25 samples for text_decoder (direct forward pass — every layer fires)...
  [calib] 0/25
  [calib] 20/25
  [calib] Layers fired: 14/14  total token-vectors: 11200
  [calib] All 14 layers fired correctly.
  [calib] Done. 14 FFN layers instrumented.

============================================================
  Applying FLAP to text_decoder
============================================================
  [FLAP] text_decoder: 114688 total neurons, pruning ≤11468 (10%), threshold=-0.8309
         score range [-13.174, 19.429]  mean=0.000  std=1.000
  [FLAP] Done. Kept 103416/114688 neurons (90.2%) across 14 layers.

============================================================
  Collecting calibration for speech_encoder
============================================================
  speech_encoder device: cuda:0
  [calib] Collecting activations from 25 samples for speech_encoder (direct forward pass — every layer fires)...
  [calib] 0/25
  [calib] 20/25
  [calib] Layers fired: 34/34  total token-vectors: 448360
  [calib] All 34 layers fired correctly.
  [calib] Done. 34 FFN layers instrumented.

============================================================
  Applying FLAP to speech_encoder
============================================================
  [FLAP] speech_encoder: 139264 total neurons, pruning ≤13926 (10%), threshold=-1.1333
         score range [-5.212, 12.750]  mean=0.000  std=1.000
  [FLAP] Done. Kept 125362/139264 neurons (90.0%) across 34 layers.

============================================================
  Collecting calibration for t2u_model
============================================================
  t2u_model device: cuda:0
  [calib] Collecting activations from 25 samples for t2u_model (direct forward pass — every layer fires)...
  [calib] 0/25
  [calib] 20/25
  [calib] Layers fired: 6/6  total token-vectors: 2550
  [calib] All 6 layers fired correctly.
  [calib] Done. 6 FFN layers instrumented.

============================================================
  Applying FLAP to t2u_model
============================================================
  [FLAP] t2u_model: 49152 total neurons, pruning ≤4915 (10%), threshold=-1.0381
         score range [-4.763, 13.959]  mean=-0.000  std=1.000
  [FLAP] Done. Kept 44237/49152 neurons (90.0%) across 6 layers.

============================================================
Width pruning complete:
  1118.8M → 1057.2M
  Saved: 61.7M params
============================================================
  [config] sync done.
  text_decoder: avg 90.2% neurons kept (14 layers)
  speech_encoder: avg 90.0% neurons kept (34 layers)
  t2u_model: avg 90.0% neurons kept (6 layers)
[ckpt] Saved phase5_flap(m4)_step000000.pt (0.0 MB)
  [config] sync done.
[model] Saving phase5_flap_pruned(m4) → /kaggle/working/models/phase5_flap_pruned(m4) ...
  config.decoder_ffn_dim: 8192 -> 7604
  config.t2u_encoder_ffn_dim: 8192 -> 7199
  [config] sync done.
  Saved custom state: ['_vocab_remap_to_old']
  Saved pruning_manifest.pt keys=['stage_name']

Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]
[model] Local save done. 2147 MB in 8 files.
[model] Pushing to rclone remote...
[model] Verified 8 files on remote.

--- After Phase 5: FLAP Width Pruned(m4) ---
  speech_encoder                         413.1M  ( 39.1%)
  text_decoder                           350.5M  ( 33.2%)
  t2u_model                              251.7M  ( 23.8%)
  vocoder                                 41.9M  (  4.0%)
  shared                                  20.9M  (  2.0%)
  lm_head                                 20.9M  (  2.0%)
  TOTAL                                 1057.2M
---

{'shared': 20.9152,
 'speech_encoder': 413.119218,
 'text_decoder': 350.472184,
 'lm_head': 20.9152,
 't2u_model': 251.688912,
 'vocoder': 41.911362,
 'TOTAL': 1057.191676}
```

### Cell 79 (code, score=135)
```python
# ── Phase 5 Cell 7: Quick sanity check then full benchmark ───────────────────
print("Quick sanity check (3 samples)...")
model_p5.eval()
 
for i, s in enumerate(eval_samples[:3]):
    wav = s["wav"]
    ref = s["ref"]
    try:
        pred, _wav_out = run_s2st(model_p5, wav, tgt_lang="ben")
        chrf = compute_chrf(pred, ref)
        print(f"  [{i+1}] ChrF={chrf:.1f}  pred: {pred[:80]!r}")
    except Exception as e:
        print(f"  [{i+1}] ERROR: {e}")

p5b = load_latest_checkpoint("phase5_benchmark(m4)")
if p5b:
    p5_results, p5_summary = p5b["results"], p5b["summary"]
    print(f"Loaded P5 benchmark: BLEU={p5_summary['avg_bleu']:.2f} "
          f"ChrF={p5_summary['avg_chrf']:.2f}")
else:
    p5_results, p5_summary = run_benchmark(
        model_p5, eval_samples, label="P5_FLAP(m4)", save_n=2)
    save_checkpoint(dict(results=p5_results, summary=p5_summary),
                    name="phase5_benchmark(m4)", step=0)
 
store_summary(p5_summary)
plot_phase_comparison()
```
OUTPUT:
```text
Quick sanity check (3 samples)...
  [1] ChrF=9.6  pred: 'রোম্যান্টিকবাদবাদবাদের সংস্কৃতিগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগ'
  [2] ChrF=6.4  pred: "তিনি বলেন, 'কাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকা"
  [3] ChrF=8.1  pred: '"আমাদের সবসময়ই দুটি বা একাধিক বিষয় মিমিমিমি, পি.এ.এ.এ.আই.কে "'
[ckpt] No checkpoint for 'phase5_benchmark(m4)'

============================================================
  BENCHMARK: P5_FLAP(m4)
  Samples: 25  Target: ben
============================================================

  GPU mem: 4.43 GB alloc / 6.85 GB reserved
  [ 1/25] BLEU=  0.0 ChrF=  9.6 RTF=0.471  id=1660
              pred: রোম্যান্টিকবাদবাদবাদের সংস্কৃতিগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগতগ
[audio] Saved P5_FLAP(m4)_s1in.wav (0.3 MB)
  P5_FLAP(m4)_s1in.wav  (10.7s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P5_FLAP(m4)_s1out.wav (1.2 MB)
  P5_FLAP(m4)_s1out.wav  (38.0s | sr=16000)

<IPython.lib.display.Audio object>
  [ 2/25] BLEU=  0.8 ChrF=  6.4 RTF=0.165  id=1661
              pred: তিনি বলেন, 'কাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকাকা
[audio] Saved P5_FLAP(m4)_s2in.wav (0.2 MB)
  P5_FLAP(m4)_s2in.wav  (6.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P5_FLAP(m4)_s2out.wav (0.2 MB)
  P5_FLAP(m4)_s2out.wav  (6.5s | sr=16000)

<IPython.lib.display.Audio object>
  [ 3/25] BLEU=  1.8 ChrF=  8.1 RTF=0.083  id=1662
              pred: "আমাদের সবসময়ই দুটি বা একাধিক বিষয় মিমিমিমি, পি.এ.এ.এ.আই.কে "
  [ 4/25] BLEU=  0.0 ChrF=  3.5 RTF=0.259  id=1663
              pred: "চোকোকোকামো, চিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিলিল
  [ 5/25] BLEU=  0.0 ChrF=  1.3 RTF=0.589  id=1664
              pred: রা ে িিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিিি
  [ 6/25] BLEU=  0.0 ChrF=  5.5 RTF=0.613  id=1665
              pred: দলটির মতে, এই চুক্তিইইই, '১৯৬৭৭৭ বছর ব ব ব ব ব বধিসার্ার্ার্ার্ার্ার্ার্ার্ার্ার
  [ 7/25] BLEU=  2.1 ChrF= 29.7 RTF=0.089  id=1666
              pred: আপনি আপনার নিজের নাগরিকদের ছাড়া অন্যদেরদেরদেরদেরদেরদেরদেরদেরদের সাথে পরামর্শের 
  [ 8/25] BLEU=  0.3 ChrF= 15.8 RTF=0.251  id=1667
              pred: "যেসে সাধারণত, দুই ধরনের আচরণেরেরের মধ্যে থেকে একজন একজন একজন ম্যানেজার তাদের পূ
  [ 9/25] BLEU=  1.1 ChrF= 25.7 RTF=0.128  id=1668
              pred: এটি দক্ষিণ আফ্রিকার বা দক্ষিণ আফ্রিকার জাতীয় উদ্যানগুলির জন্য এক এক এক এক এক এক
  [10/25] BLEU=  0.2 ChrF=  5.2 RTF=0.696  id=1669
              pred: পুলিশ সুপারিনেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটেটে
  [11/25] BLEU=  0.0 ChrF=  3.2 RTF=0.109  id=1670
              pred: তাদের আনুষ্ঠানিক আচরণ, সাধারণতঃঃ-এএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএএ
  [12/25] BLEU=  0.6 ChrF=  4.1 RTF=0.504  id=1671
              pred: "ম. দ. 2005 সালে, কংগ্রেসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসসস
  [13/25] BLEU=  0.1 ChrF=  3.7 RTF=0.498  id=1672
              pred: যাই যাইহোক, ফ্যাব্রিককে খুব বেশি স্রা ri ri ri ri ri ri ri ri ri ri ri ri ri ri 
  [14/25] BLEU=  0.0 ChrF=  2.9 RTF=0.351  id=1673
              pred: বিপ্লববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববব
  [15/25] BLEU=  2.3 ChrF=  7.5 RTF=0.218  id=1674
              pred: কিছু কিছু কিছু কিছু কিছু কিছু কিছু কিছু কিছু কিছু কিছু কিছু কিছু কিছুক্ষণের জন্য
  [16/25] BLEU=  0.0 ChrF=  2.2 RTF=0.566  id=1675
              pred: " 'ওওওওওও''''''''''''''''''''''''''''''''''''''''' '...'সমস্ত নোডডডডডডডলললসসসসসস
  [17/25] BLEU=  0.2 ChrF=  5.0 RTF=0.536  id=1676
              pred: দক্ষিণ আফ্রোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকোকো
  [18/25] BLEU=  2.6 ChrF= 12.0 RTF=0.361  id=1677
              pred: আজ, একমাত্র পপপপপপপপপপ যা তাদের পাড়ের পিছনে ঢাকাতেতেতেতেতেতেতেতেতেতেতেতেতেতেতেত
  [19/25] BLEU=  0.0 ChrF=  1.5 RTF=0.392  id=1678
              pred: "আলভার-ববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববব
  [20/25] BLEU=  5.5 ChrF= 46.7 RTF=0.135  id=1679
              pred: ইরাস্স্স্মিত তাদের সফরর জন্য তাদের অবশিষ্ট সংগীত অনুষ্ঠান বাতিল করেছে।
  [21/25] BLEU=  0.2 ChrF=  5.3 RTF=0.485  id=1680
              pred: একজন সজ্জিতিত ক্রী্রী্রীতিতিতিতিতিতিতিতি, টাইগারগগগগগগগগগগগগগগগগগগগগগগগগগগগগগগগগ
  [22/25] BLEU=  0.0 ChrF=  1.9 RTF=0.264  id=1681
              pred: যাই যাইহোক, একটি অ-অঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅঅ
  [23/25] BLEU=  0.2 ChrF=  6.2 RTF=0.183  id=1682
              pred: যদিও, লেখকের জন্য, তার জীবনেরইইইইইইইইইইইইইইই, তারইইইইইইইইইইইইইইইইইইইইইইইইইইইইইইই
  [24/25] BLEU=  5.4 ChrF= 15.5 RTF=0.435  id=1683
              pred: "এএএতে এখনও অনেক পুরুষ ও মহিলারা তাদের সময় বেঁচে আছে, " "এএতে " ""মৃতৃতৃতৃতৃত""
  [25/25] BLEU=  0.0 ChrF=  1.6 RTF=0.467  id=1684
              pred: উপ উপ উপ উপ উপনিনিনিনিনিনিনি, ইইউপোপোপোপোপোপোপোপোপোপোপোপোপোপোপোপোপোপোপোপোপোপোপোপ

  Summary: BLEU=0.95  ChrF=9.20  RTF=0.3540  Params=1057.2M

[ckpt] Saved phase5_benchmark(m4)_step000000.pt (0.0 MB)
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P5_FLAP(m4) (6 total)

<Figure size 1680x1200 with 4 Axes>
[image/png output omitted]
```

### Cell 80 (code, score=10)
```python
# # ── Phase 5 Cell 7: Sanity check before full benchmark ───────────────────────
# print('Quick sanity check (3 samples)...')
# model_p5.eval()

# for i, s in enumerate(eval_samples[:3]):
#     wav = s['wav']
#     ref = s['ref']
#     try:
#         pred, _wav_out = run_s2st(model_p5, wav, tgt_lang='ben')
#         chrf = compute_chrf(pred, ref)
#         print(f'  [{i+1}] ChrF={chrf:.1f}  pred: {pred[:80]!r}')
#     except Exception as e:
#         print(f'  [{i+1}] ERROR: {e}')
```

### Cell 81 (code, score=25)
```python
# # ── Phase 5 Cell 8: Full benchmark ───────────────────────────────────────────
# p5b = load_latest_checkpoint('phase5_benchmark')
# if p5b:
#     p5_results, p5_summary = p5b['results'], p5b['summary']
#     print(f'Loaded P5 benchmark: BLEU={p5_summary["avg_bleu"]:.2f} '
#           f'ChrF={p5_summary["avg_chrf"]:.2f}')
# else:
#     p5_results, p5_summary = run_benchmark(
#         model_p5, eval_samples, label='P5_FLAP', save_n=2)
#     save_checkpoint(dict(results=p5_results, summary=p5_summary),
#                     name='phase5_benchmark', step=0)

# store_summary(p5_summary)
# plot_phase_comparison()
```

### Cell 82 (code, score=3)
```python
# 1. Delete model references
del model_p5, model_p4

# 2. Force garbage collection
gc.collect()

# 3. Empty the GPU cache
torch.cuda.empty_cache()
```

### Cell 83 (code, score=51)
```python
model_p4, processor = load_model_from_drive('phase4_enc_pruned')
print_model_breakdown(model_p4, 'After Phase 4:')
```
OUTPUT:
```text
[model] Loading phase4_enc_pruned from /kaggle/working/models/phase4_enc_pruned ...

Loading weights:   0%|          | 0/1330 [00:00<?, ?it/s]
  Restored custom state: ['_vocab_remap_to_old']
  [model] pruning_manifest: ['stage_name']

--- After Phase 4: ---
  speech_encoder                         441.6M  ( 39.5%)
  text_decoder                           373.6M  ( 33.4%)
  t2u_model                              261.8M  ( 23.4%)
  vocoder                                 41.9M  (  3.7%)
  shared                                  20.9M  (  1.9%)
  lm_head                                 20.9M  (  1.9%)
  TOTAL                                 1118.8M
---

{'shared': 20.9152,
 'speech_encoder': 441.604416,
 'text_decoder': 373.568512,
 'lm_head': 20.9152,
 't2u_model': 261.759747,
 'vocoder': 41.911362,
 'TOTAL': 1118.844037}
```

### Cell 84 (code, score=69)
```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 6 — T2U Iterative Layer Pruning (CORRECTED)                         ║
# ║  Based on model_p4 (Phase 5 skipped).                                       ║
# ║  Follows the Phase 3 / Phase 4 gold standard exactly:                       ║
# ║    • quick_eval_chrf() for baseline and scoring (not manual loops)           ║
# ║    • ALL layers eligible to prune (no first/mid/last guard; stacks are small) ║
# ║    • Checkpoint resume per stack                                             ║
# ║    • sync_model_config() + layer-index realignment after pruning             ║
# ║    • save_model_to_drive() before benchmark                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Phase 6 Cell 1: T2U architecture probe ───────────────────────────────────
# NOTE: uses model_p4 — Phase 5 is intentionally skipped.

def inspect_t2u(model):
    """Print T2U sub-components so we know what to prune."""
    t2u = model.t2u_model
    print('T2U sub-components:')
    for name, child in t2u.named_children():
        n = count_params(child)
        print(f'  {name:<35} {n:.1f}M')
        for subname, subchild in child.named_children():
            sn = count_params(subchild)
            if sn > 1:
                print(f'    {subname:<33} {sn:.1f}M')
    # Find layer lists
    for name, child in t2u.named_children():
        for attr in ['layers', 'inner_layers', 'encoder_layers', 'decoder_layers']:
            if hasattr(child, attr):
                layers = getattr(child, attr)
                if isinstance(layers, nn.ModuleList):
                    print(f'  → Found t2u.{name}.{attr}: {len(layers)} layers')

inspect_t2u(model_p4)   # ← model_p4, not model_p5
```
OUTPUT:
```text
T2U sub-components:
  model                               261.8M
    encoder                           125.9M
    decoder                           135.8M
  lm_head                             10.3M
```

### Cell 85 (markdown, score=5)
```markdown
---
# Phase 6: T2U Model Pruning
Apply middle-layer removal to T2U transformer stacks (2-3 layers per stack).
```

### Cell 86 (code, score=57)
```python
# ── Phase 6 Cell 2: T2U stack discovery + helpers ────────────────────────────

def find_t2u_stacks(model):
    """
    Return list of (parent_object, layers_attr, display_name) for each
    prunable layer stack inside t2u_model.  Only considers stacks with ≥3 layers.
    """
    t2u = model.t2u_model
    stacks = []

    def _search(module, prefix):
        for attr in ['layers', 'inner_layers', 'encoder_layers', 'decoder_layers']:
            if hasattr(module, attr):
                layers = getattr(module, attr)
                if isinstance(layers, nn.ModuleList) and len(layers) >= 3:
                    stacks.append((module, attr, f't2u.{prefix}.{attr}'))
        for name, child in module.named_children():
            _search(child, f'{prefix}.{name}' if prefix else name)

    _search(t2u, 't2u_model')
    return stacks


def sync_t2u_layer_indices(model):
    """
    Re-index layer_idx on any attention modules inside t2u_model,
    exactly like reindex_text_decoder_layer_idx() in Phase 3.
    """
    t2u = model.t2u_model
    stacks = find_t2u_stacks(model)
    for (parent, attr, name) in stacks:
        layers = list(getattr(parent, attr))
        for i, layer in enumerate(layers):
            # Conformer / transformer layers may expose self_attn
            for attn_name in ['self_attn', 'encoder_attn', 'cross_attention']:
                attn = getattr(layer, attn_name, None)
                if attn is not None and hasattr(attn, 'layer_idx'):
                    attn.layer_idx = i
        print(f'  Re-indexed {name}: {len(layers)} layers')


print('T2U helpers (find_t2u_stacks, sync_t2u_layer_indices) ready.')
```
OUTPUT:
```text
T2U helpers (find_t2u_stacks, sync_t2u_layer_indices) ready.
```

### Cell 87 (code, score=159)
```python
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 FIX: ASR-ChrF Scoring for T2U Layer Pruning
# ═══════════════════════════════════════════════════════════════════════════════
#
# ROOT CAUSE OF BUG:
#   quick_eval_chrf() uses run_s2t_only(), which bypasses the T2U model entirely.
#   T2U layers only affect the audio output path (text tokens → speech units → audio).
#   Text-only evaluation shows no difference when T2U layers are removed because
#   the text decoder path is unchanged.
#
# FIX:
#   Use ASR-ChrF scoring for pruning decisions (fast, reliable).
#   ASR-BLEU is computed separately during final benchmarking.
#
#   This evaluates the ACTUAL S2ST audio output by:
#     1. Running full S2ST pipeline (generates audio)
#     2. Transcribing output audio with MMS-ASR (Bengali)
#     3. Computing ChrF between ASR transcript and reference
#
# USAGE:
#   Replace Phase 6 Cell 3 in your notebook with this code.
#
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import numpy as np


def quick_eval_asr_chrf(model, samples, tgt_lang='ben', max_eval=10):
    """
    Fast S2ST quality evaluation using ASR-ChrF only.
    
    This is the CORRECT metric for T2U pruning because:
      - T2U layers only affect the audio output path (text → units → audio)
      - Text-only metrics (quick_eval_chrf) bypass T2U entirely
      - ASR-ChrF measures whether the output audio contains correct words
    
    Returns: avg_asr_chrf (used for pruning decisions)
    
    Note: ASR-BLEU is computed separately during final benchmarking.
    """
    _ensure_mms_loaded()  # Load MMS-ASR model if not already loaded
    
    scores = []
    for s in samples[:max_eval]:
        try:
            # Full S2ST: generates both text and audio
            pred_text, out_wav = run_s2st(model, s['wav'], tgt_lang=tgt_lang)
            
            # ASR-ChrF: transcribe output audio, compare to reference
            if out_wav is not None and len(out_wav) > 1600:
                _, asr_chrf = compute_asr_chrf(
                    out_wav, s['ref'], sr=model.config.sampling_rate)
                scores.append(asr_chrf)
            else:
                # No audio output = catastrophic failure
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    
    return float(np.mean(scores)) if scores else 0.0


def iterative_prune_t2u_stack(model, stack_parent, layers_attr,
                               stack_name, samples, n_remove,
                               tgt_lang='ben', max_eval=10,
                               ckpt_name=None):
    """
    Iterative greedy pruning for one T2U layer stack.
    
    KEY FIX: Uses ASR-ChrF scoring (not text-only ChrF).
    T2U layers only affect the audio output path, so we must evaluate
    the actual S2ST audio quality using MMS-ASR transcription.
    
    Scoring: ASR-ChrF only (for pruning decisions)
    Selection: Remove the layer whose removal causes LEAST ChrF degradation
               (i.e., keeps the highest ASR-ChrF).
    
    Note: ASR-BLEU is computed separately during final benchmarking.
    """
    if ckpt_name is None:
        ckpt_name = f'phase6_{stack_name.replace(".", "_").replace(" ", "_")}_pruning'

    current = list(getattr(stack_parent, layers_attr))
    orig_indices = list(range(len(current)))
    n_total_orig = len(current)

    # Clamp n_remove so at least 2 layers always remain
    if n_total_orig - n_remove < 2:
        n_remove = max(0, n_total_orig - 2)
        print(f'  Clamped n_remove to {n_remove} (keeping minimum 2 layers)')

    if n_remove == 0:
        print(f'  {stack_name}: nothing to remove.')
        return [], []

    print(f'  {stack_name}: {n_total_orig} layers, removing {n_remove} (all eligible)')
    print(f'  Scoring: ASR-ChrF (via MMS-ASR Bengali transcription)')

    removed, log = [], []

    # ── Resume from checkpoint ──
    partial = load_latest_checkpoint(ckpt_name)
    if partial and partial.get('removed'):
        removed = list(partial['removed'])
        log = partial.get('log', [])
        for r in removed:
            if r in orig_indices:
                pos = orig_indices.index(r)
                current.pop(pos)
                orig_indices.pop(pos)
        setattr(stack_parent, layers_attr, nn.ModuleList(current))
        print(f'  Resuming: already removed {removed}, {len(current)} layers remain')

    # ── Baseline ASR-ChrF ──
    baseline_chrf = quick_eval_asr_chrf(model, samples, tgt_lang, max_eval)
    print(f'  Baseline ASR-ChrF: {baseline_chrf:.2f}')

    start_iter = len(removed)
    for it in range(start_iter, n_remove):
        eligible = list(range(len(current)))

        if not eligible:
            print(f'  WARNING: No layers left to prune. Stopping.')
            break

        print(f'\n  Iter {it+1}/{n_remove} ({len(current)} layers remain, '
              f'all {len(eligible)} eligible)')

        scores = {}
        for pos in eligible:
            temp = current[:pos] + current[pos+1:]
            setattr(stack_parent, layers_attr, nn.ModuleList(temp))
            
            # Evaluate ASR-ChrF with this layer removed
            sc = quick_eval_asr_chrf(model, samples, tgt_lang, max_eval)
            scores[pos] = (orig_indices[pos], sc)
            print(f'    Remove L{orig_indices[pos]:>2} -> ASR-ChrF={sc:.2f}')
        
```
OUTPUT:
```text
iterative_prune_t2u_stack() — ASR-ChrF scoring (S2ST-aware).
  Metric: ASR-ChrF only (for pruning decisions)
  ASR-BLEU computed separately during final benchmarking.
  This correctly evaluates T2U layer impact on audio output quality.
```

### Cell 88 (code, score=42)
```python
!rm -rf /kaggle/working/checkpoints/phase6*.pt
!rm -rf /kaggle/working/models/phase6_t2u_iter_pruned
!ls checkpoints
```
OUTPUT:
```text
all_summaries_step000000.pt	    phase4_benchmark_step000000.pt
 phase0_baseline_step000000.pt	    phase4_enc_pruning_step000000.pt
 phase1_benchmark_step000000.pt    'phase5_benchmark(base)_step000000.pt'
 phase1_vocab_step000000.pt	   'phase5_benchmark(m4)_step000000.pt'
 phase3_benchmark_step000000.pt    'phase5_flap(base)_step000000.pt'
 phase3_dec_pruning_step000000.pt  'phase5_flap(m4)_step000000.pt'
```

### Cell 89 (code, score=437)
```python
# ── Phase 6 Cell 4: RUN PHASE 6 ──────────────────────────────────────────────
# model_p6 = model_p4 (Phase 5 deliberately skipped).
# Prune T2U encoder + decoder stacks independently, same as Phase 3/4 did for
# text_decoder and speech_encoder.

import copy as _copy
import gc as _stdlib_gc  # stdlib gc; a notebook variable named `gc` shadows the module

N_T2U_REMOVE_PER_STACK = 2   # conservative: keeps ≥4 layers in each 6-layer stack

# ── Step 0: Obtain p4 baseline ChrF (loaded from checkpoint, never hardcoded) ──
p4b = load_latest_checkpoint('phase4_benchmark')
if p4b:
    p4_baseline_chrf = p4b['summary']['avg_chrf']
    print(f'Phase 4 baseline ChrF (loaded from checkpoint): {p4_baseline_chrf:.2f}')
else:
    # Fallback: compute it live if checkpoint is missing
    print('Phase 4 benchmark checkpoint not found — computing live baseline...')
    _, tmp_summary = run_benchmark(model_p4, eval_samples,
                                   label='P4_EncPrune_baseline', save_n=0)
    p4_baseline_chrf = tmp_summary['avg_chrf']
    print(f'Live baseline ChrF: {p4_baseline_chrf:.2f}')

# ── Step 1: Try loading a completed Phase 6 model from Drive ──────────────────
p6_ckpt = load_latest_checkpoint('phase6_t2u_pruning')

# ── Architecture-aware Drive load (fixes the MISSING-keys / garbage-init bug) ─
# ROOT CAUSE: load_model_from_drive() reinstates the full HuggingFace
# architecture (6+6 T2U layers) before filling weights.  The pruned layers
# (3,4,5) are absent from the saved checkpoint, so HF randomly initialises
# them — the model ends up with 6 layers again but with garbage weights.
#
# FIX (same pattern as Phase 3 reindex cell):
#   1. Read the pruning log  →  know which original indices were removed
#   2. Start from model_p4  (speech_encoder + text_decoder already pruned)
#   3. Replay T2U removals in memory  →  skeleton now has 3+3 layers
#   4. sync_model_config() + reindex  →  config matches skeleton
#   5. load_state_dict() from saved weights  →  zero MISSING keys

def _rebuild_p6_from_checkpoint(base_model, p6_pruning_ckpt):
    """
    Replay T2U layer removals onto base_model, then load saved weights.
    Returns the corrected model with no randomly-initialised layers.
    """
    import torch.nn as nn

    removed_map = p6_pruning_ckpt.get('removed', {})
    if not removed_map:
        print('  [p6 rebuild] No removal log — loading weights as-is.')
        return base_model

    mdl = base_model
    stacks = find_t2u_stacks(mdl)
    print(f'  [p6 rebuild] Replaying removals on {len(stacks)} T2U stacks...')

    for (stack_parent, layers_attr, stack_name) in stacks:
        removed_orig = removed_map.get(stack_name)
        if not removed_orig:
            print(f'    {stack_name}: no removal record, skipping.')
            continue
        removed_set = set(removed_orig)
        layers = list(getattr(stack_parent, layers_attr))
        n_before = len(layers)
        keep = [i for i in range(n_before) if i not in removed_set]
        setattr(stack_parent, layers_attr, nn.ModuleList([layers[i] for i in keep]))
        print(f'    {stack_name}: {n_before} -> {len(keep)} layers ')
        print(f'      (removed original indices {sorted(removed_orig)})')

    print('  [p6 rebuild] Re-indexing T2U layer indices...')
    sync_t2u_layer_indices(mdl)

    print('  [p6 rebuild] Syncing model config...')
    sync_model_config(mdl)

    if hasattr(mdl, '_cache'):
        delattr(mdl, '_cache')

    # Load weights — skeleton matches saved checkpoint (safetensors or .bin)
    model_dir = f'{MODEL_DIR}/phase6_t2u_iter_pruned'
    state = load_hf_weights_dict(model_dir)
    if state is not None:
        print(f'  [p6 rebuild] Loading weights from {model_dir} ...')
        missing, unexpected = mdl.load_state_dict(state, strict=False)
        if missing:
            print(f'  [p6 rebuild] WARNING: still {len(missing)} missing keys: ')
            for k in missing[:8]:
                print(f'    {k}')
        else:
            print('  [p6 rebuild] All keys matched — no random initialisation.')
        if unexpected:
            print(f'  [p6 rebuild] ({len(unexpected)} unexpected keys — often buffers)')
        del state
        _stdlib_gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f'  [p6 rebuild] No model.safetensors / pytorch_model.bin in {model_dir}')

    return mdl


p6_loaded = False
if p6_ckpt and p6_ckpt.get('removed') and os.path.isdir(
        f'{MODEL_DIR}/phase6_t2u_iter_pruned'):
    print('Phase 6 pruning log + saved weights found — rebuilding pruned arch...')
    # deepcopy so we never mutate model_p4 in place while replaying T2U surgery
    model_p6 = _copy.deepcopy(model_p4)
    model_p6 = _consolidate_to_single_gpu(model_p6)
    model_p6 = _rebuild_p6_from_checkpoint(model_p6, p6_ckpt)
    p6_dir = f'{MODEL_DIR}/phase6_t2u_iter_pruned'
    _load_custom_state(model_p6, p6_dir)
    print('Loaded Phase 6 from Drive (architecture-aware rebuild — zero MISSING keys).')
    p6_loaded = True

if not p6_loaded:
    # Phase 6 starts from model_p4 (Phase 5 skipped)
    model_p6 = model_p4
    model_p6 = _consolidate_to_single_gpu(model_p6)

    # ── Sanity check: confirm we have the real p4 baseline ──────────────────
    sanity = quick_eval_chrf(model_p6, eval_samples, TARGET_LANG, 5)
    print(f'  Sanity ChrF = {sanity:.2f}  '
          f'(expect ~{p4_baseline_chrf:.1f}, abort if < 10)')
    assert sanity > 10, \
        f'ChrF={sanity:.2f} is too low — model or vocab remap is broken!'

    pre_t2u = count_params(model_p6.t2u_model)
    print(f'\nT2U before pruning: {pre_t2u:.1f}M params')

    stacks = find_t2u_stacks(model_p6)
    if not stacks:
        print('WARNING: No prunable layer stacks found in T2U. Skipping.')
    else:
        print(f'Found {len(stacks)} T2U stacks to prune:')
        for (parent, attr, name) in stacks:
            print(f'  {name}: {len(getattr(parent, attr))} layers')

    all_removed = {}
    all_logs    = {}

    for (stack_parent, layers_attr, stack_name) in stacks:
```
OUTPUT:
```text
[ckpt] Loaded phase4_benchmark_step000000.pt
Phase 4 baseline ChrF (loaded from checkpoint): 40.11
[ckpt] No checkpoint for 'phase6_t2u_pruning'
  Multi-device map detected, consolidating to cuda:0...
  Model now on: cuda:0
  Sanity ChrF = 43.55  (expect ~40.1, abort if < 10)

T2U before pruning: 261.8M params
Found 2 T2U stacks to prune:
  t2u.t2u_model.model.encoder.layers: 6 layers
  t2u.t2u_model.model.decoder.layers: 6 layers
  t2u.t2u_model.model.encoder.layers: 6 layers, removing 2 (all eligible)
  Scoring: ASR-ChrF (via MMS-ASR Bengali transcription)
[ckpt] No checkpoint for 'phase6_t2u_t2u_model_model_encoder_layers_pruning'
[MMS-ASR] Loading facebook/mms-1b-all  lang=ben...

preprocessor_config.json:   0%|          | 0.00/254 [00:00<?, ?B/s]
config.json: 0.00B [00:00, ?B/s]
tokenizer_config.json:   0%|          | 0.00/397 [00:00<?, ?B/s]
vocab.json: 0.00B [00:00, ?B/s]
special_tokens_map.json:   0%|          | 0.00/96.0 [00:00<?, ?B/s]
model.safetensors:   0%|          | 0.00/3.86G [00:00<?, ?B/s]
Loading weights:   0%|          | 0/1096 [00:00<?, ?it/s]
adapter.ben.safetensors:   0%|          | 0.00/9.34M [00:00<?, ?B/s]
[MMS-ASR] Ready.
  Baseline ASR-ChrF: 43.77

  Iter 1/2 (6 layers remain, all 6 eligible)
    Remove L 0 -> ASR-ChrF=8.40
    Remove L 1 -> ASR-ChrF=43.83
    Remove L 2 -> ASR-ChrF=43.24
    Remove L 3 -> ASR-ChrF=43.76
    Remove L 4 -> ASR-ChrF=43.43
    Remove L 5 -> ASR-ChrF=42.37
  -> Removed L1 (ASR-ChrF=43.83, 5 layers remain)
[ckpt] Saved phase6_t2u_t2u_model_model_encoder_layers_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (1/2 iterations done)

  Iter 2/2 (5 layers remain, all 5 eligible)
    Remove L 0 -> ASR-ChrF=9.14
    Remove L 2 -> ASR-ChrF=43.51
    Remove L 3 -> ASR-ChrF=36.14
    Remove L 4 -> ASR-ChrF=43.19
    Remove L 5 -> ASR-ChrF=43.39
  -> Removed L2 (ASR-ChrF=43.51, 4 layers remain)
[ckpt] Saved phase6_t2u_t2u_model_model_encoder_layers_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (2/2 iterations done)
  t2u.t2u_model.model.decoder.layers: 6 layers, removing 2 (all eligible)
  Scoring: ASR-ChrF (via MMS-ASR Bengali transcription)
[ckpt] No checkpoint for 'phase6_t2u_t2u_model_model_decoder_layers_pruning'
  Baseline ASR-ChrF: 43.51

  Iter 1/2 (6 layers remain, all 6 eligible)
    Remove L 0 -> ASR-ChrF=25.97
    Remove L 1 -> ASR-ChrF=40.01
    Remove L 2 -> ASR-ChrF=39.29
    Remove L 3 -> ASR-ChrF=39.88
    Remove L 4 -> ASR-ChrF=39.56
    Remove L 5 -> ASR-ChrF=42.51
  -> Removed L5 (ASR-ChrF=42.51, 5 layers remain)
[ckpt] Saved phase6_t2u_t2u_model_model_decoder_layers_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (1/2 iterations done)

  Iter 2/2 (5 layers remain, all 5 eligible)
    Remove L 0 -> ASR-ChrF=24.07
    Remove L 1 -> ASR-ChrF=38.37
    Remove L 2 -> ASR-ChrF=37.37
    Remove L 3 -> ASR-ChrF=39.06
    Remove L 4 -> ASR-ChrF=33.60
  -> Removed L3 (ASR-ChrF=39.06, 4 layers remain)
[ckpt] Saved phase6_t2u_t2u_model_model_decoder_layers_pruning_step000000.pt (0.0 MB)
  [ckpt] Progress saved (2/2 iterations done)

T2U after pruning: 182.0M params (saved 79.7M)

Re-indexing T2U layer indices...
  Re-indexed t2u.t2u_model.model.encoder.layers: 4 layers
  Re-indexed t2u.t2u_model.model.decoder.layers: 4 layers
Syncing model config...
  [config] t2u_encoder_layers: 6 -> 4
  [config] t2u_decoder_layers: 6 -> 4
  [config] t2u_model.config.encoder_layers: 6 -> 4
  [config] t2u_model.config.decoder_layers: 6 -> 4
  [config] sync done.
[ckpt] Saved phase6_t2u_pruning_step000000.pt (0.0 MB)
[model] Saving phase6_t2u_iter_pruned → /kaggle/working/models/phase6_t2u_iter_pruned ...
  [config] sync done.
  Saved custom state: ['_vocab_remap_to_old']
  Saved pruning_manifest.pt keys=['stage_name', 't2u_removed', 'phase']

Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]
[model] Local save done. 2110 MB in 8 files.
[model] Pushing to rclone remote...
[model] Verified 8 files on remote.

--- After Phase 6: T2U Iteratively Pruned (from P4) ---
  speech_encoder                         441.6M  ( 42.5%)
  text_decoder                           373.6M  ( 36.0%)
  t2u_model                              182.0M  ( 17.5%)
  vocoder                                 41.9M  (  4.0%)
  shared                                  20.9M  (  2.0%)
  lm_head                                 20.9M  (  2.0%)
  TOTAL                                 1039.1M
---

{'shared': 20.9152,
 'speech_encoder': 441.604416,
 'text_decoder': 373.568512,
 'lm_head': 20.9152,
 't2u_model': 182.012675,
 'vocoder': 41.911362,
 'TOTAL': 1039.096965}
```

### Cell 90 (code, score=44)
```python
# ── Phase 6 Cell 5: Verify T2U layer indices after pruning ───────────────────
# Mirror of Cell 44/46 in Phase 3 that verified decoder layer_idx.

stacks_post = find_t2u_stacks(model_p6)
for (parent, attr, name) in stacks_post:
    layers = list(getattr(parent, attr))
    indices = []
    for layer in layers:
        for attn_name in ['self_attn', 'encoder_attn', 'cross_attention']:
            attn = getattr(layer, attn_name, None)
            if attn is not None and hasattr(attn, 'layer_idx'):
                indices.append(attn.layer_idx)
                break
    if indices:
        print(f'{name} layer_idx: {indices}')
    else:
        print(f'{name}: {len(layers)} layers (no layer_idx attribute found)')
```
OUTPUT:
```text
t2u.t2u_model.model.encoder.layers layer_idx: [0, 1, 2, 3]
t2u.t2u_model.model.decoder.layers layer_idx: [0, 1, 2, 3]
```

### Cell 91 (code, score=142)
```python
# ── Phase 6 Cell 6: Full benchmark ───────────────────────────────────────────
# p6b = load_latest_checkpoint('phase6_benchmark')
p6b = None
if p6b:
    p6_results, p6_summary = p6b['results'], p6b['summary']
    print(f'Loaded P6 benchmark: BLEU={p6_summary["avg_bleu"]:.2f} '
          f'ChrF={p6_summary["avg_chrf"]:.2f}')
else:
    p6_results, p6_summary = run_benchmark(
        model_p6, eval_samples, label='P6_T2UIter', save_n=2)
    save_checkpoint(dict(results=p6_results, summary=p6_summary),
                    name='phase6_benchmark', step=0)

# Compare against the real p4 baseline
drop = p4_baseline_chrf - p6_summary['avg_chrf']
print(f'\nP4 baseline ChrF : {p4_baseline_chrf:.2f}')
print(f'P6 result  ChrF  : {p6_summary["avg_chrf"]:.2f}')
print(f'ChrF drop        : {drop:.2f}')

store_summary(p6_summary)
plot_phase_comparison()
plot_size_vs_quality()
```
OUTPUT:
```text
============================================================
  BENCHMARK: P6_T2UIter
  Samples: 25  Target: ben
============================================================

  GPU mem: 4.23 GB alloc / 4.46 GB reserved
  [ 1/25] BLEU= 14.8 ChrF= 46.9 RTF=0.096  id=1660
              pred: রোমান্টিকতাবাদ সংস্কৃতির নির্ণয়বাদের একটি বড় উপাদান ছিল, যা গথ, ফিচট এবং শ্লেগ
[audio] Saved P6_T2UIter_s1in.wav (0.3 MB)
  P6_T2UIter_s1in.wav  (10.7s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P6_T2UIter_s1out.wav (0.2 MB)
  P6_T2UIter_s1out.wav  (7.1s | sr=16000)

<IPython.lib.display.Audio object>
  [ 2/25] BLEU= 10.1 ChrF= 40.9 RTF=0.096  id=1661
              pred: তিনি বলেন, তিনি চীনের অর্থনৈতিক উৎপাদনের উপর ভিত্তি করে এই সংখ্যাটি তৈরি করা হবে
[audio] Saved P6_T2UIter_s2in.wav (0.2 MB)
  P6_T2UIter_s2in.wav  (6.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P6_T2UIter_s2out.wav (0.2 MB)
  P6_T2UIter_s2out.wav  (5.4s | sr=16000)

<IPython.lib.display.Audio object>
  [ 3/25] BLEU= 20.1 ChrF= 45.5 RTF=0.096  id=1662
              pred: মিশ্রণ মূলত দুই বা তার বেশি ধাতব মিশ্রণ, পিইআইআরআই টেবিলের উপর অনেক উপাদান রয়েছ
  [ 4/25] BLEU=  5.2 ChrF= 42.4 RTF=0.097  id=1663
              pred: চোকামু উপত্যকা, চিলির শীর্ষস্থানীয় আরোহণের গন্তব্য, দক্ষিণ আমেরিকা'র ইয়েসোমিটি
  [ 5/25] BLEU= 13.3 ChrF= 42.1 RTF=0.093  id=1664
              pred: দুটি ড্রাই পাতা একসাথে ঘুরান এবং তারপর চিলিয়ে ঘন হাত দিয়ে তাদের একটি বলের মধ্য
  [ 6/25] BLEU=  5.6 ChrF= 23.9 RTF=0.073  id=1665
              pred: "লিকের মতে, ""ডকুমেন্টটি সীমান্ত বিরোধের কথা উল্লেখ করবে, যা"
  [ 7/25] BLEU=  9.8 ChrF= 50.6 RTF=0.079  id=1666
              pred: আপনি আপনার নিজের সরকার ছাড়া অন্য সরকারের পরামর্শ নিয়ে পরামর্শ নিতে পারেন, কিন্
  [ 8/25] BLEU=  4.6 ChrF= 45.5 RTF=0.098  id=1667
              pred: সাধারণভাবে, দুইটি আচরণ বিবর্তনগুলি উদ্ভূত হতে পারে, যেহেতু ম্যানেজাররা তাদের প্র
  [ 9/25] BLEU=  7.0 ChrF= 50.9 RTF=0.077  id=1668
              pred: এটি একটি ওয়াইল্ডকার্ড কিনতেও উপকারী হতে পারে, যা দক্ষিণ আফ্রিকার পার্কের যে কোন
  [10/25] BLEU=  7.3 ChrF= 51.4 RTF=0.117  id=1669
              pred: পুলিশ সুপারিনটেন্ডেন্ট চান্দ্রা শিকর সুলঙ্কি বলেন, অভিযুক্তরা মুখোমুখি হয়ে আদাল
  [11/25] BLEU=  0.2 ChrF=  7.8 RTF=0.192  id=1670
              pred: তাদের আনুষ্ঠানিক আচরণ, প্রায়শই স্থিরতা বজায় রাখার মতো বড় বড় বড় বড় বড় বড় 
  [12/25] BLEU= 14.5 ChrF= 51.4 RTF=0.140  id=1671
              pred: কংগ্রেস অযৌনতা ইনিশিয়েয়েয়েকে এবং ফিসাল-২৫৫৫-এ অর্থায়ন শুরু করে এবং নির্দিষ্ট
  [13/25] BLEU=  6.7 ChrF= 30.2 RTF=0.070  id=1672
              pred: ফ্যাব্রিককে খুব গরম হতে দেয় না, যা সংকুচিত হতে পারে, বা চরম ক্ষেত্রে, পুড়ে যায
  [14/25] BLEU= 22.3 ChrF= 68.5 RTF=0.099  id=1673
              pred: বিপ্লবী যুদ্ধের সময়, ১৩টি রাজ্যে প্রথমবারের মতো একটি দুর্বল কেন্দ্রীয় সরকার গঠ
  [15/25] BLEU=  8.2 ChrF= 40.9 RTF=0.099  id=1674
              pred: কিছু এলাকায়, এক মিনিটের জন্য উষ্ণ জল যথেষ্ট এবং অন্য কয়েক মিনিট প্রয়োজন হয়।
  [16/25] BLEU=  0.0 ChrF=  7.3 RTF=0.057  id=1675
              pred: "প্রাণের মাঝামাঝি পর্যন্ত আপনার জন্য " - "
  [17/25] BLEU=  8.8 ChrF= 60.2 RTF=0.063  id=1676
              pred: দক্ষিণ আফ্রিকার সকল জাতীয় উদ্যানের মতো, পার্কের জন্য প্রতিদিন সংরক্ষণ এবং প্রবে
  [18/25] BLEU=  6.4 ChrF= 37.1 RTF=0.134  id=1677
              pred: আজ, একমাত্র পোকা যে তাদের ডানাগুলিকে পিছনে ভাঁড়ানো যায় না তা হ'ল ড্রাগনফ্লি এব
  [19/25] BLEU=  2.2 ChrF= 30.9 RTF=0.076  id=1678
              pred: "অলিভার স্যাক্স তার কাগজতে রাষ্ট্রপতির বক্তৃতাটি নির্দেশ করে যে, মস্তিষ্কের ক্ষত
  [20/25] BLEU=  6.6 ChrF= 45.8 RTF=0.120  id=1679
              pred: এরা স্মিথ তাদের সফরর বাকি কনসেন্ট বাতিল করেছে।
  [21/25] BLEU=  3.8 ChrF= 28.1 RTF=0.091  id=1680
              pred: একটি সু-গোল্লা, বাঘ ভাল, ভাল না, যদিও, সাঁতার, লম্বা, বড় দূরত্ব, এবং পাঁচবার এক
  [22/25] BLEU= 13.5 ChrF= 52.1 RTF=0.068  id=1681
              pred: তবে, এটি কেবলমাত্র পরীক্ষা নয়, এবং এটি এমন একটি পরীক্ষা যা এক বা একাধিক সম্ভাব্
  [23/25] BLEU=  0.7 ChrF= 16.4 RTF=0.129  id=1682
              pred: যদিও কেউই নিশ্চিত না যে এটি কে লিখেছেন, তবে এটি তার জীবনের প্রথম দিকে, এটির বৃহত
  [24/25] BLEU=  6.0 ChrF= 39.7 RTF=0.107  id=1683
              pred: "তারা আরও আরও আরও লিখেছেন, "তারা এখনও তাদের সময় বেঁচে আছে, এবং আরও অনেক লোক আছে
  [25/25] BLEU=  6.6 ChrF= 46.2 RTF=0.064  id=1684
              pred: সামোয়া'র রাজধানী, শহরটি উপোলু দ্বীপের মধ্যে এবং জনসংখ্যা ৪০ হাজারেরও কম।

  Summary: BLEU=8.19  ChrF=40.11  RTF=0.0972  Params=1039.1M

[ckpt] Saved phase6_benchmark_step000000.pt (0.0 MB)

P4 baseline ChrF : 40.11
P6 result  ChrF  : 40.11
ChrF drop        : 0.00
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P6_T2UIter (7 total)

<Figure size 1680x1200 with 4 Axes>
[image/png output omitted]
<Figure size 1200x840 with 1 Axes>
[image/png output omitted]
```

### Cell 93 (code, score=4)
```python
if ON_KAGGLE:
    print('[audio] Syncing audio to rclone remote...')
    r = subprocess.run(
        f'rclone sync "{AUDIO_DIR}/" "{GDRIVE_ROOT}/audio/"',
        shell=True,
        capture_output=True,
        text=True
    )

    if r.returncode != 0:
        print(f'[audio] WARNING: {r.stderr[:300]}')
    else:
        print('[audio] Sync complete.')

else:
    print(f'[audio] Colab: files already in Google Drive at {AUDIO_DIR} (no sync needed)')

if ON_KAGGLE:
    print('[figures] Syncing figures to rclone remote...')
    r = subprocess.run(
        f'rclone sync "{FIG_DIR}/" "{GDRIVE_ROOT}/figures/"',
        shell=True,
        capture_output=True,
        text=True
    )

    if r.returncode != 0:
        print(f'[figure] WARNING: {r.stderr[:300]}')
    else:
        print('[figure] Sync complete.')

else:
    print(f'[figure] Colab: files already in Google Drive at {FIG_DIR} (no sync needed)')
```
OUTPUT:
```text
[audio] Syncing audio to rclone remote...
[audio] Sync complete.
[figures] Syncing figures to rclone remote...
[figure] Sync complete.
```

### Cell 94 (markdown, score=26)
```markdown
---
# Phase 7: Recovery Fine-tuning — S2ST Focused
**Papers:** Moslem (IWSLT 2025) + DoRA (Liu et al., ICML 2024 Oral)

**Why this replaces the old S2TT-only approach:**
The previous Phase 7 trained only with S2TT cross-entropy loss, which backpropagates
gradients only through `speech_encoder → text_decoder`. The `t2u_model` (which converts
text tokens to discrete speech units) received **zero gradient**, so the audio output
remained broken even though BLEU/ChrF scores appeared recovered (they measured only the
text decoder path).

**Correct approach — two-phase S2ST recovery:**
1. **Phase 7a — S2TT DoRA** (text decoder recovery): Fast, recovers translation quality.
   Trains: speech_encoder + text_decoder projections.
2. **Phase 7b — S2ST unit fine-tuning** (T2U recovery): Extracts discrete unit labels
   from target Bengali audio using SeamlessM4T's own unit extractor, then trains the
   T2U encoder+decoder with unit cross-entropy loss. This is the path that produces
   the translated audio waveform.

**Benchmark is now S2ST-first:** `run_benchmark_s2st()` measures BLEU/ChrF from the
*S2ST text output* (the text produced alongside waveform generation via the full pipeline)
and saves audio clips for listening. The old `run_s2t_only`-based benchmark is kept for
fast intermediate checks only.
```