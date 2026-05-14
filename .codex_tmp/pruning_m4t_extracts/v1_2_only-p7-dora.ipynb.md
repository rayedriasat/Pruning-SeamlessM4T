# Final Notebooks\v1_2 only-p7-dora.ipynb

Extracted notebook map containing markdown headings plus code/output cells likely to matter for reports, reproduction, or agent steering.

## Markdown headings
cell 1: # SeamlessM4T v2 Large: Structured Compression 2.3B to ~1B ## Compression Pipeline
cell 3: ## Setup Cells 1-8
cell 20: ## Core Library: Model, Benchmark, Plotting
cell 26: # Phase 0: Baseline Benchmark
cell 31: # Phase 7: Recovery Fine-tuning with DoRA
cell 39: # Phase 8: Final Results + Paper Table

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

### Cell 12 (code, score=37)
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 50.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.1/84.1 kB 7.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.8/100.8 kB 8.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 101.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.6/121.6 kB 10.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 788.2/788.2 kB 48.5 MB/s eta 0:00:00
All packages installed.
```

### Cell 13 (code, score=78)
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
16 file(s) found:
  all_summaries_step000000.pt                                  0.0 MB
  phase0_baseline_step000000.pt                                0.0 MB
  phase1_benchmark_step000000.pt                               0.0 MB
  phase1_vocab_step000000.pt                                   0.1 MB
  phase3_benchmark_step000000.pt                               0.0 MB
  phase3_dec_pruning_step000000.pt                             0.0 MB
  phase4_benchmark_step000000.pt                               0.0 MB
  phase4_enc_pruning_step000000.pt                             0.0 MB
  phase5_benchmark(base)_step000000.pt                         0.0 MB
  phase5_benchmark(m4)_step000000.pt                           0.0 MB
  phase5_flap(base)_step000000.pt                              0.0 MB
  phase5_flap(m4)_step000000.pt                                0.0 MB
  phase6_benchmark_step000000.pt                               0.0 MB
  phase6_t2u_pruning_step000000.pt                             0.0 MB
  phase6_t2u_t2u_model_model_decoder_layers_pruning_step000000.pt      0.0 MB
  phase6_t2u_t2u_model_model_encoder_layers_pruning_step000000.pt      0.0 MB
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

### Cell 16 (code, score=67)
```python
sync_checkpoints_from_drive()
```
OUTPUT:
```text
[ckpt] Syncing checkpoints from rclone remote...
[ckpt] 16 checkpoint(s) available
  all_summaries_step000000.pt                                 0.0 MB
  phase0_baseline_step000000.pt                               0.0 MB
  phase1_benchmark_step000000.pt                              0.0 MB
  phase1_vocab_step000000.pt                                  0.1 MB
  phase3_benchmark_step000000.pt                              0.0 MB
  phase3_dec_pruning_step000000.pt                            0.0 MB
  phase4_benchmark_step000000.pt                              0.0 MB
  phase4_enc_pruning_step000000.pt                            0.0 MB
  phase5_benchmark(base)_step000000.pt                        0.0 MB
  phase5_benchmark(m4)_step000000.pt                          0.0 MB
  phase5_flap(base)_step000000.pt                             0.0 MB
  phase5_flap(m4)_step000000.pt                               0.0 MB
  phase6_benchmark_step000000.pt                              0.0 MB
  phase6_t2u_pruning_step000000.pt                            0.0 MB
  phase6_t2u_t2u_model_model_decoder_layers_pruning_step000000.pt     0.0 MB
  phase6_t2u_t2u_model_model_encoder_layers_pruning_step000000.pt     0.0 MB
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

### Cell 19 (code, score=87)
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
  Platform : kaggle   Time : 2026-04-19 08:43
  Checkpoint files in /kaggle/working/checkpoints: 16
    all_summaries_step000000.pt                             0.0 MB
    phase0_baseline_step000000.pt                           0.0 MB
    phase1_benchmark_step000000.pt                          0.0 MB
    phase1_vocab_step000000.pt                              0.1 MB
    phase3_benchmark_step000000.pt                          0.0 MB
    phase3_dec_pruning_step000000.pt                        0.0 MB
    phase4_benchmark_step000000.pt                          0.0 MB
    phase4_enc_pruning_step000000.pt                        0.0 MB
    phase5_benchmark(base)_step000000.pt                    0.0 MB
    phase5_benchmark(m4)_step000000.pt                      0.0 MB
    phase5_flap(base)_step000000.pt                         0.0 MB
    phase5_flap(m4)_step000000.pt                           0.0 MB
    phase6_benchmark_step000000.pt                          0.0 MB
    phase6_t2u_pruning_step000000.pt                        0.0 MB
    phase6_t2u_t2u_model_model_decoder_layers_pruning_step000000.pt      0.0 MB
    phase6_t2u_t2u_model_model_encoder_layers_pruning_step000000.pt      0.0 MB
  GPU: Tesla T4
  VRAM: 15.6 GB
============================================================
```

### Cell 20 (markdown, score=1)
```markdown
## Core Library: Model, Benchmark, Plotting
```

### Cell 21 (code, score=31)
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

### Cell 22 (code, score=72)
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

### Cell 23 (code, score=145)
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

### Cell 24 (code, score=46)
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

### Cell 25 (code, score=92)
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
Loaded 7 existing summaries: ['P0_Baseline', 'P1_VocabTrim', 'P3_DecPrune', 'P4_EncPrune', 'P5_FLAP(base)', 'P5_FLAP(m4)', 'P6_T2UIter']
Plotting helpers ready.
```

### Cell 26 (markdown, score=5)
```markdown
---
# Phase 0: Baseline Benchmark
Load the full teacher model, measure size and translation quality.
```

### Cell 27 (code, score=87)
```python
## Dataset loading
LOCAL_PARQUET_CACHE = "/kaggle/input/datasets/coderayed/fleurs-en-bn-parquet"

import concurrent.futures

BASE_PARQUET_URL = (
    "https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet"
)

def _list_parquet_urls(lang, split):
    return [f"{BASE_PARQUET_URL}/{lang}/{split}/0000.parquet?download=true"]

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

### Cell 28 (code, score=71)
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

30
```

### Cell 29 (code, score=73)
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

### Cell 31 (markdown, score=8)
```markdown
---
# Phase 7: Recovery Fine-tuning with DoRA
**Paper:** DoRA (Liu et al., ICML 2024 Oral)

Fine-tune using S2TT cross-entropy loss + DoRA for memory-efficient recovery of pruned model quality.
```

### Cell 32 (code, score=72)
```python
model_p6, processor = load_model_from_drive('phase6_t2u_iter_pruned')
print_model_breakdown(model_p6, 'Phase 6 (input to Phase 7)')
model_p6 = _consolidate_to_single_gpu(model_p6)
# save_model_to_drive(model_p6, processor, 'phase6_t2u_iter_pruned')
```
OUTPUT:
```text
[model] Not in local cache, pulling from remote...
[rclone] Pulled phase6_t2u_iter_pruned → /kaggle/working/models/phase6_t2u_iter_pruned
[model] Loading phase6_t2u_iter_pruned from /kaggle/working/models/phase6_t2u_iter_pruned ...

Instantiating a decoder SeamlessM4Tv2Attention without passing `layer_idx` is not recommended and will lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` when creating this class.

Loading weights:   0%|          | 0/1266 [00:00<?, ?it/s]
  Restored custom state: ['_vocab_remap_to_old']
  [model] pruning_manifest: ['stage_name', 't2u_removed', 'phase']

--- Phase 6 (input to Phase 7) ---
  speech_encoder                         441.6M  ( 42.5%)
  text_decoder                           373.6M  ( 36.0%)
  t2u_model                              182.0M  ( 17.5%)
  vocoder                                 41.9M  (  4.0%)
  shared                                  20.9M  (  2.0%)
  lm_head                                 20.9M  (  2.0%)
  TOTAL                                 1039.1M
---
  Multi-device map detected, consolidating to cuda:0...
  Model now on: cuda:0
```

### Cell 33 (code, score=31)
```python
subprocess.run(['pip', 'install', '-q', 'peft>=0.10.0'], check=True)

from peft import LoraConfig, get_peft_model, TaskType

def discover_lora_targets(mdl, scope_keywords=('text_decoder', 't2u_model', 'speech_encoder')):
    found_by_scope = {}
    for name, mod in mdl.named_modules():
        if not isinstance(mod, nn.Linear): continue
        scope = next((kw for kw in scope_keywords if kw in name), None)
        if scope is None: continue
        leaf = name.split('.')[-1]
        found_by_scope.setdefault(scope, set()).add(leaf)
    print("Linear layer leaf names by scope:")
    all_leaves = set()
    for scope, leaves in sorted(found_by_scope.items()):
        print(f"  {scope}: {sorted(leaves)}")
        all_leaves |= leaves
    attn_ffn_candidates = {
        'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2',
    }
    targets = sorted(all_leaves & attn_ffn_candidates)
    count = sum(1 for name, mod in mdl.named_modules()
                if isinstance(mod, nn.Linear)
                and name.split('.')[-1] in targets
                and any(kw in name for kw in scope_keywords))
    print(f"\nTarget modules: {targets}  ({count} Linear layers)")
    return targets

targets = discover_lora_targets(model_p6)
```
OUTPUT:
```text
Linear layer leaf names by scope:
  speech_encoder: ['intermediate_dense', 'linear_k', 'linear_out', 'linear_q', 'linear_v', 'output_dense', 'projection']
  t2u_model: ['fc1', 'fc2', 'k_proj', 'lm_head', 'out_proj', 'proj', 'q_proj', 'v_proj']
  text_decoder: ['fc1', 'fc2', 'k_proj', 'out_proj', 'q_proj', 'v_proj']

Target modules: ['fc1', 'fc2', 'k_proj', 'out_proj', 'q_proj', 'v_proj']  (180 Linear layers)
```

### Cell 34 (code, score=458)
```python
LORA_R     = 16
LORA_ALPHA = 32
LORA_DROP  = 0.05

lora_cfg = LoraConfig(
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROP,
    bias           = 'none',
    use_dora       = True,
    target_modules = targets,
)

model_p7 = get_peft_model(model_p6, lora_cfg)
model_p7.print_trainable_parameters()

model_p7 = _consolidate_to_single_gpu(model_p7)
model_p7.train()
```
OUTPUT:
```text
trainable params: 10,340,352 || all params: 1,049,437,317 || trainable%: 0.9853

PeftModel(
  (base_model): LoraModel(
    (model): SeamlessM4Tv2ForSpeechToSpeech(
      (shared): Embedding(20425, 1024, padding_idx=0)
      (speech_encoder): SeamlessM4Tv2SpeechEncoder(
        (feature_projection): SeamlessM4Tv2ConformerFeatureProjection(
          (layer_norm): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
          (projection): Linear(in_features=160, out_features=1024, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (encoder): SeamlessM4Tv2ConformerEncoder(
          (dropout): Dropout(p=0.0, inplace=False)
          (layers): ModuleList(
            (0-15): 16 x SeamlessM4Tv2ConformerEncoderLayer(
              (ffn1_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (ffn1): SeamlessM4Tv2ConformerFeedForward(
                (intermediate_dropout): Dropout(p=0.0, inplace=False)
                (intermediate_dense): Linear(in_features=1024, out_features=4096, bias=True)
                (intermediate_act_fn): SiLU()
                (output_dense): Linear(in_features=4096, out_features=1024, bias=True)
                (output_dropout): Dropout(p=0.0, inplace=False)
              )
              (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (self_attn_dropout): Dropout(p=0.0, inplace=False)
              (self_attn): SeamlessM4Tv2ConformerSelfAttention(
                (linear_q): Linear(in_features=1024, out_features=1024, bias=True)
                (linear_k): Linear(in_features=1024, out_features=1024, bias=True)
                (linear_v): Linear(in_features=1024, out_features=1024, bias=True)
                (linear_out): Linear(in_features=1024, out_features=1024, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (distance_embedding): Embedding(73, 64)
              )
              (conv_module): SeamlessM4Tv2ConformerConvolutionModule(
                (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (pointwise_conv1): Conv1d(1024, 2048, kernel_size=(1,), stride=(1,), bias=False)
                (glu): GLU(dim=1)
                (depthwise_conv): Conv1d(1024, 1024, kernel_size=(31,), stride=(1,), groups=1024, bias=False)
                (depthwise_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (activation): SiLU()
                (pointwise_conv2): Conv1d(1024, 1024, kernel_size=(1,), stride=(1,), bias=False)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (ffn2_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (ffn2): SeamlessM4Tv2ConformerFeedForward(
                (intermediate_dropout): Dropout(p=0.0, inplace=False)
                (intermediate_dense): Linear(in_features=1024, out_features=4096, bias=True)
                (intermediate_act_fn): SiLU()
                (output_dense): Linear(in_features=4096, out_features=1024, bias=True)
                (output_dropout): Dropout(p=0.0, inplace=False)
              )
              (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            )
          )
          (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
        (intermediate_ffn): SeamlessM4Tv2ConformerFeedForward(
          (intermediate_dropout): Dropout(p=0.0, inplace=False)
          (intermediate_dense): Linear(in_features=1024, out_features=4096, bias=True)
          (intermediate_act_fn): ReLU()
          (output_dense): Linear(in_features=4096, out_features=1024, bias=True)
          (output_dropout): Dropout(p=0.0, inplace=False)
        )
        (adapter): SeamlessM4Tv2ConformerAdapter(
          (layers): ModuleList(
            (0): SeamlessM4Tv2ConformerAdapterLayer(
              (residual_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (residual_conv): Conv1d(1024, 2048, kernel_size=(8,), stride=(8,), padding=(4,))
              (activation): GLU(dim=1)
              (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (self_attn_conv): Conv1d(1024, 2048, kernel_size=(8,), stride=(8,), padding=(4,))
              (self_attn): SeamlessM4Tv2ConformerSelfAttention(
                (linear_q): Linear(in_features=1024, out_features=1024, bias=True)
                (linear_k): Linear(in_features=1024, out_features=1024, bias=True)
                (linear_v): Linear(in_features=1024, out_features=1024, bias=True)
                (linear_out): Linear(in_features=1024, out_features=1024, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (self_attn_dropout): Dropout(p=0.1, inplace=False)
              (ffn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (ffn): SeamlessM4Tv2ConformerFeedForward(
                (intermediate_dropout): Dropout(p=0.1, inplace=False)
                (intermediate_dense): Linear(in_features=1024, out_features=4096, bias=True)
                (intermediate_act_fn): ReLU()
                (output_dense): Linear(in_features=4096, out_features=1024, bias=True)
                (output_dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (inner_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
      (text_decoder): SeamlessM4Tv2Decoder(
        (embed_tokens): SeamlessM4Tv2ScaledWordEmbedding(20425, 1024, padding_idx=0)
        (embed_positions): SeamlessM4Tv2SinusoidalPositionalEmbedding()
        (layers): ModuleList(
          (0-13): 14 x SeamlessM4Tv2DecoderLayer(
            (self_attn): SeamlessM4Tv2Attention(
              (k_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict(
                  (default): lora.dora.DoraLinearLayer()
                )
              )
              (v_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict(
                  (default): lora.dora.DoraLinearLayer()
                )
              )
              (q_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict(
                  (default): lora.dora.DoraLinearLayer()
                )
              )
              (out_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict(
                  (default): lora.dora.DoraLinearLayer()
                )
              )
            )
            (activation_fn): ReLU()
            (attn_dropout): Dropout(p=0.1, inplace=False)
            (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (cross_attention): SeamlessM4Tv2Attention(
              (k_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict(
                  (default): lora.dora.DoraLinearLayer()
                )
              )
              (v_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict(
                  (default): lora.dora.DoraLinearLayer()
                )
              )
              (q_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict(
                  (default): lora.dora.DoraLinearLayer()
                )
              )
              (out_proj): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict(
                  (default): lora.dora.DoraLinearLayer()
                )
              )
            )
            (cross_attention_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (ffn): SeamlessM4Tv2FeedForwardNetwork(
              (fc1): lora.Linear(
                (base_layer): Linear(in_features=1024, out_features=8192, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=8192, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict(
                  (default): lora.dora.DoraLinearLayer()
                )
              )
              (fc2): lora.Linear(
                (base_layer): Linear(in_features=8192, out_features=1024, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=8192, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict(
                  (default): lora.dora.DoraLinearLayer()
                )
              )
              (dropout): Dropout(p=0.0,
```

### Cell 35 (code, score=199)
```python
import torch
import random
import time
import logging
import gc as _stdlib_gc

MAX_STEPS  = 2500
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR         = 3e-4
GRAD_CLIP  = 1.0
LOG_EVERY  = 50
SAVE_EVERY = 250

trainable = [p for p in model_p7.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(trainable, lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS)

def prepare_s2tt_batch(batch, processor, device, tgt_lang, mdl):
    audios = [s['wav'] for s in batch]
    targets = [s['ref'] for s in batch]
    audio_enc = processor(audio=audios, sampling_rate=16000, return_tensors='pt', padding=True)
    input_feats = audio_enc['input_features'].to(device)
    attn_mask = audio_enc['attention_mask'].to(device)
    tok = processor.tokenizer
    text_enc = tok(text_target=targets, tgt_lang=tgt_lang, return_tensors='pt', padding=True)
    labels = text_enc['input_ids'].to(device)
    pad = tok.pad_token_id
    if pad is not None:
        labels = labels.masked_fill(labels == pad, -100)
    labels = remap_label_ids(labels, mdl)
    return input_feats, attn_mask, labels

def compute_s2tt_loss(model, input_feats, attn_mask, labels):
    outputs = model(input_features=input_feats, attention_mask=attn_mask,
                    labels=labels, return_dict=True)
    return outputs.loss

ft_ckpt = load_latest_checkpoint('phase7_ft')
start_step = 0
loss_log = []

if ft_ckpt and ft_ckpt.get('step', 0) > 0:
    start_step = ft_ckpt['step']
    loss_log = ft_ckpt.get('loss_log', [])
    ostate = ft_ckpt.get('optimizer_state') or ft_ckpt.get('opt')
    sstate = ft_ckpt.get('scheduler_state') or ft_ckpt.get('sched')
    if ostate: optimizer.load_state_dict(ostate)
    if sstate: scheduler.load_state_dict(sstate)
    print(f'Resuming from step {start_step}')
else:
    print("Starting Phase 7 from scratch.")

_m4t_train_log = logging.getLogger(
    'transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2')
_prev_hf_level = _m4t_train_log.level
_m4t_train_log.setLevel(logging.ERROR)
try:
    model_p7.train()
    device = next(model_p7.parameters()).device
    optim_steps = start_step
    micro_step = 0
    consecutive_errors = 0
    optimizer.zero_grad()
    t0 = time.time()

    while optim_steps < MAX_STEPS:
        batch = random.sample(ft_samples, min(BATCH_SIZE, len(ft_samples)))
        try:
            input_feats, attn_mask, labels = prepare_s2tt_batch(
                batch, processor, device, TARGET_LANG, model_p7)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = compute_s2tt_loss(model_p7, input_feats, attn_mask, labels)
                loss = loss / GRAD_ACCUM
            loss.backward()
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            print(f'  [ERR] Step {optim_steps}: {e}')
            if consecutive_errors > 5:
                print('CRITICAL: Too many errors.')
                break
            continue

        loss_log.append(loss.item() * GRAD_ACCUM)

        if (micro_step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optim_steps += 1

            if optim_steps % LOG_EVERY == 0:
                avg_loss = sum(loss_log[-LOG_EVERY:]) / LOG_EVERY
                elapsed = time.time() - t0
                print(f'Step {optim_steps}/{MAX_STEPS} | loss: {avg_loss:.4f} | elapsed: {elapsed/60:.1f}min')

            if optim_steps % SAVE_EVERY == 0:
                model_p7.save_pretrained(f'{MODEL_DIR}/phase7_dora_adapter')
                save_checkpoint(
                    dict(step=optim_steps, loss_log=loss_log,
                         optimizer_state=optimizer.state_dict(),
                         scheduler_state=scheduler.state_dict()),
                    name='phase7_ft', step=optim_steps)

        micro_step += 1

    print(f'\nTraining complete. Final step: {optim_steps}  Total time: {(time.time()-t0)/60:.1f} min')
    model_p7.save_pretrained(f'{MODEL_DIR}/phase7_dora_adapter')
    save_checkpoint(
        dict(step=optim_steps, loss_log=loss_log,
             optimizer_state=optimizer.state_dict(),
             scheduler_state=scheduler.state_dict()),
        name='phase7_ft', step=optim_steps)
finally:
    _m4t_train_log.setLevel(_prev_hf_level)
```
OUTPUT:
```text
[ckpt] No checkpoint for 'phase7_ft'
Starting Phase 7 from scratch.
Step 50/2500 | loss: 5.9578 | elapsed: 3.6min
Step 100/2500 | loss: 3.8861 | elapsed: 7.5min
Step 150/2500 | loss: 3.5615 | elapsed: 11.4min
Step 200/2500 | loss: 3.0916 | elapsed: 15.4min
Step 250/2500 | loss: 2.9988 | elapsed: 19.2min
[ckpt] Saved phase7_ft_step000250.pt (64.7 MB)
Step 300/2500 | loss: 2.6668 | elapsed: 23.6min
Step 350/2500 | loss: 2.6807 | elapsed: 27.4min
Step 400/2500 | loss: 2.4485 | elapsed: 31.1min
Step 450/2500 | loss: 2.4662 | elapsed: 34.9min
Step 500/2500 | loss: 2.3160 | elapsed: 38.8min
[ckpt] Saved phase7_ft_step000500.pt (64.7 MB)
Step 550/2500 | loss: 2.2929 | elapsed: 42.7min
Step 600/2500 | loss: 2.3755 | elapsed: 46.6min
Step 650/2500 | loss: 2.3123 | elapsed: 50.4min
Step 700/2500 | loss: 2.2579 | elapsed: 54.3min
Step 750/2500 | loss: 2.0619 | elapsed: 58.0min
[ckpt] Saved phase7_ft_step000750.pt (64.7 MB)
Step 800/2500 | loss: 2.2597 | elapsed: 61.9min
Step 850/2500 | loss: 2.1205 | elapsed: 65.7min
Step 900/2500 | loss: 2.1786 | elapsed: 69.6min
Step 950/2500 | loss: 2.2055 | elapsed: 73.4min
Step 1000/2500 | loss: 2.0664 | elapsed: 77.2min
[ckpt] Saved phase7_ft_step001000.pt (64.7 MB)
Step 1050/2500 | loss: 2.0606 | elapsed: 81.9min
Step 1100/2500 | loss: 2.2489 | elapsed: 85.7min
Step 1150/2500 | loss: 2.0166 | elapsed: 89.4min
Step 1200/2500 | loss: 2.0961 | elapsed: 93.4min
Step 1250/2500 | loss: 2.0528 | elapsed: 97.2min
[ckpt] Saved phase7_ft_step001250.pt (64.8 MB)
Step 1300/2500 | loss: 2.1048 | elapsed: 101.4min
Step 1350/2500 | loss: 2.1347 | elapsed: 105.3min
Step 1400/2500 | loss: 1.9568 | elapsed: 109.2min
Step 1450/2500 | loss: 2.0711 | elapsed: 113.0min
Step 1500/2500 | loss: 1.9723 | elapsed: 116.8min
[ckpt] Saved phase7_ft_step001500.pt (64.8 MB)
Step 1550/2500 | loss: 2.0834 | elapsed: 120.7min
Step 1600/2500 | loss: 1.9303 | elapsed: 124.6min
Step 1650/2500 | loss: 1.8952 | elapsed: 128.4min
Step 1700/2500 | loss: 2.0044 | elapsed: 132.3min
Step 1750/2500 | loss: 2.0161 | elapsed: 136.2min
[ckpt] Saved phase7_ft_step001750.pt (64.8 MB)
Step 1800/2500 | loss: 1.8893 | elapsed: 140.4min
Step 1850/2500 | loss: 2.0325 | elapsed: 144.2min
Step 1900/2500 | loss: 1.9008 | elapsed: 148.0min
Step 1950/2500 | loss: 1.7931 | elapsed: 151.9min
Step 2000/2500 | loss: 1.9636 | elapsed: 155.8min
[ckpt] Saved phase7_ft_step002000.pt (64.8 MB)
Step 2050/2500 | loss: 1.8895 | elapsed: 159.7min
Step 2100/2500 | loss: 1.9524 | elapsed: 163.7min
Step 2150/2500 | loss: 2.0062 | elapsed: 167.7min
Step 2200/2500 | loss: 1.9206 | elapsed: 171.6min
Step 2250/2500 | loss: 1.7829 | elapsed: 175.4min
[ckpt] Saved phase7_ft_step002250.pt (64.8 MB)
Step 2300/2500 | loss: 1.8205 | elapsed: 179.4min
Step 2350/2500 | loss: 1.9449 | elapsed: 183.3min
Step 2400/2500 | loss: 1.8622 | elapsed: 187.2min
Step 2450/2500 | loss: 1.8643 | elapsed: 191.0min
Step 2500/2500 | loss: 1.9909 | elapsed: 194.9min
[ckpt] Saved phase7_ft_step002500.pt (64.8 MB)

Training complete. Final step: 2500  Total time: 195.0 min
[ckpt] Saved phase7_ft_step002500.pt (64.8 MB)
```

### Cell 36 (code, score=125)
```python
import gc as _stdlib_gc

print('Merging DoRA adapters into base model...')
model_p7_merged = model_p7.merge_and_unload()
model_p7_merged.eval()
_stdlib_gc.collect(); torch.cuda.empty_cache()
print('Merge complete.')

sync_model_config(model_p7_merged)
# Peft merge can leave config.decoder_layers stale vs pruned ModuleList → bad saves / missing keys on load.
if hasattr(model_p7_merged, 'text_decoder') and getattr(model_p7_merged.text_decoder, 'layers', None) is not None:
    n = len(model_p7_merged.text_decoder.layers)
    if getattr(model_p7_merged.config, 'decoder_layers', None) != n:
        print(f'  [phase7 merge] decoder_layers {model_p7_merged.config.decoder_layers} -> {n} (ModuleList)')
        model_p7_merged.config.decoder_layers = n
if hasattr(model_p7_merged, 'speech_encoder') and getattr(model_p7_merged.speech_encoder, 'encoder', None):
    enc = model_p7_merged.speech_encoder.encoder
    if getattr(enc, 'layers', None) is not None:
        n = len(enc.layers)
        if getattr(model_p7_merged.config, 'speech_encoder_layers', None) != n:
            print(f'  [phase7 merge] speech_encoder_layers {model_p7_merged.config.speech_encoder_layers} -> {n}')
            model_p7_merged.config.speech_encoder_layers = n
t2u_e = getattr(getattr(model_p7_merged.t2u_model, 'model', None), 'encoder', None)
t2u_d = getattr(getattr(model_p7_merged.t2u_model, 'model', None), 'decoder', None)
if t2u_e is not None and getattr(t2u_e, 'layers', None) is not None:
    n = len(t2u_e.layers)
    if getattr(model_p7_merged.config, 't2u_encoder_layers', None) != n:
        print(f'  [phase7 merge] t2u_encoder_layers {model_p7_merged.config.t2u_encoder_layers} -> {n}')
        model_p7_merged.config.t2u_encoder_layers = n
if t2u_d is not None and getattr(t2u_d, 'layers', None) is not None:
    n = len(t2u_d.layers)
    if getattr(model_p7_merged.config, 't2u_decoder_layers', None) != n:
        print(f'  [phase7 merge] t2u_decoder_layers {model_p7_merged.config.t2u_decoder_layers} -> {n}')
        model_p7_merged.config.t2u_decoder_layers = n

save_model_to_drive(model_p7_merged, processor, 'phase7_dora_merged_v1')
print_model_breakdown(model_p7_merged, 'After Phase 7: DoRA Fine-tuned & Merged')
```
OUTPUT:
```text
Merging DoRA adapters into base model...
Merge complete.
  [config] sync done.
[model] Saving phase7_dora_merged_v1 → /kaggle/working/models/phase7_dora_merged_v1 ...
  [config] sync done.
  Saved custom state: ['_vocab_remap_to_old']
  Saved pruning_manifest.pt keys=['stage_name']

Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]
[model] Local save done. 2110 MB in 8 files.
[model] Pushing to rclone remote...
[model] Verified 8 files on remote.

--- After Phase 7: DoRA Fine-tuned & Merged ---
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

### Cell 37 (code, score=41)
```python
ft_ckpt = load_latest_checkpoint('phase7_ft')
if ft_ckpt and ft_ckpt.get('loss_log'):
    losses = ft_ckpt['loss_log']
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(losses, alpha=0.25, color='steelblue', lw=0.5, label='Raw')
    ema, val = [], losses[0]
    for l in losses:
        val = 0.05 * l + 0.95 * val
        ema.append(val)
    ax.plot(ema, color='steelblue', lw=2, label='EMA')
    ax.set_xlabel('Step'); ax.set_ylabel('S2TT Cross-Entropy Loss')
    ax.set_title('Phase 7: DoRA Fine-tuning Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, 'phase7_loss.png')
    plt.show()
```
OUTPUT:
```text
[ckpt] Loaded phase7_ft_step002500.pt

<Figure size 1440x480 with 1 Axes>
[image/png output omitted]
```

### Cell 38 (code, score=191)
```python
p7b = load_latest_checkpoint('phase7_benchmark')
if p7b:
    p7_results, p7_summary = p7b['results'], p7b['summary']
    print(f'Loaded P7 benchmark: BLEU={p7_summary["avg_bleu"]:.2f}  '
          f'ChrF={p7_summary["avg_chrf"]:.2f}')
else:
    p7_results, p7_summary = run_benchmark(
        model_p7_merged, eval_samples, label='P7_DoRA', save_n=4)
    save_checkpoint(dict(results=p7_results, summary=p7_summary),
                    name='phase7_benchmark', step=0)

p4b = load_latest_checkpoint('phase4_benchmark')
p6b = load_latest_checkpoint('phase6_benchmark')
p4_chrf = p4b['summary']['avg_chrf'] if p4b else 0.0
p6_chrf = p6b['summary']['avg_chrf'] if p6b else 0.0
p7_chrf = p7_summary['avg_chrf']

print(f'\n{"="*55}')
print(f'  Phase 4 ChrF : {p4_chrf:.2f}')
print(f'  Phase 6 ChrF : {p6_chrf:.2f}  (drop: {p4_chrf - p6_chrf:.2f})')
print(f'  Phase 7 ChrF : {p7_chrf:.2f}  (recovery: +{p7_chrf - p6_chrf:.2f})')
print(f'{"="*55}')

store_summary(p7_summary)
```
OUTPUT:
```text
[ckpt] No checkpoint for 'phase7_benchmark'

============================================================
  BENCHMARK: P7_DoRA
  Samples: 25  Target: ben
============================================================

  GPU mem: 2.23 GB alloc / 2.37 GB reserved
  [ 1/25] BLEU= 14.8 ChrF= 49.4 RTF=0.182  id=1660
              pred: রোমান্টিকতার মধ্যে সংস্কৃতির নির্ধারকতা এর একটি বড় উপাদান ছিল যা গথ ফিচ এবং শ্ল
[audio] Saved P7_DoRA_s1in.wav (0.3 MB)
  P7_DoRA_s1in.wav  (10.7s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P7_DoRA_s1out.wav (0.2 MB)
  P7_DoRA_s1out.wav  (6.4s | sr=16000)

<IPython.lib.display.Audio object>
  [ 2/25] BLEU=  6.2 ChrF= 39.5 RTF=0.127  id=1661
              pred: তিনি চীনের অর্থনৈতিক উৎপাদনের উপর ভিত্তি করে কাট করার জন্য কোনও সংখ্যা নির্ধারণ 
[audio] Saved P7_DoRA_s2in.wav (0.2 MB)
  P7_DoRA_s2in.wav  (6.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P7_DoRA_s2out.wav (0.2 MB)
  P7_DoRA_s2out.wav  (5.0s | sr=16000)

<IPython.lib.display.Audio object>
  [ 3/25] BLEU= 10.8 ChrF= 45.4 RTF=0.112  id=1662
              pred: অ্যালোয়ি মূলত 2 বা একাধিক ধাতুর মিশ্রণ মনে রাখবেন না যে পিআইআর তে অনেক উপাদান র
[audio] Saved P7_DoRA_s3in.wav (0.3 MB)
  P7_DoRA_s3in.wav  (8.4s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P7_DoRA_s3out.wav (0.1 MB)
  P7_DoRA_s3out.wav  (4.5s | sr=16000)

<IPython.lib.display.Audio object>
  [ 4/25] BLEU=  6.2 ChrF= 46.2 RTF=0.108  id=1663
              pred: চোকামো উপত্যকা চিলির শীর্ষস্থানীয় পর্বতারোহণের গন্তব্য যা দক্ষিণ আমেরিকার য়োসি
[audio] Saved P7_DoRA_s4in.wav (0.4 MB)
  P7_DoRA_s4in.wav  (12.1s | sr=16000)

<IPython.lib.display.Audio object>
[audio] Saved P7_DoRA_s4out.wav (0.3 MB)
  P7_DoRA_s4out.wav  (8.0s | sr=16000)

<IPython.lib.display.Audio object>
  [ 5/25] BLEU=  8.8 ChrF= 48.0 RTF=0.107  id=1664
              pred: দুটি শুকনো শক্তি একসাথে ঘূর্ণিয়ে তারপর পরিষ্কার পাতলা হাত দিয়ে তাদের একটি বলে 
  [ 6/25] BLEU=  8.5 ChrF= 40.3 RTF=0.126  id=1665
              pred: লিক অনুসারে নথিটি সীমান্ত বিবাদের দিকে উল্লেখ করবে যা পেলেস্টাইন 1969 এর মধ্য-পূ
  [ 7/25] BLEU=  6.8 ChrF= 38.7 RTF=0.105  id=1666
              pred: আপনার নিজের নয় এমন সরকারের পরামর্শের সাথে পরামর্শ নিতে আপনি হয়তো আগ্রহী কিন্তু
  [ 8/25] BLEU=  7.5 ChrF= 51.9 RTF=0.094  id=1667
              pred: সাধারণভাবে বলতে গেলে ম্যানেজাররা তাদের প্রাক্তন সমনীতিকে নেতৃত্ব দিতে শুরু করলে 
  [ 9/25] BLEU= 12.6 ChrF= 51.1 RTF=0.090  id=1668
              pred: একজনের জন্য ওয়াইল্ডকার্ড কেনাও উপকারী হতে পারে যা দক্ষিণ আফ্রিকার পার্কগুলির কো
  [10/25] BLEU= 12.1 ChrF= 57.7 RTF=0.147  id=1669
              pred: পুলিশ সুপারিনটেনডেন্ট চান্দ্রা শিকার সোলাঙ্কি বলেছিলেন যে অভিযুক্তরা মুখের ঢাকা 
  [11/25] BLEU=  6.3 ChrF= 35.9 RTF=0.073  id=1670
              pred: তাদের আনুষ্ঠানিক আচরণ পৃথিবীতে বড় বড় ক্যাসের মতো ধারাবাহিক নয় তবে এগুলি ভূখণ্
  [12/25] BLEU= 17.7 ChrF= 57.2 RTF=0.122  id=1671
              pred: কংগ্রেস 2005 সালে অশ্লীলতামূলক পদক্ষেপটি অর্থায়ন শুরু করে এবং নির্দিষ্ট করে যে 
  [13/25] BLEU=  5.9 ChrF= 28.8 RTF=0.099  id=1672
              pred: ফ্যাব্রিককে খুব গরম হতে না দেয়ার ব্যাপারে সাবধান যেটি সংকোচন বা চরম অবস্থায় দা
  [14/25] BLEU= 42.0 ChrF= 76.2 RTF=0.104  id=1673
              pred: বিপ্লবী যুদ্ধের সময় ১৩টি রাজ্য প্রথম একটি দুর্বল কেন্দ্রীয় সরকার গঠন করেছিল যে
  [15/25] BLEU=  4.0 ChrF= 30.9 RTF=0.140  id=1674
              pred: কিছু অঞ্চলে এক মিনিটের জন্য উষ্ণ জল যথেষ্ট এবং অন্য কয়েক মিনিটের প্রয়োজন হয়
  [16/25] BLEU=  7.3 ChrF= 31.2 RTF=0.093  id=1675
              pred: বাক্যের মাঝামাঝি পর্যন্ত আপনার জন্য সমস্ত নাম সবসময় বড় আক্ষরে শুরু হয়
  [17/25] BLEU= 10.2 ChrF= 62.2 RTF=0.089  id=1676
              pred: সমস্ত দক্ষিণ আফ্রিকার জাতীয় উদ্যানগুলির মতোই পার্কটির জন্য প্রতিদিন সংরক্ষণ এবং
  [18/25] BLEU= 23.6 ChrF= 46.3 RTF=0.158  id=1677
              pred: আজ একমাত্র কীট যে তাদের ডানা পিছনে রাখতে পারে না তা হল ড্রাগনফ্লাইস এবং মে ফ্লাই
  [19/25] BLEU=  2.3 ChrF= 46.5 RTF=0.100  id=1678
              pred: অলিভার স্যাক্স তার রাষ্ট্রপতি বক্তৃতায় আলোকিত করেছিলেন যে মস্তিষ্কের ক্ষতির কার
  [20/25] BLEU=  6.6 ChrF= 44.8 RTF=0.175  id=1679
              pred: এরাস্মিথ তাদের সফরে তাদের বাকি সংগীত বাতিল করেছে
  [21/25] BLEU=  3.5 ChrF= 31.7 RTF=0.119  id=1680
              pred: একটি ভাল চক্রান্ত অ্যাথলেট হিসাবে বাঘ ভাল না হলেও ভাল চড়ে যায় সাঁতার দিয়ে অনে
  [22/25] BLEU=  4.2 ChrF= 46.3 RTF=0.070  id=1681
              pred: এটি কেবলমাত্র পরীক্ষা নয় এবং একটি পরীক্ষা এমন একটি পরীক্ষা যা সম্ভাব্য অনুমানগু
  [23/25] BLEU=  1.6 ChrF= 26.4 RTF=0.085  id=1682
              pred: যদিও কেউ এটা লিখেনি তা নিশ্চিত না জানা যায় যে তার জীবনের শুরুতে একটি বিশাল পার্
  [24/25] BLEU= 13.6 ChrF= 48.4 RTF=0.110  id=1683
              pred: এখানে অনেক পুরুষ এবং মহিলা বেঁচে আছেন যারা তাদের সময় বেঁচে আছেন এবং আরও অনেক যা
  [25/25] BLEU= 12.1 ChrF= 47.4 RTF=0.088  id=1684
              pred: অ্যাপিয়া সামোয়া'র রাজধানী শহরটি উপোভু দ্বীপের উপর এবং এর জনসংখ্যা প্রায় ৪০ হা

  Summary: BLEU=10.20  ChrF=45.14  RTF=0.1129  Params=1039.1M

[ckpt] Saved phase7_benchmark_step000000.pt (0.0 MB)
[ckpt] Loaded phase4_benchmark_step000000.pt
[ckpt] Loaded phase6_benchmark_step000000.pt

=======================================================
  Phase 4 ChrF : 40.11
  Phase 6 ChrF : 40.11  (drop: 0.00)
  Phase 7 ChrF : 45.14  (recovery: +5.03)
=======================================================
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P7_DoRA (8 total)
```

### Cell 39 (markdown, score=3)
```markdown
---
# Phase 8: Final Results + Paper Table
```

### Cell 40 (code, score=63)
```python
sc = load_latest_checkpoint('all_summaries')
if sc and 'summaries' in sc: ALL_SUMMARIES = sc['summaries']

print('\n' + '='*80)
print('  FINAL: SeamlessM4T v2 Large  Structured Compression')
print('  Task: English to Bengali Speech Translation (FLEURS test)')
print('='*80)
hdr = f'{"Phase":<25} {"Params(M)":>10} {"Delta":>8} {"BLEU":>7} {"ChrF":>7} {"RTF":>7}'
print(hdr); print('-'*len(hdr))
bp = ALL_SUMMARIES[0]['params_M'] if ALL_SUMMARIES else 2300
for s in ALL_SUMMARIES:
    d = (1 - s['params_M']/bp)*100 if bp else 0
    ds = f'-{d:.1f}%' if d > 0 else 'base'
    print(f'  {s["label"]:<23} {s["params_M"]:>8.1f}  {ds:>7}  {s["avg_bleu"]:>6.2f}  {s["avg_chrf"]:>6.2f}  {s["avg_rtf"]:>6.4f}')
print('='*80)
if len(ALL_SUMMARIES) >= 2:
    f, b = ALL_SUMMARIES[-1], ALL_SUMMARIES[0]
    print(f'  Param reduction: {(1-f["params_M"]/b["params_M"])*100:.1f}%')
    if f['avg_rtf'] > 0:
        print(f'  Speed (RTF): {b["avg_rtf"]/f["avg_rtf"]:.2f}x faster')
```
OUTPUT:
```text
[ckpt] Loaded all_summaries_step000000.pt

================================================================================
  FINAL: SeamlessM4T v2 Large  Structured Compression
  Task: English to Bengali Speech Translation (FLEURS test)
================================================================================
Phase                      Params(M)    Delta    BLEU    ChrF     RTF
---------------------------------------------------------------------
  P0_Baseline               1805.5     base   11.63   50.52  0.2681
  P1_VocabTrim              1564.2   -13.4%   11.43   49.07  0.1734
  P3_DecPrune               1312.3   -27.3%    8.09   43.58  0.0994
  P4_EncPrune               1118.8   -38.0%    8.19   40.11  0.0937
  P5_FLAP(base)             1713.7    -5.1%    6.34   35.48  0.2341
  P5_FLAP(m4)               1057.2   -41.4%    0.95    9.20  0.3540
  P6_T2UIter                1039.1   -42.4%    8.19   40.11  0.0972
  P7_DoRA                   1039.1   -42.4%   10.20   45.14  0.1129
================================================================================
  Param reduction: 42.4%
  Speed (RTF): 2.37x faster
```

### Cell 41 (code, score=75)
```python
# Fix get_summaries to handle ALL_SUMMARIES as a list (not a dict)
def get_summaries():
    return sorted(ALL_SUMMARIES, key=lambda s: s['label'])

if len(ALL_SUMMARIES) >= 2:
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('SeamlessM4T Compression Pipeline Results', fontsize=16, fontweight='bold')
    labels = [s['label'] for s in ALL_SUMMARIES]
    x = range(len(labels))

    ax1 = fig.add_subplot(2, 3, 1)
    ps = [s['params_M'] for s in ALL_SUMMARIES]
    ax1.bar(x, ps, color='#9C27B0', alpha=0.85)
    ax1.set_ylabel('Params (M)'); ax1.set_title('Model Size', fontweight='bold')
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, rotation=40, ha='right', fontsize=7)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(list(x), [s['avg_bleu'] for s in ALL_SUMMARIES], 'o-', color='#2196F3', lw=2)
    ax2.set_ylabel('BLEU'); ax2.set_title('BLEU (higher=better)', fontweight='bold')
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, rotation=40, ha='right', fontsize=7)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(list(x), [s['avg_chrf'] for s in ALL_SUMMARIES], 's-', color='#4CAF50', lw=2)
    ax3.set_ylabel('ChrF'); ax3.set_title('ChrF (higher=better)', fontweight='bold')
    ax3.set_xticks(list(x)); ax3.set_xticklabels(labels, rotation=40, ha='right', fontsize=7)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.bar(list(x), [s['avg_rtf'] for s in ALL_SUMMARIES], color='#FF9800', alpha=0.85)
    ax4.set_ylabel('RTF'); ax4.set_title('RTF (lower=faster)', fontweight='bold')
    ax4.set_xticks(list(x)); ax4.set_xticklabels(labels, rotation=40, ha='right', fontsize=7)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(ps, [s['avg_bleu'] for s in ALL_SUMMARIES], s=100, c='#2196F3', label='BLEU')
    ax5.scatter(ps, [s['avg_chrf'] for s in ALL_SUMMARIES], s=100, c='#4CAF50', marker='s', label='ChrF')
    ax5.set_xlabel('Params (M)'); ax5.set_ylabel('Score')
    ax5.set_title('Size vs Quality', fontweight='bold'); ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(2, 3, 6)
    bp = ALL_SUMMARIES[0]['params_M'] or 1
    bb = ALL_SUMMARIES[0]['avg_bleu'] or 1
    bc = ALL_SUMMARIES[0]['avg_chrf'] or 1
    comp = [(1-s['params_M']/bp)*100 for s in ALL_SUMMARIES]
    ax6.plot(comp, [s['avg_bleu']/bb*100 for s in ALL_SUMMARIES], 'o-', color='#2196F3', label='BLEU %')
    ax6.plot(comp, [s['avg_chrf']/bc*100 for s in ALL_SUMMARIES], 's-', color='#4CAF50', label='ChrF %')
    ax6.axhline(y=90, color='gray', ls='--', alpha=0.5)
    ax6.set_xlabel('Compression %'); ax6.set_ylabel('Quality Retention %')
    ax6.set_title('Compression vs Quality', fontweight='bold'); ax6.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, 'final_comprehensive.png')
    plt.show()

plot_phase_comparison()
plot_size_vs_quality()
```
OUTPUT:
```text
<Figure size 2160x1440 with 6 Axes>
[image/png output omitted]
<Figure size 1680x1200 with 4 Axes>
[image/png output omitted]
<Figure size 1200x840 with 1 Axes>
[image/png output omitted]
```

### Cell 42 (code, score=80)
```python
print('\nDone. All results saved to Drive.')
session_status()
```
OUTPUT:
```text
Done. All results saved to Drive.
============================================================
  Platform : kaggle   Time : 2026-04-19 12:07
  Checkpoint files in /kaggle/working/checkpoints: 20
    all_summaries_step000000.pt                             0.0 MB
    phase0_baseline_step000000.pt                           0.0 MB
    phase1_benchmark_step000000.pt                          0.0 MB
    phase1_vocab_step000000.pt                              0.1 MB
    phase3_benchmark_step000000.pt                          0.0 MB
    phase3_dec_pruning_step000000.pt                        0.0 MB
    phase4_benchmark_step000000.pt                          0.0 MB
    phase4_enc_pruning_step000000.pt                        0.0 MB
    phase5_benchmark(base)_step000000.pt                    0.0 MB
    phase5_benchmark(m4)_step000000.pt                      0.0 MB
    phase5_flap(base)_step000000.pt                         0.0 MB
    phase5_flap(m4)_step000000.pt                           0.0 MB
    phase6_benchmark_step000000.pt                          0.0 MB
    phase6_t2u_pruning_step000000.pt                        0.0 MB
    phase6_t2u_t2u_model_model_decoder_layers_pruning_step000000.pt      0.0 MB
    phase6_t2u_t2u_model_model_encoder_layers_pruning_step000000.pt      0.0 MB
    phase7_benchmark_step000000.pt                          0.0 MB
    phase7_ft_step002000.pt                                64.8 MB
    phase7_ft_step002250.pt                                64.8 MB
    phase7_ft_step002500.pt                                64.8 MB
  GPU: Tesla T4
  VRAM: 15.6 GB
============================================================
```

### Cell 43 (code, score=8)
```python
if ON_KAGGLE:
    print('[audio] Uploading audio to rclone remote...')
    r = subprocess.run(
        f'rclone copy "{AUDIO_DIR}/" "{GDRIVE_ROOT}/audio/"',
        shell=True,
        capture_output=True,
        text=True
    )

    if r.returncode != 0:
        print(f'[audio] WARNING: {r.stderr[:300]}')
    else:
        print('[audio] Uploading complete.')

else:
    print(f'[audio] Colab: files already in Google Drive at {AUDIO_DIR} (no sync needed)')

if ON_KAGGLE:
    print('[figures] Uploading figures to rclone remote...')
    r = subprocess.run(
        f'rclone copy "{FIG_DIR}/" "{GDRIVE_ROOT}/figures/"',
        shell=True,
        capture_output=True,
        text=True
    )

    if r.returncode != 0:
        print(f'[figure] WARNING: {r.stderr[:300]}')
    else:
        print('[figure] Uploading complete.')

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

### Cell 44 (code, score=24)
```python
print("=============== KD PHASE ===============")
```
OUTPUT:
```text
=============== KD PHASE ===============
```