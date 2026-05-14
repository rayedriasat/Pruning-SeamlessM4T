# mission500m\bidirectional-tracking-base.ipynb

Extracted notebook map containing markdown headings plus code/output cells likely to matter for reports, reproduction, or agent steering.

## Markdown headings
cell 1: # SeamlessM4T v2 Large: Structured Compression 2.3B to ~1B ## Compression Pipeline
cell 2: ## Setup Cells 1-8
cell 20: ## Core Library: Model, Benchmark, Plotting
cell 26: # Phase 0: Baseline Benchmark

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

### Cell 2 (markdown, score=0)
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
rclone v1.73.5
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 73.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.1/84.1 kB 6.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.8/100.8 kB 8.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 85.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.6/121.6 kB 11.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 788.2/788.2 kB 38.6 MB/s eta 0:00:00
All packages installed.
```

### Cell 13 (code, score=100)
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
27 file(s) found:
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
  phase7_benchmark_step000000.pt                               0.0 MB
  phase7_ft_step000250.pt                                     64.7 MB
  phase7_ft_step000500.pt                                     64.7 MB
  phase7_ft_step000750.pt                                     64.7 MB
  phase7_ft_step001000.pt                                     64.7 MB
  phase7_ft_step001250.pt                                     64.8 MB
  phase7_ft_step001500.pt                                     64.8 MB
  phase7_ft_step001750.pt                                     64.8 MB
  phase7_ft_step002000.pt                                     64.8 MB
  phase7_ft_step002250.pt                                     64.8 MB
  phase7_ft_step002500.pt                                     64.8 MB
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

### Cell 16 (code, score=89)
```python
sync_checkpoints_from_drive()
```
OUTPUT:
```text
[ckpt] Syncing checkpoints from rclone remote...
[ckpt] 27 checkpoint(s) available
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
  phase7_benchmark_step000000.pt                              0.0 MB
  phase7_ft_step000250.pt                                    64.7 MB
  phase7_ft_step000500.pt                                    64.7 MB
  phase7_ft_step000750.pt                                    64.7 MB
  phase7_ft_step001000.pt                                    64.7 MB
  phase7_ft_step001250.pt                                    64.8 MB
  phase7_ft_step001500.pt                                    64.8 MB
  phase7_ft_step001750.pt                                    64.8 MB
  phase7_ft_step002000.pt                                    64.8 MB
  phase7_ft_step002250.pt                                    64.8 MB
  phase7_ft_step002500.pt                                    64.8 MB
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

### Cell 19 (code, score=95)
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
  Platform : kaggle   Time : 2026-04-19 18:26
  Checkpoint files in /kaggle/working/checkpoints: 27
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
    phase7_ft_step000250.pt                                64.7 MB
    phase7_ft_step000500.pt                                64.7 MB
    phase7_ft_step000750.pt                                64.7 MB
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

### Cell 25 (code, score=93)
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
Loaded 8 existing summaries: ['P0_Baseline', 'P1_VocabTrim', 'P3_DecPrune', 'P4_EncPrune', 'P5_FLAP(base)', 'P5_FLAP(m4)', 'P6_T2UIter', 'P7_DoRA']
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

60
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

### Cell 30 (code, score=17)
```python
# model_p7_merged, processor = load_model_from_drive('phase7_dora_merged_v1')
# print_model_breakdown(model_p7_merged, 'DoRA finetuned after Phase 7)')
# model_p7_merged = _consolidate_to_single_gpu(model_p7_merged)
# # save_model_to_drive(model_p7_merged, processor, 'phase7_dora_merged_v1')
```

### Cell 31 (code, score=89)
```python
base_model, processor = load_base_model()
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
text_encoder.layers.{0...23}.self_attn.k_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.q_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn_layer_norm.bias   | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc2.bias                | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.k_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn_layer_norm.weight | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.out_proj.weight   | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.v_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc2.weight              | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn_layer_norm.weight       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc1.weight              | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.out_proj.bias     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc1.bias                | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn_layer_norm.bias         | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.q_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.v_proj.bias       | UNEXPECTED |  | 
text_encoder.layer_norm.bias                             | UNEXPECTED |  | 
text_encoder.layer_norm.weight                           | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

generation_config.json: 0.00B [00:00, ?B/s]
Model loaded.
  GPU mem: 1.79 GB alloc / 1.80 GB reserved
```

### Cell 32 (code, score=24)
```python
print("=============== KD PHASE ===============")
```
OUTPUT:
```text
=============== KD PHASE ===============
```

### Cell 33 (code, score=35)
```python
# ===============================================================================
# BIDIRECTIONAL TRANSLATION ANALYSIS
# Analyze layer importance for Bengali→English vs English→Bengali translation
# ===============================================================================

print("\n" + "="*80)
print("  BIDIRECTIONAL TRANSLATION & LAYER ANALYSIS (Base Model)")
print("="*80)

# Load the final model for analysis
model_analysis = base_model
model_analysis.eval()

# Prepare samples for both directions
print("Preparing bidirectional translation samples...")

# English to Bengali (already have this)
eng_to_ben_samples = eval_samples[:10]  # Use first 10 samples

# Bengali to English (reverse direction)
ben_to_eng_samples = []
for sample in eval_samples[:10]:
    # Use the Bengali reference as input, English text as reference
    ben_audio_data = tgt_by_id[sample['id']]['bn_audio']
    ben_to_eng_samples.append({
        'id': sample['id'],
        'wav': _load_wav(ben_audio_data),
        'ref': sample['en_text'],  # English as reference
        'bn_text': sample['ref']   # Bengali as input context
    })

print(f"Prepared {len(eng_to_ben_samples)} EN→BN samples")
print(f"Prepared {len(ben_to_eng_samples)} BN→EN samples")
```
OUTPUT:
```text
================================================================================
  BIDIRECTIONAL TRANSLATION & LAYER ANALYSIS (Base Model)
================================================================================
Preparing bidirectional translation samples...
Prepared 10 EN→BN samples
Prepared 10 BN→EN samples
```

### Cell 34 (code, score=73)
```python
# ===============================================================================
# LAYER ACTIVATION ANALYSIS
# Hook into transformer layers to capture activation magnitudes
# ===============================================================================

import torch.nn.functional as F
from collections import defaultdict
import numpy as np

class LayerActivationHook:
    def __init__(self):
        self.activations = defaultdict(list)
        self.hooks = []
    
    def register_hooks(self, model, component_name):
        """Register hooks on transformer layers"""
        component = getattr(model, component_name, None)
        if component is None:
            return
        
        # Handle different layer structures
        layers = None
        if hasattr(component, 'layers'):
            layers = component.layers
        elif hasattr(component, 'encoder') and hasattr(component.encoder, 'layers'):
            layers = component.encoder.layers
        elif hasattr(component, 'model'):
            if hasattr(component.model, 'encoder') and hasattr(component.model.encoder, 'layers'):
                layers = component.model.encoder.layers
            elif hasattr(component.model, 'decoder') and hasattr(component.model.decoder, 'layers'):
                layers = component.model.decoder.layers
        
        if layers is None:
            return
        
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(
                self.make_hook_fn(f"{component_name}_layer_{i}")
            )
            self.hooks.append(hook)
    
    def make_hook_fn(self, layer_name):
        def hook_fn(module, input, output):
            # Capture activation magnitude (L2 norm)
            if isinstance(output, tuple):
                activation = output[0]  # Usually the main output
            else:
                activation = output
            
            if isinstance(activation, torch.Tensor):
                # Compute mean L2 norm across batch and sequence dimensions
                magnitude = torch.norm(activation, p=2, dim=-1).mean().item()
                self.activations[layer_name].append(magnitude)
        
        return hook_fn
    
    def clear(self):
        self.activations.clear()
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_average_activations(self):
        """Get average activation magnitude per layer"""
        avg_activations = {}
        for layer_name, magnitudes in self.activations.items():
            if magnitudes:
                avg_activations[layer_name] = np.mean(magnitudes)
        return avg_activations

# Initialize activation tracker
activation_tracker = LayerActivationHook()

# Register hooks on key components
print("Registering activation hooks...")
activation_tracker.register_hooks(model_analysis, 'speech_encoder')
activation_tracker.register_hooks(model_analysis, 'text_decoder')

# Also register T2U model hooks if present
if hasattr(model_analysis, 't2u_model'):
    t2u = model_analysis.t2u_model
    if hasattr(t2u, 'model'):
        # Register encoder hooks
        if hasattr(t2u.model, 'encoder') and hasattr(t2u.model.encoder, 'layers'):
            for i, layer in enumerate(t2u.model.encoder.layers):
                hook = layer.register_forward_hook(
                    activation_tracker.make_hook_fn(f"t2u_encoder_layer_{i}")
                )
                activation_tracker.hooks.append(hook)
        
        # Register decoder hooks
        if hasattr(t2u.model, 'decoder') and hasattr(t2u.model.decoder, 'layers'):
            for i, layer in enumerate(t2u.model.decoder.layers):
                hook = layer.register_forward_hook(
                    activation_tracker.make_hook_fn(f"t2u_decoder_layer_{i}")
                )
                activation_tracker.hooks.append(hook)

print(f"Registered {len(activation_tracker.hooks)} activation hooks")
```
OUTPUT:
```text
Registering activation hooks...
Registered 60 activation hooks
```

### Cell 35 (code, score=83)
```python
# ===============================================================================
# RUN BIDIRECTIONAL TRANSLATION WITH ACTIVATION TRACKING
# ===============================================================================

def run_translation_with_tracking(samples, direction, target_lang):
    """Run translation and track layer activations"""
    print(f"\nAnalyzing {direction} translation...")
    activation_tracker.clear()
    
    results = []
    for i, sample in enumerate(samples):
        try:
            # Run translation
            pred_text = run_s2t_only(model_analysis, sample['wav'], tgt_lang=target_lang)
            bleu = compute_bleu(pred_text, sample['ref'])
            chrf = compute_chrf(pred_text, sample['ref'])
            
            results.append({
                'id': sample['id'],
                'pred': pred_text,
                'ref': sample['ref'],
                'bleu': bleu,
                'chrf': chrf
            })
            
            print(f"  [{i+1:>2}/{len(samples)}] BLEU={bleu:5.1f} ChrF={chrf:5.1f}")
            
        except Exception as e:
            print(f"  [{i+1:>2}/{len(samples)}] ERROR: {e}")
            results.append({
                'id': sample['id'],
                'pred': '',
                'ref': sample['ref'],
                'bleu': 0.0,
                'chrf': 0.0
            })
    
    # Get average activations for this direction
    avg_activations = activation_tracker.get_average_activations()
    
    return results, avg_activations

# Run English to Bengali translation
print("\n" + "-"*60)
print("ENGLISH → BENGALI TRANSLATION")
print("-"*60)

eng_to_ben_results, eng_to_ben_activations = run_translation_with_tracking(
    eng_to_ben_samples, "EN→BN", "ben"
)

# Run Bengali to English translation  
print("\n" + "-"*60)
print("BENGALI → ENGLISH TRANSLATION")
print("-"*60)

ben_to_eng_results, ben_to_eng_activations = run_translation_with_tracking(
    ben_to_eng_samples, "BN→EN", "eng"
)

# Clean up hooks
activation_tracker.remove_hooks()
```
OUTPUT:
```text
------------------------------------------------------------
ENGLISH → BENGALI TRANSLATION
------------------------------------------------------------

Analyzing EN→BN translation...
  [ 1/10] BLEU= 10.7 ChrF= 49.2
  [ 2/10] BLEU= 10.4 ChrF= 45.8
  [ 3/10] BLEU= 11.6 ChrF= 56.7
  [ 4/10] BLEU=  5.4 ChrF= 42.5
  [ 5/10] BLEU= 11.2 ChrF= 45.8
  [ 6/10] BLEU=  9.7 ChrF= 48.3
  [ 7/10] BLEU= 10.4 ChrF= 53.6
  [ 8/10] BLEU= 19.1 ChrF= 60.3
  [ 9/10] BLEU=  7.0 ChrF= 46.5
  [10/10] BLEU=  9.7 ChrF= 43.2

------------------------------------------------------------
BENGALI → ENGLISH TRANSLATION
------------------------------------------------------------

Analyzing BN→EN translation...
  [ 1/10] BLEU=  7.6 ChrF= 45.0
  [ 2/10] BLEU= 12.4 ChrF= 55.2
  [ 3/10] BLEU= 45.8 ChrF= 60.1
  [ 4/10] BLEU=  3.8 ChrF= 36.5
  [ 5/10] BLEU= 22.8 ChrF= 59.9
  [ 6/10] BLEU= 12.0 ChrF= 51.4
  [ 7/10] BLEU=  9.0 ChrF= 46.8
  [ 8/10] BLEU=  9.2 ChrF= 45.2
  [ 9/10] BLEU=  3.0 ChrF= 33.3
  [10/10] BLEU= 39.7 ChrF= 68.9
```

### Cell 36 (code, score=179)
```python
# ===============================================================================
# ANALYZE LAYER IMPORTANCE BY DIRECTION
# ===============================================================================

print("\n" + "="*80)
print("  LAYER IMPORTANCE ANALYSIS")
print("="*80)

def analyze_layer_importance(activations_dict, direction_name):
    """Analyze and rank layer importance"""
    print(f"\n{direction_name} Layer Importance:")
    print("-" * 50)
    
    # Group by component
    components = defaultdict(dict)
    for layer_name, activation in activations_dict.items():
        if '_layer_' in layer_name:
            component = layer_name.split('_layer_')[0]
            layer_idx = int(layer_name.split('_layer_')[1])
            components[component][layer_idx] = activation
    
    # Rank layers within each component
    component_rankings = {}
    for component, layer_activations in components.items():
        if not layer_activations:
            continue
            
        # Sort by activation magnitude (descending)
        sorted_layers = sorted(layer_activations.items(), 
                             key=lambda x: x[1], reverse=True)
        
        component_rankings[component] = sorted_layers
        
        print(f"\n{component.upper()} Layers (ranked by importance):")
        for rank, (layer_idx, activation) in enumerate(sorted_layers, 1):
            print(f"  {rank:>2}. Layer {layer_idx:>2}: {activation:>8.4f}")
    
    return component_rankings

# Analyze both directions
eng_to_ben_rankings = analyze_layer_importance(eng_to_ben_activations, "ENGLISH → BENGALI")
ben_to_eng_rankings = analyze_layer_importance(ben_to_eng_activations, "BENGALI → ENGLISH")
```
OUTPUT:
```text
================================================================================
  LAYER IMPORTANCE ANALYSIS
================================================================================

ENGLISH → BENGALI Layer Importance:
--------------------------------------------------

SPEECH_ENCODER Layers (ranked by importance):
   1. Layer  1:  26.4578
   2. Layer  2:  25.5938
   3. Layer 10:  25.5734
   4. Layer  0:  25.2422
   5. Layer  9:  25.1391
   6. Layer  4:  24.6359
   7. Layer 14:  22.5531
   8. Layer 11:  22.5484
   9. Layer  5:  22.4609
  10. Layer 15:  22.4172
  11. Layer  8:  22.2766
  12. Layer 12:  21.8000
  13. Layer 16:  21.2875
  14. Layer 17:  21.2391
  15. Layer  3:  20.9547
  16. Layer 13:  20.3359
  17. Layer  6:  20.3281
  18. Layer 18:  19.7188
  19. Layer 19:  18.4500
  20. Layer 20:  15.8438
  21. Layer 21:  13.6000
  22. Layer 22:  12.3430
  23. Layer  7:   8.6766
  24. Layer 23:   5.2938

TEXT_DECODER Layers (ranked by importance):
   1. Layer 23: 7769.9116
   2. Layer 22: 7470.9945
   3. Layer 21: 6767.7348
   4. Layer 20: 6310.4862
   5. Layer 19: 6015.3536
   6. Layer 18: 5718.6133
   7. Layer 17: 5345.3591
   8. Layer 16: 4898.4862
   9. Layer 15: 4353.9282
  10. Layer 14: 3869.0387
  11. Layer 13: 3447.9088
  12. Layer 12: 3072.9586
  13. Layer 11: 2779.6961
  14. Layer 10: 2517.3508
  15. Layer  9: 2284.7072
  16. Layer  8: 2089.0718
  17. Layer  7: 1889.5635
  18. Layer  6: 1696.3094
  19. Layer  5: 1540.7044
  20. Layer  4: 1364.3812
  21. Layer  3: 1206.5939
  22. Layer  2: 1038.7970
  23. Layer  1: 832.1008
  24. Layer  0: 581.9834

T2U_ENCODER Layers (ranked by importance):
   1. Layer  5: 294.3375
   2. Layer  4: 170.5500
   3. Layer  3: 116.7250
   4. Layer  2:  74.9062
   5. Layer  1:  62.6781
   6. Layer  0:  56.4469

T2U_DECODER Layers (ranked by importance):
   1. Layer  2:  37.8406
   2. Layer  5:  32.8406
   3. Layer  1:  31.8891
   4. Layer  3:  31.8672
   5. Layer  4:  31.8453
   6. Layer  0:  31.4500

BENGALI → ENGLISH Layer Importance:
--------------------------------------------------

SPEECH_ENCODER Layers (ranked by importance):
   1. Layer  1:  26.6187
   2. Layer  2:  25.8031
   3. Layer  0:  25.1703
   4. Layer 10:  25.0422
   5. Layer  4:  24.6891
   6. Layer  9:  24.4719
   7. Layer 11:  22.2500
   8. Layer 15:  22.2156
   9. Layer  5:  22.1953
  10. Layer 14:  22.0594
  11. Layer  8:  21.8156
  12. Layer 12:  21.5266
  13. Layer 16:  21.3734
  14. Layer 17:  21.3469
  15. Layer  3:  21.2437
  16. Layer  6:  20.2484
  17. Layer 18:  19.8969
  18. Layer 13:  19.8484
  19. Layer 19:  18.6625
  20. Layer 20:  16.0695
  21. Layer 21:  13.9008
  22. Layer 22:  12.4961
  23. Layer  7:   8.6383
  24. Layer 23:   5.2605

TEXT_DECODER Layers (ranked by importance):
   1. Layer 23: 11597.8863
   2. Layer 22: 11079.2107
   3. Layer 21: 9388.8294
   4. Layer 20: 7275.6923
   5. Layer 19: 6067.1906
   6. Layer 18: 5409.0234
   7. Layer 17: 4907.3846
   8. Layer 16: 4410.7960
   9. Layer 15: 3973.7926
  10. Layer 14: 3612.8294
  11. Layer 13: 3280.1672
  12. Layer 12: 2971.3244
  13. Layer 11: 2699.4482
  14. Layer 10: 2469.9732
  15. Layer  9: 2248.0569
  16. Layer  8: 2062.6288
  17. Layer  7: 1898.5217
  18. Layer  6: 1737.3110
  19. Layer  5: 1592.0702
  20. Layer  4: 1437.5134
  21. Layer  3: 1284.3294
  22. Layer  2: 1096.5368
  23. Layer  1: 870.2592
  24. Layer  0: 635.2115

T2U_ENCODER Layers (ranked by importance):
   1. Layer  5: 266.0000
   2. Layer  4: 133.1687
   3. Layer  3:  79.6375
   4. Layer  2:  64.3406
   5. Layer  0:  60.5688
   6. Layer  1:  60.5250

T2U_DECODER Layers (ranked by importance):
   1. Layer  2:  37.6938
   2. Layer  5:  33.4250
   3. Layer  1:  31.8766
   4. Layer  3:  31.8422
   5. Layer  4:  31.8313
   6. Layer  0:  31.4375
```

### Cell 37 (code, score=78)
```python
# ===============================================================================
# COMPARATIVE ANALYSIS
# ===============================================================================

print("\n" + "="*80)
print("  COMPARATIVE LAYER ANALYSIS")
print("="*80)

def compare_layer_importance(rankings1, rankings2, name1, name2):
    """Compare layer importance between two translation directions"""
    
    for component in set(rankings1.keys()) | set(rankings2.keys()):
        if component not in rankings1 or component not in rankings2:
            continue
            
        print(f"\n{component.upper()} - Direction Comparison:")
        print(f"{'Layer':<8} {name1:<15} {name2:<15} {'Difference':<12}")
        print("-" * 55)
        
        # Get all layers present in both rankings
        layers1 = dict(rankings1[component])
        layers2 = dict(rankings2[component])
        all_layers = sorted(set(layers1.keys()) | set(layers2.keys()))
        
        layer_diffs = []
        for layer_idx in all_layers:
            act1 = layers1.get(layer_idx, 0.0)
            act2 = layers2.get(layer_idx, 0.0)
            diff = act1 - act2
            layer_diffs.append((layer_idx, act1, act2, diff))
            
            print(f"{layer_idx:<8} {act1:<15.4f} {act2:<15.4f} {diff:<12.4f}")
        
        # Find layers that are much more important for one direction
        layer_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
        
        print(f"\nMost Direction-Specific Layers in {component.upper()}:")
        for layer_idx, act1, act2, diff in layer_diffs[:5]:
            if abs(diff) > 0.001:  # Only show meaningful differences
                direction = name1 if diff > 0 else name2
                print(f"  Layer {layer_idx:>2}: More important for {direction} (diff: {abs(diff):.4f})")

compare_layer_importance(eng_to_ben_rankings, ben_to_eng_rankings, 
                        "EN→BN", "BN→EN")
```
OUTPUT:
```text
================================================================================
  COMPARATIVE LAYER ANALYSIS
================================================================================

T2U_DECODER - Direction Comparison:
Layer    EN→BN           BN→EN           Difference  
-------------------------------------------------------
0        31.4500         31.4375         0.0125      
1        31.8891         31.8766         0.0125      
2        37.8406         37.6938         0.1469      
3        31.8672         31.8422         0.0250      
4        31.8453         31.8313         0.0141      
5        32.8406         33.4250         -0.5844     

Most Direction-Specific Layers in T2U_DECODER:
  Layer  5: More important for BN→EN (diff: 0.5844)
  Layer  2: More important for EN→BN (diff: 0.1469)
  Layer  3: More important for EN→BN (diff: 0.0250)
  Layer  4: More important for EN→BN (diff: 0.0141)
  Layer  1: More important for EN→BN (diff: 0.0125)

SPEECH_ENCODER - Direction Comparison:
Layer    EN→BN           BN→EN           Difference  
-------------------------------------------------------
0        25.2422         25.1703         0.0719      
1        26.4578         26.6187         -0.1609     
2        25.5938         25.8031         -0.2094     
3        20.9547         21.2437         -0.2891     
4        24.6359         24.6891         -0.0531     
5        22.4609         22.1953         0.2656      
6        20.3281         20.2484         0.0797      
7        8.6766          8.6383          0.0383      
8        22.2766         21.8156         0.4609      
9        25.1391         24.4719         0.6672      
10       25.5734         25.0422         0.5312      
11       22.5484         22.2500         0.2984      
12       21.8000         21.5266         0.2734      
13       20.3359         19.8484         0.4875      
14       22.5531         22.0594         0.4938      
15       22.4172         22.2156         0.2016      
16       21.2875         21.3734         -0.0859     
17       21.2391         21.3469         -0.1078     
18       19.7188         19.8969         -0.1781     
19       18.4500         18.6625         -0.2125     
20       15.8438         16.0695         -0.2258     
21       13.6000         13.9008         -0.3008     
22       12.3430         12.4961         -0.1531     
23       5.2938          5.2605          0.0332      

Most Direction-Specific Layers in SPEECH_ENCODER:
  Layer  9: More important for EN→BN (diff: 0.6672)
  Layer 10: More important for EN→BN (diff: 0.5312)
  Layer 14: More important for EN→BN (diff: 0.4938)
  Layer 13: More important for EN→BN (diff: 0.4875)
  Layer  8: More important for EN→BN (diff: 0.4609)

T2U_ENCODER - Direction Comparison:
Layer    EN→BN           BN→EN           Difference  
-------------------------------------------------------
0        56.4469         60.5688         -4.1219     
1        62.6781         60.5250         2.1531      
2        74.9062         64.3406         10.5656     
3        116.7250        79.6375         37.0875     
4        170.5500        133.1687        37.3813     
5        294.3375        266.0000        28.3375     

Most Direction-Specific Layers in T2U_ENCODER:
  Layer  4: More important for EN→BN (diff: 37.3813)
  Layer  3: More important for EN→BN (diff: 37.0875)
  Layer  5: More important for EN→BN (diff: 28.3375)
  Layer  2: More important for EN→BN (diff: 10.5656)
  Layer  0: More important for BN→EN (diff: 4.1219)

TEXT_DECODER - Direction Comparison:
Layer    EN→BN           BN→EN           Difference  
-------------------------------------------------------
0        581.9834        635.2115        -53.2281    
1        832.1008        870.2592        -38.1584    
2        1038.7970       1096.5368       -57.7398    
3        1206.5939       1284.3294       -77.7355    
4        1364.3812       1437.5134       -73.1322    
5        1540.7044       1592.0702       -51.3658    
6        1696.3094       1737.3110       -41.0016    
7        1889.5635       1898.5217       -8.9582     
8        2089.0718       2062.6288       26.4431     
9        2284.7072       2248.0569       36.6503     
10       2517.3508       2469.9732       47.3776     
11       2779.6961       2699.4482       80.2480     
12       3072.9586       2971.3244       101.6341    
13       3447.9088       3280.1672       167.7416    
14       3869.0387       3612.8294       256.2092    
15       4353.9282       3973.7926       380.1355    
16       4898.4862       4410.7960       487.6902    
17       5345.3591       4907.3846       437.9745    
18       5718.6133       5409.0234       309.5898    
19       6015.3536       6067.1906       -51.8370    
20       6310.4862       7275.6923       -965.2061   
21       6767.7348       9388.8294       -2621.0946  
22       7470.9945       11079.2107      -3608.2162  
23       7769.9116       11597.8863      -3827.9747  

Most Direction-Specific Layers in TEXT_DECODER:
  Layer 23: More important for BN→EN (diff: 3827.9747)
  Layer 22: More important for BN→EN (diff: 3608.2162)
  Layer 21: More important for BN→EN (diff: 2621.0946)
  Layer 20: More important for BN→EN (diff: 965.2061)
  Layer 16: More important for EN→BN (diff: 487.6902)
```

### Cell 38 (code, score=47)
```python
# ===============================================================================
# SUMMARY STATISTICS
# ===============================================================================

print("\n" + "="*80)
print("  TRANSLATION QUALITY SUMMARY")
print("="*80)

def summarize_results(results, direction):
    valid_results = [r for r in results if r['bleu'] > 0 or r['chrf'] > 0]
    if not valid_results:
        return
    
    avg_bleu = np.mean([r['bleu'] for r in valid_results])
    avg_chrf = np.mean([r['chrf'] for r in valid_results])
    
    print(f"\n{direction}:")
    print(f"  Samples: {len(valid_results)}/{len(results)}")
    print(f"  Avg BLEU: {avg_bleu:.2f}")
    print(f"  Avg ChrF: {avg_chrf:.2f}")

summarize_results(eng_to_ben_results, "English → Bengali")
summarize_results(ben_to_eng_results, "Bengali → English")
```
OUTPUT:
```text
================================================================================
  TRANSLATION QUALITY SUMMARY
================================================================================

English → Bengali:
  Samples: 10/10
  Avg BLEU: 10.52
  Avg ChrF: 49.19

Bengali → English:
  Samples: 10/10
  Avg BLEU: 16.54
  Avg ChrF: 50.22
```

### Cell 39 (code, score=52)
```python
# ===============================================================================
# VISUALIZATION
# ===============================================================================

print("\nGenerating layer importance visualizations...")

def plot_layer_activations(rankings_dict, title, save_name):
    """Plot layer activation patterns"""
    fig, axes = plt.subplots(len(rankings_dict), 1, 
                           figsize=(12, 4 * len(rankings_dict)))
    if len(rankings_dict) == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i, (component, layer_data) in enumerate(rankings_dict.items()):
        ax = axes[i]
        
        if not layer_data:
            continue
            
        layers, activations = zip(*layer_data)
        
        # Create color map based on importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
        
        bars = ax.bar(range(len(layers)), activations, color=colors)
        ax.set_title(f'{component.upper()} Layer Activations', fontweight='bold')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Activation Magnitude')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(l) for l in layers])
        
        # Add value labels on bars
        for bar, act in zip(bars, activations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{act:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()

# Plot for both directions
if eng_to_ben_rankings:
    plot_layer_activations(eng_to_ben_rankings, 
                          'Layer Importance: English → Bengali Translation',
                          'layer_importance_en_to_bn.png')

if ben_to_eng_rankings:
    plot_layer_activations(ben_to_eng_rankings,
                          'Layer Importance: Bengali → English Translation', 
                          'layer_importance_bn_to_en.png')
```
OUTPUT:
```text
Generating layer importance visualizations...

<Figure size 1440x1920 with 4 Axes>
[image/png output omitted]
<Figure size 1440x1920 with 4 Axes>
[image/png output omitted]
```

### Cell 40 (code, score=57)
```python
# Comparative heatmap
if eng_to_ben_rankings and ben_to_eng_rankings:
    print("Creating comparative heatmap...")
    
    # Find common components
    common_components = set(eng_to_ben_rankings.keys()) & set(ben_to_eng_rankings.keys())
    
    for component in common_components:
        layers1 = dict(eng_to_ben_rankings[component])
        layers2 = dict(ben_to_eng_rankings[component])
        all_layers = sorted(set(layers1.keys()) | set(layers2.keys()))
        
        if len(all_layers) < 2:
            continue
            
        # Create comparison matrix
        data = np.zeros((2, len(all_layers)))
        for i, layer_idx in enumerate(all_layers):
            data[0, i] = layers1.get(layer_idx, 0.0)  # EN→BN
            data[1, i] = layers2.get(layer_idx, 0.0)  # BN→EN
        
        fig, ax = plt.subplots(figsize=(max(8, len(all_layers)), 4))
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        
        ax.set_title(f'{component.upper()} Layer Activations: Direction Comparison', 
                    fontweight='bold')
        ax.set_xlabel('Layer Index')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['EN→BN', 'BN→EN'])
        ax.set_xticks(range(len(all_layers)))
        ax.set_xticklabels([str(l) for l in all_layers])
        
        # Add text annotations
        for i in range(2):
            for j in range(len(all_layers)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                             ha="center", va="center", color="white", fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Activation Magnitude')
        plt.tight_layout()
        save_figure(fig, f'layer_comparison_{component}.png')
        plt.show()

print("\n" + "="*80)
print("  BIDIRECTIONAL ANALYSIS COMPLETE")
print("="*80)
print("Layer importance rankings and visualizations have been generated.")
print("Check the figures directory for detailed plots.")

# Clean up memory
del model_analysis
torch.cuda.empty_cache()
gc.collect()
```
OUTPUT:
```text
Creating comparative heatmap...

<Figure size 960x480 with 2 Axes>
[image/png output omitted]
<Figure size 2880x480 with 2 Axes>
[image/png output omitted]
<Figure size 2880x480 with 2 Axes>
[image/png output omitted]
<Figure size 960x480 with 2 Axes>
[image/png output omitted]

================================================================================
  BIDIRECTIONAL ANALYSIS COMPLETE
================================================================================
Layer importance rankings and visualizations have been generated.
Check the figures directory for detailed plots.

43623
```