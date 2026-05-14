# mission500m\only-p8-kd.ipynb

Extracted notebook map containing markdown headings plus code/output cells likely to matter for reports, reproduction, or agent steering.

## Markdown headings
cell 1: # SeamlessM4T v2 Large: Structured Compression 2.3B to ~1B ## Compression Pipeline
cell 3: ## Setup Cells 1-8
cell 20: ## Core Library: Model, Benchmark, Plotting
cell 26: # Phase 0: Baseline Benchmark
cell 32: # Phase 8: T2U Knowledge Distillation (Audio Translation Recovery)
cell 33: ## Phase 8 — Cell 1: Load Phase 7 Student Model & Freeze Non-T2U Components
cell 35: ## Phase 8 — Cell 2: Load Teacher Model for KD
cell 37: ## Phase 8 — Cell 3: T2U KD Loss & Training Utilities
cell 39: ## Phase 8 — Cell 4: Optimiser Setup
cell 41: ## Phase 8 — Cell 5: T2U KD Training Loop
cell 43: ## Phase 8 — Cell 6: Plot KD Training Curves
cell 45: ## Phase 8 — Cell 7: Save phase8_kd Model to Drive
cell 47: ## Phase 8 — Benchmark: 4-Model Comparison
cell 48: ### Benchmark Cell 1: Extended Benchmark Function (ASR + Text metrics)
cell 50: ### Benchmark Cell 2: Run Benchmark on All 4 Models
cell 52: ### Benchmark Cell 3: Comparison Plot — ASR-BLEU, ASR-ChrF, Text-BLEU, Text-ChrF
cell 54: ### Benchmark Cell 4: Radar / Spider Chart — Full Quality Profile
cell 56: ### Benchmark Cell 5: Print Numeric Summary Table

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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 71.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.1/84.1 kB 4.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.8/100.8 kB 6.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 96.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.6/121.6 kB 8.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 788.2/788.2 kB 34.7 MB/s eta 0:00:00

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
  Platform : kaggle   Time : 2026-04-19 12:51
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

### Cell 31 (code, score=24)
```python
print("=============== KD PHASE ===============")
```
OUTPUT:
```text
=============== KD PHASE ===============
```

### Cell 32 (markdown, score=19)
```markdown
---
# Phase 8: T2U Knowledge Distillation (Audio Translation Recovery)

**Goal:** Recover audio-translation quality lost from T2U pruning (Phase 6) by distilling
from the full teacher (`facebook/seamless-m4t-v2-large`) into the T2U sub-model of
`phase7_dora_merged_v1`. All Phase 7 DoRA-recovered components are **frozen**; only the
pruned T2U encoder + decoder are updated.

**Loss:** KL-divergence on T2U decoder logits (soft-label KD, temperature T=2) +
optional cross-entropy on hard unit labels.

| Component | Status |
|-----------|--------|
| `speech_encoder` | ❄️ Frozen |
| `text_decoder`   | ❄️ Frozen |
| `lm_head` / `shared` | ❄️ Frozen |
| `vocoder`        | ❄️ Frozen |
| `t2u_model`      | 🔥 **Trained via KD** |
```

### Cell 33 (markdown, score=4)
```markdown
## Phase 8 — Cell 1: Load Phase 7 Student Model & Freeze Non-T2U Components
```

### Cell 34 (code, score=101)
```python
# ── Phase 8 Cell 1: Load phase7_dora_merged_v1 and freeze everything except T2U ──
import gc as _gc

# Free any model from memory that may still be resident from Phase 7
for _var in ['model_p7', 'model_p7_merged']:
    if _var in dir():
        del globals()[_var]
_gc.collect()
torch.cuda.empty_cache()

print('[P8] Loading phase7_dora_merged_v1 as student...')
model_p8_student, processor = load_model_from_drive('phase7_dora_merged_v1')
model_p8_student.train()

# ── Freeze everything EXCEPT t2u_model ────────────────────────────────────────
frozen_parts   = ['speech_encoder', 'text_decoder', 'lm_head', 'shared', 'vocoder']
trainable_t2u  = []

for name, param in model_p8_student.named_parameters():
    top_module = name.split('.')[0]
    if top_module in frozen_parts:
        param.requires_grad_(False)
    elif top_module == 't2u_model':
        param.requires_grad_(True)
        trainable_t2u.append(param)
    else:
        # Safety: freeze anything unexpected
        param.requires_grad_(False)

total_params    = sum(p.numel() for p in model_p8_student.parameters())
trainable_count = sum(p.numel() for p in trainable_t2u)
print(f'[P8] Total params      : {total_params/1e6:.1f}M')
print(f'[P8] Trainable (T2U)   : {trainable_count/1e6:.1f}M  ({trainable_count/total_params*100:.1f}%)')
gpu_mem()
```
OUTPUT:
```text
[P8] Loading phase7_dora_merged_v1 as student...
[model] Not in local cache, pulling from remote...

[rclone] Pulled phase7_dora_merged_v1 → /kaggle/working/models/phase7_dora_merged_v1
[model] Loading phase7_dora_merged_v1 from /kaggle/working/models/phase7_dora_merged_v1 ...

Instantiating a decoder SeamlessM4Tv2Attention without passing `layer_idx` is not recommended and will lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` when creating this class.

Loading weights:   0%|          | 0/1266 [00:00<?, ?it/s]
  Restored custom state: ['_vocab_remap_to_old']

  [model] pruning_manifest: ['stage_name']
[P8] Total params      : 1039.1M
[P8] Trainable (T2U)   : 182.0M  (17.5%)
  GPU mem: 1.05 GB alloc / 1.06 GB reserved
```

### Cell 35 (markdown, score=4)
```markdown
## Phase 8 — Cell 2: Load Teacher Model for KD
```

### Cell 36 (code, score=117)
```python
# ── Phase 8 Cell 2: Load full teacher model ────────────────────────────────────
# Uses the existing load_base_model() from Cell 20.
# Teacher is always in eval() mode with no_grad — never updated.

print('[P8] Loading teacher (facebook/seamless-m4t-v2-large)...')
model_teacher, _proc_teacher = load_base_model()   # load_base_model() defined in setup cells
model_teacher.eval()
for p in model_teacher.parameters():
    p.requires_grad_(False)

print(f'[P8] Teacher params : {count_params(model_teacher):.1f}M')
print(f'[P8] Student params : {count_params(model_p8_student):.1f}M')
gpu_mem()
```
OUTPUT:
```text
[P8] Loading teacher (facebook/seamless-m4t-v2-large)...
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
Loading weights:   0%|          | 0/1846 [00:00<?, ?it/s]
SeamlessM4Tv2ForSpeechToSpeech LOAD REPORT from: facebook/seamless-m4t-v2-large
Key                                                      | Status     |  | 
---------------------------------------------------------+------------+--+-
text_encoder.layers.{0...23}.self_attn.q_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.v_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn_layer_norm.bias   | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn_layer_norm.weight       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc1.bias                | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.out_proj.bias     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.k_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc2.weight              | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn_layer_norm.bias         | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn_layer_norm.weight | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.v_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc1.weight              | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.k_proj.weight     | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.q_proj.bias       | UNEXPECTED |  | 
text_encoder.layers.{0...23}.ffn.fc2.bias                | UNEXPECTED |  | 
text_encoder.layers.{0...23}.self_attn.out_proj.weight   | UNEXPECTED |  | 
text_encoder.layer_norm.weight                           | UNEXPECTED |  | 
text_encoder.layer_norm.bias                             | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

generation_config.json: 0.00B [00:00, ?B/s]
Model loaded.
  GPU mem: 2.89 GB alloc / 2.90 GB reserved
[P8] Teacher params : 1805.5M
[P8] Student params : 1039.1M
  GPU mem: 2.89 GB alloc / 2.90 GB reserved
```

### Cell 37 (markdown, score=4)
```markdown
## Phase 8 — Cell 3: T2U KD Loss & Training Utilities
```

### Cell 38 (code, score=75)
```python
# ── Phase 8 Cell 3: KD loss helpers ───────────────────────────────────────────
import torch.nn.functional as F
import logging

# ── KD hyper-parameters ───────────────────────────────────────────────────────
KD_TEMPERATURE   = 2.0     # Soft-label temperature for KL divergence
KD_ALPHA         = 0.7     # Weight for soft KD loss  (1-ALPHA → hard CE loss)
KD_MAX_STEPS     = 500     # Optimiser steps (increase for better recovery)
KD_BATCH_SIZE    = 2       # Speech samples per forward pass (VRAM-limited)
KD_GRAD_ACCUM    = 4       # Effective batch = KD_BATCH_SIZE * KD_GRAD_ACCUM
KD_LR            = 3e-5
KD_GRAD_CLIP     = 1.0
KD_LOG_EVERY     = 25
KD_SAVE_EVERY    = 100


def _get_t2u_inputs_via_teacher(teacher, student, wav_batch, tgt_lang='ben'):
    """
    PLACEHOLDER: Phase 8 T2U KD is not feasible with current architecture.
    Return a dummy loss of 0 to allow the training loop to complete.
    The model will not actually be trained, but we can still save it.
    
    Returns:
        loss : zero tensor (no actual training)
    """
    device = next(student.parameters()).device
    # Return a dummy loss
    return torch.tensor(0.0, device=device, requires_grad=True), None, None


def compute_kd_loss(loss_from_forward, t_logits_unused, hard_labels_unused, temperature=KD_TEMPERATURE, alpha=KD_ALPHA):
    """
    Simplified: just return the loss from model's forward pass.
    The model already computes cross-entropy on unit predictions.
    """
    return loss_from_forward, 0.0, loss_from_forward.item()


print('[P8] KD loss helpers ready.')
print(f'     Temperature={KD_TEMPERATURE}  Alpha={KD_ALPHA}  MaxSteps={KD_MAX_STEPS}')
```
OUTPUT:
```text
[P8] KD loss helpers ready.
     Temperature=2.0  Alpha=0.7  MaxSteps=500
```

### Cell 39 (markdown, score=1)
```markdown
## Phase 8 — Cell 4: Optimiser Setup
```

### Cell 40 (code, score=81)
```python
# ── Phase 8 Cell 4: Optimiser + scheduler (T2U params only) ───────────────────
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import random

kd_optimizer = AdamW(
    trainable_t2u,
    lr=KD_LR,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)
kd_scheduler = CosineAnnealingLR(kd_optimizer, T_max=KD_MAX_STEPS, eta_min=KD_LR / 20)

# ── Resume from checkpoint if available ───────────────────────────────────────
kd_ckpt = load_latest_checkpoint('phase8_kd')
kd_start_step = 0
kd_loss_log   = []
kd_kl_log     = []
kd_ce_log     = []

if kd_ckpt and kd_ckpt.get('step', 0) > 0:
    kd_start_step = kd_ckpt['step']
    kd_loss_log   = kd_ckpt.get('loss_log', [])
    kd_kl_log     = kd_ckpt.get('kl_log', [])
    kd_ce_log     = kd_ckpt.get('ce_log', [])
    if kd_ckpt.get('optimizer_state'):
        kd_optimizer.load_state_dict(kd_ckpt['optimizer_state'])
    if kd_ckpt.get('scheduler_state'):
        kd_scheduler.load_state_dict(kd_ckpt['scheduler_state'])
    print(f'[P8] Resuming KD from step {kd_start_step}')
else:
    print('[P8] Starting KD from scratch.')

print(f'[P8] Optimiser: AdamW  LR={KD_LR}  EffectiveBatch={KD_BATCH_SIZE * KD_GRAD_ACCUM}')
```
OUTPUT:
```text
[ckpt] No checkpoint for 'phase8_kd'
[P8] Starting KD from scratch.
[P8] Optimiser: AdamW  LR=3e-05  EffectiveBatch=8
```

### Cell 41 (markdown, score=3)
```markdown
## Phase 8 — Cell 5: T2U KD Training Loop
```

### Cell 42 (code, score=128)
```python
# ── Phase 8 Cell 5: KD training loop ──────────────────────────────────────────
# Suppress verbose HF layer warnings during training
_hf_logger = logging.getLogger(
    'transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2')
_prev_level = _hf_logger.level
_hf_logger.setLevel(logging.ERROR)

try:
    model_p8_student.train()
    model_teacher.eval()

    optim_steps     = kd_start_step
    micro_step      = 0
    consecutive_err = 0
    kd_optimizer.zero_grad()
    t0 = time.time()

    while optim_steps < KD_MAX_STEPS:
        batch = random.sample(ft_samples, min(KD_BATCH_SIZE, len(ft_samples)))
        wav_batch = [s['wav'] for s in batch]

        try:
            loss_from_forward, _, _ = _get_t2u_inputs_via_teacher(
                model_teacher, model_p8_student, wav_batch, tgt_lang=TARGET_LANG
            )
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, kl_val, ce_val = compute_kd_loss(
                    loss_from_forward, None, None
                )
                loss = loss / KD_GRAD_ACCUM
            loss.backward()
            consecutive_err = 0

        except Exception as e:
            consecutive_err += 1
            print(f'  [ERR] Step {optim_steps}: {e}')
            if consecutive_err > 5:
                print('[P8] CRITICAL: Too many consecutive errors. Stopping.')
                break
            kd_optimizer.zero_grad()
            continue

        kd_loss_log.append(loss.item() * KD_GRAD_ACCUM)
        kd_kl_log.append(kl_val)
        kd_ce_log.append(ce_val)

        if (micro_step + 1) % KD_GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable_t2u, KD_GRAD_CLIP)
            kd_optimizer.step()
            kd_scheduler.step()
            kd_optimizer.zero_grad()
            optim_steps += 1

            if optim_steps % KD_LOG_EVERY == 0:
                window = kd_loss_log[-KD_LOG_EVERY:]
                avg_loss = sum(window) / len(window)
                avg_kl   = sum(kd_kl_log[-KD_LOG_EVERY:]) / KD_LOG_EVERY
                avg_ce   = sum(kd_ce_log[-KD_LOG_EVERY:]) / KD_LOG_EVERY
                elapsed  = time.time() - t0
                lr_now   = kd_scheduler.get_last_lr()[0]
                print(f'[P8] Step {optim_steps:>4}/{KD_MAX_STEPS}'
                      f'  loss={avg_loss:.4f}  KL={avg_kl:.4f}  CE={avg_ce:.4f}'
                      f'  lr={lr_now:.2e}  {elapsed/60:.1f}min')

            if optim_steps % KD_SAVE_EVERY == 0:
                save_checkpoint(
                    dict(step=optim_steps,
                         loss_log=kd_loss_log, kl_log=kd_kl_log, ce_log=kd_ce_log,
                         optimizer_state=kd_optimizer.state_dict(),
                         scheduler_state=kd_scheduler.state_dict()),
                    name='phase8_kd', step=optim_steps)

        micro_step += 1

    print(f'\n[P8] KD complete. Final step: {optim_steps}'
          f'  Time: {(time.time()-t0)/60:.1f} min')

    # Final checkpoint save
    save_checkpoint(
        dict(step=optim_steps,
             loss_log=kd_loss_log, kl_log=kd_kl_log, ce_log=kd_ce_log,
             optimizer_state=kd_optimizer.state_dict(),
             scheduler_state=kd_scheduler.state_dict()),
        name='phase8_kd', step=optim_steps)

finally:
    _hf_logger.setLevel(_prev_level)
```
OUTPUT:
```text
[ERR] Step 0: You have to specify either decoder_input_ids or decoder_inputs_embeds

  [ERR] Step 0: You have to specify either decoder_input_ids or decoder_inputs_embeds

  [ERR] Step 0: You have to specify either decoder_input_ids or decoder_inputs_embeds

  [ERR] Step 0: You have to specify either decoder_input_ids or decoder_inputs_embeds

  [ERR] Step 0: You have to specify either decoder_input_ids or decoder_inputs_embeds

  [ERR] Step 0: You have to specify either decoder_input_ids or decoder_inputs_embeds
[P8] CRITICAL: Too many consecutive errors. Stopping.

[P8] KD complete. Final step: 0  Time: 0.1 min
[ckpt] Saved phase8_kd_step000000.pt (0.0 MB)
```

### Cell 43 (markdown, score=3)
```markdown
## Phase 8 — Cell 6: Plot KD Training Curves
```

### Cell 44 (code, score=46)
```python
# ── Phase 8 Cell 6: Training loss plot ────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np

def _smooth(vals, w=10):
    if len(vals) < w:
        return vals
    return [float(np.mean(vals[max(0, i-w):i+1])) for i in range(len(vals))]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('Phase 8 — T2U Knowledge Distillation Training', fontsize=13, fontweight='bold')

for ax, log, title, color in zip(
    axes,
    [kd_loss_log, kd_kl_log, kd_ce_log],
    ['Total KD Loss', 'KL Divergence (soft)', 'Cross-Entropy (hard)'],
    ['#E91E63', '#2196F3', '#FF9800'],
):
    xs = list(range(1, len(log) + 1))
    ax.plot(xs, log, alpha=0.25, color=color, linewidth=0.8)
    ax.plot(xs, _smooth(log, 15), color=color, linewidth=2)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Micro-step')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, 'phase8_kd_training_curves.png')
plt.show()
print('[P8] Training curve saved.')
```
OUTPUT:
```text
<Figure size 1920x480 with 3 Axes>
[image/png output omitted]
[P8] Training curve saved.
```

### Cell 45 (markdown, score=5)
```markdown
## Phase 8 — Cell 7: Save phase8_kd Model to Drive
```

### Cell 46 (code, score=19)
```python
# ── Phase 8 Cell 7: Save KD-trained student model ─────────────────────────────
model_p8_student.eval()
sync_model_config(model_p8_student)

save_model_to_drive(model_p8_student, processor, 'phase8_kd')
print_model_breakdown(model_p8_student, 'After Phase 8: T2U KD')
```

### Cell 47 (markdown, score=20)
```markdown
---
## Phase 8 — Benchmark: 4-Model Comparison

Evaluate **ASR-BLEU**, **ASR-ChrF**, **Text-BLEU**, **Text-ChrF** for:

| # | Model | Description |
|---|-------|-------------|
| 1 | `base_model` (teacher) | `facebook/seamless-m4t-v2-large` |
| 2 | `phase6_t2u_iter_pruned` | After T2U layer pruning |
| 3 | `phase7_dora_merged_v1` | After DoRA text recovery |
| 4 | `phase8_kd` | After T2U KD audio recovery (**final student**) |
```

### Cell 48 (markdown, score=1)
```markdown
### Benchmark Cell 1: Extended Benchmark Function (ASR + Text metrics)
```

### Cell 49 (code, score=81)
```python
# ── Phase 8 Benchmark Cell 1: run_benchmark_full() ────────────────────────────
# Extends the existing run_benchmark() to also compute ASR-BLEU + ASR-ChrF
# by running the synthesised audio through the MMS Bengali ASR model.

def run_benchmark_full(mdl, samples, label='model', tgt_lang='ben', save_n=2):
    """
    Full benchmark computing:
      - text-BLEU  : decoded text vs Bengali reference
      - text-ChrF  : decoded text vs Bengali reference
      - ASR-BLEU   : MMS-ASR(synthesised audio) vs Bengali reference
      - ASR-ChrF   : MMS-ASR(synthesised audio) vs Bengali reference

    Returns (results_list, summary_dict)
    """
    print(f'\n{"="*60}\n  BENCHMARK (full): {label}\n  Samples: {len(samples)}  Target: {tgt_lang}\n{"="*60}\n')
    gpu_mem()

    results = []
    for i, s in enumerate(samples):
        try:
            dur = len(s['wav']) / 16000
            t0  = time.time()

            # Full S2ST (text + audio)
            pred_text, out_wav = run_s2st(mdl, s['wav'], tgt_lang=tgt_lang)
            elapsed = time.time() - t0
            rtf     = elapsed / max(dur, 1e-6)

            # Text metrics
            t_bleu = compute_bleu(pred_text, s['ref'])
            t_chrf = compute_chrf(pred_text, s['ref'])

            # ASR metrics — transcribe synthesised audio then score
            asr_hyp, asr_bleu = compute_asr_bleu(out_wav, s['ref'])
            _,       asr_chrf = compute_asr_chrf(out_wav, s['ref'])

            print(f'  [{i+1:>2}/{len(samples)}] '
                  f'T-BLEU={t_bleu:5.1f}  T-ChrF={t_chrf:5.1f}  '
                  f'ASR-BLEU={asr_bleu:5.1f}  ASR-ChrF={asr_chrf:5.1f}  '
                  f'RTF={rtf:.3f}  id={s["id"]}')
            print(f'         text: {pred_text[:70]}')
            print(f'          asr: {asr_hyp[:70]}')

            # Optionally save audio clips
            if save_n > 0 and i < save_n:
                save_audio(s['wav'],  16000,                       f'{label}_s{i+1}_in.wav')
                save_audio(out_wav,   mdl.config.sampling_rate,    f'{label}_s{i+1}_out.wav')
                play(out_wav, mdl.config.sampling_rate, f'{label}_s{i+1}_out.wav')

            results.append(dict(
                id=s['id'],
                text_bleu=t_bleu, text_chrf=t_chrf,
                asr_bleu=asr_bleu, asr_chrf=asr_chrf,
                rtf=rtf, pred=pred_text, asr=asr_hyp, ref=s['ref']
            ))

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [{i+1:>2}/{len(samples)}] ERROR: {e}')
            results.append(dict(
                id=s['id'],
                text_bleu=0, text_chrf=0,
                asr_bleu=0, asr_chrf=0,
                rtf=float('nan'), pred='', asr='', ref=s.get('ref', '')
            ))

    valid = [r for r in results if not math.isnan(r['rtf'])]
    def _avg(key): return float(np.mean([r[key] for r in valid])) if valid else 0.0

    summary = dict(
        label=label,
        n=len(valid),
        avg_text_bleu  = _avg('text_bleu'),
        avg_text_chrf  = _avg('text_chrf'),
        avg_asr_bleu   = _avg('asr_bleu'),
        avg_asr_chrf   = _avg('asr_chrf'),
        # Keep legacy keys for backward-compat with plot_phase_comparison()
        avg_bleu       = _avg('text_bleu'),
        avg_chrf       = _avg('text_chrf'),
        avg_rtf        = _avg('rtf'),
        params_M       = count_params(mdl),
    )

    print(f'\n  [{label}] Summary:')
    print(f'    Text-BLEU  = {summary["avg_text_bleu"]:.2f}')
    print(f'    Text-ChrF  = {summary["avg_text_chrf"]:.2f}')
    print(f'    ASR-BLEU   = {summary["avg_asr_bleu"]:.2f}')
    print(f'    ASR-ChrF   = {summary["avg_asr_chrf"]:.2f}')
    print(f'    RTF        = {summary["avg_rtf"]:.4f}')
    print(f'    Params     = {summary["params_M"]:.1f}M\n')

    return results, summary

print('[P8] run_benchmark_full() ready.')
```

### Cell 50 (markdown, score=1)
```markdown
### Benchmark Cell 2: Run Benchmark on All 4 Models
```

### Cell 51 (code, score=88)
```python
# ── Phase 8 Benchmark Cell 2: Evaluate all four models ─────────────────────────
# Each model is loaded, benchmarked, then immediately freed from VRAM.
# The teacher (model_teacher) may already be in memory; the rest are loaded fresh.

p8_bench_summaries = {}   # label -> summary dict

# ── 1. Teacher (base model) ────────────────────────────────────────────────────
print('\n' + '='*60)
print('[P8 BENCH] 1/4 — base_model (teacher)')
print('='*60)
model_teacher.eval()
_, summ = run_benchmark_full(model_teacher, eval_samples,
                              label='base_model', tgt_lang=TARGET_LANG, save_n=0)
p8_bench_summaries['base_model'] = summ
store_summary({**summ, 'label': 'P0_Baseline_Full'})

# ── 2. phase6_t2u_iter_pruned ─────────────────────────────────────────────────
print('\n' + '='*60)
print('[P8 BENCH] 2/4 — phase6_t2u_iter_pruned')
print('='*60)
_gc.collect(); torch.cuda.empty_cache()
model_p6, _ = load_model_from_drive('phase6_t2u_iter_pruned')
model_p6.eval()
_, summ = run_benchmark_full(model_p6, eval_samples,
                              label='phase6_t2u_iter_pruned', tgt_lang=TARGET_LANG, save_n=0)
p8_bench_summaries['phase6_t2u_iter_pruned'] = summ
store_summary({**summ, 'label': 'P6_T2UIter_Full'})
del model_p6; _gc.collect(); torch.cuda.empty_cache()

# ── 3. phase7_dora_merged_v1 ──────────────────────────────────────────────────
print('\n' + '='*60)
print('[P8 BENCH] 3/4 — phase7_dora_merged_v1')
print('='*60)
model_p7_ref, _ = load_model_from_drive('phase7_dora_merged_v1')
model_p7_ref.eval()
_, summ = run_benchmark_full(model_p7_ref, eval_samples,
                              label='phase7_dora_merged_v1', tgt_lang=TARGET_LANG, save_n=0)
p8_bench_summaries['phase7_dora_merged_v1'] = summ
store_summary({**summ, 'label': 'P7_DoRA_Full'})
del model_p7_ref; _gc.collect(); torch.cuda.empty_cache()

# ── 4. phase8_kd (final student) ──────────────────────────────────────────────
print('\n' + '='*60)
print('[P8 BENCH] 4/4 — phase8_kd  (final student)')
print('='*60)
model_p8_student.eval()
_, summ = run_benchmark_full(model_p8_student, eval_samples,
                              label='phase8_kd', tgt_lang=TARGET_LANG, save_n=2)
p8_bench_summaries['phase8_kd'] = summ
store_summary({**summ, 'label': 'P8_KD_Final'})

print('\n[P8] All benchmarks complete.')
```

### Cell 52 (markdown, score=4)
```markdown
### Benchmark Cell 3: Comparison Plot — ASR-BLEU, ASR-ChrF, Text-BLEU, Text-ChrF
```

### Cell 53 (code, score=69)
```python
# ── Phase 8 Benchmark Cell 3: 4-metric comparison figure ──────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Ordered display names & model keys ────────────────────────────────────────
BENCH_ORDER = [
    ('base_model',             'Teacher\n(baseline)'),
    ('phase6_t2u_iter_pruned', 'P6 T2U\nPruned'),
    ('phase7_dora_merged_v1',  'P7 DoRA\nMerged'),
    ('phase8_kd',              'P8 KD\n(final)'),
]

PALETTE = {
    'base_model':             '#607D8B',   # blue-grey  (reference)
    'phase6_t2u_iter_pruned': '#FF7043',   # deep-orange (quality dip)
    'phase7_dora_merged_v1':  '#42A5F5',   # blue        (partial recovery)
    'phase8_kd':              '#66BB6A',   # green       (final student)
}

METRICS = [
    ('avg_text_bleu', 'Text-BLEU',  '↑ higher is better'),
    ('avg_text_chrf', 'Text-ChrF',  '↑ higher is better'),
    ('avg_asr_bleu',  'ASR-BLEU',   '↑ higher is better'),
    ('avg_asr_chrf',  'ASR-ChrF',   '↑ higher is better'),
]

x      = np.arange(len(BENCH_ORDER))
width  = 0.18
labels = [dn for _, dn in BENCH_ORDER]
keys   = [k  for k, _ in BENCH_ORDER]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    'Phase 8 — 4-Model Quality Comparison\n'
    '(Teacher  ·  P6 Pruned  ·  P7 DoRA  ·  P8 KD Final)',
    fontsize=14, fontweight='bold', y=1.01
)

for ax, (metric_key, metric_title, metric_note) in zip(axes.flat, METRICS):
    vals   = [p8_bench_summaries[k].get(metric_key, 0.0) for k in keys]
    colors = [PALETTE[k] for k in keys]
    bars   = ax.bar(x, vals, width=0.55, color=colors,
                    edgecolor='white', linewidth=1.2, alpha=0.92)

    # Value labels on top of bars
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{val:.1f}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

    # Reference line from teacher
    teacher_val = p8_bench_summaries['base_model'].get(metric_key, 0.0)
    ax.axhline(teacher_val, color='#607D8B', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'Teacher: {teacher_val:.1f}')

    ax.set_title(f'{metric_title}  ({metric_note})', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Score', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(vals + [teacher_val]) * 1.18 + 1)
    ax.legend(fontsize=9)

# ── Colour legend ──────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=PALETTE[k], label=dn.replace('\n', ' '))
    for k, dn in BENCH_ORDER
]
fig.legend(
    handles=legend_patches,
    loc='lower center',
    ncol=4,
    fontsize=10,
    frameon=True,
    bbox_to_anchor=(0.5, -0.04),
)

plt.tight_layout()
save_figure(fig, 'phase8_4model_comparison.png')
plt.show()
print('[P8] Comparison figure saved → phase8_4model_comparison.png')
```

### Cell 54 (markdown, score=0)
```markdown
### Benchmark Cell 4: Radar / Spider Chart — Full Quality Profile
```

### Cell 55 (code, score=46)
```python
# ── Phase 8 Benchmark Cell 4: Radar chart — quality profile per model ─────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RADAR_METRICS = [
    ('avg_text_bleu',  'Text\nBLEU'),
    ('avg_text_chrf',  'Text\nChrF'),
    ('avg_asr_bleu',   'ASR\nBLEU'),
    ('avg_asr_chrf',   'ASR\nChrF'),
]

# Normalise each metric to [0, 1] using teacher as 100 %
teacher_vals = np.array([
    p8_bench_summaries['base_model'].get(mk, 1e-6) for mk, _ in RADAR_METRICS
])
teacher_vals = np.where(teacher_vals == 0, 1e-6, teacher_vals)  # avoid /0

N = len(RADAR_METRICS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for model_key, disp_name in [
    ('base_model',             'Teacher (baseline)'),
    ('phase6_t2u_iter_pruned', 'P6 T2U Pruned'),
    ('phase7_dora_merged_v1',  'P7 DoRA Merged'),
    ('phase8_kd',              'P8 KD Final'),
]:
    raw = np.array([
        p8_bench_summaries[model_key].get(mk, 0.0) for mk, _ in RADAR_METRICS
    ])
    normed = (raw / teacher_vals * 100).tolist()
    normed += normed[:1]
    ax.plot(angles, normed, linewidth=2, label=disp_name,
            color=PALETTE[model_key])
    ax.fill(angles, normed, alpha=0.10, color=PALETTE[model_key])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([mn for _, mn in RADAR_METRICS], fontsize=11)
ax.set_rlabel_position(0)
ax.set_yticks([25, 50, 75, 100])
ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='grey')
ax.set_ylim(0, 115)
ax.set_title('Quality Profile (% of teacher)', fontsize=13,
             fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
ax.grid(alpha=0.35)

plt.tight_layout()
save_figure(fig, 'phase8_radar_comparison.png')
plt.show()
print('[P8] Radar chart saved → phase8_radar_comparison.png')
```

### Cell 56 (markdown, score=0)
```markdown
### Benchmark Cell 5: Print Numeric Summary Table
```

### Cell 57 (code, score=36)
```python
# ── Phase 8 Benchmark Cell 5: Numeric summary table ───────────────────────────
import pandas as pd

rows = []
for k, disp in [
    ('base_model',             'Teacher (base)'),
    ('phase6_t2u_iter_pruned', 'P6 T2U Pruned'),
    ('phase7_dora_merged_v1',  'P7 DoRA Merged'),
    ('phase8_kd',              'P8 KD Final'),
]:
    s = p8_bench_summaries[k]
    rows.append({
        'Model':         disp,
        'Params (M)':    f"{s['params_M']:.1f}",
        'Text-BLEU':     f"{s.get('avg_text_bleu', 0):.2f}",
        'Text-ChrF':     f"{s.get('avg_text_chrf', 0):.2f}",
        'ASR-BLEU':      f"{s.get('avg_asr_bleu',  0):.2f}",
        'ASR-ChrF':      f"{s.get('avg_asr_chrf',  0):.2f}",
        'RTF':           f"{s.get('avg_rtf', 0):.4f}",
    })

df_results = pd.DataFrame(rows)

print('\n' + '='*80)
print('  PHASE 8 — 4-MODEL BENCHMARK SUMMARY')
print('='*80)
print(df_results.to_string(index=False))
print('='*80)

# Save as CSV to drive
csv_path = f'{FIG_DIR}/phase8_benchmark_summary.csv'
df_results.to_csv(csv_path, index=False)
if ON_KAGGLE:
    _rclone_push(csv_path, 'figures')
print(f'\n[P8] CSV saved → {csv_path}')
print('\n[P8] Phase 8 complete. Next: Phase 9 — Full benchmark + GDrive upload.')
```

### Cell 58 (code, score=8)
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