# Final Notebooks\v2 Multilingual full-final-notebook.ipynb

Extracted notebook map containing markdown headings plus code/output cells likely to matter for reports, reproduction, or agent steering.

## Markdown headings
cell 1: # SeamlessM4T v2 # 5 Languages support: English, Bangla, Hindi, Chinese, Arabic ### Pruning, LoRA finetuning
cell 2: ## ⚙️ Setup — run ALL at the start of EVERY Kaggle session
cell 16: ## 📊 Enhanced Per-Language Tracking Enabled # After running benchmark # Save both
cell 32: ## Phase 0: V1 Baseline Capture
cell 39: ## Phase 1: Vocabulary Pruning — 5 Languages
cell 43: ## Phase 2: Speech Encoder Moderate Pruning (24 → 16 layers)
cell 51: ## Phase 3: T2U LaCo RDSC Merge (6+6 → 4+6 layers)
cell 58: ## Phase 4: Speech Encoder Additional Pruning (16 → 14 layers)
cell 64: ## Phase 5: Text Decoder Pruning (24 → 14 layers)
cell 70: # PHASE 6

## Key cells

### Cell 1 (markdown, score=1)
```markdown
# SeamlessM4T v2 
# 5 Languages support: English, Bangla, Hindi, Chinese, Arabic
### Pruning, LoRA finetuning
```

### Cell 2 (markdown, score=0)
```markdown
## ⚙️ Setup — run ALL at the start of EVERY Kaggle session
```

### Cell 3 (code, score=9)
```python
import os, sys, subprocess, pathlib, re, glob, json, gc, copy, time, math, shutil, random
import warnings; warnings.filterwarnings('ignore')

ON_KAGGLE = os.path.exists('/kaggle/working')
ON_COLAB  = not ON_KAGGLE
PLATFORM  = 'kaggle' if ON_KAGGLE else 'colab'

GDRIVE_MOUNT = '/content/drive/MyDrive/seamTL'   # ← NEW project folder
KAGGLE_WORK  = '/kaggle/working'

WORK_DIR  = KAGGLE_WORK if ON_KAGGLE else GDRIVE_MOUNT
CKPT_DIR  = f'{WORK_DIR}/checkpoints'
AUDIO_DIR = f'{WORK_DIR}/audio'
FIG_DIR   = f'{WORK_DIR}/figures'
MODEL_DIR = f'{WORK_DIR}/models'

GDRIVE_ROOT = 'gdrive:seamTL'   # rclone remote root (Kaggle only)
```

### Cell 4 (code, score=3)
```python
if ON_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print(f'Drive mounted. Working folder: {GDRIVE_MOUNT}')
else:
    print('Kaggle: skipping Drive mount.')
```
OUTPUT:
```text
Kaggle: skipping Drive mount.
```

### Cell 5 (code, score=25)
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

### Cell 6 (code, score=3)
```python
if ON_KAGGLE:
    subprocess.run('curl -s https://rclone.org/install.sh | sudo bash',
                   shell=True, capture_output=True)
    ver = subprocess.run('rclone version', shell=True, capture_output=True, text=True)
    print(ver.stdout.split('\n')[0])
else:
    print('Colab: rclone not needed — using mounted Drive directly.')
    if not os.path.exists('/content/drive/MyDrive'):
        print('WARNING: Drive does not appear to be mounted.')
    else:
        print('Drive mount: OK')
```
OUTPUT:
```text
rclone v1.74.1
```

### Cell 7 (code, score=36)
```python
def _get_secret(key):
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
            raise RuntimeError(f'Colab secret {key!r} not found: {e}')

if ON_KAGGLE:
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
    print('Colab: skipping rclone config.')

try:
    HF_TOKEN = _get_secret('HF_TOKEN')
    from huggingface_hub import login
    login(HF_TOKEN)
    print('HuggingFace login: OK')
except Exception as e:
    print(f'HF login skipped: {e}')
```
OUTPUT:
```text
Drive root:
           0 2026-04-17 11:03:10        -1 Colab Notebooks
           0 2025-11-10 11:33:43        -1 ScholarMate
           0 2026-04-05 12:59:09        -1 cse465
           0 2026-04-12 12:42:04        -1 cse465v5
           0 2026-04-23 09:47:12        -1 seamTL
           0 2026-04-23 20:52:48  
HuggingFace login: OK
```

### Cell 8 (code, score=37)
```python
subprocess.run([
    'pip', 'install', '-q',
    'transformers>=4.41.0', 'datasets', 'torchaudio', 'speechbrain>=1.0.0',
    'peft>=0.10.0', 'librosa', 'jiwer', 'evaluate', 'sacrebleu', 'pyarrow',
    'sentencepiece', 'accelerate', 'matplotlib', 'seaborn',
    'soundfile', 'requests', 'pandas',
], check=True)
print('All packages installed.')
```
OUTPUT:
```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 75.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.1/84.1 kB 6.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.8/100.8 kB 7.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 93.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.6/121.6 kB 10.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 788.2/788.2 kB 37.1 MB/s eta 0:00:00
All packages installed.
```

### Cell 10 (code, score=3)
```python
import torch
import random
import numpy as np

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

### Cell 11 (code, score=115)
```python
# ── pulled verbatim from seamless-cse465v5 Cell 14 ──
import torch
import os, glob, queue, threading, subprocess
from datetime import datetime

_CUSTOM_STATE_FILE = '_custom_state.pt'
_PRUNING_MANIFEST  = 'pruning_manifest.pt'

def _rclone_push(local_path, remote_subpath):
    if not ON_KAGGLE: return
    r = subprocess.run(
        f'rclone copy "{local_path}" "{GDRIVE_ROOT}/{remote_subpath}/" --transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M',
        shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'[rclone] WARNING: push failed for {local_path}: {r.stderr[:200]}')

def _rclone_pull_model(stage_name):
    if not ON_KAGGLE: return
    local = f'{MODEL_DIR}/{stage_name}'
    os.makedirs(local, exist_ok=True)
    r = subprocess.run(
        f'rclone sync "{GDRIVE_ROOT}/models/{stage_name}/" "{local}/" --transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M',
        shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'[rclone] model pull failed for {stage_name}: {r.stderr[:300]}')
    print(f'[rclone] Pulled {stage_name} → {local}')

# --- background upload state ---
_upload_q = queue.Queue()
_upload_pending = set()
_upload_lock = threading.Lock()
_worker_started = False

def _start_upload_worker():
    global _worker_started
    if _worker_started or not ON_KAGGLE:
        return
    t = threading.Thread(target=_upload_worker_loop, daemon=True)
    t.start()
    _worker_started = True

def _upload_worker_loop():
    while True:
        local_path, remote_subpath = _upload_q.get()
        try:
            _rclone_push_blocking(local_path, remote_subpath)
        finally:
            with _upload_lock:
                _upload_pending.discard(local_path)
            _upload_q.task_done()

def _rclone_push_async(local_path, remote_subpath):
    if not ON_KAGGLE:
        return
    _start_upload_worker()
    with _upload_lock:
        _upload_pending.add(local_path)
    _upload_q.put((local_path, remote_subpath))

def _rclone_push_blocking(local_path, remote_subpath):
    # rclone prints progress every 10s
    cmd = [
        "rclone", "copy",
        local_path, f"{GDRIVE_ROOT}/{remote_subpath}/",
        "--transfers=8",
        "--multi-thread-streams=4",
        "--drive-chunk-size=64M",
        "--progress",
        "--stats=10s",
        "--stats-one-line-date",
    ]

    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    for line in p.stdout:
        print(f"[rclone] {line.rstrip()}")

    rc = p.wait()
    if rc != 0:
        print(f"[rclone] WARNING: push failed for {local_path}")

def wait_for_uploads():
    # call once at the very end of training if you need all uploads guaranteed done
    if ON_KAGGLE:
        _upload_q.join()

def save_checkpoint(state, name, step=0, keep=3):
    fname = f"{name}_step{step:06d}.pt"
    path = f"{CKPT_DIR}/{fname}"
    torch.save(state, path)
    mb = os.path.getsize(path) / 1e6
    print(f"[ckpt] Saved {fname} ({mb:.1f} MB)")

    if ON_KAGGLE:
        _rclone_push_async(path, "checkpoints")  # non-blocking enqueue

    # Don't delete checkpoints still uploading
    old = sorted(glob.glob(f"{CKPT_DIR}/{name}_step*.pt"))
    for f in old[:-keep]:
        with _upload_lock:
            in_flight = f in _upload_pending
        if (not in_flight) and os.path.exists(f):
            os.remove(f)


def load_latest_checkpoint(name):
    files = sorted(glob.glob(f'{CKPT_DIR}/{name}_step*.pt'))
    if not files:
        print(f'[ckpt] No checkpoint for {name!r}')
        return None
    state = torch.load(files[-1], map_location='cpu', weights_only=False)
    print(f'[ckpt] Loaded {os.path.basename(files[-1])}')
    return state

def sync_checkpoints_from_drive():
    if ON_KAGGLE:
        print('[ckpt] Syncing from rclone remote...')
        r = subprocess.run(
            f'rclone sync "{GDRIVE_ROOT}/checkpoints/" "{CKPT_DIR}/" --transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M',
            shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print(f'[ckpt] WARNING: {r.stderr[:300]}')
    else:
        print(f'[ckpt] Colab: reading directly from {CKPT_DIR}')
    files = sorted(os.listdir(CKPT_DIR)) if os.path.exists(CKPT_DIR) else []
    print(f'[ckpt] {len(files)} file(s) available')
    for f in files:
        mb = os.path.getsize(f'{CKPT_DIR}/{f}') / 1e6
        print(f'  {f:<55} {mb:>7.1f} MB')

print('Checkpoint helpers ready.')
```
OUTPUT:
```text
Checkpoint helpers ready.
```

### Cell 12 (code, score=236)
```python
import torch.nn as nn, torch.nn.functional as F

_CUSTOM_ATTR_NAMES = ['_vocab_remap_to_old']

def _save_custom_state(mdl, path):
    state = {a: getattr(mdl, a) for a in _CUSTOM_ATTR_NAMES if hasattr(mdl, a)}
    if state:
        torch.save(state, os.path.join(path, _CUSTOM_STATE_FILE))
        print(f'  Saved custom state: {list(state.keys())}')

def _load_custom_state(mdl, path):
    fpath = os.path.join(path, _CUSTOM_STATE_FILE)
    if not os.path.exists(fpath): return
    state = torch.load(fpath, map_location='cpu', weights_only=False)
    for k, v in state.items(): setattr(mdl, k, v)
    print(f'  Restored custom state: {list(state.keys())}')

def _find_layers(component):
    for attr in ['layers', 'inner_layers', 'layer']:
        mod = getattr(component, attr, None)
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            return mod
    return None

def _get_t2u_encoder_decoder(mdl):
    t2u   = getattr(mdl, 't2u_model', None)
    if t2u is None: return None, None
    inner = getattr(t2u, 'model', None)
    if inner is None: return None, None
    return getattr(inner, 'encoder', None), getattr(inner, 'decoder', None)

def sync_model_config(mdl):
    """Keep config in sync with actual ModuleList depths after pruning."""    
    cfg = mdl.config
    if hasattr(mdl, 'speech_encoder'):
        enc = mdl.speech_encoder
        parent = enc.encoder if hasattr(enc, 'encoder') else enc
        if hasattr(parent, 'layers'):
            actual = len(parent.layers)
            for k in ['speech_encoder_layers']:
                if hasattr(cfg, k) and getattr(cfg, k) != actual:
                    print(f'  [config] {k}: {getattr(cfg,k)} -> {actual}')
                    setattr(cfg, k, actual)
            sc = getattr(mdl.speech_encoder, 'config', None)
            if sc and hasattr(sc, 'num_hidden_layers') and sc.num_hidden_layers != actual:
                sc.num_hidden_layers = actual
    if hasattr(mdl, 'text_decoder') and mdl.text_decoder is not None:
        layers = _find_layers(mdl.text_decoder)
        if layers is not None:
            actual = len(layers)
            if hasattr(cfg, 'decoder_layers') and cfg.decoder_layers != actual:
                print(f'  [config] decoder_layers: {cfg.decoder_layers} -> {actual}')
                cfg.decoder_layers = actual
    t2u_enc, t2u_dec = _get_t2u_encoder_decoder(mdl)
    for sub, attr in [(t2u_enc,'t2u_encoder_layers'), (t2u_dec,'t2u_decoder_layers')]:
        if sub is None: continue
        layers = _find_layers(sub)
        if layers and hasattr(cfg, attr) and getattr(cfg, attr) != len(layers):
            print(f'  [config] {attr}: {getattr(cfg,attr)} -> {len(layers)}')
            setattr(cfg, attr, len(layers))
    t2u = getattr(mdl, 't2u_model', None)
    if t2u and hasattr(t2u, 'config'):
        tc = t2u.config
        for sub, attr in [(t2u_enc,'encoder_layers'), (t2u_dec,'decoder_layers')]:
            if sub is None: continue
            layers = _find_layers(sub)
            if layers and hasattr(tc, attr) and getattr(tc, attr) != len(layers):
                print(f'  [config] t2u.config.{attr}: {getattr(tc,attr)} -> {len(layers)}')
                setattr(tc, attr, len(layers))
    print('  [config] sync done.')

def _consolidate_to_single_gpu(mdl):
    """Move model to cuda:0 if split by device_map='auto'."""    
    if not torch.cuda.is_available(): return mdl
    if not (hasattr(mdl, 'hf_device_map') and len(set(mdl.hf_device_map.values())) > 1):
        return mdl
    print('  Multi-device → consolidating to cuda:0...')
    try:
        from accelerate.hooks import remove_hook_from_submodules
        remove_hook_from_submodules(mdl)
    except Exception: pass
    mdl = mdl.to('cuda:0')
    torch.cuda.empty_cache()
    print(f'  Model now on: {next(mdl.parameters()).device}')
    return mdl

def load_hf_weights_dict(model_dir):
    from pathlib import Path
    safe = Path(model_dir) / 'model.safetensors'
    if safe.is_file():
        try:
            from safetensors.torch import load_file
            return load_file(str(safe))
        except ImportError: pass
    pt = Path(model_dir) / 'pytorch_model.bin'
    if pt.is_file():
        blob = torch.load(str(pt), map_location='cpu', weights_only=False)
        return blob.get('model', blob) if isinstance(blob, dict) else blob
    return None

def _infer_t2u_layer_counts(model_dir):
    sd = load_hf_weights_dict(model_dir)
    if not sd: return None, None
    enc_idx, dec_idx = set(), set()
    for k in sd:
        if k.startswith('t2u_model.model.encoder.layers.'):
            r = k.split('.')[4]
            if r.isdigit(): enc_idx.add(int(r))
        elif k.startswith('t2u_model.model.decoder.layers.'):
            r = k.split('.')[4]
            if r.isdigit(): dec_idx.add(int(r))
    return (max(enc_idx)+1 if enc_idx else None), (max(dec_idx)+1 if dec_idx else None)

def save_model_to_drive(mdl, proc, stage_name, manifest_extra=None):
    """Save model to Drive using HF save_pretrained (battle-tested from v5)."""    
    target = f'{MODEL_DIR}/{stage_name}'
    os.makedirs(target, exist_ok=True)
    print(f'[model] Saving {stage_name} → {target} ...')
    sync_model_config(mdl)
    _save_custom_state(mdl, target)
    man = {'stage_name': stage_name}
    if manifest_extra: man.update(manifest_extra)
    torch.save(man, os.path.join(target, _PRUNING_MANIFEST))
    try:
        mdl.save_pretrained(target, safe_serialization=True)
    except Exception as e:
        print(f'  safe_serialization failed ({e}); trying .bin')
        mdl.save_pretrained(target)
    if proc is not None: proc.save_pretrained(target)
    total = sum(os.path.getsize(f'{target}/{f}') for f in os.listdir(target)) / 1e6
    print(f'[model] Local: {total:.0f} MB in {len(os.listdir(target))} files.')
    if ON_KAGGLE:
        r = subprocess.run(f'rclone sync "{target}/" "{GDRIVE_ROOT}/models/{stage_name}/" --transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M',
                           shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print(f'[model] WARNING rclone push failed: {r.stderr[:300]}')
        else:
            print(f'[model] Pushed to remote: {GDRIVE_ROOT}/models/{stage_name}/')
    else:
        print('[model] Colab: saved directly to Drive.')
```
OUTPUT:
```text
Model I/O helpers ready.
```

### Cell 13 (code, score=61)
```python
import numpy as np, matplotlib.pyplot as plt, matplotlib, seaborn as sns
matplotlib.rcParams.update({'font.size': 11, 'figure.dpi': 120, 'savefig.bbox': 'tight'})
sns.set_style('whitegrid')

N_GPU = torch.cuda.device_count()
print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs {N_GPU}')
for i in range(N_GPU):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU{i}: {torch.cuda.get_device_name(i)}  {p.total_memory/1e9:.1f} GB')

def count_params(module):
    return sum(p.numel() for p in module.parameters()) / 1e6

def count_params_detailed(model):
    bd = {n: count_params(c) for n, c in model.named_children()}
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
        for i in range(N_GPU):
            a = torch.cuda.memory_allocated(i)/1e9
            r = torch.cuda.memory_reserved(i)/1e9
            print(f'  GPU{i}: {a:.2f}GB alloc / {r:.2f}GB reserved')

def save_figure(fig, name):
    fig.savefig(f'{FIG_DIR}/{name}', dpi=150, bbox_inches='tight')
    if ON_KAGGLE: _rclone_push(f'{FIG_DIR}/{name}', 'figures')
    print(f'[fig] Saved {name}')

import torchaudio
from IPython.display import Audio as IPAudio, display

def play(audio, sr, label=''):
    if hasattr(audio, 'numpy'): audio = audio.squeeze().numpy()
    print(f'  {label}  ({len(audio)/sr:.1f}s | sr={sr})')
    display(IPAudio(audio, rate=int(sr)))

def save_audio(audio, sr, filename):
    path = f'{AUDIO_DIR}/{filename}'
    if not isinstance(audio, torch.Tensor): audio = torch.tensor(audio)
    torchaudio.save(path, audio.squeeze().unsqueeze(0).float().cpu(), sr)
    print(f'[audio] Saved {filename}')

print('Core utilities ready.')
```
OUTPUT:
```text
PyTorch 2.10.0+cu128 | CUDA True | GPUs 2
  GPU0: Tesla T4  15.6 GB
  GPU1: Tesla T4  15.6 GB
Core utilities ready.
```

### Cell 14 (code, score=82)
```python
def _load_summaries_from_drive():
    ckpt = load_latest_checkpoint('all_summaries')
    if ckpt and 'summaries' in ckpt:
        return {s['label']: s for s in ckpt['summaries']}
    return {}

ALL_SUMMARIES: dict = _load_summaries_from_drive()
print(f'Loaded {len(ALL_SUMMARIES)} existing summaries: {list(ALL_SUMMARIES.keys())}')

def store_summary(s):
    label = s['label']
    ALL_SUMMARIES[label] = s.copy()
    save_checkpoint({'summaries': list(ALL_SUMMARIES.values())}, 'all_summaries', 0)
    print(f'[summary] Stored {label} ({len(ALL_SUMMARIES)} total)')

def get_summaries():
    return sorted(ALL_SUMMARIES.values(), key=lambda s: s['label'])

def plot_phase_comparison(summaries=None, save_name='phase_comparison.png'):
    data = summaries or get_summaries()
    if not data: 
        print('No summaries yet.'); 
        return
    
    # Sort by label to ensure consistent ordering
    data = sorted(data, key=lambda s: s['label'])
    labels = [s['label'] for s in data]
    
    print(f'Plotting {len(data)} phases: {labels}')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Textless S2ST Compression Pipeline: Phase Comparison',
                 fontsize=15, fontweight='bold')
    metrics = [('avg_bleu', 'ASR-BLEU (higher=better)', '#2196F3'),
               ('avg_chrf', 'ASR-ChrF (higher=better)', '#4CAF50'),
               ('avg_rtf',  'RTF (lower=faster)',        '#FF9800'),
               ('params_M', 'Parameters (M)',            '#9C27B0')]
    
    for ax, (key, title, color) in zip(axes.flat, metrics):
        vals = [s.get(key, 0) for s in data]
        x_pos = range(len(labels))
        bars = ax.bar(x_pos, vals, color=color, alpha=0.85, edgecolor='white', width=0.7)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, v in zip(bars, vals):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height, 
                       f'{v:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
def plot_size_vs_quality(summaries=None, save_name='size_vs_quality.png'):
    data = summaries or get_summaries()
    if not data: return
    fig, ax = plt.subplots(figsize=(10, 7))
    params = [s['params_M'] for s in data]
    chrf   = [s['avg_chrf'] for s in data]
    bleu   = [s['avg_bleu'] for s in data]
    ax.scatter(params, bleu, s=120, c='#2196F3', zorder=5, label='ASR-BLEU')
    ax.scatter(params, chrf, s=120, c='#4CAF50', marker='s', zorder=5, label='ASR-ChrF')
    for i, lbl in enumerate([s['label'] for s in data]):
        ax.annotate(lbl, (params[i], bleu[i]), fontsize=7, xytext=(5,5),
                    textcoords='offset points')
    ax.set_xlabel('Parameters (M)'); ax.set_ylabel('Score')
    ax.set_title('Model Size vs Translation Quality', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()

print('Plotting helpers ready.')
```
OUTPUT:
```text
[ckpt] No checkpoint for 'all_summaries'
Loaded 0 existing summaries: []
Plotting helpers ready.
```

### Cell 15 (code, score=241)
```python
"""
QUICK INTEGRATION SNIPPET
Copy-paste this entire cell into seamless-final.ipynb after the existing summary functions
This is a condensed version for immediate use
"""

# ============================================================================
# ENHANCED TRACKING - Insert after ALL_SUMMARIES definition
# ============================================================================

def _load_detailed_summaries_from_drive():
    ckpt = load_latest_checkpoint('all_detailed_summaries')
    if ckpt and 'detailed_summaries' in ckpt:
        return {s['label']: s for s in ckpt['detailed_summaries']}
    return {}

ALL_DETAILED_SUMMARIES = _load_detailed_summaries_from_drive()
print(f'Loaded {len(ALL_DETAILED_SUMMARIES)} detailed summaries')

def store_detailed_summary(s):
    label = s['label']
    ALL_DETAILED_SUMMARIES[label] = s.copy()
    save_checkpoint({'detailed_summaries': list(ALL_DETAILED_SUMMARIES.values())}, 
                    'all_detailed_summaries', 0)
    print(f'[detailed] Stored {label}')

def compute_detailed_summary(results, label, params_M):
    from collections import defaultdict
    by_pair = defaultdict(list)
    for r in results:
        if not math.isnan(r.get('rtf', float('nan'))):
            by_pair[f"{r['src_lang']}→{r['tgt_lang']}"].append(r)
    
    pair_stats = {}
    for pair_key, pair_results in by_pair.items():
        pair_stats[pair_key] = {
            'n_samples': len(pair_results),
            'avg_bleu': float(np.mean([r['bleu'] for r in pair_results])),
            'avg_chrf': float(np.mean([r['chrf'] for r in pair_results])),
            'avg_rtf': float(np.mean([r['rtf'] for r in pair_results])),
            'std_chrf': float(np.std([r['chrf'] for r in pair_results])),
        }
    
    valid = [r for r in results if not math.isnan(r.get('rtf', float('nan')))]
    by_src = defaultdict(list)
    by_tgt = defaultdict(list)
    for r in valid:
        by_src[r['src_lang']].append(r)
        by_tgt[r['tgt_lang']].append(r)
    
    return {
        'label': label, 'params_M': params_M, 'n_total': len(valid),
        'avg_bleu': float(np.mean([r['bleu'] for r in valid])),
        'avg_chrf': float(np.mean([r['chrf'] for r in valid])),
        'avg_rtf': float(np.mean([r['rtf'] for r in valid])),
        'std_chrf': float(np.std([r['chrf'] for r in valid])),
        'pair_stats': pair_stats,
        'by_src_lang': {lang: {
            'n_samples': len(rs),
            'avg_chrf': float(np.mean([r['chrf'] for r in rs])),
            'avg_bleu': float(np.mean([r['bleu'] for r in rs])),
        } for lang, rs in by_src.items()},
        'by_tgt_lang': {lang: {
            'n_samples': len(rs),
            'avg_chrf': float(np.mean([r['chrf'] for r in rs])),
            'avg_bleu': float(np.mean([r['bleu'] for r in rs])),
        } for lang, rs in by_tgt.items()},
    }

def plot_detailed_phase_comparison(save_name='detailed_comparison.png'):
    """
    Drop-in replacement: saves each panel as an individual research-paper-quality PNG.
    save_name is used as a prefix base (extension is ignored).
    e.g. 'detailed_comparison.png' → 'detailed_comparison_01_overall_quality.png', etc.
    """
    summaries = sorted(ALL_DETAILED_SUMMARIES.values(), key=lambda s: s['label'])
    if not summaries:
        print('No detailed summaries yet.')
        return

    print(f'Plotting detailed comparison for {len(summaries)} phases: {[s["label"] for s in summaries]}')

    # ── Derive a clean prefix from save_name ──────────────────────────────────
    base_prefix = save_name.replace('.png', '').replace('.jpg', '').replace('.pdf', '')

    labels = [s['label'] for s in summaries]
    chrfs  = [s['avg_chrf'] for s in summaries]
    bleus  = [s['avg_bleu'] for s in summaries]

    # ── Shared style — applied via rcParams context manager ───────────────────
    STYLE = {
        'font.family':       'DejaVu Sans',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.3,
        'grid.linestyle':    '--',
        'axes.titlesize':    15,
        'axes.titleweight':  'bold',
        'axes.labelsize':    13,
        'xtick.labelsize':   11,
        'ytick.labelsize':   11,
        'legend.fontsize':   11,
        'figure.dpi':        180,
    }

    COLOR_CHRF  = '#2E86AB'
    COLOR_BLEU  = '#E84855'
    COLOR_RTF   = '#F4A261'
    ALPHA       = 0.88
    FIG_W       = 10
    FIG_H       = 6

    saved = []

    def _savefig(fig, tag, title_for_log):
        fname = f'{base_prefix}_{tag}.png'
        fig.savefig(fname, dpi=180, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close(fig)
        saved.append(fname)
        print(f'  ✓ Saved: {fname}  [{title_for_log}]')

    # ══════════════════════════════════════════════════════════════════════════
    # FIG 1 — Overall ChrF / BLEU per phase
    # ══════════════════════════════════════════════════════════════════════════
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        x  = np.arange(len(labels))
        bw = 0.35
        ax.bar(x - bw/2, chrfs, bw, label='ChrF', color=COLOR_CHRF, alpha=ALPHA, edgecolor='white', linewidth=0.8)
        ax.bar(x + bw/2, bleus, bw, label='BLEU', color=COLOR_BLEU, alpha=ALPHA, edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Overall Translation Quality per Phase')
        ax.legend()
        for i, (c, b) in enumerate(zip(chrfs, bleus)):
            ax.text(i - bw/2, c + 0.4, f'{c:.1f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + bw/2, b + 0.4, f'{b:.1f}', ha='center', va='bottom', fontsize=9)
```
OUTPUT:
```text
[ckpt] No checkpoint for 'all_detailed_summaries'
Loaded 0 detailed summaries
✓ Enhanced tracking loaded: store_detailed_summary(), compute_detailed_summary(), plot_detailed_phase_comparison(), print_detailed_summary_table()
```

### Cell 16 (markdown, score=23)
```markdown
## 📊 Enhanced Per-Language Tracking Enabled

**New functions available:**
- `compute_detailed_summary(results, label, params_M)` - Extract per-language metrics
- `store_detailed_summary(summary)` - Save to checkpoint
- `plot_detailed_phase_comparison()` - 9-panel visualization
- `print_detailed_summary_table(phase_label)` - Text output

**To use in benchmark cells:**
```python
# After running benchmark
p0_results, p0_summary = run_benchmark_asr(model, samples, 'P0_Label', save_n=4)
p0_detailed = compute_detailed_summary(p0_results, 'P0_Label', p0_summary['params_M'])

# Save both
save_checkpoint({
    'results': p0_results,
    'summary': p0_summary,
    'detailed_summary': p0_detailed  # NEW
}, 'phase0_benchmark', 0)

store_summary(p0_summary)
store_detailed_summary(p0_detailed)  # NEW
print_detailed_summary_table('P0_Label')  # NEW
plot_detailed_phase_comparison()  # NEW
```

**All per-language data now preserved in checkpoints!**
```

### Cell 17 (code, score=94)
```python
# ── MMS-ASR for Bengali, Hindi, Arabic ──────────────────────────────────────
import gc as _stdlib_gc

_MMS_MODEL_ID = 'facebook/mms-1b-all'
_mms_asr_models = {}  # Cache models per language
_mms_asr_processors = {}

def _ensure_mms_loaded(lang_code):
    """Load MMS model for specific language (ben, hin, ara)"""
    global _mms_asr_models, _mms_asr_processors
    if lang_code in _mms_asr_models: return
    
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    print(f'[MMS-ASR] Loading {_MMS_MODEL_ID} lang={lang_code}...')
    _mms_asr_processors[lang_code] = AutoProcessor.from_pretrained(
        _MMS_MODEL_ID, target_lang=lang_code)
    _mms_asr_models[lang_code] = Wav2Vec2ForCTC.from_pretrained(
        _MMS_MODEL_ID, target_lang=lang_code,
        ignore_mismatched_sizes=True, torch_dtype=torch.float16)
    _mms_asr_models[lang_code].load_adapter(lang_code)
    _mms_asr_models[lang_code] = _mms_asr_models[lang_code].eval()
    try: 
        _mms_asr_models[lang_code] = _mms_asr_models[lang_code].to('cuda:0')
    except RuntimeError: 
        pass
    print(f'[MMS-ASR] {lang_code} ready.')

def asr_transcribe_mms(audio_np, lang_code, sr=16000):
    _ensure_mms_loaded(lang_code)
    if audio_np is None or len(audio_np) < 400:
        return ''

    if sr != 16000:
        audio_np = torchaudio.functional.resample(
            torch.tensor(audio_np), sr, 16000).numpy()

    model = _mms_asr_models[lang_code]
    processor = _mms_asr_processors[lang_code]

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    inputs = processor(audio_np, sampling_rate=16000, return_tensors='pt')
    input_values = inputs.input_values.to(device).to(dtype)

    with torch.no_grad():
        logits = model(input_values=input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0].strip()

# ── Whisper-medium for English and Chinese ──────────────────────────────────
_whisper_model = None
_whisper_processor = None

def _ensure_whisper_loaded():
    global _whisper_model, _whisper_processor
    if _whisper_model is not None: return
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    print('[Whisper] Loading openai/whisper-medium...')
    _whisper_processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    _whisper_model = WhisperForConditionalGeneration.from_pretrained(
        'openai/whisper-medium', torch_dtype=torch.float16)
    _whisper_model = _whisper_model.eval()
    try:
        device = 'cuda:1' if N_GPU > 1 else 'cuda:0'
        _whisper_model = _whisper_model.to(device)
    except RuntimeError:
        pass
    print('[Whisper] Ready.')

def asr_transcribe_whisper(audio_np, lang='en', sr=16000):
    """
    Transcribe audio using Whisper-medium for English or Chinese.
    lang: 'en' for English, 'zh' for Chinese
    """
    _ensure_whisper_loaded()
    if audio_np is None or len(audio_np) < 400: return ''
    
    # Resample if needed
    if sr != 16000:
        audio_np = torchaudio.functional.resample(
            torch.tensor(audio_np), sr, 16000).numpy()
    
    device = next(_whisper_model.parameters()).device
    dtype = next(_whisper_model.parameters()).dtype
    
    # Whisper language codes
    whisper_lang = 'en'
    
    try:
        # Process audio - ensure correct dtype
        inputs = _whisper_processor(
            audio_np, 
            sampling_rate=16000, 
            return_tensors='pt',
            return_attention_mask=True)
        
        # Move to device and convert to model dtype
        input_features = inputs['input_features'].to(device).to(dtype)
        
        # Use modern task/language parameters instead of forced_decoder_ids
        with torch.no_grad():
            predicted_ids = _whisper_model.generate(
                input_features,
                language=whisper_lang,
                task='transcribe',
                max_new_tokens=256,
                num_beams=1,
                do_sample=False)
        
        transcription = _whisper_processor.batch_decode(
            predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except Exception as e:
        print(f'[Whisper] Error: {e}')
        import traceback
        traceback.print_exc()
        return ''

# ── M4T lang → ASR backend mapping ──────────────────────────────────────────
M4T_FLEURS_MAP = {
    'eng': 'en_us', 'ben': 'bn_in', 'cmn': 'cmn_hans_cn',
    'arb': 'ar_eg', 'hin': 'hi_in',
}

# MMS language codes - UPDATED to include Chinese
MMS_LANG_MAP = {
    'ben': 'ben',  # Bengali
    'hin': 'hin',  # Hindi
    'arb': 'ara',  # Arabic (MMS uses 'ara' for Arabic)
    'cmn': 'cmn',  # Chinese Mandarin (ADDED)
}

# UPDATED: Whisper only for English, MMS for all others
LANG_ASR_CONFIG = {
    'ben': ('mms', 'ben'),       # MMS for Bengali
    'hin': ('mms', 'hin'),       # MMS for Hindi
    'arb': ('mms', 'ara'),       # MMS for Arabic
    'cmn': ('mms', 'cmn-script_simplified'),       # MMS for Chinese (CHANGED from Whisper)
```
OUTPUT:
```text
ASR stack ready:
  - Whisper-medium: English only
  - MMS-1b-all: Bengali, Hindi, Arabic, Chinese
```

### Cell 18 (code, score=3)
```python
# print(len(eval_samples))
# sasa = eval_samples[52]
# print(sasa['tgt_lang'])
# _, wav_out = run_s2st(model_v1, sasa['wav'], tgt_lang=sasa['tgt_lang'])
# asr_transcribe(wav_out, sasa['tgt_lang'] , sr=16000)
```

### Cell 20 (code, score=40)
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
    """Full S2ST for models with text decoder (Phases 0-3)."""
    inputs = processor(audio=wav, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}
    with torch.no_grad():
        try:
            out  = mdl.generate(**inputs, tgt_lang=tgt_lang,
                                return_intermediate_token_ids=True)
            text_ids = _remap_ids_for_decode(mdl, out.sequences.cpu())
            text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
            wav_out = out.waveform.cpu().numpy().squeeze() if out.waveform is not None else np.zeros(16000)
            return text, wav_out
        except RuntimeError:
            return run_s2t_only(mdl, wav, tgt_lang), np.zeros(16000)

def run_s2t_only(mdl, wav, tgt_lang='ben'):
    """Text-only generation (for benchmarking text-decoder models)."""
    inputs = processor(audio=wav, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(_model_input_device(mdl)) for k, v in inputs.items()}
    orig_voc = mdl.vocoder
    inp_dev  = next(iter(inputs.values())).device
    class _Noop(nn.Module):
        def forward(self, *a, **kw): return torch.zeros(1,1,device=inp_dev), [1]
    mdl.vocoder = _Noop()
    try:
        with torch.no_grad():
            out = mdl.generate(**inputs, tgt_lang=tgt_lang,
                               return_intermediate_token_ids=True)
    finally:
        mdl.vocoder = orig_voc
    text_ids = _remap_ids_for_decode(mdl, out.sequences.cpu())
    return processor.batch_decode(text_ids, skip_special_tokens=True)[0]

def quick_eval_chrf(mdl, samples, max_samples=16, group_size=25):
    """
    Optimized: Only load audio for samples we actually use.
    """
    scores = []
    num_langs = len(samples) // group_size
    per_lang = max(1, max_samples // num_langs)
    
    for i in range(num_langs):
        start = i * group_size
        
        # ✅ OPTIMIZED: Only load the samples we need
        for j in range(per_lang):
            
            idx = start + j
            if idx >= len(samples):
                break
            
            s = samples[idx]  # Load only this one sample
            tgt = s.get('tgt_lang', 'ben')
            _, wav_out = run_s2st(mdl, s['wav'], tgt_lang=tgt)
            pred = asr_transcribe(wav_out, tgt)
            scores.append(compute_chrf(pred, s['ref']))
    
    return float(np.mean(scores))


print('Benchmark functions ready.')
```
OUTPUT:
```text
Benchmark functions ready.
```

### Cell 21 (code, score=69)
```python
import jieba

def zh_tokenize(text):
    return " ".join(jieba.lcut(text.replace(" ", "")))


def run_benchmark_asr(mdl, samples, label='model', save_n=4):
    """ASR-based benchmark: translate audio → ASR transcribe → compute ASR-ChrF/BLEU."""
    print(f'\n{"="*60}\n  BENCHMARK (ASR): {label}  Samples:{len(samples)}\n{"="*60}')
    gpu_mem()
    results = []
    
    # Group samples by language pair for organized output
    from collections import defaultdict
    by_pair = defaultdict(list)
    for s in samples:
        by_pair[f"{s['src_lang']}→{s['tgt_lang']}"].append(s)
    
    for pair_key, pair_samples in by_pair.items():
        print(f'\n  === {pair_key} ({len(pair_samples)} samples) ===')
        for i, s in enumerate(pair_samples):
            try:
                dur = len(s['wav']) / 16000
                t0  = time.time()
                # Run S2ST translation
                _, wav_out = run_s2st(mdl, s['wav'], tgt_lang=s['tgt_lang'])
                rtf  = (time.time() - t0) / dur
                
                # ASR transcribe output audio
                pred = asr_transcribe(wav_out, s['tgt_lang'])
                
                ref = s['ref']
                hyp = pred
                if s['tgt_lang'] == 'cmn':
                    print("bench cmn")
                    ref_clean = ref.replace(" ", "")
                    hyp_clean = hyp.replace(" ", "")
                
                    # BLEU (tokenized)
                    ref_bleu = zh_tokenize(ref_clean)
                    hyp_bleu = zh_tokenize(hyp_clean)
                
                    # chrF (raw)
                    ref_chrf = ref_clean
                    hyp_chrf = hyp_clean

                    bleu = compute_bleu(hyp_bleu, ref_bleu)
                    chrf = compute_chrf(hyp_chrf, ref_chrf)    
                else:
                    bleu = compute_bleu(pred, ref)
                    chrf = compute_chrf(pred, ref)

                print(f'  [{i+1:>2}/{len(pair_samples)}] ASR-BLEU={bleu:5.1f} ASR-ChrF={chrf:5.1f} RTF={rtf:.3f}')
                print(f'              pred: {pred[:80]}')
                
                if save_n > 0 and i < save_n:
                    play(s['wav'], 16000, label=f'{label}_{pair_key}_s{i+1}in.wav')
                    save_audio(s['wav'], 16000, f'{label}_{pair_key}_s{i+1}in.wav')
                    play(wav_out, 16000, label=f'{label}_{pair_key}_s{i+1}out.wav')
                    save_audio(wav_out, 16000, f'{label}_{pair_key}_s{i+1}out.wav')
                
                results.append(dict(
                    id=s['id'], src_lang=s['src_lang'], tgt_lang=s['tgt_lang'],
                    bleu=bleu, chrf=chrf, rtf=rtf, pred=pred, ref=s['ref']))
            except Exception as e:
                import traceback; traceback.print_exc()
                results.append(dict(
                    id=s['id'], src_lang=s.get('src_lang','?'), tgt_lang=s.get('tgt_lang','?'),
                    bleu=0, chrf=0, rtf=float('nan'), pred='', ref=s.get('ref','')))
    
    valid = [r for r in results if not math.isnan(r['rtf'])]
    summary = dict(
        label=label, n=len(valid),
        avg_bleu=float(np.mean([r['bleu'] for r in valid])) if valid else 0,
        avg_chrf=float(np.mean([r['chrf'] for r in valid])) if valid else 0,
        avg_rtf =float(np.mean([r['rtf']  for r in valid])) if valid else 0,
        params_M=count_params(mdl)
    )
    
    # Per-pair breakdown
    print(f'\n  === Summary by Language Pair ===')
    for pair_key in by_pair.keys():
        pair_res = [r for r in valid if f"{r['src_lang']}→{r['tgt_lang']}" == pair_key]
        if pair_res:
            avg_chrf_pair = np.mean([r['chrf'] for r in pair_res])
            avg_bleu_pair = np.mean([r['bleu'] for r in pair_res])
            print(f'  {pair_key:<12} ASR-ChrF={avg_chrf_pair:5.2f}  ASR-BLEU={avg_bleu_pair:5.2f}')
    
    print(f'\n  Overall: ASR-BLEU={summary["avg_bleu"]:.2f} ASR-ChrF={summary["avg_chrf"]:.2f}'
          f' RTF={summary["avg_rtf"]:.4f} Params={summary["params_M"]:.1f}M')
    return results, summary

# Alias for backward compatibility
run_benchmark = run_benchmark_asr
```

### Cell 22 (code, score=51)
```python
from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor

try:
    HF_TOKEN = _get_secret('HF_TOKEN')
    from huggingface_hub import login
    login(HF_TOKEN)
    print('Logged into HuggingFace Hub.')
except Exception as e:
    print(f'HF login skipped: {e}')

MODEL_NAME = 'facebook/seamless-m4t-v2-large'
processor = None   # Will be set when loading any model

def load_base_model():
    global processor
    print(f'Loading processor from {MODEL_NAME}...')
    proc = SeamlessM4TProcessor.from_pretrained(MODEL_NAME)
    print(f'Loading model -- may take 5-10 min...')
    mdl = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map='auto')
    mdl.eval()
    print('Model loaded.'); gpu_mem()
    processor = proc
    return mdl, proc

print('load_base_model() ready. Call it to load teacher/base model.')
```
OUTPUT:
```text
Logged into HuggingFace Hub.
load_base_model() ready. Call it to load teacher/base model.
```

### Cell 23 (code, score=97)
```python
## Dataset loading — battle-tested from seamless-cse465v5 (Cells 24-26)
import concurrent.futures, io, soundfile as sfile, pandas as pd

# LOCAL_PARQUET_CACHE = '/kaggle/working/fleurs_parquet'
LOCAL_PARQUET_CACHE = '/kaggle/input/datasets/rayedriasat/fleurs5'
BASE_PARQUET_URL = 'https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet'
DRIVE_FLEURS_PATH = f'{GDRIVE_ROOT}/fleurs_parquet'

def _list_parquet_urls(lang, split):
    import requests
    urls, i = [], 0
    while True:
        url = f'{BASE_PARQUET_URL}/{lang}/{split}/{i:04d}.parquet?download=true'
        try:
            r = requests.head(url, timeout=15, allow_redirects=True)
            if r.status_code == 200: urls.append(url); i += 1
            else: break
        except: break
    if not urls:
        urls = [f'{BASE_PARQUET_URL}/{lang}/{split}/0000.parquet?download=true']
        print(f'  [WARN] fallback to shard 0000 for {lang}/{split}')
    print(f'  [shards] {lang}/{split}: {len(urls)} shard(s)')
    return urls

def _download_shard(args):
    import requests
    url, dest = args
    dest = pathlib.Path(dest)
    if dest.exists() and dest.stat().st_size > 1024*1024:
        return url, True, 'cached'
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(8*1024*1024):
                    if chunk: f.write(chunk)
            if dest.stat().st_size > 1024*1024: return url, True, 'downloaded'
            raise RuntimeError('Downloaded file too small')
        except Exception as e:
            if dest.exists(): dest.unlink()
            if attempt == 2: return url, False, str(e)
    return url, False, 'unknown'

def load_fleurs_parallel(src_lang, tgt_lang, split='train', n_workers=4):
    from datasets import Dataset
    tasks = []
    for lang in [src_lang, tgt_lang]:
        urls = _list_parquet_urls(lang, split)
        for i, url in enumerate(urls):
            dest = f'{LOCAL_PARQUET_CACHE}/{lang}/{split}_{i:04d}.parquet'
            tasks.append((url, dest))
    print(f'[Parallel] Downloading {len(tasks)} shards...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for url, ok, msg in pool.map(_download_shard, tasks):
            print(f'  {"OK" if ok else "FAIL"}: {msg}')
    def _load_lang(lang):
        if lang == "": 
            return None
        files = sorted(glob.glob(f'{LOCAL_PARQUET_CACHE}/{lang}/{split}_*.parquet'))
        if not files: raise FileNotFoundError(f'No cached shards for {lang}')
        return Dataset.from_pandas(pd.read_parquet(files[0]))
    return _load_lang(src_lang), _load_lang(tgt_lang)

def push_fleurs_to_drive():
    if not ON_KAGGLE: return
    subprocess.run(f'rclone copy "{LOCAL_PARQUET_CACHE}/" "{DRIVE_FLEURS_PATH}/" --transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M',
                   shell=True, capture_output=True)

def load_fleurs_from_drive(src_lang, tgt_lang, split='train'):
    from datasets import Dataset
    if not ON_KAGGLE: return None, None
    if not os.path.exists(LOCAL_PARQUET_CACHE):
        r = subprocess.run(f'rclone copy "{DRIVE_FLEURS_PATH}/" "{LOCAL_PARQUET_CACHE}/" --transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M',
                           shell=True, capture_output=True, text=True)
        if r.returncode != 0: return None, None
    def _load_lang(lang):
        if lang == "": 
            return None
        files = sorted(glob.glob(f'{LOCAL_PARQUET_CACHE}/{lang}/{split}_*.parquet'))
        if not files: return None
        return Dataset.from_pandas(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
    src_ds = _load_lang(src_lang); tgt_ds = _load_lang(tgt_lang)
    if src_ds and tgt_ds: print(f'[gdrive] Loaded: {len(src_ds)} src, {len(tgt_ds)} tgt')
    return src_ds, tgt_ds

def _load_wav(audio_cell):
    """Verbatim from v5 Cell 25 — handles both HF Dataset and parquet byte formats."""    
    audio = audio_cell
    if isinstance(audio, dict) and 'array' in audio:
        arr, sr = audio['array'], audio['sampling_rate']
    elif isinstance(audio, dict) and 'bytes' in audio:
        wav, sr = sfile.read(io.BytesIO(audio['bytes']))
        if wav.ndim > 1: wav = wav.mean(axis=1)
        arr = wav
    else:
        raise RuntimeError(f'Unsupported audio format: {type(audio)}')
    arr = np.array(arr, dtype=np.float32)
    if sr != 16000:
        arr = torchaudio.functional.resample(torch.tensor(arr), sr, 16000).numpy()
    return arr

print('FLEURS data loaders ready.')
```
OUTPUT:
```text
FLEURS data loaders ready.
```

### Cell 24 (code, score=7)
```python
# from datasets import Dataset
# tasks = []
# split = 'validation'
# for lang in ['ar_eg', 'bn_in', 'cmn_hans_cn', 'en_us', 'hi_in']:
#     urls = _list_parquet_urls(lang, split)
#     for i, url in enumerate(urls):
#         dest = f'{LOCAL_PARQUET_CACHE}/{lang}/{split}_{i:04d}.parquet'
#         tasks.append((url, dest))
# print(f'[Parallel] Downloading {len(tasks)} shards...')
# with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
#     for url, ok, msg in pool.map(_download_shard, tasks):
#         print(f'  {"OK" if ok else "FAIL"}: {msg}')
# push_fleurs_to_drive()
```

### Cell 25 (code, score=106)
```python
import os, glob, torch
from datetime import datetime

def session_status():
    print('=' * 65)
    print(f'  Platform : {PLATFORM}   Time : {datetime.now():%Y-%m-%d %H:%M}')
    if os.path.exists(CKPT_DIR):
        files = [f for f in glob.glob(f'{CKPT_DIR}/**/*.pt', recursive=True) if os.path.isfile(f)]
        print(f'  Checkpoint files: {len(files)}')
        for f in sorted(files)[:20]:
            print(f'    {os.path.relpath(f,CKPT_DIR):<50} {os.path.getsize(f)/1e6:>8.1f} MB')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU: {torch.cuda.get_device_name(0)}  VRAM: {props.total_memory/1e9:.1f} GB')
    print('=' * 65)

if not os.path.exists(LOCAL_PARQUET_CACHE):
    r = subprocess.run(f'rclone copy "{DRIVE_FLEURS_PATH}/" "{LOCAL_PARQUET_CACHE}/" --transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M',
                       shell=True, capture_output=True, text=True)

sync_checkpoints_from_drive()
session_status()
print('\n✓ ALL SETUP CELLS COMPLETE — proceed to phases.')
```
OUTPUT:
```text
[ckpt] Syncing from rclone remote...
[ckpt] 14 file(s) available
  all_detailed_summaries_step000000.pt                        0.0 MB
  all_summaries_step000000.pt                                 0.0 MB
  phase0_benchmark_step000000.pt                              0.1 MB
  phase1_benchmark_step000000.pt                              0.1 MB
  phase2_benchmark_step000000.pt                              0.1 MB
  phase2_enc_pruning_step000000.pt                            0.0 MB
  phase3_benchmark_step000000.pt                              0.1 MB
  phase3_laco_done_step000000.pt                              0.0 MB
  phase4_benchmark_step000000.pt                              0.1 MB
  phase4_enc_pruning_step000000.pt                            0.0 MB
  phase5_benchmark_step000000.pt                              0.1 MB
  phase5_dec_pruning_step000000.pt                            0.0 MB
  phase6_benchmark_step000000.pt                              0.1 MB
  phase7_benchmark_step000000.pt                              0.1 MB
=================================================================
  Platform : kaggle   Time : 2026-05-14 04:17
  Checkpoint files: 14
    all_detailed_summaries_step000000.pt                    0.0 MB
    all_summaries_step000000.pt                             0.0 MB
    phase0_benchmark_step000000.pt                          0.1 MB
    phase1_benchmark_step000000.pt                          0.1 MB
    phase2_benchmark_step000000.pt                          0.1 MB
    phase2_enc_pruning_step000000.pt                        0.0 MB
    phase3_benchmark_step000000.pt                          0.1 MB
    phase3_laco_done_step000000.pt                          0.0 MB
    phase4_benchmark_step000000.pt                          0.1 MB
    phase4_enc_pruning_step000000.pt                        0.0 MB
    phase5_benchmark_step000000.pt                          0.1 MB
    phase5_dec_pruning_step000000.pt                        0.0 MB
    phase6_benchmark_step000000.pt                          0.1 MB
    phase7_benchmark_step000000.pt                          0.1 MB
  GPU: Tesla T4  VRAM: 15.6 GB
=================================================================

✓ ALL SETUP CELLS COMPLETE — proceed to phases.
```

### Cell 26 (code, score=63)
```python
# ══════════════════════════════════════════════════════════════════════════════
# RAM-Efficient Parquet Streaming Dataset
# Loads audio on-demand, not during initialization
# RAM: ~4MB for 4000 samples (vs ~20GB with old approach)
# ══════════════════════════════════════════════════════════════════════════════

import pyarrow.parquet as pq

class ParquetStreamingDataset:
    """Memory-efficient dataset that streams from parquet files."""
    
    def __init__(self, parquet_cache_dir, src_lang, tgt_lang, split='train', 
                 max_samples_per_pair=500):
        self.cache_dir = pathlib.Path(parquet_cache_dir)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.split = split
        self.max_samples = max_samples_per_pair
        self.samples = []
        self._build_index()
    
    def _build_index(self):
        """Build lightweight index (metadata only, no audio)."""
        src_files = sorted(self.cache_dir.glob(f'{M4T_FLEURS_MAP.get(self.src_lang)}/{self.split}_*.parquet'))
        tgt_files = sorted(self.cache_dir.glob(f'{M4T_FLEURS_MAP.get(self.tgt_lang)}/{self.split}_*.parquet'))
        
        if not src_files or not tgt_files:
            print(f'  WARNING: No parquet files for {self.src_lang}/{self.tgt_lang}')
            return
        
        # Read only ID columns (fast, <1MB RAM)
        src_ids = []
        for f in src_files:
            df = pd.read_parquet(f, columns=['id'])
            src_ids.extend([(str(f), idx, row_id) for idx, row_id in enumerate(df['id'])])
        
        tgt_ids = []
        for f in tgt_files:
            df = pd.read_parquet(f, columns=['id', 'transcription'])
            df = df[df['transcription'].str.strip().str.len() > 0]
            tgt_ids.extend([(str(f), idx, row_id, trans) 
                           for idx, (row_id, trans) in enumerate(zip(df['id'], df['transcription']))])
        
        # Create lookup dicts
        src_lookup = {row_id: (f, idx) for f, idx, row_id in src_ids}
        tgt_lookup = {row_id: (f, idx, trans) for f, idx, row_id, trans in tgt_ids}
        
        # Find matching IDs
        common_ids = set(src_lookup.keys()) & set(tgt_lookup.keys())
        
        # Build sample index (metadata only)
        for sample_id in list(common_ids)[:self.max_samples]:
            src_file, src_idx = src_lookup[sample_id]
            tgt_file, tgt_idx, tgt_text = tgt_lookup[sample_id]
            
            self.samples.append({
                'id': f"{self.src_lang}2{self.tgt_lang}_{sample_id}",
                'src_lang': self.src_lang,
                'tgt_lang': self.tgt_lang,
                'ref': tgt_text,
                '_src_file': src_file,
                '_src_idx': src_idx,
            })
        
        print(f'  Indexed {len(self.samples)} samples from {self.src_lang}→{self.tgt_lang}')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get sample with audio loaded on-demand."""
        sample = self.samples[idx].copy()
        
        # Load audio only when accessed
        if '_src_file' in sample:
            audio = self._load_audio_from_parquet(
                sample['_src_file'], 
                sample['_src_idx']
            )
            sample['wav'] = audio
            del sample['_src_file']
            del sample['_src_idx']
        
        return sample
    
    def _load_audio_from_parquet(self, parquet_file, row_idx):
        """Load single audio sample from parquet file."""
        table = pq.read_table(parquet_file, columns=['audio'])
        audio_cell = table.to_pandas().iloc[row_idx]['audio']
        return _load_wav(audio_cell)


class MultilingualStreamingDataset:
    """Combines multiple language pairs into a single streaming dataset."""
    
    def __init__(self, parquet_cache_dir, lang_pairs, split='train', 
                 max_samples_per_pair=25):
        self.datasets = []
        
        for src_lang, tgt_lang in lang_pairs:
            ds = ParquetStreamingDataset(
                parquet_cache_dir, src_lang, tgt_lang, split, max_samples_per_pair
            )
            if len(ds) > 0:
                self.datasets.append(ds)
        
        # Build flat index
        self.index = []
        for ds_idx, ds in enumerate(self.datasets):
            for sample_idx in range(len(ds)):
                self.index.append((ds_idx, sample_idx))
        
        print(f'\n✓ Multilingual dataset ready: {len(self.index)} total samples')
        print(f'  RAM usage: ~{len(self.index) * 0.001:.1f} MB (metadata only)')
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """Get sample from appropriate sub-dataset."""
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self[i] for i in indices]
        ds_idx, sample_idx = self.index[idx]
        return self.datasets[ds_idx][sample_idx]
    
    def __iter__(self):
        """Allow iteration."""
        for i in range(len(self)):
            yield self[i]

print('✓ Streaming dataset classes ready.')
```
OUTPUT:
```text
✓ Streaming dataset classes ready.
```

### Cell 27 (code, score=6)
```python
# ── Load Multilingual Eval Samples: En→X and X→En (all 5 languages) ─────────
# PLAN.md Section 5: 5 languages — EN, BN, ZH, AR, HI
N_EVAL_PER_PAIR = 25
EVAL_LANG_PAIRS = [
    ('eng', 'ben'), ('ben', 'eng'),  # English ↔ Bengali
    ('eng', 'cmn'), ('cmn', 'eng'),  # English ↔ Mandarin
    ('eng', 'arb'), ('arb', 'eng'),  # English ↔ Arabic
    ('eng', 'hin'), ('hin', 'eng'),  # English ↔ Hindi
]
```

### Cell 28 (code, score=64)
```python
# ── Load Multilingual Eval Samples: En→X and X→En (all 5 languages) ──────────
# STREAMING VERSION: Only loads audio when accessed
# RAM: ~200KB for 200 samples (vs ~1GB with old approach)

print('Loading evaluation samples (streaming mode)...')
eval_samples = MultilingualStreamingDataset(
    parquet_cache_dir=LOCAL_PARQUET_CACHE,
    lang_pairs=EVAL_LANG_PAIRS,
    split='test',
    max_samples_per_pair=N_EVAL_PER_PAIR
)

print(f'\n✓ Loaded {len(eval_samples)} multilingual eval samples')
print(f'  Language pairs: {len(EVAL_LANG_PAIRS)}')
print(f'  RAM usage: ~{len(eval_samples) * 0.001:.1f} MB (metadata only)')

# Test: Load one sample to verify it works
test_sample = eval_samples[26]
print(f'\n✓ Test sample loaded:')
print(f'  ID: {test_sample["id"]}')
print(f'  Audio shape: {test_sample["wav"].shape}')
print(f'  Reference: {test_sample["ref"][:50]}...')

play(test_sample["wav"], 16000, label='hello.wav')
```
OUTPUT:
```text
Loading evaluation samples (streaming mode)...
  Indexed 25 samples from eng→ben
  Indexed 25 samples from ben→eng
  Indexed 25 samples from eng→cmn
  Indexed 25 samples from cmn→eng
  Indexed 25 samples from eng→arb
  Indexed 25 samples from arb→eng
  Indexed 25 samples from eng→hin
  Indexed 25 samples from hin→eng

✓ Multilingual dataset ready: 200 total samples
  RAM usage: ~0.2 MB (metadata only)

✓ Loaded 200 multilingual eval samples
  Language pairs: 8
  RAM usage: ~0.2 MB (metadata only)

✓ Test sample loaded:
  ID: ben2eng_1661
  Audio shape: (187200,)
  Reference: he did not set a figure for the cuts saying they w...
  hello.wav  (11.7s | sr=16000)

<IPython.lib.display.Audio object>
```

### Cell 29 (code, score=3)
```python
# ── Load Multilingual Training Samples: En→X and X→En (all 5 languages) ─────
N_TRAIN_PER_PAIR = 1200  # 500 samples per direction = 4000 total
```

### Cell 30 (code, score=66)
```python
# ── Load Multilingual Training Samples: En→X and X→En (all 5 languages) ──────
# STREAMING VERSION: Only loads audio when accessed
# RAM: ~4MB for 4000 samples (vs ~20GB with old approach)

print('Loading training samples (streaming mode)...')
ft_samples = MultilingualStreamingDataset(
    parquet_cache_dir=LOCAL_PARQUET_CACHE,
    lang_pairs=EVAL_LANG_PAIRS,
    split='train',
    max_samples_per_pair=N_TRAIN_PER_PAIR
)

print(f'\n✓ Loaded {len(ft_samples)} multilingual training samples')
print(f'  Language pairs: {len(EVAL_LANG_PAIRS)}')
print(f'  RAM usage: ~{len(ft_samples) * 0.001:.1f} MB (metadata only)')
print(f'  RAM saved: ~{len(ft_samples) * 5:.0f} MB (would be with old approach)')

# Summary by language pair
print('\nSamples per language pair:')
pair_counts = {}
for i in range(len(ft_samples)):
    sample_meta = ft_samples.datasets[ft_samples.index[i][0]].samples[ft_samples.index[i][1]]
    pair = f"{sample_meta['src_lang']}→{sample_meta['tgt_lang']}"
    pair_counts[pair] = pair_counts.get(pair, 0) + 1

for pair, count in sorted(pair_counts.items()):
    print(f'  {pair}: {count}')
```
OUTPUT:
```text
Loading training samples (streaming mode)...
  Indexed 1200 samples from eng→ben
  Indexed 1200 samples from ben→eng
  Indexed 1200 samples from eng→cmn
  Indexed 1200 samples from cmn→eng
  Indexed 1200 samples from eng→arb
  Indexed 1200 samples from arb→eng
  Indexed 1200 samples from eng→hin
  Indexed 1200 samples from hin→eng

✓ Multilingual dataset ready: 9600 total samples
  RAM usage: ~9.6 MB (metadata only)

✓ Loaded 9600 multilingual training samples
  Language pairs: 8
  RAM usage: ~9.6 MB (metadata only)
  RAM saved: ~48000 MB (would be with old approach)

Samples per language pair:
  arb→eng: 1200
  ben→eng: 1200
  cmn→eng: 1200
  eng→arb: 1200
  eng→ben: 1200
  eng→cmn: 1200
  eng→hin: 1200
  hin→eng: 1200
```

### Cell 31 (code, score=35)
```python
# ── Multilingual eval samples now integrated into eval_samples ──────────────
# All 5 languages (EN, BN, HI, ZH, AR) with bidirectional pairs are loaded above
print(f'Multilingual eval ready: {len(eval_samples)} samples across {len(EVAL_LANG_PAIRS)} pairs')
print(f'Language pairs: {EVAL_LANG_PAIRS}')

print(f'\n✓ Loaded {len(ft_samples)} multilingual training samples across {len(EVAL_LANG_PAIRS)} pairs')
```
OUTPUT:
```text
Multilingual eval ready: 200 samples across 8 pairs
Language pairs: [('eng', 'ben'), ('ben', 'eng'), ('eng', 'cmn'), ('cmn', 'eng'), ('eng', 'arb'), ('arb', 'eng'), ('eng', 'hin'), ('hin', 'eng')]

✓ Loaded 9600 multilingual training samples across 8 pairs
```

### Cell 32 (markdown, score=7)
```markdown
---
## Phase 0: V1 Baseline Capture
Load V1 pipeline (1039M from previous work) and run ASR-ChrF benchmark across 5 languages.
These scores become the quality ceiling for the textless model.
```

### Cell 33 (code, score=9)
```python
# model_v1, processor = load_base_model()

# print_model_breakdown(model_v1, 'V1 Baseline Model')
# gpu_mem()
```

### Cell 34 (code, score=10)
```python
# # QUICK FIX: Add this cell right before running the benchmark in Phase 0

# # Fix the processor None issue
# if processor is None:
#     print("WARNING: processor is None, reloading...")
#     from transformers import SeamlessM4TProcessor
#     MODEL_NAME = 'facebook/seamless-m4t-v2-large'
#     processor = SeamlessM4TProcessor.from_pretrained(MODEL_NAME)
#     print("Processor reloaded successfully")
# else:
#     print("Processor is already loaded")

# # Verify processor is working
# print(f"Processor type: {type(processor)}")
# print(f"Processor model: {getattr(processor, 'model_name', 'Unknown')}")
```

### Cell 35 (code, score=35)
```python
# # ── Test Whisper ASR: Bengali → English Translation ──────────────────────────
# # This cell tests the complete pipeline: BN audio → S2ST → EN audio → Whisper ASR

# print('='*70)
# print('  WHISPER ASR TEST: Bengali → English Translation')
# print('='*70)

# # Load a Bengali test sample
# test_sample = eval_samples[0]  # First Bengali sample
# print(f'\n📥 Input:')
# print(f'  Language: Bengali → English')
# print(f'  Duration: {len(test_sample["wav"])/16000:.1f}s')
# print(f'  Reference (EN): {test_sample["ref"][:100]}...')

# # Run S2ST translation (Bengali → English)
# print(f'\n🔄 Running S2ST translation...')
# try:
#     _, wav_out = run_s2st(model_v1, test_sample['wav'], tgt_lang='eng')
#     print(f'  ✓ Translation complete')
#     print(f'  Output duration: {len(wav_out)/16000:.1f}s')
# except Exception as e:
#     print(f'  ✗ Translation failed: {e}')
#     import traceback
#     traceback.print_exc()

# # Test Whisper ASR on English output
# print(f'\n🎤 Testing Whisper ASR (English)...')
# try:
#     # Ensure Whisper is loaded
#     _ensure_whisper_loaded()
    
#     # Transcribe using Whisper
#     hyp = asr_transcribe_whisper(wav_out, lang='en', sr=16000)
    
#     print(f'  ✓ Whisper transcription complete')
#     print(f'\n📝 Results:')
#     print(f'  Reference: {test_sample["ref"][:150]}')
#     print(f'  Whisper:   {hyp[:150]}')
    
#     # Compute metrics
#     from sacrebleu.metrics import BLEU, CHRF
#     bleu_metric = BLEU(effective_order=True)
#     chrf_metric = CHRF()
    
#     bleu_score = bleu_metric.sentence_score(hyp, [test_sample['ref']]).score
#     chrf_score = chrf_metric.sentence_score(hyp, [test_sample['ref']]).score
    
#     print(f'\n📊 Metrics:')
#     print(f'  ASR-BLEU: {bleu_score:.2f}')
#     print(f'  ASR-ChrF: {chrf_score:.2f}')
    
#     # Play audio (optional)
#     print(f'\n🔊 Audio samples:')
#     play(test_sample['wav'], 16000, 'Input (Bengali)')
#     play(wav_out, 16000, 'Output (English, voice-cloned)')
    
#     print(f'\n✅ Whisper ASR test PASSED')
    
# except Exception as e:
#     print(f'  ✗ Whisper ASR failed: {e}')
#     import traceback
#     traceback.print_exc()
#     print(f'\n❌ Whisper ASR test FAILED')

# print('='*70)
```

### Cell 36 (code, score=37)
```python
# print(len(eval_samples))
# sasa = eval_samples[52]
# print(sasa['tgt_lang'])
# _, wav_out = run_s2st(model_v1, sasa['wav'], tgt_lang=sasa['tgt_lang'])

# pred = asr_transcribe(wav_out, 'cmn', sr=16000)

# ref = sasa['ref']
# hyp = pred
# if sasa['tgt_lang'] == 'cmn':
#     print("inside")
#     ref_clean = ref.replace(" ", "")
#     hyp_clean = hyp.replace(" ", "")

#     # BLEU (tokenized)
#     ref_bleu = zh_tokenize(ref_clean)
#     hyp_bleu = zh_tokenize(hyp_clean)

#     # chrF (raw)
#     ref_chrf = ref_clean
#     hyp_chrf = hyp_clean

#     bleu = compute_bleu(hyp_bleu, ref_bleu)
#     chrf = compute_chrf(hyp_chrf, ref_chrf) 

# print(f'ref: {ref}')
# print(f'hyp: {hyp}')
# print(f'ref_chrf: {ref_chrf}')
# print(f'hyp_chrf: {hyp_chrf}')
# print("=======")
# print(f'ref_bleu: {ref_bleu}')
# print(f'hyp_bleu: {hyp_bleu}')

# print(f'bleu: {bleu}')
# print(f'chrf: {chrf}')

# nbleu = compute_bleu(hyp, ref)
# nchrf = compute_chrf(hyp, ref) 
# print(f'nBleu: {nbleu}')
# print(f'nchrf: {nchrf}')
```

### Cell 37 (code, score=139)
```python
p0_bench = load_latest_checkpoint('phase0_benchmark')
if p0_bench and p0_bench.get('summary', {}).get('avg_bleu', 0) > 0:
    p0_results = p0_bench['results']
    p0_summary = p0_bench['summary']
    p0_detailed = p0_bench.get('detailed_summary')
    print('Loaded Phase 0 benchmark results from checkpoint.')
    
    # Recompute detailed if missing
    if not p0_detailed:
        p0_detailed = compute_detailed_summary(p0_results, 'P0_V1_Baseline', p0_summary['params_M'])
else:
    p0_results, p0_summary = run_benchmark_asr(
        model_v1, eval_samples, label='P0_V1_Baseline', save_n=4)
    p0_detailed = compute_detailed_summary(p0_results, 'P0_V1_Baseline', p0_summary['params_M'])
    save_checkpoint(dict(
        results=p0_results, 
        summary=p0_summary,
        detailed_summary=p0_detailed
    ), 'phase0_benchmark', 0)

store_summary(p0_summary)
store_detailed_summary(p0_detailed)
print_detailed_summary_table('P0_V1_Baseline')
plot_phase_comparison()
plot_detailed_phase_comparison()
```
OUTPUT:
```text
[ckpt] Loaded phase0_benchmark_step000000.pt
Loaded Phase 0 benchmark results from checkpoint.
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P0_V1_Baseline (1 total)
[ckpt] Saved all_detailed_summaries_step000000.pt (0.0 MB)
[detailed] Stored P0_V1_Baseline

================================================================================
  P0_V1_Baseline - 1805.5M params
================================================================================
Overall: ChrF=46.49±16.02  BLEU=15.88  RTF=0.2455

Per-Pair (8 pairs):
  Pair               N     ChrF     BLEU      RTF
  arb→eng           25    52.40    18.44   0.2802
  ben→eng           25    49.63    15.44   0.2119
  cmn→eng           25    49.84    15.53   0.2471
  eng→arb           25    46.62    14.29   0.2336
  eng→ben           25    47.95    11.09   0.3270
  eng→cmn           25    18.34     9.95   0.1970
  eng→hin           25    54.52    26.27   0.2434
  hin→eng           25    52.61    16.02   0.2238

By Source Language:
     ARB: ChrF= 52.40  BLEU= 18.44  (n=25)
     BEN: ChrF= 49.63  BLEU= 15.44  (n=25)
     CMN: ChrF= 49.84  BLEU= 15.53  (n=25)
     ENG: ChrF= 41.86  BLEU= 15.40  (n=100)
     HIN: ChrF= 52.61  BLEU= 16.02  (n=25)

By Target Language:
     ARB: ChrF= 46.62  BLEU= 14.29  (n=25)
     BEN: ChrF= 47.95  BLEU= 11.09  (n=25)
     CMN: ChrF= 18.34  BLEU=  9.95  (n=25)
     ENG: ChrF= 51.12  BLEU= 16.36  (n=100)
     HIN: ChrF= 54.52  BLEU= 26.27  (n=25)
================================================================================
Plotting 1 phases: ['P0_V1_Baseline']
[fig] Saved phase_comparison.png

<Figure size 1920x1200 with 4 Axes>
[image/png output omitted]
Plotting detailed comparison for 1 phases: ['P0_V1_Baseline']

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_01_overall_quality.png  [Overall Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_02_chrf_by_pair.png  [ChrF by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_03_bleu_by_pair.png  [BLEU by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_04_src_lang_trends.png  [Source Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_05_tgt_lang_trends.png  [Target Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_06_size_vs_quality.png  [Size vs Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_07_inference_rtf.png  [Inference Speed RTF]

<Figure size 1800x540 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_08_summary_table.png  [Summary Table]

✅ All 8 figures saved individually:
   📄 detailed_comparison_01_overall_quality.png
   📄 detailed_comparison_02_chrf_by_pair.png
   📄 detailed_comparison_03_bleu_by_pair.png
   📄 detailed_comparison_04_src_lang_trends.png
   📄 detailed_comparison_05_tgt_lang_trends.png
   📄 detailed_comparison_06_size_vs_quality.png
   📄 detailed_comparison_07_inference_rtf.png
   📄 detailed_comparison_08_summary_table.png
```

### Cell 38 (code, score=8)
```python
# # ── Free V1 from VRAM before next phase ──────────────────────────────────────
# del model_v1
# gc.collect(); torch.cuda.empty_cache()
# print('V1 unloaded from VRAM.')
# gpu_mem()
```

### Cell 39 (markdown, score=8)
```markdown
---
## Phase 1: Vocabulary Pruning — 5 Languages
Extend V1 vocab trim to EN, BN, ZH, AR, HI. Saves ~215M params, zero quality loss.
Paper: Asahi et al. (EMNLP 2023) — vocabulary trimming methodology.
```

### Cell 40 (code, score=63)
```python
# def identify_used_tokens(proc, target_lang_codes, n_corpus=5000):
#     """Scan FLEURS corpora for all 5 target languages and collect used token IDs."""
#     from datasets import load_dataset
#     # Extended fleurs codes for all 5 languages
#     fleurs_codes = dict(
#         eng='en_us', ben='bn_in', cmn='cmn_hans_cn',
#         arb='ar_eg', hin='hi_in',
#         # Keep these for safety (may appear in multilingual text)
#         fra='fr_fr', deu='de_de',
#     )
#     BASE = 'hf://datasets/google/fleurs@refs%2Fconvert%2Fparquet'
#     used = set()
#     tok = proc.tokenizer
#     # Always keep special tokens and language tokens
#     if hasattr(tok, 'all_special_ids'): used.update(tok.all_special_ids)
#     for tid in range(len(tok)):
#         t = tok.convert_ids_to_tokens(tid)
#         if t and t.startswith('__') and t.endswith('__'): used.add(tid)
#     for lang, fc in fleurs_codes.items():
#         if lang not in target_lang_codes: continue
#         print(f'  Scanning {lang} ({fc})...')
#         try:
#             ds = load_dataset('parquet',
#                               data_files={'train': f'{BASE}/{fc}/train/*.parquet'},
#                               split='train')
#             for i, ex in enumerate(ds):
#                 if i >= n_corpus: break
#                 text = ex.get('transcription', '')
#                 if text: used.update(tok.encode(text, add_special_tokens=False))
#         except Exception as e:
#             print(f'    Warning: {lang}: {e}')
#     print(f'  Unique tokens: {len(used)} / {len(tok)}')
#     return sorted(used)

# print('Vocabulary pruning helpers ready.')



# def trim_vocabulary(mdl, proc, keep_ids):
#     """Trim NLLB vocabulary to only the kept token IDs. Battle-tested from seamless-cse465v5."""
#     keep_t = torch.tensor(keep_ids, dtype=torch.long)
#     old_v = mdl.config.vocab_size
#     new_v = len(keep_ids)
#     hidden = mdl.config.hidden_size
#     print(f'  Vocabulary: {old_v} -> {new_v} ({new_v/old_v*100:.1f}%)')
#     old_shared = mdl.shared
#     dev = old_shared.weight.device
#     dtype = old_shared.weight.dtype
#     keep_t_dev = keep_t.to(dev)
#     old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
#     old_pad = old_shared.padding_idx
#     new_pad = old_to_new.get(old_pad) if old_pad is not None else None
#     embed_scale = getattr(mdl.text_decoder.embed_tokens, 'embed_scale', 1.0)
#     print(f'  text_decoder.embed_tokens.embed_scale = {embed_scale}')
#     # Create trimmed shared embedding
#     new_shared = nn.Embedding(new_v, hidden, padding_idx=new_pad)
#     new_shared.weight.data = old_shared.weight.data[keep_t_dev].clone()
#     mdl.shared = new_shared.to(dev)
#     # Decoder embed_tokens (must preserve embed_scale)
#     from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ScaledWordEmbedding
#     new_embed = SeamlessM4Tv2ScaledWordEmbedding(new_v, hidden, padding_idx=new_pad, embed_scale=embed_scale)
#     new_embed.weight = mdl.shared.weight
#     mdl.text_decoder.embed_tokens = new_embed
#     print(f'  text_decoder.embed_tokens: tied to shared, embed_scale={embed_scale}')
#     # LM head
#     old_lm = mdl.lm_head
#     new_lm = nn.Linear(hidden, new_v, bias=False)
#     new_lm.weight = mdl.shared.weight
#     mdl.lm_head = new_lm
#     print(f'  lm_head: tied to shared [20425, 1024]' if new_v == 20425 else f'  lm_head: tied to shared [{new_v}, {hidden}]')
#     # Update config
#     mdl.config.vocab_size = new_v
#     # Remap generation config id_to_text
#     gen_cfg = mdl.generation_config
#     if hasattr(gen_cfg, 'id_to_text') and gen_cfg.id_to_text:
#         new_map = {}
#         for key_str, text_val in gen_cfg.id_to_text.items():
#             old_id = int(key_str)
#             if old_id in old_to_new:
#                 new_map[str(old_to_new[old_id])] = text_val
#         gen_cfg.id_to_text = new_map
#         print(f'  id_to_text: {len(gen_cfg.id_to_text)} entries')
#     # Remap lang code IDs
#     for attr in ['text_decoder_lang_to_code_id', 'id_to_lang']:
#         if hasattr(gen_cfg, attr):
#             old_map = getattr(gen_cfg, attr)
#             if isinstance(old_map, dict):
#                 new_m = {}
#                 for k, v in old_map.items():
#                     if isinstance(v, int):
#                         new_v_id = old_to_new.get(v, v)
#                         new_m[k] = new_v_id
#                     else:
#                         new_m[k] = v
#                 setattr(gen_cfg, attr, new_m)
#     saved_M = (old_v - new_v) * hidden / 1e6
#     print(f'  Done: ~{saved_M:.0f}M shared-embedding params removed (lm_head tied, not double-counted)')
#     mdl._vocab_remap_to_old = keep_t.cpu()
#     return mdl

# print('trim_vocabulary() ready.')
```

### Cell 41 (code, score=49)
```python
# # ── Run Phase 1 — try Drive, else trim from V1 ───────────────────────────────
# try:
#     model_p1, processor = load_model_from_drive('phase1_vocab_5lang')
#     p1_ck = load_latest_checkpoint('phase1_vocab')
#     if p1_ck and 'keep_ids' in p1_ck:
#         keep_ids = p1_ck['keep_ids']
#         model_p1._vocab_remap_to_old = torch.tensor(keep_ids, dtype=torch.long)
#         print(f'  Restored vocab remap ({len(keep_ids)} tokens)')
#     print('Loaded Phase 1 from Drive.')
# except Exception as e:
#     print(f'Load failed ({e}), running vocab trim...')
#     # Reload V1 fresh
#     model_v1_fresh, processor = load_base_model()
#     TARGET_5LANGS = ['eng', 'ben', 'cmn', 'arb', 'hin']
#     keep_ids = identify_used_tokens(processor, TARGET_5LANGS, n_corpus=3000)
#     pre = count_params(model_v1_fresh)
#     model_p1 = trim_vocabulary(model_v1_fresh, processor, keep_ids)
#     post = count_params(model_p1)
#     print(f'  Params: {pre:.1f}M -> {post:.1f}M (saved {pre-post:.1f}M)')
#     save_checkpoint(dict(keep_ids=keep_ids, pre=pre, post=post), 'phase1_vocab', 0)
#     save_model_to_drive(model_p1, processor, 'phase1_vocab_5lang')

# print_model_breakdown(model_p1, 'After Phase 1: Vocab Trimmed (5L)')
```

### Cell 42 (code, score=122)
```python
p1_bench = load_latest_checkpoint('phase1_benchmark')
if p1_bench and p1_bench.get('summary', {}).get('avg_bleu', 0) > 0:
    p1_results = p1_bench['results']
    p1_summary = p1_bench['summary']
    p1_detailed = p1_bench.get('detailed_summary')
    if not p1_detailed:
        p1_detailed = compute_detailed_summary(p1_results, 'P1_Vocab5L', p1_summary['params_M'])
else:
    p1_results, p1_summary = run_benchmark_asr(
        model_p1, eval_samples, label='P1_Vocab5L', save_n=4)
    p1_detailed = compute_detailed_summary(p1_results, 'P1_Vocab5L', p1_summary['params_M'])
    save_checkpoint(dict(
        results=p1_results, 
        summary=p1_summary,
        detailed_summary=p1_detailed
    ), 'phase1_benchmark', 0)

store_summary(p1_summary)
store_detailed_summary(p1_detailed)
print_detailed_summary_table('P1_Vocab5L')
plot_phase_comparison()
plot_detailed_phase_comparison()
```
OUTPUT:
```text
[ckpt] Loaded phase1_benchmark_step000000.pt
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P1_Vocab5L (2 total)
[ckpt] Saved all_detailed_summaries_step000000.pt (0.0 MB)
[detailed] Stored P1_Vocab5L

================================================================================
  P1_Vocab5L - 1566.6M params
================================================================================
Overall: ChrF=41.74±18.87  BLEU=13.65  RTF=0.2435

Per-Pair (8 pairs):
  Pair               N     ChrF     BLEU      RTF
  arb→eng           25    47.26    17.35   0.2728
  ben→eng           25    45.22    11.78   0.2003
  cmn→eng           25    45.06    14.26   0.2518
  eng→arb           25    45.15    13.48   0.2150
  eng→ben           25    47.20    10.96   0.3127
  eng→cmn           25     6.13     3.26   0.2671
  eng→hin           25    54.07    25.37   0.2199
  hin→eng           25    43.84    12.70   0.2082

By Source Language:
     ARB: ChrF= 47.26  BLEU= 17.35  (n=25)
     BEN: ChrF= 45.22  BLEU= 11.78  (n=25)
     CMN: ChrF= 45.06  BLEU= 14.26  (n=25)
     ENG: ChrF= 38.14  BLEU= 13.27  (n=100)
     HIN: ChrF= 43.84  BLEU= 12.70  (n=25)

By Target Language:
     ARB: ChrF= 45.15  BLEU= 13.48  (n=25)
     BEN: ChrF= 47.20  BLEU= 10.96  (n=25)
     CMN: ChrF=  6.13  BLEU=  3.26  (n=25)
     ENG: ChrF= 45.35  BLEU= 14.02  (n=100)
     HIN: ChrF= 54.07  BLEU= 25.37  (n=25)
================================================================================
Plotting 2 phases: ['P0_V1_Baseline', 'P1_Vocab5L']
[fig] Saved phase_comparison.png

<Figure size 1920x1200 with 4 Axes>
[image/png output omitted]
Plotting detailed comparison for 2 phases: ['P0_V1_Baseline', 'P1_Vocab5L']

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_01_overall_quality.png  [Overall Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_02_chrf_by_pair.png  [ChrF by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_03_bleu_by_pair.png  [BLEU by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_04_src_lang_trends.png  [Source Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_05_tgt_lang_trends.png  [Target Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_06_size_vs_quality.png  [Size vs Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_07_inference_rtf.png  [Inference Speed RTF]

<Figure size 1800x540 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_08_summary_table.png  [Summary Table]

✅ All 8 figures saved individually:
   📄 detailed_comparison_01_overall_quality.png
   📄 detailed_comparison_02_chrf_by_pair.png
   📄 detailed_comparison_03_bleu_by_pair.png
   📄 detailed_comparison_04_src_lang_trends.png
   📄 detailed_comparison_05_tgt_lang_trends.png
   📄 detailed_comparison_06_size_vs_quality.png
   📄 detailed_comparison_07_inference_rtf.png
   📄 detailed_comparison_08_summary_table.png
```

### Cell 43 (markdown, score=7)
```markdown
---
## Phase 2: Speech Encoder Moderate Pruning (24 → 16 layers)
Target: remove 8 of 24 layers (~33%). Method: BI-guided iterative greedy (same as v5 Phase 4).
Conservative — encoder layers are language-neutral, no cliff expected.
Papers: ShortGPT (ACL 2025) Block Influence · Moslem IWSLT 2025 iterative greedy.
```

### Cell 44 (code, score=9)
```python
# model_p1, processor = load_model_from_drive('phase1_vocab_5lang')
# print_model_breakdown(model_p1, 'After Phase 1: Vocab Trimmed (5L)')
```

### Cell 45 (code, score=87)
```python
# ── get_speech_encoder_layers, compute_block_influence, iterative_enc_prune
# ── verbatim from v5 Cells 54 + 58 ──────────────────────────────────────────

def get_speech_encoder_layers(mdl):
    enc = mdl.speech_encoder
    if hasattr(enc,'layers') and isinstance(enc.layers,nn.ModuleList) and len(enc.layers)>0:
        return enc, 'layers'
    if hasattr(enc,'encoder') and hasattr(enc.encoder,'layers') and len(enc.encoder.layers)>0:
        return enc.encoder, 'layers'
    for child_name, child in enc.named_children():
        if hasattr(child,'layers') and isinstance(child.layers,nn.ModuleList) and len(child.layers)>0:
            print(f'  Found layers at speech_encoder.{child_name}.layers')
            return child, 'layers'
    raise RuntimeError(f'Cannot find speech encoder layers. children: {[n for n,_ in enc.named_children()]}')


def compute_block_influence(mdl, samples, max_n=50):
    """ShortGPT (ACL 2025) BI: 1 - cos(layer_input, layer_output)."""    
    parent, la = get_speech_encoder_layers(mdl)
    layers = getattr(parent, la); n = len(layers)
    bi = {i: [] for i in range(n)}
    hooks = []
    for i in range(n):
        def make_hook(idx):
            def hook(mod, inp, out):
                x = inp[0]
                if x is None or not isinstance(x, torch.Tensor): return
                y = out[0] if isinstance(out,tuple) else out
                if y is None or not isinstance(y, torch.Tensor): return
                x = x.detach().float().reshape(-1,x.shape[-1])
                y = y.detach().to(x.device).float().reshape(-1,y.shape[-1])
                bi[idx].append(1.0 - F.cosine_similarity(x,y,dim=-1).mean().item())
            return hook
        hooks.append(layers[i].register_forward_hook(make_hook(i)))
    mdl.eval()
    dev = next(mdl.speech_encoder.parameters()).device
    ok = 0
    for idx, s in enumerate(samples[:max_n]):
        if idx % 10 == 0: print(f'  Calibrating BI {idx}/{min(max_n,len(samples))}...')
        try:
            inputs = processor(audio=s['wav'], sampling_rate=16000, return_tensors='pt')
            feats  = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad(): mdl.speech_encoder(**feats)
            ok += 1
        except Exception as e: print(f'  Sample {idx} failed: {e}')
    for h in hooks: h.remove()
    scores = {i: float(np.mean(v)) if v else 0.0 for i,v in bi.items()}
    print(f'  Calibrated {ok}/{min(max_n,len(samples))} samples.')
    ranked = sorted(scores.items(), key=lambda x: x[1])
    print('  BI ranking (low=redundant):')
    for rank, (li, bv) in enumerate(ranked):
        print(f'    Rank{rank+1:>2}  L{li:>2}  BI={bv:.4f}')
    return scores


def _get_protected_enc(n_total):
    return {0, n_total//2, n_total-1}


def iterative_enc_prune(mdl, samples, n_remove, tgt_lang='ben', max_eval=16,
                        ckpt_name='phase2_enc_pruning', bi_scores=None,
                        bi_candidate_ratio=0.5, protected=None):
    """BI-guided iterative greedy encoder pruning — verbatim from v5 Cell 58."""    
    parent, la = get_speech_encoder_layers(mdl)
    current    = list(getattr(parent, la))
    orig_idx   = list(range(len(current)))
    n_total    = len(current)
    removed, log = [], []
    if protected is None: protected = _get_protected_enc(n_total)
    print(f'  Protected layers (first/mid/last): {sorted(protected)}')
    partial = load_latest_checkpoint(ckpt_name)
    if partial and partial.get('removed'):
        removed = list(partial['removed']); log = partial.get('log',[])
        for r in removed:
            if r in orig_idx:
                pos = orig_idx.index(r); current.pop(pos); orig_idx.pop(pos)
        setattr(parent, la, nn.ModuleList(current))
        print(f'  Resuming: removed {removed}, {len(current)} layers remain')
    # Use multilingual samples for baseline
    baseline = quick_eval_chrf(mdl, samples, max_samples=max_eval)
    print(f'  Baseline ChrF: {baseline:.2f}')
    for it in range(len(removed), n_remove):
        eligible = [pos for pos in range(len(current)) if orig_idx[pos] not in protected]
        if bi_scores and len(eligible) > 2:
            by_bi = sorted(eligible, key=lambda pos: bi_scores.get(orig_idx[pos], float('inf')))
            n_cands = max(2, int(len(by_bi)*bi_candidate_ratio))
            cands   = by_bi[:n_cands]
            print(f'\n  Iter {it+1}/{n_remove} | BI pre-filter: {len(cands)}/{len(eligible)} cands')
        else:
            cands = eligible
            print(f'\n  Iter {it+1}/{n_remove} | all {len(cands)} eligible (no BI)')
        if not cands: print('  No candidates left, stopping.'); break
        scores = {}
        for pos in cands:
            temp = current[:pos]+current[pos+1:]
            setattr(parent, la, nn.ModuleList(temp))
            sc = quick_eval_chrf(mdl, samples, max_samples=max_eval)
            bi_note = f'  BI={bi_scores.get(orig_idx[pos],0):.4f}' if bi_scores else ''
            print(f'    Remove L{orig_idx[pos]:>2} -> ChrF={sc:.2f}{bi_note}')
            scores[pos] = (orig_idx[pos], sc)
        setattr(parent, la, nn.ModuleList(current))
        best_pos = max(scores, key=lambda k: scores[k][1])
        best_orig, best_sc = scores[best_pos]
        current.pop(best_pos); orig_idx.pop(best_pos)
        setattr(parent, la, nn.ModuleList(current))
        removed.append(best_orig)
        log.append(dict(iter=it+1, removed=best_orig, chrf=best_sc,
                        remaining=len(current),
                        bi_score=bi_scores.get(best_orig) if bi_scores else None))
        print(f'  -> Removed L{best_orig} ChrF={best_sc:.2f} ({len(current)} remain)')
        if bi_scores and best_orig in bi_scores: del bi_scores[best_orig]
        save_checkpoint(dict(removed=removed, log=log, bi_scores=bi_scores or {}),
                        ckpt_name, step=0)
        torch.cuda.empty_cache()
    return removed, log

print('Phase 2 helpers ready (BI-guided iterative enc prune).')
```
OUTPUT:
```text
Phase 2 helpers ready (BI-guided iterative enc prune).
```

### Cell 46 (code, score=66)
```python
# # ── RUN Phase 2 ───────────────────────────────────────────────────────────────
# N_ENC_REMOVE      = 8
# ENC_BI_CAND_RATIO = 0.5   # evaluate bottom 50% by BI — halves ChrF calls

# p2_ckpt    = load_latest_checkpoint('phase2_enc_pruning')
# p2_complete= p2_ckpt and len(p2_ckpt.get('removed',[])) >= N_ENC_REMOVE

# if p2_complete:
#     removed_enc = p2_ckpt['removed']; bi_scores = p2_ckpt.get('bi_scores',{}); p2_log = p2_ckpt['log']
#     print(f'Phase 2 complete: removed {removed_enc}')
#     try:
#         model_p2, processor = load_model_from_drive('phase2_enc_16L')
#     except:
#         print('  Rebuilding from checkpoint + model_p1...')
#         model_p2 = model_p1
#         parent, la = get_speech_encoder_layers(model_p2)
#         cur = list(getattr(parent, la))
#         keep = [i for i in range(len(cur)) if i not in removed_enc]
#         setattr(parent, la, nn.ModuleList([cur[i] for i in keep]))
#         sync_model_config(model_p2)
#         save_model_to_drive(model_p2, processor, 'phase2_enc_16L')
# else:
#     done = len(p2_ckpt['removed']) if p2_ckpt else 0
#     print(f'{"Resuming" if done else "Running"} Phase 2: enc pruning ({done}/{N_ENC_REMOVE} done)...')
#     model_p2 = _consolidate_to_single_gpu(model_p1)
#     sanity = quick_eval_chrf(model_p2, eval_samples, 10)
#     print(f'  Sanity ChrF={sanity:.2f}  (abort if < 10)')
#     assert sanity > 10, f'Sanity too low: {sanity:.2f}'
#     if not (p2_ckpt and p2_ckpt.get('bi_scores')):
#         print('Computing Block Influence scores...')
#         bi_scores = compute_block_influence(model_p2, eval_samples, max_n=50)
#         save_checkpoint(dict(removed=[], log=[], bi_scores=bi_scores), 'phase2_enc_pruning', 0)
#     else:
#         bi_scores = p2_ckpt['bi_scores']
#         print(f'  BI scores loaded ({len(bi_scores)} layers)')
#     parent_tmp, la_tmp = get_speech_encoder_layers(model_p2)
#     n_enc = len(getattr(parent_tmp, la_tmp))
#     enc_protected = _get_protected_enc(n_enc)
#     removed_enc, p2_log = iterative_enc_prune(
#         model_p2, eval_samples, N_ENC_REMOVE, max_eval=16,
#         ckpt_name='phase2_enc_pruning', bi_scores=bi_scores,
#         bi_candidate_ratio=ENC_BI_CAND_RATIO, protected=enc_protected)
#     sync_model_config(model_p2)
#     save_checkpoint(dict(removed=removed_enc, log=p2_log, bi_scores=bi_scores),
#                     'phase2_enc_pruning', 0)
#     save_model_to_drive(model_p2, processor, 'phase2_enc_16L')

# print(f'Encoder layers removed: {removed_enc}')
# print_model_breakdown(model_p2, 'After Phase 2: Enc 16L')
```

### Cell 47 (code, score=4)
```python
# !rm -rf checkpoints/phase2_enc_pruning_step000000.pt
```

### Cell 48 (code, score=36)
```python
!ls checkpoints
```
OUTPUT:
```text
all_detailed_summaries_step000000.pt  phase3_laco_done_step000000.pt
all_summaries_step000000.pt	      phase4_benchmark_step000000.pt
phase0_benchmark_step000000.pt	      phase4_enc_pruning_step000000.pt
phase1_benchmark_step000000.pt	      phase5_benchmark_step000000.pt
phase2_benchmark_step000000.pt	      phase5_dec_pruning_step000000.pt
phase2_enc_pruning_step000000.pt      phase6_benchmark_step000000.pt
phase3_benchmark_step000000.pt	      phase7_benchmark_step000000.pt
```

### Cell 49 (code, score=121)
```python
p2_bench = load_latest_checkpoint('phase2_benchmark')
if p2_bench:
    p2_results = p2_bench['results']
    p2_summary = p2_bench['summary']
    p2_detailed = p2_bench.get('detailed_summary')
    if not p2_detailed:
        p2_detailed = compute_detailed_summary(p2_results, 'P2_Enc16L', p2_summary['params_M'])
else:
    p2_results, p2_summary = run_benchmark_asr(model_p2, eval_samples, 'P2_Enc16L', save_n=4)
    p2_detailed = compute_detailed_summary(p2_results, 'P2_Enc16L', p2_summary['params_M'])
    save_checkpoint(dict(
        results=p2_results, 
        summary=p2_summary,
        detailed_summary=p2_detailed
    ), 'phase2_benchmark', 0)

store_summary(p2_summary)
store_detailed_summary(p2_detailed)
print_detailed_summary_table('P2_Enc16L')
plot_phase_comparison()
plot_detailed_phase_comparison()
```
OUTPUT:
```text
[ckpt] Loaded phase2_benchmark_step000000.pt
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P2_Enc16L (3 total)
[ckpt] Saved all_detailed_summaries_step000000.pt (0.0 MB)
[detailed] Stored P2_Enc16L

================================================================================
  P2_Enc16L - 1373.1M params
================================================================================
Overall: ChrF=38.97±17.74  BLEU=11.13  RTF=0.1617

Per-Pair (8 pairs):
  Pair               N     ChrF     BLEU      RTF
  arb→eng           25    44.47    13.14   0.1621
  ben→eng           25    37.85     8.31   0.1285
  cmn→eng           25    41.46     9.99   0.1527
  eng→arb           25    46.07    13.28   0.1398
  eng→ben           25    46.95    10.24   0.1761
  eng→cmn           25     6.26     3.16   0.2026
  eng→hin           25    52.63    23.04   0.1478
  hin→eng           25    36.09     7.86   0.1838

By Source Language:
     ARB: ChrF= 44.47  BLEU= 13.14  (n=25)
     BEN: ChrF= 37.85  BLEU=  8.31  (n=25)
     CMN: ChrF= 41.46  BLEU=  9.99  (n=25)
     ENG: ChrF= 37.98  BLEU= 12.43  (n=100)
     HIN: ChrF= 36.09  BLEU=  7.86  (n=25)

By Target Language:
     ARB: ChrF= 46.07  BLEU= 13.28  (n=25)
     BEN: ChrF= 46.95  BLEU= 10.24  (n=25)
     CMN: ChrF=  6.26  BLEU=  3.16  (n=25)
     ENG: ChrF= 39.97  BLEU=  9.83  (n=100)
     HIN: ChrF= 52.63  BLEU= 23.04  (n=25)
================================================================================
Plotting 3 phases: ['P0_V1_Baseline', 'P1_Vocab5L', 'P2_Enc16L']
[fig] Saved phase_comparison.png

<Figure size 1920x1200 with 4 Axes>
[image/png output omitted]
Plotting detailed comparison for 3 phases: ['P0_V1_Baseline', 'P1_Vocab5L', 'P2_Enc16L']

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_01_overall_quality.png  [Overall Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_02_chrf_by_pair.png  [ChrF by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_03_bleu_by_pair.png  [BLEU by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_04_src_lang_trends.png  [Source Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_05_tgt_lang_trends.png  [Target Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_06_size_vs_quality.png  [Size vs Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_07_inference_rtf.png  [Inference Speed RTF]

<Figure size 1800x567 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_08_summary_table.png  [Summary Table]

✅ All 8 figures saved individually:
   📄 detailed_comparison_01_overall_quality.png
   📄 detailed_comparison_02_chrf_by_pair.png
   📄 detailed_comparison_03_bleu_by_pair.png
   📄 detailed_comparison_04_src_lang_trends.png
   📄 detailed_comparison_05_tgt_lang_trends.png
   📄 detailed_comparison_06_size_vs_quality.png
   📄 detailed_comparison_07_inference_rtf.png
   📄 detailed_comparison_08_summary_table.png
```

### Cell 51 (markdown, score=3)
```markdown
---
## Phase 3: T2U LaCo RDSC Merge (6+6 → 4+6 layers)
LaCo reserves weight differences instead of outright removal, preserving >80% capacity.
Better than iterative removal for T2U because every layer matters for unit quality.
Paper: Yang et al. EMNLP Findings 2024 (LaCo arXiv:2402.11187).
```

### Cell 52 (code, score=93)
```python
# # ── find_t2u_stacks, sync_t2u_layer_indices from v5 Cell 85 (verbatim) ───────

# def find_t2u_stacks(model):
#     t2u, stacks = model.t2u_model, []
#     def _search(module, prefix):
#         for attr in ['layers', 'inner_layers', 'encoder_layers', 'decoder_layers']:
#             if hasattr(module, attr):
#                 layers = getattr(module, attr)
#                 if isinstance(layers, nn.ModuleList) and len(layers) >= 3:
#                     stacks.append((module, attr, f't2u.{prefix}.{attr}'))
#         for name, child in module.named_children():
#             _search(child, f'{prefix}.{name}' if prefix else name)
#     _search(t2u, 't2u_model')
#     return stacks

# def sync_t2u_layer_indices(model):
#     for (parent, attr, name) in find_t2u_stacks(model):
#         for i, layer in enumerate(list(getattr(parent, attr))):
#             for aname in ['self_attn', 'encoder_attn', 'cross_attention']:
#                 attn = getattr(layer, aname, None)
#                 if attn and hasattr(attn,'layer_idx'): attn.layer_idx = i
#         print(f'  Re-indexed {name}: {len(getattr(parent,attr))} layers')

# # ── LaCo RDSC merge (PLAN.md Section 7 Phase 3) ──────────────────────────────
# def laco_rdsc_merge(layer_i, layer_j, alpha=0.5):
#     """RDSC: W_merged = W_j + alpha*(W_j - W_i)  — preserves weight differences."""    
#     merged = copy.deepcopy(layer_j)
#     sd_i = layer_i.state_dict(); sd_j = layer_j.state_dict()
#     merged_sd = {k: (sd_j[k].float() + alpha*(sd_j[k].float()-sd_i[k].float())).to(sd_j[k].dtype)
#                  if k in sd_i and sd_i[k].shape == sd_j[k].shape else sd_j[k]
#                  for k in sd_j}
#     merged.load_state_dict(merged_sd)
#     return merged

# def _cosine_sim_layers(merged, orig_j, calib_tensors, device):
#     """Measure output similarity between merged and original layer_j."""    
#     orig_j = orig_j.to(device).eval()
#     merged = merged.to(device).eval()
    
#     # Get the dtype from the model layers
#     model_dtype = next(orig_j.parameters()).dtype
    
#     sims = []
#     for x in calib_tensors[:5]:
#         if x is None: continue
#         # Convert calibration tensor to match model dtype
#         x = x.to(device=device, dtype=model_dtype)
#         with torch.no_grad():
#             try:
#                 # T2U layers expect (hidden_states, attention_mask=None, ...)
#                 o = orig_j(x, attention_mask=None)
#                 o = o[0] if isinstance(o, tuple) else o
#                 m = merged(x, attention_mask=None)
#                 m = m[0] if isinstance(m, tuple) else m
#                 # Compute cosine similarity on flattened tensors
#                 sim = F.cosine_similarity(o.reshape(-1), m.reshape(-1), dim=0).item()
#                 sims.append(sim)
#             except Exception as e:
#                 # Debug: print what went wrong
#                 print(f' [sim_err: {str(e)[:50]}]', end='')
#                 pass
#     return float(np.mean(sims)) if sims else 0.0
# def apply_laco_t2u(model, sim_threshold=0.85, alpha=0.5, max_per_stack=2):
#     """Apply LaCo RDSC to T2U encoder + decoder stacks (2 merges each)."""    
#     t2u_enc, t2u_dec = _get_t2u_encoder_decoder(model)
#     device = next(model.t2u_model.parameters()).device
#     # Build calibration tensors from speech encoder outputs
#     calib = []
#     for s in eval_samples[:8]:
#         try:
#             inp = processor(audio=s['wav'], sampling_rate=16000, return_tensors='pt')
#             inp = {k: v.to(device) for k,v in inp.items() if isinstance(v,torch.Tensor)}
#             with torch.no_grad():
#                 enc_out = model.speech_encoder(
#                     input_features=inp['input_features'],
#                     attention_mask=inp.get('attention_mask')).last_hidden_state
#             calib.append(enc_out.cpu().float())
#         except: pass
#     print(f'  Built {len(calib)} calibration tensors.')
#     for stack_obj, sname in [(t2u_enc, 'T2U-Enc'), (t2u_dec, 'T2U-Dec')]:
#         if stack_obj is None or not hasattr(stack_obj,'layers'): continue
#         layers = list(stack_obj.layers)
#         collapsed, n_rm = [layers[0]], 0
#         print(f'\n  {sname}: {len(layers)} layers -> merging up to {max_per_stack}')
#         for i in range(1, len(layers)):
#             if n_rm >= max_per_stack:
#                 collapsed.append(layers[i]); continue
#             candidate = laco_rdsc_merge(collapsed[-1], layers[i], alpha)
#             sim = _cosine_sim_layers(candidate, layers[i], calib, device)
#             print(f'  L{i}: sim={sim:.4f}', end='')
#             if sim > sim_threshold:
#                 collapsed[-1] = candidate; n_rm += 1
#                 print(f' -> MERGED [{n_rm}/{max_per_stack}]')
#             else:
#                 collapsed.append(layers[i])
#                 print(f' -> kept (below {sim_threshold})')
#         stack_obj.layers = nn.ModuleList(collapsed)
#         print(f'  {sname}: {len(layers)} -> {len(collapsed)} layers')
#     sync_t2u_layer_indices(model)
#     sync_model_config(model)
#     return model

# print('LaCo RDSC merge ready.')
```

### Cell 54 (code, score=43)
```python
# # ── RUN Phase 3 ───────────────────────────────────────────────────────────────
# p3_done = load_latest_checkpoint('phase3_laco_done')
# if p3_done:
#     print('Phase 3 already complete — loading from Drive.')
#     try:
#         model_p3, processor = load_model_from_drive('phase3_t2u_laco')
#     except:
#         print('  Rebuilding in-memory...')
#         model_p3 = model_p2
#         model_p3 = apply_laco_t2u(model_p3)
#         save_model_to_drive(model_p3, processor, 'phase3_t2u_laco')
# else:
#     print('Running Phase 3: LaCo T2U merge...')
#     model_p3 = _consolidate_to_single_gpu(model_p2)
#     model_p3 = apply_laco_t2u(model_p3, sim_threshold=0.85, alpha=0.5, max_per_stack=2)
#     print_model_breakdown(model_p3, 'After Phase 3: LaCo T2U 4+4L')
#     save_model_to_drive(model_p3, processor, 'phase3_t2u_laco')
#     save_checkpoint({'done': True, 'alpha': 0.5, 'sim_threshold': 0.85},
#                     'phase3_laco_done', 0)

# print_model_breakdown(model_p3, 'Phase 3 Model (Enc16L + T2U 4+6L)')

# # Quick verify T2U stacks
# for (parent, attr, name) in find_t2u_stacks(model_p3):
#     print(f'  {name}: {len(getattr(parent,attr))} layers remaining')
```

### Cell 55 (code, score=121)
```python
p3_bench = load_latest_checkpoint('phase3_benchmark')
if p3_bench:
    p3_results = p3_bench['results']
    p3_summary = p3_bench['summary']
    p3_detailed = p3_bench.get('detailed_summary')
    if not p3_detailed:
        p3_detailed = compute_detailed_summary(p3_results, 'P3_LaCoT2U', p3_summary['params_M'])
else:
    p3_results, p3_summary = run_benchmark_asr(
        model_p3, eval_samples, 'P3_LaCoT2U', save_n=4)
    p3_detailed = compute_detailed_summary(p3_results, 'P3_LaCoT2U', p3_summary['params_M'])
    save_checkpoint(dict(
        results=p3_results, 
        summary=p3_summary,
        detailed_summary=p3_detailed
    ), 'phase3_benchmark', 0)

store_summary(p3_summary)
store_detailed_summary(p3_detailed)
print_detailed_summary_table('P3_LaCoT2U')
plot_phase_comparison()
plot_detailed_phase_comparison()
```
OUTPUT:
```text
[ckpt] Loaded phase3_benchmark_step000000.pt
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P3_LaCoT2U (4 total)
[ckpt] Saved all_detailed_summaries_step000000.pt (0.0 MB)
[detailed] Stored P3_LaCoT2U

================================================================================
  P3_LaCoT2U - 1331.2M params
================================================================================
Overall: ChrF=38.47±17.92  BLEU=11.21  RTF=0.1646

Per-Pair (8 pairs):
  Pair               N     ChrF     BLEU      RTF
  arb→eng           25    44.59    13.15   0.1646
  ben→eng           25    38.26     8.59   0.1348
  cmn→eng           25    41.57     9.94   0.1540
  eng→arb           25    43.39    12.37   0.1403
  eng→ben           25    46.34    10.07   0.1814
  eng→cmn           25     4.81     2.91   0.2058
  eng→hin           25    53.06    24.67   0.1510
  hin→eng           25    35.72     7.99   0.1851

By Source Language:
     ARB: ChrF= 44.59  BLEU= 13.15  (n=25)
     BEN: ChrF= 38.26  BLEU=  8.59  (n=25)
     CMN: ChrF= 41.57  BLEU=  9.94  (n=25)
     ENG: ChrF= 36.90  BLEU= 12.51  (n=100)
     HIN: ChrF= 35.72  BLEU=  7.99  (n=25)

By Target Language:
     ARB: ChrF= 43.39  BLEU= 12.37  (n=25)
     BEN: ChrF= 46.34  BLEU= 10.07  (n=25)
     CMN: ChrF=  4.81  BLEU=  2.91  (n=25)
     ENG: ChrF= 40.03  BLEU=  9.92  (n=100)
     HIN: ChrF= 53.06  BLEU= 24.67  (n=25)
================================================================================
Plotting 4 phases: ['P0_V1_Baseline', 'P1_Vocab5L', 'P2_Enc16L', 'P3_LaCoT2U']
[fig] Saved phase_comparison.png

<Figure size 1920x1200 with 4 Axes>
[image/png output omitted]
Plotting detailed comparison for 4 phases: ['P0_V1_Baseline', 'P1_Vocab5L', 'P2_Enc16L', 'P3_LaCoT2U']

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_01_overall_quality.png  [Overall Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_02_chrf_by_pair.png  [ChrF by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_03_bleu_by_pair.png  [BLEU by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_04_src_lang_trends.png  [Source Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_05_tgt_lang_trends.png  [Target Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_06_size_vs_quality.png  [Size vs Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_07_inference_rtf.png  [Inference Speed RTF]

<Figure size 1800x666 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_08_summary_table.png  [Summary Table]

✅ All 8 figures saved individually:
   📄 detailed_comparison_01_overall_quality.png
   📄 detailed_comparison_02_chrf_by_pair.png
   📄 detailed_comparison_03_bleu_by_pair.png
   📄 detailed_comparison_04_src_lang_trends.png
   📄 detailed_comparison_05_tgt_lang_trends.png
   📄 detailed_comparison_06_size_vs_quality.png
   📄 detailed_comparison_07_inference_rtf.png
   📄 detailed_comparison_08_summary_table.png
```

### Cell 56 (code, score=7)
```python
# # Upload all artefacts
# if ON_KAGGLE:
#     subprocess.run(f'rclone copy "{AUDIO_DIR}/" "{GDRIVE_ROOT}/audio/" --transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M', shell=True)
#     subprocess.run(f'rclone copy "{FIG_DIR}/" "{GDRIVE_ROOT}/figures/" --transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M', shell=True)
#     print('[rclone] Audio + figures synced to Drive.')

# session_status()
# print('\n✓ Phase 7 complete. All results persisted to Drive.')
```

### Cell 58 (markdown, score=8)
```markdown
## Phase 4: Speech Encoder Additional Pruning (16 → 14 layers)
Target: Remove 4 more layers from speech encoder (currently 16L after Phase 2).
Method: BI-guided iterative greedy (same as Phase 2).
Conservative pruning to maintain quality.
```

### Cell 59 (code, score=15)
```python
# # Load Phase 3 model
# model_p3, processor = load_model_from_drive('phase3_t2u_laco')
# print_model_breakdown(model_p3, 'Phase 3 Model (Enc16L + T2U 4+4L)')

# # Consolidate to single GPU
# model_p4 = _consolidate_to_single_gpu(model_p3)

# # Parameters
# N_ENC_REMOVE_P4 = 2
# ENC_BI_RATIO_P4 = 0.5
```

### Cell 60 (code, score=61)
```python
# # Check for existing checkpoint
# p4_ckpt = load_latest_checkpoint('phase4_enc_pruning')
# p4_complete = p4_ckpt and len(p4_ckpt.get('removed', [])) >= N_ENC_REMOVE_P4

# if p4_complete:
#     print(f'Phase 4 complete: removed {p4_ckpt["removed"]}')
#     try:
#         model_p4, processor = load_model_from_drive('phase4_enc_14L')
#     except:
#         print('  Rebuilding from checkpoint...')
#         parent, la = get_speech_encoder_layers(model_p4)
#         cur = list(getattr(parent, la))
#         keep = [i for i in range(len(cur)) if i not in p4_ckpt['removed']]
#         setattr(parent, la, nn.ModuleList([cur[i] for i in keep]))
#         sync_model_config(model_p4)
#         save_model_to_drive(model_p4, processor, 'phase4_enc_14L')
# else:
#     done = len(p4_ckpt['removed']) if p4_ckpt else 0
#     print(f'{"Resuming" if done else "Running"} Phase 4: enc pruning ({done}/{N_ENC_REMOVE_P4} done)...')
    
#     # Sanity check
#     sanity = quick_eval_chrf(model_p4, eval_samples)
#     print(f'  Sanity ChrF={sanity:.2f}')
#     assert sanity > 10, f'Sanity too low: {sanity:.2f}'
    
#     # Compute or load BI scores
#     if not (p4_ckpt and p4_ckpt.get('bi_scores')):
#         print('Computing Block Influence scores...')
#         bi_scores = compute_block_influence(model_p4, eval_samples, max_n=50)
#         save_checkpoint(dict(removed=[], log=[], bi_scores=bi_scores), 
#                        'phase4_enc_pruning', 0)
#     else:
#         bi_scores = p4_ckpt['bi_scores']
#         print(f'  BI scores loaded ({len(bi_scores)} layers)')
    
#     # Get protected layers
#     parent_tmp, la_tmp = get_speech_encoder_layers(model_p4)
#     n_enc = len(getattr(parent_tmp, la_tmp))
#     enc_protected = _get_protected_enc(n_enc)
    
#     # Run iterative pruning
#     removed_enc, p4_log = iterative_enc_prune(
#         model_p4, eval_samples, N_ENC_REMOVE_P4, max_eval=16,
#         ckpt_name='phase4_enc_pruning', bi_scores=bi_scores,
#         bi_candidate_ratio=ENC_BI_RATIO_P4, protected=enc_protected)
    
#     sync_model_config(model_p4)
#     save_checkpoint(dict(removed=removed_enc, log=p4_log, bi_scores=bi_scores),
#                    'phase4_enc_pruning', 0)
#     save_model_to_drive(model_p4, processor, 'phase4_enc_14L')

# print(f'Encoder layers removed: {removed_enc}')
# print_model_breakdown(model_p4, 'After Phase 4: Enc 14L')
```

### Cell 61 (code, score=68)
```python
# ═══════════════════════════════════════════════════════════════════════════════
# CREATE PUBLIC KAGGLE DATASET from downloaded data
# Run this ONCE after the first successful data download.
# The created dataset can then be mounted in any future Kaggle session as
# /kaggle/input/<your-username>/fleurs5/
# ═══════════════════════════════════════════════════════════════════════════════
import os, json, pathlib, subprocess

def create_kaggle_dataset(
    dataset_title='fleurs5',
    source_dir='/kaggle/working/fleurs_parquet'
):
    if not ON_KAGGLE:
        print('Kaggle dataset creation only works on Kaggle.')
        return

    # ── Kaggle API key ─────────────────────────────────────
    try:
        kag_json = _get_secret('KAGGLE_API_TOKEN')
        kag_path = pathlib.Path.home() / '.kaggle/kaggle.json'
        kag_path.parent.mkdir(parents=True, exist_ok=True)
        kag_path.write_text(kag_json)
        kag_path.chmod(0o600)
        print('[kaggle] API key ready.')
    except Exception as e:
        print(f'[kaggle] ERROR: API key missing: {e}')
        return

    # ── Metadata ───────────────────────────────────────────
    try:
        kag_data = json.loads(kag_json)
        username = kag_data['username']
        print(f"user: {username}")
    except:
        username = 'rayedriasat'

    metadata = {
        'title': dataset_title,
        'id': f'{username}/{dataset_title}',
        'licenses': [{'name': 'CC0-1.0'}]
    }

    meta_path = os.path.join(source_dir, 'dataset-metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'[kaggle] Metadata written at {meta_path}')

    # ── Create dataset (NO COPYING, direct folder) ─────────
    r = subprocess.run(
        ['kaggle', 'datasets', 'create', '-p', source_dir, '--dir-mode', 'tar'],
        text=True  # no capture_output
    )
    
    if r.returncode == 0:
        print(f'✅ Dataset created: {username}/{dataset_title}')
        print(f'Use later at: /kaggle/input/{dataset_title}/')
    else:
        print('[kaggle] Create failed. Trying version update...')
    
        r2 = subprocess.run(
            ['kaggle', 'datasets', 'version', '-p', source_dir,
             '-m', 'update', '--dir-mode', 'tar'],
            text=True
        )
    
        if r2.returncode == 0:
            print('[kaggle] Version updated ✅')
        else:
            print('[kaggle] Update also failed ❌')
            
# ── Usage: uncomment to run ───────────────────────────────────────────────────
# create_kaggle_dataset()
print('Kaggle dataset creator ready.  Uncomment create_kaggle_dataset() to publish.')
print('Future sessions: mount your dataset at /kaggle/input/<dataset-name>/')
print()
print('If you already have the Kaggle dataset mounted, set FLEURS_CACHE to the mounted path:')
print('  FLEURS_CACHE = "/kaggle/input/datasets/rayedriasat/fleurs5"')
```
OUTPUT:
```text
Kaggle dataset creator ready.  Uncomment create_kaggle_dataset() to publish.
Future sessions: mount your dataset at /kaggle/input/<dataset-name>/

If you already have the Kaggle dataset mounted, set FLEURS_CACHE to the mounted path:
  FLEURS_CACHE = "/kaggle/input/datasets/rayedriasat/fleurs5"
```

### Cell 62 (code, score=121)
```python
# Benchmark
p4_bench = load_latest_checkpoint('phase4_benchmark')
if p4_bench:
    p4_results = p4_bench['results']
    p4_summary = p4_bench['summary']
    p4_detailed = p4_bench.get('detailed_summary')
    if not p4_detailed:
        p4_detailed = compute_detailed_summary(p4_results, 'P4_Enc14L', p4_summary['params_M'])
else:
    p4_results, p4_summary = run_benchmark_asr(
        model_p4, eval_samples, 'P4_Enc14L', save_n=4)
    p4_detailed = compute_detailed_summary(p4_results, 'P4_Enc14L', p4_summary['params_M'])
    save_checkpoint(dict(
        results=p4_results, 
        summary=p4_summary,
        detailed_summary=p4_detailed
    ), 'phase4_benchmark', 0)

store_summary(p4_summary)
store_detailed_summary(p4_detailed)
print_detailed_summary_table('P4_Enc14L')
plot_phase_comparison()
plot_detailed_phase_comparison()
```
OUTPUT:
```text
[ckpt] Loaded phase4_benchmark_step000000.pt
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P4_Enc14L (5 total)
[ckpt] Saved all_detailed_summaries_step000000.pt (0.0 MB)
[detailed] Stored P4_Enc14L

================================================================================
  P4_Enc14L - 1282.8M params
================================================================================
Overall: ChrF=35.74±16.30  BLEU=9.67  RTF=0.1635

Per-Pair (8 pairs):
  Pair               N     ChrF     BLEU      RTF
  arb→eng           25    40.67     9.82   0.1582
  ben→eng           25    33.04     5.66   0.1205
  cmn→eng           25    37.53     7.70   0.1698
  eng→arb           25    39.60    10.76   0.1424
  eng→ben           25    43.15     9.56   0.1587
  eng→cmn           25     4.12     2.31   0.2472
  eng→hin           25    49.04    22.78   0.1619
  hin→eng           25    38.79     8.81   0.1495

By Source Language:
     ARB: ChrF= 40.67  BLEU=  9.82  (n=25)
     BEN: ChrF= 33.04  BLEU=  5.66  (n=25)
     CMN: ChrF= 37.53  BLEU=  7.70  (n=25)
     ENG: ChrF= 33.98  BLEU= 11.35  (n=100)
     HIN: ChrF= 38.79  BLEU=  8.81  (n=25)

By Target Language:
     ARB: ChrF= 39.60  BLEU= 10.76  (n=25)
     BEN: ChrF= 43.15  BLEU=  9.56  (n=25)
     CMN: ChrF=  4.12  BLEU=  2.31  (n=25)
     ENG: ChrF= 37.51  BLEU=  8.00  (n=100)
     HIN: ChrF= 49.04  BLEU= 22.78  (n=25)
================================================================================
Plotting 5 phases: ['P0_V1_Baseline', 'P1_Vocab5L', 'P2_Enc16L', 'P3_LaCoT2U', 'P4_Enc14L']
[fig] Saved phase_comparison.png

<Figure size 1920x1200 with 4 Axes>
[image/png output omitted]
Plotting detailed comparison for 5 phases: ['P0_V1_Baseline', 'P1_Vocab5L', 'P2_Enc16L', 'P3_LaCoT2U', 'P4_Enc14L']

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_01_overall_quality.png  [Overall Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_02_chrf_by_pair.png  [ChrF by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_03_bleu_by_pair.png  [BLEU by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_04_src_lang_trends.png  [Source Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_05_tgt_lang_trends.png  [Target Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_06_size_vs_quality.png  [Size vs Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_07_inference_rtf.png  [Inference Speed RTF]

<Figure size 1800x765 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_08_summary_table.png  [Summary Table]

✅ All 8 figures saved individually:
   📄 detailed_comparison_01_overall_quality.png
   📄 detailed_comparison_02_chrf_by_pair.png
   📄 detailed_comparison_03_bleu_by_pair.png
   📄 detailed_comparison_04_src_lang_trends.png
   📄 detailed_comparison_05_tgt_lang_trends.png
   📄 detailed_comparison_06_size_vs_quality.png
   📄 detailed_comparison_07_inference_rtf.png
   📄 detailed_comparison_08_summary_table.png
```

### Cell 63 (code, score=40)
```python
# ══════════════════════════════════════════════════════════════════════════════
# RELOAD ALL SUMMARIES FROM CHECKPOINT (in case we skipped earlier phases)
# ══════════════════════════════════════════════════════════════════════════════

print('Reloading all summaries from checkpoint...')
ALL_SUMMARIES = _load_summaries_from_drive()
ALL_DETAILED_SUMMARIES = _load_detailed_summaries_from_drive()
print(f'Loaded {len(ALL_SUMMARIES)} summaries: {list(ALL_SUMMARIES.keys())}')
print(f'Loaded {len(ALL_DETAILED_SUMMARIES)} detailed summaries: {list(ALL_DETAILED_SUMMARIES.keys())}')
```
OUTPUT:
```text
Reloading all summaries from checkpoint...
[ckpt] Loaded all_summaries_step000000.pt
[ckpt] Loaded all_detailed_summaries_step000000.pt
Loaded 5 summaries: ['P0_V1_Baseline', 'P1_Vocab5L', 'P2_Enc16L', 'P3_LaCoT2U', 'P4_Enc14L']
Loaded 5 detailed summaries: ['P0_V1_Baseline', 'P1_Vocab5L', 'P2_Enc16L', 'P3_LaCoT2U', 'P4_Enc14L']
```

### Cell 64 (markdown, score=7)
```markdown
## Phase 5: Text Decoder Pruning (24 → 14 layers)
Target: Remove 12 layers from text decoder (currently 24L, unpruned).
Method: BI-guided iterative greedy (adapted for decoder).
This is the first time we prune the text decoder.
```

### Cell 65 (code, score=79)
```python
# Helper functions for text decoder
def get_text_decoder_layers(mdl):
    """Get text decoder layers ModuleList."""
    dec = mdl.text_decoder
    if hasattr(dec, 'layers') and isinstance(dec.layers, nn.ModuleList):
        return dec, 'layers'
    for attr in ['decoder', 'model']:
        if hasattr(dec, attr):
            sub = getattr(dec, attr)
            if hasattr(sub, 'layers') and isinstance(sub.layers, nn.ModuleList):
                return sub, 'layers'
    raise RuntimeError('Cannot find text decoder layers')

def compute_decoder_block_influence(mdl, samples, max_n=50):
    """
    Compute Block Influence for text decoder layers.
    BI = 1 - cos(layer_input, layer_output)
    """
    parent, la = get_text_decoder_layers(mdl)
    layers = getattr(parent, la)
    n = len(layers)
    bi = {i: [] for i in range(n)}
    hooks = []
    
    for i in range(n):
        def make_hook(idx):
            def hook(mod, inp, out):
                x = inp[0]
                if x is None or not isinstance(x, torch.Tensor):
                    return
                y = out[0] if isinstance(out, tuple) else out
                if y is None or not isinstance(y, torch.Tensor):
                    return
                x = x.detach().float().reshape(-1, x.shape[-1])
                y = y.detach().to(x.device).float().reshape(-1, y.shape[-1])
                bi[idx].append(1.0 - F.cosine_similarity(x, y, dim=-1).mean().item())
            return hook
        hooks.append(layers[i].register_forward_hook(make_hook(i)))
    
    mdl.eval()
    dev = next(mdl.text_decoder.parameters()).device
    ok = 0
    
    for idx, s in enumerate(samples[:max_n]):
        if idx % 10 == 0:
            print(f'  Calibrating decoder BI {idx}/{min(max_n, len(samples))}...')
        try:
            inputs = processor(audio=s['wav'], sampling_rate=16000, return_tensors='pt')
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                _ = mdl.generate(**inputs, tgt_lang=s['tgt_lang'])
            ok += 1
        except Exception as e:
            print(f'  Sample {idx} failed: {e}')
    
    for h in hooks:
        h.remove()
    
    scores = {i: float(np.mean(v)) if v else 0.0 for i, v in bi.items()}
    print(f'  Calibrated {ok}/{min(max_n, len(samples))} samples.')
    
    ranked = sorted(scores.items(), key=lambda x: x[1])
    print('  Decoder BI ranking (low=redundant):')
    for rank, (li, bv) in enumerate(ranked):
        print(f'    Rank{rank+1:>2}  L{li:>2}  BI={bv:.4f}')
    
    return scores

def _get_protected_dec(n_total):
    """Protect first, middle, and last decoder layers."""
    return {0, n_total//2, n_total-1}

def iterative_dec_prune(mdl, samples, n_remove, tgt_lang='ben', max_eval=16,
                       ckpt_name='phase5_dec_pruning', bi_scores=None,
                       bi_candidate_ratio=0.5, protected=None):
    """BI-guided iterative greedy decoder pruning."""
    parent, la = get_text_decoder_layers(mdl)
    current = list(getattr(parent, la))
    orig_idx = list(range(len(current)))
    n_total = len(current)
    removed, log = [], []
    
    if protected is None:
        protected = _get_protected_dec(n_total)
    print(f'  Protected decoder layers (first/mid/last): {sorted(protected)}')
    
    partial = load_latest_checkpoint(ckpt_name)
    if partial and partial.get('removed'):
        removed = list(partial['removed'])
        log = partial.get('log', [])
        for r in removed:
            if r in orig_idx:
                pos = orig_idx.index(r)
                current.pop(pos)
                orig_idx.pop(pos)
        setattr(parent, la, nn.ModuleList(current))
        print(f'  Resuming: removed {removed}, {len(current)} layers remain')
    
    baseline = quick_eval_chrf(mdl, samples, max_samples=max_eval)
    print(f'  Baseline ChrF: {baseline:.2f}')
    
    for it in range(len(removed), n_remove):
        eligible = [pos for pos in range(len(current)) if orig_idx[pos] not in protected]
        
        if bi_scores and len(eligible) > 2:
            by_bi = sorted(eligible, key=lambda pos: bi_scores.get(orig_idx[pos], float('inf')))
            n_cands = max(2, int(len(by_bi) * bi_candidate_ratio))
            cands = by_bi[:n_cands]
            print(f'\n  Iter {it+1}/{n_remove} | BI pre-filter: {len(cands)}/{len(eligible)} cands')
        else:
            cands = eligible
            print(f'\n  Iter {it+1}/{n_remove} | all {len(cands)} eligible (no BI)')
        
        if not cands:
            print('  No candidates left, stopping.')
            break
        
        scores = {}
        for pos in cands:
            temp = current[:pos] + current[pos+1:]
            setattr(parent, la, nn.ModuleList(temp))
            sc = quick_eval_chrf(mdl, samples, max_samples=max_eval)
            bi_note = f'  BI={bi_scores.get(orig_idx[pos], 0):.4f}' if bi_scores else ''
            print(f'    Remove L{orig_idx[pos]:>2} -> ChrF={sc:.2f}{bi_note}')
            scores[pos] = (orig_idx[pos], sc)
        
        setattr(parent, la, nn.ModuleList(current))
        
        best_pos = max(scores, key=lambda k: scores[k][1])
        best_orig, best_sc = scores[best_pos]
        current.pop(best_pos)
        orig_idx.pop(best_pos)
        setattr(parent, la, nn.ModuleList(current))
        removed.append(best_orig)
        
        log.append(dict(
            iter=it+1, removed=best_orig, chrf=best_sc,
            remaining=len(current),
            bi_score=bi_scores.get(best_orig) if bi_scores else None))
        
```
OUTPUT:
```text
Decoder pruning helpers ready.
```

### Cell 66 (code, score=71)
```python
# # Load Phase 4 model
# model_p4, processor = load_model_from_drive('phase4_enc_14L')
# print_model_breakdown(model_p4, 'Phase 4 Model (Enc14L + Dec24L + T2U 6+4L)')

# model_p5 = _consolidate_to_single_gpu(model_p4)

# N_DEC_REMOVE = 10
# DEC_BI_RATIO = 0.5

# p5_ckpt = load_latest_checkpoint('phase5_dec_pruning')
# p5_complete = p5_ckpt and len(p5_ckpt.get('removed', [])) >= N_DEC_REMOVE

# if p5_complete:
#     print(f'Phase 5 complete: removed {p5_ckpt["removed"]}')
#     try:
#         model_p5, processor = load_model_from_drive('phase5_dec_14L')
#     except:
#         print('  Rebuilding from checkpoint...')
#         parent, la = get_text_decoder_layers(model_p5)
#         cur = list(getattr(parent, la))
#         keep = [i for i in range(len(cur)) if i not in p5_ckpt['removed']]
#         setattr(parent, la, nn.ModuleList([cur[i] for i in keep]))
#         sync_model_config(model_p5)
#         save_model_to_drive(model_p5, processor, 'phase5_dec_14L')
# else:
#     done = len(p5_ckpt['removed']) if p5_ckpt else 0
#     print(f'{"Resuming" if done else "Running"} Phase 5: dec pruning ({done}/{N_DEC_REMOVE} done)...')
    
#     sanity = quick_eval_chrf(model_p5, eval_samples, 10)
#     print(f'  Sanity ChrF={sanity:.2f}')
#     assert sanity > 10, f'Sanity too low: {sanity:.2f}'
    
#     if not (p5_ckpt and p5_ckpt.get('bi_scores')):
#         print('Computing decoder Block Influence scores...')
#         bi_scores = compute_decoder_block_influence(model_p5, eval_samples, max_n=50)
#         save_checkpoint(dict(removed=[], log=[], bi_scores=bi_scores),
#                        'phase5_dec_pruning', 0)
#     else:
#         bi_scores = p5_ckpt['bi_scores']
#         print(f'  Decoder BI scores loaded ({len(bi_scores)} layers)')
    
#     parent_tmp, la_tmp = get_text_decoder_layers(model_p5)
#     n_dec = len(getattr(parent_tmp, la_tmp))
#     dec_protected = _get_protected_dec(n_dec)
    
#     removed_dec, p5_log = iterative_dec_prune(
#         model_p5, eval_samples, N_DEC_REMOVE, max_eval=16,
#         ckpt_name='phase5_dec_pruning', bi_scores=bi_scores,
#         bi_candidate_ratio=DEC_BI_RATIO, protected=dec_protected)
    
#     sync_model_config(model_p5)
#     save_checkpoint(dict(removed=removed_dec, log=p5_log, bi_scores=bi_scores),
#                    'phase5_dec_pruning', 0)
#     save_model_to_drive(model_p5, processor, 'phase5_dec_14L')

# print(f'Decoder layers removed: {removed_dec}')
# print_model_breakdown(model_p5, 'After Phase 5: Enc14L + Dec14L')
```

### Cell 68 (code, score=12)
```python
# # Load Phase 4 model
# model_p5, processor = load_model_from_drive('phase5_dec_14L')
# print_model_breakdown(model_p5, 'Phase 5 Model (Enc14L + Dec14L + T2U 6+4L)')
```

### Cell 69 (code, score=123)
```python
p5_bench = load_latest_checkpoint('phase5_benchmark')
if p5_bench:
    p5_results = p5_bench['results']
    p5_summary = p5_bench['summary']
    p5_detailed = p5_bench.get('detailed_summary')
    if not p5_detailed:
        p5_detailed = compute_detailed_summary(p5_results, 'P5_Dec14L', p5_summary['params_M'])
else:
    p5_results, p5_summary = run_benchmark_asr(
        model_p5, eval_samples, 'P5_Dec14L', save_n=4)
    p5_detailed = compute_detailed_summary(p5_results, 'P5_Dec14L', p5_summary['params_M'])
    save_checkpoint(dict(
        results=p5_results,
        summary=p5_summary,
        detailed_summary=p5_detailed
    ), 'phase5_benchmark', 0)

store_summary(p5_summary)
store_detailed_summary(p5_detailed)
print_detailed_summary_table('P5_Dec14L')
plot_phase_comparison()
plot_detailed_phase_comparison()

# del model_p4
# gc.collect()
# torch.cuda.empty_cache()
# print('P4 model freed.')
```
OUTPUT:
```text
[ckpt] Loaded phase5_benchmark_step000000.pt
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P5_Dec14L (6 total)
[ckpt] Saved all_detailed_summaries_step000000.pt (0.0 MB)
[detailed] Stored P5_Dec14L

================================================================================
  P5_Dec14L - 1030.9M params
================================================================================
Overall: ChrF=25.32±17.31  BLEU=5.83  RTF=0.1881

Per-Pair (8 pairs):
  Pair               N     ChrF     BLEU      RTF
  arb→eng           25    30.24     6.11   0.1846
  ben→eng           25    24.24     4.29   0.1294
  cmn→eng           25    26.58     4.55   0.1626
  eng→arb           25    23.93     4.48   0.1902
  eng→ben           25    27.42     5.09   0.2519
  eng→cmn           25     1.78     1.00   0.2502
  eng→hin           25    43.27    17.05   0.1327
  hin→eng           25    25.11     4.03   0.2035

By Source Language:
     ARB: ChrF= 30.24  BLEU=  6.11  (n=25)
     BEN: ChrF= 24.24  BLEU=  4.29  (n=25)
     CMN: ChrF= 26.58  BLEU=  4.55  (n=25)
     ENG: ChrF= 24.10  BLEU=  6.91  (n=100)
     HIN: ChrF= 25.11  BLEU=  4.03  (n=25)

By Target Language:
     ARB: ChrF= 23.93  BLEU=  4.48  (n=25)
     BEN: ChrF= 27.42  BLEU=  5.09  (n=25)
     CMN: ChrF=  1.78  BLEU=  1.00  (n=25)
     ENG: ChrF= 26.54  BLEU=  4.75  (n=100)
     HIN: ChrF= 43.27  BLEU= 17.05  (n=25)
================================================================================
Plotting 6 phases: ['P0_V1_Baseline', 'P1_Vocab5L', 'P2_Enc16L', 'P3_LaCoT2U', 'P4_Enc14L', 'P5_Dec14L']
[fig] Saved phase_comparison.png

<Figure size 1920x1200 with 4 Axes>
[image/png output omitted]
Plotting detailed comparison for 6 phases: ['P0_V1_Baseline', 'P1_Vocab5L', 'P2_Enc16L', 'P3_LaCoT2U', 'P4_Enc14L', 'P5_Dec14L']

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_01_overall_quality.png  [Overall Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_02_chrf_by_pair.png  [ChrF by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_03_bleu_by_pair.png  [BLEU by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_04_src_lang_trends.png  [Source Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_05_tgt_lang_trends.png  [Target Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_06_size_vs_quality.png  [Size vs Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_07_inference_rtf.png  [Inference Speed RTF]

<Figure size 1800x864 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_08_summary_table.png  [Summary Table]

✅ All 8 figures saved individually:
   📄 detailed_comparison_01_overall_quality.png
   📄 detailed_comparison_02_chrf_by_pair.png
   📄 detailed_comparison_03_bleu_by_pair.png
   📄 detailed_comparison_04_src_lang_trends.png
   📄 detailed_comparison_05_tgt_lang_trends.png
   📄 detailed_comparison_06_size_vs_quality.png
   📄 detailed_comparison_07_inference_rtf.png
   📄 detailed_comparison_08_summary_table.png
```

### Cell 70 (markdown, score=1)
```markdown
# PHASE 6
```

### Cell 71 (code, score=55)
```python
# ══════════════════════════════════════════════════════════════════════════════
# CHUNKED PARQUET DATASET
# Strategy: pre-load audio in chunks of CHUNK_SIZE samples into RAM.
# When a chunk is exhausted, load the next one (evict previous).
# RAM usage: CHUNK_SIZE × ~0.5MB ≈ 512MB for CHUNK_SIZE=1000
# ══════════════════════════════════════════════════════════════════════════════

import pyarrow.parquet as pq
import numpy as np
import threading

CHUNK_SIZE = 4000  # must match ChunkedMultilingualDataset chunk_size

class ChunkedStreamingDataset:
    """
    Loads audio in RAM chunks. 
    - CHUNK_SIZE=1000 → ~500MB RAM, ~50x fewer disk reads vs pure streaming
    - Evicts previous chunk when moving to next
    - Thread-safe chunk loading with prefetch
    """
    
    def __init__(self, index_samples, chunk_size=CHUNK_SIZE, prefetch=True):
        """
        index_samples: list of metadata dicts (no audio) from ParquetStreamingDataset
        chunk_size: how many samples to hold in RAM at once
        """
        self.index_samples = index_samples  # metadata only, tiny
        self.chunk_size    = chunk_size
        self.prefetch      = prefetch
        
        self._chunk_start  = -1             # which chunk is currently loaded
        self._chunk_data   = {}             # {local_idx: wav_array}
        self._next_chunk   = {}             # prefetched next chunk
        self._lock         = threading.Lock()
        self._prefetch_thread = None
        
        # Group by parquet file for efficient batch reads
        self._file_groups  = self._build_file_groups()
        
        print(f'  ChunkedStreamingDataset: {len(index_samples)} samples | '
              f'chunk={chunk_size} | '
              f'RAM/chunk≈{chunk_size*0.5:.0f}MB')
    
    def _build_file_groups(self):
        """Group sample indices by source parquet file."""
        groups = {}
        for idx, s in enumerate(self.index_samples):
            f = s['_src_file']
            if f not in groups:
                groups[f] = []
            groups[f].append((idx, s['_src_idx']))
        return groups
    
    def _load_chunk_into(self, chunk_start, target_dict):
        """Load one chunk of audio into target_dict."""
        target_dict.clear()
        end = min(chunk_start + self.chunk_size, len(self.index_samples))
        chunk_indices = list(range(chunk_start, end))
        
        # Group by parquet file to batch reads
        by_file = {}
        for idx in chunk_indices:
            s = self.index_samples[idx]
            f = s['_src_file']
            if f not in by_file:
                by_file[f] = []
            by_file[f].append((idx, s['_src_idx']))
        
        # Read each file once, grab all needed rows
        for parquet_file, idx_pairs in by_file.items():
            try:
                table  = pq.read_table(parquet_file, columns=['audio'])
                df_col = table.column('audio')
                
                for global_idx, row_idx in idx_pairs:
                    try:
                        audio_cell = df_col[row_idx].as_py()
                        target_dict[global_idx] = _load_wav(audio_cell)
                    except Exception as e:
                        target_dict[global_idx] = np.zeros(16000, dtype=np.float32)
                
                del table, df_col  # free immediately
            except Exception as e:
                print(f'  [ChunkLoad] Failed {parquet_file}: {e}')
                for global_idx, _ in idx_pairs:
                    target_dict[global_idx] = np.zeros(16000, dtype=np.float32)
    
    def _ensure_chunk(self, idx):
        """Make sure the chunk containing idx is loaded."""
        chunk_start = (idx // self.chunk_size) * self.chunk_size
        
        if chunk_start == self._chunk_start:
            return  # already loaded
        
        with self._lock:
            if chunk_start == self._chunk_start:
                return  # double-check after lock
            
            # Check if prefetch already has it
            if self._next_chunk and chunk_start != self._chunk_start:
                next_start = self._chunk_start + self.chunk_size
                if chunk_start == next_start and self._next_chunk:
                    # Wait for prefetch thread if still running
                    if self._prefetch_thread and self._prefetch_thread.is_alive():
                        self._prefetch_thread.join()
                    self._chunk_data  = self._next_chunk
                    self._chunk_start = chunk_start
                    self._next_chunk  = {}
                else:
                    # Random access — load directly
                    new_chunk = {}
                    self._load_chunk_into(chunk_start, new_chunk)
                    self._chunk_data  = new_chunk
                    self._chunk_start = chunk_start
            else:
                # Fresh load
                new_chunk = {}
                self._load_chunk_into(chunk_start, new_chunk)
                self._chunk_data  = new_chunk
                self._chunk_start = chunk_start
            
            # Trigger prefetch of next chunk in background
            if self.prefetch:
                next_start = chunk_start + self.chunk_size
                if next_start < len(self.index_samples):
                    self._next_chunk = {}
                    self._prefetch_thread = threading.Thread(
                        target=self._load_chunk_into,
                        args=(next_start, self._next_chunk),
                        daemon=True
                    )
                    self._prefetch_thread.start()
    
    def __len__(self):
        return len(self.index_samples)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        
```

### Cell 72 (code, score=18)
```python
class ChunkedMultilingualDataset:
    """
    Drop-in replacement for MultilingualStreamingDataset.
    Collects all metadata first, then wraps in ChunkedStreamingDataset.
    """
    
    def __init__(self, parquet_cache_dir, lang_pairs, split='train',
                 max_samples_per_pair=1200, chunk_size=CHUNK_SIZE):
        
        all_metadata = []
        
        for src_lang, tgt_lang in lang_pairs:
            ds = ParquetStreamingDataset(
                parquet_cache_dir, src_lang, tgt_lang,
                split, max_samples_per_pair
            )
            # Grab metadata (no audio loaded yet)
            all_metadata.extend(ds.samples)
        
        self._chunked = ChunkedStreamingDataset(
            all_metadata, chunk_size=chunk_size, prefetch=True
        )
        
        # Build pair counts for reporting
        pair_counts = {}
        for s in all_metadata:
            pair = f"{s['src_lang']}→{s['tgt_lang']}"
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        print(f'\n✓ ChunkedMultilingualDataset: {len(all_metadata)} samples')
        for pair, count in sorted(pair_counts.items()):
            print(f'  {pair}: {count}')
        print(f'  Chunk size: {chunk_size} | '
              f'Est. peak RAM: ~{chunk_size*0.5:.0f}MB')
    
    def __len__(self):
        return len(self._chunked)
    
    def __getitem__(self, idx):
        return self._chunked[idx]
    
    def __iter__(self):
        return iter(self._chunked)
    
    def notify_shuffle(self):
        """Call after shuffling indices to invalidate chunk cache."""
        self._chunked.invalidate_cache()
```

### Cell 73 (code, score=6)
```python
# Shuffle at chunk granularity, sequential within chunks

def chunk_friendly_shuffle(n_samples, chunk_size, batch_size):
    """Shuffle chunk order, keep sequential access within each chunk."""
    chunks = list(range(0, n_samples, chunk_size))
    random.shuffle(chunks)
    order = []
    for chunk_start in chunks:
        chunk_indices = list(range(chunk_start, min(chunk_start+chunk_size, n_samples)))
        random.shuffle(chunk_indices)   # shuffle within chunk too
        order.extend(chunk_indices)
    return order
```

### Cell 74 (code, score=48)
```python
# ── 1. Rebuild ft_samples with chunking ──────────────────────────────────────
print('Rebuilding ft_samples with chunk caching...')
ft_samples = ChunkedMultilingualDataset(
    parquet_cache_dir = LOCAL_PARQUET_CACHE,
    lang_pairs        = EVAL_LANG_PAIRS,
    split             = 'train',
    max_samples_per_pair = N_TRAIN_PER_PAIR,
    chunk_size        = CHUNK_SIZE,    # ~500MB RAM, tune down to 500 if needed
)
```
OUTPUT:
```text
Rebuilding ft_samples with chunk caching...
  Indexed 1200 samples from eng→ben
  Indexed 1200 samples from ben→eng
  Indexed 1200 samples from eng→cmn
  Indexed 1200 samples from cmn→eng
  Indexed 1200 samples from eng→arb
  Indexed 1200 samples from arb→eng
  Indexed 1200 samples from eng→hin
  Indexed 1200 samples from hin→eng
  ChunkedStreamingDataset: 9600 samples | chunk=4000 | RAM/chunk≈2000MB

✓ ChunkedMultilingualDataset: 9600 samples
  arb→eng: 1200
  ben→eng: 1200
  cmn→eng: 1200
  eng→arb: 1200
  eng→ben: 1200
  eng→cmn: 1200
  eng→hin: 1200
  hin→eng: 1200
  Chunk size: 4000 | Est. peak RAM: ~2000MB
```

### Cell 75 (code, score=37)
```python
# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Online Knowledge Distillation Recovery
# Strategy: Dual-loss (CE + KL divergence on logits) with LoRA on student,
#            frozen teacher on GPU-1, student on GPU-0.
#            Teacher produces soft targets ONLINE — no offline caching.
#
# Architecture reminder:
#   Teacher  = facebook/seamless-m4t-v2-large  (full, frozen, fp16, cuda:1)
#   Student  = phase5_dec_14L                  (pruned 1B, trainable, fp16+LoRA, cuda:0)
#
# Inspired by:
#   • Moslem 2025 (IWSLT): full FT + KD after pruning → 97-100 % quality retention
#   • Self-Data Distillation (2410.09982): SDD > plain SFT for pruned LLMs
#   • DistillLens (2602.13567): intermediate hidden-state matching improves student
#   • Sparse Logit KD (2503.16870): top-K logit matching is memory-efficient
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Cell 33  — Install extras (run once per session)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess
subprocess.run(['pip', 'install', '-q', 'peft>=0.10.0', 'bitsandbytes'], check=False)
print('peft + bitsandbytes ready.')
```
OUTPUT:
```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.7/60.7 MB 32.2 MB/s eta 0:00:00
peft + bitsandbytes ready.
```

### Cell 76 (code, score=92)
```python
# ─────────────────────────────────────────────────────────────────────────────
# Cell 34 — Load pruned student + full teacher (one GPU each)
# ─────────────────────────────────────────────────────────────────────────────
import gc, torch, os
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType

assert torch.cuda.device_count() >= 2, "Need 2 GPUs for this phase."

# ── Student on cuda:0 ────────────────────────────────────────────────────────
print('Loading STUDENT (phase5_dec_14L) on cuda:0 …')
student, processor = load_model_from_drive('phase5_dec_14L', device_map='cuda:0')
student = student.to(torch.float16)      # keep fp16
student.train()

# ── Teacher on cuda:1 ────────────────────────────────────────────────────────
print('\nLoading TEACHER (facebook/seamless-m4t-v2-large) on cuda:1 …')
from transformers import SeamlessM4Tv2ForSpeechToSpeech
teacher = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
    'facebook/seamless-m4t-v2-large',
    torch_dtype=torch.float16,
    device_map='cuda:1',
)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

print('\n✓ Both models loaded.')
gpu_mem()

# ── Sanity: student vocab size vs teacher vocab size ─────────────────────────
S_VOCAB = student.config.vocab_size        # 22767
T_VOCAB = teacher.config.vocab_size        # 256102
print(f'Student vocab: {S_VOCAB}   Teacher vocab: {T_VOCAB}')
# We match logits only in the student's (remapped) vocab space — see loss fn.
```
OUTPUT:
```text
Loading STUDENT (phase5_dec_14L) on cuda:0 …
[model] Not in local cache — pulling from remote...
[rclone] Pulled phase5_dec_14L → /kaggle/working/models/phase5_dec_14L
[model] Loading phase5_dec_14L from /kaggle/working/models/phase5_dec_14L ...

Loading weights:   0%|          | 0/1234 [00:00<?, ?it/s]
  Restored custom state: ['_vocab_remap_to_old']
[model] Loaded phase5_dec_14L.

Loading TEACHER (facebook/seamless-m4t-v2-large) on cuda:1 …

config.json: 0.00B [00:00, ?B/s]
model.safetensors.index.json: 0.00B [00:00, ?B/s]
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]
Loading weights:   0%|          | 0/1846 [00:00<?, ?it/s]
generation_config.json: 0.00B [00:00, ?B/s]

✓ Both models loaded.
  GPU0: 2.09GB alloc / 2.10GB reserved
  GPU1: 3.64GB alloc / 3.65GB reserved
Student vocab: 22767   Teacher vocab: 256102
```

### Cell 77 (code, score=62)
```python
# ─────────────────────────────────────────────────────────────────────────────
# Cell 35 — Attach LoRA adapters to student
# Target: text_decoder q/v/out projections + cross-attention + FFN fc1/fc2
# r=32 gives ~12 M trainable params — substantial but not exploding VRAM
# ─────────────────────────────────────────────────────────────────────────────

LORA_R          = 32
LORA_ALPHA      = 64          # α = 2×r is the standard effective-init rule
LORA_DROPOUT    = 0.05
LORA_TARGET_MODULES = [
    # text decoder self-attention
    'text_decoder.layers.{i}.self_attn.q_proj',
    'text_decoder.layers.{i}.self_attn.v_proj',
    'text_decoder.layers.{i}.self_attn.out_proj',
    # text decoder cross-attention
    'text_decoder.layers.{i}.cross_attention.q_proj',
    'text_decoder.layers.{i}.cross_attention.v_proj',
    'text_decoder.layers.{i}.cross_attention.out_proj',
    # text decoder FFN
    'text_decoder.layers.{i}.ffn.fc1',
    'text_decoder.layers.{i}.ffn.fc2',
]
# Build list without format strings (PEFT matches by suffix)
_lora_targets = [
    'self_attn.q_proj', 'self_attn.v_proj', 'self_attn.out_proj',
    'cross_attention.q_proj', 'cross_attention.v_proj', 'cross_attention.out_proj',
    'ffn.fc1', 'ffn.fc2',
    # also t2u encoder/decoder attention for speech quality recovery
    't2u_model.model.encoder.layers.{}.self_attn.q_proj'.format,
]
# Simpler: just target by module name suffix
LORA_TARGET_SUFFIXES = [
    'q_proj', 'v_proj', 'out_proj', 'fc1', 'fc2',
    'k_proj',                              # include k too for cross-attn quality
]

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias='none',
    target_modules=LORA_TARGET_SUFFIXES,
    # PEFT will skip modules whose in/out dim doesn't match — safe for all layers
)

student = get_peft_model(student, lora_cfg)
student.print_trainable_parameters()

# Enable gradient checkpointing on the student base model to halve activation VRAM
student.base_model.model.text_decoder.gradient_checkpointing = True
# For conformer encoder layers too
try:
    student.base_model.model.speech_encoder.encoder.gradient_checkpointing = True
except AttributeError:
    pass

print('LoRA + gradient checkpointing applied to student.')
gpu_mem()
```
OUTPUT:
```text
trainable params: 20,578,304 || all params: 1,051,482,117 || trainable%: 1.9571
LoRA + gradient checkpointing applied to student.
  GPU0: 2.17GB alloc / 2.19GB reserved
  GPU1: 3.64GB alloc / 3.65GB reserved
```

### Cell 78 (code, score=50)
```python
# ─────────────────────────────────────────────────────────────────────────────
# Cell A — Vocab remap tables
# ─────────────────────────────────────────────────────────────────────────────
 
import torch, gc, random, math, time
import torch.nn.functional as F
import numpy as np
 
_student_base = student.base_model.model
S_VOCAB = _student_base.shared.num_embeddings   # e.g. 22767
T_VOCAB = teacher.config.vocab_size             # 256102
UNK_ID_STUDENT = 3
 
print(f'Student vocab : {S_VOCAB}')
print(f'Teacher vocab : {T_VOCAB}')
 
if hasattr(_student_base, '_vocab_remap_to_old'):
    remap_to_old = _student_base._vocab_remap_to_old
    old_to_new   = torch.full((T_VOCAB,), UNK_ID_STUDENT, dtype=torch.long)
    for new_id, old_id in enumerate(remap_to_old.tolist()):
        if 0 <= old_id < T_VOCAB:
            old_to_new[old_id] = new_id
    print(f'Remap table built. Mapped: {(old_to_new != UNK_ID_STUDENT).sum().item()}')
else:
    print('WARNING: no _vocab_remap_to_old — using clamp fallback.')
    old_to_new = torch.arange(T_VOCAB, dtype=torch.long)
    old_to_new[S_VOCAB:] = UNK_ID_STUDENT
 
OLD_TO_NEW_CPU = old_to_new  # [T_VOCAB] long CPU
 
def remap_ids(ids_cpu: torch.Tensor) -> torch.Tensor:
    flat = ids_cpu.reshape(-1).clamp(0, T_VOCAB - 1)
    return OLD_TO_NEW_CPU[flat].reshape(ids_cpu.shape)
 
_tok        = processor.tokenizer
PAD_ID_FULL = _tok.pad_token_id  if _tok.pad_token_id  is not None else 1
BOS_ID_FULL = _tok.bos_token_id  if _tok.bos_token_id  is not None else 0
 
def _remap_special(full_id):
    if full_id is None or full_id < 0: return UNK_ID_STUDENT
    return int(OLD_TO_NEW_CPU[min(full_id, T_VOCAB - 1)].item())
 
PAD_ID_S = _remap_special(PAD_ID_FULL)
BOS_ID_S = _remap_special(BOS_ID_FULL)
 
def _lang_token_id_student(lang_code):
    full_id = _tok.convert_tokens_to_ids(f'__{lang_code}__')
    if full_id is None or full_id == _tok.unk_token_id:
        return UNK_ID_STUDENT
    s = _remap_special(full_id)
    return max(0, min(s, S_VOCAB - 1))
 
for _l in ['eng', 'ben', 'hin', 'cmn', 'arb']:
    _s = _lang_token_id_student(_l)
    assert 0 <= _s < S_VOCAB, f'{_l} → {_s} out of range'
    print(f'  {_l}: student_id={_s}')
 
print('✓ Vocab remap ready.')
```
OUTPUT:
```text
Student vocab : 22767
Teacher vocab : 256102
Remap table built. Mapped: 22766
  eng: student_id=22690
  ben: student_id=22677
  hin: student_id=22701
  cmn: student_id=22684
  arb: student_id=22671
✓ Vocab remap ready.
```

### Cell 80 (code, score=15)
```python
# ─────────────────────────────────────────────────────────────────────────────
# Pristine Vocabulary Mapping Configuration
# ─────────────────────────────────────────────────────────────────────────────
import torch

_student_base = student.base_model.model
S_VOCAB = _student_base.shared.num_embeddings   # 22767
T_VOCAB = teacher.config.vocab_size             # 256102
UNMAPPED_SENTINEL = -1

# Build clean inverse lookup mapping table
old_to_new_clean = torch.full((T_VOCAB,), UNMAPPED_SENTINEL, dtype=torch.long)
if hasattr(_student_base, '_vocab_remap_to_old'):
    remap_to_old = _student_base._vocab_remap_to_old
    for new_id, old_id in enumerate(remap_to_old.tolist()):
        if 0 <= old_id < T_VOCAB:
            old_to_new_clean[old_id] = new_id

OLD_TO_NEW_CPU = old_to_new_clean

def remap_ids(ids_cpu: torch.Tensor) -> torch.Tensor:
    """Remap full tokenizer tensor IDs to student vocab space."""
    flat = ids_cpu.reshape(-1).clamp(0, T_VOCAB - 1)
    remapped = OLD_TO_NEW_CPU[flat].reshape(ids_cpu.shape)
    # Map any unmapped punctuation/tokens cleanly to student UNK (3)
    remapped[remapped < 0] = 3 
    return remapped

_tok = processor.tokenizer
PAD_ID_FULL = _tok.pad_token_id if _tok.pad_token_id is not None else 1
BOS_ID_FULL = _tok.bos_token_id if _tok.bos_token_id is not None else 0
EOS_ID_FULL = _tok.eos_token_id if _tok.eos_token_id is not None else 2

def _remap_special_token(full_id):
    if full_id is None or full_id < 0: return 3
    mapped = OLD_TO_NEW_CPU[min(full_id, T_VOCAB - 1)].item()
    return mapped if mapped >= 0 else 3

PAD_ID_S = _remap_special_token(PAD_ID_FULL)
BOS_ID_S = _remap_special_token(BOS_ID_FULL)
EOS_ID_S = _remap_special_token(EOS_ID_FULL)

def _lang_token_id_student(lang_code):
    full_id = _tok.convert_tokens_to_ids(f'__{lang_code}__')
    mapped = OLD_TO_NEW_CPU[min(full_id, T_VOCAB - 1)].item()
    return mapped if mapped >= 0 else 3

# print(f"Vocab Setup Complete. BOS_S={BOS_S}, EOS_S={EOS_S}, PAD_S={PAD_S}")
```

### Cell 81 (code, score=16)
```python
# ─────────────────────────────────────────────────────────────────────────────
# Pristine Autoregressive Data Collation
# ─────────────────────────────────────────────────────────────────────────────
MAX_AUDIO_SEC  = 20
MAX_TGT_TOKENS = 128

def collate_s2t_batch(samples):
    valid = [s for s in samples if len(s['wav']) / 16000 <= MAX_AUDIO_SEC]
    if not valid: return None

    wavs      = [s['wav']      for s in valid]
    tgt_refs  = [s['ref']      for s in valid]
    tgt_langs = [s['tgt_lang'] for s in valid]

    feat_out = processor(audio=wavs, sampling_rate=16000, return_tensors='pt', padding=True)

    # Extract text contents directly without automatic tokenizer special tokens
    enc_full = processor.tokenizer(
        tgt_refs, padding=True, truncation=True,
        max_length=MAX_TGT_TOKENS, return_tensors='pt',
        add_special_tokens=False
    )
    content_ids_full = enc_full['input_ids']
    content_mask     = enc_full['attention_mask']
    B, T = content_ids_full.shape

    # Construct clean prefix components
    bos_full      = torch.full((B, 1), BOS_ID_FULL, dtype=torch.long)
    eos_full      = torch.full((B, 1), EOS_ID_FULL, dtype=torch.long)
    tgt_lang_full = torch.tensor([_tok.convert_tokens_to_ids(f'__{lg}__') for lg in tgt_langs], dtype=torch.long).unsqueeze(1)

    # 1. Master Teacher Context sequence Layout
    dec_full = torch.cat([bos_full, tgt_lang_full, content_ids_full], dim=1)

    # 2. Student Context sequence Layout
    content_ids_s = remap_ids(content_ids_full)
    bos_s         = torch.full((B, 1), BOS_ID_S, dtype=torch.long)
    tgt_lang_s    = torch.tensor([_lang_token_id_student(lg) for lg in tgt_langs], dtype=torch.long).unsqueeze(1)
    
    dec_s = torch.cat([bos_s, tgt_lang_s, content_ids_s], dim=1).clamp(0, S_VOCAB - 1)

    # 3. Target Label alignment Layout
    eos_s       = torch.full((B, 1), EOS_ID_S, dtype=torch.long)
    labels_full = torch.cat([tgt_lang_s, content_ids_s, eos_s], dim=1)

    # Construct padding mask: target language token and EOS token must never be masked
    prefix_mask = torch.ones(B, 1, dtype=torch.long)
    suffix_mask = torch.ones(B, 1, dtype=torch.long)
    labels_mask = torch.cat([prefix_mask, content_mask, suffix_mask], dim=1)

    labels_s = labels_full.clone()
    labels_s[labels_mask == 0] = -100
    labels_s[labels_s >= 0] = labels_s[labels_s >= 0].clamp(0, S_VOCAB - 1)

    return dict(
        feat=feat_out,
        dec_s=dec_s,          # Shape: [B, T + 2]
        dec_full=dec_full,    # Shape: [B, T + 2]
        labels_s=labels_s,    # Shape: [B, T + 2]
        tgt_langs=tgt_langs
    )

print("✓ Collation workflow successfully synchronized.")
```
OUTPUT:
```text
✓ Collation workflow successfully synchronized.
```

### Cell 82 (code, score=6)
```python
# batch = collate_s2t_batch([ft_samples[i] for i in range(2)])

# print('Alignment check (dec_s[i] should predict labels_s[i]):')
# b = 0
# dec   = batch['dec_s'][b].tolist()
# labs  = batch['labels_s'][b].tolist()

# for i in range(min(10, len(dec))):
#     d_full = remap_to_old[dec[i]].item() if dec[i] < len(remap_to_old) else 0
#     d_tok  = processor.tokenizer.convert_ids_to_tokens([d_full])
#     if labs[i] == -100:
#         l_tok = 'PAD'
#     else:
#         l_full = remap_to_old[labs[i]].item() if labs[i] < len(remap_to_old) else 0
#         l_tok  = processor.tokenizer.convert_ids_to_tokens([l_full])
#     print(f'  [{i}] dec={dec[i]:5d} {str(d_tok):20s} → label={labs[i]:5d} {str(l_tok)}')

# # Expected output:
# # [0] dec=BOS        → label=__ben__
# # [1] dec=__ben__    → label=__eng__  (or src_lang)
# # [2] dec=__eng__    → label=▁প
# # [3] dec=▁প         → label=াইল
```

### Cell 84 (code, score=5)
```python
# ─────────────────────────────────────────────────────────────────────────────
# Cell C — Forward helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _to_dev(batch_dict, device, dtype=torch.float16):
    return {
        k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
           if isinstance(v, torch.Tensor) else v
        for k, v in batch_dict.items()
    }
 
@torch.no_grad()
def teacher_logits_cpu(feat_cpu, dec_full_cpu):
    b1     = _to_dev(feat_cpu, 'cuda:1')
    dec_t1 = dec_full_cpu.to('cuda:1')
    with torch.cuda.amp.autocast(dtype=torch.float16):
        out = teacher(
            input_features    = b1['input_features'],
            attention_mask    = b1.get('attention_mask'),
            decoder_input_ids = dec_t1,
        )
    return out.logits.float().cpu()   # [B, T, T_VOCAB]  fp32  CPU
 
def student_logits_gpu(feat_cuda0, dec_s_cpu):
    dec_s = dec_s_cpu.clamp(0, S_VOCAB - 1).to('cuda:0')
    feat0 = _to_dev(feat_cuda0, 'cuda:0')
    with torch.cuda.amp.autocast(dtype=torch.float16):
        out = student(
            input_features    = feat0['input_features'],
            attention_mask    = feat0.get('attention_mask'),
            decoder_input_ids = dec_s,
        )
    return out.logits   # [B, T, S_VOCAB]  fp16  cuda:0
 
print('✓ Forward helpers ready.')
```
OUTPUT:
```text
✓ Forward helpers ready.
```

### Cell 85 (code, score=97)
```python
# ─────────────────────────────────────────────────────────────────────────────
# Cell D — Loss functions
# ─────────────────────────────────────────────────────────────────────────────
 
# Adjust these before calling run_phase6_training()
KD_ALPHA       = 0.15   # was 0.3 till step 960 # was 0.5 — let CE lead, KD assist
KD_TEMPERATURE = 3.0   # was 4.0 — slightly sharper teacher targets
TOP_K_LOGITS   = 256   # was 512 — focus on highest-mass mapped tokens
 
def _label_smoothed_nll(logits, targets, smoothing=0.1, ignore_index=-100):
    """
    logits  : [N, V]   (already contiguous — caller must ensure this)
    targets : [N]
    """
    V    = logits.size(-1)
    mask = targets != ignore_index
    if not mask.any():
        return logits.sum() * 0.0
 
    with torch.no_grad():
        smooth = torch.full_like(logits, smoothing / (V - 1))
        smooth.scatter_(-1, targets.clamp(min=0).unsqueeze(-1), 1.0 - smoothing)
 
    log_p = F.log_softmax(logits, dim=-1)
    loss  = -(smooth * log_p).sum(-1)
    return loss[mask].mean()
 
 
# ── Rebuild OLD_TO_NEW_CPU with -1 sentinel (not UNK_ID=3) ──────────────────
# This is the only change needed from Cell 36

UNMAPPED_SENTINEL = -1

old_to_new_clean = torch.full((T_VOCAB,), UNMAPPED_SENTINEL, dtype=torch.long)
for new_id, old_id in enumerate(remap_to_old.tolist()):
    if 0 <= old_id < T_VOCAB:
        old_to_new_clean[old_id] = new_id

# Token 3 (</s>) should now correctly map to 3, not sentinel
assert old_to_new_clean[3].item() == 3, "EOS should map correctly"
assert old_to_new_clean[698].item() == -1, "▁? should be unmapped"

OLD_TO_NEW_CPU = old_to_new_clean

n_mapped = (OLD_TO_NEW_CPU >= 0).sum().item()
covered  = OLD_TO_NEW_CPU[3].item()  # EOS check
print(f'Mapped: {n_mapped}/{T_VOCAB} ({100*n_mapped/T_VOCAB:.1f}%)')
print(f'EOS (token 3) maps to student: {covered}  ← should be 3')


def remap_ids(ids_cpu: torch.Tensor) -> torch.Tensor:
    """Remap full-vocab IDs to student vocab. Unmapped → UNK_ID_STUDENT."""
    flat    = ids_cpu.reshape(-1).clamp(0, T_VOCAB - 1)
    remapped = OLD_TO_NEW_CPU[flat].reshape(ids_cpu.shape)
    # Replace sentinel with UNK for label tensors
    remapped[remapped < 0] = UNK_ID_STUDENT
    return remapped


def sparse_kl_loss(s_logits, t_logits_cpu, T=KD_TEMPERATURE, k=TOP_K_LOGITS):
    """
    s_logits     : [B, L, S_VOCAB]  fp16  cuda:0
    t_logits_cpu : [B, L, T_VOCAB]  fp32  CPU
    Skips unmapped teacher tokens cleanly.
    """
    V_s = s_logits.shape[-1]
    B, L, _ = s_logits.shape

    # Teacher top-k in full vocab
    t_soft = F.softmax(t_logits_cpu.float() / T, dim=-1)        # [B,L,T_VOCAB] CPU
    kk = min(k, t_logits_cpu.shape[-1])
    topk_vals, topk_idx_full = torch.topk(t_soft, kk, dim=-1)   # [B,L,k] CPU

    # Remap teacher indices → student vocab (-1 = unmapped)
    flat             = topk_idx_full.reshape(-1).clamp(0, T_VOCAB - 1)
    flat_s           = OLD_TO_NEW_CPU[flat]                       # [B*L*k]
    topk_idx_student = flat_s.reshape(B, L, kk)                  # [B,L,k]

    # Valid mask: mapped tokens only
    valid = (topk_idx_student >= 0)                               # [B,L,k] bool CPU

    # Zero-out unmapped teacher probs, renormalize
    topk_vals_m = topk_vals * valid.float()
    denom       = topk_vals_m.sum(-1, keepdim=True).clamp(min=1e-9)
    topk_t      = topk_vals_m / denom                            # [B,L,k] CPU

    # Clamp indices for gather (unmapped slots → 0, masked out anyway)
    topk_idx_clamped = topk_idx_student.clamp(min=0, max=V_s - 1)

    # Move to GPU
    idx_gpu   = topk_idx_clamped.to(s_logits.device)
    topk_t_gpu = topk_t.to(s_logits.device, dtype=torch.float32)
    valid_gpu  = valid.to(s_logits.device)

    # Gather + mask + log_softmax
    s_f      = s_logits.float().contiguous()
    gathered = s_f.gather(-1, idx_gpu)                           # [B,L,k]
    gathered = gathered.masked_fill(~valid_gpu, -1e9)            # mask unmapped
    s_log    = F.log_softmax(gathered / T, dim=-1)               # [B,L,k]

    # Only compute KL at positions where ≥1 teacher token mapped
    has_valid = valid_gpu.any(-1)                                 # [B,L]
    if not has_valid.any():
        return s_logits.sum() * 0.0

    kl = F.kl_div(
        s_log[has_valid],
        topk_t_gpu[has_valid],
        reduction='batchmean'
    ) * (T ** 2)

    return kl


print('✓ sparse_kl_loss fixed with clean sentinel handling')
print('✓ Ready to train')
 
 
# Redefine loss with new constants
def compute_recovery_loss(s_logits, labels_dev, t_logits_cpu,
                          alpha=KD_ALPHA, smoothing=0.1):
    s_flat = s_logits.contiguous().reshape(-1, s_logits.size(-1))
    l_flat = labels_dev.contiguous().reshape(-1)
    ce     = _label_smoothed_nll(s_flat, l_flat, smoothing=smoothing)
    kd     = sparse_kl_loss(s_logits, t_logits_cpu)
    return (1.0 - alpha) * ce + alpha * kd, ce.item(), kd.item()

print('✓ Hyperparams adjusted. Ready to train.')
print('✓ Loss functions ready.')
```
OUTPUT:
```text
Mapped: 22767/256102 (8.9%)
EOS (token 3) maps to student: 3  ← should be 3
✓ sparse_kl_loss fixed with clean sentinel handling
✓ Ready to train
✓ Hyperparams adjusted. Ready to train.
✓ Loss functions ready.
```

### Cell 86 (code, score=18)
```python
# def set_kd_alpha(new_alpha):
#     global KD_ALPHA
#     KD_ALPHA = new_alpha
    
#     def compute_recovery_loss(s_logits, labels_dev, t_logits_cpu,
#                               alpha=KD_ALPHA, smoothing=0.1):
#         s_flat = s_logits.contiguous().reshape(-1, s_logits.size(-1))
#         l_flat = labels_dev.contiguous().reshape(-1)
#         ce     = _label_smoothed_nll(s_flat, l_flat, smoothing=smoothing)
#         if alpha == 0.0:
#             return ce, ce.item(), 0.0
#         kd     = sparse_kl_loss(s_logits, t_logits_cpu)
#         return (1.0 - alpha) * ce + alpha * kd, ce.item(), kd.item()
    
#     import builtins
#     globals()['compute_recovery_loss'] = compute_recovery_loss
#     print(f'KD_ALPHA set to {new_alpha} — CE weight={(1-new_alpha):.0%}')

# # Use at appropriate steps:
# # set_kd_alpha(0.05)  ← run at step ~1400
# # set_kd_alpha(0.0)   ← run at step ~1600
```

### Cell 87 (code, score=44)
```python
# ─────────────────────────────────────────────────────────────────────────────
# Cell E — Optimizer + scheduler
# ─────────────────────────────────────────────────────────────────────────────
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
 
BATCH_SIZE      = 4
GRAD_ACCUM      = 8
LR_PEAK         = 1.5e-4 #was 9e-5
WEIGHT_DECAY    = 1e-2
MAX_EPOCHS      = 5
EVAL_STEPS      = 100
LOG_STEPS       = 10
WARMUP_FRACTION = 0.01
 
N_TRAIN         = len(ft_samples)
STEPS_PER_EPOCH = math.ceil(N_TRAIN / (BATCH_SIZE * GRAD_ACCUM))
TOTAL_STEPS     = STEPS_PER_EPOCH * MAX_EPOCHS
 
trainable_params = [p for p in student.parameters() if p.requires_grad]
print(f'Trainable: {sum(p.numel() for p in trainable_params)/1e6:.2f} M  |  '
      f'Total steps: {TOTAL_STEPS}')
 
optimizer = AdamW(trainable_params, lr=LR_PEAK, weight_decay=WEIGHT_DECAY,
                  betas=(0.9, 0.98), eps=1e-6)
scheduler = OneCycleLR(optimizer, max_lr=LR_PEAK, total_steps=TOTAL_STEPS,
                       pct_start=WARMUP_FRACTION, anneal_strategy='cos',
                       div_factor=25.0, final_div_factor=1e4)
scaler    = torch.cuda.amp.GradScaler()
 
print('✓ Optimizer ready.')
```
OUTPUT:
```text
Trainable: 20.58 M  |  Total steps: 1500
✓ Optimizer ready.
```

### Cell 88 (code, score=60)
```python
# ─────────────────────────────────────────────────────────────────────────────
# Cell F — Quick eval helper
# ─────────────────────────────────────────────────────────────────────────────

ASR_EVAL_EXCLUDE_LANGS     = {'cmn'}

def quick_eval_chrf_fixed(mdl, samples, max_samples=32, group_size=25):
    text_scores, asr_scores = [], []
    num_langs = len(samples) // group_size
    per_lang  = max(1, max_samples // num_langs)

    # ── Use base model for generation, not PeftModel wrapper ─────────────────
    base_mdl = mdl.base_model.model if hasattr(mdl, 'base_model') else mdl

    for i in range(num_langs):
        for j in range(per_lang):
            idx = i * group_size + j
            if idx >= len(samples):
                break
            s   = samples[idx]
            tgt = s.get('tgt_lang', 'ben')
            if tgt in ASR_EVAL_EXCLUDE_LANGS:
                continue
            try:
                text_pred, wav_out = run_s2st(base_mdl, s['wav'], tgt_lang=tgt)
                asr_pred = asr_transcribe(wav_out, tgt)
                text_scores.append(compute_chrf(text_pred, s['ref']))
                asr_scores.append(compute_chrf(asr_pred,  s['ref']))
            except RuntimeError as e:
                if 'Kernel size' in str(e) or 'padded input size' in str(e):
                    # Short sequence degenerate case — skip silently
                    continue
                else:
                    raise

    return (float(np.mean(text_scores)) if text_scores else 0.0,
            float(np.mean(asr_scores))  if asr_scores  else 0.0)


def _eval_quick(n_samples=32):
    student.eval()
    try:
        text_chrf, asr_chrf = quick_eval_chrf_fixed(
            student, eval_samples, max_samples=n_samples)
    except Exception as e:
        print(f'  [eval_quick error] {e}')
        text_chrf, asr_chrf = 0.0, 0.0
    student.train()
    return text_chrf, asr_chrf

print('✓ _eval_quick fixed — uses base_model.model + catches short-seq errors')
```
OUTPUT:
```text
✓ _eval_quick fixed — uses base_model.model + catches short-seq errors
```

### Cell 89 (code, score=3)
```python
# # Quick sanity check
# batch = collate_s2t_batch([ft_samples[i] for i in range(4)])
# print("Labels non-masked:", (batch['labels_s'] != -100).sum())
# print("Dec_s range:", batch['dec_s'].min(), batch['dec_s'].max(), "vs S_VOCAB:", S_VOCAB)
# print("Labels range:", batch['labels_s'][batch['labels_s']!=-100].unique()[:10])
```

### Cell 90 (code, score=36)
```python
!ls checkpoints
```
OUTPUT:
```text
all_detailed_summaries_step000000.pt  phase3_laco_done_step000000.pt
all_summaries_step000000.pt	      phase4_benchmark_step000000.pt
phase0_benchmark_step000000.pt	      phase4_enc_pruning_step000000.pt
phase1_benchmark_step000000.pt	      phase5_benchmark_step000000.pt
phase2_benchmark_step000000.pt	      phase5_dec_pruning_step000000.pt
phase2_enc_pruning_step000000.pt      phase6_benchmark_step000000.pt
phase3_benchmark_step000000.pt	      phase7_benchmark_step000000.pt
```

### Cell 91 (code, score=20)
```python
# !rm -rf checkpoints/phase6_kd_step000080.pt
# !rm -rf checkpoints/phase6_kd_step000160.pt
# !rm -rf checkpoints/phase6_kd_step000240.pt
# !rm -rf checkpoints/phase6_kd_step000480.pt
# !rm -rf checkpoints/phase6_kd_step001280.pt
```

### Cell 92 (code, score=175)
```python
# ─────────────────────────────────────────────────────────────────────────────
# Cell G — Training loop (scaler fix)
# ─────────────────────────────────────────────────────────────────────────────
MULTILINGUAL_BASELINE_CHRF = 38.17 # ✓ New best 38.17 @ 1120

# Modified training loop with:
# 1. Saved shuffle state
# 2. Exact resume position  
# 3. Memory fixes
# 4. Adjusted loss weights

import gc
import torch
import ctypes

def free_cpu_ram():
    """Aggressively free CPU RAM."""
    gc.collect()
    # Release Python memory back to OS (Linux only — works on Kaggle)
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass

# KD_ALPHA schedule config — define BEFORE calling run_phase6_training()
# KD_ALPHA_SCHEDULE = {
#     # step: alpha
#     0:    0.30,   # initial
#     560:  0.15,   # already applied
#     1400: 0.05,   # step trigger
#     1600: 0.0,    # pure CE
# }

# OR: use linear decay (recommended)
KD_ALPHA_START      = 0.15   # current value
KD_ALPHA_END        = 0.0    # final value
KD_DECAY_START_STEP = 99999   # essetially not used any decay
KD_DECAY_END_STEP   = 99999   # when to reach zero


def _get_kd_alpha(step):
    """Linear decay of KD_ALPHA from start to end over decay window."""
    if step < KD_DECAY_START_STEP:
        return KD_ALPHA_START
    if step >= KD_DECAY_END_STEP:
        return KD_ALPHA_END
    progress = (step - KD_DECAY_START_STEP) / (KD_DECAY_END_STEP - KD_DECAY_START_STEP)
    return KD_ALPHA_START * (1.0 - progress)


def _apply_kd_alpha(step):
    """Update compute_recovery_loss with current KD_ALPHA for this step."""
    global KD_ALPHA
    new_alpha = _get_kd_alpha(step)
    
    # Only redefine if alpha changed meaningfully (avoid constant redefinition)
    if abs(new_alpha - KD_ALPHA) < 1e-4:
        return
    
    KD_ALPHA = new_alpha

    def compute_recovery_loss(s_logits, labels_dev, t_logits_cpu,
                              alpha=new_alpha, smoothing=0.1):
        s_flat = s_logits.contiguous().reshape(-1, s_logits.size(-1))
        l_flat = labels_dev.contiguous().reshape(-1)
        ce     = _label_smoothed_nll(s_flat, l_flat, smoothing=smoothing)
        if alpha < 1e-6:
            return ce, ce.item(), 0.0
        kd     = sparse_kl_loss(s_logits, t_logits_cpu)
        return (1.0 - alpha) * ce + alpha * kd, ce.item(), kd.item()

    globals()['compute_recovery_loss'] = compute_recovery_loss


def run_phase6_training():
    global KD_ALPHA
    best_chrf      = MULTILINGUAL_BASELINE_CHRF
    best_chrf_step = 1120
    patience_left  = 20
    opt_step       = 0
    epoch_seeds    = {}

    ckpt = load_latest_checkpoint('phase6_kd')
    if ckpt:
        try:
            student.load_state_dict(ckpt['student_state'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            opt_step       = ckpt.get('opt_step', 0)
            best_chrf      = ckpt.get('best_chrf', MULTILINGUAL_BASELINE_CHRF)
            best_chrf_step = ckpt.get('best_chrf_step', 0)
            epoch_seeds    = ckpt.get('epoch_seeds', {})
            print(f'[resume] step={opt_step}  best={best_chrf:.2f}')
        except Exception as e:
            print(f'[resume] fresh start ({e})')
        finally:
            del ckpt
            free_cpu_ram()

    # Apply correct KD_ALPHA for resume step immediately
    # _apply_kd_alpha(opt_step)
    print(f'[resume] KD_ALPHA={KD_ALPHA:.4f} at step {opt_step}')

    start_epoch               = opt_step // STEPS_PER_EPOCH
    steps_done_in_start_epoch = opt_step % STEPS_PER_EPOCH
    batches_to_skip           = steps_done_in_start_epoch * GRAD_ACCUM

    print(f'[resume] epoch={start_epoch} '
          f'steps_into_epoch={steps_done_in_start_epoch} '
          f'batches_to_skip={batches_to_skip}')
    print(f'[resume] Remaining steps: {TOTAL_STEPS - opt_step}')
    print(f'[resume] KD decay: {KD_ALPHA_START}→{KD_ALPHA_END} '
          f'over steps {KD_DECAY_START_STEP}→{KD_DECAY_END_STEP}')

    print(f'\n{"="*65}')
    print(f'  PHASE 6 — Exact resume from step {opt_step}')
    print(f'{"="*65}\n')

    for epoch in range(start_epoch, MAX_EPOCHS):
        ep_ce = ep_kd = ep_n = 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0

        if epoch not in epoch_seeds:
            epoch_seeds[epoch] = random.randint(0, 2**31)
        seed = epoch_seeds[epoch]
        random.seed(seed)
        all_idx = chunk_friendly_shuffle(len(ft_samples), CHUNK_SIZE, BATCH_SIZE)
        random.seed(42)

        print(f'  Epoch {epoch+1} | seed={seed} | KD_ALPHA={KD_ALPHA:.4f}')

        for batch_idx, batch_start in enumerate(
                range(0, len(all_idx), BATCH_SIZE)):

            if epoch == start_epoch and batch_idx < batches_to_skip:
                continue

            if opt_step >= TOTAL_STEPS:
                break
```

### Cell 93 (code, score=23)
```python
# # ─────────────────────────────────────────────────────────────────────────────
# # Cell H — Merge LoRA + save
# # ─────────────────────────────────────────────────────────────────────────────
 
# print('\nMerging LoRA adapters …')
# best_ckpt = load_latest_checkpoint('phase6_kd')
# if best_ckpt and 'student_state' in best_ckpt:
#     student.load_state_dict(best_ckpt['student_state'], strict=False)
#     print(f"  Best weights restored (step {best_ckpt.get('opt_step','?')})")
 
# student.eval()
# merged = student.merge_and_unload()
# merged.eval()
# print(f'  Merged params: {count_params(merged):.1f} M')
 
# save_model_to_drive(merged, processor, 'phase6_kd_merged',
#                     manifest_extra={'strategy': 'online_kd_lora',
#                                     'best_chrf': final_chrf})
# print('✓ Saved phase6_kd_merged')
```

### Cell 94 (code, score=139)
```python
# Full benchmark with checkpoint loading
p6_bench = load_latest_checkpoint('phase6_benchmark')
if p6_bench and p6_bench.get('summary', {}).get('avg_bleu', 0) > 0:
    p6_results = p6_bench['results']
    p6_summary = p6_bench['summary']
    p6_detailed = p6_bench.get('detailed_summary')
    print('Loaded Phase 6 benchmark results from checkpoint.')
    # Recompute detailed if missing
    if not p6_detailed:
        p6_detailed = compute_detailed_summary(p6_results, 'P6_KD_Merged', p6_summary['params_M'])
else:
    merged_clean, _ = load_model_from_drive('phase6_kd_merged', device_map='cuda:0')
    merged_clean.eval()
    p6_results, p6_summary = run_benchmark_asr(merged_clean, list(eval_samples), 'P6_KD_Merged', save_n=4)
    p6_detailed = compute_detailed_summary(p6_results, 'P6_KD_Merged', p6_summary['params_M'])
    save_checkpoint(dict(results=p6_results, summary=p6_summary,
                         detailed_summary=p6_detailed), 'phase6_benchmark', 0)

store_summary(p6_summary)
store_detailed_summary(p6_detailed)
print_detailed_summary_table('P6_KD_Merged')
plot_detailed_phase_comparison()
```
OUTPUT:
```text
[ckpt] Loaded phase6_benchmark_step000000.pt
Loaded Phase 6 benchmark results from checkpoint.
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P6_KD_Merged (7 total)
[ckpt] Saved all_detailed_summaries_step000000.pt (0.0 MB)
[detailed] Stored P6_KD_Merged

================================================================================
  P6_KD_Merged - 1030.9M params
================================================================================
Overall: ChrF=33.07±15.97  BLEU=7.95  RTF=0.1484

Per-Pair (8 pairs):
  Pair               N     ChrF     BLEU      RTF
  arb→eng           24    39.74     9.41   0.1073
  ben→eng           25    31.45     4.76   0.1225
  cmn→eng           25    34.78     7.77   0.1070
  eng→arb           25    36.43     8.94   0.1014
  eng→ben           25    40.10     8.91   0.1239
  eng→cmn           25     2.41     1.30   0.3948
  eng→hin           25    42.02    14.25   0.1266
  hin→eng           25    37.92     8.31   0.1024

By Source Language:
     ARB: ChrF= 39.74  BLEU=  9.41  (n=24)
     BEN: ChrF= 31.45  BLEU=  4.76  (n=25)
     CMN: ChrF= 34.78  BLEU=  7.77  (n=25)
     ENG: ChrF= 30.24  BLEU=  8.35  (n=100)
     HIN: ChrF= 37.92  BLEU=  8.31  (n=25)

By Target Language:
     ARB: ChrF= 36.43  BLEU=  8.94  (n=25)
     BEN: ChrF= 40.10  BLEU=  8.91  (n=25)
     CMN: ChrF=  2.41  BLEU=  1.30  (n=25)
     ENG: ChrF= 35.93  BLEU=  7.54  (n=99)
     HIN: ChrF= 42.02  BLEU= 14.25  (n=25)
================================================================================
Plotting detailed comparison for 7 phases: ['P0_V1_Baseline', 'P1_Vocab5L', 'P2_Enc16L', 'P3_LaCoT2U', 'P4_Enc14L', 'P5_Dec14L', 'P6_KD_Merged']

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_01_overall_quality.png  [Overall Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_02_chrf_by_pair.png  [ChrF by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_03_bleu_by_pair.png  [BLEU by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_04_src_lang_trends.png  [Source Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_05_tgt_lang_trends.png  [Target Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_06_size_vs_quality.png  [Size vs Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_07_inference_rtf.png  [Inference Speed RTF]

<Figure size 1800x963 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_08_summary_table.png  [Summary Table]

✅ All 8 figures saved individually:
   📄 detailed_comparison_01_overall_quality.png
   📄 detailed_comparison_02_chrf_by_pair.png
   📄 detailed_comparison_03_bleu_by_pair.png
   📄 detailed_comparison_04_src_lang_trends.png
   📄 detailed_comparison_05_tgt_lang_trends.png
   📄 detailed_comparison_06_size_vs_quality.png
   📄 detailed_comparison_07_inference_rtf.png
   📄 detailed_comparison_08_summary_table.png
```

### Cell 95 (code, score=3)
```python
import torch
import gc

def cleanup():
    # 1. Clear references if they exist in global scope
    # (Adjust variable names as needed)
    global student, merged, merged_clean
    
    if 'student' in globals(): del student
    if 'merged' in globals(): del merged
    if 'merged_clean' in globals(): del merged_clean
    
    # 2. Python garbage collection
    gc.collect()
    
    # 3. PyTorch specific VRAM clearing
    torch.cuda.empty_cache()
    
    # 4. (Optional) Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

cleanup()
```

### Cell 96 (code, score=15)
```python
# ─────────────────────────────────────────────────────────────────────────────
# FORWARD HELPERS — FULLY GPU, NO CPU MIDDLEMAN
# Teacher runs on cuda:1, top-K selected on cuda:1,
# transferred directly cuda:1 → cuda:0 via PCIe
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_TEACHER  = 256
KD_TEMPERATURE = 3.0

def _to_dev(batch_dict, device, dtype=torch.float16):
    return {
        k: v.to(device=device,
                dtype=dtype if v.is_floating_point() else v.dtype)
           if isinstance(v, torch.Tensor) else v
        for k, v in batch_dict.items()
    }


@torch.no_grad()
def teacher_topk_direct(feat_cpu, dec_full_cpu, k=TOP_K_TEACHER, T=KD_TEMPERATURE):
    """
    Teacher forward on cuda:1.
    Top-K selection on cuda:1.
    Direct cuda:1 → cuda:0 transfer (no CPU involved).

    Returns:
        topk_vals: [B, T, k]  fp32  cuda:0  (softmax probabilities at temp T)
        topk_idx:  [B, T, k]  long  cuda:0  (indices in T_VOCAB space)
    """
    # Load batch to cuda:1
    feat1 = _to_dev(feat_cpu, 'cuda:1')
    dec1  = dec_full_cpu.to('cuda:1')

    with torch.cuda.amp.autocast(dtype=torch.float16):
        out = teacher(
            input_features    = feat1['input_features'],
            attention_mask    = feat1.get('attention_mask'),
            decoder_input_ids = dec1,
        )

    # All operations on cuda:1 — no CPU touch
    logits_1 = out.logits.float()                          # [B, T, T_VOCAB] fp32 cuda:1
    probs_1  = torch.softmax(logits_1 / T, dim=-1)         # [B, T, T_VOCAB] fp32 cuda:1
    topk_vals_1, topk_idx_1 = torch.topk(
        probs_1,
        k=min(k, probs_1.shape[-1]),
        dim=-1,
        sorted=False,
    )                                                       # [B, T, k] on cuda:1

    # Direct GPU-to-GPU transfer: cuda:1 → cuda:0
    # PyTorch handles this via PCIe without CPU staging
    topk_vals_0 = topk_vals_1.to('cuda:0')                 # [B, T, k] fp32 cuda:0
    topk_idx_0  = topk_idx_1.to('cuda:0')                  # [B, T, k] long cuda:0

    del feat1, dec1, out, logits_1, probs_1, topk_vals_1, topk_idx_1
    torch.cuda.synchronize('cuda:1')

    return topk_vals_0, topk_idx_0




print('✓ Fully GPU forward helpers ready.')
print(f'  teacher_topk_direct: cuda:1 → top-{TOP_K_TEACHER} → direct to cuda:0')
print(f'  No CPU involved in teacher→student logit transfer')
```
OUTPUT:
```text
✓ Fully GPU forward helpers ready.
  teacher_topk_direct: cuda:1 → top-256 → direct to cuda:0
  No CPU involved in teacher→student logit transfer
```

### Cell 97 (code, score=79)
```python
# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS — all tensors already on cuda:0, no CPU remapping needed
# ─────────────────────────────────────────────────────────────────────────────

KD_ALPHA = 0.15
OLD_TO_NEW_GPU = OLD_TO_NEW_CPU.to('cuda:0')   # remap table on GPU
# Shape: [T_VOCAB] long cuda:0
# Built once, reused every batch — ~1MB VRAM

def _label_smoothed_nll(logits, targets, smoothing=0.1, ignore_index=-100):
    """logits: [N, V] fp32 cuda:0,  targets: [N] long cuda:0"""
    V    = logits.size(-1)
    mask = targets != ignore_index
    if not mask.any():
        return logits.sum() * 0.0
    with torch.no_grad():
        smooth = torch.full_like(logits, smoothing / (V - 1))
        smooth.scatter_(-1, targets.clamp(min=0).unsqueeze(-1), 1.0 - smoothing)
    log_p = torch.nn.functional.log_softmax(logits, dim=-1)
    loss  = -(smooth * log_p).sum(-1)
    return loss[mask].mean()


def sparse_kl_from_topk_gpu(s_logits, topk_vals_0, topk_idx_full_0,
                              T=KD_TEMPERATURE):
    """
    KD loss — everything on cuda:0, no CPU involved.

    s_logits:        [B, L, S_VOCAB]  fp32  cuda:0
    topk_vals_0:     [B, L, k]        fp32  cuda:0  (teacher probs at temp T)
    topk_idx_full_0: [B, L, k]        long  cuda:0  (T_VOCAB indices)
    """
    B, L, V_s = s_logits.shape
    k = topk_vals_0.shape[-1]

    # Remap T_VOCAB indices → S_VOCAB using GPU lookup table
    flat_full  = topk_idx_full_0.reshape(-1).clamp(0, T_VOCAB - 1)
    flat_s     = OLD_TO_NEW_GPU[flat_full]             # [B*L*k] long cuda:0
    topk_idx_s = flat_s.reshape(B, L, k)              # [B, L, k] cuda:0

    # Valid: mapped tokens only (sentinel = -1)
    valid = topk_idx_s >= 0                            # [B, L, k] bool cuda:0

    # Renormalize teacher distribution over mapped tokens only
    vals_masked = topk_vals_0 * valid.float()
    denom       = vals_masked.sum(-1, keepdim=True).clamp(min=1e-9)
    topk_t      = vals_masked / denom                  # [B, L, k] cuda:0

    # Gather student logits at teacher positions
    idx_clamped = topk_idx_s.clamp(min=0, max=V_s - 1)
    gathered    = s_logits.gather(-1, idx_clamped)     # [B, L, k] cuda:0
    gathered    = gathered.masked_fill(~valid, -1e9)

    # Student log-probs (temp-scaled)
    s_log = torch.nn.functional.log_softmax(
        gathered / T, dim=-1)                          # [B, L, k] cuda:0

    # Only positions with at least one valid mapped token
    has_valid = valid.any(-1)                          # [B, L] cuda:0
    if not has_valid.any():
        return s_logits.sum() * 0.0

    kl = torch.nn.functional.kl_div(
        s_log[has_valid],
        topk_t[has_valid],
        reduction='batchmean',
    ) * (T ** 2)

    return kl


def compute_recovery_loss_gpu(s_logits, labels_dev,
                               topk_vals_0, topk_idx_0,
                               alpha=KD_ALPHA, smoothing=0.1):
    """
    Full loss — all tensors on cuda:0, no CPU ops.
    s_logits:     [B, L, S_VOCAB]  fp32  cuda:0
    labels_dev:   [B, L]           long  cuda:0
    topk_vals_0:  [B, L, k]        fp32  cuda:0
    topk_idx_0:   [B, L, k]        long  cuda:0
    """
    s_flat = s_logits.contiguous().reshape(-1, s_logits.size(-1))
    l_flat = labels_dev.contiguous().reshape(-1)
    ce     = _label_smoothed_nll(s_flat, l_flat, smoothing=smoothing)
    kd     = sparse_kl_from_topk_gpu(s_logits, topk_vals_0, topk_idx_0)
    return (1.0 - alpha) * ce + alpha * kd, ce.item(), kd.item()


print('✓ GPU-only loss functions ready.')
print(f'  OLD_TO_NEW_GPU: remap table on cuda:0 '
      f'({OLD_TO_NEW_GPU.numel()*4/1e6:.1f}MB VRAM)')
print('  No CPU involved in any loss computation')
```
OUTPUT:
```text
✓ GPU-only loss functions ready.
  OLD_TO_NEW_GPU: remap table on cuda:0 (1.0MB VRAM)
  No CPU involved in any loss computation
```

### Cell 98 (code, score=273)
```python
# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — Option C: Hybrid LoRA — SPEED FIXED VERSION
# ─────────────────────────────────────────────────────────────────────────────

import gc, math, os, random, time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from peft import get_peft_model, LoraConfig, TaskType

# ── 0. Load directly to cuda:0 — skip CPU staging ────────────────────────────
# device_map='cuda:0' loads directly onto GPU, no PCIe round trip
print('Loading phase6_kd_merged directly to cuda:0...')
model_p7, processor = load_model_from_drive(
    'phase6_kd_merged', device_map='cuda:0')
model_p7 = model_p7.to(torch.float16)
print(f'Loaded. Params: {count_params(model_p7):.1f}M')
gpu_mem()

# ── 1. KILL gradient checkpointing immediately after load ─────────────────────
# It may have been saved into config during Phase 6.
# Must be disabled BEFORE get_peft_model wraps the model.

def disable_all_gradient_checkpointing(model):
    killed = []

    # HuggingFace API
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        killed.append('model.gradient_checkpointing_disable()')

    # Config flags
    if hasattr(model, 'config'):
        if getattr(model.config, 'gradient_checkpointing', False):
            model.config.gradient_checkpointing = False
            killed.append('model.config.gradient_checkpointing')

    # Every submodule attribute
    for name, module in model.named_modules():
        if getattr(module, 'gradient_checkpointing', False):
            module.gradient_checkpointing = False
            killed.append(f'{name}.gradient_checkpointing')
        if hasattr(module, 'config'):
            if getattr(module.config, 'gradient_checkpointing', False):
                module.config.gradient_checkpointing = False
                killed.append(f'{name}.config.gradient_checkpointing')

    # Direct known locations
    for attr_path in [
        'text_decoder',
        'speech_encoder',
        'speech_encoder.encoder',
        't2u_model',
        't2u_model.model',
        't2u_model.model.encoder',
        't2u_model.model.decoder',
    ]:
        try:
            obj = model
            for part in attr_path.split('.'):
                obj = getattr(obj, part)
            if getattr(obj, 'gradient_checkpointing', False):
                obj.gradient_checkpointing = False
                killed.append(f'{attr_path}.gradient_checkpointing direct')
        except AttributeError:
            pass

    if killed:
        print(f'✓ Killed gradient checkpointing at {len(killed)} locations:')
        for k in killed:
            print(f'    {k}')
    else:
        print('✓ No gradient checkpointing found (clean)')
    return killed

killed = disable_all_gradient_checkpointing(model_p7)

# Verify
still_on = []
for name, module in model_p7.named_modules():
    if getattr(module, 'gradient_checkpointing', False):
        still_on.append(name)
if still_on:
    print(f'⚠ Still on after disable: {still_on}')
    for name, module in model_p7.named_modules():
        if name in still_on:
            module.gradient_checkpointing = False
    print('  Force-killed remaining')
else:
    print('✓ Confirmed: gradient checkpointing fully OFF')

# ── 2. Verify key shapes ──────────────────────────────────────────────────────
assert model_p7.shared.num_embeddings == 22767
assert model_p7.lm_head.out_features == 22767
assert len(model_p7.text_decoder.layers) == 14
assert len(model_p7.t2u_model.model.encoder.layers) == 4
print('✓ Architecture assertions passed')

# ── 3. LoRA config ────────────────────────────────────────────────────────────
LORA_TARGET_MODULES = [
    'k_proj', 'v_proj', 'q_proj', 'out_proj',
    'fc1', 'fc2',
]
LORA_MODULES_TO_SAVE = [
    'lm_head',
    'shared',
    't2u_model.lm_head',
]

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=LORA_TARGET_MODULES,
    modules_to_save=LORA_MODULES_TO_SAVE,
)

model_p7 = get_peft_model(model_p7, lora_config)

# Disable checkpointing again — get_peft_model can re-enable it
disable_all_gradient_checkpointing(model_p7)
disable_all_gradient_checkpointing(model_p7.base_model.model)

# ── 4. Cast trainable params to fp32 (fixes GradScaler ValueError) ────────────
n_cast = 0
for name, param in model_p7.named_parameters():
    if ('lora_A' in name or
        'lora_B' in name or
        'modules_to_save' in name):
        param.data = param.data.to(torch.float32)
        n_cast += param.numel()
print(f'✓ Cast {n_cast/1e6:.2f}M trainable params to fp32')

model_p7.train()

# ── 5. Verify wrapping ────────────────────────────────────────────────────────
print('\n=== LORA WRAPPING AUDIT ===')
```
OUTPUT:
```text
Loading phase6_kd_merged directly to cuda:0...
[model] Not in local cache — pulling from remote...
[rclone] Pulled phase6_kd_merged → /kaggle/working/models/phase6_kd_merged
[model] Loading phase6_kd_merged from /kaggle/working/models/phase6_kd_merged ...

Loading weights:   0%|          | 0/1234 [00:00<?, ?it/s]
  Restored custom state: ['_vocab_remap_to_old']
[model] Loaded phase6_kd_merged.
Loaded. Params: 1030.9M
  GPU0: 4.26GB alloc / 4.27GB reserved
  GPU1: 3.64GB alloc / 3.65GB reserved
✓ Killed gradient checkpointing at 1 locations:
    model.gradient_checkpointing_disable()
✓ Confirmed: gradient checkpointing fully OFF
✓ Architecture assertions passed
✓ Killed gradient checkpointing at 1 locations:
    model.gradient_checkpointing_disable()
✓ Killed gradient checkpointing at 1 locations:
    model.gradient_checkpointing_disable()
✓ Cast 98.11M trainable params to fp32

=== LORA WRAPPING AUDIT ===
LoRA in text_decoder:   0
LoRA in t2u_model:      0
LoRA in speech_encoder: 0  ← must be 0
LoRA in vocoder:        0  ← must be 0
modules_to_save:        3
trainable params: 98,107,392 || all params: 1,129,011,205 || trainable%: 8.6897
✓ Speech adapter unfrozen + fp32: 54.553M params
✓ All trainable params fp32
Trainable: 152.7M / 1129.0M
  GPU0: 4.76GB alloc / 4.89GB reserved
  GPU1: 3.64GB alloc / 3.65GB reserved

VRAM + speed check...
  Batch load: 25.70s
  Forward pass: 3.91s  ← should be <5s
  Backward pass: 0.25s  ← should be <10s, was 1000s before
  GPU0: 5.28GB alloc / 6.16GB reserved
  GPU1: 3.64GB alloc / 3.65GB reserved
✓ Speed check complete
```

### Cell 99 (code, score=62)
```python
# ─────────────────────────────────────────────────────────────────────────────
# SCOPE FIX: Freeze back any LoRA that leaked into vocoder or speech_encoder
# ─────────────────────────────────────────────────────────────────────────────

n_frozen_back = 0
for name, param in model_p7.named_parameters():
    if 'vocoder' in name and param.requires_grad:
        param.requires_grad_(False)
        n_frozen_back += param.numel()
        print(f'  Frozen back (vocoder leak): {name}')

if n_frozen_back == 0:
    print('✓ No vocoder leakage detected')
else:
    print(f'  Frozen back {n_frozen_back/1e6:.3f}M vocoder params')

for name, param in model_p7.named_parameters():
    if ('speech_encoder' in name and
        'lora_' in name and
        param.requires_grad and
        not any(pat in name for pat in SPEECH_UNFREEZE)):
        param.requires_grad_(False)
        print(f'  Frozen back (speech LoRA leak): {name}')

# Final confirmed count
all_trainable_params = [p for p in model_p7.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in all_trainable_params)
print(f'\nFinal trainable: {n_trainable/1e6:.1f}M params')

# Final dtype check
bad = [(n, p.dtype) for n, p in model_p7.named_parameters()
       if p.requires_grad and p.dtype != torch.float32]
if bad:
    print(f'⚠ {len(bad)} trainable params not fp32 — fixing...')
    for name, _ in bad:
        p = dict(model_p7.named_parameters())[name]
        p.data = p.data.to(torch.float32)
    print('  ✓ Fixed')
else:
    print('✓ All trainable params are fp32 — GradScaler will work correctly')
```
OUTPUT:
```text
✓ No vocoder leakage detected

Final trainable: 152.7M params
✓ All trainable params are fp32 — GradScaler will work correctly
```

### Cell 100 (code, score=108)
```python
# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE      = 8
GRAD_ACCUM      = 4
LR_PEAK         = 4e-5
WEIGHT_DECAY    = 1e-2
MAX_EPOCHS      = 6 # was 3
EVAL_STEPS      = 100
LOG_STEPS       = 5
WARMUP_FRACTION = 0.05
# KD_ALPHA        = 0.2
# CHUNK_SIZE      = 4000
S_VOCAB         = 22767

N_TRAIN         = len(ft_samples)
STEPS_PER_EPOCH = math.ceil(N_TRAIN / (BATCH_SIZE * GRAD_ACCUM))
TOTAL_STEPS     = STEPS_PER_EPOCH * MAX_EPOCHS

print(f'Effective batch size: {BATCH_SIZE * GRAD_ACCUM}')
print(f'Steps per epoch:      {STEPS_PER_EPOCH}')
print(f'Total steps:          {TOTAL_STEPS}')
print(f'Vocab size:           {S_VOCAB}')

# Confirm all trainable are fp32 before building optimizer
# (optimizer stores fp32 master weights internally — params must be fp32)
for name, param in model_p7.named_parameters():
    if param.requires_grad and param.dtype != torch.float32:
        print(f'⚠ Fixing {name}: {param.dtype} → fp32')
        param.data = param.data.to(torch.float32)

# Parameter groups with different LRs
trainable_params_grouped = [
    {
        'params': [p for n, p in model_p7.named_parameters()
                   if p.requires_grad and
                   any(x in n for x in ['lm_head', 'shared'])],
        'lr': LR_PEAK * 0.3,
        'name': 'output_layers',
    },
    {
        'params': [p for n, p in model_p7.named_parameters()
                   if p.requires_grad and
                   not any(x in n for x in ['lm_head', 'shared'])],
        'lr': LR_PEAK,
        'name': 'lora_and_adapter',
    },
]

# Verify no parameter missed or double-counted
all_trainable_ids = set(
    id(p) for p in model_p7.parameters() if p.requires_grad)
grouped_ids = set(
    id(p) for g in trainable_params_grouped for p in g['params'])
assert all_trainable_ids == grouped_ids, \
    f'Parameter mismatch: {len(all_trainable_ids)} trainable ' \
    f'but {len(grouped_ids)} in groups'

for g in trainable_params_grouped:
    n = sum(p.numel() for p in g['params'])
    print(f"  Group '{g['name']}': {n/1e6:.2f}M params @ lr={g['lr']:.1e}")

optimizer = AdamW(
    trainable_params_grouped,
    weight_decay = WEIGHT_DECAY,
    betas        = (0.9, 0.98),
    eps          = 1e-6,
)
scheduler = OneCycleLR(
    optimizer,
    max_lr           = [g['lr'] for g in trainable_params_grouped],
    total_steps      = TOTAL_STEPS,
    pct_start        = WARMUP_FRACTION,
    anneal_strategy  = 'cos',
    div_factor       = 25.0,
    final_div_factor = 1e4,
)

# GradScaler — works correctly now that trainable params are fp32
scaler = torch.cuda.amp.GradScaler()

all_trainable_params = [p for p in model_p7.parameters() if p.requires_grad]
print(f'\n✓ Phase 7 optimizer ready.')
print(f'  Total trainable: {sum(p.numel() for p in all_trainable_params)/1e6:.1f}M')
print(f'  All fp32: {all(p.dtype == torch.float32 for p in all_trainable_params)}')
```
OUTPUT:
```text
Effective batch size: 32
Steps per epoch:      300
Total steps:          1800
Vocab size:           22767
  Group 'output_layers': 56.95M params @ lr=1.2e-05
  Group 'lora_and_adapter': 95.71M params @ lr=4.0e-05

✓ Phase 7 optimizer ready.
  Total trainable: 152.7M
  All fp32: True
```

### Cell 101 (code, score=54)
```python
# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 FORWARD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def student_logits_gpu_p7(feat_dict, dec_s_cpu):
    """
    Student forward on cuda:0.
    Returns fp32 logits on cuda:0.
    """
    dec_s          = dec_s_cpu.clamp(0, S_VOCAB - 1).to('cuda:0')
    input_features = feat_dict['input_features'].to('cuda:0')
    attention_mask = feat_dict.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to('cuda:0')
    else:
        attention_mask = torch.ones(
            input_features.shape[:2], device='cuda:0')

    with torch.cuda.amp.autocast(dtype=torch.float16):
        out = model_p7(
            input_features    = input_features,
            attention_mask    = attention_mask,
            decoder_input_ids = dec_s,
        )
    return out.logits.float()   # [B, T, S_VOCAB] fp32 cuda:0


def _eval_quick_p7(n_samples=32):
    model_p7.eval()
    try:
        text_chrf, asr_chrf = quick_eval_chrf_fixed(
            model_p7, eval_samples, max_samples=n_samples)
    except Exception as e:
        print(f'  [eval error] {e}')
        text_chrf, asr_chrf = 0.0, 0.0
    finally:
        model_p7.train()
        disable_all_gradient_checkpointing(model_p7)
    return text_chrf, asr_chrf


print('✓ Phase 7 forward helpers ready.')
print(f'  student_logits_gpu_p7: returns fp32 logits [B, T, {S_VOCAB}]')
print(f'  _eval_quick_p7: uses model_p7 for generation')
```
OUTPUT:
```text
✓ Phase 7 forward helpers ready.
  student_logits_gpu_p7: returns fp32 logits [B, T, 22767]
  _eval_quick_p7: uses model_p7 for generation
```

### Cell 104 (code, score=43)
```python
# ── Run this once before calling run_phase7_training() ───────────────────────

# KD_ALPHA   = 0.15   # reduced
# MAX_EPOCHS = 6      # extend to 6

STEPS_PER_EPOCH = math.ceil(len(ft_samples) / (BATCH_SIZE * GRAD_ACCUM))
TOTAL_STEPS     = STEPS_PER_EPOCH * MAX_EPOCHS

# Rebuild scheduler — keeps optimizer Adam state, resets LR curve
# load_latest_checkpoint will load model + optimizer state but
# we skip loading scheduler_state by patching the loop
scheduler = OneCycleLR(
    optimizer,
    max_lr           = [LR_PEAK * 0.3, LR_PEAK],
    total_steps      = TOTAL_STEPS,
    pct_start        = 0.0,       # no warmup, already warm
    anneal_strategy  = 'cos',
    div_factor       = 1.0,       # start at LR_PEAK immediately
    final_div_factor = 1e3,
)

print(f'KD_ALPHA={KD_ALPHA}  MAX_EPOCHS={MAX_EPOCHS}  TOTAL_STEPS={TOTAL_STEPS}')
print(f'New scheduler created — optimizer Adam state preserved')

# Then call normally:
# final_step, final_chrf = run_phase7_training()
```
OUTPUT:
```text
KD_ALPHA=0.15  MAX_EPOCHS=6  TOTAL_STEPS=1800
New scheduler created — optimizer Adam state preserved
```

### Cell 105 (code, score=129)
```python
# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 TRAINING LOOP — FULLY GPU, NO CPU MIDDLEMAN
# ─────────────────────────────────────────────────────────────────────────────

def run_phase7_training():
    best_chrf      = 0.0
    best_chrf_step = 0
    patience_left  = 25
    opt_step       = 0
    epoch_seeds    = {}

    ckpt = load_latest_checkpoint('phase7_ft')
    if ckpt:
        try:
            model_p7.load_state_dict(ckpt['model_state'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer_state'])
            # scheduler.load_state_dict(ckpt['scheduler_state'])
            opt_step       = ckpt.get('opt_step', 0)
            best_chrf      = ckpt.get('best_chrf', 0.0)
            best_chrf_step = ckpt.get('best_chrf_step', 0)
            epoch_seeds    = ckpt.get('epoch_seeds', {})
            print(f'[resume] step={opt_step}  best={best_chrf:.2f}')
            for name, param in model_p7.named_parameters():
                if param.requires_grad and param.dtype != torch.float32:
                    param.data = param.data.to(torch.float32)
            disable_all_gradient_checkpointing(model_p7)
        except Exception as e:
            print(f'[resume failed] {e}')
        finally:
            del ckpt
            free_cpu_ram()

    start_epoch               = opt_step // STEPS_PER_EPOCH
    steps_done_in_start_epoch = opt_step  % STEPS_PER_EPOCH
    batches_to_skip           = steps_done_in_start_epoch * GRAD_ACCUM

    print(f'\n{"="*65}')
    print(f'  PHASE 7 — Hybrid LoRA, Fully GPU Pipeline')
    print(f'  Teacher: cuda:1 → top-{TOP_K_TEACHER} → direct cuda:0')
    print(f'  Student: cuda:0')
    print(f'  Loss:    all cuda:0, no CPU ops')
    print(f'  Gradient checkpointing: OFF')
    print(f'  Trainable: {sum(p.numel() for p in all_trainable_params)/1e6:.1f}M')
    print(f'  BATCH={BATCH_SIZE}  ACCUM={GRAD_ACCUM}  LR={LR_PEAK:.1e}')
    print(f'  TOTAL_STEPS={TOTAL_STEPS}')
    print(f'{"="*65}\n')

    step_times = []

    for epoch in range(start_epoch, MAX_EPOCHS):
        ep_ce = ep_kd = ep_n = 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0

        if epoch not in epoch_seeds:
            epoch_seeds[epoch] = random.randint(0, 2**31)
        seed = epoch_seeds[epoch]
        random.seed(seed)
        all_idx = chunk_friendly_shuffle(
            len(ft_samples), CHUNK_SIZE, BATCH_SIZE)
        random.seed(42)

        print(f'  Epoch {epoch+1}/{MAX_EPOCHS} | seed={seed}')
        t_epoch = time.time()

        for batch_idx, batch_start in enumerate(
                range(0, len(all_idx), BATCH_SIZE)):

            if epoch == start_epoch and batch_idx < batches_to_skip:
                continue
            if opt_step >= TOTAL_STEPS:
                break

            t0 = time.time()

            # ── Batch ─────────────────────────────────────────────────────
            raw   = [ft_samples[i] for i in
                     all_idx[batch_start:batch_start + BATCH_SIZE]]
            batch = collate_s2t_batch(raw)
            del raw
            if batch is None:
                continue

            # ── Teacher: top-K on cuda:1, direct transfer to cuda:0 ───────
            topk_vals = topk_idx = None
            try:
                topk_vals, topk_idx = teacher_topk_direct(
                    batch['feat'], batch['dec_full'])
                L         = batch['labels_s'].shape[1]
                topk_vals = topk_vals[:, :L, :].contiguous()
                topk_idx  = topk_idx[:, :L, :].contiguous()
            except Exception as e:
                print(f'  [teacher skip] {e}')
                del batch
                free_cpu_ram()
                continue

            # ── Student: forward on cuda:0 ────────────────────────────────
            try:
                s_log = student_logits_gpu_p7(batch['feat'], batch['dec_s'])
                s_log = s_log[:, :L, :]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                free_cpu_ram()
                print(f'  [OOM] step {opt_step}')
                del batch, topk_vals, topk_idx
                continue

            labels_dev = batch['labels_s'].to('cuda:0')
            del batch

            # ── Loss: fully on cuda:0 ─────────────────────────────────────
            try:
                loss, ce_v, kd_v = compute_recovery_loss_gpu(
                    s_log, labels_dev, topk_vals, topk_idx)
            except Exception as e:
                print(f'  [loss error] {e}')
                del s_log, topk_vals, topk_idx, labels_dev
                continue
            finally:
                del topk_vals, topk_idx

            scaler.scale(loss / GRAD_ACCUM).backward()
            del s_log, labels_dev, loss

            accum += 1
            ep_ce += ce_v
            ep_kd += kd_v
            ep_n  += 1

            if accum >= GRAD_ACCUM:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum    = 0
                opt_step += 1

```

### Cell 106 (code, score=43)
```python
# # ─────────────────────────────────────────────────────────────────────────────
# # SAVE PHASE 7 BEST
# # ─────────────────────────────────────────────────────────────────────────────

# print('Loading best Phase 7 checkpoint...')
# best_ckpt = load_latest_checkpoint('phase7_ft')
# model_p7.load_state_dict(best_ckpt['model_state'], strict=False)
# best_step  = best_ckpt.get('best_chrf_step', '?')
# best_score = best_ckpt.get('best_chrf', 0.0)
# del best_ckpt
# gc.collect()

# print(f'Best checkpoint: step={best_step}  ASR-ChrF={best_score:.2f}')

# # Option A: Save as PEFT checkpoint (smaller, keeps LoRA structure)
# model_p7.eval()
# save_model_to_drive(model_p7, processor, 'phase7_peft_best')
# print('✓ Saved PEFT checkpoint (phase7_peft_best)')

# # Option B: Merge and save as plain model
# print('Merging LoRA into weights...')
# merged_p7 = model_p7.merge_and_unload()
# merged_p7.eval()
# print(f'Merged params: {count_params(merged_p7):.1f}M')
# save_model_to_drive(merged_p7, processor, 'phase7_final_merged')
# print('✓ Saved merged model (phase7_final_merged)')
```

### Cell 107 (code, score=12)
```python
# print('Loading phase7_final_merged directly to cuda:0...')
# merged_p7, processor = load_model_from_drive(
#     'phase7_final_merged', device_map='cuda:0')
# merged_p7 = merged_p7.to(torch.float16)
# print(f'Loaded. Params: {count_params(merged_p7):.1f}M')
# gpu_mem()
```

### Cell 108 (code, score=136)
```python
# Full benchmark on merged model with checkpoint loading
p7_bench = load_latest_checkpoint('phase7_benchmark')
if p7_bench and p7_bench.get('summary', {}).get('avg_bleu', 0) > 0:
    results = p7_bench['results']
    summary = p7_bench['summary']
    detailed = p7_bench.get('detailed_summary')
    print('Loaded Phase 7 benchmark results from checkpoint.')
    # Recompute detailed if missing
    if not detailed:
        detailed = compute_detailed_summary(results, 'P7_Final', summary['params_M'])
else:
    results, summary = run_benchmark_asr(merged_p7, list(eval_samples), 'P7_Final', save_n=4)
    detailed = compute_detailed_summary(results, 'P7_Final', summary['params_M'])
    save_checkpoint(dict(results=results, summary=summary,
                         detailed_summary=detailed), 'phase7_benchmark', 0)

store_summary(summary)
store_detailed_summary(detailed)
print_detailed_summary_table('P7_Final')
plot_detailed_phase_comparison()
```
OUTPUT:
```text
[ckpt] Loaded phase7_benchmark_step000000.pt
Loaded Phase 7 benchmark results from checkpoint.
[ckpt] Saved all_summaries_step000000.pt (0.0 MB)
[summary] Stored P7_Final (8 total)
[ckpt] Saved all_detailed_summaries_step000000.pt (0.0 MB)
[detailed] Stored P7_Final

================================================================================
  P7_Final - 1087.9M params
================================================================================
Overall: ChrF=33.73±15.65  BLEU=8.16  RTF=0.1336

Per-Pair (8 pairs):
  Pair               N     ChrF     BLEU      RTF
  arb→eng           25    39.62    10.42   0.1241
  ben→eng           25    34.40     6.11   0.0969
  cmn→eng           25    35.62     7.33   0.1288
  eng→arb           25    32.96     6.51   0.1144
  eng→ben           25    39.32     8.49   0.1409
  eng→cmn           25     5.61     2.29   0.2529
  eng→hin           25    44.68    15.41   0.1014
  hin→eng           25    37.65     8.75   0.1094

By Source Language:
     ARB: ChrF= 39.62  BLEU= 10.42  (n=25)
     BEN: ChrF= 34.40  BLEU=  6.11  (n=25)
     CMN: ChrF= 35.62  BLEU=  7.33  (n=25)
     ENG: ChrF= 30.64  BLEU=  8.17  (n=100)
     HIN: ChrF= 37.65  BLEU=  8.75  (n=25)

By Target Language:
     ARB: ChrF= 32.96  BLEU=  6.51  (n=25)
     BEN: ChrF= 39.32  BLEU=  8.49  (n=25)
     CMN: ChrF=  5.61  BLEU=  2.29  (n=25)
     ENG: ChrF= 36.82  BLEU=  8.15  (n=100)
     HIN: ChrF= 44.68  BLEU= 15.41  (n=25)
================================================================================
Plotting detailed comparison for 8 phases: ['P0_V1_Baseline', 'P1_Vocab5L', 'P2_Enc16L', 'P3_LaCoT2U', 'P4_Enc14L', 'P5_Dec14L', 'P6_KD_Merged', 'P7_Final']

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_01_overall_quality.png  [Overall Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_02_chrf_by_pair.png  [ChrF by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_03_bleu_by_pair.png  [BLEU by Language Pair]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_04_src_lang_trends.png  [Source Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_05_tgt_lang_trends.png  [Target Language Trends]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_06_size_vs_quality.png  [Size vs Quality]

<Figure size 1800x1080 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_07_inference_rtf.png  [Inference Speed RTF]

<Figure size 1800x1062 with 1 Axes>
[image/png output omitted]
  ✓ Saved: detailed_comparison_08_summary_table.png  [Summary Table]

✅ All 8 figures saved individually:
   📄 detailed_comparison_01_overall_quality.png
   📄 detailed_comparison_02_chrf_by_pair.png
   📄 detailed_comparison_03_bleu_by_pair.png
   📄 detailed_comparison_04_src_lang_trends.png
   📄 detailed_comparison_05_tgt_lang_trends.png
   📄 detailed_comparison_06_size_vs_quality.png
   📄 detailed_comparison_07_inference_rtf.png
   📄 detailed_comparison_08_summary_table.png
```

### Cell 109 (code, score=49)
```python
# import gc, torch

# # These are PEFT checkpoints — must use base_model.model for generation
# CANDIDATES = [
#     # 'phase6_kd_step001120.pt',
#     # 'phase6_kd_step001200.pt',
#     'phase6_kd_step001360.pt',
#     'phase6_kd_step001280.pt',
# ]

# shootout_results = {}

# for ckpt_file in CANDIDATES:
#     path = f'{CKPT_DIR}/{ckpt_file}'
#     if not os.path.exists(path):
#         # Try pulling from Drive
#         r = subprocess.run(
#             f'rclone copy "{GDRIVE_ROOT}/checkpoints/{ckpt_file}" "{CKPT_DIR}/"',
#             shell=True, capture_output=True, text=True)
#         if not os.path.exists(path):
#             print(f'Missing: {ckpt_file}')
#             continue

#     print(f'\n{"="*60}')
#     print(f'  Loading {ckpt_file}')
#     print(f'{"="*60}')

#     ckpt = torch.load(path, map_location='cpu', weights_only=False)
#     step = ckpt.get('opt_step', '?')
#     student.load_state_dict(ckpt['student_state'], strict=False)
#     del ckpt
#     torch.cuda.empty_cache()
#     gc.collect()

#     student.eval()
#     label = f'P6_S{step}'

#     # PEFT model — pass base_model.model to bypass PEFT generate wrapper
#     results, summary = run_benchmark_asr(
#         student.base_model.model,
#         eval_samples,
#         label=label,
#         save_n=2
#     )

#     detailed = compute_detailed_summary(results, label, summary['params_M'])
#     store_summary(summary)
#     store_detailed_summary(detailed)
#     print_detailed_summary_table(label)

#     shootout_results[ckpt_file] = {
#         'step': step,
#         'asr_chrf': summary['avg_chrf'],
#         'bleu': summary['avg_bleu'],
#         'rtf': summary['avg_rtf'],
#     }

#     torch.cuda.empty_cache()
#     gc.collect()

# # Print winner
# print(f'\n{"="*60}')
# print('SHOOTOUT RESULTS:')
# for f, r in sorted(shootout_results.items(), 
#                     key=lambda x: -x[1]['asr_chrf']):
#     print(f'  {f}: ASR-ChrF={r["asr_chrf"]:.2f}  '
#           f'BLEU={r["bleu"]:.2f}  RTF={r["rtf"]:.3f}')
# winner = max(shootout_results.items(), key=lambda x: x[1]['asr_chrf'])
# print(f'\n  WINNER: {winner[0]} (ASR-ChrF={winner[1]["asr_chrf"]:.2f})')
# print('='*60)

# plot_detailed_phase_comparison()
```

### Cell 112 (code, score=5)
```python
# save_checkpoint(dict(
#     student_state   = student.state_dict(),
#     optimizer_state = optimizer.state_dict(),
#     scheduler_state = scheduler.state_dict(),
#     opt_step        = 560,
#     best_chrf       = 35.00,
# ), 'phase6_kd', 560, keep=3)
```

### Cell 113 (code, score=41)
```python
# # Quick diagnosis: eval on a few training samples vs eval samples
# student.eval()

# print('=== Training sample quality (should be HIGH if overfitting) ===')
# train_scores = []
# for i in range(8):
#     s = ft_samples[i]  # samples the model has seen twice
#     _, wav_out = run_s2st(student.base_model.model, s['wav'], tgt_lang=s['tgt_lang'])
#     pred = asr_transcribe(wav_out, s['tgt_lang'])
#     chrf = compute_chrf(pred, s['ref'])
#     train_scores.append(chrf)
#     print(f'  train[{i}] ChrF={chrf:.1f}  pred={pred[:40]}')

# print(f'  Avg train ChrF: {np.mean(train_scores):.2f}')

# print('\n=== Eval sample quality (should be LOWER if overfitting) ===')
# eval_scores = []
# for i in range(8):
#     s = eval_samples[i]  # unseen samples
#     _, wav_out = run_s2st(student.base_model.model, s['wav'], tgt_lang=s['tgt_lang'])
#     pred = asr_transcribe(wav_out, s['tgt_lang'])
#     chrf = compute_chrf(pred, s['ref'])
#     eval_scores.append(chrf)
#     print(f'  eval[{i}] ChrF={chrf:.1f}  pred={pred[:40]}')

# print(f'  Avg eval ChrF: {np.mean(eval_scores):.2f}')

# gap = np.mean(train_scores) - np.mean(eval_scores)
# print(f'\nTrain-Eval gap: {gap:.2f}')
# print(f'  gap < 5:  healthy generalization')
# print(f'  gap 5-15: mild overfitting')
# print(f'  gap > 15: severe overfitting → stop training')

# student.train()
```

### Cell 114 (code, score=62)
```python
# # ── Full clean restart — run ALL of these in order ───────────────────────────

# import gc
# torch.cuda.empty_cache()
# gc.collect()

# # 1. Reload pristine student
# print('Reloading phase5_dec_14L...')
# student, processor = load_model_from_drive('phase5_dec_14L', device_map='cuda:0')
# student = student.to(torch.float16)
# student.train()

# # 2. Fresh LoRA
# lora_cfg = LoraConfig(
#     r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
#     bias='none', target_modules=LORA_TARGET_SUFFIXES,
# )
# student = get_peft_model(student, lora_cfg)
# student.print_trainable_parameters()
# student.base_model.model.text_decoder.gradient_checkpointing = True
# try:
#     student.base_model.model.speech_encoder.encoder.gradient_checkpointing = True
# except AttributeError:
#     pass

# # 3. Rebuild trainable_params from NEW model
# trainable_params = [p for p in student.parameters() if p.requires_grad]
# print(f'Trainable params: {sum(p.numel() for p in trainable_params)/1e6:.2f}M')
# print(f'All point to cuda:0: '
#       f'{all(p.device.type == "cuda" for p in trainable_params)}')

# # 4. Fresh optimizer + scheduler + scaler
# optimizer = AdamW(
#     trainable_params, lr=LR_PEAK,
#     weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-6
# )
# scheduler = OneCycleLR(
#     optimizer, max_lr=LR_PEAK, total_steps=TOTAL_STEPS,
#     pct_start=0.04, anneal_strategy='cos',
#     div_factor=25.0, final_div_factor=1e4
# )
# scaler = torch.cuda.amp.GradScaler()   # fresh scaler — critical

# print(f'Optimizer param groups: {len(optimizer.param_groups)}')
# print(f'Scaler scale: {scaler.get_scale()}')

# # 5. Rebuild _student_base reference
# _student_base = student.base_model.model
# S_VOCAB = _student_base.shared.num_embeddings
# print(f'S_VOCAB={S_VOCAB}')

# # 6. Quick sanity: one manual forward+backward
# print('\nSanity forward+backward...')
# _batch = collate_s2t_batch([ft_samples[i] for i in range(2)])
# assert _batch is not None, 'collate returned None'

# _s = student_logits_gpu(_batch['feat'], _batch['dec_s'])
# _L = _batch['labels_s'].shape[1]
# _s = _s[:, :_L, :]
# _t = teacher_logits_cpu(_batch['feat'], _batch['dec_full'])
# _t = _t[:, :_L, :]
# _lab = _batch['labels_s'].to('cuda:0')

# _loss, _ce, _kd = compute_recovery_loss(_s, _lab, _t)
# print(f'  loss={_loss.item():.4f}  CE={_ce:.4f}  KD={_kd:.4f}')
# assert _loss.requires_grad, 'Loss has no grad_fn!'

# # Use scaler properly
# scaler.scale(_loss).backward()
# scaler.unscale_(optimizer)
# _norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
# scaler.step(optimizer)
# scaler.update()
# optimizer.zero_grad(set_to_none=True)
# print(f'  grad_norm={_norm:.4f}  scaler_scale={scaler.get_scale()}')
# print('✓ Sanity check passed — safe to run training loop')
```

### Cell 115 (code, score=6)
```python
# batch = collate_s2t_batch([ft_samples[i] for i in range(2)])
# b = 0
# dec  = batch['dec_s'][b].tolist()
# labs = batch['labels_s'][b].tolist()
# print(f'Alignment (tgt={batch["tgt_langs"][b]}):')
# for i in range(min(8, len(dec))):
#     d_full = remap_to_old[dec[i]].item() if dec[i] < len(remap_to_old) else 0
#     d_tok  = processor.tokenizer.convert_ids_to_tokens([d_full])[0]
#     if labs[i] == -100:
#         l_str = 'PAD'
#     else:
#         l_full = remap_to_old[labs[i]].item() if labs[i] < len(remap_to_old) else 0
#         l_str  = processor.tokenizer.convert_ids_to_tokens([l_full])[0]
#     print(f'  [{i}] {d_tok:20s} → {l_str}')
```

### Cell 116 (code, score=5)
```python
# final_step, final_chrf = run_phase6_training()
```

### Cell 122 (code, score=22)
```python
!ls
```
OUTPUT:
```text
audio  checkpoints  figures  models
```