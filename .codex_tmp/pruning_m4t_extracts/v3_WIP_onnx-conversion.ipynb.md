# Final Notebooks\v3_WIP onnx-conversion.ipynb

Extracted notebook map containing markdown headings plus code/output cells likely to matter for reports, reproduction, or agent steering.

## Markdown headings
cell 1: # SeamlessM4T v2
cell 4: ## ⚙️ Setup — run ALL at the start of EVERY Kaggle session
cell 19: ## 📊 Enhanced Per-Language Tracking Enabled # After running benchmark # Save both

## Key cells

### Cell 1 (markdown, score=0)
```markdown
# SeamlessM4T v2 

**Papers:** S2UT (Lee ACL 2022) · SeamlessExpressive (arXiv:2312.05187) ·
LaCo (Yang EMNLP 2024) · CIF (Dong & Xu ICASSP 2020) · ECAPA-TDNN (Desplanques IS 2020) ·
 · ShortGPT (ACL 2025) · MMS (Pratap 2023)
```

### Cell 2 (code, score=6)
```python
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# ── OOM mitigation ────────────────────────────────────────────────────────────
# PyTorch's own error message recommended this.  Set before any CUDA allocations.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Lower the audio cap only if the no_grad fix alone is not enough.
# Fix 2 (no_grad on frozen conditioning) removes ~2-3 GB of activation graphs,
# which should be the entire OOM margin.  Start here; only drop further if needed.
# MAX_AUDIO_SEC_C = 10   # was 12 — safe starting point after the no_grad fix
#                         # raise back to 12 if you stay OOM-free for 300+ steps
```

### Cell 4 (markdown, score=0)
```markdown
## ⚙️ Setup — run ALL at the start of EVERY Kaggle session
```

### Cell 5 (code, score=9)
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

### Cell 6 (code, score=3)
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

### Cell 7 (code, score=25)
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

### Cell 8 (code, score=3)
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
rclone v1.74.0
```

### Cell 9 (code, score=36)
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

### Cell 11 (code, score=37)
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 35.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.1/84.1 kB 7.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.8/100.8 kB 9.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 79.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.6/121.6 kB 9.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 788.2/788.2 kB 46.4 MB/s eta 0:00:00
All packages installed.
```

### Cell 13 (code, score=3)
```python
import torch
import random
import numpy as np

autocast_dtype        = torch.float16
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

### Cell 14 (code, score=115)
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

### Cell 15 (code, score=187)
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

### Cell 16 (code, score=61)
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

### Cell 17 (code, score=82)
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

### Cell 18 (code, score=208)
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
    summaries = sorted(ALL_DETAILED_SUMMARIES.values(), key=lambda s: s['label'])
    if not summaries: 
        print('No detailed summaries yet.')
        return
    
    print(f'Plotting detailed comparison for {len(summaries)} phases: {[s["label"] for s in summaries]}')
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Detailed Phase Comparison: Per-Language Breakdown', fontsize=14, fontweight='bold')
    
    labels = [s['label'] for s in summaries]
    
    # Panel 1: Overall ChrF/BLEU
    ax1 = plt.subplot(3, 3, 1)
    chrfs = [s['avg_chrf'] for s in summaries]
    bleus = [s['avg_bleu'] for s in summaries]
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, chrfs, 0.4, label='ChrF', color='#4CAF50', alpha=0.85)
    ax1.bar(x + 0.2, bleus, 0.4, label='BLEU', color='#2196F3', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax1.set_title('Overall Quality', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Panel 2: Per-pair ChrF comparison across ALL phases (vertical bars, grouped by language pair)
    ax2 = plt.subplot(3, 3, 2)
    
    # Collect all unique language pairs across all phases
    all_pairs = set()
    for s in summaries:
        if 'pair_stats' in s:
            all_pairs.update(s['pair_stats'].keys())
    all_pairs = sorted(all_pairs)
    
    if all_pairs:
        n_pairs = len(all_pairs)
        n_phases = len(summaries)
        bar_width = 0.8 / n_phases
        x_pos = np.arange(n_pairs)
        
        for phase_idx, s in enumerate(summaries):
            pair_stats = s.get('pair_stats', {})
            chrf_vals = [pair_stats.get(pair, {}).get('avg_chrf', 0) for pair in all_pairs]
            offset = (phase_idx - n_phases/2 + 0.5) * bar_width
            ax2.bar(x_pos + offset, chrf_vals, bar_width, 
                   label=s['label'], alpha=0.85)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(all_pairs, rotation=45, ha='right', fontsize=6)
        ax2.set_ylabel('ASR-ChrF')
        ax2.set_title('ChrF by Language Pair (All Phases)', fontweight='bold', fontsize=9)
        ax2.legend(fontsize=6, ncol=2)
        ax2.grid(alpha=0.3, axis='y')
    
    # Panel 3: BLEU by pair for all phases
    ax3 = plt.subplot(3, 3, 3)
    if all_pairs:
        for phase_idx, s in enumerate(summaries):
            pair_stats = s.get('pair_stats', {})
            bleu_vals = [pair_stats.get(pair, {}).get('avg_bleu', 0) for pair in all_pairs]
            offset = (phase_idx - n_phases/2 + 0.5) * bar_width
            ax3.bar(x_pos + offset, bleu_vals, bar_width, 
                   label=s['label'], alpha=0.85)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(all_pairs, rotation=45, ha='right', fontsize=6)
        ax3.set_ylabel('ASR-BLEU')
        ax3.set_title('BLEU by Language Pair (All Phases)', fontweight='bold', fontsize=9)
        ax3.legend(fontsize=6, ncol=2)
```
OUTPUT:
```text
[ckpt] No checkpoint for 'all_detailed_summaries'
Loaded 0 detailed summaries
✓ Enhanced tracking loaded: store_detailed_summary(), compute_detailed_summary(), plot_detailed_phase_comparison(), print_detailed_summary_table()
```

### Cell 19 (markdown, score=23)
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

### Cell 20 (code, score=94)
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

### Cell 21 (code, score=41)
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
            with torch.cuda.amp.autocast(dtype=autocast_dtype):   # ← FIX
                out = mdl.generate(**inputs, tgt_lang=tgt_lang,
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
            with torch.cuda.amp.autocast(dtype=autocast_dtype):   # ← FIX
                out = mdl.generate(**inputs, tgt_lang=tgt_lang,
                                   return_intermediate_token_ids=True)
    finally:
        mdl.vocoder = orig_voc
    text_ids = _remap_ids_for_decode(mdl, out.sequences.cpu())
    return processor.batch_decode(text_ids, skip_special_tokens=True)[0]

def quick_eval_chrf(mdl, samples, max_samples=32, group_size=25):
    """
    Optimized: Only load audio for samples we actually use.
    """
    text_scores = []
    asr_scores = []
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
            text_pred, wav_out = run_s2st(mdl, s['wav'], tgt_lang=tgt)
            asr_pred = asr_transcribe(wav_out, tgt)
            text_scores.append(compute_chrf(text_pred, s['ref']))
            asr_scores.append(compute_chrf(asr_pred, s['ref']))
    
    return float(np.mean(text_scores)), float(np.mean(asr_scores))

print('Benchmark functions ready.')
```
OUTPUT:
```text
Benchmark functions ready.
```

### Cell 22 (code, score=69)
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

### Cell 23 (code, score=51)
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

### Cell 24 (code, score=97)
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

### Cell 25 (code, score=7)
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

### Cell 26 (code, score=106)
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
  Platform : kaggle   Time : 2026-05-07 17:02
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

### Cell 27 (code, score=63)
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

### Cell 28 (code, score=6)
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

### Cell 29 (code, score=64)
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

### Cell 30 (code, score=3)
```python
# ── Load Multilingual Training Samples: En→X and X→En (all 5 languages) ─────
N_TRAIN_PER_PAIR = 1600  # 500 samples per direction = 4000 total
```

### Cell 31 (code, score=66)
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
  Indexed 1449 samples from eng→ben
  Indexed 1449 samples from ben→eng
  Indexed 1469 samples from eng→cmn
  Indexed 1469 samples from cmn→eng
  Indexed 1251 samples from eng→arb
  Indexed 1251 samples from arb→eng
  Indexed 1280 samples from eng→hin
  Indexed 1280 samples from hin→eng

✓ Multilingual dataset ready: 10898 total samples
  RAM usage: ~10.9 MB (metadata only)

✓ Loaded 10898 multilingual training samples
  Language pairs: 8
  RAM usage: ~10.9 MB (metadata only)
  RAM saved: ~54490 MB (would be with old approach)

Samples per language pair:
  arb→eng: 1251
  ben→eng: 1449
  cmn→eng: 1469
  eng→arb: 1251
  eng→ben: 1449
  eng→cmn: 1469
  eng→hin: 1280
  hin→eng: 1280
```

### Cell 32 (code, score=35)
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

✓ Loaded 10898 multilingual training samples across 8 pairs
```

### Cell 33 (code, score=55)
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

### Cell 34 (code, score=18)
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

### Cell 35 (code, score=6)
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

### Cell 36 (code, score=48)
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
  Indexed 1449 samples from eng→ben
  Indexed 1449 samples from ben→eng
  Indexed 1469 samples from eng→cmn
  Indexed 1469 samples from cmn→eng
  Indexed 1251 samples from eng→arb
  Indexed 1251 samples from arb→eng
  Indexed 1280 samples from eng→hin
  Indexed 1280 samples from hin→eng
  ChunkedStreamingDataset: 10898 samples | chunk=4000 | RAM/chunk≈2000MB

✓ ChunkedMultilingualDataset: 10898 samples
  arb→eng: 1251
  ben→eng: 1449
  cmn→eng: 1469
  eng→arb: 1251
  eng→ben: 1449
  eng→cmn: 1469
  eng→hin: 1280
  hin→eng: 1280
  Chunk size: 4000 | Est. peak RAM: ~2000MB
```

### Cell 37 (code, score=37)
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.7/60.7 MB 31.5 MB/s eta 0:00:00
peft + bitsandbytes ready.
```

### Cell 38 (code, score=92)
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

### Cell 39 (code, score=62)
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
  GPU0: 2.17GB alloc / 2.18GB reserved
  GPU1: 3.64GB alloc / 3.65GB reserved
```

### Cell 40 (code, score=50)
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

### Cell 42 (code, score=15)
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

### Cell 43 (code, score=16)
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

### Cell 44 (code, score=6)
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

### Cell 46 (code, score=5)
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

### Cell 47 (code, score=97)
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

### Cell 48 (code, score=18)
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

### Cell 49 (code, score=44)
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
Trainable: 20.58 M  |  Total steps: 1705
✓ Optimizer ready.
```

### Cell 50 (code, score=60)
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

### Cell 51 (code, score=3)
```python
# # Quick sanity check
# batch = collate_s2t_batch([ft_samples[i] for i in range(4)])
# print("Labels non-masked:", (batch['labels_s'] != -100).sum())
# print("Dec_s range:", batch['dec_s'].min(), batch['dec_s'].max(), "vs S_VOCAB:", S_VOCAB)
# print("Labels range:", batch['labels_s'][batch['labels_s']!=-100].unique()[:10])
```

### Cell 52 (code, score=36)
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

### Cell 53 (code, score=175)
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

### Cell 54 (code, score=23)
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

### Cell 55 (code, score=20)
```python
# # Full benchmark
# merged_clean, _ = load_model_from_drive('phase6_kd_merged', device_map='cuda:0')
# merged_clean.eval()

# p6_results, p6_summary = run_benchmark_asr(
#     merged_clean, list(eval_samples), 'P6_KD_Merged', save_n=4)
 
# p6_detailed = compute_detailed_summary(p6_results, 'P6_KD_Merged',
#                                        p6_summary['params_M'])
# save_checkpoint(dict(results=p6_results, summary=p6_summary,
#                      detailed_summary=p6_detailed), 'phase6_benchmark', 0)
# store_summary(p6_summary)
# store_detailed_summary(p6_detailed)
# print_detailed_summary_table('P6_KD_Merged')
# plot_detailed_phase_comparison()
```

### Cell 56 (code, score=42)
```python
heed
```
OUTPUT:
```text
ERROR: NameError name 'heed' is not defined
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_57/239245293.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mheed[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mNameError[0m: name 'heed' is not defined
```

### Cell 57 (code, score=36)
```python
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
```

### Cell 58 (code, score=29)
```python
import subprocess, sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "onnx>=1.15.0",
    "onnxruntime>=1.17.0",
    "onnxsim",
    "onnxscript",
    "optimum",
], check=True)

print("✓ ONNX tooling installed")
```
OUTPUT:
```text
✓ ONNX tooling installed
```

### Cell 59 (code, score=73)
```python
import torch, gc, json, zipfile, time
import torch.nn as nn
import numpy as np
from pathlib import Path

DEVICE = "cuda:0"

model_onnx, processor = load_model_from_drive(
    "phase7_final_merged",
    device_map=DEVICE
)
model_onnx.eval()
disable_all_gradient_checkpointing(model_onnx)

for p in model_onnx.parameters():
    p.requires_grad_(False)

print(f"✓ Model on {DEVICE}")
print(f"  Params: {count_params(model_onnx):.1f}M")
gpu_mem()

EXPORT_DIR = Path("/kaggle/working/seamless_onnx_raw")
SIM_DIR    = Path("/kaggle/working/seamless_onnx_sim")
INT8_DIR   = Path("/kaggle/working/seamless_onnx_int8")

for d in [EXPORT_DIR, SIM_DIR, INT8_DIR]:
    d.mkdir(exist_ok=True)

print("✓ Directories ready")
```
OUTPUT:
```text
[model] Loading phase7_final_merged from /kaggle/working/models/phase7_final_merged ...

Loading weights:   0%|          | 0/1237 [00:00<?, ?it/s]
  Restored custom state: ['_vocab_remap_to_old']
[model] Loaded phase7_final_merged.
✓ Killed gradient checkpointing at 1 locations:
    model.gradient_checkpointing_disable()
✓ Model on cuda:0
  Params: 1087.9M
  GPU0: 4.94GB alloc / 6.60GB reserved
  GPU1: 3.64GB alloc / 3.65GB reserved
✓ Directories ready
```

### Cell 60 (code, score=69)
```python
# ─────────────────────────────────────────────────────────────
# CELL 3 — Speech Encoder (patch chunk attention config)
# ─────────────────────────────────────────────────────────────

def get_hidden(x):
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    if isinstance(x, (tuple, list)):
        return x[0]
    raise RuntimeError(f"Unknown output type: {type(x)}")


class SpeechEncoderONNX(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.fp  = m.speech_encoder.feature_projection
        self.enc = m.speech_encoder.encoder
        self.ifn = m.speech_encoder.intermediate_ffn
        self.adp = m.speech_encoder.adapter
        self.ln  = m.speech_encoder.inner_layer_norm

    def forward(self,
                input_features: torch.Tensor,
                attention_mask: torch.Tensor):
        h = self.fp(input_features)
        if isinstance(h, tuple):
            h = h[0]

        h = get_hidden(self.enc(
            hidden_states=h,
            attention_mask=None))   # None skips chunk attention code path

        h = get_hidden(self.ifn(h))

        h = get_hidden(self.adp(
            hidden_states=h,
            attention_mask=attention_mask))

        h = self.ln(h)

        # Subsampled mask — all ones, same device as h
        sub_mask = attention_mask.new_ones(h.shape[0], h.shape[1])

        return h, sub_mask


# ── Patch config to disable chunk attention ───────────────────
enc_config = model_onnx.speech_encoder.encoder.config
original_chunk_size = enc_config.speech_encoder_chunk_size

print(f"Original speech_encoder_chunk_size: {original_chunk_size}")
enc_config.speech_encoder_chunk_size = None   # ← disable chunk attention
print(f"Patched  speech_encoder_chunk_size: {enc_config.speech_encoder_chunk_size}")

enc_wrapper = SpeechEncoderONNX(model_onnx).to(DEVICE).eval()

dummy_feat = torch.randn(1, 400, 160, device=DEVICE, dtype=torch.float32)
dummy_mask = torch.ones(1, 400, device=DEVICE, dtype=torch.long)

# Test forward
print("\nTesting speech encoder...")
with torch.no_grad():
    try:
        h, sub_mask = enc_wrapper(dummy_feat, dummy_mask)
        print(f"✓ hidden:   {h.shape}")
        print(f"✓ sub_mask: {sub_mask.shape}")
    except Exception as e:
        enc_config.speech_encoder_chunk_size = original_chunk_size
        print(f"✗ {e}")
        import traceback
        traceback.print_exc()
        raise

# Export
print("\nExporting speech_encoder.onnx...")
t = time.time()

with torch.no_grad():
    torch.onnx.export(
        enc_wrapper,
        (dummy_feat, dummy_mask),
        str(EXPORT_DIR / "speech_encoder.onnx"),
        input_names=["input_features", "attention_mask"],
        output_names=["encoder_hidden_states", "encoder_attention_mask"],
        dynamic_axes={
            "input_features":         {0: "batch", 1: "time"},
            "attention_mask":         {0: "batch", 1: "time"},
            "encoder_hidden_states":  {0: "batch", 1: "enc_seq"},
            "encoder_attention_mask": {0: "batch", 1: "enc_seq"},
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )

mb = (EXPORT_DIR / "speech_encoder.onnx").stat().st_size / 1e6
print(f"✓ speech_encoder.onnx  {mb:.1f}MB  {time.time()-t:.1f}s")

# Restore config
enc_config.speech_encoder_chunk_size = original_chunk_size
print(f"Restored speech_encoder_chunk_size: {original_chunk_size}")

del enc_wrapper
torch.cuda.empty_cache()
gc.collect()
```
OUTPUT:
```text
Original speech_encoder_chunk_size: 20000
Patched  speech_encoder_chunk_size: None

Testing speech encoder...
✓ hidden:   torch.Size([1, 51, 1024])
✓ sub_mask: torch.Size([1, 51])

Exporting speech_encoder.onnx...
✓ speech_encoder.onnx  786.9MB  9.8s
Restored speech_encoder_chunk_size: 20000

5308
```

### Cell 61 (code, score=57)
```python
# ─────────────────────────────────────────────────────────────
# CELL 4 — Text Decoder
# Fixed: enc dummy length matches subsampled enc output (~50)
#         do_constant_folding=False to prevent shape baking
# ─────────────────────────────────────────────────────────────

class TextDecoderONNX(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.embed = m.shared
        self.dec   = m.text_decoder
        self.lm    = m.lm_head

    def forward(self,
                decoder_input_ids,
                encoder_hidden_states,
                encoder_attention_mask):
        e   = self.embed(decoder_input_ids)
        out = self.dec(
            inputs_embeds=e,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            past_key_values=None,
        )
        return self.lm(get_hidden(out))


dec_wrapper = TextDecoderONNX(model_onnx).to(DEVICE).eval()

# Use subsampled length (50) not raw input length (575)
dummy_dec  = torch.ones(1, 5,  device=DEVICE, dtype=torch.long)
dummy_enc  = torch.randn(1, 50, 1024, device=DEVICE, dtype=torch.float32)
dummy_emsk = torch.ones(1, 50, device=DEVICE, dtype=torch.long)

# Verify forward works
print("Verifying text decoder forward...")
with torch.no_grad():
    try:
        test_emb = model_onnx.shared(dummy_dec)
        test_out = model_onnx.text_decoder(
            inputs_embeds=test_emb,
            encoder_hidden_states=dummy_enc,
            encoder_attention_mask=dummy_emsk,
            use_cache=False,
        )
        print(f"✓ forward OK: {get_hidden(test_out).shape}")
    except Exception as e:
        print(f"✗ {e}")
        raise

print("\nExporting text_decoder.onnx...")
t = time.time()

with torch.no_grad():
    torch.onnx.export(
        dec_wrapper,
        (dummy_dec, dummy_enc, dummy_emsk),
        str(EXPORT_DIR / "text_decoder.onnx"),
        input_names=[
            "decoder_input_ids",
            "encoder_hidden_states",
            "encoder_attention_mask",
        ],
        output_names=["logits"],
        dynamic_axes={
            "decoder_input_ids":      {0: "batch", 1: "dec_seq"},
            "encoder_hidden_states":  {0: "batch", 1: "enc_seq"},
            "encoder_attention_mask": {0: "batch", 1: "enc_seq"},
            "logits":                 {0: "batch", 1: "dec_seq"},
        },
        opset_version=17,
        do_constant_folding=False,
        export_params=True,
        dynamo=False,
    )

mb = (EXPORT_DIR / "text_decoder.onnx").stat().st_size / 1e6
print(f"✓ text_decoder.onnx  {mb:.1f}MB  {time.time()-t:.1f}s")

del dec_wrapper
torch.cuda.empty_cache()
gc.collect()
```
OUTPUT:
```text
Verifying text decoder forward...
✓ forward OK: torch.Size([1, 5, 1024])

Exporting text_decoder.onnx...
✓ text_decoder.onnx  1614.3MB  21.4s

8
```

### Cell 62 (code, score=100)
```python
# ─────────────────────────────────────────────────────────────
# CELL 5 — T2U Model
# Key insight: use torch.jit.trace directly
# then export to ONNX from traced script
# This avoids torch.export which fails on T2U
# ─────────────────────────────────────────────────────────────

class T2UEncoderONNX(nn.Module):
    """Export T2U encoder separately."""
    def __init__(self, m):
        super().__init__()
        self.encoder = m.t2u_model.model.encoder

    def forward(self, inputs_embeds, attention_mask):
        out = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return get_hidden(out)


class T2UDecoderONNX(nn.Module):
    """Export T2U decoder separately."""
    def __init__(self, m):
        super().__init__()
        self.decoder = m.t2u_model.model.decoder
        self.lm_head = m.t2u_model.lm_head

    def forward(self,
                decoder_input_ids,
                encoder_hidden_states,
                encoder_attention_mask):
        out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return self.lm_head(get_hidden(out))


class T2UEmbedding(nn.Module):
    """Export T2U token embedding separately."""
    def __init__(self, m):
        super().__init__()
        # T2U encoder uses shared embedding
        self.embed = m.t2u_model.model.encoder.embed_tokens \
            if hasattr(m.t2u_model.model.encoder, 'embed_tokens') \
            else m.shared

    def forward(self, input_ids):
        return self.embed(input_ids)


# ── Find correct embed_tokens ─────────────────────────────────
print("Inspecting T2U model structure...")
t2u = model_onnx.t2u_model

# Check what attributes exist
print(f"  t2u_model type: {type(t2u).__name__}")
print(f"  t2u_model.model type: {type(t2u.model).__name__}")

enc = t2u.model.encoder
print(f"  encoder type: {type(enc).__name__}")
print(f"  encoder attrs: {[a for a in dir(enc) if not a.startswith('_') and 'embed' in a.lower()]}")

dec = t2u.model.decoder
print(f"  decoder attrs: {[a for a in dir(dec) if not a.startswith('_') and 'embed' in a.lower()]}")
```
OUTPUT:
```text
Inspecting T2U model structure...
  t2u_model type: SeamlessM4Tv2TextToUnitForConditionalGeneration
  t2u_model.model type: SeamlessM4Tv2TextToUnitModel
  encoder type: SeamlessM4Tv2Encoder
  encoder attrs: ['get_input_embeddings', 'get_output_embeddings', 'get_position_embeddings', 'resize_position_embeddings', 'resize_token_embeddings', 'set_input_embeddings', 'set_output_embeddings']
  decoder attrs: ['embed_char', 'embed_char_positions', 'embed_positions', 'embed_scale', 'embed_tokens', 'get_input_embeddings', 'get_output_embeddings', 'get_position_embeddings', 'resize_position_embeddings', 'resize_token_embeddings', 'set_input_embeddings', 'set_output_embeddings']
```

### Cell 63 (code, score=42)
```python
# ─────────────────────────────────────────────────────────────
# CELL 5b — T2U Export (Correct Architecture)
#
# SeamlessM4Tv2 T2U is NOT a standard seq2seq decoder.
# Architecture:
#   text_ids → T2U Encoder → hidden_states
#            → Duration Predictor → char_count_per_id (int per token)
#            → Upsample hidden_states by char_count
#            → T2U Decoder (char-level) → unit logits
#
# We export TWO separate ONNX models:
#   1. t2u_encoder.onnx  — text tokens → encoder hidden states + durations
#   2. t2u_decoder.onnx  — upsampled hidden states → unit logits
#
# Upsampling (repeat_interleave) is done in Python between them.
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import time
import gc


def get_hidden(x):
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    if isinstance(x, (tuple, list)):
        return x[0]
    raise RuntimeError(f"Unknown output type: {type(x)}")


# ── Inspect T2U model internals ───────────────────────────────
print("=== Inspecting T2U Model ===")
t2u = model_onnx.t2u_model
print(f"t2u type: {type(t2u).__name__}")

t2u_inner = t2u.model
print(f"t2u.model type: {type(t2u_inner).__name__}")

print("\nt2u.model children:")
for name, mod in t2u_inner.named_children():
    print(f"  {name}: {type(mod).__name__}")

print("\nt2u children:")
for name, mod in t2u.named_children():
    print(f"  {name}: {type(mod).__name__}")
```
OUTPUT:
```text
=== Inspecting T2U Model ===
t2u type: SeamlessM4Tv2TextToUnitForConditionalGeneration
t2u.model type: SeamlessM4Tv2TextToUnitModel

t2u.model children:
  encoder: SeamlessM4Tv2Encoder
  decoder: SeamlessM4Tv2TextToUnitDecoder

t2u children:
  model: SeamlessM4Tv2TextToUnitModel
  lm_head: Linear
```

### Cell 64 (code, score=78)
```python
# ─────────────────────────────────────────────────────────────
# CELL 5c — CORRECTED T2U Encoder Wrapper
#
# Facts learned:
#   1. T2U encoder requires inputs_embeds, NOT input_ids
#   2. duration_predictor is inside model.decoder
#   3. Embedding lives in model.decoder.embed_tokens (check)
# ─────────────────────────────────────────────────────────────

print("=== T2U Structure Deep Dive ===")
t2u       = model_onnx.t2u_model
t2u_inner = t2u.model

print("\nmodel.encoder children:")
for name, mod in t2u_inner.encoder.named_children():
    print(f"  {name}: {type(mod).__name__}")

print("\nmodel.decoder children:")
for name, mod in t2u_inner.decoder.named_children():
    print(f"  {name}: {type(mod).__name__}")

# Find the embedding table
print("\nLooking for embed_tokens...")
for obj_name, obj in [("t2u_model", t2u),
                       ("t2u_model.model", t2u_inner),
                       ("t2u_model.model.encoder", t2u_inner.encoder),
                       ("t2u_model.model.decoder", t2u_inner.decoder)]:
    if hasattr(obj, "embed_tokens"):
        emb = obj.embed_tokens
        print(f"  Found at {obj_name}.embed_tokens: "
              f"{type(emb).__name__}  weight={emb.weight.shape}")
    if hasattr(obj, "embed_scale"):
        print(f"  {obj_name}.embed_scale = {obj.embed_scale}")
```
OUTPUT:
```text
=== T2U Structure Deep Dive ===

model.encoder children:
  layers: ModuleList
  layer_norm: LayerNorm

model.decoder children:
  embed_tokens: Embedding
  embed_char: Embedding
  embed_char_positions: SeamlessM4Tv2SinusoidalPositionalEmbedding
  duration_predictor: SeamlessM4Tv2VariancePredictor
  embed_positions: SeamlessM4Tv2SinusoidalPositionalEmbedding
  layers: ModuleList
  layer_norm: LayerNorm

Looking for embed_tokens...
  Found at t2u_model.model.decoder.embed_tokens: Embedding  weight=torch.Size([10082, 1024])
  t2u_model.model.decoder.embed_scale = 32.0
```

### Cell 65 (code, score=104)
```python
# ─────────────────────────────────────────────────────────────
# CELL 5d — T2U Encoder ONNX Wrapper (inputs_embeds path)
# ─────────────────────────────────────────────────────────────

class T2UEmbedWrapper(nn.Module):
    """
    Converts text_token_ids → inputs_embeds for T2U encoder.
    Exported as a tiny separate ONNX (embedding lookup only).
    """
    def __init__(self, t2u_model):
        super().__init__()
        inner = t2u_model.model

        # Find embed_tokens — check decoder first (common in SeamlessM4Tv2)
        if hasattr(inner.decoder, "embed_tokens"):
            self.embed = inner.decoder.embed_tokens
            src = "decoder.embed_tokens"
        elif hasattr(inner.encoder, "embed_tokens"):
            self.embed = inner.encoder.embed_tokens
            src = "encoder.embed_tokens"
        elif hasattr(t2u_model, "shared"):
            self.embed = t2u_model.shared
            src = "t2u_model.shared"
        else:
            # Walk all modules
            for name, mod in t2u_model.named_modules():
                if isinstance(mod, nn.Embedding) and "embed_tokens" in name:
                    self.embed = mod
                    src = name
                    break
            else:
                raise RuntimeError("Cannot find embed_tokens in T2U model")

        print(f"  Using embedding from: {src}  shape={self.embed.weight.shape}")

        # embed_scale (some models scale embeddings by sqrt(d_model))
        self.scale = getattr(inner.decoder, "embed_scale",
                    getattr(inner.encoder, "embed_scale", 1.0))
        print(f"  embed_scale = {self.scale}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids) * self.scale


class T2UEncoderWrapper(nn.Module):
    """
    inputs_embeds + attention_mask → encoder hidden states
    """
    def __init__(self, t2u_model):
        super().__init__()
        self.encoder = t2u_model.model.encoder

    def forward(self,
                inputs_embeds:  torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return get_hidden(out)


class T2UDurationWrapper(nn.Module):
    """
    encoder_hidden_states → log_durations
    duration_predictor lives in model.decoder
    """
    def __init__(self, t2u_model):
        super().__init__()
        self.dur_pred = t2u_model.model.decoder.duration_predictor

    def forward(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        # SeamlessM4Tv2VariancePredictor returns [B, T, 1] or [B, T]
        out = self.dur_pred(encoder_hidden)
        if out.dim() == 3:
            out = out.squeeze(-1)   # [B, T]
        return out                  # log durations


# ── Instantiate & Test ────────────────────────────────────────
print("Building wrappers...")
t2u_embed_w = T2UEmbedWrapper(model_onnx.t2u_model).to(DEVICE).eval()
t2u_enc_w   = T2UEncoderWrapper(model_onnx.t2u_model).to(DEVICE).eval()
t2u_dur_w   = T2UDurationWrapper(model_onnx.t2u_model).to(DEVICE).eval()

print("\nTesting forward passes...")
with torch.no_grad():
    test_ids  = torch.ones(1, 10, device=DEVICE, dtype=torch.long)
    test_mask = torch.ones(1, 10, device=DEVICE, dtype=torch.long)

    # Step 1: embed
    embeds = t2u_embed_w(test_ids)
    print(f"✓ embeds:    {embeds.shape}")

    # Step 2: encode
    enc_h = t2u_enc_w(embeds, test_mask)
    print(f"✓ enc_h:     {enc_h.shape}")

    # Step 3: duration
    log_dur = t2u_dur_w(enc_h)
    print(f"✓ log_dur:   {log_dur.shape}  "
          f"range=[{log_dur.min():.2f}, {log_dur.max():.2f}]")

    # Sanity: convert to integer durations
    durations = torch.clamp(torch.round(torch.exp(log_dur)), 1, 20).long()
    print(f"  durations: {durations[0].tolist()}")
```
OUTPUT:
```text
Building wrappers...
  Using embedding from: decoder.embed_tokens  shape=torch.Size([10082, 1024])
  embed_scale = 32.0

Testing forward passes...
✓ embeds:    torch.Size([1, 10, 1024])
✓ enc_h:     torch.Size([1, 10, 1024])
✓ log_dur:   torch.Size([1, 10])  range=[1.26, 3.32]
  durations: [4, 5, 5, 5, 5, 5, 5, 5, 5, 20]
```

### Cell 66 (code, score=55)
```python
# ─────────────────────────────────────────────────────────────
# CELL 5e-INSPECT — Check exact layer forward signature
# ─────────────────────────────────────────────────────────────

import inspect

dec = model_onnx.t2u_model.model.decoder

# Check layer forward signature
layer0 = dec.layers[0]
print(f"Layer type: {type(layer0).__name__}")
print("\nforward() signature:")
sig = inspect.signature(layer0.forward)
for name, param in sig.parameters.items():
    print(f"  {name}: default={param.default}")

# Check embed_char
print(f"\nembed_char: {dec.embed_char}")
print(f"embed_char_positions: {type(dec.embed_char_positions).__name__}")
print(f"embed_positions: {type(dec.embed_positions).__name__}")

# Check the full decoder forward signature
print("\nDecoder forward() signature:")
sig2 = inspect.signature(dec.forward)
for name, param in sig2.parameters.items():
    print(f"  {name}: default={param.default}")
```
OUTPUT:
```text
Layer type: SeamlessM4Tv2TextToUnitDecoderLayer

forward() signature:
  hidden_states: default=<class 'inspect._empty'>
  attention_mask: default=None
  padding_mask: default=None
  output_attentions: default=False

embed_char: Embedding(10943, 1024)
embed_char_positions: SeamlessM4Tv2SinusoidalPositionalEmbedding
embed_positions: SeamlessM4Tv2SinusoidalPositionalEmbedding

Decoder forward() signature:
  char_input_ids: default=None
  char_count_per_id: default=None
  encoder_hidden_states: default=None
  output_attentions: default=None
  output_hidden_states: default=None
  return_dict: default=None
  kwargs: default=<class 'inspect._empty'>
```

### Cell 67 (code, score=39)
```python
# ─────────────────────────────────────────────────────────────
# CELL 5e-FINAL — Fix pos_emb call + confirm clean forward
# SeamlessM4Tv2SinusoidalPositionalEmbedding.forward(input, past_key_values_length)
# where input is the embeddings tensor itself [B, T, D]
# ─────────────────────────────────────────────────────────────

import inspect

dec = model_onnx.t2u_model.model.decoder

# Inspect the positional embedding forward signature
char_pos = dec.embed_char_positions
print(f"char_pos_emb type: {type(char_pos).__name__}")
print("forward() sig:", inspect.signature(char_pos.forward))

# Try calling it the correct way
with torch.no_grad():
    test_h = torch.randn(1, 64, 1024, device=DEVICE)
    
    # Method 1: pass the tensor directly
    try:
        pe = char_pos(test_h)
        print(f"✓ Method 1 (tensor): pe={pe.shape}")
    except Exception as e:
        print(f"✗ Method 1: {e}")
    
    # Method 2: pass (seq_len,) tuple
    try:
        pe = char_pos((1, 64))
        print(f"✓ Method 2 (tuple): pe={pe.shape}")
    except Exception as e:
        print(f"✗ Method 2: {e}")
    
    # Method 3: pass positions tensor
    try:
        pos = torch.arange(64, device=DEVICE).unsqueeze(0)  # [1, 64]
        pe = char_pos(pos)
        print(f"✓ Method 3 (arange): pe={pe.shape}")
    except Exception as e:
        print(f"✗ Method 3: {e}")

    # Method 4: the HF sinusoidal takes (input_ids_shape, past_key_values_length)
    # where input_ids_shape = (batch, seq_len)
    try:
        pe = char_pos((1, 64), 0)
        print(f"✓ Method 4 (shape_tuple, 0): pe={pe.shape}")
    except Exception as e:
        print(f"✗ Method 4: {e}")
```
OUTPUT:
```text
char_pos_emb type: SeamlessM4Tv2SinusoidalPositionalEmbedding
forward() sig: (input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None, past_key_values_length: int = 0)
✗ Method 1: too many values to unpack (expected 2)
✗ Method 2: 'tuple' object has no attribute 'size'
✓ Method 3 (arange): pe=torch.Size([1, 64, 1024])
✗ Method 4: 'tuple' object has no attribute 'size'
```

### Cell 68 (code, score=81)
```python
# ─────────────────────────────────────────────────────────────
# CELL 5e-WRAPPER CORRECTED
# SeamlessM4Tv2SinusoidalPositionalEmbedding.forward:
#   (input_ids=None, inputs_embeds=None, past_key_values_length=0)
# Pass inputs_embeds=h to get positional embeddings
# ─────────────────────────────────────────────────────────────

class T2UCharDecoderWrapper(nn.Module):
    """
    upsampled_hidden [B, T_char, D] → unit_logits [B, T_char, 10082]
    Self-attention only decoder (no cross-attention).
    """
    def __init__(self, t2u_model):
        super().__init__()
        dec = t2u_model.model.decoder

        self.layers       = dec.layers
        self.layer_norm   = dec.layer_norm
        self.char_pos_emb = dec.embed_char_positions
        self.lm_head      = t2u_model.lm_head

        print(f"  layers:       {len(self.layers)}")
        print(f"  layer_norm:   {type(self.layer_norm).__name__}")
        print(f"  pos_emb:      {type(self.char_pos_emb).__name__}")
        print(f"  lm_head out:  {self.lm_head.out_features}")

    def forward(self, upsampled_hidden: torch.Tensor) -> torch.Tensor:
        """
        upsampled_hidden: [B, T_char, D]
        returns:          [B, T_char, 10082]
        """
        h = upsampled_hidden  # [B, T_char, D]

        # Add sinusoidal positional embeddings
        # Pass inputs_embeds=h, past_key_values_length=0
        pe = self.char_pos_emb(
            inputs_embeds=h,
            past_key_values_length=0
        )  # [B, T_char, D]
        
        h = h + pe

        # Self-attention layers
        for layer in self.layers:
            out = layer(
                hidden_states=h,
                attention_mask=None,
                padding_mask=None,
                output_attentions=False,
            )
            h = get_hidden(out)

        h = self.layer_norm(h)
        return self.lm_head(h)


print("Building final T2UCharDecoderWrapper...")
t2u_char_dec_w = T2UCharDecoderWrapper(
    model_onnx.t2u_model).to(DEVICE).eval()

# Test end-to-end
print("\nTesting end-to-end T2U pipeline (PyTorch)...")
with torch.no_grad():
    try:
        # Use enc_h and durations from Cell 5d
        ups_h = enc_h.repeat_interleave(durations[0], dim=1)
        print(f"  upsampled: {ups_h.shape}  "
              f"(sum_durations={durations[0].sum().item()})")

        unit_logits = t2u_char_dec_w(ups_h)
        print(f"✓ unit_logits: {unit_logits.shape}")

        unit_ids = unit_logits.argmax(-1)
        print(f"  sample units: {unit_ids[0].tolist()[:20]}...")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
```
OUTPUT:
```text
Building final T2UCharDecoderWrapper...
  layers:       6
  layer_norm:   LayerNorm
  pos_emb:      SeamlessM4Tv2SinusoidalPositionalEmbedding
  lm_head out:  10082

Testing end-to-end T2U pipeline (PyTorch)...
  upsampled: torch.Size([1, 64, 1024])  (sum_durations=64)
✓ unit_logits: torch.Size([1, 64, 10082])
  sample units: [8980, 5553, 7947, 5573, 5202, 477, 3770, 5209, 5209, 5209, 3351, 5686, 5284, 5284, 5284, 5209, 5209, 7108, 7108, 7108]...
```

### Cell 69 (code, score=222)
```python
# ─────────────────────────────────────────────────────────────
# CELL 5f — Export All 4 T2U ONNX Models
# ─────────────────────────────────────────────────────────────

D = 1024  # confirmed from Cell 5d

print("=" * 50)

# ── 1. t2u_embed.onnx ────────────────────────────────────────
print("[1/4] Exporting t2u_embed.onnx...")
t = time.time()
dummy_ids = torch.ones(1, 20, device=DEVICE, dtype=torch.long)

with torch.no_grad():
    torch.onnx.export(
        t2u_embed_w,
        (dummy_ids,),
        str(EXPORT_DIR / "t2u_embed.onnx"),
        input_names=["input_ids"],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids":     {0: "batch", 1: "seq"},
            "inputs_embeds": {0: "batch", 1: "seq"},
        },
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
    )
mb = (EXPORT_DIR / "t2u_embed.onnx").stat().st_size / 1e6
print(f"  ✓ {mb:.1f}MB  {time.time()-t:.1f}s")

# ── 2. t2u_encoder.onnx ──────────────────────────────────────
print("[2/4] Exporting t2u_encoder.onnx...")
t = time.time()
dummy_emb  = torch.randn(1, 20, D, device=DEVICE)
dummy_mask = torch.ones(1, 20,    device=DEVICE, dtype=torch.long)

with torch.no_grad():
    torch.onnx.export(
        t2u_enc_w,
        (dummy_emb, dummy_mask),
        str(EXPORT_DIR / "t2u_encoder.onnx"),
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "inputs_embeds":         {0: "batch", 1: "seq"},
            "attention_mask":        {0: "batch", 1: "seq"},
            "encoder_hidden_states": {0: "batch", 1: "seq"},
        },
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
    )
mb = (EXPORT_DIR / "t2u_encoder.onnx").stat().st_size / 1e6
print(f"  ✓ {mb:.1f}MB  {time.time()-t:.1f}s")

# ── 3. t2u_duration.onnx ─────────────────────────────────────
print("[3/4] Exporting t2u_duration.onnx...")
t = time.time()
dummy_enc_h = torch.randn(1, 20, D, device=DEVICE)

with torch.no_grad():
    torch.onnx.export(
        t2u_dur_w,
        (dummy_enc_h,),
        str(EXPORT_DIR / "t2u_duration.onnx"),
        input_names=["encoder_hidden_states"],
        output_names=["log_durations"],
        dynamic_axes={
            "encoder_hidden_states": {0: "batch", 1: "seq"},
            "log_durations":         {0: "batch", 1: "seq"},
        },
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
    )
mb = (EXPORT_DIR / "t2u_duration.onnx").stat().st_size / 1e6
print(f"  ✓ {mb:.1f}MB  {time.time()-t:.1f}s")

# ── 4. t2u_decoder.onnx ──────────────────────────────────────
print("[4/4] Exporting t2u_decoder.onnx...")
t = time.time()
dummy_ups = torch.randn(1, 60, D, device=DEVICE)

with torch.no_grad():
    torch.onnx.export(
        t2u_char_dec_w,
        (dummy_ups,),
        str(EXPORT_DIR / "t2u_decoder.onnx"),
        input_names=["upsampled_hidden"],
        output_names=["unit_logits"],
        dynamic_axes={
            "upsampled_hidden": {0: "batch", 1: "char_seq"},
            "unit_logits":      {0: "batch", 1: "char_seq"},
        },
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
    )
mb = (EXPORT_DIR / "t2u_decoder.onnx").stat().st_size / 1e6
print(f"  ✓ {mb:.1f}MB  {time.time()-t:.1f}s")

# ── Cleanup ──────────────────────────────────────────────────
for w in [t2u_embed_w, t2u_enc_w, t2u_dur_w, t2u_char_dec_w]:
    del w
torch.cuda.empty_cache()
gc.collect()

# ── Summary ──────────────────────────────────────────────────
print("\n=== All Exported ONNX Files ===")
total = 0
for f in sorted(EXPORT_DIR.glob("*.onnx")):
    mb = f.stat().st_size / 1e6
    total += mb
    print(f"  {f.name:35s}  {mb:7.1f}MB")
print(f"  {'TOTAL':35s}  {total:7.1f}MB")
```
OUTPUT:
```text
==================================================
[1/4] Exporting t2u_embed.onnx...

W0507 18:00:57.328000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'input' from (input, rois, spatial_scale: 'float', pooled_height: 'int', pooled_width: 'int', sampling_ratio: 'int' = -1, aligned: 'bool' = False). Treating as an Input.
W0507 18:00:57.329000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'rois' from (input, rois, spatial_scale: 'float', pooled_height: 'int', pooled_width: 'int', sampling_ratio: 'int' = -1, aligned: 'bool' = False). Treating as an Input.
W0507 18:00:57.331000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'input' from (input, boxes, output_size: 'Sequence[int]', spatial_scale: 'float' = 1.0). Treating as an Input.
W0507 18:00:57.332000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'boxes' from (input, boxes, output_size: 'Sequence[int]', spatial_scale: 'float' = 1.0). Treating as an Input.

[torch.onnx] Obtain model graph for `T2UEmbedWrapper([...]` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `T2UEmbedWrapper([...]` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decomposition...
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
  ✓ 0.0MB  0.9s
[2/4] Exporting t2u_encoder.onnx...

W0507 18:00:58.236000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'input' from (input, rois, spatial_scale: 'float', pooled_height: 'int', pooled_width: 'int', sampling_ratio: 'int' = -1, aligned: 'bool' = False). Treating as an Input.
W0507 18:00:58.238000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'rois' from (input, rois, spatial_scale: 'float', pooled_height: 'int', pooled_width: 'int', sampling_ratio: 'int' = -1, aligned: 'bool' = False). Treating as an Input.
W0507 18:00:58.240000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'input' from (input, boxes, output_size: 'Sequence[int]', spatial_scale: 'float' = 1.0). Treating as an Input.
W0507 18:00:58.241000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'boxes' from (input, boxes, output_size: 'Sequence[int]', spatial_scale: 'float' = 1.0). Treating as an Input.

[torch.onnx] Obtain model graph for `T2UEncoderWrapper([...]` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `T2UEncoderWrapper([...]` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decomposition...
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
  ✓ 0.3MB  8.4s
[3/4] Exporting t2u_duration.onnx...

W0507 18:01:06.680000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'input' from (input, rois, spatial_scale: 'float', pooled_height: 'int', pooled_width: 'int', sampling_ratio: 'int' = -1, aligned: 'bool' = False). Treating as an Input.
W0507 18:01:06.681000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'rois' from (input, rois, spatial_scale: 'float', pooled_height: 'int', pooled_width: 'int', sampling_ratio: 'int' = -1, aligned: 'bool' = False). Treating as an Input.
W0507 18:01:06.683000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'input' from (input, boxes, output_size: 'Sequence[int]', spatial_scale: 'float' = 1.0). Treating as an Input.
W0507 18:01:06.685000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'boxes' from (input, boxes, output_size: 'Sequence[int]', spatial_scale: 'float' = 1.0). Treating as an Input.

[torch.onnx] Obtain model graph for `T2UDurationWrapper([...]` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `T2UDurationWrapper([...]` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decomposition...
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
  ✓ 0.0MB  1.3s
[4/4] Exporting t2u_decoder.onnx...

W0507 18:01:07.946000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'input' from (input, rois, spatial_scale: 'float', pooled_height: 'int', pooled_width: 'int', sampling_ratio: 'int' = -1, aligned: 'bool' = False). Treating as an Input.
W0507 18:01:07.947000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'rois' from (input, rois, spatial_scale: 'float', pooled_height: 'int', pooled_width: 'int', sampling_ratio: 'int' = -1, aligned: 'bool' = False). Treating as an Input.
W0507 18:01:07.949000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'input' from (input, boxes, output_size: 'Sequence[int]', spatial_scale: 'float' = 1.0). Treating as an Input.
W0507 18:01:07.950000 57 torch/onnx/_internal/exporter/_schemas.py:455] Missing annotation for parameter 'boxes' from (input, boxes, output_size: 'Sequence[int]', spatial_scale: 'float' = 1.0). Treating as an Input.

[torch.onnx] Obtain model graph for `T2UCharDecoderWrapper([...]` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `T2UCharDecoderWrapper([...]` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decomposition...
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
  ✓ 0.3MB  7.2s

=== All Exported ONNX Files ===
  speech_encoder.onnx                      2.2MB
  t2u_decoder.onnx                         0.3MB
  t2u_duration.onnx                        0.0MB
  t2u_embed.onnx                           0.0MB
  t2u_encoder.onnx                         0.3MB
  text_decoder.onnx                        1.7MB
  vocoder_duration.onnx                   45.3MB
  vocoder_hifigan.onnx                    38.5MB
  TOTAL                                   88.2MB
```

### Cell 70 (code, score=52)
```python
# ─────────────────────────────────────────────────────────────
# CELL 5g — T2U inference helper + updated Cell 9
# ─────────────────────────────────────────────────────────────

def run_t2u_onnx(dec_ids:        np.ndarray,
                 t2u_embed_sess,
                 t2u_enc_sess,
                 t2u_dur_sess,
                 t2u_dec_sess) -> np.ndarray:
    """
    dec_ids: [1, T] int64 — text token ids

    Pipeline:
      embed → t2u_encoder → duration → upsample → t2u_decoder

    Returns: unit_ids [1, T_char] int64
    """
    t2u_mask = np.ones(dec_ids.shape, dtype=np.int64)

    # 1. Embed text tokens
    embeds = t2u_embed_sess.run(
        ["inputs_embeds"],
        {"input_ids": dec_ids}
    )[0]                                         # [1, T, D]

    # 2. T2U Encoder
    enc_h = t2u_enc_sess.run(
        ["encoder_hidden_states"],
        {"inputs_embeds":  embeds,
         "attention_mask": t2u_mask}
    )[0]                                         # [1, T, D]

    # 3. Duration prediction
    log_dur = t2u_dur_sess.run(
        ["log_durations"],
        {"encoder_hidden_states": enc_h}
    )[0][0]                                      # [T]

    # Clamp durations: exp(log_dur), round, clip to [1, 20]
    durations = np.clip(
        np.round(np.exp(log_dur)).astype(np.int32),
        1, 20
    )
    total_chars = int(durations.sum())
    print(f"    T2U: {dec_ids.shape[1]} tokens → "
          f"{total_chars} chars  "
          f"(dur range [{durations.min()},{durations.max()}])")

    # 4. Upsample encoder hidden by duration
    ups_h = np.repeat(enc_h[0], durations, axis=0)[np.newaxis]
    # [1, T_char, D]

    # 5. Char decoder (self-attention only)
    unit_logits = t2u_dec_sess.run(
        ["unit_logits"],
        {"upsampled_hidden": ups_h.astype(np.float32)}
    )[0]                                         # [1, T_char, 10082]

    # 6. Greedy argmax → unit ids
    unit_ids = np.argmax(
        unit_logits[0], axis=-1
    )[np.newaxis].astype(np.int64)               # [1, T_char]

    return unit_ids


print("✓ run_t2u_onnx() defined")
print("""
─────────────────────────────────────────
CELL 9 — replace old T2U block with:
─────────────────────────────────────────

# Load sessions (add these to session loading block):
t2u_embed_sess = ort.InferenceSession(str(INT8_DIR/"t2u_embed.onnx"),    opts, providers=providers)
t2u_enc_sess   = ort.InferenceSession(str(INT8_DIR/"t2u_encoder.onnx"),  opts, providers=providers)
t2u_dur_sess   = ort.InferenceSession(str(INT8_DIR/"t2u_duration.onnx"), opts, providers=providers)
t2u_dec_sess   = ort.InferenceSession(str(INT8_DIR/"t2u_decoder.onnx"),  opts, providers=providers)

# Run T2U:
t = time.time()
unit_ids = run_t2u_onnx(dec_ids, t2u_embed_sess,
                         t2u_enc_sess, t2u_dur_sess, t2u_dec_sess)
print(f"✓ T2U: {time.time()-t:.2f}s  {unit_ids.shape[1]} units")
─────────────────────────────────────────
""")
```
OUTPUT:
```text
✓ run_t2u_onnx() defined

─────────────────────────────────────────
CELL 9 — replace old T2U block with:
─────────────────────────────────────────

# Load sessions (add these to session loading block):
t2u_embed_sess = ort.InferenceSession(str(INT8_DIR/"t2u_embed.onnx"),    opts, providers=providers)
t2u_enc_sess   = ort.InferenceSession(str(INT8_DIR/"t2u_encoder.onnx"),  opts, providers=providers)
t2u_dur_sess   = ort.InferenceSession(str(INT8_DIR/"t2u_duration.onnx"), opts, providers=providers)
t2u_dec_sess   = ort.InferenceSession(str(INT8_DIR/"t2u_decoder.onnx"),  opts, providers=providers)

# Run T2U:
t = time.time()
unit_ids = run_t2u_onnx(dec_ids, t2u_embed_sess,
                         t2u_enc_sess, t2u_dur_sess, t2u_dec_sess)
print(f"✓ T2U: {time.time()-t:.2f}s  {unit_ids.shape[1]} units")
─────────────────────────────────────────
```

### Cell 72 (code, score=30)
```python
# ─────────────────────────────────────────────────────────────
# CELL 6-INSPECT — Check vocoder forward signature
# ─────────────────────────────────────────────────────────────

import inspect

voc = model_onnx.vocoder
print(f"Vocoder type: {type(voc).__name__}")
print("\nforward() signature:")
sig = inspect.signature(voc.forward)
for name, param in sig.parameters.items():
    print(f"  {name}: {param.annotation if param.annotation != inspect._empty else '?'}")

# Test with correct args
print("\nTesting vocoder directly...")
with torch.no_grad():
    test_units = torch.ones(1, 50, device=DEVICE, dtype=torch.long)
    test_spk   = torch.zeros(1, 1,  device=DEVICE, dtype=torch.long)
    test_lang  = torch.zeros(1, 1,  device=DEVICE, dtype=torch.long)
    
    # Try different argument names
    for kwargs in [
        {"input_ids": test_units, "speaker_id": test_spk, "lang_id": test_lang},
        {"input_ids": test_units, "spkr_id": test_spk, "lang_id": test_lang},
        {"input_ids": test_units, "speaker_id": test_spk, "tgt_lang": test_lang},
    ]:
        try:
            out = voc(**kwargs)
            print(f"✓ Works with args: {list(kwargs.keys())}")
            if hasattr(out, "waveform"):
                print(f"  → out.waveform: {out.waveform.shape}")
            elif isinstance(out, torch.Tensor):
                print(f"  → tensor: {out.shape}")
            elif isinstance(out, (tuple, list)):
                print(f"  → tuple[0]: {out[0].shape}")
            break
        except Exception as e:
            print(f"✗ {list(kwargs.keys())}: {e}")
```
OUTPUT:
```text
Vocoder type: SeamlessM4Tv2CodeHifiGan

forward() signature:
  input_ids: <class 'torch.LongTensor'>
  speaker_id: <class 'torch.Tensor'>
  lang_id: <class 'torch.Tensor'>
  kwargs: ?

Testing vocoder directly...
✓ Works with args: ['input_ids', 'speaker_id', 'lang_id']
  → tuple[0]: torch.Size([1, 16000])
```

### Cell 73 (code, score=29)
```python
# ─────────────────────────────────────────────────────────────
# CELL 6-INSPECT — Check vocoder internals
# ─────────────────────────────────────────────────────────────

import inspect

voc = model_onnx.vocoder
print(f"Vocoder type: {type(voc).__name__}")
print("\nChildren:")
for name, mod in voc.named_children():
    print(f"  {name}: {type(mod).__name__}")

# Check if duration predictor exists
if hasattr(voc, 'dur_predictor'):
    print(f"\n✓ Found dur_predictor: {type(voc.dur_predictor).__name__}")
elif hasattr(voc, 'duration_predictor'):
    print(f"\n✓ Found duration_predictor: {type(voc.duration_predictor).__name__}")
else:
    print("\n⚠ No duration_predictor found")
    for name, mod in voc.named_modules():
        if 'dur' in name.lower():
            print(f"  Found: {name} -> {type(mod).__name__}")

# Check if HiFi-GAN exists
if hasattr(voc, 'hifi_gan'):
    print(f"✓ Found hifi_gan: {type(voc.hifi_gan).__name__}")
```
OUTPUT:
```text
Vocoder type: SeamlessM4Tv2CodeHifiGan

Children:
  dur_predictor: SeamlessM4Tv2VariancePredictor
  unit_embedding: Embedding
  speaker_embedding: Embedding
  language_embedding: Embedding
  hifi_gan: SeamlessM4Tv2HifiGan

✓ Found dur_predictor: SeamlessM4Tv2VariancePredictor
✓ Found hifi_gan: SeamlessM4Tv2HifiGan
```

### Cell 74 (code, score=46)
```python
# ─────────────────────────────────────────────────────────────
# CELL 6A — Vocoder Duration Predictor Wrapper
# ─────────────────────────────────────────────────────────────

class VocoderDurationWrapper(nn.Module):
    """
    Extract duration prediction from vocoder.
    Replaces expm1 with exp to make it ONNX-compatible.
    """
    def __init__(self, vocoder):
        super().__init__()
        self.unit_embedding     = vocoder.unit_embedding
        self.speaker_embedding  = vocoder.speaker_embedding
        self.language_embedding = vocoder.language_embedding
        self.dur_predictor      = vocoder.dur_predictor

    def forward(self,
                input_ids:  torch.Tensor,
                speaker_id: torch.Tensor,
                lang_id:    torch.Tensor) -> torch.Tensor:
        """
        input_ids:  [B, T_units] long
        speaker_id: [B, 1]       long
        lang_id:    [B, 1]       long

        returns: log_durations [B, T_units] float
        """
        # Embed units, speaker, lang
        hidden_states = self.unit_embedding(input_ids).transpose(1, 2)  # [B, D, T]
        spkr = self.speaker_embedding(speaker_id).transpose(1, 2)       # [B, D, 1]
        lang = self.language_embedding(lang_id).transpose(1, 2)         # [B, D, 1]

        # Duration prediction (from hidden_states transposed back)
        log_dur = self.dur_predictor(hidden_states.transpose(1, 2))     # [B, T]

        return log_dur  # don't apply expm1 here, let Python do it


print("Building vocoder duration wrapper...")
voc_dur_w = VocoderDurationWrapper(model_onnx.vocoder).to(DEVICE).eval()

print("\nTesting vocoder duration forward...")
with torch.no_grad():
    try:
        test_units = torch.ones(1, 50, device=DEVICE, dtype=torch.long)
        test_spk   = torch.zeros(1, 1,  device=DEVICE, dtype=torch.long)
        test_lang  = torch.zeros(1, 1,  device=DEVICE, dtype=torch.long)

        log_dur = voc_dur_w(test_units, test_spk, test_lang)
        print(f"✓ log_dur: {log_dur.shape}  "
              f"range=[{log_dur.min():.2f}, {log_dur.max():.2f}]")

        # Simulate duration conversion
        durations = torch.clamp(
            torch.round(torch.expm1(log_dur)), 1, 20
        ).long()
        print(f"  durations: {durations[0].tolist()[:10]}...")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
```
OUTPUT:
```text
Building vocoder duration wrapper...

Testing vocoder duration forward...
✓ log_dur: torch.Size([1, 50])  range=[0.70, 0.71]
  durations: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]...
```

### Cell 75 (code, score=58)
```python
# ─────────────────────────────────────────────────────────────
# CELL 6B — Vocoder HiFi-GAN Wrapper
# Takes pre-upsampled hidden states, bypasses duration logic
# ─────────────────────────────────────────────────────────────

class VocoderHiFiGANWrapper(nn.Module):
    """
    HiFi-GAN only: upsampled hidden → waveform.
    Bypasses embedding and duration predictor.
    """
    def __init__(self, vocoder):
        super().__init__()
        self.hifi_gan = vocoder.hifi_gan

    def forward(self, upsampled_hidden: torch.Tensor) -> torch.Tensor:
        """
        upsampled_hidden: [B, 1792, T_upsampled] float32
                          where 1792 = 256 (lang) + 1280 (units) + 256 (speaker)

        returns: waveform [B, T_samples] float32
        """
        waveform = self.hifi_gan(upsampled_hidden)
        # HiFi-GAN returns [B, 1, T_samples], squeeze channel dim
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        return waveform


print("Building vocoder HiFi-GAN wrapper...")
voc_hifigan_w = VocoderHiFiGANWrapper(model_onnx.vocoder).to(DEVICE).eval()

print("\nTesting HiFi-GAN forward...")
with torch.no_grad():
    try:
        # Simulate full vocoder flow
        test_units = torch.ones(1, 50, device=DEVICE, dtype=torch.long)
        test_spk   = torch.zeros(1, 1,  device=DEVICE, dtype=torch.long)
        test_lang  = torch.zeros(1, 1,  device=DEVICE, dtype=torch.long)

        # Get embeddings (replicate vocoder's embedding logic)
        hidden_states = model_onnx.vocoder.unit_embedding(test_units).transpose(1, 2)  # [1, 1280, 50]
        spkr = model_onnx.vocoder.speaker_embedding(test_spk).transpose(1, 2)          # [1, 256, 1]
        lang = model_onnx.vocoder.language_embedding(test_lang).transpose(1, 2)        # [1, 256, 1]

        # Get durations
        log_dur = voc_dur_w(test_units, test_spk, test_lang)
        durations = torch.clamp(torch.round(torch.expm1(log_dur)), 1, 20).long()

        # Upsample hidden by duration
        ups_hidden = torch.repeat_interleave(
            hidden_states, durations[0], dim=2)  # [1, 1280, T_upsampled]

        # Repeat speaker/lang to match upsampled length
        T_ups = ups_hidden.shape[2]
        spkr_rep = spkr.repeat(1, 1, T_ups)  # [1, 256, T_ups]
        lang_rep = lang.repeat(1, 1, T_ups)  # [1, 256, T_ups]

        # Concat: [lang, hidden, spkr] → [1, 1792, T_ups]
        concat_hidden = torch.cat([lang_rep, ups_hidden, spkr_rep], dim=1)
        print(f"  concat_hidden: {concat_hidden.shape}")

        # HiFi-GAN forward
        waveform = voc_hifigan_w(concat_hidden)
        print(f"✓ waveform: {waveform.shape}  "
              f"({waveform.shape[-1]/16000:.2f}s audio)")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
```
OUTPUT:
```text
Building vocoder HiFi-GAN wrapper...

Testing HiFi-GAN forward...
  concat_hidden: torch.Size([1, 1792, 50])
✓ waveform: torch.Size([1, 16000])  (1.00s audio)
```

### Cell 76 (code, score=35)
```python
# ─────────────────────────────────────────────────────────────
# CELL 6C — Export Vocoder Duration (ONNX-safe)
# ─────────────────────────────────────────────────────────────

print("Exporting vocoder_duration.onnx...")
t = time.time()

dummy_units = torch.ones(1, 50, device=DEVICE, dtype=torch.long)
dummy_spk   = torch.zeros(1, 1,  device=DEVICE, dtype=torch.long)
dummy_lang  = torch.zeros(1, 1,  device=DEVICE, dtype=torch.long)

with torch.no_grad():
    torch.onnx.export(
        voc_dur_w,
        (dummy_units, dummy_spk, dummy_lang),
        str(EXPORT_DIR / "vocoder_duration.onnx"),
        input_names=["input_ids", "speaker_id", "lang_id"],
        output_names=["log_durations"],
        dynamic_axes={
            "input_ids":     {0: "batch", 1: "units"},
            "log_durations": {0: "batch", 1: "units"},
        },
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,   # JIT path
    )

mb = (EXPORT_DIR / "vocoder_duration.onnx").stat().st_size / 1e6
print(f"✓ vocoder_duration.onnx  {mb:.1f}MB  {time.time()-t:.1f}s")

del voc_dur_w
torch.cuda.empty_cache()
gc.collect()
```
OUTPUT:
```text
Exporting vocoder_duration.onnx...
✓ vocoder_duration.onnx  45.3MB  0.3s

18
```

### Cell 77 (code, score=125)
```python
# ─────────────────────────────────────────────────────────────
# CELL 6D — Export Vocoder HiFi-GAN (fp16 input fix)
# ─────────────────────────────────────────────────────────────

print("Exporting vocoder_hifigan.onnx...")
t = time.time()

# Cast to fp16 to match model weights
dummy_ups = torch.randn(1, 1792, 320, device=DEVICE, dtype=torch.float16)

with torch.no_grad():
    torch.onnx.export(
        voc_hifigan_w,
        (dummy_ups,),
        str(EXPORT_DIR / "vocoder_hifigan.onnx"),
        input_names=["upsampled_hidden"],
        output_names=["waveform"],
        dynamic_axes={
            "upsampled_hidden": {0: "batch", 2: "time"},
            "waveform":         {0: "batch", 1: "samples"},
        },
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )

mb = (EXPORT_DIR / "vocoder_hifigan.onnx").stat().st_size / 1e6
print(f"✓ vocoder_hifigan.onnx  {mb:.1f}MB  {time.time()-t:.1f}s")

del voc_hifigan_w
torch.cuda.empty_cache()
gc.collect()

# Summary
print("\n=== All ONNX Files ===")
total = 0
for f in sorted(EXPORT_DIR.glob("*.onnx")):
    mb = f.stat().st_size / 1e6
    total += mb
    print(f"  {f.name:35s}  {mb:7.1f}MB")
print(f"  {'TOTAL':35s}  {total:7.1f}MB")
```
OUTPUT:
```text
Exporting vocoder_hifigan.onnx...

ERROR: NameError name 'voc_hifigan_w' is not defined
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_57/1792120779.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     11[0m [0;32mwith[0m [0mtorch[0m[0;34m.[0m[0mno_grad[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m     torch.onnx.export(
[0;32m---> 13[0;31m         [0mvoc_hifigan_w[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     14[0m         [0;34m([0m[0mdummy_ups[0m[0;34m,[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m         [0mstr[0m[0;34m([0m[0mEXPORT_DIR[0m [0;34m/[0m [0;34m"vocoder_hifigan.onnx"[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: name 'voc_hifigan_w' is not defined
```

### Cell 78 (code, score=54)
```python
# ─────────────────────────────────────────────────────────────
# CELL 6E — Full vocoder inference helper for Cell 9
# ─────────────────────────────────────────────────────────────

def run_vocoder_onnx(unit_ids:      np.ndarray,
                     speaker_id:    np.ndarray,
                     lang_id:       np.ndarray,
                     voc_dur_sess,
                     voc_hifi_sess,
                     voc_embed_unit_w,
                     voc_embed_spkr_w,
                     voc_embed_lang_w) -> np.ndarray:
    """
    Full vocoder pipeline via 2 ONNX sessions + numpy upsampling.

    unit_ids:   [1, T_units] int64
    speaker_id: [1, 1]       int64
    lang_id:    [1, 1]       int64

    Returns: waveform [1, T_samples] float32
    """
    # 1. Duration prediction
    log_dur = voc_dur_sess.run(
        ["log_durations"],
        {"input_ids":  unit_ids,
         "speaker_id": speaker_id,
         "lang_id":    lang_id}
    )[0][0]  # [T_units]

    durations = np.clip(
        np.round(np.expm1(log_dur)).astype(np.int32), 1, 20)
    T_ups = int(durations.sum())
    print(f"    vocoder: {unit_ids.shape[1]} units → {T_ups} frames")

    # 2. Get embeddings in PyTorch (small, fast)
    with torch.no_grad():
        u_ids = torch.from_numpy(unit_ids).to(DEVICE)
        s_id  = torch.from_numpy(speaker_id).to(DEVICE)
        l_id  = torch.from_numpy(lang_id).to(DEVICE)

        hidden = model_onnx.vocoder.unit_embedding(u_ids).transpose(1, 2)   # [1, 1280, T]
        spkr   = model_onnx.vocoder.speaker_embedding(s_id).transpose(1, 2) # [1, 256, 1]
        lang   = model_onnx.vocoder.language_embedding(l_id).transpose(1, 2)# [1, 256, 1]

        # 3. Upsample
        ups_h = torch.repeat_interleave(hidden, torch.from_numpy(durations).to(DEVICE), dim=2)

        # 4. Concat [lang, hidden, spkr]
        spkr_r = spkr.repeat(1, 1, T_ups)
        lang_r = lang.repeat(1, 1, T_ups)
        concat = torch.cat([lang_r, ups_h, spkr_r], dim=1)  # [1, 1792, T_ups]

        # Convert to float16 to match HiFi-GAN weights
        concat_np = concat.cpu().to(torch.float16).numpy()

    # 5. HiFi-GAN
    waveform = voc_hifi_sess.run(
        ["waveform"],
        {"upsampled_hidden": concat_np}
    )[0]  # [1, T_samples]

    return waveform.astype(np.float32)


print("✓ run_vocoder_onnx() defined")
print("""
Cell 9 vocoder sessions:
  voc_dur_sess  = ort.InferenceSession(str(INT8_DIR/"vocoder_duration.onnx"),  opts, providers=providers)
  voc_hifi_sess = ort.InferenceSession(str(INT8_DIR/"vocoder_hifigan.onnx"),   opts, providers=providers)

Cell 9 vocoder call:
  t = time.time()
  waveform = run_vocoder_onnx(unit_ids, np.zeros((1,1),dtype=np.int64),
                               np.zeros((1,1),dtype=np.int64),
                               voc_dur_sess, voc_hifi_sess)
  print(f"✓ Vocoder: {time.time()-t:.2f}s  {waveform.shape[-1]/16000:.1f}s audio")
""")
```
OUTPUT:
```text
✓ run_vocoder_onnx() defined

Cell 9 vocoder sessions:
  voc_dur_sess  = ort.InferenceSession(str(INT8_DIR/"vocoder_duration.onnx"),  opts, providers=providers)
  voc_hifi_sess = ort.InferenceSession(str(INT8_DIR/"vocoder_hifigan.onnx"),   opts, providers=providers)

Cell 9 vocoder call:
  t = time.time()
  waveform = run_vocoder_onnx(unit_ids, np.zeros((1,1),dtype=np.int64),
                               np.zeros((1,1),dtype=np.int64),
                               voc_dur_sess, voc_hifi_sess)
  print(f"✓ Vocoder: {time.time()-t:.2f}s  {waveform.shape[-1]/16000:.1f}s audio")
```

### Cell 80 (code, score=97)
```python
# ─────────────────────────────────────────────────────────────
# CELL 4-INSPECT — Deep dive into text decoder
# ─────────────────────────────────────────────────────────────
import inspect

dec = model_onnx.text_decoder
print(f"Decoder type: {type(dec).__name__}")

# Check decoder config
if hasattr(dec, 'config'):
    cfg = dec.config
    print(f"\nDecoder config:")
    for attr in ['max_position_embeddings', 'max_source_positions',
                 'max_target_positions', 'd_model', 'decoder_attention_heads',
                 'decoder_layers', 'encoder_attention_heads']:
        if hasattr(cfg, attr):
            print(f"  {attr}: {getattr(cfg, attr)}")

# Check full model config for encoder
if hasattr(model_onnx, 'config'):
    cfg2 = model_onnx.config
    print(f"\nFull model config relevant fields:")
    for attr in ['max_position_embeddings', 'speech_encoder_max_source_positions',
                 'unit_hifi_gan_vocab_size', 'vocab_size']:
        if hasattr(cfg2, attr):
            print(f"  {attr}: {getattr(cfg2, attr)}")

# Check decoder forward signature
print(f"\nDecoder forward() signature:")
sig = inspect.signature(dec.forward)
for name, param in sig.parameters.items():
    print(f"  {name}: default={param.default!r}")

# Check cross-attention layer
layer0 = dec.layers[0]
print(f"\nDecoder layer type: {type(layer0).__name__}")
print(f"\nCross attention forward() signature:")
sig2 = inspect.signature(layer0.cross_attention.forward)
for name, param in sig2.parameters.items():
    print(f"  {name}: default={param.default!r}")

# Run a test forward and trace shapes
print("\n=== Shape tracing test ===")
model_onnx.eval()
with torch.no_grad():
    # Use actual encoder output size
    enc_h  = torch.randn(1, 72, 1024, device=DEVICE, dtype=torch.float32)
    dec_in = torch.ones(1, 1, dtype=torch.long, device=DEVICE)
    
    # Embed
    emb = model_onnx.shared(dec_in)
    print(f"decoder embed:       {emb.shape}")
    
    # Try cross attention mask shapes
    for mask_len in [72, 144, 575, 1078]:
        try:
            enc_mask = torch.ones(1, mask_len, device=DEVICE, dtype=torch.long)
            out = model_onnx.text_decoder(
                inputs_embeds=emb,
                encoder_hidden_states=enc_h,
                encoder_attention_mask=enc_mask,
                use_cache=False,
            )
            print(f"enc_mask len={mask_len}: ✓ output={get_hidden(out).shape}")
        except Exception as e:
            print(f"enc_mask len={mask_len}: ✗ {str(e)[:80]}")

# Check what shape the attention bias expects
print("\n=== Checking internal attention bias shapes ===")
for name, buf in dec.named_buffers():
    print(f"  buffer {name}: {buf.shape}")

for name, param in dec.named_parameters():
    if 'bias' in name.lower() or 'embed' in name.lower():
        if any(s == 575 or s == 1078 or s == 72 for s in param.shape):
            print(f"  param {name}: {param.shape}  ← matches!")
```
OUTPUT:
```text
Decoder type: SeamlessM4Tv2Decoder

Decoder config:
  max_position_embeddings: 4096
  decoder_attention_heads: 16
  decoder_layers: 14
  encoder_attention_heads: 16

Full model config relevant fields:
  max_position_embeddings: 4096
  unit_hifi_gan_vocab_size: 10000
  vocab_size: 22767

Decoder forward() signature:
  input_ids: default=None
  attention_mask: default=None
  encoder_hidden_states: default=None
  encoder_attention_mask: default=None
  past_key_values: default=None
  inputs_embeds: default=None
  use_cache: default=None
  output_attentions: default=None
  output_hidden_states: default=None
  return_dict: default=None
  cache_position: default=None
  kwargs: default=<class 'inspect._empty'>

Decoder layer type: SeamlessM4Tv2DecoderLayer

Cross attention forward() signature:
  hidden_states: default=<class 'inspect._empty'>
  encoder_hidden_states: default=None
  past_key_values: default=None
  attention_mask: default=None
  output_attentions: default=False
  cache_position: default=None

=== Shape tracing test ===
decoder embed:       torch.Size([1, 1, 1024])
enc_mask len=72: ✓ output=torch.Size([1, 1, 1024])
enc_mask len=144: ✗ The size of tensor a (72) must match the size of tensor b (144) at non-singleton
enc_mask len=575: ✗ The size of tensor a (72) must match the size of tensor b (575) at non-singleton
enc_mask len=1078: ✗ The size of tensor a (72) must match the size of tensor b (1078) at non-singleto

=== Checking internal attention bias shapes ===
  buffer embed_positions.weights: torch.Size([4098, 1024])
```

### Cell 81 (code, score=59)
```python
# ─────────────────────────────────────────────────────────────
# CELL 4-INSPECT2 — Check what 575 and 1078 actually are
# ─────────────────────────────────────────────────────────────

print("=== Searching for 575 and 1078 in model ===")
for name, buf in model_onnx.named_buffers():
    if any(s in [575, 1078, 4606] for s in buf.shape):
        print(f"  buffer {name}: {buf.shape}")

for name, param in model_onnx.named_parameters():
    if any(s in [575, 1078, 4606] for s in param.shape):
        print(f"  param  {name}: {param.shape}")

# Check speech encoder output size relationship
print("\n=== Speech encoder output analysis ===")
print(f"Input features shape: {inp_feat.shape}")
print(f"Input mask shape:     {inp_mask.shape}")
print(f"Encoder output:       (1, 72, 1024)")
print(f"Input mask length:    {inp_mask.shape[1]}")
print(f"575 / input_mask_len: {575 / inp_mask.shape[1]:.4f}")
print(f"1078 / enc_out_len:   {1078 / 72:.4f}")
print(f"1078 / 2:             {1078 / 2}")
print(f"inp_mask.sum():       {inp_mask.sum()}")

# Check if there's a subsampling factor
print("\n=== Subsampling analysis ===")
print(f"inp_feat.shape[1]:                  {inp_feat.shape[1]}")
print(f"enc_out len (72):                   72")
print(f"inp_feat / enc_out:                 {inp_feat.shape[1] / 72:.2f}")
print(f"inp_mask length:                    {inp_mask.shape[1]}")
print(f"inp_mask / enc_out:                 {inp_mask.shape[1] / 72:.2f}")
print(f"inp_mask sum (active):              {inp_mask[0].sum()}")
print(f"inp_mask sum / enc_out:             {inp_mask[0].sum() / 72:.2f}")

# The text decoder likely expects SPEECH encoder attention mask
# not the subsampled encoder output mask
# Let's check what the original model uses
print("\n=== Testing with speech encoder subsampled mask ===")
with torch.no_grad():
    # Compute what mask the speech encoder actually produces
    feat_t = torch.from_numpy(inp_feat).to(DEVICE)
    mask_t = torch.from_numpy(inp_mask).to(DEVICE)
    
    # Run actual speech encoder to get its output + mask
    enc_out_real = model_onnx.speech_encoder(
        input_features=feat_t,
        attention_mask=mask_t,
    )
    
    print(f"Speech encoder output type: {type(enc_out_real)}")
    if hasattr(enc_out_real, 'last_hidden_state'):
        print(f"  last_hidden_state: {enc_out_real.last_hidden_state.shape}")
    
    # Check if speech encoder returns a mask
    for attr in dir(enc_out_real):
        if not attr.startswith('_'):
            val = getattr(enc_out_real, attr, None)
            if isinstance(val, torch.Tensor):
                print(f"  {attr}: {val.shape}")
```
OUTPUT:
```text
=== Searching for 575 and 1078 in model ===

=== Speech encoder output analysis ===
Input features shape: (1, 575, 160)
Input mask shape:     (1, 575)
Encoder output:       (1, 72, 1024)
Input mask length:    575
575 / input_mask_len: 1.0000
1078 / enc_out_len:   14.9722
1078 / 2:             539.0
inp_mask.sum():       575

=== Subsampling analysis ===
inp_feat.shape[1]:                  575
enc_out len (72):                   72
inp_feat / enc_out:                 7.99
inp_mask length:                    575
inp_mask / enc_out:                 7.99
inp_mask sum (active):              575
inp_mask sum / enc_out:             7.99

=== Testing with speech encoder subsampled mask ===
Speech encoder output type: <class 'transformers.modeling_outputs.Wav2Vec2BaseModelOutput'>
  last_hidden_state: torch.Size([1, 72, 1024])
  last_hidden_state: torch.Size([1, 72, 1024])
```

### Cell 82 (code, score=57)
```python
# ─────────────────────────────────────────────────────────────
# CELL 4-INSPECT3 — Run full model forward to see what mask
#                   the text decoder actually receives
# ─────────────────────────────────────────────────────────────
import types

# Monkey-patch text decoder to print what mask it receives
original_forward = model_onnx.text_decoder.forward

def patched_forward(self_or_hidden, *args, **kwargs):
    print(f"\n  text_decoder received:")
    if 'encoder_hidden_states' in kwargs:
        print(f"    encoder_hidden_states: {kwargs['encoder_hidden_states'].shape}")
    if 'encoder_attention_mask' in kwargs:
        mask = kwargs['encoder_attention_mask']
        if mask is not None:
            print(f"    encoder_attention_mask: {mask.shape}  sum={mask.sum()}")
        else:
            print(f"    encoder_attention_mask: None")
    return original_forward(*args, **kwargs)

# Run the real model forward
print("Running real SeamlessM4Tv2 forward to trace mask shapes...")
with torch.no_grad():
    try:
        feat_t = torch.from_numpy(inp_feat).to(DEVICE)
        mask_t = torch.from_numpy(inp_mask).to(DEVICE)
        
        # Use generate to trigger the full pipeline
        out = model_onnx.generate(
            input_features=feat_t,
            attention_mask=mask_t,
            tgt_lang="ben",
            generate_speech=False,   # text only first
            max_new_tokens=10,
        )
        print(f"\nGenerate output type: {type(out)}")
        if isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                if hasattr(o, 'shape'):
                    print(f"  [{i}]: {o.shape}")
                else:
                    print(f"  [{i}]: {type(o)}")
        elif hasattr(out, 'sequences'):
            print(f"  sequences: {out.sequences.shape}")
        else:
            print(f"  shape: {out.shape}")

    except Exception as e:
        print(f"Generate failed: {e}")
        import traceback
        traceback.print_exc()
```
OUTPUT:
```text
Running real SeamlessM4Tv2 forward to trace mask shapes...
Generate failed: The following `model_kwargs` are not used by the model: ['generate_speech'] (note: typos in the generate arguments will also show up in this list)

Traceback (most recent call last):
  File "/tmp/ipykernel_57/1565500629.py", line 30, in <cell line: 0>
    out = model_onnx.generate(
          ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py", line 3757, in generate
    text_generation_output = super().generate(input_features, **kwargs_text)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/generation/utils.py", line 2496, in generate
    self._validate_model_kwargs(model_kwargs.copy())
  File "/usr/local/lib/python3.12/dist-packages/transformers/generation/utils.py", line 1661, in _validate_model_kwargs
    raise ValueError(
ValueError: The following `model_kwargs` are not used by the model: ['generate_speech'] (note: typos in the generate arguments will also show up in this list)
```

### Cell 83 (code, score=64)
```python
# ─────────────────────────────────────────────────────────────
# CELL 4-INSPECT4 — Check what mask shape the model creates
#                   internally during speech encoder + decoder
# ─────────────────────────────────────────────────────────────

print("Checking _compute_sub_sample_lengths or similar...")

# Check if the model has a mask computation method
for name in dir(model_onnx):
    if 'mask' in name.lower() or 'sub' in name.lower() or 'sample' in name.lower():
        print(f"  model attr: {name}")

# Check speech encoder adapter output
print("\n=== Testing mask through full encode path ===")
with torch.no_grad():
    feat_t = torch.from_numpy(inp_feat).to(DEVICE)
    mask_t = torch.from_numpy(inp_mask).to(DEVICE)

    # Step through manually
    # 1. Feature projection
    fp_out = model_onnx.speech_encoder.feature_projection(feat_t)
    if isinstance(fp_out, tuple):
        fp_out = fp_out[0]
    print(f"feature_projection out: {fp_out.shape}")

    # 2. Encoder
    enc_out2 = model_onnx.speech_encoder.encoder(
        hidden_states=fp_out, attention_mask=mask_t)
    h = get_hidden(enc_out2)
    print(f"encoder out: {h.shape}")

    # 3. IFN
    ifn_out = model_onnx.speech_encoder.intermediate_ffn(h)
    h = get_hidden(ifn_out)
    print(f"intermediate_ffn out: {h.shape}")

    # 4. Adapter — this likely subsamples AND returns a new mask
    print(f"\nAdapter type: {type(model_onnx.speech_encoder.adapter).__name__}")
    print("Adapter forward signature:")
    sig = inspect.signature(model_onnx.speech_encoder.adapter.forward)
    for pname, param in sig.parameters.items():
        print(f"  {pname}: default={param.default!r}")

    # Try adapter with mask return
    try:
        adp_out = model_onnx.speech_encoder.adapter(
            hidden_states=h, attention_mask=mask_t)
        if isinstance(adp_out, tuple):
            print(f"\nAdapter returns tuple of {len(adp_out)}:")
            for i, o in enumerate(adp_out):
                if isinstance(o, torch.Tensor):
                    print(f"  [{i}]: {o.shape}")
                else:
                    print(f"  [{i}]: {o}")
        else:
            print(f"\nAdapter returns: {get_hidden(adp_out).shape}")
    except Exception as e:
        print(f"Adapter error: {e}")
        traceback.print_exc()
```
OUTPUT:
```text
Checking _compute_sub_sample_lengths or similar...
  model attr: __init_subclass__
  model attr: __subclasshook__
  model attr: _compute_sub_sample_lengths_from_attention_mask
  model attr: _count_character_length_in_subword
  model attr: _hard_upsample
  model attr: _indices_to_subwords
  model attr: _prepare_attention_mask_for_generation
  model attr: _sample
  model attr: create_extended_attention_mask_for_decoder
  model attr: get_extended_attention_mask
  model attr: get_submodule
  model attr: invert_attention_mask
  model attr: set_submodule
  model attr: warn_if_padding_and_no_attention_mask

=== Testing mask through full encode path ===
feature_projection out: torch.Size([1, 575, 1024])
encoder out: torch.Size([1, 575, 1024])
intermediate_ffn out: torch.Size([1, 575, 1024])

Adapter type: SeamlessM4Tv2ConformerAdapter
Adapter forward signature:
  hidden_states: default=<class 'inspect._empty'>
  attention_mask: default=<class 'inspect._empty'>

Adapter returns: torch.Size([1, 72, 1024])
```

### Cell 86 (code, score=127)
```python
# ─────────────────────────────────────────────────────────────
# CELL 7 — Simplify + Convert to FP32 (skip text_decoder)
# ─────────────────────────────────────────────────────────────
import onnx
import onnxsim
from onnx import numpy_helper
import numpy as np
import shutil

def convert_model_to_fp32(model):
    """Convert all fp16 tensors/types to fp32 in an ONNX model."""
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT16:
            arr = numpy_helper.to_array(initializer).astype(np.float32)
            new_init = numpy_helper.from_array(arr, name=initializer.name)
            initializer.CopyFrom(new_init)

    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    for out in model.graph.output:
        if out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            out.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT16:
                attr.i = onnx.TensorProto.FLOAT

    return model


print("Simplifying + converting to fp32...")
for f in sorted(EXPORT_DIR.glob("*.onnx")):
    try:
        print(f"  {f.name}...", end=" ")
        
        # Skip simplification for text_decoder (it bakes in shape constants)
        if f.name == "text_decoder.onnx":
            model = onnx.load(str(f))
            model = convert_model_to_fp32(model)
            onnx.save(model, str(SIM_DIR / f.name))
            print(f"(skipped simplification, fp32 only)")
            continue
        
        orig_size = f.stat().st_size / 1e6
        model = onnx.load(str(f))
        model = convert_model_to_fp32(model)
        
        # Simplify
        sim_model, ok = onnxsim.simplify(model)
        if not ok:
            print("simplify failed, using converted model")
            sim_model = model
        
        onnx.save(sim_model, str(SIM_DIR / f.name))
        new_size = (SIM_DIR / f.name).stat().st_size / 1e6
        print(f"{orig_size:.1f}→{new_size:.1f}MB")
        
    except Exception as e:
        print(f"ERROR: {e}")
        shutil.copy(f, SIM_DIR / f.name)

print("\n=== Simplified (FP32) ===")
total = 0
for f in sorted(SIM_DIR.glob("*.onnx")):
    mb = f.stat().st_size / 1e6
    total += mb
    print(f"  {f.name:35s}  {mb:7.1f}MB")
print(f"  {'TOTAL':35s}  {total:7.1f}MB")
```
OUTPUT:
```text
Simplifying + converting to fp32...
  speech_encoder.onnx... 786.9→1573.3MB
  t2u_decoder.onnx... 0.3→511.3MB
  t2u_duration.onnx... 0.0→3.9MB
  t2u_embed.onnx... 0.0→41.3MB
  t2u_encoder.onnx... 0.3→335.8MB
  text_decoder.onnx... (skipped simplification, fp32 only)
  vocoder_duration.onnx... 45.3→90.6MB
  vocoder_hifigan.onnx... 38.5→76.9MB

=== Simplified (FP32) ===
  speech_encoder.onnx                   1573.3MB
  t2u_decoder.onnx                       511.3MB
  t2u_duration.onnx                        3.9MB
  t2u_embed.onnx                          41.3MB
  t2u_encoder.onnx                       335.8MB
  text_decoder.onnx                     1614.3MB
  vocoder_duration.onnx                   90.6MB
  vocoder_hifigan.onnx                    76.9MB
  vocoder_hifigan_fp32.onnx               76.9MB
  TOTAL                                 4324.4MB
```

### Cell 87 (code, score=74)
```python
# ─────────────────────────────────────────────────────────────
# CELL 8 — INT8 Quantization
# ─────────────────────────────────────────────────────────────
from onnxruntime.quantization import quantize_dynamic, QuantType

print("Quantizing to INT8...")
total_orig = total_quant = 0

for f in sorted(SIM_DIR.glob("*.onnx")):
    out = INT8_DIR / f.name
    try:
        quantize_dynamic(
            str(f),
            str(out),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        orig = f.stat().st_size / 1e6
        q    = out.stat().st_size / 1e6
        total_orig  += orig
        total_quant += q
        print(f"  ✓ {f.name}: {orig:.1f}→{q:.1f}MB "
              f"({100*(1-q/orig):.0f}% smaller)")
    except Exception as e:
        print(f"  ✗ {f.name}: {e}")

print(f"\n  Total: {total_orig:.1f}MB → {total_quant:.1f}MB "
      f"({100*(1-total_quant/total_orig):.0f}% reduction)")
```
OUTPUT:
```text
Quantizing to INT8...
  ✗ speech_encoder-inferred.onnx: Unable to parse proto from file: /kaggle/working/seamless_onnx_sim/speech_encoder-inferred.onnx. Please check if it is a valid protobuf file of proto. 
  ✗ speech_encoder.onnx: Error parsing message with type 'onnx.ModelProto'

WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 

  ✗ t2u_decoder.onnx: Failed to find proper ai.onnx domain
  ✗ t2u_duration.onnx: Failed to find proper ai.onnx domain

WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md
```

### Cell 88 (code, score=50)
```python
# Quick fix: convert vocoder_hifigan.onnx to fp32 and re-quantize
import onnx
from onnx import numpy_helper
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType

def convert_model_to_fp32(model):
    """Convert all fp16 tensors/types to fp32."""
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT16:
            arr = numpy_helper.to_array(initializer).astype(np.float32)
            new_init = numpy_helper.from_array(arr, name=initializer.name)
            initializer.CopyFrom(new_init)

    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    for out in model.graph.output:
        if out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            out.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT16:
                attr.i = onnx.TensorProto.FLOAT

    return model

# Load, convert, save
print("Fixing vocoder_hifigan.onnx...")
model = onnx.load(str(SIM_DIR / "vocoder_hifigan.onnx"))
model = convert_model_to_fp32(model)
onnx.save(model, str(SIM_DIR / "vocoder_hifigan_fp32.onnx"))

# Re-quantize
quantize_dynamic(
    str(SIM_DIR / "vocoder_hifigan_fp32.onnx"),
    str(INT8_DIR / "vocoder_hifigan.onnx"),
    weight_type=QuantType.QInt8,
    per_channel=True,
)
print("✓ vocoder_hifigan.onnx fixed and re-quantized")
```

### Cell 89 (code, score=38)
```python
# ─────────────────────────────────────────────────────────────
# CELL 9 — Full S2ST Validation (all fixes applied)
# ─────────────────────────────────────────────────────────────
import onnxruntime as ort
import numpy as np
import time
import torch

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
providers = ["CPUExecutionProvider"]

print("Loading sessions...")
enc_sess      = ort.InferenceSession(str(INT8_DIR / "speech_encoder.onnx"),    opts, providers=providers)
dec_sess      = ort.InferenceSession(str(INT8_DIR / "text_decoder.onnx"),      opts, providers=providers)
t2u_embed_sess= ort.InferenceSession(str(INT8_DIR / "t2u_embed.onnx"),         opts, providers=providers)
t2u_enc_sess  = ort.InferenceSession(str(INT8_DIR / "t2u_encoder.onnx"),       opts, providers=providers)
t2u_dur_sess  = ort.InferenceSession(str(INT8_DIR / "t2u_duration.onnx"),      opts, providers=providers)
t2u_dec_sess  = ort.InferenceSession(str(INT8_DIR / "t2u_decoder.onnx"),       opts, providers=providers)
voc_dur_sess  = ort.InferenceSession(str(INT8_DIR / "vocoder_duration.onnx"),  opts, providers=providers)
voc_hifi_sess = ort.InferenceSession(str(INT8_DIR / "vocoder_hifigan.onnx"),   opts, providers=providers)
print("✓ All sessions loaded")

# ── Load test sample ─────────────────────────────────────────
sample = eval_samples[0]
wav    = sample["wav"]
ref    = sample["ref"]
print(f"\nTest: {ref[:60]}")

feat     = processor(audio=wav, sampling_rate=16000,
                     return_tensors="np", padding=True)
inp_feat = feat["input_features"].astype(np.float32)
inp_mask = feat.get(
    "attention_mask",
    np.ones(inp_feat.shape[:2], dtype=np.int64)
).astype(np.int64)

# ── 1. Speech Encoder ────────────────────────────────────────
print("\n[1/5] Speech Encoder...")
t = time.time()
enc_out, enc_sub_mask = enc_sess.run(
    ["encoder_hidden_states", "encoder_attention_mask"],
    {"input_features": inp_feat, "attention_mask": inp_mask}
)
print(f"  ✓ {time.time()-t:.2f}s")
print(f"    hidden:   {enc_out.shape}")
print(f"    sub_mask: {enc_sub_mask.shape}")

# ── 2. Text Decoder (greedy) ─────────────────────────────────
print("[2/5] Text Decoder...")
t = time.time()
dec_ids = np.array([[BOS_ID_S]], dtype=np.int64)

for step in range(128):
    logits = dec_sess.run(
        ["logits"],
        {"decoder_input_ids":      dec_ids,
         "encoder_hidden_states":  enc_out,
         "encoder_attention_mask": enc_sub_mask}  # ← subsampled mask
    )[0]
    next_tok = int(np.argmax(logits[0, -1, :]))
    if next_tok == EOS_ID_S:
        break
    dec_ids = np.concatenate([dec_ids, [[next_tok]]], axis=1)

text = processor.tokenizer.decode(
    dec_ids[0, 1:].tolist(), skip_special_tokens=True)
print(f"  ✓ {time.time()-t:.2f}s  {dec_ids.shape[1]-1} tokens")
print(f"    Text: {text[:80]}")

# ── 3. T2U ───────────────────────────────────────────────────
print("[3/5] T2U...")
t = time.time()
t2u_mask = np.ones_like(dec_ids)

embeds    = t2u_embed_sess.run(
    ["inputs_embeds"], {"input_ids": dec_ids})[0]

t2u_enc_h = t2u_enc_sess.run(
    ["encoder_hidden_states"],
    {"inputs_embeds": embeds, "attention_mask": t2u_mask})[0]

log_dur   = t2u_dur_sess.run(
    ["log_durations"],
    {"encoder_hidden_states": t2u_enc_h})[0][0]

durations = np.clip(np.round(np.exp(log_dur)).astype(np.int32), 1, 20)
ups_h     = np.repeat(t2u_enc_h[0], durations, axis=0)[np.newaxis]

unit_logits = t2u_dec_sess.run(
    ["unit_logits"],
    {"upsampled_hidden": ups_h.astype(np.float32)})[0]

unit_ids = np.argmax(unit_logits[0], axis=-1)[np.newaxis].astype(np.int64)
print(f"  ✓ {time.time()-t:.2f}s  {unit_ids.shape[1]} units")

# ── 4. Vocoder ───────────────────────────────────────────────
print("[4/5] Vocoder...")
t = time.time()

speaker_id = np.zeros((1, 1), dtype=np.int64)
lang_id    = np.zeros((1, 1), dtype=np.int64)

# 4a. Duration prediction
voc_log_dur  = voc_dur_sess.run(
    ["log_durations"],
    {"input_ids":  unit_ids,
     "speaker_id": speaker_id,
     "lang_id":    lang_id})[0][0]

voc_durations = np.clip(
    np.round(np.expm1(voc_log_dur)).astype(np.int32), 1, 20)
T_ups = int(voc_durations.sum())
print(f"    {unit_ids.shape[1]} units → {T_ups} frames")

# 4b. Embeddings + upsample (in PyTorch, fast)
with torch.no_grad():
    u_t = torch.from_numpy(unit_ids).to(DEVICE)
    s_t = torch.from_numpy(speaker_id).to(DEVICE)
    l_t = torch.from_numpy(lang_id).to(DEVICE)

    hidden = model_onnx.vocoder.unit_embedding(u_t).transpose(1, 2)   # [1,1280,T]
    spkr   = model_onnx.vocoder.speaker_embedding(s_t).transpose(1, 2)# [1,256,1]
    lang_e = model_onnx.vocoder.language_embedding(l_t).transpose(1, 2)# [1,256,1]

    ups_h_voc = torch.repeat_interleave(
        hidden,
        torch.from_numpy(voc_durations).to(DEVICE),
        dim=2)                                                          # [1,1280,T_ups]

    concat = torch.cat([
        lang_e.repeat(1, 1, T_ups),
        ups_h_voc,
        spkr.repeat(1, 1, T_ups)
    ], dim=1)                                                           # [1,1792,T_ups]

    concat_np = concat.cpu().numpy().astype(np.float32)

# 4c. HiFi-GAN
waveform = voc_hifi_sess.run(
```

### Cell 90 (code, score=38)
```python
# ─────────────────────────────────────────────────────────────
# CELL 10 — Save Metadata + Package + Upload
# ─────────────────────────────────────────────────────────────
import subprocess

# Save metadata
meta = {
    "vocab_size":    int(S_VOCAB),
    "bos_token_id":  int(BOS_ID_S),
    "eos_token_id":  int(EOS_ID_S),
    "pad_token_id":  int(PAD_ID_S),
    "sample_rate":   16000,
    "n_mels":        160,
    "max_tokens":    256,
    "supported_langs": ["ben", "eng", "arb", "hin", "cmn"],
    "components": {
        "speech_encoder": "speech_encoder.onnx",
        "text_decoder":   "text_decoder.onnx",
        "t2u_model":      "t2u_model.onnx",
        "vocoder":        "vocoder.onnx",
    }
}

with open(INT8_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# Save tokenizer
tok_dir = INT8_DIR / "tokenizer"
tok_dir.mkdir(exist_ok=True)
processor.tokenizer.save_pretrained(str(tok_dir))

# Save vocab remap
np.save(str(INT8_DIR / "vocab_remap.npy"),
        OLD_TO_NEW_CPU.numpy().astype(np.int32))

# Package
PKG = Path("/kaggle/working/seamless_android_onnx.zip")
print("Packaging...")
with zipfile.ZipFile(PKG, "w",
                     compression=zipfile.ZIP_DEFLATED,
                     compresslevel=6) as zf:
    for fp in sorted(INT8_DIR.rglob("*")):
        if fp.is_file():
            arc = fp.relative_to(INT8_DIR)
            zf.write(fp, arc)
            print(f"  + {arc}  ({fp.stat().st_size/1e6:.1f}MB)")

print(f"\n✓ Package: {PKG.stat().st_size/1e6:.1f}MB")

r = subprocess.run(
    f'rclone copy "{PKG}" "{GDRIVE_ROOT}/android_onnx/"',
    shell=True, capture_output=True, text=True)
print("✓ Uploaded to Drive" if r.returncode == 0
      else f"⚠ Upload: {r.stderr[:100]}")

print("\n=== FINAL PACKAGE CONTENTS ===")
total = 0
for fp in sorted(INT8_DIR.rglob("*")):
    if fp.is_file():
        mb = fp.stat().st_size / 1e6
        total += mb
        print(f"  {str(fp.relative_to(INT8_DIR)):40s}  {mb:7.1f}MB")
print(f"  {'TOTAL':40s}  {total:7.1f}MB")
```

### Cell 94 (code, score=3)
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

### Cell 95 (code, score=15)
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

### Cell 96 (code, score=55)
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

### Cell 97 (code, score=183)
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

### Cell 98 (code, score=36)
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

### Cell 99 (code, score=76)
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

### Cell 100 (code, score=30)
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

### Cell 103 (code, score=21)
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

### Cell 104 (code, score=129)
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

### Cell 105 (code, score=54)
```python
# ─────────────────────────────────────────────────────────────────────────────
# SAVE PHASE 7 BEST
# ─────────────────────────────────────────────────────────────────────────────

print('Loading best Phase 7 checkpoint...')
best_ckpt = load_latest_checkpoint('phase7_ft')
model_p7.load_state_dict(best_ckpt['model_state'], strict=False)
best_step  = best_ckpt.get('best_chrf_step', '?')
best_score = best_ckpt.get('best_chrf', 0.0)
del best_ckpt
gc.collect()

print(f'Best checkpoint: step={best_step}  ASR-ChrF={best_score:.2f}')

# Option A: Save as PEFT checkpoint (smaller, keeps LoRA structure)
model_p7.eval()
save_model_to_drive(model_p7, processor, 'phase7_peft_best')
print('✓ Saved PEFT checkpoint (phase7_peft_best)')

# Option B: Merge and save as plain model
print('Merging LoRA into weights...')
merged_p7 = model_p7.merge_and_unload()
merged_p7.eval()
print(f'Merged params: {count_params(merged_p7):.1f}M')
save_model_to_drive(merged_p7, processor, 'phase7_final_merged')
print('✓ Saved merged model (phase7_final_merged)')

# Full benchmark on merged model
results, summary = run_benchmark_asr(
    merged_p7, list(eval_samples), 'P7_Final', save_n=4)
detailed = compute_detailed_summary(
    results, 'P7_Final', summary['params_M'])
store_summary(summary)
store_detailed_summary(detailed)
print_detailed_summary_table('P7_Final')
plot_detailed_phase_comparison()
```

### Cell 106 (code, score=49)
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

### Cell 109 (code, score=5)
```python
# save_checkpoint(dict(
#     student_state   = student.state_dict(),
#     optimizer_state = optimizer.state_dict(),
#     scheduler_state = scheduler.state_dict(),
#     opt_step        = 560,
#     best_chrf       = 35.00,
# ), 'phase6_kd', 560, keep=3)
```

### Cell 110 (code, score=41)
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

### Cell 111 (code, score=62)
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

### Cell 112 (code, score=6)
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

### Cell 113 (code, score=5)
```python
# final_step, final_chrf = run_phase6_training()
```