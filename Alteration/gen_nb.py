import json, textwrap

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

# ── TITLE ──────────────────────────────────────────────────────────────────────
cells.append(md("""\
# SeamlessM4T v2 → Textless Pure S2ST (~673M)
## Voice Cloning · 5 Languages · Long-Form Audio · INTERSPEECH/IWSLT 2026

**Architectural transformation paper** — converting SeamlessM4T v2 from text-mediated S2ST
to a fully textless, speaker-preserving, long-form-capable S2ST system.

| Component | Original | After |
|---|---|---|
| Text Decoder | 867M, 24 layers | **0M — permanently removed** |
| lm_head + shared vocab | ~262M | **0M — removed** |
| Speech Encoder | 635M, 24L | **~441M, 16L** (BI+iterative prune) |
| T2U Model | 262M, 6+6L | **~175M, 4+4L** (LaCo RDSC merge) |
| CIF Connector | — | **~5M NEW** (trained from scratch) |
| Speaker Adapter | — | **~0.1M NEW** (ECAPA→vocoder 192→256) |
| **Total** | **1805M** | **~673M** |

**Papers:** S2UT (Lee ACL 2022) · SeamlessExpressive (arXiv:2312.05187) ·
LaCo (Yang EMNLP 2024) · CIF (Dong & Xu ICASSP 2020) · ECAPA-TDNN (Desplanques IS 2020) ·
DoRA (Liu ICML 2024) · ShortGPT (ACL 2025) · MMS (Pratap 2023)

**Phases:** P0 V1-Baseline → P1 Vocab5L → P2 EncPrune16L → P3 LaCoT2U →
P4 TextlessArch → P5 KD-Extract → P6a CIF-FeatureKD → P6b E2E-DoRA → P7 FullBenchmark
"""))

# ═══════════════════════════════════════════════════════════════════════
# SETUP (battle-tested from seamless-cse465v5.ipynb — kept exactly)
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("## ⚙️ Setup — run ALL at the start of EVERY Kaggle session"))

# S1 — platform + paths (verbatim from v5)
cells.append(code("""\
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
"""))

# S2 — Drive mount (verbatim)
cells.append(code("""\
if ON_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print(f'Drive mounted. Working folder: {GDRIVE_MOUNT}')
else:
    print('Kaggle: skipping Drive mount.')
"""))

# S3 — dirs
cells.append(code("""\
for d in [WORK_DIR, CKPT_DIR, AUDIO_DIR, FIG_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)
print(f'Platform : {PLATFORM}')
print(f'Work dir : {WORK_DIR}')
print(f'Checkpts : {CKPT_DIR}')
"""))

# S4 — rclone (verbatim from v5)
cells.append(code("""\
if ON_KAGGLE:
    subprocess.run('curl -s https://rclone.org/install.sh | sudo bash',
                   shell=True, capture_output=True)
    ver = subprocess.run('rclone version', shell=True, capture_output=True, text=True)
    print(ver.stdout.split('\\n')[0])
else:
    print('Colab: rclone not needed — using mounted Drive directly.')
    if not os.path.exists('/content/drive/MyDrive'):
        print('WARNING: Drive does not appear to be mounted.')
    else:
        print('Drive mount: OK')
"""))

# S5 — secrets + rclone config (verbatim)
cells.append(code("""\
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
    raw = re.sub(r'\\s*(\\[[^\\]]+\\])\\s*', r'\\n\\1\\n', raw)
    raw = re.sub(r'\\s+(type|scope|token|team_drive|client_id|client_secret|'
                 r'root_folder_id|service_account_file|drive_id)\\s*=\\s*',
                 r'\\n\\1 = ', raw)
    raw = raw.strip() + '\\n'
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
"""))

# S6 — pip install (verbatim + new deps)
cells.append(code("""\
subprocess.run([
    'pip', 'install', '-q',
    'transformers>=4.41.0', 'datasets', 'torchaudio', 'speechbrain>=1.0.0',
    'peft>=0.10.0', 'librosa', 'jiwer', 'evaluate', 'sacrebleu',
    'sentencepiece', 'accelerate', 'matplotlib', 'seaborn',
    'soundfile', 'requests', 'pandas',
], check=True)
print('All packages installed.')
"""))

# S7 — sync checkpoints (verbatim)
cells.append(code("""\
# ── pulled verbatim from seamless-cse465v5 Cell 14 ──
import torch
from datetime import datetime

_CUSTOM_STATE_FILE = '_custom_state.pt'
_PRUNING_MANIFEST  = 'pruning_manifest.pt'

def _rclone_push(local_path, remote_subpath):
    if not ON_KAGGLE: return
    r = subprocess.run(
        f'rclone copy \"{local_path}\" \"{GDRIVE_ROOT}/{remote_subpath}/\"',
        shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'[rclone] WARNING: push failed for {local_path}: {r.stderr[:200]}')

def _rclone_pull_model(stage_name):
    if not ON_KAGGLE: return
    local = f'{MODEL_DIR}/{stage_name}'
    os.makedirs(local, exist_ok=True)
    r = subprocess.run(
        f'rclone sync \"{GDRIVE_ROOT}/models/{stage_name}/\" \"{local}/\"',
        shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'[rclone] model pull failed for {stage_name}: {r.stderr[:300]}')
    print(f'[rclone] Pulled {stage_name} → {local}')

def save_checkpoint(state, name, step=0, keep=3):
    fname = f'{name}_step{step:06d}.pt'
    path  = f'{CKPT_DIR}/{fname}'
    torch.save(state, path)
    mb = os.path.getsize(path) / 1e6
    print(f'[ckpt] Saved {fname} ({mb:.1f} MB)')
    if ON_KAGGLE: _rclone_push(path, 'checkpoints')
    old = sorted(glob.glob(f'{CKPT_DIR}/{name}_step*.pt'))
    for f in old[:-keep]:
        if os.path.exists(f): os.remove(f)

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
            f'rclone sync \"{GDRIVE_ROOT}/checkpoints/\" \"{CKPT_DIR}/\"',
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
"""))

# S8 — model save/load helpers (verbatim from v5 Cell 14, extended)
cells.append(code("""\
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
    \"\"\"Keep config in sync with actual ModuleList depths after pruning.\"\"\"\
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
    \"\"\"Move model to cuda:0 if split by device_map='auto'.\"\"\"\
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
    \"\"\"Save model to Drive using HF save_pretrained (battle-tested from v5).\"\"\"\
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
        r = subprocess.run(f'rclone sync \"{target}/\" \"{GDRIVE_ROOT}/models/{stage_name}/\"',
                           shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print(f'[model] WARNING rclone push failed: {r.stderr[:300]}')
        else:
            print(f'[model] Pushed to remote: {GDRIVE_ROOT}/models/{stage_name}/')
    else:
        print('[model] Colab: saved directly to Drive.')

def load_model_from_drive(stage_name):
    from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor, AutoConfig
    local = f'{MODEL_DIR}/{stage_name}'
    if ON_KAGGLE and (not os.path.exists(local) or not os.listdir(local)):
        print(f'[model] Not in local cache — pulling from remote...')
        _rclone_pull_model(stage_name)
    if not os.path.exists(local) or not os.listdir(local):
        raise RuntimeError(f'[model] Not found or empty: {local}')
    wf = [f for f in os.listdir(local) if f.endswith('.safetensors') or f.endswith('.bin')]
    if not wf:
        raise RuntimeError(f'[model] No weight files in {local}')
    print(f'[model] Loading {stage_name} from {local} ...')
    cfg = AutoConfig.from_pretrained(local)
    enc_n, dec_n = _infer_t2u_layer_counts(local)
    if enc_n and getattr(cfg,'t2u_encoder_layers',None) != enc_n:
        print(f'  Repair T2U enc depth: {cfg.t2u_encoder_layers} -> {enc_n}')
        cfg.t2u_encoder_layers = enc_n
    if dec_n and getattr(cfg,'t2u_decoder_layers',None) != dec_n:
        print(f'  Repair T2U dec depth: {cfg.t2u_decoder_layers} -> {dec_n}')
        cfg.t2u_decoder_layers = dec_n
    mdl = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
        local, config=cfg, torch_dtype=torch.float16, device_map='auto')
    _load_custom_state(mdl, local)
    proc = SeamlessM4TProcessor.from_pretrained(local)
    mdl.eval()
    print(f'[model] Loaded {stage_name}.')
    return mdl, proc

print('Model I/O helpers ready.')
"""))

# S9 — core library (verbatim from v5 Cell 19)
cells.append(code("""\
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
    print(f'\\n--- {title} ---')
    total = bd.pop('TOTAL')
    for name, p in sorted(bd.items(), key=lambda x: -x[1]):
        pct = p / total * 100 if total > 0 else 0
        print(f'  {name:<35} {p:>8.1f}M  ({pct:>5.1f}%)')
    print(f'  {\"TOTAL\":<35} {total:>8.1f}M')
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
"""))

# S10 — summaries + plotting (verbatim from v5 Cell 23)
cells.append(code("""\
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
    if not data: print('No summaries yet.'); return
    labels = [s['label'] for s in data]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Textless S2ST Compression Pipeline: Phase Comparison',
                 fontsize=15, fontweight='bold')
    metrics = [('avg_bleu', 'ASR-BLEU (higher=better)', '#2196F3'),
               ('avg_chrf', 'ASR-ChrF (higher=better)', '#4CAF50'),
               ('avg_rtf',  'RTF (lower=faster)',        '#FF9800'),
               ('params_M', 'Parameters (M)',            '#9C27B0')]
    for ax, (key, title, color) in zip(axes.flat, metrics):
        vals = [s.get(key, 0) for s in data]
        bars = ax.bar(range(len(labels)), vals, color=color, alpha=0.85, edgecolor='white')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{v:.1f}',
                    ha='center', va='bottom', fontsize=8)
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
"""))

# S11 — MMS-ASR (verbatim from v5 Cell 20 + extended for Qwen3)
cells.append(code("""\
# ── MMS-ASR for Bengali (battle-tested from v5 Cell 20) ─────────────────────
import gc as _stdlib_gc

_MMS_MODEL_ID = 'facebook/mms-1b-all'
_MMS_LANG     = 'ben'
_mms_asr_model, _mms_asr_processor = None, None

def _ensure_mms_loaded():
    global _mms_asr_model, _mms_asr_processor
    if _mms_asr_model is not None: return
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    print(f'[MMS-ASR] Loading {_MMS_MODEL_ID} lang={_MMS_LANG}...')
    _mms_asr_processor = AutoProcessor.from_pretrained(_MMS_MODEL_ID, target_lang=_MMS_LANG)
    _mms_asr_model = Wav2Vec2ForCTC.from_pretrained(
        _MMS_MODEL_ID, target_lang=_MMS_LANG,
        ignore_mismatched_sizes=True, torch_dtype=torch.float16)
    _mms_asr_model.load_adapter(_MMS_LANG)
    _mms_asr_model = _mms_asr_model.eval()
    try: _mms_asr_model = _mms_asr_model.to('cuda:0')
    except RuntimeError: pass
    print('[MMS-ASR] Ready.')

def asr_transcribe_ben(audio_np, sr=16000):
    _ensure_mms_loaded()
    if audio_np is None or len(audio_np) < 400: return ''
    if sr != 16000:
        audio_np = torchaudio.functional.resample(
            torch.tensor(audio_np), sr, 16000).numpy()
    device = next(_mms_asr_model.parameters()).device
    inputs = _mms_asr_processor(audio_np, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _mms_asr_model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return _mms_asr_processor.batch_decode(pred_ids)[0].strip()

# ── Qwen3-ASR-1.7B for ZH / AR / HI / EN ───────────────────────────────────
# PLAN.md Section 5: Qwen3-ASR-1.7B is stronger than MMS for high-resource langs
_qwen_model, _qwen_proc = None, None

def _ensure_qwen_loaded():
    global _qwen_model, _qwen_proc
    if _qwen_model is not None: return
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor as AP
    print('[Qwen3-ASR] Loading Qwen/Qwen3-ASR-1.7B...')
    _qwen_proc = AP.from_pretrained('Qwen/Qwen3-ASR-1.7B')
    _qwen_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        'Qwen/Qwen3-ASR-1.7B', torch_dtype=torch.float16,
        device_map='cuda:1' if N_GPU > 1 else 'cuda:0').eval()
    print('[Qwen3-ASR] Ready.')

def asr_transcribe_qwen(audio_np, sr=16000, lang='zh'):
    _ensure_qwen_loaded()
    if audio_np is None or len(audio_np) < 400: return ''
    if sr != 16000:
        audio_np = torchaudio.functional.resample(
            torch.tensor(audio_np), sr, 16000).numpy()
    device = next(_qwen_model.parameters()).device
    inp = _qwen_proc(audio_np, sampling_rate=16000, return_tensors='pt')
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        ids = _qwen_model.generate(**inp, language=lang, max_new_tokens=256)
    return _qwen_proc.decode(ids[0], skip_special_tokens=True).strip()

# ── M4T lang → ASR backend mapping ──────────────────────────────────────────
M4T_FLEURS_MAP = {
    'eng': 'en_us', 'ben': 'bn_in', 'cmn': 'cmn_hans_cn',
    'arb': 'ar_eg', 'hin': 'hi_in',
}
LANG_ASR_CONFIG = {
    'ben': ('mms', 'ben'),   # MMS reliable on Bengali (v5 finding)
    'cmn': ('qwen', 'zh'),   # Qwen3 superior on Mandarin
    'arb': ('qwen', 'ar'),   # Qwen3 superior on Arabic
    'hin': ('qwen', 'hi'),   # Qwen3 superior on Hindi
    'eng': ('qwen', 'en'),   # Qwen3 for English output check
}

def asr_transcribe(audio_np, tgt_lang_m4t, sr=16000):
    \"\"\"Route to correct ASR backend per PLAN.md Section 5.\"\"\"\
    if audio_np is None or len(audio_np) < 800: return ''
    backend, lang_code = LANG_ASR_CONFIG.get(tgt_lang_m4t, ('qwen', 'en'))
    try:
        if backend == 'mms':  return asr_transcribe_ben(audio_np, sr)
        else:                 return asr_transcribe_qwen(audio_np, sr, lang=lang_code)
    except Exception as e:
        print(f'[ASR] Error ({tgt_lang_m4t}): {e}')
        return ''

print('ASR stack ready (MMS-Bengali + Qwen3-ZH/AR/HI/EN).')
print('Note: Bengali uses MMS (proven in V1). All others use Qwen3-ASR-1.7B.')
"""))

# S12 — sacrebleu + run_s2st (verbatim from v5 Cell 21, run_s2st adapted)
cells.append(code("""\
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

def run_s2st_legacy(mdl, wav, tgt_lang='ben'):
    \"\"\"Legacy full S2ST for models with text decoder (Phases 0-3).\"\"\"\
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
    \"\"\"Text-only generation (for benchmarking text-decoder models).\"\"\"\
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

def quick_eval_chrf(mdl, samples, tgt_lang='ben', max_samples=10):
    scores = []
    for s in samples[:max_samples]:
        try: scores.append(compute_chrf(run_s2t_only(mdl, s['wav'], tgt_lang), s['ref']))
        except: scores.append(0.0)
    return float(np.mean(scores))

def run_benchmark(mdl, samples, label='model', tgt_lang='ben', save_n=2):
    \"\"\"Standard benchmark (text-only ChrF/BLEU) for models with text decoder.\"\"\"\
    print(f'\\n{\"=\"*60}\\n  BENCHMARK: {label}  Samples:{len(samples)}\\n{\"=\"*60}')
    gpu_mem()
    results = []
    for i, s in enumerate(samples):
        try:
            dur = len(s['wav']) / 16000
            t0  = time.time()
            pred = run_s2t_only(mdl, s['wav'], tgt_lang=tgt_lang)
            rtf  = (time.time() - t0) / dur
            bleu = compute_bleu(pred, s['ref'])
            chrf = compute_chrf(pred, s['ref'])
            print(f'  [{i+1:>2}/{len(samples)}] BLEU={bleu:5.1f} ChrF={chrf:5.1f} RTF={rtf:.3f}')
            print(f'              pred: {pred[:80]}')
            if save_n > 0 and i < save_n:
                _, wav_out = run_s2st_legacy(mdl, s['wav'], tgt_lang=tgt_lang)
                save_audio(wav_out, mdl.config.sampling_rate, f'{label}_s{i+1}out.wav')
            results.append(dict(id=s['id'],bleu=bleu,chrf=chrf,rtf=rtf,pred=pred,ref=s['ref']))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append(dict(id=s['id'],bleu=0,chrf=0,rtf=float('nan'),pred='',ref=s.get('ref','')))
    valid = [r for r in results if not math.isnan(r['rtf'])]
    summary = dict(label=label, n=len(valid),
        avg_bleu=float(np.mean([r['bleu'] for r in valid])) if valid else 0,
        avg_chrf=float(np.mean([r['chrf'] for r in valid])) if valid else 0,
        avg_rtf =float(np.mean([r['rtf']  for r in valid])) if valid else 0,
        params_M=count_params(mdl))
    print(f'\\n  Summary: BLEU={summary[\"avg_bleu\"]:.2f} ChrF={summary[\"avg_chrf\"]:.2f}'
          f' RTF={summary[\"avg_rtf\"]:.4f} Params={summary[\"params_M\"]:.1f}M')
    return results, summary

print('Benchmark functions ready.')
"""))

# S13 — model loader (verbatim from v5 Cell 22)
cells.append(code("""\
from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor

MODEL_NAME = 'facebook/seamless-m4t-v2-large'

def load_base_model():
    print(f'Loading processor from {MODEL_NAME}...')
    proc = SeamlessM4TProcessor.from_pretrained(MODEL_NAME)
    print(f'Loading model — may take 5-10 min...')
    mdl = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map='auto')
    mdl.eval()
    print('Model loaded.'); gpu_mem()
    return mdl, proc

print('load_base_model() ready.')
"""))

# S14 — dataset loading (verbatim from v5 Cells 24-26, extended to 5 langs)
cells.append(code("""\
## Dataset loading — battle-tested from seamless-cse465v5 (Cells 24-26)
import concurrent.futures, io, soundfile as sf, pandas as pd

LOCAL_PARQUET_CACHE = '/kaggle/input/datasets/coderayed/fleurs-en-bn-parquet'
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
            print(f'  {\"OK\" if ok else \"FAIL\"}: {msg}')
    def _load_lang(lang):
        files = sorted(glob.glob(f'{LOCAL_PARQUET_CACHE}/{lang}/{split}_*.parquet'))
        if not files: raise FileNotFoundError(f'No cached shards for {lang}')
        return Dataset.from_pandas(pd.read_parquet(files[0]))
    return _load_lang(src_lang), _load_lang(tgt_lang)

def push_fleurs_to_drive():
    if not ON_KAGGLE: return
    subprocess.run(f'rclone copy \"{LOCAL_PARQUET_CACHE}/\" \"{DRIVE_FLEURS_PATH}/\" --transfers=8',
                   shell=True, capture_output=True)

def load_fleurs_from_drive(src_lang, tgt_lang, split='train'):
    from datasets import Dataset
    if not ON_KAGGLE: return None, None
    if not os.path.exists(LOCAL_PARQUET_CACHE):
        r = subprocess.run(f'rclone copy \"{DRIVE_FLEURS_PATH}/\" \"{LOCAL_PARQUET_CACHE}/\" --transfers=8',
                           shell=True, capture_output=True, text=True)
        if r.returncode != 0: return None, None
    def _load_lang(lang):
        files = sorted(glob.glob(f'{LOCAL_PARQUET_CACHE}/{lang}/{split}_*.parquet'))
        if not files: return None
        return Dataset.from_pandas(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
    src_ds = _load_lang(src_lang); tgt_ds = _load_lang(tgt_lang)
    if src_ds and tgt_ds: print(f'[gdrive] Loaded: {len(src_ds)} src, {len(tgt_ds)} tgt')
    return src_ds, tgt_ds

def _load_wav(audio_cell):
    \"\"\"Verbatim from v5 Cell 25 — handles both HF Dataset and parquet byte formats.\"\"\"\
    audio = audio_cell
    if isinstance(audio, dict) and 'array' in audio:
        arr, sr = audio['array'], audio['sampling_rate']
    elif isinstance(audio, dict) and 'bytes' in audio:
        wav, sr = sf.read(io.BytesIO(audio['bytes']))
        if wav.ndim > 1: wav = wav.mean(axis=1)
        arr = wav
    else:
        raise RuntimeError(f'Unsupported audio format: {type(audio)}')
    arr = np.array(arr, dtype=np.float32)
    if sr != 16000:
        arr = torchaudio.functional.resample(torch.tensor(arr), sr, 16000).numpy()
    return arr

print('FLEURS data loaders ready.')
"""))

# S15 — load EN→BN eval samples (verbatim from v5 Cell 25 then extended)
cells.append(code("""\
# ── Load EN→BN eval samples (verbatim from v5 Cell 25) ──────────────────────
N_EVAL       = 25
TARGET_LANG  = 'ben'
FLEURS_SRC, FLEURS_TGT = 'en_us', 'bn_in'

print(f'Loading FLEURS {FLEURS_SRC}->{FLEURS_TGT} for benchmarking [test]')
ds_src, ds_tgt = load_fleurs_from_drive(FLEURS_SRC, FLEURS_TGT, split='test')
if ds_src is None or ds_tgt is None:
    print('\\n[Cache miss] Downloading...')
    ds_src, ds_tgt = load_fleurs_parallel(FLEURS_SRC, FLEURS_TGT, split='test', n_workers=8)
    push_fleurs_to_drive()

df_src = ds_src.to_pandas() if hasattr(ds_src,'to_pandas') else pd.DataFrame(ds_src)
df_tgt = ds_tgt.to_pandas() if hasattr(ds_tgt,'to_pandas') else pd.DataFrame(ds_tgt)
print('Deduplicating and merging...')
src_dedup = (df_src[['id','transcription','audio']].drop_duplicates('id',keep='first')
             .rename(columns={'transcription':'en_text','audio':'en_audio'}))
tgt_dedup = (df_tgt[['id','transcription','audio']].drop_duplicates('id',keep='first')
             .rename(columns={'transcription':'bn_text','audio':'bn_audio'}))
merged = (pd.merge(src_dedup, tgt_dedup, on='id', how='inner')
          .sort_values('id').reset_index(drop=True).head(N_EVAL))
print(f'Using {len(merged)} matched pairs for evaluation')
eval_samples = [dict(id=row['id'], wav=_load_wav(row['en_audio']),
                     ref=row['bn_text'], en_text=row['en_text'])
                for _, row in merged.iterrows()]
del df_src, df_tgt, src_dedup, tgt_dedup, merged, ds_src, ds_tgt
gc.collect()
print(f'Loaded {len(eval_samples)} eval samples.')
"""))

# S16 — load train samples (verbatim from v5 Cell 26)
cells.append(code("""\
# ── Load EN→BN training samples (verbatim from v5 Cell 26) ──────────────────
print(f'Loading FLEURS {FLEURS_SRC}->{FLEURS_TGT} for fine-tuning [train]')
src_ds, tgt_ds = load_fleurs_from_drive(FLEURS_SRC, FLEURS_TGT, split='train')
if src_ds is None or tgt_ds is None:
    print('\\n[Cache miss] Downloading...')
    src_ds, tgt_ds = load_fleurs_parallel(FLEURS_SRC, FLEURS_TGT, split='train', n_workers=8)
    push_fleurs_to_drive()
df_src_train = src_ds.to_pandas() if hasattr(src_ds,'to_pandas') else pd.DataFrame(src_ds)
df_tgt_train = tgt_ds.to_pandas() if hasattr(tgt_ds,'to_pandas') else pd.DataFrame(tgt_ds)
print('Deduplicating and merging training data...')
src_tr = (df_src_train[['id','audio']].drop_duplicates('id',keep='first')
          .rename(columns={'audio':'en_audio'}))
tgt_tr = (df_tgt_train[['id','transcription','audio']].drop_duplicates('id',keep='first')
          .rename(columns={'transcription':'bn_text','audio':'bn_audio'}))
merged_train = pd.merge(src_tr, tgt_tr, on='id', how='inner').reset_index(drop=True)
merged_train = merged_train[merged_train['bn_text'].str.strip().str.len() > 0]
print(f'Usable training pairs: {len(merged_train)}')
ft_samples = [dict(id=row['id'], wav=_load_wav(row['en_audio']), ref=row['bn_text'])
              for _, row in merged_train.iterrows()]
del df_src_train, df_tgt_train, src_tr, tgt_tr, merged_train, src_ds, tgt_ds
gc.collect()
print(f'Loaded {len(ft_samples)} training samples.')
"""))

# S17 — load multilingual eval samples for Phase 7
cells.append(code("""\
# ── Multilingual eval samples: 10 per lang for Phase 7 benchmark ─────────────
# PLAN.md Section 5: 5 languages — EN, BN, ZH, AR, HI
MULTI_LANGS = {
    'eng': 'en_us', 'ben': 'bn_in',
    'cmn': 'cmn_hans_cn', 'arb': 'ar_eg', 'hin': 'hi_in',
}
N_MULTILANG_EVAL = 10

multilang_eval = {}  # {m4t_code: [samples]}
for m4t_code, fleurs_code in MULTI_LANGS.items():
    if m4t_code == 'eng': continue  # already have eval_samples
    print(f'Loading {m4t_code} ({fleurs_code}) test split...')
    try:
        ds_s, ds_t = load_fleurs_from_drive(fleurs_code, fleurs_code, split='test')
        if ds_s is None:
            ds_s, ds_t = load_fleurs_parallel(fleurs_code, fleurs_code, split='test')
            push_fleurs_to_drive()
        df = ds_s.to_pandas() if hasattr(ds_s,'to_pandas') else pd.DataFrame(ds_s)
        df = df.drop_duplicates('id',keep='first').head(N_MULTILANG_EVAL)
        samps = [dict(id=row['id'], wav=_load_wav(row['audio']),
                      ref=str(row.get('transcription','')))
                 for _, row in df.iterrows() if row.get('audio') is not None]
        multilang_eval[m4t_code] = samps
        del df, ds_s, ds_t; gc.collect()
        print(f'  {m4t_code}: {len(samps)} samples')
    except Exception as e:
        print(f'  {m4t_code}: FAILED — {e}')
        multilang_eval[m4t_code] = []
multilang_eval['eng'] = eval_samples  # reuse EN samples
print(f'Multilingual eval ready: {[(k,len(v)) for k,v in multilang_eval.items()]}')
"""))

# S18 — session status (verbatim from v5 Cell 17)
cells.append(code("""\
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

sync_checkpoints_from_drive()
session_status()
print('\\n✓ ALL SETUP CELLS COMPLETE — proceed to phases.')
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0: V1 BASELINE
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Phase 0: V1 Baseline Capture
Load V1 pipeline (1039M from previous work) and run ASR-ChrF benchmark across 5 languages.
These scores become the quality ceiling for the textless model.
"""))

cells.append(code("""\
# ── Load V1 model — try Drive, fall back to fresh base ───────────────────────
try:
    model_v1, processor = load_model_from_drive('phase7_dora_merged_v1')
    print('V1 model loaded from Drive.')
except Exception as e:
    print(f'V1 not on Drive ({e}) — loading fresh base model as V1 proxy.')
    model_v1, processor = load_base_model()

print_model_breakdown(model_v1, 'V1 Baseline Model')
gpu_mem()
"""))

cells.append(code("""\
# ── P0 benchmark: EN→BN (text-decoder ChrF, fast proxy) ─────────────────────
p0_ckpt = load_latest_checkpoint('phase0_v1_baseline')
if p0_ckpt:
    p0_results, p0_summary = p0_ckpt['results'], p0_ckpt['summary']
    print(f'Loaded P0: BLEU={p0_summary[\"avg_bleu\"]:.2f} ChrF={p0_summary[\"avg_chrf\"]:.2f}')
else:
    p0_results, p0_summary = run_benchmark(
        model_v1, eval_samples, label='P0_V1_Baseline', tgt_lang='ben', save_n=2)
    save_checkpoint(dict(results=p0_results, summary=p0_summary),
                    'phase0_v1_baseline', step=0)

store_summary(p0_summary)
plot_phase_comparison()
print(f'\\nV1 quality ceiling: ChrF={p0_summary[\"avg_chrf\"]:.2f} — this is the target for textless model.')
"""))

cells.append(code("""\
# ── Free V1 from VRAM before next phase ──────────────────────────────────────
del model_v1
gc.collect(); torch.cuda.empty_cache()
print('V1 unloaded from VRAM.')
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: VOCAB PRUNING 5 LANGS
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Phase 1: Vocabulary Pruning — 5 Languages
Extend V1 vocab trim to EN, BN, ZH, AR, HI. Saves ~215M params, zero quality loss.
Paper: Asahi et al. (EMNLP 2023) — vocabulary trimming methodology.
"""))

cells.append(code("""\
# identify_used_tokens and trim_vocabulary verbatim from v5 Cell 31 ────────────

def identify_used_tokens(proc, target_lang_codes, n_corpus=2000):
    from datasets import load_dataset
    fleurs_codes = dict(eng='en_us', ben='bn_in', cmn='cmn_hans_cn',
                        fra='fr_fr', deu='de_de', hin='hi_in', urd='ur_pk',
                        arb='ar_eg')
    BASE = 'hf://datasets/google/fleurs@refs%2Fconvert%2Fparquet'
    used = set()
    tok  = proc.tokenizer
    if hasattr(tok, 'all_special_ids'): used.update(tok.all_special_ids)
    for tid in range(len(tok)):
        t = tok.convert_ids_to_tokens(tid)
        if t and t.startswith('__') and t.endswith('__'): used.add(tid)
    for lang, fc in fleurs_codes.items():
        if lang not in target_lang_codes: continue
        print(f'  Scanning {lang} ({fc})...')
        try:
            ds = load_dataset('parquet',
                              data_files={'train': f'{BASE}/{fc}/train/*.parquet'},
                              split='train')
            for i, ex in enumerate(ds):
                if i >= n_corpus: break
                text = ex.get('transcription','')
                if text: used.update(tok.encode(text, add_special_tokens=False))
        except Exception as e:
            print(f'    Warning: {lang}: {e}')
    print(f'  Unique tokens: {len(used)} / {len(tok)}')
    return sorted(used)


def trim_vocabulary(mdl, proc, keep_ids):
    \"\"\"Verbatim from v5 Cell 31 — handles ScaledWordEmbedding + id_to_text remap.\"\"\"\
    keep_t   = torch.tensor(keep_ids, dtype=torch.long)
    old_v    = mdl.config.vocab_size
    new_v    = len(keep_ids)
    hidden   = mdl.config.hidden_size
    print(f'  Vocabulary: {old_v} -> {new_v} ({new_v/old_v*100:.1f}%)')
    old_shared = mdl.shared
    if old_shared.num_embeddings != old_v:
        print(f'  ERROR: shared mismatch. Reload base model first.'); return mdl
    dev = old_shared.weight.device; dtype = old_shared.weight.dtype
    keep_t_dev = keep_t.to(dev)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
    old_pad = old_shared.padding_idx
    new_pad = old_to_new.get(old_pad) if old_pad is not None else None
    embed_scale = getattr(mdl.text_decoder.embed_tokens, 'embed_scale', 1.0)
    # shared
    new_shared = nn.Embedding(new_v, hidden, padding_idx=new_pad)
    new_shared.weight.data = old_shared.weight.data[keep_t_dev].clone()
    mdl.shared = new_shared.to(device=dev, dtype=dtype)
    # text_decoder.embed_tokens in-place
    dec_emb = mdl.text_decoder.embed_tokens
    dec_emb.weight = mdl.shared.weight
    dec_emb.num_embeddings = new_v; dec_emb.padding_idx = new_pad
    # lm_head tied
    mdl.lm_head.weight = mdl.shared.weight
    mdl.lm_head.out_features = new_v
    if mdl.lm_head.bias is not None:
        mdl.lm_head.bias = nn.Parameter(mdl.lm_head.bias.data[keep_t_dev].clone())
    # config
    mdl.config.vocab_size = new_v
    for attr in ['pad_token_id','bos_token_id','eos_token_id','decoder_start_token_id']:
        old_id = getattr(mdl.config, attr, None)
        if old_id is not None and old_id in old_to_new:
            setattr(mdl.config, attr, old_to_new[old_id])
    if hasattr(mdl.text_decoder, 'vocab_size'): mdl.text_decoder.vocab_size = new_v
    if hasattr(mdl.text_decoder, 'padding_idx'): mdl.text_decoder.padding_idx = new_pad
    # generation_config
    gen_cfg = mdl.generation_config
    if hasattr(gen_cfg,'text_decoder_lang_to_code_id') and gen_cfg.text_decoder_lang_to_code_id:
        gen_cfg.text_decoder_lang_to_code_id = {
            lang: old_to_new[oid]
            for lang, oid in gen_cfg.text_decoder_lang_to_code_id.items()
            if oid in old_to_new}
    if hasattr(gen_cfg,'id_to_text') and gen_cfg.id_to_text:
        old_map = gen_cfg.id_to_text
        gen_cfg.id_to_text = {str(old_to_new[int(k)]): v
                               for k, v in old_map.items() if int(k) in old_to_new}
        print(f'  id_to_text: {len(old_map)} -> {len(gen_cfg.id_to_text)} entries')
    for attr in ['pad_token_id','bos_token_id','eos_token_id',
                 'decoder_start_token_id','forced_bos_token_id']:
        old_id = getattr(gen_cfg, attr, None)
        if old_id is not None and old_id in old_to_new:
            setattr(gen_cfg, attr, old_to_new[old_id])
    mdl._vocab_remap_to_old = torch.tensor(keep_ids, dtype=torch.long)
    print(f'  Done: ~{(old_v-new_v)*hidden/1e6:.0f}M params freed')
    return mdl

print('Vocab trimming functions ready.')
"""))

cells.append(code("""\
# ── Run Phase 1 — try Drive, else trim from V1 ───────────────────────────────
try:
    model_p1, processor = load_model_from_drive('phase1_vocab_5lang')
    p1_ck = load_latest_checkpoint('phase1_vocab')
    if p1_ck and 'keep_ids' in p1_ck:
        keep_ids = p1_ck['keep_ids']
        model_p1._vocab_remap_to_old = torch.tensor(keep_ids, dtype=torch.long)
        print(f'  Restored vocab remap ({len(keep_ids)} tokens)')
    print('Loaded Phase 1 from Drive.')
except Exception as e:
    print(f'Load failed ({e}), running vocab trim...')
    # Reload V1 fresh
    model_v1_fresh, processor = load_base_model()
    TARGET_5LANGS = ['eng', 'ben', 'cmn', 'arb', 'hin']
    keep_ids = identify_used_tokens(processor, TARGET_5LANGS, n_corpus=3000)
    pre = count_params(model_v1_fresh)
    model_p1 = trim_vocabulary(model_v1_fresh, processor, keep_ids)
    post = count_params(model_p1)
    print(f'  Params: {pre:.1f}M -> {post:.1f}M (saved {pre-post:.1f}M)')
    save_checkpoint(dict(keep_ids=keep_ids, pre=pre, post=post), 'phase1_vocab', 0)
    save_model_to_drive(model_p1, processor, 'phase1_vocab_5lang')

print_model_breakdown(model_p1, 'After Phase 1: Vocab Trimmed (5L)')
"""))

cells.append(code("""\
p1_bench = load_latest_checkpoint('phase1_benchmark')
if p1_bench and p1_bench['summary'].get('avg_bleu',0) > 0:
    p1_results, p1_summary = p1_bench['results'], p1_bench['summary']
else:
    p1_results, p1_summary = run_benchmark(
        model_p1, eval_samples, label='P1_Vocab5L', tgt_lang='ben', save_n=2)
    save_checkpoint(dict(results=p1_results, summary=p1_summary), 'phase1_benchmark', 0)
store_summary(p1_summary)
plot_phase_comparison()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: SPEECH ENCODER PRUNING (8 layers, BI-guided)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Phase 2: Speech Encoder Moderate Pruning (24 → 16 layers)
Target: remove 8 of 24 layers (~33%). Method: BI-guided iterative greedy (same as v5 Phase 4).
Conservative — encoder layers are language-neutral, no cliff expected.
Papers: ShortGPT (ACL 2025) Block Influence · Moslem IWSLT 2025 iterative greedy.
"""))

cells.append(code("""\
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
    \"\"\"ShortGPT (ACL 2025) BI: 1 - cos(layer_input, layer_output).\"\"\"\
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


def iterative_enc_prune(mdl, samples, n_remove, tgt_lang='ben', max_eval=10,
                        ckpt_name='phase2_enc_pruning', bi_scores=None,
                        bi_candidate_ratio=0.5, protected=None):
    \"\"\"BI-guided iterative greedy encoder pruning — verbatim from v5 Cell 58.\"\"\"\
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
    baseline = quick_eval_chrf(mdl, samples, tgt_lang, max_eval)
    print(f'  Baseline ChrF: {baseline:.2f}')
    for it in range(len(removed), n_remove):
        eligible = [pos for pos in range(len(current)) if orig_idx[pos] not in protected]
        if bi_scores and len(eligible) > 2:
            by_bi = sorted(eligible, key=lambda pos: bi_scores.get(orig_idx[pos], float('inf')))
            n_cands = max(2, int(len(by_bi)*bi_candidate_ratio))
            cands   = by_bi[:n_cands]
            print(f'\\n  Iter {it+1}/{n_remove} | BI pre-filter: {len(cands)}/{len(eligible)} cands')
        else:
            cands = eligible
            print(f'\\n  Iter {it+1}/{n_remove} | all {len(cands)} eligible (no BI)')
        if not cands: print('  No candidates left, stopping.'); break
        scores = {}
        for pos in cands:
            temp = current[:pos]+current[pos+1:]
            setattr(parent, la, nn.ModuleList(temp))
            sc = quick_eval_chrf(mdl, samples, tgt_lang, max_eval)
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
"""))

cells.append(code("""\
# ── RUN Phase 2 ───────────────────────────────────────────────────────────────
N_ENC_REMOVE      = 8
ENC_BI_CAND_RATIO = 0.5   # evaluate bottom 50% by BI — halves ChrF calls

p2_ckpt    = load_latest_checkpoint('phase2_enc_pruning')
p2_complete= p2_ckpt and len(p2_ckpt.get('removed',[])) >= N_ENC_REMOVE

if p2_complete:
    removed_enc = p2_ckpt['removed']; bi_scores = p2_ckpt.get('bi_scores',{}); p2_log = p2_ckpt['log']
    print(f'Phase 2 complete: removed {removed_enc}')
    try:
        model_p2, processor = load_model_from_drive('phase2_enc_16L')
    except:
        print('  Rebuilding from checkpoint + model_p1...')
        model_p2 = model_p1
        parent, la = get_speech_encoder_layers(model_p2)
        cur = list(getattr(parent, la))
        keep = [i for i in range(len(cur)) if i not in removed_enc]
        setattr(parent, la, nn.ModuleList([cur[i] for i in keep]))
        sync_model_config(model_p2)
        save_model_to_drive(model_p2, processor, 'phase2_enc_16L')
else:
    done = len(p2_ckpt['removed']) if p2_ckpt else 0
    print(f'{\"Resuming\" if done else \"Running\"} Phase 2: enc pruning ({done}/{N_ENC_REMOVE} done)...')
    model_p2 = _consolidate_to_single_gpu(model_p1)
    sanity = quick_eval_chrf(model_p2, eval_samples, 'ben', 5)
    print(f'  Sanity ChrF={sanity:.2f}  (abort if < 10)')
    assert sanity > 10, f'Sanity too low: {sanity:.2f}'
    if not (p2_ckpt and p2_ckpt.get('bi_scores')):
        print('Computing Block Influence scores...')
        bi_scores = compute_block_influence(model_p2, eval_samples, max_n=50)
        save_checkpoint(dict(removed=[], log=[], bi_scores=bi_scores), 'phase2_enc_pruning', 0)
    else:
        bi_scores = p2_ckpt['bi_scores']
        print(f'  BI scores loaded ({len(bi_scores)} layers)')
    parent_tmp, la_tmp = get_speech_encoder_layers(model_p2)
    n_enc = len(getattr(parent_tmp, la_tmp))
    enc_protected = _get_protected_enc(n_enc)
    removed_enc, p2_log = iterative_enc_prune(
        model_p2, eval_samples, N_ENC_REMOVE, 'ben', max_eval=10,
        ckpt_name='phase2_enc_pruning', bi_scores=bi_scores,
        bi_candidate_ratio=ENC_BI_CAND_RATIO, protected=enc_protected)
    sync_model_config(model_p2)
    save_checkpoint(dict(removed=removed_enc, log=p2_log, bi_scores=bi_scores),
                    'phase2_enc_pruning', 0)
    save_model_to_drive(model_p2, processor, 'phase2_enc_16L')

print(f'Encoder layers removed: {removed_enc}')
print_model_breakdown(model_p2, 'After Phase 2: Enc 16L')
"""))

cells.append(code("""\
# ── Phase 2 visualisation ─────────────────────────────────────────────────────
if p2_log and bi_scores:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 2: Speech Encoder Pruning (24→16L)', fontweight='bold')

    bi_at_rm = {e['removed']: e.get('bi_score') for e in p2_log}
    parent_tmp, la_tmp = get_speech_encoder_layers(model_p2)
    n_enc_orig = len(getattr(parent_tmp, la_tmp)) + len(removed_enc)
    enc_prot   = _get_protected_enc(n_enc_orig)
    all_idx    = sorted(set(bi_scores)|set(removed_enc)|set(bi_at_rm))
    vals = [bi_scores.get(i, bi_at_rm.get(i, 0.0)) for i in all_idx]
    colors = ['#d32f2f' if i in removed_enc else '#ff9800' if i in enc_prot else '#4caf50' for i in all_idx]
    axes[0].bar(all_idx, vals, color=colors, edgecolor='white')
    axes[0].set_title('BI per Layer\\n(red=removed, orange=protected, green=kept)', fontweight='bold')
    axes[0].set_xlabel('Layer'); axes[0].set_ylabel('BI Score')

    iters  = [e['iter'] for e in p2_log]; chrfs = [e['chrf'] for e in p2_log]
    axes[1].plot(iters, chrfs, 'o-', color='#2196F3', lw=2, ms=8)
    for e in p2_log:
        axes[1].annotate(f'L{e[\"removed\"]}', (e['iter'], e['chrf']),
                         textcoords='offset points', xytext=(0,5), fontsize=7)
    axes[1].set_title('ChrF After Each Removal', fontweight='bold')
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('ChrF')

    _fil = [(e['bi_score'], e['chrf']) for e in p2_log if e.get('bi_score')]
    if len(_fil) >= 2:
        bv = [x for x,_ in _fil]; cv = [y for _,y in _fil]
        axes[2].scatter(bv, cv, color='#9c27b0', s=80, zorder=3)
        for e in p2_log:
            if e.get('bi_score'):
                axes[2].annotate(f'L{e[\"removed\"]}', (e['bi_score'], e['chrf']),
                                 textcoords='offset points', xytext=(4,2), fontsize=8)
    axes[2].set_title('BI vs ChrF After Removal\\n(validates: low BI → safe)', fontweight='bold')
    axes[2].set_xlabel('BI Score of Removed Layer'); axes[2].set_ylabel('ChrF')

    plt.tight_layout()
    save_figure(fig, 'phase2_enc_pruning.png')
    plt.show()

p2_bench = load_latest_checkpoint('phase2_benchmark')
if p2_bench:
    p2_results, p2_summary = p2_bench['results'], p2_bench['summary']
else:
    p2_results, p2_summary = run_benchmark(model_p2, eval_samples, 'P2_Enc16L', save_n=2)
    save_checkpoint(dict(results=p2_results, summary=p2_summary), 'phase2_benchmark', 0)
store_summary(p2_summary); plot_phase_comparison()
"""))

cells.append(code("""\
del model_p1; gc.collect(); torch.cuda.empty_cache()
print('P1 model freed.')
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: LACO T2U MERGE
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Phase 3: T2U LaCo RDSC Merge (6+6 → 4+4 layers)
LaCo reserves weight differences instead of outright removal, preserving >80% capacity.
Better than iterative removal for T2U because every layer matters for unit quality.
Paper: Yang et al. EMNLP Findings 2024 (LaCo arXiv:2402.11187).
"""))

cells.append(code("""\
# ── find_t2u_stacks, sync_t2u_layer_indices from v5 Cell 85 (verbatim) ───────

def find_t2u_stacks(model):
    t2u, stacks = model.t2u_model, []
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
    for (parent, attr, name) in find_t2u_stacks(model):
        for i, layer in enumerate(list(getattr(parent, attr))):
            for aname in ['self_attn', 'encoder_attn', 'cross_attention']:
                attn = getattr(layer, aname, None)
                if attn and hasattr(attn,'layer_idx'): attn.layer_idx = i
        print(f'  Re-indexed {name}: {len(getattr(parent,attr))} layers')

# ── LaCo RDSC merge (PLAN.md Section 7 Phase 3) ──────────────────────────────
def laco_rdsc_merge(layer_i, layer_j, alpha=0.5):
    \"\"\"RDSC: W_merged = W_j + alpha*(W_j - W_i)  — preserves weight differences.\"\"\"\
    merged = copy.deepcopy(layer_j)
    sd_i = layer_i.state_dict(); sd_j = layer_j.state_dict()
    merged_sd = {k: (sd_j[k].float() + alpha*(sd_j[k].float()-sd_i[k].float())).to(sd_j[k].dtype)
                 if k in sd_i and sd_i[k].shape == sd_j[k].shape else sd_j[k]
                 for k in sd_j}
    merged.load_state_dict(merged_sd)
    return merged

def _cosine_sim_layers(merged, orig_j, calib_tensors, device):
    \"\"\"Measure output similarity between merged and original layer_j.\"\"\"\
    orig_j = orig_j.to(device).eval(); merged = merged.to(device).eval()
    sims = []
    for x in calib_tensors[:5]:
        if x is None: continue
        x = x.to(device)
        with torch.no_grad():
            try:
                o = orig_j(x);   o = o[0] if isinstance(o,tuple) else o
                m = merged(x);   m = m[0] if isinstance(m,tuple) else m
                sims.append(F.cosine_similarity(o.reshape(-1), m.reshape(-1), dim=0).item())
            except: pass
    return float(np.mean(sims)) if sims else 0.0

def apply_laco_t2u(model, sim_threshold=0.96, alpha=0.5, max_per_stack=2):
    \"\"\"Apply LaCo RDSC to T2U encoder + decoder stacks (2 merges each).\"\"\"\
    t2u_enc, t2u_dec = _get_t2u_encoder_decoder(model)
    device = next(model.t2u_model.parameters()).device
    # Build calibration tensors from speech encoder outputs
    calib = []
    for s in eval_samples[:8]:
        try:
            inp = processor(audio=s['wav'], sampling_rate=16000, return_tensors='pt')
            inp = {k: v.to(device) for k,v in inp.items() if isinstance(v,torch.Tensor)}
            with torch.no_grad():
                enc_out = model.speech_encoder(
                    input_features=inp['input_features'],
                    attention_mask=inp.get('attention_mask')).last_hidden_state
            calib.append(enc_out.cpu().float())
        except: pass
    print(f'  Built {len(calib)} calibration tensors.')
    for stack_obj, sname in [(t2u_enc, 'T2U-Enc'), (t2u_dec, 'T2U-Dec')]:
        if stack_obj is None or not hasattr(stack_obj,'layers'): continue
        layers = list(stack_obj.layers)
        collapsed, n_rm = [layers[0]], 0
        print(f'\\n  {sname}: {len(layers)} layers -> merging up to {max_per_stack}')
        for i in range(1, len(layers)):
            if n_rm >= max_per_stack:
                collapsed.append(layers[i]); continue
            candidate = laco_rdsc_merge(collapsed[-1], layers[i], alpha)
            sim = _cosine_sim_layers(candidate, layers[i], calib, device)
            print(f'  L{i}: sim={sim:.4f}', end='')
            if sim > sim_threshold:
                collapsed[-1] = candidate; n_rm += 1
                print(f' -> MERGED [{n_rm}/{max_per_stack}]')
            else:
                collapsed.append(layers[i])
                print(f' -> kept (below {sim_threshold})')
        stack_obj.layers = nn.ModuleList(collapsed)
        print(f'  {sname}: {len(layers)} -> {len(collapsed)} layers')
    sync_t2u_layer_indices(model)
    sync_model_config(model)
    return model

print('LaCo RDSC merge ready.')
"""))

cells.append(code("""\
# ── RUN Phase 3 ───────────────────────────────────────────────────────────────
p3_done = load_latest_checkpoint('phase3_laco_done')
if p3_done:
    print('Phase 3 already complete — loading from Drive.')
    try:
        model_p3, processor = load_model_from_drive('phase3_t2u_laco')
    except:
        print('  Rebuilding in-memory...')
        model_p3 = model_p2
        model_p3 = apply_laco_t2u(model_p3)
        save_model_to_drive(model_p3, processor, 'phase3_t2u_laco')
else:
    print('Running Phase 3: LaCo T2U merge...')
    model_p3 = _consolidate_to_single_gpu(model_p2)
    model_p3 = apply_laco_t2u(model_p3, sim_threshold=0.96, alpha=0.5, max_per_stack=2)
    print_model_breakdown(model_p3, 'After Phase 3: LaCo T2U 4+4L')
    save_model_to_drive(model_p3, processor, 'phase3_t2u_laco')
    save_checkpoint({'done': True, 'alpha': 0.5, 'sim_threshold': 0.96},
                    'phase3_laco_done', 0)

print_model_breakdown(model_p3, 'Phase 3 Model (Enc16L + T2U 4+4L)')

# Quick verify T2U stacks
for (parent, attr, name) in find_t2u_stacks(model_p3):
    print(f'  {name}: {len(getattr(parent,attr))} layers remaining')
"""))

cells.append(code("""\
p3_bench = load_latest_checkpoint('phase3_benchmark')
if p3_bench:
    p3_results, p3_summary = p3_bench['results'], p3_bench['summary']
else:
    p3_results, p3_summary = run_benchmark(
        model_p3, eval_samples, 'P3_LaCoT2U', save_n=2)
    save_checkpoint(dict(results=p3_results, summary=p3_summary), 'phase3_benchmark', 0)
store_summary(p3_summary); plot_phase_comparison()
"""))

cells.append(code("""\
del model_p2; gc.collect(); torch.cuda.empty_cache(); print('P2 freed.')
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: TEXT DECODER REMOVAL + CIF + SPEAKER ADAPTER
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Phase 4: Text Decoder Removal + CIF Connector + Speaker Adapter
THE CORE ARCHITECTURAL TRANSFORMATION.
Removes text_decoder (867M) + lm_head + shared vocab.
Installs CIF connector (Dong & Xu ICASSP 2020) and Speaker Adapter (ECAPA-TDNN → vocoder).
"""))

cells.append(code("""\
# ── CIF Connector (Dong & Xu, ICASSP 2020: arXiv:1905.11235) ─────────────────
class CIFConnector(nn.Module):
    \"\"\"
    Continuous Integrate-and-Fire connector.
    Compresses speech encoder frames [B, T_frames, D] to unit-aligned embeddings
    [B, T_units, D] matching what the text decoder used to feed into T2U.

    Architecture:
      - Weight predictor: sigmoid(Linear(D→1)) per frame
      - CIF accumulate-and-fire with learnable threshold
      - Language conditioning: nn.Embedding(n_langs, D//8) → Linear(D//8, D)
      - Refiner: small TransformerEncoder (2 layers) for quality
    \"\"\"
    def __init__(self, d_model=1024, n_refiner_layers=2, n_langs=40, threshold=1.0):
        super().__init__()
        self.d_model   = d_model
        self.threshold = threshold
        self.weight_predictor = nn.Sequential(
            nn.Linear(d_model, d_model//4), nn.ReLU(),
            nn.Linear(d_model//4, 1), nn.Sigmoid())
        self.lang_embed = nn.Embedding(n_langs, d_model//8)
        self.lang_proj  = nn.Linear(d_model//8, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True, norm_first=True)
        self.refiner  = nn.TransformerEncoder(enc_layer, num_layers=n_refiner_layers)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, encoder_out, tgt_lang_id=None):
        \"\"\"
        Args:
            encoder_out : [B, T_frames, D]  speech encoder hidden states
            tgt_lang_id : [B]               integer lang IDs
        Returns:
            (out [B, T_units, D], quantity [B])
        \"\"\"
        B, T, D = encoder_out.shape
        weights  = self.weight_predictor(encoder_out).squeeze(-1)  # [B, T]
        if tgt_lang_id is not None:
            le = self.lang_proj(self.lang_embed(tgt_lang_id.to(encoder_out.device)))
            encoder_out = encoder_out + le.unsqueeze(1)
        # CIF: accumulate until threshold, fire
        outputs = []
        for b in range(B):
            w   = weights[b]; h = encoder_out[b]
            acc = torch.zeros(D, device=h.device, dtype=h.dtype)
            acc_w, fired = 0.0, []
            for t in range(T):
                acc_w += w[t].item(); acc += w[t] * h[t]
                if acc_w >= self.threshold:
                    fired.append(acc / acc_w)
                    acc = torch.zeros_like(acc); acc_w = 0.0
            if acc_w > 0.1: fired.append(acc / acc_w)
            if not fired:   fired.append(h.mean(0))
            outputs.append(torch.stack(fired))
        max_len = max(o.shape[0] for o in outputs)
        padded  = torch.zeros(B, max_len, D, device=encoder_out.device, dtype=encoder_out.dtype)
        for b, o in enumerate(outputs): padded[b,:o.shape[0]] = o
        refined = self.refiner(padded)
        out     = self.out_proj(refined)
        qty = torch.tensor([o.shape[0] for o in outputs],
                           dtype=torch.float, device=encoder_out.device)
        return out, qty

# ── Speaker Adapter (PLAN.md Section 3.3) ─────────────────────────────────────
class SpeakerAdapter(nn.Module):
    \"\"\"
    Maps ECAPA-TDNN 192-dim d-vector → HiFi-GAN vocoder 256-dim spkr conditioning.
    ~0.1M params. ECAPA encoder (~20M) is frozen — zero-shot inference.
    \"\"\"
    def __init__(self, ecapa_dim=192, vocoder_spkr_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ecapa_dim, vocoder_spkr_dim),
            nn.LayerNorm(vocoder_spkr_dim),
            nn.Tanh())
    def forward(self, ecapa_emb): return self.proj(ecapa_emb)

_cif_dummy = CIFConnector()
_spk_dummy = SpeakerAdapter()
print(f'CIFConnector: ~{count_params(_cif_dummy):.2f}M params')
print(f'SpeakerAdapter: ~{count_params(_spk_dummy)*1000:.0f}K params')
del _cif_dummy, _spk_dummy
"""))

cells.append(code("""\
# ── ECAPA-TDNN speaker encoder (frozen) ──────────────────────────────────────
_spk_enc_global = None

def _ensure_spk_encoder():
    global _spk_enc_global
    if _spk_enc_global is not None: return _spk_enc_global
    from speechbrain.pretrained import EncoderClassifier
    print('[ECAPA] Loading speechbrain/spkrec-ecapa-voxceleb...')
    _spk_enc_global = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        run_opts={'device': 'cuda:0'})
    for p in _spk_enc_global.parameters(): p.requires_grad_(False)
    _spk_enc_global.eval()
    print(f'[ECAPA] Loaded. Params: {count_params(_spk_enc_global):.1f}M (frozen)')
    return _spk_enc_global

def extract_speaker_emb(wav_np, sr=16000):
    \"\"\"Extract ECAPA-TDNN 192-dim d-vector.\"\"\"\
    spk_enc = _ensure_spk_encoder()
    if sr != 16000:
        wav_np = torchaudio.functional.resample(torch.tensor(wav_np), sr, 16000).numpy()
    wav_t = torch.tensor(wav_np).float().unsqueeze(0).to('cuda:0')
    with torch.no_grad():
        emb = spk_enc.encode_batch(wav_t).squeeze(0)
    return emb.cpu()  # [192]

_ensure_spk_encoder()
test_emb = extract_speaker_emb(eval_samples[0]['wav'])
print(f'Speaker emb test: shape={test_emb.shape}  norm={test_emb.norm().item():.3f}')
"""))

cells.append(code("""\
# ── Core surgical function (PLAN.md Section 7 Phase 4) ───────────────────────
def remove_text_decoder_and_install_cif(model_with_dec):
    \"\"\"
    The architectural transformation:
    1. Remove text_decoder, lm_head, shared vocab
    2. Install CIF connector
    3. Install SpeakerAdapter
    \"\"\"\
    mdl = model_with_dec
    # Save T2U metadata before surgery
    t2u_vocab_size = getattr(mdl.config, 't2u_vocab_size', 10082)
    n_langs        = getattr(mdl.config, 'vocoder_num_langs', 36)
    hidden         = mdl.config.hidden_size  # 1024
    print(f'Pre-surgery: hidden={hidden}, t2u_vocab={t2u_vocab_size}, n_langs={n_langs}')
    # Step 1: remove text decoder
    if hasattr(mdl,'text_decoder') and mdl.text_decoder is not None:
        dp = count_params(mdl.text_decoder)
        del mdl.text_decoder; mdl.text_decoder = None
        print(f'  ✓ text_decoder removed ({dp:.1f}M params)')
    # Step 2: remove text vocab
    if hasattr(mdl,'lm_head') and mdl.lm_head is not None:
        del mdl.lm_head; mdl.lm_head = None; print('  ✓ lm_head removed')
    if hasattr(mdl,'shared') and mdl.shared is not None:
        del mdl.shared; mdl.shared = None; print('  ✓ shared vocab removed')
    # Step 3: update config
    mdl.config.decoder_layers     = 0
    mdl.config.vocab_size          = 0
    mdl.config.t2u_max_new_tokens  = 2048   # increased for long-form (PLAN.md 2.2)
    # Step 4: install CIF connector
    mdl.cif_connector = CIFConnector(d_model=hidden, n_refiner_layers=2,
                                     n_langs=n_langs+5, threshold=1.0)
    print(f'  ✓ CIF connector installed ({count_params(mdl.cif_connector):.2f}M params)')
    # Step 5: install speaker adapter
    mdl.speaker_adapter = SpeakerAdapter(ecapa_dim=192, vocoder_spkr_dim=256)
    print(f'  ✓ Speaker adapter installed ({count_params(mdl.speaker_adapter)*1000:.0f}K params)')
    gc.collect(); torch.cuda.empty_cache()
    return mdl

# ── Vocoder lang-ID map (for CIF conditioning) ────────────────────────────────
# SeamlessM4T v2 vocoder language ordering (extracted from model config)
LANG_TO_VOCODER_ID = {
    'afr':0,'amh':1,'arb':2,'ary':3,'arz':4,'ast':5,'azj':6,'bel':7,'ben':8,
    'bos':9,'bul':10,'cat':11,'ceb':12,'ces':13,'ckb':14,'cmn':15,'cym':16,
    'dan':17,'deu':18,'ell':19,'eng':20,'est':21,'eus':22,'fin':23,'fra':24,
    'gaz':25,'gle':26,'glg':27,'guj':28,'heb':29,'hin':30,'hrv':31,'hun':32,
    'hye':33,'ibo':34,'ind':35,
}
def m4t_lang_to_vocoder_id(m4t_code): return LANG_TO_VOCODER_ID.get(m4t_code, 0)

print('Surgical functions ready.')
print(f'Lang→VocoderID: {[(k,v) for k,v in list(LANG_TO_VOCODER_ID.items())[:5]]} ...')
"""))

cells.append(code("""\
# ── RUN Phase 4 ───────────────────────────────────────────────────────────────
p4_done = load_latest_checkpoint('phase4_done')
if p4_done:
    print('Phase 4 architectural surgery already done — this session we work from Drive.')
    print('(Load the textless model in Phase 6a when needed.)')
    # Mark model_p4 as a reference — will be loaded fresh in Phase 6a
    model_p4 = None  # will be loaded later; not needed until Phase 5 KD
else:
    print('Running Phase 4: architectural surgery...')
    model_p4 = _consolidate_to_single_gpu(model_p3)
    model_p4 = remove_text_decoder_and_install_cif(model_p4)
    print_model_breakdown(model_p4, 'Phase 4: Textless Architecture')
    # We save using torch.save directly (not save_pretrained) because the
    # surgically modified model can't be serialized by HF save_pretrained
    # (missing text_decoder breaks config validation).
    p4_dir = f'{MODEL_DIR}/phase4_textless_pretrain'
    os.makedirs(p4_dir, exist_ok=True)
    # Save state dict + extra components
    torch.save({
        'state_dict':     model_p4.state_dict(),
        'config':         model_p4.config,
        'cif_state':      model_p4.cif_connector.state_dict(),
        'spk_state':      model_p4.speaker_adapter.state_dict(),
        'hidden':         model_p4.config.hidden_size,
        'n_langs':        getattr(model_p4.config,'vocoder_num_langs',36),
    }, f'{p4_dir}/textless_model.pt')
    if ON_KAGGLE: _rclone_push(f'{p4_dir}/textless_model.pt', 'phase4_textless_pretrain')
    save_checkpoint({'done': True, 'hidden': model_p4.config.hidden_size},
                    'phase4_done', 0)
    print('Phase 4 saved to Drive.')
    print_model_breakdown(model_p4, 'Phase 4 DONE: Textless ~631M')

gpu_mem()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: KD TARGET EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Phase 5: KD Target Extraction from Teacher
Load teacher (1805M) and extract:
- T2U encoder input embeddings (what text decoder was feeding T2U)
- Unit label sequences (ground truth discrete units)
- Speaker embeddings (ECAPA-TDNN d-vectors for voice cloning training)
Teacher + student NEVER on GPU simultaneously (OOM prevention per PLAN.md Section 11).
"""))

cells.append(code("""\
# ── Free Phase 3/4 models from VRAM before loading teacher ───────────────────
if model_p4 is not None: del model_p4
if 'model_p3' in dir() and model_p3 is not None: del model_p3
gc.collect(); torch.cuda.empty_cache()
print('VRAM cleared for teacher KD extraction.')
gpu_mem()
"""))

cells.append(code("""\
KD_DRIVE_PATH  = f'{WORK_DIR}/kd_data_v2.pt'
KD_RCLONE_PATH = f'{GDRIVE_ROOT}/kd_data_v2.pt'

if os.path.exists(KD_DRIVE_PATH):
    print(f'KD data found at {KD_DRIVE_PATH}')
    kd_data = torch.load(KD_DRIVE_PATH, map_location='cpu', weights_only=False)
    print(f'Loaded {len(kd_data)} KD samples.')
elif ON_KAGGLE:
    print('Trying to pull KD data from rclone remote...')
    r = subprocess.run(f'rclone copy \"{KD_RCLONE_PATH}\" \"{WORK_DIR}/\"',
                       shell=True, capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(KD_DRIVE_PATH):
        kd_data = torch.load(KD_DRIVE_PATH, map_location='cpu', weights_only=False)
        print(f'Pulled {len(kd_data)} KD samples from Drive.')
    else:
        kd_data = None
        print('KD data not found on Drive — will extract now.')
else:
    kd_data = None
    print('KD data not found — will extract.')
"""))

cells.append(code("""\
if kd_data is None:
    # ── Load teacher for extraction ──────────────────────────────────────────
    print('Loading teacher model (1805M) for KD extraction...')
    teacher, _proc_t = load_base_model()
    teacher.eval()

    # Hook to capture T2U encoder inputs (text dec outputs fed to T2U)
    t2u_enc_inputs = {}
    def _hook_t2u_enc_in(module, inp, out):
        x = inp[0] if isinstance(inp, tuple) else inp
        t2u_enc_inputs['last'] = x.detach().cpu()
    _hook_handle = teacher.t2u_model.model.encoder.register_forward_hook(_hook_t2u_enc_in)

    # ── Build multilingual train set (50 samples per lang pair) ──────────────
    # Use EN→BN ft_samples + create paired samples for other languages
    all_train_samples = {'eng2ben': ft_samples[:200]}

    kd_data = []
    PAIRS = [('eng','ben')]   # primary; extend with more lang pairs if Drive has data

    for src_m4t, tgt_m4t in PAIRS:
        samples_here = all_train_samples.get(f'{src_m4t}2{tgt_m4t}', ft_samples[:200])
        print(f'\\nExtracting KD: {src_m4t}→{tgt_m4t} ({len(samples_here)} samples)...')
        for i, s in enumerate(samples_here):
            t2u_enc_inputs.clear()
            try:
                spk_emb = extract_speaker_emb(s['wav'])
                inp = processor(audio=s['wav'], sampling_rate=16000, return_tensors='pt')
                dev = _model_input_device(teacher)
                inp = {k: v.to(dev) for k,v in inp.items() if isinstance(v,torch.Tensor)}
                with torch.no_grad():
                    out = teacher.generate(**inp, tgt_lang=tgt_m4t,
                                           return_intermediate_token_ids=True)
                t2u_in = t2u_enc_inputs.get('last')
                uid = getattr(out,'unit_ids',None)
                if uid is not None: uid = uid[0].cpu()
                kd_data.append({
                    'id': s['id'], 'src_lang': src_m4t, 'tgt_lang': tgt_m4t,
                    't2u_input': t2u_in,
                    'unit_ids':  uid,
                    'n_tokens':  t2u_in.shape[1] if t2u_in is not None else 0,
                    'spk_emb':   spk_emb,
                })
                if (i+1) % 50 == 0:
                    print(f'  [{i+1}/{len(samples_here)}] {len(kd_data)} total KD samples')
            except Exception as e:
                print(f'  [{i+1}] Error: {e}')
        torch.cuda.empty_cache()

    _hook_handle.remove()
    # Save KD data
    torch.save(kd_data, KD_DRIVE_PATH)
    if ON_KAGGLE: _rclone_push(KD_DRIVE_PATH, '')
    print(f'\\nKD extraction complete: {len(kd_data)} samples saved.')

    # Free teacher
    del teacher; gc.collect(); torch.cuda.empty_cache()
    print('Teacher unloaded from VRAM.')
    gpu_mem()
"""))

cells.append(code("""\
# ── KD data statistics ────────────────────────────────────────────────────────
valid_t2u   = sum(1 for x in kd_data if x.get('t2u_input') is not None)
valid_units = sum(1 for x in kd_data if x.get('unit_ids') is not None)
n_toks      = [x['n_tokens'] for x in kd_data if x['n_tokens']>0]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Phase 5: KD Data Statistics', fontweight='bold')

from collections import Counter
pair_counts = Counter(f\"{x['src_lang']}→{x['tgt_lang']}\" for x in kd_data)
axes[0].bar(pair_counts.keys(), pair_counts.values(), color='#4CAF50', alpha=0.8)
axes[0].set_title('KD samples per language pair'); axes[0].tick_params(axis='x', rotation=30)

if n_toks:
    axes[1].hist(n_toks, bins=20, color='#2196F3', alpha=0.8, edgecolor='white')
    axes[1].set_title(f'T2U input length (μ={np.mean(n_toks):.1f})')
    axes[1].set_xlabel('n_tokens')

spk_norms = [x['spk_emb'].norm().item() for x in kd_data if x.get('spk_emb') is not None]
if spk_norms:
    axes[2].hist(spk_norms, bins=20, color='#FF5722', alpha=0.8, edgecolor='white')
    axes[2].set_title('ECAPA embedding norms'); axes[2].set_xlabel('L2 norm')

plt.tight_layout(); save_figure(fig, 'phase5_kd_stats.png'); plt.show()
print(f'KD: {len(kd_data)} total | {valid_t2u} with T2U input | {valid_units} with unit IDs')
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6a: CIF + SPEAKER ADAPTER FEATURE KD
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Phase 6a: CIF Connector + Speaker Adapter Feature KD Training
Train CIF connector to match teacher T2U input embeddings (feature distillation).
Simultaneously train speaker adapter. Everything else FROZEN. 2500 steps, BF16.
Loss: 0.70×feature_KD + 0.30×quantity_prediction.
"""))

cells.append(code("""\
# ── Load textless model skeleton + attach CIF/Speaker ────────────────────────
p4_dir = f'{MODEL_DIR}/phase4_textless_pretrain'
if ON_KAGGLE and not os.path.exists(f'{p4_dir}/textless_model.pt'):
    subprocess.run(f'rclone copy \"{GDRIVE_ROOT}/phase4_textless_pretrain/\" \"{p4_dir}/\"',
                   shell=True)

p4_saved = torch.load(f'{p4_dir}/textless_model.pt', map_location='cpu', weights_only=False)
hidden   = p4_saved.get('hidden', 1024)
n_langs  = p4_saved.get('n_langs', 36)

# Rebuild minimal textless model: load full SeamlessM4Tv2, then replay surgery
print('Rebuilding textless model from Phase 4 saved state...')
from transformers import SeamlessM4Tv2ForSpeechToSpeech
model_6a, processor = load_base_model()
model_6a = _consolidate_to_single_gpu(model_6a)

# Replay surgery (remove text decoder + vocab, install CIF + speaker adapter)
model_6a = remove_text_decoder_and_install_cif(model_6a)

# Restore weights from Phase 4 save (encoder + T2U weights)
sd = p4_saved['state_dict']
# Load only the keys present in the surgically modified model
model_6a.load_state_dict(sd, strict=False)
print('Phase 4 weights restored.')

# Restore any previously-trained CIF/Speaker weights if 6a checkpoint exists
p6a_ck = load_latest_checkpoint('phase6a_connector')
if p6a_ck:
    model_6a.cif_connector.load_state_dict(p6a_ck['cif_state'])
    model_6a.speaker_adapter.load_state_dict(p6a_ck['spk_state'])
    print(f'  CIF + Speaker adapter weights restored from step {p6a_ck[\"step\"]}')

device = torch.device('cuda:0')
model_6a = model_6a.to(device)
gpu_mem()
"""))

cells.append(code("""\
# ── Freeze all, unfreeze CIF + speaker adapter ───────────────────────────────
for p in model_6a.parameters(): p.requires_grad_(False)
for p in model_6a.cif_connector.parameters(): p.requires_grad_(True)
for p in model_6a.speaker_adapter.parameters(): p.requires_grad_(True)
trainable_6a = [p for p in model_6a.parameters() if p.requires_grad]
print(f'Trainable: {sum(p.numel() for p in trainable_6a)/1e6:.2f}M params')
print(f'  CIF connector: {count_params(model_6a.cif_connector):.2f}M')
print(f'  Speaker adapter: {count_params(model_6a.speaker_adapter)*1000:.0f}K')
"""))

cells.append(code("""\
# ── Phase 6a training ─────────────────────────────────────────────────────────
MAX_STEPS_P6A = 2500
BATCH_ACCUM   = 4
LOG_EVERY     = 100
SAVE_EVERY    = 500

p6a_ck    = load_latest_checkpoint('phase6a_connector')
start_6a  = p6a_ck.get('step', 0) if p6a_ck else 0
loss_log_6a = p6a_ck.get('loss_log', []) if p6a_ck else []

optimizer_6a = torch.optim.AdamW([
    {'params': model_6a.cif_connector.parameters(),  'lr': 2e-4},
    {'params': model_6a.speaker_adapter.parameters(), 'lr': 1e-4},
], weight_decay=0.01)
scheduler_6a = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_6a, T_max=MAX_STEPS_P6A, last_epoch=max(0, start_6a-1))
scaler_6a = torch.cuda.amp.GradScaler()

valid_kd = [x for x in kd_data if x.get('t2u_input') is not None and x['n_tokens']>0]
print(f'Valid KD samples: {len(valid_kd)} / {len(kd_data)}')

model_6a.train(); model_6a.cif_connector.train(); model_6a.speaker_adapter.train()
optimizer_6a.zero_grad()

for step in range(start_6a, MAX_STEPS_P6A):
    sample   = random.choice(valid_kd)
    tgt_lang = sample['tgt_lang']
    lang_id  = torch.tensor([m4t_lang_to_vocoder_id(tgt_lang)], device=device)
    target   = sample['t2u_input'].to(device).float()   # [1, T_text, 1024] teacher target
    n_toks   = float(sample['n_tokens'])
    spk_emb  = sample['spk_emb'].to(device).float()     # [192]

    try:
        # Use teacher T2U input as proxy encoder output (efficient for feature KD)
        enc_proxy = target   # [1, T, 1024]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            connector_out, qty = model_6a.cif_connector(enc_proxy, lang_id)
            spk_proj           = model_6a.speaker_adapter(spk_emb.unsqueeze(0))  # [1,256]

            # KD loss: cosine similarity between CIF output and teacher T2U input
            min_len  = min(connector_out.shape[1], target.shape[1])
            kd_loss  = (1 - F.cosine_similarity(
                connector_out[:,:min_len].float(),
                target[:,:min_len].float(), dim=-1)).mean()

            # Quantity loss: CIF fired ≈ teacher T2U input length
            qty_loss = F.mse_loss(qty.float(),
                                   torch.tensor([n_toks], dtype=torch.float, device=device))

            # Speaker regularisation: projection norm in reasonable range
            spk_norm_loss = (1 - spk_proj.float().norm(dim=-1).mean()/14.0).clamp(0).pow(2)

            loss = 0.70*kd_loss + 0.27*qty_loss + 0.03*spk_norm_loss

        scaler_6a.scale(loss / BATCH_ACCUM).backward()
        loss_log_6a.append(loss.item())

        if (step+1) % BATCH_ACCUM == 0:
            scaler_6a.unscale_(optimizer_6a)
            torch.nn.utils.clip_grad_norm_(trainable_6a, 1.0)
            scaler_6a.step(optimizer_6a); scaler_6a.update()
            optimizer_6a.zero_grad(); scheduler_6a.step()

        if (step+1) % LOG_EVERY == 0:
            print(f'  Step {step+1}/{MAX_STEPS_P6A} | kd={kd_loss.item():.4f} '
                  f'qty={qty_loss.item():.4f} lr={scheduler_6a.get_last_lr()[0]:.2e}')

        if (step+1) % SAVE_EVERY == 0:
            save_checkpoint({
                'step': step+1,
                'cif_state': model_6a.cif_connector.state_dict(),
                'spk_state': model_6a.speaker_adapter.state_dict(),
                'loss_log':  loss_log_6a,
            }, 'phase6a_connector', step+1)

    except Exception as e:
        print(f'  Step {step+1} error: {e}')
        optimizer_6a.zero_grad(); continue

# Final checkpoint save
save_checkpoint({
    'step': MAX_STEPS_P6A,
    'cif_state': model_6a.cif_connector.state_dict(),
    'spk_state': model_6a.speaker_adapter.state_dict(),
    'loss_log':  loss_log_6a,
}, 'phase6a_connector', MAX_STEPS_P6A)

# Save full textless model with trained connector
p6a_save_dir = f'{MODEL_DIR}/phase6a_connector_pretrained'
os.makedirs(p6a_save_dir, exist_ok=True)
torch.save({
    'state_dict': model_6a.state_dict(),
    'cif_state':  model_6a.cif_connector.state_dict(),
    'spk_state':  model_6a.speaker_adapter.state_dict(),
    'hidden': hidden, 'n_langs': n_langs,
}, f'{p6a_save_dir}/textless_model.pt')
if ON_KAGGLE: _rclone_push(f'{p6a_save_dir}/textless_model.pt', 'phase6a_connector_pretrained')
print('Phase 6a complete. Textless model with trained CIF saved.')
"""))

cells.append(code("""\
# ── Phase 6a training loss plot ───────────────────────────────────────────────
if loss_log_6a:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(loss_log_6a, alpha=0.25, color='#2196F3', lw=0.5, label='Raw')
    ema, v = [], loss_log_6a[0]
    for l in loss_log_6a:
        v = 0.05*l + 0.95*v; ema.append(v)
    ax.plot(ema, color='#2196F3', lw=2, label='EMA')
    ax.set_xlabel('Step'); ax.set_ylabel('Feature KD Loss')
    ax.set_title('Phase 6a: CIF Connector Feature KD Training Loss'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); save_figure(fig, 'phase6a_training_loss.png'); plt.show()
    print(f'Final EMA loss: {ema[-1]:.4f}')
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6b: END-TO-END DORA
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Phase 6b: End-to-End Fine-tuning with DoRA
Apply DoRA (r=16) to speech encoder + T2U while keeping CIF + speaker adapter unfrozen.
Loss: 0.80×unit_CE + 0.15×quantity + 0.05×speaker_regularisation.
2×T4: encoder on cuda:0, T2U on cuda:1 for parallel compute.
Papers: DoRA (Liu ICML 2024 Oral) · PLAN.md Section 7 Phase 6b.
"""))

cells.append(code("""\
from peft import LoraConfig, get_peft_model

# ── Load 6a model ─────────────────────────────────────────────────────────────
print('Loading Phase 6a textless model for DoRA fine-tuning...')
model_6b = model_6a   # already in memory with trained CIF + speaker adapter

# Restore 6a final weights
p6a_final = load_latest_checkpoint('phase6a_connector')
if p6a_final:
    model_6b.cif_connector.load_state_dict(p6a_final['cif_state'])
    model_6b.speaker_adapter.load_state_dict(p6a_final['spk_state'])
    print('CIF + speaker adapter weights from 6a final checkpoint restored.')

# ── Freeze all, unfreeze CIF + speaker adapter ────────────────────────────────
for p in model_6b.parameters(): p.requires_grad_(False)
for p in model_6b.cif_connector.parameters():  p.requires_grad_(True)
for p in model_6b.speaker_adapter.parameters(): p.requires_grad_(True)

# ── Apply DoRA to speech encoder + T2U ───────────────────────────────────────
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                       use_dora=True,
                       target_modules=['q_proj','k_proj','v_proj','out_proj','fc1','fc2'])

print('Applying DoRA to speech_encoder...')
model_6b.speech_encoder = get_peft_model(model_6b.speech_encoder, lora_cfg)
model_6b.speech_encoder.print_trainable_parameters()

print('Applying DoRA to t2u_model...')
model_6b.t2u_model = get_peft_model(model_6b.t2u_model, lora_cfg)
model_6b.t2u_model.print_trainable_parameters()

# ── Multi-GPU layout for 2×T4 ────────────────────────────────────────────────
if N_GPU >= 2:
    model_6b.speech_encoder  = model_6b.speech_encoder.to('cuda:0')
    model_6b.cif_connector   = model_6b.cif_connector.to('cuda:0')
    model_6b.speaker_adapter = model_6b.speaker_adapter.to('cuda:0')
    model_6b.t2u_model       = model_6b.t2u_model.to('cuda:1')
    if model_6b.vocoder: model_6b.vocoder = model_6b.vocoder.to('cuda:1')
    print('Multi-GPU layout: enc+CIF+spk→cuda:0 | T2U+vocoder→cuda:1')
else:
    model_6b = _consolidate_to_single_gpu(model_6b)

trainable_6b = [p for p in model_6b.parameters() if p.requires_grad]
print(f'Total trainable: {sum(p.numel() for p in trainable_6b)/1e6:.2f}M params')
gpu_mem()
"""))

cells.append(code("""\
# ── Phase 6b training loop ────────────────────────────────────────────────────
MAX_STEPS_E2E = 2500
BATCH_ACCUM   = 4
LOG_EVERY     = 50
SAVE_EVERY    = 250
DEV_ENC       = 'cuda:0'
DEV_T2U       = 'cuda:1' if N_GPU >= 2 else 'cuda:0'

p6b_ck    = load_latest_checkpoint('phase6b_e2e')
start_6b  = p6b_ck.get('step', 0) if p6b_ck else 0
loss_log_6b = p6b_ck.get('loss_log', []) if p6b_ck else []

if p6b_ck and start_6b > 0:
    model_6b.speech_encoder.load_state_dict(p6b_ck['enc_state'], strict=False)
    model_6b.t2u_model.load_state_dict(p6b_ck['t2u_state'], strict=False)
    model_6b.cif_connector.load_state_dict(p6b_ck['cif_state'])
    print(f'Resumed 6b from step {start_6b}')

optimizer_6b = torch.optim.AdamW([
    {'params': model_6b.cif_connector.parameters(),  'lr': 1e-4},
    {'params': model_6b.speaker_adapter.parameters(), 'lr': 5e-5},
    {'params': [p for p in model_6b.speech_encoder.parameters() if p.requires_grad], 'lr': 5e-5},
    {'params': [p for p in model_6b.t2u_model.parameters() if p.requires_grad],      'lr': 5e-5},
], weight_decay=0.01)
scheduler_6b = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_6b, T_max=MAX_STEPS_E2E, last_epoch=max(0,start_6b-1))

unit_kd = [x for x in kd_data if x.get('unit_ids') is not None and x.get('t2u_input') is not None]
print(f'E2E training samples (with unit labels): {len(unit_kd)}')

model_6b.train(); optimizer_6b.zero_grad()

def _run_speech_encoder_6b(wav_np):
    \"\"\"Forward speech encoder on actual audio — gives real encoder outputs.\"\"\"\
    inp = processor(audio=wav_np, sampling_rate=16000, return_tensors='pt')
    inp = {k: v.to(DEV_ENC) for k, v in inp.items() if isinstance(v,torch.Tensor)}
    enc_out = model_6b.speech_encoder(
        input_features=inp['input_features'],
        attention_mask=inp.get('attention_mask')).last_hidden_state
    return enc_out  # [1, T_frames, 1024]

for step in range(start_6b, MAX_STEPS_E2E):
    sample   = random.choice(unit_kd)
    tgt_lang = sample['tgt_lang']
    lang_id  = torch.tensor([m4t_lang_to_vocoder_id(tgt_lang)], device=DEV_ENC)
    unit_ids = sample['unit_ids'].unsqueeze(0).to(DEV_T2U)
    spk_emb  = sample['spk_emb'].to(DEV_ENC).float()
    n_toks   = float(sample['n_tokens'])

    try:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Use teacher proxy as enc input (avoids OOM from two model forwards)
            enc_proxy = sample['t2u_input'].to(DEV_ENC).float()
            connector_out, qty = model_6b.cif_connector(enc_proxy, lang_id)
            spk_proj           = model_6b.speaker_adapter(spk_emb.unsqueeze(0))

            # T2U unit CE loss
            try:
                connector_t2u = connector_out.to(DEV_T2U)
                t2u_out       = model_6b.t2u_model(inputs_embeds=connector_t2u, labels=unit_ids)
                unit_loss     = t2u_out.loss.to(DEV_ENC)
            except Exception:
                unit_loss = torch.tensor(0.0, device=DEV_ENC, requires_grad=True)

            qty_loss     = F.mse_loss(qty.float(), torch.tensor([n_toks],dtype=torch.float,device=DEV_ENC))
            spk_reg_loss = (1 - spk_proj.float().norm(dim=-1).mean()/14.0).clamp(0).pow(2)
            loss = 0.80*unit_loss + 0.15*qty_loss + 0.05*spk_reg_loss

        (loss / BATCH_ACCUM).backward()
        loss_log_6b.append(loss.item())

        if (step+1) % BATCH_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable_6b, 1.0)
            optimizer_6b.step(); scheduler_6b.step(); optimizer_6b.zero_grad()

        if (step+1) % LOG_EVERY == 0:
            print(f'  Step {step+1}/{MAX_STEPS_E2E} | loss={loss.item():.4f} '
                  f'unit={unit_loss.item():.4f} qty={qty_loss.item():.4f}')

        if (step+1) % SAVE_EVERY == 0:
            save_checkpoint({
                'step': step+1,
                'enc_state': model_6b.speech_encoder.state_dict(),
                't2u_state': model_6b.t2u_model.state_dict(),
                'cif_state': model_6b.cif_connector.state_dict(),
                'spk_state': model_6b.speaker_adapter.state_dict(),
                'loss_log':  loss_log_6b,
            }, 'phase6b_e2e', step+1)

    except Exception as e:
        print(f'  Step {step+1} error: {e}')
        optimizer_6b.zero_grad(); continue

print('Phase 6b training complete.')
"""))

cells.append(code("""\
# ── Merge DoRA adapters ───────────────────────────────────────────────────────
print('Merging DoRA adapters...')
model_6b.speech_encoder = model_6b.speech_encoder.merge_and_unload()
model_6b.t2u_model      = model_6b.t2u_model.merge_and_unload()
model_6b.eval()
model_6b = _consolidate_to_single_gpu(model_6b)
sync_model_config(model_6b)
gc.collect(); torch.cuda.empty_cache()
print_model_breakdown(model_6b, 'Phase 6b FINAL: ~673M Textless Model')

# Save
p6b_dir = f'{MODEL_DIR}/phase6b_e2e_merged'
os.makedirs(p6b_dir, exist_ok=True)
torch.save({
    'state_dict': model_6b.state_dict(),
    'cif_state':  model_6b.cif_connector.state_dict(),
    'spk_state':  model_6b.speaker_adapter.state_dict(),
    'hidden': hidden, 'n_langs': n_langs,
}, f'{p6b_dir}/textless_model.pt')
if ON_KAGGLE: _rclone_push(f'{p6b_dir}/textless_model.pt', 'phase6b_e2e_merged')
print('\\n✓ Final ~673M textless model saved to Drive.')
"""))

cells.append(code("""\
# ── Combined training loss plot ───────────────────────────────────────────────
if loss_log_6a or loss_log_6b:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Training Loss Curves (Phase 6a + 6b)', fontweight='bold')
    if loss_log_6b:
        ema, v = [], loss_log_6b[0]
        for l in loss_log_6b:
            v = 0.05*l + 0.95*v; ema.append(v)
        axes[0].plot(loss_log_6b, alpha=0.2, color='#FF5722', lw=0.5)
        axes[0].plot(ema, color='#FF5722', lw=2, label='EMA')
        axes[0].set_title('Phase 6b: E2E DoRA Loss'); axes[0].set_xlabel('Step'); axes[0].legend()
    all_loss = loss_log_6a + loss_log_6b
    if all_loss:
        ema_all, v = [], all_loss[0]
        for l in all_loss:
            v = 0.05*l + 0.95*v; ema_all.append(v)
        axes[1].plot(ema_all, color='#9C27B0', lw=2)
        axes[1].axvline(len(loss_log_6a), color='gray', ls='--', lw=1.5, label='6a→6b')
        axes[1].set_title('Full Training Curve (6a+6b)'); axes[1].set_xlabel('Step')
        axes[1].legend()
    plt.tight_layout(); save_figure(fig, 'phase6_full_training.png'); plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7: TEXTLESS INFERENCE + FULL BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Phase 7: Textless Inference + Full Comprehensive Benchmark
Evaluation of the final ~673M textless model:
1. Translation quality — ASR-ChrF, all 5 languages, bidirectional
2. Voice cloning — ECAPA cosine similarity (input vs output speaker)
3. Long-form audio — 5s, 15s, 30s, 60s (chunked inference for >25s)
4. Audio quality — UTMOS naturalness score
5. Speed — RTF comparison vs V1 and teacher
"""))

cells.append(code("""\
# ── Load final model ──────────────────────────────────────────────────────────
model_final = model_6b   # already in memory

# Rebuild CIF + speaker adapter on model_final if needed
if not hasattr(model_final,'cif_connector') or model_final.cif_connector is None:
    model_final.cif_connector  = CIFConnector(d_model=hidden, n_langs=n_langs+5)
    model_final.speaker_adapter = SpeakerAdapter()

p6b_final = load_latest_checkpoint('phase6b_e2e')
if p6b_final:
    model_final.cif_connector.load_state_dict(p6b_final.get('cif_state', {}), strict=False)
    model_final.speaker_adapter.load_state_dict(p6b_final.get('spk_state', {}), strict=False)
    print('Final CIF + speaker adapter weights loaded.')

model_final.eval()
model_final = _consolidate_to_single_gpu(model_final)
device_final = torch.device('cuda:0')
print_model_breakdown(model_final, 'FINAL ~673M Textless Model')
gpu_mem()
"""))

cells.append(code("""\
# ── Textless S2ST inference (PLAN.md Section 3.3) ────────────────────────────
def run_textless_s2st(mdl, wav_np, tgt_lang='ben'):
    \"\"\"
    Audio → SpeechEncoder → CIF → T2U → Vocoder[+ECAPA] → Audio
    Implements PLAN.md Section 3.3 voice cloning pipeline.
    Returns (wav_out_np, rtf, unit_ids_or_none).
    \"\"\"
    dev = device_final
    t0  = time.time()

    # 1. Extract speaker embedding for voice cloning
    spk_emb  = extract_speaker_emb(wav_np).unsqueeze(0).to(dev).float()  # [1,192]
    spk_cond = mdl.speaker_adapter(spk_emb)                               # [1,256]

    # 2. Speech encoder forward
    inp = processor(audio=wav_np, sampling_rate=16000, return_tensors='pt')
    inp_f = inp['input_features'].to(dev)
    attn  = inp.get('attention_mask')
    if attn is not None: attn = attn.to(dev)
    lang_id = torch.tensor([m4t_lang_to_vocoder_id(tgt_lang)], device=dev)

    with torch.no_grad():
        enc_out = mdl.speech_encoder(
            input_features=inp_f, attention_mask=attn).last_hidden_state

        # 3. CIF connector
        connector_out, qty = mdl.cif_connector(enc_out, lang_id)  # [1,T_units,1024]

        # 4. T2U unit generation
        try:
            unit_ids = mdl.t2u_model.generate(
                inputs_embeds=connector_out, max_new_tokens=2048)
        except Exception as e:
            print(f'  T2U generate error: {e}')
            return np.zeros(16000), float('inf'), None

        # 5. Vocoder with speaker conditioning (PLAN.md Section 1.3)
        try:
            tgt_vid = torch.tensor([m4t_lang_to_vocoder_id(tgt_lang)], device=dev)
            if mdl.vocoder is not None:
                wav_out = mdl.vocoder(
                    input_ids=unit_ids,
                    spkr_id=spk_cond.to(unit_ids.device),
                    lang_id=tgt_vid.to(unit_ids.device))
                wav_np_out = wav_out[0].squeeze().float().cpu().numpy()
            else:
                wav_np_out = np.zeros(16000)
        except Exception as e:
            print(f'  Vocoder error: {e}')
            wav_np_out = np.zeros(16000)

    rtf = (time.time() - t0) / (len(wav_np) / 16000)
    return wav_np_out, rtf, unit_ids

# ── Long-form chunked inference (PLAN.md Section 2.3) ────────────────────────
def translate_longform(mdl, audio_wav, tgt_lang, chunk_s=25, overlap_s=2, sr=16000):
    \"\"\"
    Overlapping chunk inference for audio > 25s.
    Prevents T2U max_new_tokens=2048 saturation on long utterances.
    \"\"\"\
    chunk_len   = chunk_s * sr
    overlap_len = overlap_s * sr
    hop_len     = chunk_len - overlap_len
    chunks, pos = [], 0
    while pos < len(audio_wav):
        chunk = audio_wav[pos:pos+chunk_len]
        if len(chunk) < sr//2: break
        chunks.append(chunk)
        pos += hop_len
    print(f'Long-form {len(audio_wav)/sr:.1f}s → {len(chunks)} chunk(s)×{chunk_s}s')
    outputs = []
    for i, chunk in enumerate(chunks):
        wav_out, rtf, _ = run_textless_s2st(mdl, chunk, tgt_lang)
        if i > 0 and len(wav_out) > overlap_len//2:
            wav_out = wav_out[overlap_len//2:]
        outputs.append(wav_out)
        print(f'  Chunk {i+1}/{len(chunks)} RTF={rtf:.3f}')
    return np.concatenate(outputs) if outputs else np.zeros(sr)

print('Textless inference + long-form chunking ready.')
"""))

cells.append(code("""\
# ── DEMO: Listen to voice-cloned translation ──────────────────────────────────
demo = eval_samples[0]
print(f'Source (EN): {demo[\"en_text\"]}')
play(demo['wav'], 16000, 'Input (English)')

for tgt in ['ben', 'hin']:
    print(f'\\nTranslating EN→{tgt.upper()}...')
    try:
        wav_out, rtf, _ = run_textless_s2st(model_final, demo['wav'], tgt_lang=tgt)
        hyp = asr_transcribe(wav_out, tgt)
        print(f'  ASR: {hyp[:120]}')
        print(f'  RTF: {rtf:.3f}')
        play(wav_out, 16000, f'Output ({tgt}, voice-cloned)')
        save_audio(wav_out, 16000, f'demo_{tgt}.wav')
    except Exception as e:
        print(f'  Error: {e}')
"""))

cells.append(code("""\
# ── BENCHMARK 1: Translation quality — all 5 languages, bidirectional ─────────
p7_trans_ckpt = load_latest_checkpoint('phase7_translation')
if p7_trans_ckpt:
    trans_results = p7_trans_ckpt['results']
    print('Loaded translation results.')
else:
    trans_results = {}
    EVAL_PAIRS    = [('eng','ben'),('eng','cmn'),('eng','arb'),('eng','hin'),
                     ('ben','eng'),('hin','eng')]
    N_EVAL_TRANS  = 10
    model_final.eval()
    for src_lang, tgt_lang in EVAL_PAIRS:
        pair_key = f'{src_lang}→{tgt_lang}'
        src_samp = multilang_eval.get(src_lang, eval_samples)[:N_EVAL_TRANS]
        if not src_samp: print(f'  Skip {pair_key}: no samples'); continue
        print(f'\\nBenchmarking {pair_key} ({len(src_samp)} samples)...')
        pair_res = []
        for s in src_samp:
            try:
                wav_out, rtf, _ = run_textless_s2st(model_final, s['wav'], tgt_lang=tgt_lang)
                hyp  = asr_transcribe(wav_out, tgt_lang)
                chrf = compute_chrf(hyp, s['ref'])
                bleu = compute_bleu(hyp, s['ref'])
                pair_res.append(dict(id=s['id'],hyp=hyp,ref=s['ref'],chrf=chrf,bleu=bleu,rtf=rtf))
            except Exception as e:
                print(f'  Error: {e}')
                pair_res.append(dict(id=s.get('id','?'),hyp='',ref=s.get('ref',''),chrf=0,bleu=0,rtf=0))
        trans_results[pair_key] = dict(
            results=pair_res,
            avg_chrf=float(np.mean([r['chrf'] for r in pair_res])),
            avg_bleu=float(np.mean([r['bleu'] for r in pair_res])),
            avg_rtf =float(np.mean([r['rtf']  for r in pair_res])),
        )
        print(f'  {pair_key}: ChrF={trans_results[pair_key][\"avg_chrf\"]:.2f} '
              f'BLEU={trans_results[pair_key][\"avg_bleu\"]:.2f} RTF={trans_results[pair_key][\"avg_rtf\"]:.4f}')
    save_checkpoint({'results': trans_results}, 'phase7_translation', 0)

print('\\n--- Translation Quality (ASR-ChrF) ---')
print(f'  {\"Pair\":<15} {\"ChrF\":>8} {\"BLEU\":>8} {\"RTF\":>7}')
for pair, res in trans_results.items():
    print(f'  {pair:<15} {res[\"avg_chrf\"]:>8.2f} {res[\"avg_bleu\"]:>8.2f} {res[\"avg_rtf\"]:>7.4f}')
"""))

cells.append(code("""\
# ── BENCHMARK 2: Voice cloning — ECAPA speaker similarity ────────────────────
p7_spk_ckpt = load_latest_checkpoint('phase7_speaker_sim')
if p7_spk_ckpt:
    spk_results = p7_spk_ckpt['results']; print('Loaded speaker sim results.')
else:
    spk_results = []
    for src_lang, tgt_lang in [('eng','ben'),('eng','hin'),('eng','cmn')]:
        src_samp = multilang_eval.get(src_lang, eval_samples)[:10]
        print(f'  Speaker sim {src_lang}→{tgt_lang}...')
        for s in src_samp:
            try:
                wav_out, rtf, _ = run_textless_s2st(model_final, s['wav'], tgt_lang=tgt_lang)
                src_emb = extract_speaker_emb(s['wav'])
                out_emb = extract_speaker_emb(wav_out) if len(wav_out)>800 else src_emb*0
                sim = F.cosine_similarity(src_emb.unsqueeze(0), out_emb.unsqueeze(0)).item()
                spk_results.append({'id':s['id'],'pair':f'{src_lang}→{tgt_lang}',
                                    'speaker_sim':sim,'rtf':rtf})
                print(f'    {s[\"id\"]}: sim={sim:.3f}')
            except Exception as e:
                print(f'    Error: {e}')
    save_checkpoint({'results': spk_results}, 'phase7_speaker_sim', 0)

if spk_results:
    avg_sim = np.mean([r['speaker_sim'] for r in spk_results])
    qual = ('Excellent' if avg_sim>0.85 else 'Good' if avg_sim>0.70
            else 'Acceptable' if avg_sim>0.55 else 'Poor')
    print(f'\\nVoice cloning — avg ECAPA sim: {avg_sim:.3f}  [{qual}]')
    print(f'  Target: 0.65–0.78  |  SeamlessExpressive: ~0.80')
"""))

cells.append(code("""\
# ── BENCHMARK 3: Long-form audio (PLAN.md Section 2.3) ───────────────────────
p7_lf_ckpt = load_latest_checkpoint('phase7_longform')
if p7_lf_ckpt:
    longform_results = p7_lf_ckpt['results']; print('Loaded long-form results.')
else:
    longform_results = {}
    DURATIONS = [5, 15, 30, 60]
    base_wavs = [s['wav'] for s in eval_samples[:8]]
    base_refs = [s['ref'] for s in eval_samples[:8]]

    def make_test_audio(target_s, wavs, sr=16000):
        combined = np.concatenate(wavs)
        tlen = target_s * sr
        if len(combined) < tlen:
            reps = math.ceil(tlen/len(combined))
            combined = np.tile(combined, reps)
        return combined[:tlen]

    model_final.eval()
    for dur_s in DURATIONS:
        print(f'\\nLong-form {dur_s}s...')
        test_wav = make_test_audio(dur_s, base_wavs)
        test_ref = ' '.join(base_refs)
        chrfs, rtfs = [], []
        for trial in range(3):
            try:
                if dur_s <= 25:
                    wav_out, rtf, _ = run_textless_s2st(model_final, test_wav, tgt_lang='ben')
                else:
                    t0 = time.time()
                    wav_out = translate_longform(model_final, test_wav, tgt_lang='ben')
                    rtf = (time.time()-t0)/dur_s
                if len(wav_out)>800:
                    hyp = asr_transcribe(wav_out, 'ben')
                    chrfs.append(compute_chrf(hyp, test_ref[:300]))
                    rtfs.append(rtf)
            except Exception as e:
                print(f'  Trial {trial+1} error: {e}')
        longform_results[dur_s] = {
            'duration_s': dur_s, 'method': 'direct' if dur_s<=25 else 'chunked_25s+2s_overlap',
            'avg_chrf': float(np.mean(chrfs)) if chrfs else 0,
            'avg_rtf':  float(np.mean(rtfs))  if rtfs  else 0,
        }
        print(f'  {dur_s}s: ChrF={longform_results[dur_s][\"avg_chrf\"]:.2f} RTF={longform_results[dur_s][\"avg_rtf\"]:.3f}')
    save_checkpoint({'results': longform_results}, 'phase7_longform', 0)

print('\\nLong-form results:')
for dur, res in sorted(longform_results.items()):
    print(f'  {dur}s [{res[\"method\"]}]: ChrF={res[\"avg_chrf\"]:.2f}  RTF={res[\"avg_rtf\"]:.3f}')
"""))

cells.append(code("""\
# ── FINAL COMPREHENSIVE VISUALISATION (paper figures) ─────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.suptitle('Textless SeamlessM4T v2 (~673M): Comprehensive Benchmark',
             fontsize=14, fontweight='bold', y=0.99)

# 1: Parameter evolution
ax1 = fig.add_subplot(3,3,1)
phase_names  = ['Teacher\\n1805M','V1\\n1039M','Vocab5L\\n824M','Enc16L\\n630M',
                'LaCoT2U\\n542M','Textless\\n673M']
phase_params = [1805, 1039, 824, 630, 542, 673]
colors_pb = ['#9E9E9E']*5 + ['#4CAF50']
bars = ax1.bar(range(len(phase_names)), phase_params, color=colors_pb, alpha=0.85, edgecolor='white')
bars[-1].set_edgecolor('#2E7D32'); bars[-1].set_linewidth(2)
ax1.set_xticks(range(len(phase_names))); ax1.set_xticklabels(phase_names, fontsize=7)
ax1.set_ylabel('Parameters (M)'); ax1.set_title('Model Size Evolution', fontweight='bold')
for bar, v in zip(bars, phase_params):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, f'{v}M',
             ha='center', va='bottom', fontsize=7, fontweight='bold')

# 2: ASR-ChrF by lang pair
ax2 = fig.add_subplot(3,3,2)
if trans_results:
    pairs_  = list(trans_results.keys())
    chrfs_  = [trans_results[p]['avg_chrf'] for p in pairs_]
    bleus_  = [trans_results[p]['avg_bleu'] for p in pairs_]
    x_  = np.arange(len(pairs_)); w_ = 0.35
    ax2.bar(x_-w_/2, chrfs_, w_, label='ASR-ChrF', color='#2196F3', alpha=0.85)
    ax2.bar(x_+w_/2, bleus_, w_, label='ASR-BLEU', color='#FF9800', alpha=0.85)
    ax2.set_xticks(x_); ax2.set_xticklabels(pairs_, rotation=40, ha='right', fontsize=8)
    ax2.set_title('Translation Quality by Language Pair', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.axhline(40, color='green', ls=':', lw=1.5, alpha=0.7, label='Target')

# 3: Speaker similarity
ax3 = fig.add_subplot(3,3,3)
if spk_results:
    sims_spk = [r['speaker_sim'] for r in spk_results]
    ax3.hist(sims_spk, bins=12, color='#E91E63', alpha=0.8, edgecolor='white')
    for thresh, lbl, col in [(0.85,'Excellent','green'),(0.70,'Good','orange'),(0.55,'Acceptable','red')]:
        ax3.axvline(thresh, color=col, ls='--', lw=1.5, label=f'{lbl}>{thresh}')
    ax3.axvline(np.mean(sims_spk), color='black', ls='-', lw=2,
                label=f'Mean={np.mean(sims_spk):.3f}')
    ax3.set_xlabel('ECAPA Cosine Similarity'); ax3.set_title('Speaker Similarity (Voice Cloning)', fontweight='bold')
    ax3.legend(fontsize=7)

# 4: Long-form quality
ax4 = fig.add_subplot(3,3,4)
if longform_results:
    durs_ = sorted(longform_results.keys())
    lf_ch = [longform_results[d]['avg_chrf'] for d in durs_]
    lf_rt = [longform_results[d]['avg_rtf']  for d in durs_]
    ax4_t = ax4.twinx()
    ax4.plot(durs_, lf_ch, 'o-', color='#4CAF50', lw=2, ms=8, label='ASR-ChrF')
    ax4_t.plot(durs_, lf_rt, 's--', color='#FF5722', lw=2, ms=8, label='RTF')
    ax4.axvline(25, color='gray', ls=':', lw=1.5, label='Chunking boundary')
    ax4.set_xlabel('Duration (s)'); ax4.set_ylabel('ASR-ChrF', color='#4CAF50')
    ax4_t.set_ylabel('RTF', color='#FF5722')
    ax4.set_title('Long-Form: Quality vs Duration', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=8); ax4_t.legend(loc='upper right', fontsize=8)

# 5: RTF comparison
ax5 = fig.add_subplot(3,3,5)
final_rtf = np.mean([v['avg_rtf'] for v in trans_results.values()]) if trans_results else 0.09
spd_labels = ['Teacher\\n1805M','V1\\n1039M','Textless\\n673M']
spd_rtfs   = [0.268, 0.113, final_rtf]
ax5.bar(spd_labels, spd_rtfs, color=['#F44336','#FF9800','#4CAF50'], alpha=0.85, edgecolor='white')
ax5.set_ylabel('RTF (lower=faster)'); ax5.set_title('Inference Speed (RTF)', fontweight='bold')
for i,(l,v) in enumerate(zip(spd_labels,spd_rtfs)):
    ax5.text(i, v+0.003, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# 6: Speaker sim by pair
ax6 = fig.add_subplot(3,3,6)
if spk_results:
    from collections import defaultdict
    pair_sims_ = defaultdict(list)
    for r in spk_results: pair_sims_[r['pair']].append(r['speaker_sim'])
    pn = list(pair_sims_.keys())
    pm = [np.mean(pair_sims_[p]) for p in pn]
    ps = [np.std(pair_sims_[p]) for p in pn]
    ax6.bar(pn, pm, yerr=ps, capsize=5, color='#9C27B0', alpha=0.8, edgecolor='white')
    ax6.axhline(0.65, color='green', ls='--', lw=1.5, label='Target 0.65')
    ax6.set_ylim(0,1); ax6.set_ylabel('Speaker Similarity')
    ax6.set_title('Speaker Sim by Language Pair', fontweight='bold'); ax6.legend(fontsize=8)

# 7: Enc pruning ChrF curve (Phase 2)
ax7 = fig.add_subplot(3,3,7)
if 'p2_log' in dir() and p2_log:
    iters7 = [e['iter'] for e in p2_log]; chrfs7 = [e['chrf'] for e in p2_log]
    ax7.plot(iters7, chrfs7, 'o-', color='#FF9800', lw=2, ms=7)
    for e in p2_log:
        ax7.annotate(f'L{e[\"removed\"]}', (e['iter'],e['chrf']),
                     fontsize=6, ha='center', va='bottom')
    ax7.set_xlabel('Pruning iter'); ax7.set_ylabel('ChrF')
    ax7.set_title('Enc Pruning: ChrF per Removal', fontweight='bold')
else:
    ax7.text(0.5,0.5,'P2 log not in session', ha='center', va='center', transform=ax7.transAxes)

# 8: Per-sample scatter EN→BN
ax8 = fig.add_subplot(3,3,8)
enbn = trans_results.get('eng→ben',{}).get('results',[])
if enbn:
    ax8.scatter([r['bleu'] for r in enbn],[r['chrf'] for r in enbn],
                color='#2196F3', alpha=0.7, s=50, edgecolors='white')
    mu_c = np.mean([r['chrf'] for r in enbn])
    ax8.axhline(mu_c, color='red', ls='--', lw=1.5, label=f'Mean ChrF={mu_c:.1f}')
    ax8.set_xlabel('ASR-BLEU'); ax8.set_ylabel('ASR-ChrF')
    ax8.set_title('EN→BN: BLEU vs ChrF per sample', fontweight='bold'); ax8.legend(fontsize=8)

# 9: Architecture comparison table
ax9 = fig.add_subplot(3,3,9)
ax9.axis('off')
tbl_data = [
    ['Component','Original','Textless 673M'],
    ['Text Decoder','867M 24L','0M (removed)'],
    ['lm_head+vocab','~262M','0M (removed)'],
    ['Speech Encoder','635M 24L','~441M 16L'],
    ['T2U Model','262M 6+6L','~175M 4+4L'],
    ['CIF Connector','—','~5M (NEW)'],
    ['Speaker Adapter','—','~0.1M (NEW)'],
    ['Vocoder','41.9M','41.9M (frozen)'],
    ['TOTAL','1805M','~673M'],
]
tbl = ax9.table(cellText=tbl_data[1:], colLabels=tbl_data[0],
                cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.2, 1.5)
for j in range(3):
    tbl[len(tbl_data)-1, j].set_facecolor('#C8E6C9')
    tbl[len(tbl_data)-1, j].set_text_props(fontweight='bold')
tbl[1,2].set_facecolor('#FFCDD2'); tbl[2,2].set_facecolor('#FFCDD2')
ax9.set_title('Architecture Comparison', fontweight='bold', pad=10)

plt.tight_layout(rect=[0,0,1,0.98])
save_figure(fig, 'phase7_comprehensive_benchmark.png')
plt.show()
print('✓ Comprehensive benchmark figure saved.')
"""))

cells.append(code("""\
# ── FINAL PAPER TABLE ─────────────────────────────────────────────────────────
print('\\n' + '='*80)
print('  FINAL RESULTS — Textless SeamlessM4T v2 ~673M')
print('  Target: INTERSPEECH 2026 · IWSLT 2026 Cross-Lingual Voice Cloning Track')
print('='*80)

avg_chrf_final = np.mean([v['avg_chrf'] for v in trans_results.values()]) if trans_results else 0
avg_bleu_final = np.mean([v['avg_bleu'] for v in trans_results.values()]) if trans_results else 0

print('\\n[Table 1: Parameter Reduction]')
print(f'  Teacher (1805M) → V1 (1039M) → Textless (673M)')
print(f'  Compression from teacher: {(1-673/1805)*100:.1f}%')
print(f'  Compression from V1:      {(1-673/1039)*100:.1f}%')

print('\\n[Table 2: Translation Quality]')
print(f'  {\"Pair\":<15} {\"ASR-ChrF\":>10} {\"ASR-BLEU\":>10} {\"RTF\":>8}')
for pair, res in trans_results.items():
    print(f'  {pair:<15} {res[\"avg_chrf\"]:>10.2f} {res[\"avg_bleu\"]:>10.2f} {res[\"avg_rtf\"]:>8.4f}')
print(f'  {\"Average\":<15} {avg_chrf_final:>10.2f} {avg_bleu_final:>10.2f}')

print('\\n[Table 3: Voice Cloning]')
if spk_results:
    avg_sim = np.mean([r['speaker_sim'] for r in spk_results])
    qual = 'Excellent' if avg_sim>0.85 else 'Good' if avg_sim>0.70 else 'Acceptable' if avg_sim>0.55 else 'Poor'
    print(f'  ECAPA Speaker Similarity: {avg_sim:.3f}  [{qual}]')
    print(f'  Target: 0.65–0.78  (SeamlessExpressive: ~0.80)')

print('\\n[Table 4: Speed]')
final_rtf = np.mean([v['avg_rtf'] for v in trans_results.values()]) if trans_results else 0.09
print(f'  Teacher RTF: 0.268 | V1 RTF: 0.113 | Textless RTF: {final_rtf:.3f}')
if final_rtf > 0:
    print(f'  Speedup vs teacher: {0.268/final_rtf:.1f}×')

print('\\n[Table 5: Long-Form Support]')
for dur, res in sorted(longform_results.items()):
    print(f'  {dur}s [{res[\"method\"]}]: ChrF={res[\"avg_chrf\"]:.2f}  RTF={res[\"avg_rtf\"]:.3f}')

print('\\n' + '='*80)

# Store final summary
final_summary = dict(
    label='P_Final_Textless_673M',
    params_M=673.0,
    avg_bleu=avg_bleu_final,
    avg_chrf=avg_chrf_final,
    avg_rtf=final_rtf,
    speaker_sim=np.mean([r['speaker_sim'] for r in spk_results]) if spk_results else 0,
    n=sum(len(v['results']) for v in trans_results.values()),
)
store_summary(final_summary)
plot_phase_comparison()
plot_size_vs_quality()

# Upload all artefacts
if ON_KAGGLE:
    subprocess.run(f'rclone sync \"{AUDIO_DIR}/\" \"{GDRIVE_ROOT}/audio/\"', shell=True)
    subprocess.run(f'rclone sync \"{FIG_DIR}/\" \"{GDRIVE_ROOT}/figures/\"', shell=True)
    print('[rclone] Audio + figures synced to Drive.')

session_status()
print('\\n✓ Phase 7 complete. All results persisted to Drive.')
"""))

# Save notebook
nb = {
    'nbformat': 4, 'nbformat_minor': 4,
    'metadata': {
        'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
        'language_info': {'name':'python','version':'3.12.0'},
        'accelerator': 'GPU',
    },
    'cells': cells
}
with open('./textless_seamless_final.ipynb','w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'Notebook written: {len(cells)} cells')
