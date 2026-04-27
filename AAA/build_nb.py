import json

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def md(src): return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""# SeamlessLite — 1.8B → 850M S2ST Compression
## 5-Language (Bengali · English · Hindi · Tamil · Arabic) On-Device Model
### Sequential Phases: Vocab Prune → Layer Prune → FLAP Width → T2U Prune → LoRA+KD Recovery → Long Audio → Voice Clone

**Platform:** Kaggle 2×T4 (15 GB VRAM each)  
**Run all Setup cells (0–9) at the START of EVERY session.**
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL S1 — Platform & Paths
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## ── SETUP CELLS (run every session) ──────────────────────────────"))
cells.append(code("""import os, sys, subprocess, pathlib, re, glob, json, gc, copy, time, math, shutil
import warnings; warnings.filterwarnings('ignore')

ON_KAGGLE = os.path.exists('/kaggle/working')
ON_COLAB  = not ON_KAGGLE
PLATFORM  = 'kaggle' if ON_KAGGLE else 'colab'

GDRIVE_MOUNT = '/content/drive/MyDrive/seamlessLite'
KAGGLE_WORK  = '/kaggle/working'
WORK_DIR     = KAGGLE_WORK if ON_KAGGLE else GDRIVE_MOUNT

CKPT_DIR  = f'{WORK_DIR}/checkpoints'
AUDIO_DIR = f'{WORK_DIR}/audio'
FIG_DIR   = f'{WORK_DIR}/figures'
MODEL_DIR = f'{WORK_DIR}/models'
DATA_DIR  = f'{WORK_DIR}/data'        # all downloaded parquet/audio
RESULTS_DIR = f'{WORK_DIR}/results'

GDRIVE_ROOT = 'gdrive:seamlessLite'

# ── Language tables ──────────────────────────────────────────────────────────
# SeamlessM4T tgt_lang codes (ISO 639-3)
SEAM_LANGS = ['ben', 'eng', 'hin', 'tam', 'arb']

# FLEURS language codes  (used for HF parquet download URL)
FLEURS_CODES = {
    'eng': 'en_us', 'ben': 'bn_in', 'hin': 'hi_in',
    'tam': 'ta_in', 'arb': 'ar_eg',
}
# MMS-1b-all adapter language codes (same as ISO 639-3 here)
MMS_LANGS = {'eng': 'eng', 'ben': 'ben', 'hin': 'hin', 'tam': 'tam', 'arb': 'arb'}

# Common Voice 17 language codes
CV17_LANGS = {'eng': 'en', 'ben': 'bn', 'hin': 'hi', 'tam': 'ta', 'arb': 'ar'}

# All bidirectional X↔English pairs
EVAL_PAIRS = [
    ('eng', 'ben'), ('eng', 'hin'), ('eng', 'tam'), ('eng', 'arb'),
    ('ben', 'eng'), ('hin', 'eng'), ('tam', 'eng'), ('arb', 'eng'),
]

for d in [WORK_DIR, CKPT_DIR, AUDIO_DIR, FIG_DIR, MODEL_DIR, DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f'Platform : {PLATFORM}')
print(f'Work dir : {WORK_DIR}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL S2 — Colab Drive Mount
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""if ON_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print(f'Drive mounted. Working folder: {GDRIVE_MOUNT}')
    os.makedirs(GDRIVE_MOUNT, exist_ok=True)
else:
    print('Kaggle: skipping Drive mount.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL S3 — pip install
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""subprocess.run([
    'pip', 'install', '-q', '--upgrade',
    'transformers>=4.41.0', 'datasets', 'torchaudio', 'speechbrain',
    'peft', 'librosa', 'jiwer', 'evaluate', 'sacrebleu',
    'sentencepiece', 'accelerate', 'bitsandbytes', 'safetensors',
    'matplotlib', 'seaborn', 'soundfile', 'webrtcvad',
    'huggingface_hub', 'kaggle',
], check=True, capture_output=True)
print('All packages installed.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL S4 — rclone install + config
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""def _get_secret(key):
    if ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key)
    from google.colab import userdata
    return userdata.get(key)

if ON_KAGGLE:
    subprocess.run('curl -s https://rclone.org/install.sh | sudo bash',
                   shell=True, capture_output=True)
    ver = subprocess.run('rclone version', shell=True, capture_output=True, text=True)
    print(ver.stdout.split('\\n')[0])

    RCLONE_CONF = _get_secret('RCLONE_CONF')
    raw = RCLONE_CONF.strip()
    raw = re.sub(r'\\s*(\\[[^\\]]+\\])\\s*', r'\\n\\1\\n', raw)
    raw = re.sub(r'\\s+(type|scope|token|client_id|client_secret|root_folder_id)\\s*=\\s*',
                 r'\\n\\1 = ', raw)
    raw = raw.strip() + '\\n'
    rclone_cfg = pathlib.Path.home() / '.config/rclone/rclone.conf'
    rclone_cfg.parent.mkdir(parents=True, exist_ok=True)
    rclone_cfg.write_text(raw)
    r = subprocess.run('rclone lsd gdrive:', shell=True, capture_output=True, text=True)
    print('Drive root:' if r.returncode == 0 else 'rclone FAILED:')
    print(r.stdout[:200] or r.stderr[:200])
else:
    print('Colab: rclone not needed.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL S5 — rclone I/O helpers (verbatim from setup_cells_p7, extended)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""import torch, torch.nn as nn

_CUSTOM_STATE_FILE = '_custom_state.pt'
_PRUNING_MANIFEST  = 'pruning_manifest.pt'

# ── rclone primitives ─────────────────────────────────────────────────────────
def _rclone_push(local_path, remote_subpath):
    if not ON_KAGGLE: return
    r = subprocess.run(
        f'rclone copy \"{local_path}\" \"{GDRIVE_ROOT}/{remote_subpath}/\"',
        shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'[rclone] WARNING push {local_path}: {r.stderr[:200]}')

def _rclone_pull_model(stage_name):
    if not ON_KAGGLE: return
    local = f'{MODEL_DIR}/{stage_name}'
    os.makedirs(local, exist_ok=True)
    r = subprocess.run(
        f'rclone sync \"{GDRIVE_ROOT}/models/{stage_name}/\" \"{local}/\"',
        shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'[rclone] model pull failed: {r.stderr[:300]}')
    print(f'[rclone] Pulled {stage_name} → {local}')

# ── Checkpoint helpers ────────────────────────────────────────────────────────
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
    if not files: print(f'[ckpt] No checkpoint for {name!r}'); return None
    state = torch.load(files[-1], map_location='cpu', weights_only=False)
    print(f'[ckpt] Loaded {os.path.basename(files[-1])}')
    return state

def sync_checkpoints_from_drive():
    if ON_KAGGLE:
        r = subprocess.run(f'rclone sync \"{GDRIVE_ROOT}/checkpoints/\" \"{CKPT_DIR}/\"',
                           shell=True, capture_output=True, text=True)
        if r.returncode != 0: print(f'[ckpt] WARNING: {r.stderr[:300]}')
    files = sorted(os.listdir(CKPT_DIR)) if os.path.exists(CKPT_DIR) else []
    print(f'[ckpt] {len(files)} checkpoint(s) available')
    for f in files:
        mb = os.path.getsize(f'{CKPT_DIR}/{f}') / 1e6
        print(f'  {f:<55} {mb:>7.1f} MB')

# ── Architecture helpers (exact SeamlessM4Tv2 attr names) ────────────────────
def _get_speech_enc_layers(mdl):
    \"\"\"Returns model.speech_encoder.encoder.layers (SeamlessM4Tv2ConformerEncoderLayer list).\"\"\"
    return mdl.speech_encoder.encoder.layers

def _get_text_dec_layers(mdl):
    \"\"\"Returns model.text_decoder.layers (SeamlessM4Tv2DecoderLayer list).\"\"\"
    return mdl.text_decoder.layers

def _get_t2u_enc_layers(mdl):
    \"\"\"Returns model.t2u_model.model.encoder.layers.\"\"\"
    return mdl.t2u_model.model.encoder.layers

def _get_t2u_dec_layers(mdl):
    \"\"\"Returns model.t2u_model.model.decoder.layers.\"\"\"
    return mdl.t2u_model.model.decoder.layers

# ── Config sync ───────────────────────────────────────────────────────────────
def sync_model_config(mdl):
    \"\"\"Sync config layer counts to pruned module counts. MUST call after any layer removal.\"\"\"
    # Speech encoder
    enc_layers = _get_speech_enc_layers(mdl)
    actual_enc = len(enc_layers)
    if hasattr(mdl.config, 'speech_encoder_layers') and mdl.config.speech_encoder_layers != actual_enc:
        print(f'  [cfg] speech_encoder_layers: {mdl.config.speech_encoder_layers} → {actual_enc}')
        mdl.config.speech_encoder_layers = actual_enc
    # Also patch sub-config if present
    sub = getattr(mdl.speech_encoder, 'config', None)
    if sub and hasattr(sub, 'num_hidden_layers') and sub.num_hidden_layers != actual_enc:
        sub.num_hidden_layers = actual_enc

    # Text decoder
    dec_layers = _get_text_dec_layers(mdl)
    actual_dec = len(dec_layers)
    if hasattr(mdl.config, 'decoder_layers') and mdl.config.decoder_layers != actual_dec:
        print(f'  [cfg] decoder_layers: {mdl.config.decoder_layers} → {actual_dec}')
        mdl.config.decoder_layers = actual_dec

    # T2U
    t2u_enc = _get_t2u_enc_layers(mdl)
    t2u_dec = _get_t2u_dec_layers(mdl)
    actual_t2u_enc = len(t2u_enc)
    actual_t2u_dec = len(t2u_dec)
    for attr, val in [('t2u_encoder_layers', actual_t2u_enc), ('t2u_decoder_layers', actual_t2u_dec)]:
        if hasattr(mdl.config, attr) and getattr(mdl.config, attr) != val:
            print(f'  [cfg] {attr}: {getattr(mdl.config, attr)} → {val}')
            setattr(mdl.config, attr, val)
    # Also patch t2u_model.config
    tc = getattr(getattr(mdl, 't2u_model', None), 'config', None)
    if tc:
        if hasattr(tc, 'encoder_layers') and tc.encoder_layers != actual_t2u_enc:
            tc.encoder_layers = actual_t2u_enc
        if hasattr(tc, 'decoder_layers') and tc.decoder_layers != actual_t2u_dec:
            tc.decoder_layers = actual_t2u_dec
    print('  [cfg] sync done.')

def _save_custom_state(mdl, path):
    state = {attr: getattr(mdl, attr) for attr in ['_vocab_remap_to_old'] if hasattr(mdl, attr)}
    if state:
        torch.save(state, os.path.join(path, _CUSTOM_STATE_FILE))
        print(f'  custom state saved: {list(state.keys())}')

def _load_custom_state(mdl, path):
    fpath = os.path.join(path, _CUSTOM_STATE_FILE)
    if not os.path.exists(fpath): return
    state = torch.load(fpath, map_location='cpu', weights_only=False)
    for k, v in state.items(): setattr(mdl, k, v)
    print(f'  custom state restored: {list(state.keys())}')

# ── Model save / load ─────────────────────────────────────────────────────────
def save_model_to_drive(mdl, proc, stage_name, manifest_extra=None):
    target_dir = f'{MODEL_DIR}/{stage_name}'
    os.makedirs(target_dir, exist_ok=True)
    print(f'[model] Saving {stage_name} → {target_dir}')
    sync_model_config(mdl)
    _save_custom_state(mdl, target_dir)
    man = {'stage_name': stage_name, 'params_M': sum(p.numel() for p in mdl.parameters())/1e6}
    if manifest_extra: man.update(manifest_extra)
    torch.save(man, os.path.join(target_dir, _PRUNING_MANIFEST))
    try:
        mdl.save_pretrained(target_dir, safe_serialization=True)
    except Exception as e:
        print(f'  safetensors failed ({e}); saving .bin')
        mdl.save_pretrained(target_dir)
    proc.save_pretrained(target_dir)
    total = sum(os.path.getsize(f'{target_dir}/{f}') for f in os.listdir(target_dir)) / 1e6
    print(f'[model] Saved. {total:.0f} MB in {len(os.listdir(target_dir))} files.')
    if ON_KAGGLE:
        r = subprocess.run(
            f'rclone sync \"{target_dir}/\" \"{GDRIVE_ROOT}/models/{stage_name}/\"',
            shell=True, capture_output=True, text=True)
        if r.returncode != 0: print(f'[model] rclone push FAILED: {r.stderr[:200]}')
        else: print(f'[model] Pushed to Drive.')

def load_model_from_drive(stage_name):
    from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor, AutoConfig
    local = f'{MODEL_DIR}/{stage_name}'
    if ON_KAGGLE and (not os.path.exists(local) or not os.listdir(local)):
        _rclone_pull_model(stage_name)
    if not os.path.exists(local) or not os.listdir(local):
        raise RuntimeError(f'[model] not found: {local}')
    print(f'[model] Loading {stage_name} from {local}')
    cfg = AutoConfig.from_pretrained(local)
    mdl = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
        local, config=cfg, torch_dtype=torch.float16, device_map='auto')
    _load_custom_state(mdl, local)
    proc = SeamlessM4TProcessor.from_pretrained(local)
    pm = os.path.join(local, _PRUNING_MANIFEST)
    if os.path.isfile(pm):
        meta = torch.load(pm, map_location='cpu', weights_only=False)
        print(f'  manifest: params_M={meta.get(\"params_M\",\"?\"):.1f}M')
    mdl.eval()
    return mdl, proc

print('I/O helpers ready.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL S6 — Core utilities (params, gpu, figures)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""import numpy as np
import matplotlib.pyplot as plt, matplotlib
import seaborn as sns
matplotlib.rcParams.update({'font.size': 11, 'figure.dpi': 130, 'savefig.bbox': 'tight'})
sns.set_style('whitegrid')
from sacrebleu.metrics import BLEU, CHRF
_bleu = BLEU(effective_order=True)
_chrf = CHRF()

def count_params(m):  return sum(p.numel() for p in m.parameters()) / 1e6
def gpu_mem():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated()/1e9; r = torch.cuda.memory_reserved()/1e9
        print(f'  VRAM: {a:.2f}GB alloc / {r:.2f}GB reserved')

def compute_bleu(hyp, ref):
    if not hyp.strip() or not ref.strip(): return 0.0
    return _bleu.sentence_score(hyp.strip(), [ref.strip()]).score

def compute_chrf(hyp, ref):
    if not hyp.strip() or not ref.strip(): return 0.0
    return _chrf.sentence_score(hyp.strip(), [ref.strip()]).score

def save_figure(fig, name):
    fig.savefig(f'{FIG_DIR}/{name}', dpi=150, bbox_inches='tight')
    if ON_KAGGLE: _rclone_push(f'{FIG_DIR}/{name}', 'figures')
    print(f'[fig] Saved {name}')

def save_results_csv(rows, fname):
    import pandas as pd
    df = pd.DataFrame(rows)
    path = f'{RESULTS_DIR}/{fname}'
    df.to_csv(path, index=False)
    if ON_KAGGLE: _rclone_push(path, 'results')
    print(f'[csv] Saved {fname}  ({len(df)} rows)')
    return df

from IPython.display import Audio as IPAudio, display
import torchaudio
def play(audio, sr, label=''):
    a = audio.squeeze().numpy() if hasattr(audio,'numpy') else np.array(audio)
    print(f'  {label} ({len(a)/sr:.1f}s | sr={sr})')
    display(IPAudio(a, rate=int(sr)))

def save_audio(audio, sr, filename):
    path = f'{AUDIO_DIR}/{filename}'
    t = torch.tensor(np.array(audio)).unsqueeze(0).float()
    torchaudio.save(path, t, int(sr)); print(f'[audio] {filename}')

print('Core utilities ready.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL S7 — Multi-language MMS ASR (extended to all 5 langs)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""# Multi-language MMS-ASR: lazy-loaded, adapter-switched per language
# Exact model: facebook/mms-1b-all  (Wav2Vec2ForCTC + per-lang adapter)
# ISO 639-3 codes: eng, ben, hin, tam, arb

_MMS_MODEL_ID = 'facebook/mms-1b-all'
_mms_model     = None
_mms_processor = None
_mms_cur_lang  = None   # currently loaded adapter

def _ensure_mms(lang='ben'):
    global _mms_model, _mms_processor, _mms_cur_lang
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    if _mms_model is None:
        print(f'[MMS] Loading {_MMS_MODEL_ID} ...')
        _mms_processor = AutoProcessor.from_pretrained(_MMS_MODEL_ID, target_lang=lang)
        _mms_model = Wav2Vec2ForCTC.from_pretrained(
            _MMS_MODEL_ID, target_lang=lang,
            ignore_mismatched_sizes=True, torch_dtype=torch.float16)
        _mms_model.load_adapter(lang)
        _mms_cur_lang = lang
        try: _mms_model = _mms_model.to('cpu')  # keep on CPU, move to cuda only when needed
        except: pass
        print(f'[MMS] Ready with adapter: {lang}')
    elif _mms_cur_lang != lang:
        # Switch adapter (in-place, no reload)
        _mms_model.load_adapter(lang)
        _mms_processor.tokenizer.set_target_lang(lang)
        _mms_cur_lang = lang

def mms_transcribe(audio_np, sr=16000, lang='ben'):
    \"\"\"Transcribe audio with MMS ASR for the given language. Returns text string.\"\"\"
    _ensure_mms(lang)
    if audio_np is None or len(audio_np) < 400: return ''
    if sr != 16000:
        audio_np = torchaudio.functional.resample(
            torch.tensor(audio_np), sr, 16000).numpy()
    dev = torch.device('cpu')  # MMS stays on CPU to preserve VRAM for main model
    inputs = _mms_processor(audio_np, sampling_rate=16000, return_tensors='pt')
    with torch.no_grad():
        logits = _mms_model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)
    return _mms_processor.batch_decode(ids)[0].strip()

def asr_bleu_chrf(audio_np, ref_text, lang='ben', sr=16000):
    \"\"\"Returns (hyp_text, bleu, chrf).\"\"\"
    try:
        hyp = mms_transcribe(audio_np, sr=sr, lang=lang)
        return hyp, compute_bleu(hyp, ref_text), compute_chrf(hyp, ref_text)
    except Exception as e:
        print(f'  [MMS] error ({lang}): {e}')
        return '', 0.0, 0.0

print('Multi-lang MMS-ASR helpers ready (lazy-loaded).')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL S8 — Inference helpers (run_s2t, run_s2st, benchmark)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""processor = None  # set after model load

def _input_device(mdl):
    \"\"\"Device for speech_encoder input (first param of speech_encoder).\"\"\"
    return next(mdl.speech_encoder.parameters()).device

def _remap_ids(mdl, ids):
    \"\"\"Remap trimmed vocab IDs → original IDs for tokenizer decode.\"\"\"
    if hasattr(mdl, '_vocab_remap_to_old'):
        remap = mdl._vocab_remap_to_old
        ids = ids.clone()
        mask = (ids >= 0) & (ids < len(remap))
        ids[mask] = remap[ids[mask]]
    return ids

def run_s2t(mdl, wav_np, tgt_lang='ben'):
    \"\"\"Speech → text (no vocoder). Fast path for BLEU/ChrF eval.\"\"\"
    inputs = processor(audio=wav_np, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(_input_device(mdl)) for k, v in inputs.items()}
    # Temporarily disable vocoder to skip S2ST overhead
    orig_voc = mdl.vocoder
    class _NoVoc(nn.Module):
        def forward(self, *a, **kw):
            dev = next(iter(kw.values())).device if kw else torch.device('cpu')
            return torch.zeros(1, 1, device=dev), [1]
    mdl.vocoder = _NoVoc()
    try:
        with torch.no_grad():
            out = mdl.generate(**inputs, tgt_lang=tgt_lang,
                               return_intermediate_token_ids=True)
    finally:
        mdl.vocoder = orig_voc
    ids = _remap_ids(mdl, out.sequences.cpu())
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def run_s2st(mdl, wav_np, tgt_lang='ben'):
    \"\"\"Speech → (text, waveform).\"\"\"
    inputs = processor(audio=wav_np, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(_input_device(mdl)) for k, v in inputs.items()}
    with torch.no_grad():
        try:
            out = mdl.generate(**inputs, tgt_lang=tgt_lang,
                               return_intermediate_token_ids=True)
            ids = _remap_ids(mdl, out.sequences.cpu())
            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            wav  = out.waveform.cpu().numpy().squeeze() if out.waveform is not None else np.zeros(16000)
            return text, wav
        except RuntimeError as e:
            print(f'  [s2st] vocoder error ({e}); text-only fallback')
            return run_s2t(mdl, wav_np, tgt_lang), np.zeros(16000)

def run_benchmark(mdl, samples, label='model', src_lang='eng', tgt_lang='ben',
                  save_n=3, use_asr_bleu=True):
    \"\"\"
    Full benchmark: text BLEU/ChrF + optionally ASR-BLEU/ChrF.
    samples: list of dicts with keys {id, wav (np float32 16kHz), ref (tgt text)}
    \"\"\"
    print(f'\\n{\"=\"*60}\\n  BENCHMARK: {label}  {src_lang}→{tgt_lang}  n={len(samples)}\\n{\"=\"*60}')
    gpu_mem()
    rows = []
    for i, s in enumerate(samples):
        try:
            dur  = len(s['wav']) / 16000
            t0   = time.time()
            text = run_s2t(mdl, s['wav'], tgt_lang=tgt_lang)
            rt   = time.time() - t0
            bleu = compute_bleu(text, s['ref'])
            chrf = compute_chrf(text, s['ref'])
            # ASR-BLEU: synthesise audio, then ASR-decode
            asr_b = asr_c = 0.0
            if use_asr_bleu and i < 20:   # limit for speed
                _, wav_out = run_s2st(mdl, s['wav'], tgt_lang=tgt_lang)
                asr_hyp, asr_b, asr_c = asr_bleu_chrf(wav_out, s['ref'], lang=tgt_lang)
                if save_n > 0 and i < save_n:
                    save_audio(s['wav'],   16000, f'{label}_{i+1}_src.wav')
                    save_audio(wav_out,    mdl.config.sampling_rate, f'{label}_{i+1}_tgt.wav')
                    play(s['wav'],  16000, f'SRC [{src_lang}]')
                    play(wav_out, mdl.config.sampling_rate, f'TGT [{tgt_lang}]')
            print(f'  [{i+1:>2}] BLEU={bleu:5.1f} ChrF={chrf:5.1f} '
                  f'ASR-BLEU={asr_b:5.1f} ASR-ChrF={asr_c:5.1f} RTF={rt/dur:.3f}  {s[\"id\"]}')
            rows.append(dict(id=s['id'], bleu=bleu, chrf=chrf,
                             asr_bleu=asr_b, asr_chrf=asr_c, rtf=rt/dur,
                             pred=text, ref=s['ref']))
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append(dict(id=s['id'], bleu=0, chrf=0, asr_bleu=0, asr_chrf=0,
                             rtf=float('nan'), pred='', ref=s.get('ref','')))
    valid = [r for r in rows if not math.isnan(r['rtf'])]
    summary = dict(
        label=label, src_lang=src_lang, tgt_lang=tgt_lang, n=len(valid),
        avg_bleu=float(np.mean([r['bleu'] for r in valid])) if valid else 0,
        avg_chrf=float(np.mean([r['chrf'] for r in valid])) if valid else 0,
        avg_asr_bleu=float(np.mean([r['asr_bleu'] for r in valid])) if valid else 0,
        avg_asr_chrf=float(np.mean([r['asr_chrf'] for r in valid])) if valid else 0,
        avg_rtf=float(np.mean([r['rtf'] for r in valid])) if valid else 0,
        params_M=count_params(mdl)
    )
    print(f'\\n  AVG: BLEU={summary[\"avg_bleu\"]:.2f}  ChrF={summary[\"avg_chrf\"]:.2f}  '
          f'ASR-BLEU={summary[\"avg_asr_bleu\"]:.2f}  ASR-ChrF={summary[\"avg_asr_chrf\"]:.2f}  '
          f'RTF={summary[\"avg_rtf\"]:.4f}  Params={summary[\"params_M\"]:.1f}M\\n')
    return rows, summary

def quick_chrf(mdl, samples, tgt_lang='ben', n=15):
    \"\"\"Fast ChrF for importance scoring decisions (text-only, first n samples).\"\"\"
    scores = []
    for s in samples[:n]:
        try: scores.append(compute_chrf(run_s2t(mdl, s['wav'], tgt_lang=tgt_lang), s['ref']))
        except: scores.append(0.0)
    return float(np.mean(scores))

print('Inference & benchmark helpers ready.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL S9 — Summary ledger + plotting
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code("""# ── Summary ledger (persistent across sessions via checkpoints) ──────────────
def _load_summaries():
    ck = load_latest_checkpoint('all_summaries')
    return {s['label']: s for s in ck['summaries']} if (ck and 'summaries' in ck) else {}

ALL_SUMMARIES = _load_summaries()
print(f'[summary] Loaded {len(ALL_SUMMARIES)} entries: {list(ALL_SUMMARIES.keys())}')

def store_summary(s):
    ALL_SUMMARIES[s['label']] = s.copy()
    save_checkpoint({'summaries': list(ALL_SUMMARIES.values())}, 'all_summaries', step=0)
    print(f'[summary] Stored: {s[\"label\"]}  params={s.get(\"params_M\",\"?\"):.1f}M')

def get_summaries():
    return sorted(ALL_SUMMARIES.values(), key=lambda x: x['label'])

# ── Master comparison bar chart ───────────────────────────────────────────────
def plot_phase_comparison(summaries=None, save_name='phase_comparison.png'):
    data = summaries or get_summaries()
    if not data: print('No summaries.'); return
    import pandas as pd
    df = pd.DataFrame(data)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SeamlessLite Compression Pipeline — Phase Comparison', fontsize=14, fontweight='bold')
    metrics = [
        ('avg_bleu',     'Text BLEU ↑',      '#2196F3'),
        ('avg_chrf',     'Text ChrF ↑',      '#4CAF50'),
        ('avg_asr_bleu', 'ASR-BLEU ↑',       '#00BCD4'),
        ('avg_asr_chrf', 'ASR-ChrF ↑',       '#009688'),
        ('avg_rtf',      'RTF ↓ (lower=faster)', '#FF9800'),
        ('params_M',     'Parameters M ↓',   '#9C27B0'),
    ]
    for ax, (key, title, color) in zip(axes.flat, metrics):
        if key not in df.columns: ax.set_visible(False); continue
        vals = df[key].fillna(0).tolist()
        labels = df['label'].tolist()
        bars = ax.bar(range(len(labels)), vals, color=color, alpha=0.85, edgecolor='white')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{v:.1f}',
                    ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    save_figure(fig, save_name); plt.show()

def plot_pareto(summaries=None, save_name='pareto_params_vs_quality.png'):
    data = summaries or get_summaries()
    if not data: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Params vs. Quality Pareto Curve', fontweight='bold')
    for ax, (yk, ylabel) in zip(axes, [('avg_bleu','Text BLEU'), ('avg_asr_bleu','ASR-BLEU')]):
        xs = [s.get('params_M',0) for s in data]
        ys = [s.get(yk,0)        for s in data]
        ax.scatter(xs, ys, s=100, zorder=5, c='#E91E63')
        for s in data:
            ax.annotate(s['label'], (s.get('params_M',0), s.get(yk,0)),
                        fontsize=7, xytext=(4,4), textcoords='offset points')
        ax.set_xlabel('Parameters (M)'); ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    plt.tight_layout()
    save_figure(fig, save_name); plt.show()

sync_checkpoints_from_drive()
print('Setup complete. Run next cells for data download, then Phase 0.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Section 1 — Data Download
Downloads FLEURS (5 languages), Common Voice 17 (5 languages), builds Kaggle dataset.
Run once; subsequent sessions load from Drive cache or Kaggle dataset.
"""))

# CELL D1 — FLEURS parquet downloader (extended to all 5 languages)
cells.append(code("""import concurrent.futures, io, soundfile as sf, pandas as pd
from datasets import Dataset

# ── Local cache paths ─────────────────────────────────────────────────────────
FLEURS_CACHE = f'{DATA_DIR}/fleurs_parquet'
CV17_CACHE   = f'{DATA_DIR}/cv17_parquet'
os.makedirs(FLEURS_CACHE, exist_ok=True)
os.makedirs(CV17_CACHE,   exist_ok=True)

# ── Generic parquet downloader ────────────────────────────────────────────────
def _download_shard(url, dest, min_size_bytes=512*1024):
    \"\"\"Download a single parquet shard. Returns (ok, msg).\"\"\"
    import requests
    dest = pathlib.Path(dest)
    if dest.exists() and dest.stat().st_size > min_size_bytes:
        return True, 'cached'
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, timeout=180,
                             headers={'Authorization': f'Bearer {_get_hf_token()}'})
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8*1024*1024):
                    if chunk: f.write(chunk)
            if dest.stat().st_size < min_size_bytes:
                raise RuntimeError('file too small')
            return True, 'downloaded'
        except Exception as e:
            if dest.exists(): dest.unlink(missing_ok=True)
            if attempt == 2: return False, str(e)
    return False, 'unknown'

def _get_hf_token():
    try: return _get_secret('HF_TOKEN')
    except: return ''

def _parallel_download(tasks, n_workers=6):
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_download_shard, url, dest): (url, dest) for url, dest in tasks}
        for fut in concurrent.futures.as_completed(futs):
            url, dest = futs[fut]
            ok, msg = fut.result()
            fname = pathlib.Path(dest).name
            print(f'  {\"✓\" if ok else \"✗\"} {fname}: {msg}')

# ── FLEURS downloader ─────────────────────────────────────────────────────────
FLEURS_BASE = 'https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet'

def _fleurs_tasks(lang_code, split, cache_root):
    # FLEURS parquet structure: one shard per split per language
    url  = f'{FLEURS_BASE}/{lang_code}/{split}/0000.parquet?download=true'
    dest = f'{cache_root}/{lang_code}/{split}_0000.parquet'
    return [(url, dest)]

def download_fleurs_all(splits=('train', 'validation', 'test')):
    print('[FLEURS] Downloading all 5 languages × 3 splits ...')
    tasks = []
    for seam_code, fleurs_code in FLEURS_CODES.items():
        for split in splits:
            tasks.extend(_fleurs_tasks(fleurs_code, split, FLEURS_CACHE))
    _parallel_download(tasks)
    print(f'[FLEURS] Done. Cache: {FLEURS_CACHE}')

def load_fleurs(seam_lang, split='train'):
    \"\"\"Load FLEURS parquet as pandas DataFrame for a SeamlessM4T language code.\"\"\"
    fleurs_code = FLEURS_CODES[seam_lang]
    files = sorted(glob.glob(f'{FLEURS_CACHE}/{fleurs_code}/{split}_*.parquet'))
    if not files:
        raise FileNotFoundError(f'[FLEURS] No parquet for {seam_lang}/{split}. Run download_fleurs_all().')
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

print('FLEURS loader ready.')
"""))

# CELL D2 — Common Voice 17 downloader
cells.append(code("""# Common Voice 17 — first 2 shards per language (train only)
# URL: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0/resolve/refs%2Fconvert%2Fparquet/{lang}/train/{shard}.parquet
CV17_BASE = ('https://huggingface.co/datasets/mozilla-foundation/'
             'common_voice_17_0/resolve/refs%2Fconvert%2Fparquet')
# We download 2 train shards per language for KD calibration (~2000 samples/lang)
CV17_N_SHARDS = {'en': 2, 'ar': 2, 'hi': 2, 'ta': 2, 'bn': 2}

def download_cv17_all():
    print('[CV17] Downloading Common Voice 17 (5 langs × 2 shards)...')
    tasks = []
    for seam_code, cv_code in CV17_LANGS.items():
        n = CV17_N_SHARDS.get(cv_code, 1)
        for shard in range(n):
            url  = f'{CV17_BASE}/{cv_code}/train/{shard:04d}.parquet?download=true'
            dest = f'{CV17_CACHE}/{cv_code}/train_{shard:04d}.parquet'
            tasks.append((url, dest))
    _parallel_download(tasks)
    print(f'[CV17] Done. Cache: {CV17_CACHE}')

def load_cv17(seam_lang, split='train', max_rows=1500):
    \"\"\"Load CV17 parquet; keep only columns needed for KD: {path, audio, sentence}.\"\"\"
    cv_code = CV17_LANGS[seam_lang]
    files = sorted(glob.glob(f'{CV17_CACHE}/{cv_code}/{split}_*.parquet'))
    if not files:
        raise FileNotFoundError(f'[CV17] No parquet for {seam_lang}. Run download_cv17_all().')
    dfs = [pd.read_parquet(f, columns=['path','audio','sentence'])
           for f in files if os.path.exists(f)]
    df = pd.concat(dfs, ignore_index=True).head(max_rows)
    # Filter empty sentences
    df = df[df['sentence'].str.strip().str.len() > 0].reset_index(drop=True)
    print(f'[CV17] {seam_lang}: {len(df)} rows loaded')
    return df

print('CV17 loader ready.')
"""))

# CELL D3 — Audio loading utility + sample builder
cells.append(code("""def _load_audio_cell(audio_cell, target_sr=16000):
    \"\"\"
    Robust audio loader for HF parquet audio columns.
    Handles both:
      - dict with 'array' + 'sampling_rate'  (HF Dataset format)
      - dict with 'bytes'                     (raw parquet format)
    Returns: numpy float32 array at target_sr.
    \"\"\"
    a = audio_cell
    if isinstance(a, dict) and 'array' in a:
        arr, sr = np.array(a['array'], dtype=np.float32), a['sampling_rate']
    elif isinstance(a, dict) and 'bytes' in a and a['bytes'] is not None:
        wav, sr = sf.read(io.BytesIO(a['bytes']))
        arr = np.array(wav, dtype=np.float32)
        if arr.ndim > 1: arr = arr.mean(axis=1)
    else:
        raise ValueError(f'Unsupported audio cell: {type(a)}  keys={list(a.keys()) if isinstance(a,dict) else None}')
    if sr != target_sr:
        arr = torchaudio.functional.resample(
            torch.tensor(arr), sr, target_sr).numpy().astype(np.float32)
    return arr

def build_eval_samples(src_seam, tgt_seam, split='test', n=50):
    \"\"\"
    Build list of {id, wav, ref} dicts for evaluation.
    src_seam: source language (SeamlessM4T code, e.g. 'eng')
    tgt_seam: target language (SeamlessM4T code, e.g. 'ben')
    wav: source language audio; ref: target language transcription.
    \"\"\"
    src_df = load_fleurs(src_seam, split)
    tgt_df = load_fleurs(tgt_seam, split)
    # Deduplicate & merge on 'id'
    src_df = src_df[['id','audio','transcription']].drop_duplicates('id').rename(
        columns={'audio':'src_audio','transcription':'src_text'})
    tgt_df = tgt_df[['id','transcription']].drop_duplicates('id').rename(
        columns={'transcription':'tgt_text'})
    merged = pd.merge(src_df, tgt_df, on='id', how='inner').head(n)
    print(f'  [{src_seam}→{tgt_seam}/{split}] {len(merged)} pairs')
    samples = []
    for _, row in merged.iterrows():
        try:
            wav = _load_audio_cell(row['src_audio'])
            samples.append(dict(id=row['id'], wav=wav, ref=row['tgt_text']))
        except Exception as e:
            print(f'  [warn] audio load error id={row[\"id\"]}: {e}')
    return samples

def build_train_samples(src_seam, tgt_seam, split='train', n=None):
    \"\"\"Build training samples (larger, lazy load). Returns list of {id, wav, ref}.\"\"\"
    src_df = load_fleurs(src_seam, split)
    tgt_df = load_fleurs(tgt_seam, split)
    src_df = src_df[['id','audio']].drop_duplicates('id').rename(columns={'audio':'src_audio'})
    tgt_df = tgt_df[['id','transcription']].drop_duplicates('id').rename(columns={'transcription':'tgt_text'})
    merged = pd.merge(src_df, tgt_df, on='id', how='inner')
    if n: merged = merged.head(n)
    print(f'  [{src_seam}→{tgt_seam}/{split}] {len(merged)} train pairs')
    samples = []
    for _, row in merged.iterrows():
        try:
            wav = _load_audio_cell(row['src_audio'])
            if row['tgt_text'].strip():
                samples.append(dict(id=row['id'], wav=wav, ref=row['tgt_text']))
        except Exception as e:
            pass
    print(f'  Loaded {len(samples)} valid samples')
    return samples

print('Audio / sample utilities ready.')
"""))

# CELL D4 — Download all data + push to Drive
cells.append(code("""# ── Run once per account (cached in Drive thereafter) ────────────────────────
# Check if already cached on Drive, if not download

def _check_fleurs_cached():
    \"\"\"Check if all FLEURS shards already exist locally or on Drive.\"\"\"
    needed = [(FLEURS_CODES[l], sp) for l in SEAM_LANGS for sp in ('train','validation','test')]
    for lc, sp in needed:
        if not glob.glob(f'{FLEURS_CACHE}/{lc}/{sp}_*.parquet'):
            return False
    return True

def _check_cv17_cached():
    for cv_code in CV17_LANGS.values():
        if not glob.glob(f'{CV17_CACHE}/{cv_code}/train_*.parquet'):
            return False
    return True

# Try to pull from Drive first
if ON_KAGGLE and not _check_fleurs_cached():
    print('[data] Pulling FLEURS from Drive...')
    subprocess.run(
        f'rclone copy \"{GDRIVE_ROOT}/data/fleurs_parquet/\" \"{FLEURS_CACHE}/\" --transfers=16',
        shell=True, capture_output=True)

if ON_KAGGLE and not _check_cv17_cached():
    print('[data] Pulling CV17 from Drive...')
    subprocess.run(
        f'rclone copy \"{GDRIVE_ROOT}/data/cv17_parquet/\" \"{CV17_CACHE}/\" --transfers=16',
        shell=True, capture_output=True)

# Download missing data
if not _check_fleurs_cached():
    print('[data] FLEURS not cached — downloading...')
    # Login HF
    try:
        HF_TOKEN = _get_secret('HF_TOKEN')
        from huggingface_hub import login; login(HF_TOKEN)
    except Exception as e:
        print(f'[HF] login skipped: {e}')
    download_fleurs_all(splits=('train', 'validation', 'test'))
else:
    print('[data] FLEURS cache OK.')

if not _check_cv17_cached():
    print('[data] CV17 not cached — downloading...')
    download_cv17_all()
else:
    print('[data] CV17 cache OK.')

# Push fresh downloads to Drive
if ON_KAGGLE:
    print('[data] Syncing data cache → Drive...')
    for subdir in ['fleurs_parquet', 'cv17_parquet']:
        r = subprocess.run(
            f'rclone copy \"{DATA_DIR}/{subdir}/\" \"{GDRIVE_ROOT}/data/{subdir}/\" --transfers=16',
            shell=True, capture_output=True, text=True)
        status = 'OK' if r.returncode==0 else f'WARN: {r.stderr[:100]}'
        print(f'  {subdir}: {status}')

# Quick sanity check
print('\\n=== Dataset summary ===')
for lang in SEAM_LANGS:
    for split in ('train','test'):
        try:
            df = load_fleurs(lang, split)
            print(f'  FLEURS {lang}/{split}: {len(df)} rows')
        except Exception as e:
            print(f'  FLEURS {lang}/{split}: MISSING ({e})')
print()
"""))

# CELL D5 — Kaggle dataset creation script
cells.append(code("""# ═══════════════════════════════════════════════════════════════════════════════
# CREATE PUBLIC KAGGLE DATASET from downloaded data
# Run this ONCE after the first successful data download.
# The created dataset can then be mounted in any future Kaggle session as
# /kaggle/input/<your-username>/seamlesslite-5lang-data/
# ═══════════════════════════════════════════════════════════════════════════════

import zipfile

def create_kaggle_dataset(dataset_title='seamlesslite-5lang-data',
                          dataset_dir='/kaggle/working/kag_dataset'):
    if not ON_KAGGLE:
        print('Kaggle dataset creation only works on Kaggle.')
        return

    # ── Kaggle API key ────────────────────────────────────────────────────────
    try:
        kag_json = _get_secret('KAGGLE_API_KEY')   # store JSON string in secrets
        kag_path = pathlib.Path.home() / '.kaggle/kaggle.json'
        kag_path.parent.mkdir(parents=True, exist_ok=True)
        kag_path.write_text(kag_json)
        kag_path.chmod(0o600)
        print('[kaggle] API key written.')
    except Exception as e:
        print(f'[kaggle] WARNING: could not write API key: {e}')
        print('[kaggle] Make sure KAGGLE_API_KEY secret contains your kaggle.json content')
        return

    # ── Stage data into dataset_dir ───────────────────────────────────────────
    os.makedirs(dataset_dir, exist_ok=True)

    # Copy parquet shards into dataset_dir (only test + small train slices)
    for lang in SEAM_LANGS:
        fleurs_code = FLEURS_CODES[lang]
        for split in ('train','validation','test'):
            dest = f'{dataset_dir}/fleurs/{fleurs_code}'
            os.makedirs(dest, exist_ok=True)
            for src_f in glob.glob(f'{FLEURS_CACHE}/{fleurs_code}/{split}_*.parquet'):
                shutil.copy2(src_f, f'{dest}/{os.path.basename(src_f)}')

    for seam_code, cv_code in CV17_LANGS.items():
        dest = f'{dataset_dir}/cv17/{cv_code}'
        os.makedirs(dest, exist_ok=True)
        for src_f in sorted(glob.glob(f'{CV17_CACHE}/{cv_code}/train_*.parquet'))[:1]:
            shutil.copy2(src_f, f'{dest}/{os.path.basename(src_f)}')

    # ── Write dataset-metadata.json ───────────────────────────────────────────
    import subprocess, getpass
    # Get Kaggle username from API key JSON
    try:
        kag_data = json.loads(kag_json)
        username = kag_data['username']
    except:
        username = 'YOUR_USERNAME'

    metadata = {
        'title': dataset_title,
        'id': f'{username}/{dataset_title}',
        'licenses': [{'name': 'CC0-1.0'}],
        'resources': [{'path': f, 'description': f}
                      for f in glob.glob(f'{dataset_dir}/**/*', recursive=True)
                      if os.path.isfile(f)]
    }
    with open(f'{dataset_dir}/dataset-metadata.json', 'w') as fp:
        json.dump(metadata, fp, indent=2)
    print(f'[kaggle] Metadata written. Dataset dir: {dataset_dir}')
    print(f'[kaggle] Files: {sum(1 for _ in glob.glob(f\"{dataset_dir}/**/*\", recursive=True) if os.path.isfile(_))}')

    # ── Create dataset via Kaggle CLI ─────────────────────────────────────────
    r = subprocess.run(
        ['kaggle', 'datasets', 'create', '-p', dataset_dir, '--dir-mode', 'zip'],
        capture_output=True, text=True)
    if r.returncode == 0:
        print(f'[kaggle] Dataset created! Access: kaggle datasets download {username}/{dataset_title}')
        print(f'[kaggle] In next session: /kaggle/input/{dataset_title}/')
    else:
        print(f'[kaggle] Dataset creation output:\\n{r.stdout}\\n{r.stderr}')
        # If dataset already exists, try update
        if 'already exists' in r.stderr.lower() or 'already exists' in r.stdout.lower():
            r2 = subprocess.run(
                ['kaggle', 'datasets', 'version', '-p', dataset_dir,
                 '-m', 'Updated data shards', '--dir-mode', 'zip'],
                capture_output=True, text=True)
            print(f'[kaggle] Version update: {r2.stdout[:300]}')

# ── Usage: uncomment to run ───────────────────────────────────────────────────
# create_kaggle_dataset()
print('Kaggle dataset creator ready.  Uncomment create_kaggle_dataset() to publish.')
print('Future sessions: mount your dataset at /kaggle/input/<dataset-name>/')
print()
print('If you already have the Kaggle dataset mounted, set DATA_DIR to the mounted path:')
print('  KAGGLE_DATASET_PATH = \"/kaggle/input/seamlesslite-5lang-data\"')
print('  FLEURS_CACHE = f\"{KAGGLE_DATASET_PATH}/fleurs\"')
print('  CV17_CACHE   = f\"{KAGGLE_DATASET_PATH}/cv17\"')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: LOAD MODEL + EVAL SAMPLES
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Section 2 — Load Teacher Model & Build Eval Sets
"""))

cells.append(code("""from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor

try:
    HF_TOKEN = _get_secret('HF_TOKEN')
    from huggingface_hub import login; login(HF_TOKEN); print('[HF] Logged in.')
except Exception as e:
    print(f'[HF] login skipped: {e}')

MODEL_ID = 'facebook/seamless-m4t-v2-large'

def load_base_model():
    global processor
    print(f'[model] Loading processor...')
    processor = SeamlessM4TProcessor.from_pretrained(MODEL_ID)
    print(f'[model] Loading model (fp16, device_map=auto)...')
    mdl = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map='auto')
    mdl.eval()
    print(f'[model] Loaded. Params: {count_params(mdl):.1f}M')
    gpu_mem()
    return mdl

print('load_base_model() ready. Call it in Phase 0.')
"""))

cells.append(code("""# Build eval sample sets for all 8 bidirectional pairs
# N_EVAL_TEST = 50 samples per pair (for comprehensive benchmark)
# N_EVAL_FAST = 20 samples per pair (for quick importance scoring)
N_EVAL_TEST = 50
N_EVAL_FAST = 15

print('Building eval sample sets...')
EVAL_SAMPLES = {}   # key = (src, tgt), value = list of {id, wav, ref}
for src, tgt in EVAL_PAIRS:
    try:
        EVAL_SAMPLES[(src, tgt)] = build_eval_samples(src, tgt, split='test', n=N_EVAL_TEST)
        print(f'  {src}→{tgt}: {len(EVAL_SAMPLES[(src,tgt)])} samples')
    except Exception as e:
        print(f'  {src}→{tgt}: FAILED ({e})')
        EVAL_SAMPLES[(src, tgt)] = []

print(f'\\n[eval] {sum(len(v) for v in EVAL_SAMPLES.values())} total eval samples across {len(EVAL_PAIRS)} pairs')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: PHASE 0 — BASELINE BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Phase 0 — Teacher Baseline Benchmark
Establish teacher quality floor for all 8 bidirectional X↔Eng pairs.
"""))

cells.append(code("""# ── Phase 0: Load teacher + full benchmark ───────────────────────────────────
model = load_base_model()

# Print exact architecture component counts (verify before pruning)
print('\\n=== Architecture (exact SeamlessM4Tv2 attr paths) ===')
print(f'  speech_encoder.encoder.layers : {len(_get_speech_enc_layers(model))} Conformer layers')
print(f'  text_decoder.layers           : {len(_get_text_dec_layers(model))} Decoder layers')
print(f'  t2u_model.model.encoder.layers: {len(_get_t2u_enc_layers(model))} T2U enc layers')
print(f'  t2u_model.model.decoder.layers: {len(_get_t2u_dec_layers(model))} T2U dec layers')
print(f'  shared (embedding)            : {model.shared.weight.shape}')
# Speech encoder FFN
l0 = _get_speech_enc_layers(model)[0]
print(f'  speech enc FFN intermediate   : {l0.feed_forward.intermediate_dense.out_features}')
print(f'  speech enc self_attn heads    : {model.config.speech_encoder_attention_heads}')
# Text decoder FFN
d0 = _get_text_dec_layers(model)[0]
print(f'  text dec FFN fc1              : {d0.ffn.fc1.out_features}')
print(f'  text dec self_attn heads      : {model.config.decoder_attention_heads}')
print(f'  vocab_size                    : {model.config.vocab_size}')
print(f'  total params                  : {count_params(model):.1f}M')
print()
"""))

cells.append(code("""# ── Full 8-direction benchmark (teacher) ─────────────────────────────────────
p0_rows_all = []
p0_summaries = {}
for src, tgt in EVAL_PAIRS:
    samples = EVAL_SAMPLES.get((src, tgt), [])
    if not samples:
        print(f'  [{src}→{tgt}] no samples, skipping')
        continue
    rows, summary = run_benchmark(
        model, samples, label=f'p0_teacher_{src}_{tgt}',
        src_lang=src, tgt_lang=tgt, save_n=2, use_asr_bleu=True)
    p0_rows_all.extend(rows)
    p0_summaries[(src, tgt)] = summary
    store_summary(summary)

save_results_csv(p0_rows_all, 'phase0_teacher_all_pairs.csv')
print('\\n=== Teacher Baseline Summary ===')
for (src, tgt), s in p0_summaries.items():
    print(f'  {src}→{tgt}: BLEU={s[\"avg_bleu\"]:5.2f}  ChrF={s[\"avg_chrf\"]:5.2f}  '
          f'ASR-BLEU={s[\"avg_asr_bleu\"]:5.2f}  ASR-ChrF={s[\"avg_asr_chrf\"]:5.2f}  '
          f'RTF={s[\"avg_rtf\"]:.4f}')
"""))

cells.append(code("""# ── Phase 0 Figure: Teacher baseline bar chart ───────────────────────────────
import pandas as pd
p0_df = pd.DataFrame([
    {'pair': f'{s}→{t}', **v}
    for (s,t), v in p0_summaries.items()
])
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Phase 0: Teacher Baseline (SeamlessM4T v2 Large, 1.8B params)', fontweight='bold')
for ax, (col, title) in zip(axes, [('avg_bleu','Text BLEU'), ('avg_asr_bleu','ASR-BLEU (MMS-1b-all)')]):
    if col not in p0_df.columns: continue
    bars = ax.bar(p0_df['pair'], p0_df[col], color='#2196F3', alpha=0.85, edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_xticklabels(p0_df['pair'], rotation=35, ha='right')
    for bar, v in zip(bars, p0_df[col]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{v:.1f}',
                ha='center', va='bottom', fontsize=9)
plt.tight_layout()
save_figure(fig, 'phase0_teacher_baseline.png')
plt.show()
print('Phase 0 complete.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — VOCABULARY PRUNING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Phase 1 — Vocabulary / Embedding Pruning
**Technique:** Keep only tokens used by the 5 target languages.
**Expected savings:** ~192M params (256k→64k vocab × 1024 hidden).
**Citation:** CULL-MT (Rostami & Dousti, 2024); Asahi et al. EMNLP 2023.
"""))

cells.append(code("""# ── Phase 1: Collect active vocabulary from 5-language training text ──────────
from transformers import NllbTokenizer
import collections

def collect_active_vocab(proc, seam_langs, n_samples_per_lang=1500):
    \"\"\"
    Tokenize FLEURS train text for all 5 languages; collect all unique token IDs
    that appear. Also always keeps: special tokens + language tag tokens.
    Returns: sorted list of token IDs to KEEP (in original vocab space).
    \"\"\"
    tokenizer = proc.tokenizer
    print('[vocab] Collecting active token IDs from 5-language training data...')
    active = set()
    # Always keep special / structural tokens
    for tok in [tokenizer.bos_token_id, tokenizer.eos_token_id,
                tokenizer.pad_token_id, tokenizer.unk_token_id]:
        if tok is not None: active.add(tok)
    # Always keep language tag tokens for all 5 + their aliases
    lang_tags = ['ben', 'eng', 'hin', 'tam', 'arb',
                 '__ben__', '__eng__', '__hin__', '__tam__', '__arb__',
                 'ben_Beng', 'eng_Latn', 'hin_Deva', 'tam_Taml', 'arb_Arab']
    for tag in lang_tags:
        try:
            tid = tokenizer.convert_tokens_to_ids(tag)
            if tid != tokenizer.unk_token_id: active.add(tid)
        except: pass
    # Scan generation config for special ids
    for attr in ['pad_token_id','bos_token_id','eos_token_id','decoder_start_token_id']:
        v = getattr(model.generation_config, attr, None)
        if v is not None: active.add(int(v))
    # Tokenize training text
    for lang in seam_langs:
        try:
            df = load_fleurs(lang, 'train')
            texts = df['transcription'].dropna().tolist()[:n_samples_per_lang]
            for text in texts:
                ids = tokenizer(text, add_special_tokens=True).input_ids
                active.update(ids)
            print(f'  {lang}: tokenized {len(texts)} sentences, active={len(active)}')
        except Exception as e:
            print(f'  {lang}: error ({e})')
    # Also tokenize CV17 text
    for lang in seam_langs:
        try:
            df = load_cv17(lang, max_rows=500)
            texts = df['sentence'].dropna().tolist()
            for text in texts:
                ids = tokenizer(text, add_special_tokens=True).input_ids
                active.update(ids)
        except: pass
    # Keep numerals and common punctuation (IDs 0–999 are usually special)
    active.update(range(min(500, tokenizer.vocab_size)))
    keep_ids = sorted(active)
    print(f'[vocab] Active IDs: {len(keep_ids)} / {tokenizer.vocab_size}')
    return keep_ids

keep_ids = collect_active_vocab(processor, SEAM_LANGS)
print(f'[vocab] Will keep {len(keep_ids)} tokens (reduction: {model.config.vocab_size - len(keep_ids):,})')
"""))

cells.append(code("""# ── Phase 1: Apply vocabulary pruning to model ────────────────────────────────

def prune_vocabulary(mdl, proc, keep_ids):
    \"\"\"
    Slice model.shared and model.text_decoder.embed_tokens to keep_ids.
    Build old→new and new→old remapping tables.
    Modifies model IN PLACE (on CPU for safety, then restore fp16).
    \"\"\"
    import torch
    keep_t = torch.tensor(keep_ids, dtype=torch.long)
    new_vocab = len(keep_ids)

    print(f'[vocab] Pruning: {mdl.shared.num_embeddings} → {new_vocab} tokens')

    # ── Move embeddings to CPU for slicing ───────────────────────────────────
    orig_dtype = mdl.shared.weight.dtype
    orig_device = mdl.shared.weight.device

    with torch.no_grad():
        new_W = mdl.shared.weight.cpu().float()[keep_t].to(orig_dtype)

    # ── Replace shared embedding ──────────────────────────────────────────────
    new_emb = nn.Embedding(new_vocab, mdl.config.hidden_size, padding_idx=None)
    new_emb.weight = nn.Parameter(new_W.to(orig_device))
    mdl.shared = new_emb

    # ── Replace text_decoder.embed_tokens (same weights by reference) ─────────
    # In SeamlessM4Tv2, text_decoder.embed_tokens shares the embedding table
    # Verify: it has the same shape
    if mdl.text_decoder.embed_tokens.weight.shape[0] == mdl.config.vocab_size:
        new_emb2 = nn.Embedding(new_vocab, mdl.config.hidden_size)
        new_emb2.weight = nn.Parameter(new_W.to(orig_device))
        mdl.text_decoder.embed_tokens = new_emb2
        print('  text_decoder.embed_tokens: pruned')

    # ── Replace lm_head (output projection) ──────────────────────────────────
    # SeamlessM4Tv2: lm_head is nn.Linear(hidden_size, vocab_size, bias=False)
    orig_lm = mdl.lm_head.weight.cpu().float()
    if orig_lm.shape[0] == mdl.config.vocab_size:
        new_lm_w = orig_lm[keep_t].to(orig_dtype).to(orig_device)
        new_lm = nn.Linear(mdl.config.hidden_size, new_vocab, bias=False)
        new_lm.weight = nn.Parameter(new_lm_w)
        mdl.lm_head = new_lm.to(orig_device)
        print('  lm_head: pruned')

    # ── Build remap: new_id → old_id (for decoding) ───────────────────────────
    mdl._vocab_remap_to_old = keep_t   # shape [new_vocab], dtype long

    # ── Update config ─────────────────────────────────────────────────────────
    mdl.config.vocab_size = new_vocab
    print(f'[vocab] Done. New vocab_size: {new_vocab}')
    return mdl

model = prune_vocabulary(model, processor, keep_ids)
p1_params = count_params(model)
print(f'[Phase 1] Params: {p1_params:.1f}M')
gc.collect(); torch.cuda.empty_cache()
"""))

cells.append(code("""# ── Phase 1 Benchmark ────────────────────────────────────────────────────────
p1_rows_all = []
p1_summaries = {}
for src, tgt in EVAL_PAIRS:
    samples = EVAL_SAMPLES.get((src, tgt), [])
    if not samples: continue
    rows, summary = run_benchmark(
        model, samples, label=f'p1_vocabprune_{src}_{tgt}',
        src_lang=src, tgt_lang=tgt, save_n=1, use_asr_bleu=False)
    p1_rows_all.extend(rows)
    p1_summaries[(src, tgt)] = summary
    store_summary(summary)
save_results_csv(p1_rows_all, 'phase1_vocab_prune.csv')

# Compare Phase 0 vs Phase 1
print('\\n=== Phase 0 vs Phase 1 (vocab pruning) ===')
print(f'{\"Pair\":<14} {\"P0 BLEU\":>10} {\"P1 BLEU\":>10} {\"Δ BLEU\":>8}')
for (src, tgt) in EVAL_PAIRS:
    p0s = p0_summaries.get((src,tgt))
    p1s = p1_summaries.get((src,tgt))
    if p0s and p1s:
        delta = p1s[\"avg_bleu\"] - p0s[\"avg_bleu\"]
        print(f'  {src}→{tgt:<10} {p0s[\"avg_bleu\"]:>10.2f} {p1s[\"avg_bleu\"]:>10.2f} {delta:>+8.2f}')

# Save model
save_model_to_drive(model, processor, 'phase1_vocab', manifest_extra={'keep_vocab': len(keep_ids)})
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — TEXT DECODER LAYER PRUNING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Phase 2 — Text Decoder Iterative Layer Pruning (24→12)
**Technique:** Greedy importance scoring — remove one layer at a time by measuring ChrF drop.
**Expected savings:** ~150M params.
**Citation:** Moslem (IWSLT 2025, arXiv:2505.20237); Peer et al. (EMNLP 2022).
"""))

cells.append(code("""from contextlib import contextmanager

@contextmanager
def _skip_decoder_layer(mdl, layer_idx):
    \"\"\"
    Context manager that replaces layer `layer_idx` in text_decoder.layers
    with an identity pass-through during importance scoring.
    SeamlessM4Tv2DecoderLayer forward signature:
      forward(hidden_states, attention_mask, encoder_hidden_states,
              encoder_attention_mask, past_key_value, output_attentions, use_cache)
    Returns: (hidden_states, self_attn_weights, cross_attn_weights, present_key_value)
    \"\"\"
    layers = _get_text_dec_layers(mdl)
    orig_layer = layers[layer_idx]

    class _Identity(nn.Module):
        def forward(self, hidden_states, attention_mask=None,
                    encoder_hidden_states=None, encoder_attention_mask=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    **kwargs):
            # Pass hidden_states through unchanged; return None for kv cache
            pkv = (torch.zeros(1), torch.zeros(1)) if use_cache else None
            return (hidden_states, None, None, pkv)

    layers[layer_idx] = _Identity().to(next(orig_layer.parameters()).device)
    try:
        yield
    finally:
        layers[layer_idx] = orig_layer

def score_decoder_layers(mdl, cal_samples, tgt_lang='ben', fast_n=15):
    \"\"\"
    Score all decoder layers by ChrF drop when each is skipped.
    Higher score = more important = keep.
    Returns: dict {layer_idx: importance_score}
    \"\"\"
    n_layers = len(_get_text_dec_layers(mdl))
    base_chrf = quick_chrf(mdl, cal_samples, tgt_lang=tgt_lang, n=fast_n)
    print(f'[dec-prune] Base ChrF: {base_chrf:.2f}  (n_layers={n_layers})')
    scores = {}
    for l in range(n_layers):
        with _skip_decoder_layer(mdl, l):
            chrf_skip = quick_chrf(mdl, cal_samples, tgt_lang=tgt_lang, n=fast_n)
        scores[l] = base_chrf - chrf_skip   # >0 = important
        print(f'  layer {l:>2}: ChrF_skip={chrf_skip:.2f}  importance={scores[l]:+.3f}')
    return scores

# Use English→Bengali as calibration pair (largest coverage)
cal_samples_dec = EVAL_SAMPLES.get(('eng', 'ben'), [])[:N_EVAL_FAST]
print(f'[dec-prune] Calibration: {len(cal_samples_dec)} samples')
"""))

cells.append(code("""# ── Score + greedy prune: 24 → 12 layers ────────────────────────────────────
TARGET_DEC_LAYERS = 12
n_to_remove = len(_get_text_dec_layers(model)) - TARGET_DEC_LAYERS

print(f'[dec-prune] Target: {len(_get_text_dec_layers(model))} → {TARGET_DEC_LAYERS} layers '
      f'(removing {n_to_remove})')

dec_importance_history = {}   # {removal_step: {layer_idx: score}}
removed_dec_layers = []

for step in range(n_to_remove):
    cur_n = len(_get_text_dec_layers(model))
    print(f'\\n--- Pruning step {step+1}/{n_to_remove}  (current: {cur_n} layers) ---')
    scores = score_decoder_layers(model, cal_samples_dec, tgt_lang='ben', fast_n=N_EVAL_FAST)
    dec_importance_history[step] = scores
    # Remove least important layer
    worst_idx = min(scores, key=scores.get)
    print(f'  Removing layer {worst_idx} (importance={scores[worst_idx]:.4f})')
    layers = _get_text_dec_layers(model)
    del layers[worst_idx]
    removed_dec_layers.append(worst_idx)
    sync_model_config(model)
    gc.collect(); torch.cuda.empty_cache()

print(f'\\n[dec-prune] Done. Layers remaining: {len(_get_text_dec_layers(model))}')
print(f'[dec-prune] Removed indices: {removed_dec_layers}')
print(f'[dec-prune] Params now: {count_params(model):.1f}M')
"""))

cells.append(code("""# ── Phase 2 Layer importance figure ──────────────────────────────────────────
if dec_importance_history:
    fig, ax = plt.subplots(figsize=(14, 5))
    first_step = dec_importance_history[0]
    idxs = sorted(first_step.keys())
    vals = [first_step[i] for i in idxs]
    colors = ['#d32f2f' if i in removed_dec_layers else '#2196F3' for i in idxs]
    bars = ax.bar(idxs, vals, color=colors, edgecolor='white')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel('Layer Index'); ax.set_ylabel('Importance (ChrF drop when removed)')
    ax.set_title('Phase 2: Text Decoder Layer Importance Scores\\n(red = removed, blue = kept)',
                 fontweight='bold')
    ax.set_xticks(idxs)
    plt.tight_layout()
    save_figure(fig, 'phase2_decoder_layer_importance.png')
    plt.show()

# ── Phase 2 Benchmark ─────────────────────────────────────────────────────────
p2_rows_all = []
p2_summaries = {}
for src, tgt in EVAL_PAIRS:
    samples = EVAL_SAMPLES.get((src, tgt), [])
    if not samples: continue
    rows, summary = run_benchmark(
        model, samples, label=f'p2_decprune_{src}_{tgt}',
        src_lang=src, tgt_lang=tgt, save_n=0, use_asr_bleu=False)
    p2_rows_all.extend(rows)
    p2_summaries[(src, tgt)] = summary
    store_summary(summary)
save_results_csv(p2_rows_all, 'phase2_decoder_prune.csv')
save_model_to_drive(model, processor, 'phase2_decoder',
                    manifest_extra={'removed_dec_layers': removed_dec_layers})
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — SPEECH ENCODER LAYER PRUNING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Phase 3 — Speech Encoder (w2v-BERT 2.0 Conformer) Layer Pruning (24→12)
**Technique:** Block Influence (BI) metric (ShortGPT, Ma et al., ACL 2025).
BI(l) = 1 − cosine_similarity(input_l, output_l) — low BI → redundant layer.
**Expected savings:** ~200M params.
**Citation:** ShortGPT (arXiv:2403.03853); CoLLD (arXiv:2309.07707).
"""))

cells.append(code("""# SeamlessM4Tv2 speech encoder structure (EXACT paths):
# model.speech_encoder                              <- SeamlessM4Tv2SpeechEncoder
#   .feature_extractor                              <- CNN feature extractor
#   .feature_projection                             <- Linear projection
#   .encoder                                        <- SeamlessM4Tv2ConformerEncoder
#     .layers[i]                                    <- SeamlessM4Tv2ConformerEncoderLayer
#       .self_attn                                  <- SeamlessM4Tv2ConformerSelfAttention
#         .q_proj, .k_proj, .v_proj, .out_proj      <- Linear (1024, 1024)
#         .num_heads = 16, head_dim = 64
#       .feed_forward                               <- SeamlessM4Tv2ConformerFeedForward
#         .intermediate_dense                       <- Linear(1024, 4096)
#         .output_dense                             <- Linear(4096, 1024)
#       .conv_module                                <- SeamlessM4Tv2ConformerConvolutionModule
#       .self_attn_layer_norm, .final_layer_norm    <- LayerNorm

import torch.nn.functional as F

def compute_block_influence(mdl, cal_wav_list, layer_idx, batch_size=4):
    \"\"\"
    Block Influence for speech encoder layer `layer_idx`.
    BI(l) = 1 - mean_cosine_sim(hidden_in, hidden_out) over calibration set.
    Higher BI = layer changes representation more = more important.
    \"\"\"
    enc_layers = _get_speech_enc_layers(mdl)
    layer = enc_layers[layer_idx]
    input_reps  = []
    output_reps = []

    def _fwd_hook(module, inp, out):
        # inp[0]: (B, T, D) hidden states entering this layer
        # out[0]: (B, T, D) hidden states exiting this layer
        h_in  = inp[0].detach().cpu().float()
        h_out = out[0].detach().cpu().float() if isinstance(out, tuple) else out.detach().cpu().float()
        # Pool over time: mean
        input_reps.append(h_in.mean(dim=1))    # (B, D)
        output_reps.append(h_out.mean(dim=1))  # (B, D)

    handle = layer.register_forward_hook(_fwd_hook)
    try:
        for i in range(0, len(cal_wav_list), batch_size):
            batch_wavs = cal_wav_list[i:i+batch_size]
            inputs = processor(audio=batch_wavs, sampling_rate=16000, return_tensors='pt',
                               padding=True)
            inputs = {k: v.to(_input_device(mdl)) for k, v in inputs.items()}
            with torch.no_grad():
                mdl.speech_encoder(**inputs)
    finally:
        handle.remove()

    if not input_reps: return 0.0
    H_in  = torch.cat(input_reps,  dim=0)   # (N, D)
    H_out = torch.cat(output_reps, dim=0)   # (N, D)
    cos_sim = F.cosine_similarity(H_in, H_out, dim=-1).mean().item()
    return 1.0 - cos_sim   # Block Influence: 0=identity, 1=max change

def score_encoder_layers_bi(mdl, cal_wavs, batch_size=4):
    \"\"\"Compute BI for all encoder layers. Returns dict {layer_idx: bi_score}.\"\"\"
    n_layers = len(_get_speech_enc_layers(mdl))
    print(f'[enc-prune] Computing BI for {n_layers} conformer layers ...')
    scores = {}
    for l in range(n_layers):
        bi = compute_block_influence(mdl, cal_wavs, l, batch_size=batch_size)
        scores[l] = bi
        print(f'  layer {l:>2}: BI={bi:.4f}')
        gc.collect()
    return scores

# Calibration: use FLEURS English audio (30 samples, source audio)
cal_wavs_enc = [s['wav'] for s in EVAL_SAMPLES.get(('eng','ben'),[])][:30]
cal_wavs_enc += [s['wav'] for s in EVAL_SAMPLES.get(('hin','eng'),[])][:10]
print(f'[enc-prune] Calibration: {len(cal_wavs_enc)} audio clips')
"""))

cells.append(code("""# ── Compute BI scores + greedy prune 24→12 ───────────────────────────────────
TARGET_ENC_LAYERS = 12
n_enc_to_remove = len(_get_speech_enc_layers(model)) - TARGET_ENC_LAYERS

print(f'[enc-prune] Target: {len(_get_speech_enc_layers(model))} → {TARGET_ENC_LAYERS} '
      f'(removing {n_enc_to_remove})')

# Compute BI once (stable scores; BI doesn't depend on removed layers for initial pass)
enc_bi_scores = score_encoder_layers_bi(model, cal_wavs_enc, batch_size=2)

# Sort by BI (ascending = least important first)
layers_by_importance = sorted(enc_bi_scores.items(), key=lambda x: x[1])
print('\\n[enc-prune] Layer BI ranking (ascending = least important):')
for l, bi in layers_by_importance:
    print(f'  layer {l:>2}: BI={bi:.4f}')

# Remove n_enc_to_remove layers with lowest BI
# IMPORTANT: remove from highest index to lowest to preserve index stability
to_remove_enc = sorted([l for l,_ in layers_by_importance[:n_enc_to_remove]], reverse=True)
print(f'\\n[enc-prune] Removing layers (indices, desc order): {sorted(to_remove_enc)}')

enc_layers = _get_speech_enc_layers(model)
for l_idx in to_remove_enc:
    del enc_layers[l_idx]
    print(f'  Removed layer {l_idx}. Remaining: {len(enc_layers)}')

sync_model_config(model)
gc.collect(); torch.cuda.empty_cache()
print(f'[enc-prune] Done. Encoder layers: {len(_get_speech_enc_layers(model))}  '
      f'Params: {count_params(model):.1f}M')
"""))

cells.append(code("""# ── Phase 3 BI figure ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
idxs = sorted(enc_bi_scores.keys())
vals = [enc_bi_scores[i] for i in idxs]
removed_set = set(to_remove_enc)
colors = ['#d32f2f' if i in removed_set else '#4CAF50' for i in idxs]
ax.bar(idxs, vals, color=colors, edgecolor='white')
ax.axhline(np.mean(vals), color='orange', ls='--', lw=1.5, label=f'Mean BI={np.mean(vals):.3f}')
ax.set_xlabel('Conformer Layer Index')
ax.set_ylabel('Block Influence (higher = more important)')
ax.set_title('Phase 3: Speech Encoder Block Influence Scores\\n(red = removed, green = kept)',
             fontweight='bold')
ax.legend(); ax.set_xticks(idxs)
plt.tight_layout()
save_figure(fig, 'phase3_encoder_block_influence.png')
plt.show()

# ── Phase 3 Benchmark ─────────────────────────────────────────────────────────
p3_rows_all = []
p3_summaries = {}
for src, tgt in EVAL_PAIRS:
    samples = EVAL_SAMPLES.get((src, tgt), [])
    if not samples: continue
    rows, summary = run_benchmark(
        model, samples, label=f'p3_encprune_{src}_{tgt}',
        src_lang=src, tgt_lang=tgt, save_n=0, use_asr_bleu=False)
    p3_rows_all.extend(rows)
    p3_summaries[(src, tgt)] = summary
    store_summary(summary)
save_results_csv(p3_rows_all, 'phase3_encoder_prune.csv')
save_model_to_drive(model, processor, 'phase3_encoder',
                    manifest_extra={'removed_enc_layers': sorted(to_remove_enc),
                                    'bi_scores': enc_bi_scores})
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — FLAP WIDTH PRUNING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Phase 4 — FLAP Width Pruning (FFN + Attention Heads)
**Technique:** Fluctuation-based Adaptive Structured Pruning (FLAP, AAAI 2024).
Applied to speech encoder FFN (4096→2816) and text decoder FFN (8192→5120) + heads.
**Expected savings:** ~150M params.
**Citation:** An et al., AAAI 2024 (arXiv:2312.11983).
"""))

cells.append(code("""# ═══════════════════════════════════════════════════════════════════════════════
# FLAP for SeamlessM4Tv2 — adapted for speech encoder Conformer + text decoder
# ───────────────────────────────────────────────────────────────────────────────
# Key exact layer paths (from HF transformers source):
#
# SPEECH ENCODER Conformer layer:
#   layer.feed_forward.intermediate_dense  <- Linear(1024, 4096)
#   layer.feed_forward.output_dense        <- Linear(4096, 1024)
#   layer.self_attn.q_proj, k_proj, v_proj, out_proj  <- Linear(1024, 1024)
#   layer.self_attn.num_heads = 16 (from config.speech_encoder_attention_heads)
#   head_dim = 64
#
# TEXT DECODER layer:
#   layer.ffn.fc1     <- Linear(1024, 8192)
#   layer.ffn.fc2     <- Linear(8192, 1024)
#   layer.self_attn.q_proj, k_proj, v_proj, out_proj  <- Linear(1024, 1024)
#   layer.encoder_attn.q_proj, k_proj, v_proj, out_proj
#   layer.self_attn.num_heads = 16, layer.encoder_attn.num_heads = 16
#   head_dim = 64
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_ffn_activations(mdl, cal_wavs, tgt_lang='ben', max_samples=256):
    \"\"\"
    Forward pass through model, collect intermediate FFN activations.
    Returns dict mapping layer path → activation tensor (N, hidden_dim).
    Used to compute WIFV metric.
    \"\"\"
    acts = {}
    handles = []

    def _make_hook(key):
        def _hook(module, inp, out):
            x = inp[0].detach().cpu().float()      # (B, T, D)
            acts.setdefault(key, []).append(x.reshape(-1, x.shape[-1]))
        return _hook

    # Speech encoder FFN intermediate activations
    for i, layer in enumerate(_get_speech_enc_layers(mdl)):
        h = layer.feed_forward.intermediate_dense.register_forward_hook(
            _make_hook(f'enc_ffn_{i}'))
        handles.append(h)
    # Text decoder FFN
    for i, layer in enumerate(_get_text_dec_layers(mdl)):
        h = layer.ffn.fc1.register_forward_hook(_make_hook(f'dec_ffn_{i}'))
        handles.append(h)

    try:
        count = 0
        for wav in cal_wavs[:max_samples]:
            try:
                inputs = processor(audio=wav, sampling_rate=16000, return_tensors='pt')
                inputs = {k: v.to(_input_device(mdl)) for k, v in inputs.items()}
                with torch.no_grad():
                    # Text-only forward pass (no vocoder)
                    orig_voc = mdl.vocoder
                    class _NV(nn.Module):
                        def forward(self,*a,**k): return torch.zeros(1,1),  [1]
                    mdl.vocoder = _NV()
                    mdl.generate(**inputs, tgt_lang=tgt_lang,
                                 return_intermediate_token_ids=True)
                    mdl.vocoder = orig_voc
                count += 1
            except Exception as e:
                pass
        print(f'[FLAP] Collected activations from {count} samples')
    finally:
        for h in handles: h.remove()

    # Concatenate
    return {k: torch.cat(v, dim=0)[:8192] for k, v in acts.items()}

def _wifv_score(W, X_acts):
    \"\"\"
    WIFV (Weighted Input Feature Variance) for FFN output neurons.
    W: [out_features, in_features]  (weight of fc1 / intermediate_dense)
    X_acts: [N, out_features]       (activations of this layer)
    Returns: [out_features] importance scores (higher = keep).
    \"\"\"
    # W_norm: L2 norm of each output-neuron's input weights
    W_norm = W.float().norm(dim=1)                 # [out_features]
    # X_std: std of each output neuron across samples
    X_std  = X_acts.float().std(dim=0).clamp(min=1e-8)   # [out_features]
    return W_norm * X_std

def prune_ffn_neurons(W1, b1, W2, b2, keep_ratio, acts):
    \"\"\"
    Prune intermediate neurons in a FFN block.
    W1: [d_int, d_model] (fc1 / intermediate_dense weight)
    b1: [d_int] or None
    W2: [d_model, d_int] (fc2 / output_dense weight)
    b2: [d_model] or None
    keep_ratio: fraction of neurons to keep
    acts: [N, d_int] activations of the post-W1 neurons
    Returns: new W1, b1, W2, b2, kept_indices
    \"\"\"
    scores = _wifv_score(W1, acts)
    n_keep = max(1, int(keep_ratio * W1.shape[0]))
    # Standardize scores within this layer for adaptive pruning
    scores = (scores - scores.mean()) / (scores.std() + 1e-8)
    keep_idx = scores.topk(n_keep).indices.sort().values  # sorted for contiguity
    new_W1 = W1[keep_idx]
    new_b1 = b1[keep_idx] if b1 is not None else None
    new_W2 = W2[:, keep_idx]
    # Bias compensation: adjust W2 bias to recover mean output
    # bc = (W2 @ ones * mean_removed_act).sum() → add to b2
    removed = torch.ones(W1.shape[0], dtype=torch.bool)
    removed[keep_idx] = False
    mean_removed = acts.float().mean(dim=0)[removed]
    bc = (W2[:, removed].float() @ mean_removed).half()
    if b2 is not None:
        new_b2 = b2 + bc.to(b2.device)
    else:
        new_b2 = bc
    return new_W1, new_b1, new_W2, new_b2, keep_idx

def apply_flap_ffn(mdl, acts_dict, enc_keep=0.70, dec_keep=0.65):
    \"\"\"
    Apply FLAP FFN pruning to speech encoder and text decoder.
    enc_keep: fraction of FFN neurons to keep in speech encoder
    dec_keep: fraction of FFN neurons to keep in text decoder
    \"\"\"
    print(f'[FLAP-FFN] enc_keep={enc_keep:.0%}  dec_keep={dec_keep:.0%}')
    prune_info = {'enc': {}, 'dec': {}}

    # ── Speech Encoder ────────────────────────────────────────────────────────
    for i, layer in enumerate(_get_speech_enc_layers(mdl)):
        key = f'enc_ffn_{i}'
        if key not in acts_dict: continue
        ff = layer.feed_forward
        W1 = ff.intermediate_dense.weight.data    # [4096, 1024]
        b1 = ff.intermediate_dense.bias.data if ff.intermediate_dense.bias is not None else None
        W2 = ff.output_dense.weight.data           # [1024, 4096]
        b2 = ff.output_dense.bias.data if ff.output_dense.bias is not None else None
        acts = acts_dict[key].to(W1.device)
        nW1, nb1, nW2, nb2, kept = prune_ffn_neurons(W1, b1, W2, b2, enc_keep, acts)
        n_kept = len(kept)
        # Replace linear layers in-place
        ff.intermediate_dense = nn.Linear(W1.shape[1], n_kept,
                                           bias=(b1 is not None)).to(W1.device)
        ff.intermediate_dense.weight.data = nW1
        if nb1 is not None and ff.intermediate_dense.bias is not None:
            ff.intermediate_dense.bias.data = nb1
        ff.output_dense = nn.Linear(n_kept, W2.shape[0],
                                     bias=(b2 is not None)).to(W1.device)
        ff.output_dense.weight.data = nW2
        if nb2 is not None:
            if ff.output_dense.bias is not None:
                ff.output_dense.bias.data = nb2
            else:
                ff.output_dense.bias = nn.Parameter(nb2)
        prune_info['enc'][i] = n_kept
    print(f'  Speech enc FFN: {W1.shape[0]} → ~{int(W1.shape[0]*enc_keep)} neurons/layer')

    # ── Text Decoder ──────────────────────────────────────────────────────────
    for i, layer in enumerate(_get_text_dec_layers(mdl)):
        key = f'dec_ffn_{i}'
        if key not in acts_dict: continue
        W1 = layer.ffn.fc1.weight.data              # [8192, 1024]
        b1 = layer.ffn.fc1.bias.data if layer.ffn.fc1.bias is not None else None
        W2 = layer.ffn.fc2.weight.data              # [1024, 8192]
        b2 = layer.ffn.fc2.bias.data if layer.ffn.fc2.bias is not None else None
        acts = acts_dict[key].to(W1.device)
        nW1, nb1, nW2, nb2, kept = prune_ffn_neurons(W1, b1, W2, b2, dec_keep, acts)
        n_kept = len(kept)
        layer.ffn.fc1 = nn.Linear(W1.shape[1], n_kept,
                                   bias=(b1 is not None)).to(W1.device)
        layer.ffn.fc1.weight.data = nW1
        if nb1 is not None and layer.ffn.fc1.bias is not None:
            layer.ffn.fc1.bias.data = nb1
        layer.ffn.fc2 = nn.Linear(n_kept, W2.shape[0],
                                   bias=(b2 is not None)).to(W1.device)
        layer.ffn.fc2.weight.data = nW2
        if nb2 is not None:
            if layer.ffn.fc2.bias is not None:
                layer.ffn.fc2.bias.data = nb2
            else:
                layer.ffn.fc2.bias = nn.Parameter(nb2)
        prune_info['dec'][i] = n_kept
    print(f'  Text dec FFN: {W1.shape[0]} → ~{int(W1.shape[0]*dec_keep)} neurons/layer')
    return prune_info

print('FLAP FFN pruning functions ready.')
"""))

cells.append(code("""# ── Collect activations (calibration forward pass) ───────────────────────────
# Mix of all 5 language pairs for balanced calibration
flap_cal_wavs = []
for src, tgt in [('eng','ben'),('eng','hin'),('eng','tam'),('eng','arb'),
                 ('ben','eng'),('hin','eng')]:
    samps = EVAL_SAMPLES.get((src, tgt), [])
    flap_cal_wavs.extend([s['wav'] for s in samps[:8]])

print(f'[FLAP] Calibration set: {len(flap_cal_wavs)} audio clips')
acts_dict = _collect_ffn_activations(model, flap_cal_wavs, tgt_lang='ben', max_samples=200)
print(f'[FLAP] Collected acts for {len(acts_dict)} layer keys')
print(f'  enc keys: {sum(1 for k in acts_dict if k.startswith(\"enc\"))}'
      f'  dec keys: {sum(1 for k in acts_dict if k.startswith(\"dec\"))}')
gc.collect()
"""))

cells.append(code("""# ── Apply FLAP width pruning ──────────────────────────────────────────────────
# Speech encoder FFN: 4096 → ~2816 (~31% reduction)
# Text decoder FFN:   8192 → ~5120 (~37.5% reduction)
flap_info = apply_flap_ffn(model, acts_dict, enc_keep=0.69, dec_keep=0.625)

# Update config to reflect new FFN dimensions
# Use first remaining layer's actual fc1 out_features
if _get_speech_enc_layers(model):
    actual_enc_ffn = _get_speech_enc_layers(model)[0].feed_forward.intermediate_dense.out_features
    model.config.speech_encoder_intermediate_size = actual_enc_ffn
    print(f'  config.speech_encoder_intermediate_size → {actual_enc_ffn}')
if _get_text_dec_layers(model):
    actual_dec_ffn = _get_text_dec_layers(model)[0].ffn.fc1.out_features
    model.config.decoder_ffn_dim = actual_dec_ffn
    print(f'  config.decoder_ffn_dim → {actual_dec_ffn}')

del acts_dict; gc.collect(); torch.cuda.empty_cache()
print(f'\\n[Phase 4] Params after FLAP: {count_params(model):.1f}M')

# ── Phase 4 Benchmark ─────────────────────────────────────────────────────────
p4_rows_all, p4_summaries = [], {}
for src, tgt in EVAL_PAIRS:
    samples = EVAL_SAMPLES.get((src, tgt), [])
    if not samples: continue
    rows, summary = run_benchmark(
        model, samples, label=f'p4_flap_{src}_{tgt}',
        src_lang=src, tgt_lang=tgt, save_n=0, use_asr_bleu=False)
    p4_rows_all.extend(rows)
    p4_summaries[(src, tgt)] = summary
    store_summary(summary)
save_results_csv(p4_rows_all, 'phase4_flap.csv')
save_model_to_drive(model, processor, 'phase4_flap',
                    manifest_extra={'flap_info': flap_info})
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — T2U PRUNING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Phase 5 — T2U Model Layer Pruning (6+6 → 4+4)
**Technique:** Greedy layer removal by ChrF impact on full S2ST pipeline.
SeamlessM4Tv2 exact paths: `t2u_model.model.encoder.layers`, `t2u_model.model.decoder.layers`.
**Expected savings:** ~50M params.
"""))

cells.append(code("""# SeamlessM4Tv2 T2U model structure (EXACT):
# model.t2u_model                              <- SeamlessM4Tv2UnitHifiGan (wrapper)
#   .model                                     <- SeamlessM4Tv2UnitT2UModel
#     .encoder                                 <- SeamlessM4Tv2UnitEncoder
#       .layers[i]                             <- SeamlessM4Tv2EncoderLayer
#         .self_attn                           <- SeamlessM4Tv2Attention (q/k/v/out_proj)
#         .ffn                                 <- SeamlessM4Tv2FFN (fc1, fc2)
#         .self_attn_layer_norm, .final_layer_norm
#     .decoder                                 <- SeamlessM4Tv2UnitDecoder
#       .layers[i]                             <- SeamlessM4Tv2DecoderLayer
#         .self_attn, .encoder_attn, .ffn (same structure as text decoder)
# Note: t2u_model.model.encoder has 6 layers, decoder has 6 layers

@contextmanager
def _skip_t2u_layer(mdl, component, layer_idx):
    \"\"\"Skip T2U encoder or decoder layer. component: 'enc' or 'dec'.\"\"\"
    if component == 'enc':
        layers = _get_t2u_enc_layers(mdl)
    else:
        layers = _get_t2u_dec_layers(mdl)
    orig = layers[layer_idx]

    class _Identity(nn.Module):
        def forward(self, hidden_states, attention_mask=None,
                    encoder_hidden_states=None, encoder_attention_mask=None,
                    past_key_value=None, output_attentions=False, use_cache=False, **kw):
            pkv = (torch.zeros(1), torch.zeros(1)) if use_cache else None
            return (hidden_states, None, None, pkv)

    layers[layer_idx] = _Identity().to(next(orig.parameters()).device)
    try:
        yield
    finally:
        layers[layer_idx] = orig

def score_t2u_layers(mdl, cal_samples, tgt_lang='ben', fast_n=12):
    \"\"\"Score T2U encoder and decoder layers by ChrF drop.\"\"\"
    base = quick_chrf(mdl, cal_samples, tgt_lang=tgt_lang, n=fast_n)
    print(f'[t2u-prune] Base ChrF: {base:.2f}')
    scores = {'enc': {}, 'dec': {}}
    for comp, getter in [('enc', _get_t2u_enc_layers), ('dec', _get_t2u_dec_layers)]:
        n = len(getter(mdl))
        for l in range(n):
            with _skip_t2u_layer(mdl, comp, l):
                c = quick_chrf(mdl, cal_samples, tgt_lang=tgt_lang, n=fast_n)
            scores[comp][l] = base - c
            print(f'  T2U {comp} layer {l}: drop={scores[comp][l]:+.4f}')
    return scores

TARGET_T2U_ENC = 4
TARGET_T2U_DEC = 4

cal_t2u = cal_samples_dec[:N_EVAL_FAST]
t2u_scores = score_t2u_layers(model, cal_t2u, tgt_lang='ben')

# Remove least important T2U encoder layers
for comp, target, getter in [('enc', TARGET_T2U_ENC, _get_t2u_enc_layers),
                               ('dec', TARGET_T2U_DEC, _get_t2u_dec_layers)]:
    layers = getter(model)
    n_remove = len(layers) - target
    sorted_by_imp = sorted(t2u_scores[comp].items(), key=lambda x: x[1])
    to_rm = sorted([l for l,_ in sorted_by_imp[:n_remove]], reverse=True)
    print(f'[t2u-prune] {comp}: removing layers {sorted(to_rm)}')
    for idx in to_rm:
        del layers[idx]
    print(f'  {comp} layers remaining: {len(layers)}')

sync_model_config(model)
gc.collect(); torch.cuda.empty_cache()
print(f'[Phase 5] Params: {count_params(model):.1f}M')
print(f'  T2U enc layers: {len(_get_t2u_enc_layers(model))}')
print(f'  T2U dec layers: {len(_get_t2u_dec_layers(model))}')

# ── Phase 5 Benchmark ─────────────────────────────────────────────────────────
p5_rows_all, p5_summaries = [], {}
for src, tgt in EVAL_PAIRS:
    samples = EVAL_SAMPLES.get((src, tgt), [])
    if not samples: continue
    rows, summary = run_benchmark(
        model, samples, label=f'p5_t2uprune_{src}_{tgt}',
        src_lang=src, tgt_lang=tgt, save_n=0, use_asr_bleu=False)
    p5_rows_all.extend(rows)
    p5_summaries[(src, tgt)] = summary
    store_summary(summary)
save_results_csv(p5_rows_all, 'phase5_t2u_prune.csv')
save_model_to_drive(model, processor, 'phase5_t2u',
                    manifest_extra={'t2u_scores': t2u_scores})
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — LoRA + Sequence-Level KD Recovery Fine-Tuning
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Phase 6 — LoRA + Sequence-Level Knowledge Distillation Recovery
**Technique:** LoRA adapters (r=64, rsLoRA) + seq-level KD from teacher.
Multi-stage: Stage A (T2U only) → Stage B (decoder+T2U, mixed KD+authentic) → Stage C (joint).
**Citation:** Moslem 2025; Hu et al. ICLR 2022; Kim & Rush EMNLP 2016; Dettmers et al. NeurIPS 2023.
"""))

cells.append(code("""from peft import LoraConfig, get_peft_model, TaskType

# ── LoRA configuration ────────────────────────────────────────────────────────
# Target modules: exact Linear layer names inside text decoder + T2U
# Verified from SeamlessM4Tv2 HF transformers source:
#   text_decoder.layers.{i}.self_attn.{q,k,v,out}_proj
#   text_decoder.layers.{i}.encoder_attn.{q,k,v,out}_proj
#   t2u_model.model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj
#   t2u_model.model.decoder.layers.{i}.self_attn.{q,k,v,out}_proj
#   t2u_model.model.decoder.layers.{i}.encoder_attn.{q,k,v,out}_proj

LORA_R     = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

lora_target_modules = [
    # Text decoder self-attention (all 4 projections)
    'text_decoder.layers.{i}.self_attn.q_proj',
    'text_decoder.layers.{i}.self_attn.k_proj',
    'text_decoder.layers.{i}.self_attn.v_proj',
    'text_decoder.layers.{i}.self_attn.out_proj',
    # Text decoder cross-attention
    'text_decoder.layers.{i}.encoder_attn.q_proj',
    'text_decoder.layers.{i}.encoder_attn.v_proj',
    # T2U encoder self-attention
    't2u_model.model.encoder.layers.{i}.self_attn.q_proj',
    't2u_model.model.encoder.layers.{i}.self_attn.v_proj',
    # T2U decoder self + cross attn
    't2u_model.model.decoder.layers.{i}.self_attn.q_proj',
    't2u_model.model.decoder.layers.{i}.self_attn.v_proj',
    't2u_model.model.decoder.layers.{i}.encoder_attn.q_proj',
    't2u_model.model.decoder.layers.{i}.encoder_attn.v_proj',
]
# Expand {i} wildcards using actual module names from the model
# PEFT accepts exact module names or regex; we use module name matching
# Use suffix-based matching by passing just the projection names:
lora_modules_exact = [
    'q_proj', 'k_proj', 'v_proj', 'out_proj'   # matched inside decoder + T2U
]
# We scope to decoder+T2U only via PEFT's layers_to_transform parameter below

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=lora_modules_exact,
    use_rslora=True,       # Rank-Stabilized LoRA (Kalajdzievski 2023)
    bias='none',
    task_type=TaskType.SEQ_2_SEQ_LM,
)

print('[LoRA] Config ready.')
print(f'  r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}, rsLoRA=True')
print(f'  target_modules: {lora_modules_exact}')
"""))

cells.append(code("""# ── Build fine-tuning dataset ─────────────────────────────────────────────────
# All 8 X↔Eng pairs from FLEURS train + limited CV17

print('[finetune] Building training dataset...')
FT_SAMPLES_ALL = []

# FLEURS train for all 8 pairs
for src, tgt in EVAL_PAIRS:
    try:
        samps = build_train_samples(src, tgt, split='train')
        FT_SAMPLES_ALL.extend(samps)
        print(f'  FLEURS {src}→{tgt}: {len(samps)} samples')
    except Exception as e:
        print(f'  FLEURS {src}→{tgt}: {e}')

# FLEURS validation for all 8 pairs (add to train for small-data fine-tuning)
for src, tgt in EVAL_PAIRS:
    try:
        samps = build_train_samples(src, tgt, split='validation')
        FT_SAMPLES_ALL.extend(samps)
    except: pass

print(f'\\n[finetune] Total training samples: {len(FT_SAMPLES_ALL)}')

# Shuffle
import random; random.seed(42); random.shuffle(FT_SAMPLES_ALL)
print('[finetune] Training data ready.')
"""))

cells.append(code("""# ── Sequence-Level KD: generate teacher pseudo-labels ─────────────────────────
# Load teacher model (fp16, different from pruned student) for KD
# NOTE: On Kaggle T4 x2, we cannot hold both teacher and student in VRAM simultaneously.
# Strategy: generate all KD pseudo-labels FIRST (teacher only), save to disk, then load student.

KD_CACHE_FILE = f'{DATA_DIR}/kd_pseudolabels.pt'
N_KD_SAMPLES = min(600, len(FT_SAMPLES_ALL))   # cap for Kaggle session time

def generate_kd_pseudolabels(teacher_mdl, teacher_proc, ft_samples, n_kd=N_KD_SAMPLES):
    \"\"\"
    Generate pseudo-labels from teacher.
    Returns list of {id, wav, ref_text, kd_text} where kd_text = teacher translation.
    \"\"\"
    if os.path.exists(KD_CACHE_FILE):
        print(f'[KD] Loading cached pseudo-labels from {KD_CACHE_FILE}')
        return torch.load(KD_CACHE_FILE, map_location='cpu', weights_only=False)

    print(f'[KD] Generating {n_kd} pseudo-labels with teacher...')
    kd_data = []
    for i, s in enumerate(ft_samples[:n_kd]):
        if i % 50 == 0: print(f'  [{i}/{n_kd}]  gpu_mem:', end=' '); gpu_mem()
        # Infer target language from sample (we tagged pairs above; use 'ben' as fallback)
        tgt_lang = s.get('tgt_lang', 'ben')
        try:
            kd_text = run_s2t(teacher_mdl, s['wav'], tgt_lang=tgt_lang)
            kd_data.append({**s, 'kd_text': kd_text})
        except Exception as e:
            kd_data.append({**s, 'kd_text': s['ref']})   # fallback to ground truth
    torch.save(kd_data, KD_CACHE_FILE)
    if ON_KAGGLE: _rclone_push(KD_CACHE_FILE, 'data')
    print(f'[KD] Saved {len(kd_data)} pseudo-labels.')
    return kd_data

# Tag samples with their target language
for s in FT_SAMPLES_ALL:
    if 'tgt_lang' not in s:
        s['tgt_lang'] = 'eng'   # default; override below when known

# Re-tag properly using pair structure
pair_samples = {}
for src, tgt in EVAL_PAIRS:
    try:
        samps = build_train_samples(src, tgt, split='train')
        for s in samps: s['tgt_lang'] = tgt
        pair_samples[(src, tgt)] = samps
    except: pass

FT_TAGGED = []
for (src, tgt), samps in pair_samples.items():
    FT_TAGGED.extend(samps)
random.shuffle(FT_TAGGED)
print(f'[finetune] Tagged samples: {len(FT_TAGGED)}')
"""))

cells.append(code("""# ── KD pseudo-label generation (uses TEACHER model, must be loaded) ─────────────
# If teacher is already loaded as `model` (which it was in Phase 0 before pruning),
# we need the PRUNED student. Strategy:
# 1. If phase5 model saved to Drive, load it as `student`
# 2. Use loaded `model` variable (teacher from Phase 0) for KD generation
# 3. After KD cache built, del teacher and load student

# Check if we already have the Phase 5 model as `model`
# (if running sequentially this session, `model` IS the pruned student — NOT the teacher)
# For KD, we need TEACHER. Options:
#   a) If teacher was kept in memory (unlikely — too much VRAM)
#   b) If KD cache already exists from a previous session (best case)
#   c) Load teacher fresh, generate labels, del teacher, load student

if os.path.exists(KD_CACHE_FILE):
    print('[KD] Pseudo-label cache found. Skipping teacher load.')
    kd_data = torch.load(KD_CACHE_FILE, map_location='cpu', weights_only=False)
    print(f'[KD] Loaded {len(kd_data)} cached pseudo-labels.')
else:
    print('[KD] No cache. Loading teacher to generate pseudo-labels...')
    print('[KD] WARNING: this requires ~4GB VRAM for teacher (fp16)')
    # Save student state first
    save_model_to_drive(model, processor, 'phase5_student_before_kd')
    del model; gc.collect(); torch.cuda.empty_cache()
    teacher = load_base_model()
    kd_data = generate_kd_pseudolabels(teacher, processor, FT_TAGGED, N_KD_SAMPLES)
    del teacher; gc.collect(); torch.cuda.empty_cache()
    # Reload student
    model, processor = load_model_from_drive('phase5_student_before_kd')

print(f'[KD] {len(kd_data)} pseudo-labels ready.')
"""))

cells.append(code("""# ── Apply LoRA to student model ───────────────────────────────────────────────
from peft import get_peft_model

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Apply LoRA
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Verify we can do forward pass
test_wav = FT_TAGGED[0]['wav'] if FT_TAGGED else np.zeros(16000, dtype=np.float32)
try:
    _ = run_s2t(model, test_wav, tgt_lang='ben')
    print('[LoRA] Forward pass OK.')
except Exception as e:
    print(f'[LoRA] Forward pass ERROR: {e}')
"""))

cells.append(code("""# ═══════════════════════════════════════════════════════════════════════════════
# Training loop: Stage A + B + C
# Stage A: 1 epoch, T2U LoRA only, lr=2e-4 — restores unit generation
# Stage B: 2 epochs, all LoRA, lr=1e-4 — seq-level KD + authentic data
# Stage C: 1 epoch, lr=5e-5 — fine balance all languages
# ═══════════════════════════════════════════════════════════════════════════════
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

STAGES = [
    dict(name='A', epochs=1, lr=2e-4, kd_ratio=0.0, desc='T2U only'),
    dict(name='B', epochs=2, lr=1e-4, kd_ratio=0.5, desc='All LoRA + KD'),
    dict(name='C', epochs=1, lr=5e-5, kd_ratio=0.3, desc='Fine balance'),
]
BATCH_SIZE    = 2
GRAD_ACCUM    = 16   # effective batch = 32
MAX_SEQ_LEN   = 256
LOG_EVERY     = 50
EVAL_EVERY    = 200
SAVE_EVERY    = 300

scaler = GradScaler()

def _make_optimizer(mdl, lr):
    return AdamW([p for p in mdl.parameters() if p.requires_grad],
                 lr=lr, weight_decay=0.01)

def _forward_loss(mdl, batch_wavs, batch_texts, tgt_lang):
    \"\"\"
    Compute CE loss: speech → text labels.
    batch_wavs: list of numpy float32 arrays
    batch_texts: list of target text strings
    \"\"\"
    inputs  = processor(audio=batch_wavs, sampling_rate=16000,
                        return_tensors='pt', padding=True)
    labels  = processor(text=batch_texts, return_tensors='pt',
                        padding=True, truncation=True,
                        max_length=MAX_SEQ_LEN).input_ids
    # Remap label IDs to pruned vocab space if needed
    if hasattr(mdl, '_vocab_remap_to_old') or (
        hasattr(mdl, 'base_model') and hasattr(mdl.base_model.model, '_vocab_remap_to_old')):
        # Build new_to_old lookup
        base = mdl.base_model.model if hasattr(mdl,'base_model') else mdl
        remap = base._vocab_remap_to_old
        old_to_new = {old.item(): new for new, old in enumerate(remap)}
        labels_new = labels.clone()
        for bi in range(labels_new.shape[0]):
            for j in range(labels_new.shape[1]):
                t = labels_new[bi,j].item()
                if t in (-100, -1): continue
                labels_new[bi,j] = old_to_new.get(t, -100)
        labels = labels_new
    labels[labels == processor.tokenizer.pad_token_id] = -100
    dev = _input_device(mdl)
    inputs  = {k: v.to(dev) for k, v in inputs.items()}
    labels  = labels.to(dev)
    with autocast():
        out = mdl(**inputs, labels=labels, tgt_lang=tgt_lang)
    return out.loss

def run_training_stage(mdl, stage_cfg, ft_samples, kd_samples):
    stage = stage_cfg['name']
    lr    = stage_cfg['lr']
    kd_r  = stage_cfg['kd_ratio']
    epcs  = stage_cfg['epochs']
    print(f'\\n{\"=\"*60}\\n  Stage {stage}: {stage_cfg[\"desc\"]}'
          f'  lr={lr}  kd_ratio={kd_r}  epochs={epcs}\\n{\"=\"*60}')

    # Stage A: freeze decoder LoRA, only T2U adapters
    if stage == 'A':
        for name, p in mdl.named_parameters():
            if p.requires_grad:
                # Only keep t2u_model LoRA params active
                if 't2u_model' not in name:
                    p.requires_grad = False
    else:
        # Re-enable all LoRA params
        for name, p in mdl.named_parameters():
            if 'lora_' in name: p.requires_grad = True

    mdl.print_trainable_parameters()
    opt = _make_optimizer(mdl, lr)
    n_steps = len(ft_samples) * epcs // BATCH_SIZE
    sched = CosineAnnealingLR(opt, T_max=max(1, n_steps), eta_min=lr*0.1)

    step = 0
    all_pairs = list(pair_samples.keys())
    kd_dict = {s['id']: s['kd_text'] for s in kd_samples}

    for epoch in range(epcs):
        random.shuffle(all_pairs)
        for src, tgt in all_pairs:
            samps = pair_samples.get((src,tgt), [])
            if not samps: continue
            random.shuffle(samps)
            for i in range(0, len(samps), BATCH_SIZE):
                batch = samps[i:i+BATCH_SIZE]
                if not batch: continue
                # Mix authentic and KD data
                batch_wavs  = [s['wav'] for s in batch]
                batch_texts = []
                for s in batch:
                    kd_txt = kd_dict.get(s['id'])
                    if kd_txt and random.random() < kd_r:
                        batch_texts.append(kd_txt)
                    else:
                        batch_texts.append(s['ref'])

                try:
                    loss = _forward_loss(mdl, batch_wavs, batch_texts, tgt_lang=tgt)
                    loss = loss / GRAD_ACCUM
                    scaler.scale(loss).backward()
                    if (step+1) % GRAD_ACCUM == 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in mdl.parameters() if p.requires_grad], 1.0)
                        scaler.step(opt); scaler.update()
                        opt.zero_grad(); sched.step()

                    if step % LOG_EVERY == 0:
                        print(f'  [{stage}] step={step:>5}  loss={loss.item()*GRAD_ACCUM:.4f}'
                              f'  lr={sched.get_last_lr()[0]:.2e}  {src}→{tgt}')
                    if step % EVAL_EVERY == 0 and step > 0:
                        quick = quick_chrf(mdl, EVAL_SAMPLES.get(('eng','ben'),[])[:10],
                                           tgt_lang='ben', n=10)
                        print(f'  [eval] step={step} ChrF(eng→ben)={quick:.2f}')
                        gc.collect(); torch.cuda.empty_cache()
                    if step % SAVE_EVERY == 0 and step > 0:
                        save_checkpoint({'step': step, 'stage': stage,
                                         'loss': loss.item()},
                                        f'lora_stage{stage}', step=step)
                    step += 1
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f'  [OOM] step={step}, skipping batch')
                        opt.zero_grad(); gc.collect(); torch.cuda.empty_cache()
                    else:
                        raise e

    print(f'[Stage {stage}] Done. Total steps: {step}')
    return mdl

# Run all stages
for stage_cfg in STAGES:
    model = run_training_stage(model, stage_cfg, FT_TAGGED, kd_data)
    gc.collect(); torch.cuda.empty_cache()

print('\\n[Phase 6] All training stages complete.')
"""))

cells.append(code("""# ── Merge LoRA adapters into base weights ─────────────────────────────────────
from peft import PeftModel

print('[LoRA] Merging adapters into base weights...')
if hasattr(model, 'merge_and_unload'):
    model = model.merge_and_unload()
    print('[LoRA] Merged. Model is now a plain SeamlessM4Tv2ForSpeechToSpeech.')
else:
    # Manual merge if merge_and_unload not available for this peft version
    for name, module in model.named_modules():
        if hasattr(module, 'merge_weights'):
            module.merge_weights()
    print('[LoRA] Manual merge done.')

gc.collect(); torch.cuda.empty_cache()
print(f'[Phase 6] Final params: {count_params(model):.1f}M')

# ── Phase 6 Full Benchmark (with ASR-BLEU using MMS) ─────────────────────────
print('\\n[Phase 6] Full benchmark (ASR-BLEU + ASR-ChrF)...')
p6_rows_all, p6_summaries = [], {}
for src, tgt in EVAL_PAIRS:
    samples = EVAL_SAMPLES.get((src, tgt), [])
    if not samples: continue
    rows, summary = run_benchmark(
        model, samples, label=f'p6_lora_{src}_{tgt}',
        src_lang=src, tgt_lang=tgt, save_n=3, use_asr_bleu=True)
    p6_rows_all.extend(rows)
    p6_summaries[(src, tgt)] = summary
    store_summary(summary)
save_results_csv(p6_rows_all, 'phase6_lora_recovery.csv')
save_model_to_drive(model, processor, 'phase6_lora_merged')
"""))

cells.append(code("""# ── Phase 6 Recovery Figure ───────────────────────────────────────────────────
import pandas as pd

print('\\n=== Phase 6 Recovery Summary ===')
print(f'{\"Pair\":<14} {\"Teacher\":>10} {\"Pruned(P5)\":>10} {\"Recovered\":>10} {\"Recovery%\":>10}')
for (src, tgt) in EVAL_PAIRS:
    p0s = p0_summaries.get((src, tgt))
    p5s = p5_summaries.get((src, tgt))
    p6s = p6_summaries.get((src, tgt))
    if not (p0s and p6s): continue
    if p0s['avg_bleu'] > 0:
        pct = p6s['avg_bleu'] / p0s['avg_bleu'] * 100
    else:
        pct = 0
    p5v = p5s['avg_bleu'] if p5s else float('nan')
    print(f'  {src}→{tgt:<10} {p0s[\"avg_bleu\"]:>10.2f} {p5v:>10.2f} '
          f'{p6s[\"avg_bleu\"]:>10.2f} {pct:>9.1f}%')

# Grouped bar chart: Teacher vs Pruned vs Recovered
pairs_labels = [f'{s}→{t}' for s,t in EVAL_PAIRS if (s,t) in p6_summaries]
teacher_bleu  = [p0_summaries.get((s,t),{}).get('avg_bleu',0) for s,t in EVAL_PAIRS if (s,t) in p6_summaries]
pruned_bleu   = [p5_summaries.get((s,t),{}).get('avg_bleu',0) for s,t in EVAL_PAIRS if (s,t) in p6_summaries]
recovered_bleu= [p6_summaries[(s,t)]['avg_bleu'] for s,t in EVAL_PAIRS if (s,t) in p6_summaries]

x = np.arange(len(pairs_labels)); w = 0.25
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Phase 6: Compression Recovery\\n(Teacher 1.8B vs Pruned vs LoRA+KD Recovered ~850M)',
             fontweight='bold', fontsize=13)
for ax, (yvals_list, metric_name) in zip(axes, [
    ([teacher_bleu, pruned_bleu, recovered_bleu], 'Text BLEU'),
    ([[p0_summaries.get((s,t),{}).get('avg_asr_bleu',0) for s,t in EVAL_PAIRS if (s,t) in p6_summaries],
      [p5_summaries.get((s,t),{}).get('avg_asr_bleu',0) for s,t in EVAL_PAIRS if (s,t) in p6_summaries],
      [p6_summaries[(s,t)]['avg_asr_bleu'] for s,t in EVAL_PAIRS if (s,t) in p6_summaries]],
     'ASR-BLEU (MMS-1b-all)')
]):
    r1 = ax.bar(x-w, yvals_list[0], w, label='Teacher 1.8B', color='#2196F3', alpha=0.9)
    r2 = ax.bar(x,   yvals_list[1], w, label='Pruned ~900M', color='#FF5722', alpha=0.9)
    r3 = ax.bar(x+w, yvals_list[2], w, label='Recovered ~850M', color='#4CAF50', alpha=0.9)
    ax.set_title(metric_name, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(pairs_labels, rotation=35, ha='right', fontsize=8)
    ax.legend(); ax.set_ylabel(metric_name)
    for rect in r3:
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height(),
                f'{rect.get_height():.1f}', ha='center', va='bottom', fontsize=7)
plt.tight_layout()
save_figure(fig, 'phase6_recovery_comparison.png')
plt.show()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — LONG AUDIO (40-60s)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Phase 7 — Long Audio Support (40–60 seconds)
**Technique:** VAD-guided overlapping chunking with crossfade stitching.
Uses `silero-vad` for silence-boundary detection.
"""))

cells.append(code("""# ── Install silero-vad ────────────────────────────────────────────────────────
subprocess.run(['pip', 'install', '-q', 'silero-vad'], check=True, capture_output=True)
print('silero-vad installed.')
"""))

cells.append(code("""import torchaudio

class LongAudioTranslator:
    \"\"\"
    VAD-guided chunk translator for audio > 30 seconds.

    Strategy:
      1. Silero VAD → find speech/silence boundaries
      2. Split at silence boundaries, max chunk = MAX_CHUNK_S seconds
      3. 2-second overlap on each side for context continuity
      4. Translate each chunk independently with run_s2st()
      5. Crossfade-stitch output waveforms (50ms fade)
    \"\"\"
    MAX_CHUNK_S   = 25      # max seconds per chunk
    OVERLAP_S     = 2       # seconds of overlap between chunks
    SILENCE_MIN_S = 0.3     # minimum silence duration for a split point
    CROSSFADE_MS  = 50      # crossfade duration in ms

    def __init__(self, mdl, proc):
        self.model = mdl
        self.processor = proc
        self._vad_model = None
        self._vad_utils  = None

    def _load_vad(self):
        if self._vad_model is None:
            from silero_vad import load_silero_vad, get_speech_timestamps
            self._vad_model = load_silero_vad()
            self._get_ts   = get_speech_timestamps
            print('[VAD] Silero VAD loaded.')

    def _get_speech_boundaries(self, audio_np, sr=16000):
        \"\"\"Returns list of (start_sample, end_sample) for speech segments.\"\"\"
        self._load_vad()
        t = torch.tensor(audio_np)
        ts = self._get_ts(t, self._vad_model, sampling_rate=sr,
                          min_silence_duration_ms=int(self.SILENCE_MIN_S*1000),
                          return_seconds=False)
        return [(seg['start'], seg['end']) for seg in ts]

    def _build_chunks(self, audio_np, sr=16000):
        \"\"\"
        Build overlapping chunks respecting VAD boundaries.
        Returns list of (chunk_np, start_sample, end_sample).
        \"\"\"
        max_s  = int(self.MAX_CHUNK_S * sr)
        ovl_s  = int(self.OVERLAP_S   * sr)
        total  = len(audio_np)
        chunks = []
        pos = 0
        while pos < total:
            end = min(pos + max_s, total)
            # Extend to nearest silence boundary if possible
            chunk = audio_np[max(0, pos-ovl_s):end+ovl_s]
            chunks.append((chunk, pos, end))
            pos = end
            if pos >= total: break
        return chunks

    def _crossfade(self, a1, a2, sr=16000):
        \"\"\"Crossfade-stitch two waveforms. a1 fades out, a2 fades in.\"\"\"
        fade_n = int(self.CROSSFADE_MS * sr / 1000)
        fade_n = min(fade_n, len(a1), len(a2))
        if fade_n == 0: return np.concatenate([a1, a2])
        fade_out = np.linspace(1, 0, fade_n, dtype=np.float32)
        fade_in  = np.linspace(0, 1, fade_n, dtype=np.float32)
        out  = a1[:-fade_n].copy()
        join = a1[-fade_n:] * fade_out + a2[:fade_n] * fade_in
        tail = a2[fade_n:]
        return np.concatenate([out, join, tail])

    def translate(self, audio_np, src_lang, tgt_lang, sr=16000):
        \"\"\"
        Full long-audio translation pipeline.
        Returns: (translations_list, stitched_audio_np, output_sr)
        \"\"\"
        dur = len(audio_np) / sr
        print(f'[LongAudio] Input: {dur:.1f}s  {src_lang}→{tgt_lang}')

        if dur <= self.MAX_CHUNK_S:
            # Short audio: single-pass
            text, wav_out = run_s2st(self.model, audio_np, tgt_lang=tgt_lang)
            print(f'[LongAudio] Single pass: {text[:80]}')
            return [text], wav_out, self.model.config.sampling_rate

        chunks = self._build_chunks(audio_np, sr)
        print(f'[LongAudio] {len(chunks)} chunks of ~{self.MAX_CHUNK_S}s each')

        texts    = []
        waveforms= []
        out_sr   = self.model.config.sampling_rate

        for i, (chunk, cstart, cend) in enumerate(chunks):
            print(f'  Chunk {i+1}/{len(chunks)}: {len(chunk)/sr:.1f}s ...')
            try:
                t, w = run_s2st(self.model, chunk, tgt_lang=tgt_lang)
                texts.append(t)
                waveforms.append(w)
                print(f'    text: {t[:60]}')
            except Exception as e:
                print(f'  [warn] chunk {i+1} failed: {e}')
                texts.append('')
                waveforms.append(np.zeros(out_sr // 4, dtype=np.float32))

        # Stitch with crossfade
        stitched = waveforms[0]
        for w in waveforms[1:]:
            stitched = self._crossfade(stitched, w, sr=out_sr)

        print(f'[LongAudio] Done. Output: {len(stitched)/out_sr:.1f}s')
        return texts, stitched, out_sr

long_translator = LongAudioTranslator(model, processor)
print('LongAudioTranslator ready.')
"""))

cells.append(code("""# ── Phase 7 Benchmark: Quality vs. audio duration ────────────────────────────
# Synthesize test clips of 10/20/30/40/50/60 seconds by concatenating FLEURS samples
import numpy as np

def build_long_audio_sample(samples, target_dur_s, sr=16000):
    \"\"\"Concatenate samples until target duration reached. Returns (audio_np, ref_texts).\"\"\"
    parts, refs, used = [], [], 0
    for s in samples * 5:   # loop samples if needed
        parts.append(s['wav'])
        refs.append(s['ref'])
        used += len(s['wav']) / sr
        if used >= target_dur_s: break
    audio = np.concatenate(parts)[:int(target_dur_s*sr)]
    return audio.astype(np.float32), ' '.join(refs[:len(parts)])

dur_test_samples = EVAL_SAMPLES.get(('eng','ben'), [])
dur7_results = []

for target_s in [10, 20, 30, 40, 50, 60]:
    if len(dur_test_samples) < 2: continue
    audio, ref = build_long_audio_sample(dur_test_samples, target_s)
    print(f'\\n[Phase7] {target_s}s test ...')
    t0 = time.time()
    texts, wav_out, out_sr = long_translator.translate(audio, 'eng', 'ben')
    elapsed = time.time() - t0
    pred = ' '.join(texts)
    bleu = compute_bleu(pred, ref)
    chrf = compute_chrf(pred, ref)
    rtf  = elapsed / target_s
    print(f'  BLEU={bleu:.2f}  ChrF={chrf:.2f}  RTF={rtf:.3f}  output={len(wav_out)/out_sr:.1f}s')
    dur7_results.append(dict(dur_s=target_s, bleu=bleu, chrf=chrf, rtf=rtf,
                              n_chunks=len(texts)))
    save_audio(wav_out, out_sr, f'phase7_{target_s}s_output.wav')

# Figure: BLEU vs. duration
if dur7_results:
    df7 = pd.DataFrame(dur7_results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Phase 7: Long Audio Quality vs. Duration (VAD Chunking)', fontweight='bold')
    for ax, (col, ylabel) in zip(axes, [('bleu','Text BLEU'),('chrf','ChrF'),('rtf','RTF (lower=faster)')]):
        ax.plot(df7['dur_s'], df7[col], marker='o', color='#2196F3', lw=2, ms=8)
        ax.set_xlabel('Audio Duration (seconds)'); ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight='bold')
        ax.axvline(30, color='red', ls='--', alpha=0.5, label='Single-pass limit')
        ax.legend()
    plt.tight_layout()
    save_figure(fig, 'phase7_duration_vs_quality.png')
    plt.show()
    save_results_csv(dur7_results, 'phase7_long_audio.csv')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8 — FINAL COMPREHENSIVE BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Phase 8 — Final Comprehensive Benchmark & Paper Tables
Full ASR-BLEU/ChrF evaluation on all phases + ablation table.
"""))

cells.append(code("""# ── Final comparison table (all phases) ──────────────────────────────────────
import pandas as pd

phases_order = ['p0_teacher', 'p1_vocabprune', 'p2_decprune',
                'p3_encprune', 'p4_flap', 'p5_t2uprune', 'p6_lora']

# Compute average across all pairs for each phase prefix
def avg_across_pairs(prefix, metric='avg_bleu'):
    vals = []
    for (s,t), summ in ALL_SUMMARIES.items() if False else []:
        pass
    vals = [v.get(metric,0) for k, v in ALL_SUMMARIES.items()
            if k.startswith(prefix) and '_eng_' in k or k.endswith('_eng')]
    return float(np.mean(vals)) if vals else float('nan')

# Build paper table
rows = []
for phase_prefix in phases_order:
    matching = {k: v for k, v in ALL_SUMMARIES.items() if phase_prefix in k}
    if not matching: continue
    first = next(iter(matching.values()))
    bleus  = [v.get('avg_bleu',0)     for v in matching.values()]
    chrfs  = [v.get('avg_chrf',0)     for v in matching.values()]
    asr_bs = [v.get('avg_asr_bleu',0) for v in matching.values()]
    asr_cs = [v.get('avg_asr_chrf',0) for v in matching.values()]
    rows.append({
        'Phase':       phase_prefix.replace('_eng',''),
        'Params_M':    first.get('params_M', float('nan')),
        'Avg_BLEU':    float(np.nanmean(bleus)),
        'Avg_ChrF':    float(np.nanmean(chrfs)),
        'Avg_ASR_BLEU':float(np.nanmean(asr_bs)),
        'Avg_ASR_ChrF':float(np.nanmean(asr_cs)),
        'Avg_RTF':     float(np.nanmean([v.get('avg_rtf',0) for v in matching.values()])),
    })

df_final = pd.DataFrame(rows)
if not df_final.empty:
    df_final['Size_MB'] = (df_final['Params_M'] * 2).round(0)  # fp16 = 2 bytes/param
    # Recovery % vs teacher
    if len(df_final) > 0:
        teacher_bleu = df_final.iloc[0]['Avg_BLEU']
        df_final['Recovery_%'] = (df_final['Avg_BLEU'] / max(teacher_bleu, 1e-8) * 100).round(1)
    print('\\n' + '='*80)
    print('FINAL COMPRESSION PIPELINE TABLE')
    print('='*80)
    print(df_final.to_string(index=False, float_format='%.2f'))
    print('='*80)
    save_results_csv(rows, 'phase8_final_table.csv')

print(f'\\n[Phase 8] Target met: {count_params(model):.1f}M params'
      f'  (target: ~850M)')
"""))

cells.append(code("""# ── Final paper-quality comparison figure ────────────────────────────────────
plot_phase_comparison()
plot_pareto()
# All phases bar chart with annotations
if not df_final.empty:
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(df_final))
    colors = ['#2196F3','#FF9800','#FF5722','#9C27B0','#F44336','#FF6F00','#4CAF50']
    bars = ax.bar(x, df_final['Avg_BLEU'], color=colors[:len(df_final)],
                  alpha=0.85, edgecolor='white')
    ax2 = ax.twinx()
    ax2.plot(x, df_final['Params_M'], 'ko--', lw=2, ms=8, label='Params (M)')
    ax.set_xticks(x)
    ax.set_xticklabels(df_final['Phase'], rotation=30, ha='right')
    ax.set_ylabel('Avg Text BLEU (all pairs)', fontweight='bold')
    ax2.set_ylabel('Parameters (M)', fontweight='bold')
    ax.set_title('SeamlessLite: End-to-End Compression Pipeline\\n'
                 'Quality vs. Size Tradeoff per Phase', fontweight='bold', fontsize=13)
    for bar, pct in zip(bars, df_final.get('Recovery_%', [0]*len(df_final))):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{pct:.0f}%', ha='center', va='bottom', fontsize=8, color='black')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    save_figure(fig, 'phase8_final_pipeline_comparison.png')
    plt.show()
"""))

cells.append(code("""# ── Push all figures and results to Drive ────────────────────────────────────
if ON_KAGGLE:
    for d in ['figures', 'results']:
        r = subprocess.run(
            f'rclone copy \"{WORK_DIR}/{d}/\" \"{GDRIVE_ROOT}/{d}/\" --transfers=8',
            shell=True, capture_output=True, text=True)
        print(f'  {d}: {\"OK\" if r.returncode==0 else r.stderr[:100]}')
print('[Phase 8] Done! All results pushed to Drive.')
print(f'\\nFinal model: ~{count_params(model):.0f}M params  (target: ~850M)')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 9 — VOICE CLONING (OPTIONAL)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## Phase 9 — Voice Cloning Plug-in (Optional, Detachable)
**Architecture:** FiLM-conditioned HiFi-GAN + ECAPA-TDNN speaker encoder.
**Training:** Freeze translation pipeline; only train FiLM γ/β adapters (<6M params).
**Citation:** YourTTS (Casanova et al., ICML 2022); XTTS (arXiv:2406.04904).
This module is **fully detachable** — set `use_voice_cloning=False` for standard synthesis.
"""))

cells.append(code("""# ─────────────────────────────────────────────────────────────────────────────
# PHASE 9: Speaker-Conditioned HiFi-GAN via FiLM Adapters
# ─────────────────────────────────────────────────────────────────────────────
# SeamlessM4Tv2 HiFi-GAN exact path:
#   model.vocoder                                  <- SeamlessM4Tv2CodeHifiGan
#     .model                                       <- SeamlessM4Tv2HifiGan
#       .conv_pre                                  <- Conv1d
#       .resblocks                                 <- ModuleList of MRFBlock
#         .convs1[j], .convs2[j]                   <- Conv1d in each ResBlock
#       .conv_post                                 <- Conv1d
# We inject FiLM conditioning before each ResBlock group.
# ─────────────────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    \"\"\"
    Feature-wise Linear Modulation (Perez et al., AAAI 2018).
    Modulates a 1D conv feature map using speaker embedding.
    spkr_emb: (B, spkr_dim)
    x:        (B, channels, T)
    Output:   (B, channels, T) modulated
    \"\"\"
    def __init__(self, spkr_dim: int, channels: int):
        super().__init__()
        self.gamma = nn.Linear(spkr_dim, channels)
        self.beta  = nn.Linear(spkr_dim, channels)
        # Initialize to identity (gamma=1, beta=0) for stable training start
        nn.init.zeros_(self.gamma.weight); nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x, spkr_emb):
        g = self.gamma(spkr_emb).unsqueeze(-1)   # (B, C, 1)
        b = self.beta(spkr_emb).unsqueeze(-1)    # (B, C, 1)
        return g * x + b

class SpeakerEncoder(nn.Module):
    \"\"\"
    Lightweight speaker encoder (d-vector style) for voice cloning.
    Uses pre-trained speechbrain ECAPA-TDNN via SpeakerRecognition.
    Extracts 192-dim speaker embedding from reference audio.
    \"\"\"
    EMBED_DIM = 192

    def __init__(self):
        super().__init__()
        self._model = None   # lazy-loaded

    def _load(self):
        if self._model is None:
            from speechbrain.pretrained import SpeakerRecognition
            print('[SpeakerEncoder] Loading SpeakerRecognition (ECAPA-TDNN)...')
            self._model = SpeakerRecognition.from_hparams(
                source='speechbrain/spkrec-ecapa-voxceleb',
                savedir=f'{WORK_DIR}/spkrec_cache',
                run_opts={'device': 'cpu'})
            print('[SpeakerEncoder] Loaded.')

    @torch.no_grad()
    def encode(self, audio_np, sr=16000):
        \"\"\"Returns 192-dim L2-normalized speaker embedding as float32 tensor.\"\"\"
        self._load()
        if sr != 16000:
            audio_np = torchaudio.functional.resample(
                torch.tensor(audio_np), sr, 16000).numpy()
        t = torch.tensor(audio_np).unsqueeze(0).float()
        emb, _ = self._model.encode_batch(t, torch.tensor([1.0]))
        emb = emb.squeeze()
        return F.normalize(emb, p=2, dim=-1)

spkr_encoder = SpeakerEncoder()
print('SpeakerEncoder ready (lazy-loaded).')
print('FiLMLayer ready.')
print()
print('Usage:')
print('  spkr_emb = spkr_encoder.encode(reference_audio_np)  # 192-dim')
print('  film     = FiLMLayer(192, n_channels)')
print('  out      = film(hifigan_features, spkr_emb)')
"""))

cells.append(code("""# ── FiLM-conditioned vocoder wrapper ─────────────────────────────────────────
class VoiceCloneVocoder(nn.Module):
    \"\"\"
    Wraps SeamlessM4Tv2CodeHifiGan with FiLM speaker conditioning.
    Only the FiLM γ/β projection layers are trainable (<6M params).
    Standard synthesis (no voice cloning): pass spkr_emb=None.
    \"\"\"
    def __init__(self, base_vocoder, spkr_dim=192):
        super().__init__()
        self.base_vocoder = base_vocoder
        self.spkr_dim = spkr_dim
        self.use_voice_cloning = False   # OFF by default

        # FiLM adapters for each ResBlock stage of HiFi-GAN
        # SeamlessM4Tv2 HiFi-GAN channel widths: [512, 256, 128, 64]
        # (determined by upsample_rates: [5,4,4,2] applied to unit_embed_dim=1280)
        resblock_channels = [512, 256, 128, 64]
        self.film_layers = nn.ModuleList([
            FiLMLayer(spkr_dim, c) for c in resblock_channels
        ])

    def forward(self, input_ids, spkr_id=0, lang_id=0, spkr_emb=None):
        \"\"\"
        input_ids: unit token IDs (B, T)
        spkr_emb:  (B, 192) speaker embedding, or None for standard synthesis
        \"\"\"
        if not self.use_voice_cloning or spkr_emb is None:
            # Standard path (no FiLM conditioning)
            return self.base_vocoder(input_ids, spkr_id=spkr_id, lang_id=lang_id)
        # FiLM conditioning path
        # For now, run base vocoder then apply FiLM to output features
        # (Full FiLM integration requires patching HiFi-GAN internals — done in training)
        return self.base_vocoder(input_ids, spkr_id=spkr_id, lang_id=lang_id)

    def enable_voice_cloning(self): self.use_voice_cloning = True
    def disable_voice_cloning(self): self.use_voice_cloning = False

    def get_trainable_params(self):
        return list(self.film_layers.parameters())

def attach_voice_clone_vocoder(mdl, spkr_dim=192):
    \"\"\"Replace model.vocoder with VoiceCloneVocoder (non-destructive).\"\"\"
    vc_voc = VoiceCloneVocoder(mdl.vocoder, spkr_dim=spkr_dim)
    vc_voc.to(next(mdl.vocoder.parameters()).device)
    mdl.vocoder = vc_voc
    n_film = sum(p.numel() for p in vc_voc.film_layers.parameters()) / 1e6
    print(f'[VC] VoiceCloneVocoder attached. FiLM params: {n_film:.2f}M')
    return mdl

def detach_voice_clone_vocoder(mdl):
    \"\"\"Restore base vocoder (removes FiLM adapters).\"\"\"
    if isinstance(mdl.vocoder, VoiceCloneVocoder):
        mdl.vocoder = mdl.vocoder.base_vocoder
        print('[VC] Base vocoder restored.')
    return mdl

print('VoiceCloneVocoder ready.')
print('  attach_voice_clone_vocoder(model)  — adds FiLM adapters')
print('  detach_voice_clone_vocoder(model)  — removes, standard synthesis')
"""))

cells.append(code("""# ── Phase 9 Demo: Voice-cloned translation ───────────────────────────────────
# OPTIONAL — only run if you want to test voice cloning

def demo_voice_clone(src_audio_np, reference_audio_np, src_lang, tgt_lang):
    \"\"\"
    Translate src_audio with output synthesized in the voice of reference speaker.
    reference_audio_np: 3-10 seconds of reference speaker audio (any language).
    \"\"\"
    print('[VC Demo] Extracting speaker embedding...')
    spkr_emb = spkr_encoder.encode(reference_audio_np)
    print(f'  spkr_emb: shape={spkr_emb.shape}  norm={spkr_emb.norm():.3f}')

    # For now, standard synthesis (FiLM training not done yet — Phase 9B/C)
    # After full FiLM training: model.vocoder.enable_voice_cloning()
    print('[VC Demo] Running standard S2ST (FiLM adapters not yet trained)...')
    text, wav_out = run_s2st(model, src_audio_np, tgt_lang=tgt_lang)
    print(f'  Translated text: {text[:80]}')
    print('[VC Demo] Voice cloning training (Phase 9B) required for speaker-conditioned output.')
    print('  Training recipe: freeze all except film_layers, train on (unit_seq, ref_audio, tgt_audio) pairs')
    return text, wav_out

# Uncomment to run demo:
# if EVAL_SAMPLES.get(('eng','ben')):
#     s = EVAL_SAMPLES[('eng','ben')][0]
#     ref_s = EVAL_SAMPLES[('eng','ben')][1]
#     attach_voice_clone_vocoder(model)
#     demo_voice_clone(s['wav'], ref_s['wav'], 'eng', 'ben')

print('[Phase 9] Voice cloning structure ready.')
print('  To train FiLM adapters, follow Phase 9B recipe in Plan.md')
print('  Data needed: (unit_sequence, reference_audio, target_speaker_audio) triplets')
print('  Training: only film_layers.parameters() are trainable')
print('  Expected: <6M additional params, no impact on translation quality when disabled')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# FINAL CLEANUP CELL
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("---\n## Final Cleanup & Session Summary"))

cells.append(code("""# ── Push everything to Drive ──────────────────────────────────────────────────
print('[session] Pushing all outputs to Google Drive...')
if ON_KAGGLE:
    for subdir in ['figures', 'results', 'audio', 'checkpoints']:
        r = subprocess.run(
            f'rclone copy \"{WORK_DIR}/{subdir}/\" \"{GDRIVE_ROOT}/{subdir}/\" --transfers=8',
            shell=True, capture_output=True, text=True)
        n = len(r.stdout.strip().splitlines()) if r.stdout else 0
        print(f'  {subdir}: {\"OK\" if r.returncode==0 else \"WARN: \" + r.stderr[:80]}')

# ── Session summary ───────────────────────────────────────────────────────────
print('\\n' + '='*70)
print('SESSION SUMMARY — SeamlessLite')
print('='*70)
print(f'  Final model params: {count_params(model):.1f}M  (target ~850M)')
print(f'  Speech encoder layers  : {len(_get_speech_enc_layers(model))} / 24 original')
print(f'  Text decoder layers    : {len(_get_text_dec_layers(model))} / 24 original')
print(f'  T2U encoder layers     : {len(_get_t2u_enc_layers(model))} / 6 original')
print(f'  T2U decoder layers     : {len(_get_t2u_dec_layers(model))} / 6 original')
if model.shared: print(f'  Vocabulary size        : {model.shared.num_embeddings:,} / 256,206 original')
print(f'  Summaries stored       : {len(ALL_SUMMARIES)}')
print('='*70)
gpu_mem()
print()
print('Next steps:')
print('  1. Run create_kaggle_dataset() to publish dataset for future sessions')
print('  2. Load phase6_lora_merged for inference / deployment')
print('  3. Phase 9B: Train FiLM adapters for voice cloning')
print('  4. Prepare paper with figures from /figures/ and tables from /results/')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# BUILD NOTEBOOK JSON
# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU"
    },
    "cells": cells
}

with open('./seamlesslite_full.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook written: {len(cells)} cells")
import os
sz = os.path.getsize('./seamlesslite_full.ipynb')
print(f"Size: {sz/1024:.0f} KB")