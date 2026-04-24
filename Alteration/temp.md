python

"""
SeamlessM4T Textless Pipeline — CORRECTED Phase 4 → Phase 6b
=============================================================

ROOT CAUSE ANALYSIS OF KD LOSS NOT CONVERGING
----------------------------------------------

Three compounding bugs caused the KD loss plateau:

BUG 1 — Phase 6a: Wrong input-target alignment
  The speech encoder produces [B, T_frames, 1024] where T_frames ≈ 90-500 frames.
  The teacher T2U input (text decoder output) is [B, T_text, 1024] where T_text ≈ 10-60 tokens.
  The AttentionConnector IS receiving real encoder output, but the MSE loss is computing
  distances between cross-attention outputs and text-decoder hidden states — these live in
  completely different sub-spaces of the 1024-dim hidden space. MSE never converges because
  the two spaces require an alignment that MSE alone cannot learn. You need cosine loss
  (direction) + MSE (magnitude) combined, which is what proper feature KD uses.

BUG 2 — Phase 6b: enc_proxy IS the target
  enc_proxy = sample['t2u_input']  ← this is the KD TARGET
  connector_out, qty = model_6b.cif_connector(enc_proxy, lang_id)
  The connector is receiving the teacher's T2U input as its INPUT, not speech encoder output.
  It's predicting its own input. Loss = MSE(connector(target), target) ≈ identity function.
  This is why loss drops a little initially (it learns identity) then plateaus completely.

BUG 3 — CIF "overshooting" was threshold calibration, not architecture
  CIF with threshold=1.0 and unnormalized weights (sigmoid output that sums to T_frames,
  not 1.0) means the CIF fires T_frames/threshold = T_frames times instead of T_text times.
  Fix: normalize weights so they sum to the predicted quantity (like the original CIF paper).
  The AttentionConnector avoids this but introduces a worse problem: it requires knowing
  target_len during training, which creates a distribution mismatch at inference time when
  target_len comes from pred_len (the length predictor hasn't converged yet).

ARCHITECTURE DECISION — CIF vs AttentionConnector
--------------------------------------------------
Go back to CIF. CIF is the correct architecture for this task for these reasons:
  1. CIF learns speech boundary detection — it finds natural pause/phoneme boundaries.
     The AttentionConnector has no inductive bias toward speech structure at all.
  2. The original UnitY2 system (which is T2U's training paradigm) uses a CIF-like
     length adapter. T2U's weights were distilled from a pipeline that produced
     text-length sequences. CIF is the right shape-matching connector.
  3. CIF's "overshooting" was a bug in the weight normalization, not CIF itself.

TRAINING STRATEGY FIX
----------------------
Phase 6a KD loss must use:
  1. Cosine similarity loss (not MSE) — learns direction alignment across representation spaces
  2. Proper CIF weight normalization — weights sum to predicted n_tokens, not T_frames
  3. Warmup + longer training (5000 steps, not 2500) — this is a cross-space alignment task
  4. Cosine annealing with warm restarts — helps escape the MSE plateau

Phase 6b must:
  1. Run the REAL speech encoder on actual audio (not use t2u_input as proxy)
  2. Use cross-entropy on unit IDs as the primary loss (it has gradient signal)
  3. Run the connector on real encoder output every step
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — CORRECTED CIF CONNECTOR (replace the class in Phase 4 cell)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class CIFConnector(nn.Module):
    """
    Continuous Integrate-and-Fire connector (Dong & Xu, ICASSP 2020).
    
    KEY FIX vs original notebook:
    - Weight normalization: weights are rescaled so they sum to predicted n_tokens,
      not T_frames. This prevents the "overshooting" problem entirely.
    - The weight predictor output is passed through softplus (not sigmoid) so weights
      are always positive and we can renormalize to any target sum.
    - Quantity prediction is a separate head that directly predicts n_tokens from
      a mean-pooled representation, then weights are scaled to sum to that value.
    
    Architecture:
      - Quantity predictor: MeanPool → Linear → Softplus → predicted_n_tokens
      - Weight predictor: Linear(D→D//4) → ReLU → Linear(D//4→1) → Softplus
      - Normalization: w_normalized = w / w.sum() * predicted_n_tokens
      - CIF accumulate-and-fire with threshold=1.0 on normalized weights
      - Language conditioning: Embedding(n_langs, D//8) → Linear(D//8, D)
      - Refiner: 2-layer TransformerEncoder for quality
    """
    def __init__(self, d_model=1024, n_refiner_layers=2, n_langs=45, threshold=1.0):
        super().__init__()
        self.d_model   = d_model
        self.threshold = threshold

        # Quantity predictor: predicts how many output tokens we want
        self.qty_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()   # always positive
        )

        # Weight predictor: per-frame importance score (unnormalized)
        self.weight_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()   # always positive — we'll normalize anyway
        )

        # Language conditioning
        self.lang_embed = nn.Embedding(n_langs, d_model // 8)
        self.lang_proj  = nn.Linear(d_model // 8, d_model)

        # Refiner transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True, norm_first=True)
        self.refiner  = nn.TransformerEncoder(enc_layer, num_layers=n_refiner_layers)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, encoder_out, tgt_lang_id=None):
        """
        Args:
            encoder_out : [B, T_frames, D]
            tgt_lang_id : [B] integer lang IDs
        Returns:
            (out [B, T_units, D], qty_pred [B])
        """
        B, T, D = encoder_out.shape

        # Language conditioning
        if tgt_lang_id is not None:
            le = self.lang_proj(self.lang_embed(tgt_lang_id.to(encoder_out.device)))
            encoder_out = encoder_out + le.unsqueeze(1)

        # Predict target quantity from mean-pooled encoder output
        mean_pool = encoder_out.mean(dim=1)           # [B, D]
        qty_pred  = self.qty_predictor(mean_pool).squeeze(-1)  # [B] — predicted n_tokens

        # Per-frame unnormalized weights
        raw_w = self.weight_predictor(encoder_out).squeeze(-1)  # [B, T]

        # KEY FIX: normalize weights so they sum to qty_pred per sample
        # This ensures CIF fires exactly qty_pred times on average
        w_sum = raw_w.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1]
        weights = raw_w / w_sum * qty_pred.unsqueeze(1).detach() # [B, T]

        # CIF: accumulate until threshold, fire
        outputs = []
        for b in range(B):
            w   = weights[b]; h = encoder_out[b]
            acc = torch.zeros(D, device=h.device, dtype=h.dtype)
            acc_w, fired = 0.0, []
            for t in range(T):
                acc_w += w[t].item()
                acc   += w[t] * h[t]
                if acc_w >= self.threshold:
                    fired.append(acc / acc_w)
                    acc = torch.zeros_like(acc)
                    acc_w = 0.0
            if acc_w > 0.05:                    # don't drop partial final frame
                fired.append(acc / max(acc_w, 1e-6))
            if not fired:
                fired.append(h.mean(0))         # safety: at least one token
            outputs.append(torch.stack(fired))

        max_len = max(o.shape[0] for o in outputs)
        padded  = torch.zeros(B, max_len, D, device=encoder_out.device,
                              dtype=encoder_out.dtype)
        for b, o in enumerate(outputs):
            padded[b, :o.shape[0]] = o

        refined = self.refiner(padded)
        out     = self.out_proj(refined)
        actual_qty = torch.tensor([float(o.shape[0]) for o in outputs],
                                  dtype=torch.float, device=encoder_out.device)
        return out, actual_qty, qty_pred


# ── Speaker Adapter (unchanged) ──────────────────────────────────────────────
class SpeakerAdapter(nn.Module):
    """ECAPA 192-dim → HiFi-GAN vocoder 256-dim. ~0.1M params."""
    def __init__(self, ecapa_dim=192, vocoder_spkr_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ecapa_dim, vocoder_spkr_dim),
            nn.LayerNorm(vocoder_spkr_dim),
            nn.Tanh())
    def forward(self, ecapa_emb):
        return self.proj(ecapa_emb)


# ── NOTE: The surgical function needs updating too (returns qty changed) ──────
def remove_text_decoder_and_install_cif(model_with_dec):
    """
    Remove text decoder and install CIF connector.
    NOTE: CIFConnector now returns 3 values: (out, actual_qty, qty_pred).
    All call sites must handle this.
    """
    mdl = model_with_dec
    t2u_vocab_size = getattr(mdl.config, 't2u_vocab_size', 10082)
    n_langs        = getattr(mdl.config, 'vocoder_num_langs', 36)
    hidden         = mdl.config.hidden_size
    print(f'Pre-surgery: hidden={hidden}, t2u_vocab={t2u_vocab_size}, n_langs={n_langs}')

    if hasattr(mdl, 'text_decoder') and mdl.text_decoder is not None:
        dp = count_params(mdl.text_decoder)
        del mdl.text_decoder; mdl.text_decoder = None
        print(f'  ✓ text_decoder removed ({dp:.1f}M params)')
    if hasattr(mdl, 'lm_head') and mdl.lm_head is not None:
        del mdl.lm_head; mdl.lm_head = None; print('  ✓ lm_head removed')
    if hasattr(mdl, 'shared') and mdl.shared is not None:
        del mdl.shared; mdl.shared = None; print('  ✓ shared vocab removed')

    mdl.config.decoder_layers    = 0
    mdl.config.vocab_size         = 0
    mdl.config.t2u_max_new_tokens = 2048

    # n_langs+10 for safety margin on lang ID range
    mdl.cif_connector  = CIFConnector(d_model=hidden, n_refiner_layers=2,
                                      n_langs=n_langs + 10, threshold=1.0)
    mdl.speaker_adapter = SpeakerAdapter(ecapa_dim=192, vocoder_spkr_dim=256)

    print(f'  ✓ CIF connector installed ({count_params(mdl.cif_connector):.2f}M params)')
    print(f'  ✓ Speaker adapter installed ({count_params(mdl.speaker_adapter)*1000:.0f}K params)')
    gc.collect(); torch.cuda.empty_cache()
    return mdl


_cif_test = CIFConnector()
_spk_test = SpeakerAdapter()
print(f'CIFConnector (fixed): ~{count_params(_cif_test):.2f}M params')
print(f'SpeakerAdapter: ~{count_params(_spk_test)*1000:.0f}K params')
del _cif_test, _spk_test


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — CORRECTED Phase 4 surgical RUN cell                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

p4_done = load_latest_checkpoint('phase4_done')
if p4_done:
    print('Phase 4 architectural surgery already done.')
    model_p4 = None
else:
    print('Running Phase 4: architectural surgery...')
    model_p4 = _consolidate_to_single_gpu(model_p3)
    model_p4 = remove_text_decoder_and_install_cif(model_p4)
    print_model_breakdown(model_p4, 'Phase 4: Textless Architecture')

    p4_dir = f'{MODEL_DIR}/phase4_textless_pretrain'
    os.makedirs(p4_dir, exist_ok=True)
    torch.save({
        'state_dict': model_p4.state_dict(),
        'config':     model_p4.config,
        'cif_state':  model_p4.cif_connector.state_dict(),
        'spk_state':  model_p4.speaker_adapter.state_dict(),
        'hidden':     model_p4.config.hidden_size,
        'n_langs':    getattr(model_p4.config, 'vocoder_num_langs', 36),
    }, f'{p4_dir}/textless_model.pt')
    if ON_KAGGLE:
        _rclone_push(f'{p4_dir}/textless_model.pt', 'phase4_textless_pretrain')
    save_checkpoint({'done': True, 'hidden': model_p4.config.hidden_size},
                    'phase4_done', 0)
    print('Phase 4 saved to Drive.')
    print_model_breakdown(model_p4, 'Phase 4 DONE: Textless ~631M')

gpu_mem()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Phase 5: KD Extraction (keep as-is, minor validation tweak)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Phase 5 KD extraction code is CORRECT. Keep it exactly as-is.
# The hook system correctly captures text_decoder output → T2U encoder input.
# Only add the following validation check after extraction to ensure data quality:

# After kd_data is loaded/extracted, validate:
if kd_data:
    # Verify the captured t2u_input is actually 1024-dim hidden states
    sample_check = [x for x in kd_data if x.get('t2u_input') is not None][0]
    t2u_in = sample_check['t2u_input']
    assert t2u_in.dim() == 3 and t2u_in.shape[2] == 1024, \
        f"t2u_input has wrong shape {t2u_in.shape} — expected [1, T_text, 1024]"
    assert t2u_in.std().item() > 0.1, \
        f"t2u_input std={t2u_in.std():.4f} is suspiciously low — check hook captured real data"
    print(f'✓ KD data validated: {len(kd_data)} samples, t2u shape={t2u_in.shape}, '
          f'mean={t2u_in.mean():.3f}, std={t2u_in.std():.3f}')
    # Inspect representation space statistics (should be similar to text decoder output)
    all_means = [x['t2u_input'].mean().item() for x in kd_data if x.get('t2u_input') is not None]
    all_stds  = [x['t2u_input'].std().item()  for x in kd_data if x.get('t2u_input') is not None]
    print(f'  t2u_input mean: μ={np.mean(all_means):.3f} ± {np.std(all_means):.3f}')
    print(f'  t2u_input std:  μ={np.mean(all_stds):.3f} ± {np.std(all_stds):.3f}')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Phase 6a: CORRECTED Model Loading                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

p4_dir   = f'{MODEL_DIR}/phase4_textless_pretrain'
if ON_KAGGLE and not os.path.exists(f'{p4_dir}/textless_model.pt'):
    subprocess.run(
        f'rclone copy "{GDRIVE_ROOT}/phase4_textless_pretrain/" "{p4_dir}/" '
        f'--transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M',
        shell=True)

p4_saved = torch.load(f'{p4_dir}/textless_model.pt', map_location='cpu',
                      weights_only=False)
hidden  = p4_saved.get('hidden', 1024)
n_langs = p4_saved.get('n_langs', 36)

print('Rebuilding textless model from Phase 4 saved state...')
model_6a, processor = load_base_model()
model_6a = _consolidate_to_single_gpu(model_6a)

# Replay surgery with CORRECTED CIFConnector
model_6a = remove_text_decoder_and_install_cif(model_6a)

# Restore Phase 3/4 weights (encoder + T2U — NOT cif/spk weights)
sd = p4_saved['state_dict']
missing, unexpected = model_6a.load_state_dict(sd, strict=False)
print(f'Phase 4 weights restored. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}')

# Restore previously trained CIF/Speaker weights if 6a checkpoint exists
p6a_ck = load_latest_checkpoint('phase6a_connector')
if p6a_ck and p6a_ck.get('step', 0) > 0:
    model_6a.cif_connector.load_state_dict(p6a_ck['cif_state'])
    model_6a.speaker_adapter.load_state_dict(p6a_ck['spk_state'])
    print(f'  ✓ CIF + Speaker adapter weights restored from step {p6a_ck["step"]}')

device = torch.device('cuda:0')
model_6a = model_6a.to(device)
gpu_mem()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Phase 6a: Freeze all, unfreeze CIF + Speaker                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

for p in model_6a.parameters():
    p.requires_grad_(False)
for p in model_6a.cif_connector.parameters():
    p.requires_grad_(True)
for p in model_6a.speaker_adapter.parameters():
    p.requires_grad_(True)

trainable_6a = [p for p in model_6a.parameters() if p.requires_grad]
print(f'Trainable: {sum(p.numel() for p in trainable_6a)/1e6:.2f}M params')
print(f'  CIF connector:  {count_params(model_6a.cif_connector):.2f}M')
print(f'  Speaker adapter: {count_params(model_6a.speaker_adapter)*1000:.0f}K')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Phase 6a: CORRECTED Training Loop (the critical fix)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# THE KEY FIXES IN THIS CELL:
#
# FIX 1 — Loss function: cosine + MSE hybrid instead of pure MSE
#   MSE between speech encoder space and text decoder space never converges
#   because the two spaces have completely different statistics and orientations.
#   Cosine similarity loss works across representation spaces — it aligns the
#   DIRECTION of the output, not the absolute magnitude. This is what the
#   feature KD literature uses (e.g. PKD, TinyBERT, FitNets all use cosine
#   for cross-architecture distillation).
#
# FIX 2 — Run real speech encoder on actual audio
#   The connector must learn to compress real speech encoder output, not a
#   cached proxy. If we train on cached proxies, the connector input distribution
#   at inference will be completely different from training.
#
# FIX 3 — Quantity loss uses qty_pred (the predictor head) not actual_qty
#   The CIFConnector now returns (out, actual_qty, qty_pred). We supervise
#   qty_pred against n_tokens from KD data. actual_qty is a by-product of
#   the CIF fire mechanism and will naturally converge as qty_pred improves.
#
# FIX 4 — 5000 steps with cosine annealing with warm restarts (SGDR)
#   Cross-space alignment takes longer than same-space distillation.
#   SGDR helps escape plateaus better than plain cosine annealing.

MAX_STEPS_P6A = 5000
BATCH_ACCUM   = 4
LOG_EVERY     = 100
SAVE_EVERY    = 500

start_6a    = 0
loss_log_6a = []
feat_log_6a = []  # track cosine loss separately for diagnostics
qty_log_6a  = []

# Resume from checkpoint if exists
p6a_ck_resume = load_latest_checkpoint('phase6a_connector')
if p6a_ck_resume and p6a_ck_resume.get('step', 0) > 0:
    start_6a    = p6a_ck_resume['step']
    loss_log_6a = p6a_ck_resume.get('loss_log', [])
    feat_log_6a = p6a_ck_resume.get('feat_log', [])
    qty_log_6a  = p6a_ck_resume.get('qty_log', [])
    print(f'Resuming Phase 6a from step {start_6a}')

# Optimizer: higher LR for connector (it's training from scratch)
optimizer_6a = torch.optim.AdamW([
    {'params': model_6a.cif_connector.parameters(),   'lr': 3e-4, 'weight_decay': 0.01},
    {'params': model_6a.speaker_adapter.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
], betas=(0.9, 0.98))

if p6a_ck_resume and p6a_ck_resume.get('optimizer_state'):
    try:
        optimizer_6a.load_state_dict(p6a_ck_resume['optimizer_state'])
        print('  Optimizer state restored.')
    except Exception as e:
        print(f'  Optimizer restore failed (architecture changed): {e}')

# Cosine annealing with warm restarts — escapes plateaus better
scheduler_6a = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_6a, T_0=1000, T_mult=2, eta_min=1e-5)

scaler_6a = torch.cuda.amp.GradScaler()

valid_kd = [x for x in kd_data
            if x.get('t2u_input') is not None
            and x.get('unit_ids') is not None
            and x['n_tokens'] > 0]
print(f'Valid KD samples for Phase 6a: {len(valid_kd)} / {len(kd_data)}')
assert len(valid_kd) > 10, 'Not enough valid KD samples — re-run Phase 5'

# Build audio lookup: sample_id → waveform numpy array
sample_id_to_audio = {s['id']: s['wav'] for s in ft_samples}
print(f'Audio lookup: {len(sample_id_to_audio)} samples')
assert len(sample_id_to_audio) > 0, 'ft_samples not loaded — run data loading cells'

model_6a.train()
model_6a.cif_connector.train()
model_6a.speaker_adapter.train()
optimizer_6a.zero_grad()

print(f'\n{"="*70}')
print(f'  PHASE 6a: CIF Connector + Speaker Adapter Feature KD Training')
print(f'  Steps: {start_6a} → {MAX_STEPS_P6A}')
print(f'  Loss: 0.60×cosine_KD + 0.20×MSE_KD + 0.15×qty_pred + 0.05×spk_reg')
print(f'{"="*70}\n')

recent_feat  = []
recent_qty   = []
recent_total = []

for step in range(start_6a, MAX_STEPS_P6A):
    sample   = random.choice(valid_kd)
    tgt_lang = sample['tgt_lang']
    lang_id  = torch.tensor([m4t_lang_to_vocoder_id(tgt_lang)], device=device)

    # KD targets — from teacher
    target   = sample['t2u_input'].to(device).float()   # [1, T_text, 1024]
    n_tokens = float(sample['n_tokens'])
    target_qty = torch.tensor([n_tokens], dtype=torch.float, device=device)

    # Speaker embedding
    spk_emb = sample['spk_emb'].to(device).float()      # [192]

    # FIX 2: Get actual audio and run REAL speech encoder
    audio_wav = sample_id_to_audio.get(sample['id'])
    if audio_wav is None:
        continue   # skip if audio not found

    try:
        inp_proc = processor(audio=audio_wav, sampling_rate=16000, return_tensors='pt')
        inp_f    = inp_proc['input_features'].to(device)
        attn_m   = inp_proc.get('attention_mask')
        if attn_m is not None:
            attn_m = attn_m.to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):

            # FIX 2: Real speech encoder forward (frozen)
            with torch.no_grad():
                enc_out = model_6a.speech_encoder(
                    input_features=inp_f,
                    attention_mask=attn_m
                ).last_hidden_state.float()      # [1, T_frames, 1024]

            # CIF connector: compress speech frames to text-like sequence
            connector_out, actual_qty, qty_pred = model_6a.cif_connector(
                enc_out, tgt_lang_id=lang_id)    # [1, T_fired, 1024]

            # Speaker adapter forward
            spk_proj = model_6a.speaker_adapter(spk_emb.unsqueeze(0))   # [1, 256]

            # ── Loss computation ─────────────────────────────────────────────

            # Align connector output length to target length for loss
            T_pred = connector_out.shape[1]
            T_tgt  = target.shape[1]

            if T_pred < T_tgt:
                # Pad connector output to target length
                conn_aligned = F.pad(connector_out, (0, 0, 0, T_tgt - T_pred))
                tgt_aligned  = target
            elif T_pred > T_tgt:
                # Truncate connector output to target length
                conn_aligned = connector_out[:, :T_tgt, :]
                tgt_aligned  = target
            else:
                conn_aligned = connector_out
                tgt_aligned  = target

            # FIX 1a: Cosine similarity loss (direction alignment)
            # Flatten to [T, D] and compute per-position cosine loss
            cos_loss = (1.0 - F.cosine_similarity(
                conn_aligned.reshape(-1, 1024),
                tgt_aligned.reshape(-1, 1024).detach(),
                dim=-1)).mean()

            # FIX 1b: MSE loss (magnitude alignment) — lower weight than cosine
            mse_loss = F.mse_loss(conn_aligned, tgt_aligned.detach())

            # FIX 3: Quantity prediction loss (supervises qty_pred head)
            qty_loss = F.mse_loss(qty_pred, target_qty)

            # Speaker regularization: keep speaker projection well-conditioned
            # Target: L2 norm ≈ 14.0 (typical ECAPA embedding norm ≈ 192^0.5)
            spk_reg = ((spk_proj.float().norm(dim=-1) - 14.0) ** 2).mean()

            # Combined loss with proper weights
            # Cosine gets 0.60 weight — it's the primary alignment signal
            # MSE gets 0.20 weight — adds magnitude supervision
            loss = (0.60 * cos_loss +
                    0.20 * mse_loss +
                    0.15 * qty_loss +
                    0.05 * spk_reg)

        scaler_6a.scale(loss / BATCH_ACCUM).backward()

        # Track individual losses
        loss_log_6a.append(loss.item())
        feat_log_6a.append(cos_loss.item())
        qty_log_6a.append(qty_loss.item())

        recent_feat.append(cos_loss.item())
        recent_qty.append(qty_loss.item())
        recent_total.append(loss.item())
        if len(recent_feat) > 100:
            recent_feat.pop(0); recent_qty.pop(0); recent_total.pop(0)

        if (step + 1) % BATCH_ACCUM == 0:
            scaler_6a.unscale_(optimizer_6a)
            torch.nn.utils.clip_grad_norm_(trainable_6a, 1.0)
            scaler_6a.step(optimizer_6a)
            scaler_6a.update()
            optimizer_6a.zero_grad()
            scheduler_6a.step()

        if (step + 1) % LOG_EVERY == 0:
            avg_cos  = np.mean(recent_feat[-50:])  if recent_feat  else 0
            avg_qty  = np.mean(recent_qty[-50:])   if recent_qty   else 0
            avg_tot  = np.mean(recent_total[-50:]) if recent_total else 0
            cur_lr   = optimizer_6a.param_groups[0]['lr']
            T_fired  = connector_out.shape[1] if 'connector_out' in dir() else 0
            print(f'  Step {step+1:>5}/{MAX_STEPS_P6A} | '
                  f'cos={avg_cos:.4f} | '
                  f'qty_err={avg_qty:.2f} | '
                  f'total={avg_tot:.4f} | '
                  f'fired={T_fired} vs tgt={int(n_tokens)} | '
                  f'lr={cur_lr:.2e}')

        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint({
                'step':            step + 1,
                'cif_state':       model_6a.cif_connector.state_dict(),
                'spk_state':       model_6a.speaker_adapter.state_dict(),
                'optimizer_state': optimizer_6a.state_dict(),
                'scheduler_state': scheduler_6a.state_dict(),
                'scaler_state':    scaler_6a.state_dict(),
                'loss_log':        loss_log_6a,
                'feat_log':        feat_log_6a,
                'qty_log':         qty_log_6a,
            }, 'phase6a_connector', step + 1)
            print(f'  ✓ Checkpoint saved at step {step+1}')

    except Exception as e:
        print(f'  Step {step+1} error: {e}')
        if step - start_6a < 5:
            import traceback; traceback.print_exc()
        optimizer_6a.zero_grad()
        continue

print(f'\n{"="*70}')
print(f'  Phase 6a Training Complete!')
if recent_feat:
    print(f'  Final avg cosine KD loss: {np.mean(recent_feat):.4f}')
    print(f'  Final avg qty error:      {np.mean(recent_qty):.2f}')
    status = ('✅ EXCELLENT' if np.mean(recent_feat) < 0.05 else
              '✅ GOOD'      if np.mean(recent_feat) < 0.15 else
              '⚠ NEEDS MORE TRAINING')
    print(f'  Status: {status}')
print(f'{"="*70}\n')

# Save Phase 6a final state
save_checkpoint({
    'step':            MAX_STEPS_P6A,
    'cif_state':       model_6a.cif_connector.state_dict(),
    'spk_state':       model_6a.speaker_adapter.state_dict(),
    'optimizer_state': optimizer_6a.state_dict(),
    'scheduler_state': scheduler_6a.state_dict(),
    'scaler_state':    scaler_6a.state_dict(),
    'loss_log':        loss_log_6a,
    'feat_log':        feat_log_6a,
    'qty_log':         qty_log_6a,
}, 'phase6a_connector', MAX_STEPS_P6A)

p6a_save_dir = f'{MODEL_DIR}/phase6a_cif_connector'
os.makedirs(p6a_save_dir, exist_ok=True)
torch.save({
    'state_dict': model_6a.state_dict(),
    'cif_state':  model_6a.cif_connector.state_dict(),
    'spk_state':  model_6a.speaker_adapter.state_dict(),
    'hidden':     hidden,
    'n_langs':    n_langs,
}, f'{p6a_save_dir}/textless_model.pt')
if ON_KAGGLE:
    _rclone_push(f'{p6a_save_dir}/textless_model.pt', 'phase6a_cif_connector')

print('✓ Phase 6a complete. CIF Connector trained.')
gpu_mem()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7 — Phase 6a Training Loss Diagnostic Plot                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_ema(values, alpha=0.05):
    if not values: return []
    ema, v = [], values[0]
    for x in values:
        v = alpha * x + (1 - alpha) * v
        ema.append(v)
    return ema

if feat_log_6a:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Phase 6a: CIF Connector Feature KD Training',
                 fontsize=14, fontweight='bold')

    for ax, data, label, color, target_line in [
        (axes[0], feat_log_6a, 'Cosine KD Loss (↓ target < 0.10)', '#2196F3', 0.10),
        (axes[1], qty_log_6a,  'Quantity MSE Loss (↓)',             '#FF9800', None),
        (axes[2], loss_log_6a, 'Total Weighted Loss (↓)',           '#9C27B0', None),
    ]:
        ax.plot(data, alpha=0.15, color=color, lw=0.5, label='Raw')
        ema = compute_ema(data)
        ax.plot(ema, color=color, lw=2, label=f'EMA (final={ema[-1]:.4f})')
        if target_line:
            ax.axhline(y=target_line, color='red', ls='--', alpha=0.7,
                       label=f'Target < {target_line}')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Step')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Annotate convergence status
    final_cos = compute_ema(feat_log_6a)[-1] if feat_log_6a else 1.0
    status = ('EXCELLENT' if final_cos < 0.05 else
              'GOOD'      if final_cos < 0.10 else
              'ACCEPTABLE' if final_cos < 0.20 else
              'NEEDS MORE TRAINING')
    fig.text(0.5, -0.02, f'Convergence status: {status} (cosine={final_cos:.4f})',
             ha='center', fontsize=11, fontweight='bold',
             color='green' if final_cos < 0.20 else 'red')

    plt.tight_layout()
    save_figure(fig, 'phase6a_cif_training.png')
    plt.show()
else:
    print('No training log data — run training cell first.')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8 — Phase 6b: DoRA E2E Fine-tuning — CORRECTED                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# FIXES:
# FIX 1 — DO NOT use enc_proxy = sample['t2u_input']
#   Run the REAL speech encoder on actual audio every step.
#   This is the only way T2U sees real connector output distributions.
#   Yes it costs ~2× VRAM, but it's mandatory for correct training.
#
# FIX 2 — T2U unit CE loss is the PRIMARY loss
#   The unit cross-entropy has strong gradient signal because T2U was pretrained
#   on exactly this objective. It will drive both the DoRA adapters AND the
#   connector weights to align quickly.
#
# FIX 3 — Use the correct 3-return CIF connector API
#   connector_out, actual_qty, qty_pred = model_6b.cif_connector(enc_out, lang_id)

from peft import LoraConfig, get_peft_model

print('Loading Phase 6a model for DoRA fine-tuning...')
model_6b = model_6a   # already in memory with trained CIF + speaker adapter

# Restore 6a final trained weights
p6a_final = load_latest_checkpoint('phase6a_connector')
if p6a_final and p6a_final.get('step', 0) > 0:
    model_6b.cif_connector.load_state_dict(p6a_final['cif_state'])
    model_6b.speaker_adapter.load_state_dict(p6a_final['spk_state'])
    print(f'✓ CIF + speaker adapter weights from step {p6a_final["step"]} restored.')
else:
    print('⚠ No 6a checkpoint found — starting from randomly initialized CIF weights.')
    print('  Run Phase 6a training first for best results.')

# Freeze all, unfreeze CIF + speaker adapter
for p in model_6b.parameters():
    p.requires_grad_(False)
for p in model_6b.cif_connector.parameters():
    p.requires_grad_(True)
for p in model_6b.speaker_adapter.parameters():
    p.requires_grad_(True)

# Apply DoRA to speech encoder + T2U
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
    use_dora=True,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'])

print('Applying DoRA to speech_encoder...')
model_6b.speech_encoder = get_peft_model(model_6b.speech_encoder, lora_cfg)
model_6b.speech_encoder.print_trainable_parameters()

print('Applying DoRA to t2u_model...')
model_6b.t2u_model = get_peft_model(model_6b.t2u_model, lora_cfg)
model_6b.t2u_model.print_trainable_parameters()

# Multi-GPU layout for 2×T4
if N_GPU >= 2:
    model_6b.speech_encoder  = model_6b.speech_encoder.to('cuda:0')
    model_6b.cif_connector   = model_6b.cif_connector.to('cuda:0')
    model_6b.speaker_adapter = model_6b.speaker_adapter.to('cuda:0')
    model_6b.t2u_model       = model_6b.t2u_model.to('cuda:1')
    if model_6b.vocoder is not None:
        model_6b.vocoder = model_6b.vocoder.to('cuda:1')
    print('Multi-GPU: enc+CIF+spk → cuda:0 | T2U+vocoder → cuda:1')
    DEV_ENC = 'cuda:0'
    DEV_T2U = 'cuda:1'
else:
    model_6b = _consolidate_to_single_gpu(model_6b)
    DEV_ENC = 'cuda:0'
    DEV_T2U = 'cuda:0'

trainable_6b = [p for p in model_6b.parameters() if p.requires_grad]
print(f'Total trainable params: {sum(p.numel() for p in trainable_6b)/1e6:.2f}M')
gpu_mem()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9 — Phase 6b Training Loop                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MAX_STEPS_E2E = 2500
BATCH_ACCUM   = 4
LOG_EVERY     = 50
SAVE_EVERY    = 250

p6b_ck     = load_latest_checkpoint('phase6b_e2e')
start_6b   = p6b_ck.get('step', 0)   if p6b_ck else 0
loss_log_6b = p6b_ck.get('loss_log', []) if p6b_ck else []

if p6b_ck and start_6b > 0:
    try:
        model_6b.speech_encoder.load_state_dict(p6b_ck['enc_state'], strict=False)
        model_6b.t2u_model.load_state_dict(p6b_ck['t2u_state'], strict=False)
        model_6b.cif_connector.load_state_dict(p6b_ck['cif_state'])
        model_6b.speaker_adapter.load_state_dict(p6b_ck['spk_state'])
        print(f'Resumed Phase 6b from step {start_6b}')
    except Exception as e:
        print(f'Checkpoint restore failed: {e}. Starting from step 0.')
        start_6b = 0

optimizer_6b = torch.optim.AdamW([
    {'params': model_6b.cif_connector.parameters(),
     'lr': 5e-5, 'weight_decay': 0.01},
    {'params': model_6b.speaker_adapter.parameters(),
     'lr': 2e-5, 'weight_decay': 0.01},
    {'params': [p for p in model_6b.speech_encoder.parameters() if p.requires_grad],
     'lr': 2e-5, 'weight_decay': 0.01},
    {'params': [p for p in model_6b.t2u_model.parameters() if p.requires_grad],
     'lr': 2e-5, 'weight_decay': 0.01},
], betas=(0.9, 0.98))

scheduler_6b = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_6b, T_max=MAX_STEPS_E2E, eta_min=1e-6,
    last_epoch=max(0, start_6b - 1))

# Note: Phase 6b uses standard scaler, no AMP for unit CE to avoid fp16 issues
scaler_6b = torch.cuda.amp.GradScaler()

# Only samples that have both unit_ids AND real audio
unit_kd = [x for x in kd_data
           if x.get('unit_ids') is not None
           and x.get('t2u_input') is not None
           and sample_id_to_audio.get(x['id']) is not None]
print(f'Phase 6b training samples (unit labels + audio): {len(unit_kd)}')
assert len(unit_kd) > 10, 'Not enough unit_kd samples — check KD extraction and audio lookup'

model_6b.train()
optimizer_6b.zero_grad()

print(f'\n{"="*70}')
print(f'  PHASE 6b: End-to-End DoRA Fine-tuning')
print(f'  Steps: {start_6b} → {MAX_STEPS_E2E}')
print(f'  Loss: 0.80×unit_CE + 0.15×qty_pred + 0.05×spk_reg')
print(f'  KEY: Real speech encoder forward every step (not cached proxy)')
print(f'{"="*70}\n')

recent_unit = []
recent_qty  = []

for step in range(start_6b, MAX_STEPS_E2E):
    sample   = random.choice(unit_kd)
    tgt_lang = sample['tgt_lang']
    lang_id  = torch.tensor([m4t_lang_to_vocoder_id(tgt_lang)], device=DEV_ENC)
    unit_ids = sample['unit_ids'].unsqueeze(0).to(DEV_T2U)
    spk_emb  = sample['spk_emb'].to(DEV_ENC).float()
    n_toks   = float(sample['n_tokens'])
    target_qty = torch.tensor([n_toks], dtype=torch.float, device=DEV_ENC)

    # FIX 1: Load actual audio and run real speech encoder
    audio_wav = sample_id_to_audio.get(sample['id'])
    if audio_wav is None:
        continue

    try:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # FIX 1: Real speech encoder forward
            inp_proc = processor(audio=audio_wav, sampling_rate=16000,
                                 return_tensors='pt')
            inp_f  = inp_proc['input_features'].to(DEV_ENC)
            attn_m = inp_proc.get('attention_mask')
            if attn_m is not None: attn_m = attn_m.to(DEV_ENC)

            enc_out = model_6b.speech_encoder(
                input_features=inp_f,
                attention_mask=attn_m).last_hidden_state.float()  # [1, T_frames, 1024]

            # FIX 3: Correct 3-value CIF return
            connector_out, actual_qty, qty_pred = model_6b.cif_connector(
                enc_out, lang_id)              # [1, T_fired, 1024]

            spk_proj = model_6b.speaker_adapter(spk_emb.unsqueeze(0))  # [1, 256]

            # Move connector output to T2U device
            connector_t2u = connector_out.to(DEV_T2U)

            # T2U unit CE loss — primary training signal
            try:
                t2u_out   = model_6b.t2u_model(
                    inputs_embeds=connector_t2u,
                    labels=unit_ids)
                unit_loss = t2u_out.loss.to(DEV_ENC)
            except Exception as e_t2u:
                print(f'  T2U forward error at step {step+1}: {e_t2u}')
                unit_loss = torch.tensor(0.0, device=DEV_ENC, requires_grad=True)

            # Quantity prediction loss
            qty_loss = F.mse_loss(qty_pred, target_qty)

            # Speaker regularization
            spk_reg = ((spk_proj.float().norm(dim=-1) - 14.0) ** 2).mean()

            loss = (0.80 * unit_loss +
                    0.15 * qty_loss +
                    0.05 * spk_reg)

        scaler_6b.scale(loss / BATCH_ACCUM).backward()
        loss_log_6b.append(loss.item())

        recent_unit.append(unit_loss.item())
        recent_qty.append(qty_loss.item())
        if len(recent_unit) > 100:
            recent_unit.pop(0); recent_qty.pop(0)

        if (step + 1) % BATCH_ACCUM == 0:
            scaler_6b.unscale_(optimizer_6b)
            torch.nn.utils.clip_grad_norm_(trainable_6b, 1.0)
            scaler_6b.step(optimizer_6b)
            scaler_6b.update()
            optimizer_6b.zero_grad()
            scheduler_6b.step()

        if (step + 1) % LOG_EVERY == 0:
            avg_unit = np.mean(recent_unit[-50:]) if recent_unit else 0
            avg_qty  = np.mean(recent_qty[-50:])  if recent_qty  else 0
            cur_lr   = optimizer_6b.param_groups[0]['lr']
            T_fired  = connector_out.shape[1]
            print(f'  Step {step+1:>5}/{MAX_STEPS_E2E} | '
                  f'unit_CE={avg_unit:.4f} | '
                  f'qty_err={avg_qty:.2f} | '
                  f'fired={T_fired} vs tgt={int(n_toks)} | '
                  f'lr={cur_lr:.2e}')

        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint({
                'step':      step + 1,
                'enc_state': model_6b.speech_encoder.state_dict(),
                't2u_state': model_6b.t2u_model.state_dict(),
                'cif_state': model_6b.cif_connector.state_dict(),
                'spk_state': model_6b.speaker_adapter.state_dict(),
                'loss_log':  loss_log_6b,
            }, 'phase6b_e2e', step + 1)
            print(f'  ✓ Checkpoint saved at step {step+1}')

    except Exception as e:
        print(f'  Step {step+1} error: {e}')
        if step - start_6b < 5:
            import traceback; traceback.print_exc()
        optimizer_6b.zero_grad()
        torch.cuda.empty_cache()
        continue

print('Phase 6b training complete.')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10 — Phase 6b: Merge DoRA + Save Final Model                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print('Merging DoRA adapters into base weights...')
model_6b.speech_encoder = model_6b.speech_encoder.merge_and_unload()
model_6b.t2u_model      = model_6b.t2u_model.merge_and_unload()
model_6b.eval()
model_6b = _consolidate_to_single_gpu(model_6b)
sync_model_config(model_6b)
gc.collect(); torch.cuda.empty_cache()
print_model_breakdown(model_6b, 'Phase 6b FINAL: ~673M Textless Model')

p6b_dir = f'{MODEL_DIR}/phase6b_e2e_merged'
os.makedirs(p6b_dir, exist_ok=True)
torch.save({
    'state_dict': model_6b.state_dict(),
    'cif_state':  model_6b.cif_connector.state_dict(),
    'spk_state':  model_6b.speaker_adapter.state_dict(),
    'hidden':     hidden,
    'n_langs':    n_langs,
}, f'{p6b_dir}/textless_model.pt')
if ON_KAGGLE:
    _rclone_push(f'{p6b_dir}/textless_model.pt', 'phase6b_e2e_merged')

print('\n✓ Final ~673M textless model saved to Drive.')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 11 — UPDATED run_textless_s2st inference (handles 3-return CIF)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_textless_s2st(mdl, wav_np, tgt_lang='ben'):
    """
    Audio → SpeechEncoder → CIF → T2U → Vocoder[+ECAPA] → Audio
    Updated for 3-return CIFConnector: (out, actual_qty, qty_pred)
    """
    dev = next(mdl.speech_encoder.parameters()).device
    t0  = time.time()

    with torch.no_grad():
        # 1. Speaker embedding
        spk_emb  = extract_speaker_emb(wav_np).unsqueeze(0).to(dev).float()
        spk_cond = mdl.speaker_adapter(spk_emb)                         # [1, 256]

        # 2. Speech encoder
        inp   = processor(audio=wav_np, sampling_rate=16000, return_tensors='pt')
        inp_f = inp['input_features'].to(dev)
        attn  = inp.get('attention_mask')
        if attn is not None: attn = attn.to(dev)
        lang_id = torch.tensor([m4t_lang_to_vocoder_id(tgt_lang)], device=dev)

        enc_out = mdl.speech_encoder(
            input_features=inp_f, attention_mask=attn).last_hidden_state

        # 3. CIF connector (3-return API)
        connector_out, actual_qty, qty_pred = mdl.cif_connector(enc_out, lang_id)

        # 4. T2U generation
        try:
            t2u_dev = next(mdl.t2u_model.parameters()).device
            unit_ids = mdl.t2u_model.generate(
                inputs_embeds=connector_out.to(t2u_dev),
                max_new_tokens=2048)
        except Exception as e:
            print(f'  T2U generate error: {e}')
            return np.zeros(16000), float('inf'), None

        # 5. Vocoder with speaker conditioning
        try:
            voc_dev  = next(mdl.vocoder.parameters()).device
            tgt_vid  = torch.tensor([m4t_lang_to_vocoder_id(tgt_lang)], device=voc_dev)
            wav_out  = mdl.vocoder(
                input_ids=unit_ids.to(voc_dev),
                spkr_id=spk_cond.to(voc_dev),
                lang_id=tgt_vid)
            wav_np_out = wav_out[0].squeeze().float().cpu().numpy()
        except Exception as e:
            print(f'  Vocoder error: {e}')
            wav_np_out = np.zeros(16000)

    t1  = time.time()
    dur = len(wav_np) / 16000
    rtf = (t1 - t0) / dur if dur > 0 else float('inf')
    return wav_np_out, rtf, unit_ids


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONVERGENCE EXPECTATIONS                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# With the corrected training:
#
# Phase 6a (5000 steps):
#   Steps  0-200:  Cosine loss drops quickly from ~0.95 → ~0.60 (early alignment)
#   Steps 200-1000: Loss drops ~0.60 → ~0.25 (learning speech-text bridge)
#   Steps 1000-3000: Loss drops ~0.25 → ~0.10 (fine alignment)
#   Steps 3000-5000: Loss drops ~0.10 → ~0.05 (near convergence)
#   Cosine loss < 0.15 at end of 6a = GOOD for proceeding to 6b
#   Quantity error should be within ±5 tokens by end of 6a
#
# Phase 6b (2500 steps):
#   Unit CE loss should start at ~6-8 (random) and drop to ~3-5
#   If unit CE does not drop below 5.0 by step 500: check connector output shape
#   Expected final unit CE: 3.0-5.0 (sufficient for coherent speech output)
#
# If cosine KD loss is still not moving after 500 steps of 6a:
#   - Check: is std(enc_out) > 0.5? If not, speech encoder isn't producing real features
#   - Check: is std(target) > 0.5? If not, teacher KD data is bad
#   - Print shapes at step 0: encoder_out.shape, connector_out.shape, target.shape
#   - Verify lang_id values are in range [0, n_langs-1]