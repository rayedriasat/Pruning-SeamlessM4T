# Phase 6C T2U Recovery — Complete Training Documentation

## Project Context

We are training a **student SeamlessM4T-v2 model** that has been compressed via encoder pruning (speech encoder reduced from 6→4 layers). The model has three main components:

- **Speech Encoder** — pruned, frozen in this phase
- **Text Decoder** — frozen in this phase (confirmed healthy, Text-ChrF stable at 38.37 throughout all experiments)
- **T2U Model** (Text-to-Unit) — the component being trained in Phase 6C

The T2U model converts text decoder hidden states → discrete speech units → eventually waveform via a vocoder. It is a non-autoregressive model that uses a **duration predictor** to expand character-level representations into unit-length sequences.

**Baseline at start of Phase 6C:** Text-ChrF=38.37, ASR-ChrF=34.11

---

## The Core Problem

After encoder pruning, the T2U model's duration predictor produces **wrong-length unit sequences**. The pruned encoder outputs shorter/different representations than the original, so the duration predictor (calibrated for the original encoder) predicts incorrect durations → wrong output length → the entire unit sequence is misaligned with what the vocoder expects → ASR degrades.

**This is the root cause of all failures.** Every approach that tried to train the decoder with cross-entropy on unit targets *before* fixing the duration predictor failed, because CE was applied to length-misaligned sequences, teaching the decoder wrong unit-to-position mappings.

---

## What Was Tried and Why It Failed

### v1 — Anchor KL Retention (retain_kl + unit_ce)

**Config:**
- Loss = `0.80 * unit_ce + 0.20 * retain_kl`
- `retain_kl` measured KL divergence between student T2U and a frozen anchor copy
- Anchor was copied from phase5 T2U weights (post-pruning, never unit-trained)

**What happened:**
```
retain_kl ≈ 250  >>  unit_ce ≈ 8
Effective gradient: 0.80*8 + 0.20*251 = 6.56 + 50.2
Retention dominated by 7.6x despite 0.20 weight
```

**Why it failed:** The anchor was copied from broken post-pruning weights. Anchoring to a broken model actively prevented unit_ce from moving the T2U in any useful direction. The retain_kl of 250 (should be <1 for similar distributions) indicated the student had already diverged from the broken anchor the moment training began, and the huge KL gradient pulled it back to broken weights.

**ASR result:** Briefly improved at step 25 (unit_ce signal correct momentarily) then degraded as retention won.

---

### v2/v3 — No Anchor, Curriculum Audio, Shift-Aligned CE

**Config:**
- Removed anchor entirely
- Loss = `0.80 * unit_ce + 0.20 * duration_mse`
- Length curriculum: 5s→7s→9s→11s audio
- Shift-aligned CE: searched over shifts [-max_shift, +max_shift] to find best alignment
- LR=3e-6, warmup=20% (140 steps), grad_clip=0.25→1.0

**What happened (v3 log):**
```
unit_loss: 8.12 → 7.66  (slow, flatlining)
retain_kl: removed ✓
ASR step 50:  34.38 (best, +0.27)
ASR step 100: 33.71 (declining)
OOM at micro_step 852 (opt_step ~106) when curriculum moved to 7s audio
```

**Why it partially worked:** Removing the broken anchor let unit_ce signal through. Duration predictor loss started dropping (0.58→0.16).

**Why it still failed:**
1. OOM: 198M trainable params (included T2U encoder) + 7s audio → OOM in decoder attention
2. ASR declining after step 50: unit_ce on length-misaligned sequences was teaching wrong positions. The duration predictor hadn't converged yet so sequences were still wrong length.
3. The 20% warmup (140 steps) kept LR so low that unit_loss barely moved — but this was actually *safe*, not the problem.

**Key lesson:** The duration predictor must be fixed BEFORE applying unit_ce to the decoder.

---

### v4 — Aggressive LR, Adaptive dur Weight, OOM Fixes

**Changes from v3:**
- LR 3e-6 → 5e-6 (thought flatline was the problem — it wasn't)
- Warmup 20% → 5% (35 steps) — critical mistake
- Abrupt dur weight switch: when dur_loss < 0.15, instantly drop 0.20→0.05
- Decoder-only training (114M, removed encoder) — correct fix for OOM
- Curriculum 5→6→8→10s — more conservative

**What happened:**
```
ASR step 50:  32.54 (already worse than baseline)
dur converged at step 74
loss jumps after weight switch: 5.77→6.37→6.75
ASR step 100: 15.85  ← catastrophic collapse
|shift| jumps to 4.28 at step 110
```

**Why it catastrophically failed:**
1. **5% warmup hit peak LR at step 35** while unit_loss=7.8 and gradients were huge → model driven off the learned manifold
2. **Abrupt loss weight switch** (0.20→0.05 dur at step 74) changed effective gradient scale by 19% instantly → destabilized Adam moment estimates which were calibrated for the old loss scale
3. Both effects combined destroyed the output distribution → ASR 34→15

---

### v5 — Restored Stability, Gradual dur Decay, Emergency Brake

**Changes from v4:**
- LR back to 3e-6, warmup back to 15% (105 steps)
- Gradual dur weight decay: linear 0.20→0.05 over steps 50-200
- grad_clip 0.5 (tighter)
- label_smoothing=0.1 on unit CE
- Emergency brake: if ASR drops >3 from best, reload best + halve LR

**What happened:**
```
ASR step 50:  33.75 (below baseline 34.11)
ASR step 100: 31.36 (gap=2.75)
unit_loss: 8.35→7.37 (dropping — model IS learning something)
Text-ChrF: 38.37 stable throughout
```

**Why it still failed:** The pattern is now definitively clear. unit_loss drops every time (8.x→7.x) but ASR drops in lockstep. **The unit targets are not wrong in principle, but the unit_ce signal is corrupted because it is applied to a length-misaligned sequence.** The duration predictor controls output sequence length *before* the decoder sees it. Training the decoder with CE on misaligned sequences teaches it to predict the wrong units at the wrong positions. This happens regardless of LR, warmup, label smoothing, or gradient clipping.

---

### v6 — Two-Phase: Duration-Only Then Unit CE ✅ WORKING

**The key insight:** Fix duration predictor first. Only add unit_ce after alignment is corrected.

**Config:**
- **Phase A (steps 0-150):** `loss = 1.0 * duration_mse`, decoder LR=0 (decoder completely frozen)
- **Phase B (steps 150-700):** `loss = 0.90 * unit_ce + 0.10 * duration_mse`, unit_ce weight ramps in over 50 steps
- Two optimizer param groups: duration predictor (always active), decoder layers (LR=0 in Phase A)
- Gradient checkpointing on T2U decoder (saves ~40% activation VRAM)
- Emergency brake in Phase B only

**What happened:**
```
Phase A:
  dur_loss: 0.60 → 0.07 (converged cleanly)
  ASR step 50:  34.13 (+0.02 from baseline) ← stable, not degrading
  ASR step 100: 35.34 (+1.23) ← duration fix is working!
  ASR step 150: 35.58 (+1.47) ← new best, +1.44 trend

Phase B:
  unit_loss appears at 9.4 (above 8.3 baseline — bad alignment still)
  |shift| jumps: 2.5 → 4.6 → 7.9 (shift search finding spurious minima)
  lr_dec=0.00e+00 throughout Phase B ← BUG: decoder LR never actually set
  ASR step 200: 35.38 (stable, not crashing)
  ASR step 250: 35.29 (stable)
  ASR step 300: 35.45 (stable)
```

**Critical bug discovered in Phase B:** The `optimizer.param_groups[0]['lr'] = PHASE6C_BASE_LR` assignment was immediately overridden by `scheduler.step()` on the next iteration, because the cosine scheduler scales from the *initial* LR (which was 0.0), so it perpetually reset decoder LR to 0. The decoder was never trained in Phase B.

**However:** ASR remained stable at 35.29-35.58 throughout Phase B even with the broken decoder LR. This means:
1. The duration predictor fix alone accounts for most of the gain
2. The decoder was not being corrupted (because it wasn't being trained)
3. Phase B is still needed — the decoder hasn't been touched yet

**Best checkpoint: `phase6_6c_best_step000150.pt` (ASR=35.58)**

---

### v7 — Fix Scheduler Override, Two Separate Optimizers (CURRENT)

**The bug fix:** Use **two completely separate optimizers** with **two completely separate scalers**. No shared scheduler that can override LR. Decoder LR is set manually each step via `_get_dec_lr(opt_step)`.

**Config:**
```python
# Duration optimizer (Phase A active, Phase B maintenance)
opt_dur: AdamW, lr=1e-5 Phase A → 5e-7 Phase B (converged, tiny maintenance)

# Decoder optimizer (Phase B only, fresh start)  
opt_dec: AdamW, lr set manually via _get_dec_lr()
  - LR=0 during Phase A
  - Own 20-step warmup from Phase B start
  - Cosine decay to 5% of peak over remaining 530 steps

# Shift search
max_shift=3 initially (conservative, prevents spurious alignments)
Widens to curriculum values only when unit_loss drops below 8.0
(unit_loss > 8.3 baseline means shift is finding wrong alignments)

# Loss weights Phase B
w_unit: 0→0.90 ramp over 30 steps
w_dur: 0.10 maintenance

# Phase A: 150 steps (duration only)
# Phase B: 150→700 steps (decoder + maintenance dur)
```

**Status:** Submitted, awaiting results.

---

## Architecture Reference

### T2U Model Components (SeamlessM4T-v2)
```
T2U input: text decoder hidden states [B, T_text, d_model]
    ↓
T2U Encoder (frozen in all v4+ runs — OOM cause at 198M total)
    ↓
Duration Predictor → per-token durations → expand to unit length
    ↓  
T2U Decoder layers (non-autoregressive, attends to expanded sequence)
    ↓
lm_head (frozen — unit vocab mapping correct)
    ↓
Unit sequence [B, T_units, vocab_size]
```

### Parameter Counts
- Full T2U trainable (encoder+decoder): 198.25M → caused OOM at 7s audio
- Decoder-only trainable (v4+): 114.29M → fits in 14.56GB VRAM up to ~6s audio
- Duration predictor subset: ~few M params within the 114M

### VRAM Budget (14.56GB GPU)
- Model weights (student): ~6-7GB
- Activations at 5s audio + decoder: ~7GB → tight
- Activations at 7s audio: OOM
- Gradient checkpointing on T2U decoder: saves ~40% activations → enables 6-8s audio
- OOM handling: skip-and-continue, reduce audio cap by 1s, floor at 4s

---

## Key Technical Findings

### Finding 1: Duration Predictor is the Dominant Bottleneck
The pruned encoder produces different-length representations. The duration predictor was never retrained after pruning. Fixing it alone improved ASR from 34.11 → 35.58 (+1.47 ChrF). This is the most important single intervention.

### Finding 2: Unit CE Before Duration Fix is Actively Harmful
Applying cross-entropy to decoder outputs before duration is correct teaches the decoder wrong unit-to-position mappings. The model learns to predict plausible-looking units at the wrong time steps. This is why unit_loss drops (model is learning *something*) but ASR degrades in lockstep.

### Finding 3: The Anchor Was Double-Broken
The phase5 T2U was copied post-pruning but never unit-trained. retain_kl=250 (should be <1) shows it was already broken. Anchoring to it was anchoring to the failure mode.

### Finding 4: LR Aggressiveness Destroys Distribution
5e-6 with 5% warmup (peak at step 35) → ASR 34→15. 3e-6 with 15-20% warmup → stable. The T2U output distribution is fragile to large gradient steps, especially before duration is corrected and sequences are properly aligned.

### Finding 5: Abrupt Loss Weight Changes Destabilize Adam
Adam's moment estimates (m1, m2) are calibrated for the current gradient scale. Instantly changing loss weight from 0.20→0.05 changes the effective gradient magnitude by 19% → moments are stale → first steps after switch are poorly scaled → instability spike visible in loss.

### Finding 6: Scheduler Must Never Initialize a Param Group at LR=0
If `make_cosine_scheduler` is called with a param group initialized at LR=0, the scheduler scales cosine decay from 0 → all values are 0 forever. Must either: (a) use two completely separate optimizers with separate schedulers, or (b) set the LR *before* adding it to the scheduler, or (c) manage Phase B LR entirely manually outside the scheduler.

### Finding 7: Shift Search Width Must Match Alignment Quality
At max_shift=10-15 with misaligned sequences, the search finds spurious low-CE alignments (local minima where the model accidentally predicts a different valid-looking unit sequence). |shift| jumping from 2.5 to 7.9 is the signal. Start at max_shift=3, widen only when unit_loss actually drops below baseline.

### Finding 8: Text-ChrF Frozen = Reliable Sanity Check
Text-ChrF stayed exactly 38.37 throughout all experiments. This confirms the text decoder is intact and the speech encoder upstream is not being corrupted. All degradation is isolated to T2U.

---

## What Works / What Doesn't Summary Table

| Approach | ASR Result | Verdict |
|---|---|---|
| retain_kl anchor (broken weights) | Briefly +0.2, then decline | ✗ Anchor was broken |
| unit_ce only, aggressive LR | 34→15 collapse | ✗ LR too aggressive |
| unit_ce + dur, slow warmup | 34→31 gradual decline | ✗ CE before duration fixed |
| unit_ce + abrupt weight switch | 34→15 collapse | ✗ Adam moment destabilization |
| dur-only Phase A (v6) | 34.11→35.58 (+1.47) | ✓ **Core fix confirmed** |
| Phase B with broken LR | 35.58→35.29-35.45 stable | ~ Neutral (decoder untrained) |
| Phase B with fixed LR (v7) | Pending | Expected: further improvement |

---

## Resuming / Running Instructions

### If Starting Fresh
Load the **phase5 checkpoint** (pre-Phase 6C). The preflight gate is Text-ChrF > 37.0.

### If Resuming After v6 Phase A Success
Load **`phase6_6c_best_step000150.pt`** (ASR=35.58). This is the best state: duration predictor fully trained, decoder untouched and uncorrupted.

```python
phase6_logs['6c'] = run_t2u_recovery_stage(
    stage_key        = '6c',
    title            = 'T2U recovery v7',
    steps            = 700,
    max_audio_sec    = 10,
    resume_from_step = 150,   # loads phase6_6c_best automatically
)
```

### What to Monitor

**Phase A (if re-running from scratch):**
- `dur_loss`: must drop from ~0.6 toward <0.10
- ASR: must be stable or improving (decoder frozen, should not degrade)
- `|shift|`: will show 0.00 (shift search not running in Phase A)

**Phase B:**
- `lr_dec`: must be nonzero (v7 fixes this with manual LR setting)
- `unit_loss`: should start at ~8.3 (baseline), must drop. If >9.0, shift search is finding wrong alignments → `shift_widened` flag should be False
- `|shift|`: should be ≤3 until unit_loss < 8.0, then widens
- ASR: should trend upward from 35.58. Brake triggers if gap >2.0 from best
- Emergency brake: reloads best state + clears optimizer state + halves dec LR

### Hyperparameters That Matter Most (do not change without reason)
```python
PHASE6C_PHASE_A_STEPS = 150      # enough for dur_loss to reach <0.10
PHASE6C_DUR_LR_A      = 1e-5     # duration predictor peak LR — working
PHASE6C_DUR_LR_B      = 5e-7     # maintenance only, duration is converged
PHASE6C_DEC_LR        = 3e-6     # decoder Phase B peak — conservative
PHASE6C_DEC_WARMUP    = 20       # decoder-specific warmup steps
PHASE6C_GRAD_CLIP     = 0.5      # tight clip — T2U distribution is fragile
PHASE6C_LABEL_SMOOTH  = 0.1      # reduces overfit to noisy shift alignment
PHASE6C_SHIFT_INIT    = 3        # conservative until unit_loss < 8.0
PHASE6C_BRAKE_THRESH  = 2.0      # tight brake — learned from v4 collapse
```

---

## Remaining Open Questions for v7

1. **Will unit_ce actually help after duration is fixed?** The theory says yes — once sequence length is correct, shift-aligned CE should find clean alignments and improve unit prediction. We will know from whether unit_loss drops below 8.0 in Phase B.

2. **Is max_shift=3 too tight?** Even with duration fixed, there may be residual 1-2 unit offsets. Shift=3 allows ±3 unit offset which should be sufficient for well-aligned sequences.

3. **Should Phase A be longer?** dur_loss reached 0.03-0.07 by step 150 (converged). Longer Phase A would not help further.

4. **Is 550 steps of Phase B enough?** Given that the decoder has 114M params and we're using a small dataset with short audio, 550 steps may be sufficient for initial recovery. If unit_loss does not drop below 7.5 by step 400, consider extending to 1000 total steps.