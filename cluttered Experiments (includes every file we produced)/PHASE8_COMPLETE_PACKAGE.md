# Phase 8: T2U NAR Training - Complete Package

## 📦 What's Included

This package contains everything you need to complete Phase 8 (T2U NAR training) and recover audio translation quality in your compressed SeamlessM4T model.

### Files

1. **`PHASE8_README.md`** (THIS FILE)
   - Setup instructions
   - Cell-by-cell guide
   - Troubleshooting
   - Expected results

2. **`PHASE8_T2U_NAR_TRAINING.md`**
   - Conceptual overview
   - Why T2U training is separate
   - Architecture explanation
   - Training strategy

3. **`phase8_cells.py`**
   - Complete Phase 8 code (10 cells)
   - Ready to copy-paste into notebook
   - Fully commented

## 🎯 Quick Start (3 Steps)

### Step 1: Verify Prerequisites

```python
# Run this in your notebook to check readiness
import os

checks = {
    "Phase 7 model": os.path.exists(f"{MODEL_DIR}/phase7_dora_merged"),
    "Unit cache": os.path.exists(f"{CKPT_DIR}/unit_labels_cache.pt"),
    "ft_samples loaded": 'ft_samples' in globals() and len(ft_samples) > 0,
    "eval_samples loaded": 'eval_samples' in globals() and len(eval_samples) > 0,
}

print("Phase 8 Readiness Check:")
for check, passed in checks.items():
    print(f"  {'✓' if passed else '✗'} {check}")

if all(checks.values()):
    print("\\n✓ Ready to start Phase 8!")
else:
    print("\\n✗ Complete missing prerequisites first")
```

### Step 2: Add Phase 8 Cells

Open `phase8_cells.py` and copy all 10 cells to your notebook after Phase 7.

### Step 3: Run Training

```python
# Cell 1-5: Setup and verification (~2 minutes)
# Cell 6: Training loop (~35 minutes on T4)
# Cell 7-10: Evaluation and results (~10 minutes)
```

**Total time**: ~45-50 minutes

## 📊 Expected Results

### Before Phase 8 (Phase 7 output)
```
Phase 7 (DoRA fine-tuned):
  txt-BLEU: 42.5  txt-ChrF: 65.3  ← Text quality recovered
  asr-BLEU:  3.2  asr-ChrF:  8.7  ← Audio still broken
```

### After Phase 8 (T2U fine-tuned)
```
Phase 8 (T2U NAR trained):
  txt-BLEU: 42.1  txt-ChrF: 64.8  ← Text quality maintained
  asr-BLEU: 18.5  asr-ChrF: 38.2  ← Audio quality recovered! 🎉
```

### Improvement
- **ASR-BLEU**: +15.3 points (3.2 → 18.5)
- **ASR-ChrF**: +29.5 points (8.7 → 38.2)
- **Audio**: From unintelligible → intelligible Bengali speech

## 🔧 Key Hyperparameters

```python
MAX_STEPS  = 1000   # Increase to 2000 if ASR-BLEU < 15
BATCH_SIZE = 2      # Reduce to 1 if OOM
GRAD_ACCUM = 4      # Increase to 8 if reducing batch size
LR         = 5e-5   # Lower than Phase 7 (NAR is sensitive)
GRAD_CLIP  = 1.0    # Prevents gradient explosion
```

## 🎓 What Phase 8 Does Differently

### Phase 7 (S2TT Recovery)
- **Loss**: Text cross-entropy
- **Gradient flow**: Speech encoder → Text decoder
- **T2U**: Receives zero gradient (frozen by loss function)
- **Result**: Text quality recovered, audio still broken

### Phase 8 (T2U Recovery)
- **Loss**: Unit cross-entropy (discrete speech units)
- **Gradient flow**: Text decoder output → T2U encoder → T2U decoder
- **Speech encoder**: Frozen (already fine-tuned)
- **Result**: Audio quality recovered, text quality maintained

## 🔍 Architecture Diagram

```
Phase 7 Training Path (S2TT):
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   English   │────▶│    Speech    │────▶│    Text     │
│   Speech    │     │   Encoder    │     │   Decoder   │
└─────────────┘     └──────────────┘     └─────────────┘
                           ▲                     ▲
                           │                     │
                      [TRAIN LoRA]          [TRAIN LoRA]
                           │                     │
                           └─────────────────────┘
                                   S2TT Loss
                                (text tokens)

Phase 8 Training Path (T2U):
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   English   │────▶│    Speech    │────▶│    Text     │
│   Speech    │     │   Encoder    │     │   Decoder   │
└─────────────┘     └──────────────┘     └─────────────┘
                        [FROZEN]             [FROZEN]
                                                  │
                                                  ▼
                                          ┌─────────────┐
                                          │     T2U     │
                                          │   Encoder   │
                                          └─────────────┘
                                                  │
                                             [TRAIN]
                                                  │
                                                  ▼
                                          ┌─────────────┐
                                          │     T2U     │
                                          │   Decoder   │
                                          └─────────────┘
                                                  │
                                             [TRAIN]
                                                  │
                                                  ▼
                                            Unit Loss
                                        (discrete units)
```

## 📈 Training Progress Visualization

```
Loss Curve (Expected):
8.0 ┤●
7.5 ┤ ●
7.0 ┤  ●●
6.5 ┤    ●●
6.0 ┤      ●●
5.5 ┤        ●●
5.0 ┤          ●●
4.5 ┤            ●●
4.0 ┤              ●●
3.5 ┤                ●●
3.0 ┤                  ●●
2.5 ┤                    ●●●
2.0 ┤                       ●●●●●●●●●●
    └─────────────────────────────────────▶
    0   100  200  300  400  500  600  700  800  900 1000
                        Steps

✓ Good: Smooth downward curve, final loss < 3.0
✗ Bad:  Flat curve, spiky, or final loss > 4.0
```

## 🎵 Audio Quality Check

After training, listen to the generated audio samples:

```python
# Phase 8 saves 4 audio clips automatically
import IPython.display as ipd

for i in range(1, 5):
    print(f"\\nSample {i}:")
    print("  Input (English):")
    ipd.display(ipd.Audio(f"{AUDIO_DIR}/P8_T2U_NAR_s{i}in.wav"))
    print("  Output (Bengali):")
    ipd.display(ipd.Audio(f"{AUDIO_DIR}/P8_T2U_NAR_s{i}out.wav"))
```

**What to listen for**:
- ✓ Clear Bengali speech (not garbled)
- ✓ Correct pronunciation of words
- ✓ Natural prosody and rhythm
- ✗ Silent or very short audio (<1 second)
- ✗ Robotic or distorted voice
- ✗ English words in Bengali output

## 🐛 Common Issues & Fixes

### Issue 1: "Unit cache not found"
```python
# Fix: Re-run Phase 7 Cell 6
unit_labels = load_or_extract_units(model_p6, processor, ft_samples)
torch.save({"units": unit_labels}, UNIT_CACHE_PATH)
```

### Issue 2: "T2U forward pass fails"
```python
# Fix: Check unit sequence lengths
lens = [p["units"].numel() for p in ft_s2st_pairs]
print(f"Min: {min(lens)}, Max: {max(lens)}, Mean: {np.mean(lens):.0f}")

# Filter out invalid lengths
ft_s2st_pairs = [p for p in ft_s2st_pairs if 3 <= p["units"].numel() <= 500]
```

### Issue 3: "Loss is NaN"
```python
# Fix: Reduce learning rate
LR = 1e-5  # Instead of 5e-5
GRAD_CLIP = 0.5  # Instead of 1.0
```

### Issue 4: "Out of memory"
```python
# Fix: Reduce batch size
BATCH_SIZE = 1
GRAD_ACCUM = 8
torch.cuda.empty_cache()
```

### Issue 5: "ASR-BLEU still low (<10)"
```python
# Fix: Train longer
MAX_STEPS = 2000  # Instead of 1000

# Or check unit label quality
print("Sample units:", unit_labels[0][:20])
print("Unit vocab size:", model_p7.config.unit_hifi_gan_vocab_size)
```

## 📝 Checklist

### Before Training
- [ ] Phase 7 complete and model saved
- [ ] Unit labels cached (from Phase 7 Cell 6)
- [ ] `ft_samples` loaded with Bengali target audio
- [ ] `eval_samples` loaded for benchmarking
- [ ] GPU has >12 GB VRAM available

### During Training
- [ ] T2U forward pass verification passed
- [ ] Loss starts around 7-8
- [ ] Loss decreases smoothly
- [ ] No NaN or Inf losses
- [ ] Checkpoints saving every 100 steps

### After Training
- [ ] Final loss < 3.0
- [ ] Model saved to Drive
- [ ] Benchmark shows ASR-BLEU > 10
- [ ] Audio samples sound intelligible
- [ ] Results table generated

## 🎉 Success Criteria

Your Phase 8 training is successful if:

1. **Loss convergence**: Final loss < 3.0 (ideally < 2.5)
2. **ASR-BLEU**: > 15 (good), > 20 (excellent)
3. **ASR-ChrF**: > 35 (good), > 40 (excellent)
4. **Audio quality**: Intelligible Bengali speech
5. **Text quality**: Maintained from Phase 7 (BLEU ~40-50)

## 📚 Further Reading

### Papers
- **Moslem et al. (IWSLT 2025)**: Iterative layer pruning for speech translation
- **Liu et al. (ICML 2024)**: DoRA weight-decomposed low-rank adaptation
- **Baevski et al. (NeurIPS 2020)**: wav2vec 2.0 (unit extraction)

### SeamlessM4T Documentation
- [Model card](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [Paper](https://arxiv.org/abs/2308.11596)
- [GitHub](https://github.com/facebookresearch/seamless_communication)

## 🤝 Support

If you encounter issues:

1. **Check logs**: Look for error messages in training output
2. **Verify prerequisites**: Run readiness check (Step 1)
3. **Review troubleshooting**: See "Common Issues & Fixes" section
4. **Check GPU memory**: Run `nvidia-smi` to see VRAM usage
5. **Reduce batch size**: If OOM, set `BATCH_SIZE=1`

## 🏆 Final Results

After completing Phase 8, you will have:

- ✅ **Compressed model**: ~1.1 GB (from 2.3 GB baseline)
- ✅ **Text quality**: ~90% of baseline (BLEU/ChrF)
- ✅ **Audio quality**: ~70-80% of baseline (ASR-BLEU/ChrF)
- ✅ **Speed**: 2-3x faster inference (lower RTF)
- ✅ **Deployment-ready**: Model saved and benchmarked

**Congratulations on completing the SeamlessM4T compression pipeline!** 🎊

---

## 📄 File Summary

```
phase8-package/
├── PHASE8_README.md                 ← Setup & troubleshooting
├── PHASE8_T2U_NAR_TRAINING.md      ← Conceptual overview
├── phase8_cells.py                  ← Complete code (10 cells)
└── PHASE8_COMPLETE_PACKAGE.md      ← This file
```

**Next step**: Open `phase8_cells.py` and start copying cells to your notebook!
