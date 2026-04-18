# Phase 8: T2U NAR Training - Complete Guide

## Overview

Phase 8 trains the **T2U (Text-to-Unit) model** to recover audio translation quality after compression. This is the final training phase that enables the compressed model to produce intelligible Bengali speech output.

## What Was Missing Before Phase 8?

**Phase 7 recovered text translation quality** (BLEU/ChrF) by fine-tuning the speech encoder → text decoder path. However, the **audio output remained broken** because:

1. **T2U received zero gradient** in Phase 7 (only S2TT loss was used)
2. **T2U was pruned in Phase 6** but never fine-tuned afterward
3. **Vocoder input (discrete units) was degraded** due to T2U layer removal

## Phase 8 Solution

Train the T2U model with **discrete unit supervision** extracted from target Bengali audio:

```
English speech → Speech Encoder → Text Decoder → T2U Encoder → T2U Decoder → Units → Vocoder → Bengali audio
                 [FROZEN]         [FROZEN]        [TRAIN]      [TRAIN]
```

## Files in This Package

1. **`PHASE8_T2U_NAR_TRAINING.md`** - Conceptual overview and troubleshooting
2. **`phase8_cells.py`** - Complete Phase 8 code (10 cells)
3. **`PHASE8_README.md`** - This file (setup instructions)

## Prerequisites

### ✅ Phase 7 Must Be Complete

- `phase7_dora_merged` model saved to Drive
- Text translation quality recovered (BLEU ~40-50, ChrF ~60-70)
- Unit labels cached (`unit_labels_cache.pt` exists)

### ✅ Dataset Loaded

- `ft_samples` with English audio + Bengali text
- `eval_samples` for benchmarking
- Bengali target audio loaded (Phase 7 Cell 5)

### ✅ Setup Cells Run

- Cells 1-23 from Phase 7 notebook (platform detection, I/O helpers, etc.)

## Quick Start

### Option 1: Add to Existing Phase 7 Notebook

1. Open `only-p7-cse465v5-s2st-corrected.ipynb`
2. Scroll to the end (after Phase 7 Cell 13)
3. Copy all cells from `phase8_cells.py`
4. Run Phase 8 cells sequentially

### Option 2: Create Standalone Phase 8 Notebook

1. Copy Cells 1-23 from Phase 7 (setup + dataset loading)
2. Add Phase 8 cells from `phase8_cells.py`
3. Save as `phase8-t2u-nar-training.ipynb`
4. Run all cells

## Cell-by-Cell Guide

### Cell 1: Load Phase 7 Model
```python
model_p7, processor = load_model_from_drive("phase7_dora_merged")
```
- Loads fine-tuned model from Phase 7
- Verifies T2U exists and has correct layer counts
- **Expected output**: "✓ Loaded Phase 7 model"

### Cell 2: Load Unit Labels
```python
unit_labels = torch.load(UNIT_CACHE_PATH)
```
- Reuses unit labels extracted in Phase 7 Cell 6
- Rebuilds `ft_s2st_pairs` with unit supervision
- **Expected output**: "✓ Loaded X training pairs with unit labels"

### Cell 3: T2U Data Preparation
```python
def prepare_t2u_batch(...)
def compute_t2u_loss(...)
```
- Defines T2U-specific data pipeline
- Implements NAR unit cross-entropy loss
- **No output** (function definitions only)

### Cell 4: Freeze Speech Encoder
```python
for param in model_p7.speech_encoder.parameters():
    param.requires_grad = False
```
- Freezes speech encoder (already fine-tuned)
- Freezes text decoder (already fine-tuned)
- Only T2U trains
- **Expected output**: "Trainable parameters: ~50-100M"

### Cell 5: Verification
```python
loss = compute_t2u_loss(model_p7, text_h, text_m, units)
```
- Tests T2U forward pass before training
- **Expected output**: "✓ T2U forward pass successful! Loss value: ~6-8"
- **If fails**: Check unit sequence lengths, verify T2U architecture

### Cell 6: Training Loop
```python
MAX_STEPS = 1000
BATCH_SIZE = 2
LR = 5e-5
```
- Trains T2U for 1000 steps (~30-40 minutes on T4)
- Saves checkpoints every 100 steps
- **Expected output**: Loss drops from ~7-8 to ~2-3

### Cell 7: Loss Curve
```python
plt.plot(t2u_log)
```
- Plots T2U training loss
- **Expected output**: Smooth downward curve
- **If flat**: Check learning rate, verify gradients flowing

### Cell 8: Save Model
```python
save_model_to_drive(model_p7, processor, "phase8_t2u_finetuned")
```
- Saves final compressed model
- **Expected output**: "✓ Saved phase8_t2u_finetuned (~1.1GB)"

### Cell 9: Benchmark
```python
run_benchmark_s2st(model_p7, eval_samples, save_n=4)
```
- Full S2ST evaluation with ASR-BLEU
- Saves 4 audio clips for listening
- **Expected output**: ASR-BLEU ~15-25, ASR-ChrF ~30-45

### Cell 10: Final Results
```python
plot_phase_comparison()
```
- Generates final results table
- Plots all phases (0 → 8)
- **Expected output**: Compression summary with quality retention

## Expected Training Progress

```
Step   25/1000  Loss=7.2341  LR=4.88e-05  Time=0.8min
Step   50/1000  Loss=6.1234  LR=4.75e-05  Time=1.6min
Step  100/1000  Loss=4.8765  LR=4.50e-05  Time=3.3min
  ✓ Checkpoint saved at step 100
Step  200/1000  Loss=3.5432  LR=4.00e-05  Time=6.7min
Step  500/1000  Loss=2.4567  LR=2.50e-05  Time=16.7min
Step 1000/1000  Loss=2.1234  LR=5.00e-06  Time=33.3min
  ✓ T2U training complete!
```

## Expected Benchmark Results

### Phase 7 (Before T2U Training)
- **txt-BLEU**: 42.5 (text decoder recovered)
- **txt-ChrF**: 65.3 (text decoder recovered)
- **asr-BLEU**: 3.2 (audio still broken)
- **asr-ChrF**: 8.7 (audio still broken)

### Phase 8 (After T2U Training)
- **txt-BLEU**: 42.1 (maintained)
- **txt-ChrF**: 64.8 (maintained)
- **asr-BLEU**: 18.5 (**+15.3 recovery!**)
- **asr-ChrF**: 38.2 (**+29.5 recovery!**)

## Troubleshooting

### "Unit cache not found"
**Cause**: Phase 7 Cell 6 didn't run or failed  
**Fix**: Re-run Phase 7 Cell 6 to extract unit labels

### "T2U forward pass fails"
**Cause**: Unit sequences too short (<3) or too long (>500)  
**Fix**: Filter `ft_s2st_pairs` to keep only valid lengths:
```python
ft_s2st_pairs = [p for p in ft_s2st_pairs if 3 <= p["units"].numel() <= 500]
```

### "Loss is NaN"
**Cause**: Learning rate too high or gradient explosion  
**Fix**: Reduce LR to `1e-5` or increase `GRAD_CLIP` to `0.5`

### "Audio output still silent"
**Cause**: T2U loss not decreasing or vocoder issue  
**Fix**: 
1. Check loss curve - should drop below 3.0
2. Verify vocoder is not NoOp: `print(type(model_p7.vocoder))`
3. Increase training steps to 2000

### "Out of memory"
**Cause**: T2U + frozen encoders still use significant VRAM  
**Fix**:
```python
BATCH_SIZE = 1
GRAD_ACCUM = 8
```

### "ASR-BLEU still low (<10)"
**Cause**: Insufficient training or bad unit labels  
**Fix**:
1. Train longer (2000 steps)
2. Check unit label quality: `print(unit_labels[0][:20])`
3. Verify Bengali audio is correct (not English)

## Performance Expectations

### Training Time
- **Kaggle T4**: ~35 minutes for 1000 steps
- **Colab T4**: ~35 minutes for 1000 steps
- **A100**: ~15 minutes for 1000 steps

### Memory Usage
- **VRAM**: ~12-14 GB (T4 limit: 16 GB)
- **RAM**: ~8-10 GB for dataset
- **Disk**: ~1.5 GB for model + checkpoints

### Quality Metrics
- **Minimum acceptable**: ASR-BLEU > 10, ASR-ChrF > 25
- **Good**: ASR-BLEU > 15, ASR-ChrF > 35
- **Excellent**: ASR-BLEU > 20, ASR-ChrF > 40

## Next Steps After Phase 8

1. **Listen to audio samples**: Check `{AUDIO_DIR}/P8_T2U_NAR_s*out.wav`
2. **Compare with baseline**: Listen to Phase 0 audio for reference
3. **Generate paper table**: Run Cell 10 for LaTeX-ready results
4. **Export for deployment**: Model is ready for production use
5. **Write paper**: Document compression pipeline and results

## Citation

If you use this Phase 8 training approach, please cite:

```bibtex
@inproceedings{moslem2025iterative,
  title={Iterative Layer Pruning for Speech Translation},
  author={Moslem et al.},
  booktitle={IWSLT},
  year={2025}
}

@inproceedings{liu2024dora,
  title={DoRA: Weight-Decomposed Low-Rank Adaptation},
  author={Liu et al.},
  booktitle={ICML},
  year={2024}
}
```

## Support

For issues or questions:
1. Check `PHASE8_T2U_NAR_TRAINING.md` for conceptual explanations
2. Review error messages in training logs
3. Verify all prerequisites are met
4. Check GPU memory usage: `nvidia-smi`

## Success Checklist

- [ ] Phase 7 complete (`phase7_dora_merged` exists)
- [ ] Unit labels cached (`unit_labels_cache.pt` exists)
- [ ] T2U forward pass verification passed
- [ ] Training loss drops below 3.0
- [ ] ASR-BLEU > 10 on benchmark
- [ ] Audio samples sound intelligible
- [ ] Model saved to Drive
- [ ] Final results table generated

**If all checked**: Congratulations! Your compressed SeamlessM4T model is ready! 🎉
