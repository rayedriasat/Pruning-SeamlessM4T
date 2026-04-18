# Phase 8: T2U NAR Training for Audio Recovery

This is a standalone notebook that continues from Phase 7 (DoRA fine-tuned model) and trains the T2U (Text-to-Unit) model using Non-Autoregressive (NAR) architecture to recover audio translation quality.

## What Phase 8 Does

**Phase 7 recovered text translation quality** (BLEU/ChrF) by fine-tuning the speech encoder → text decoder path with DoRA adapters.

**Phase 8 recovers audio translation quality** by training the T2U model (text decoder → discrete speech units → vocoder → audio waveform).

### Why T2U Training is Separate

The T2U model in SeamlessM4Tv2 uses a **Non-Autoregressive (NAR) architecture** that requires:
1. **Discrete unit labels** extracted from target Bengali audio
2. **Duration prediction** (how many units per text token)
3. **Specialized loss functions** (unit cross-entropy + duration loss)
4. **Different training dynamics** than standard seq2seq

This is fundamentally different from the S2TT cross-entropy loss used in Phase 7, so we train it separately.

## Training Strategy

### Inputs
- **Source**: English speech (from `ft_samples`)
- **Target units**: Discrete speech units extracted from Bengali audio using SeamlessM4T's unit extractor
- **Text tokens**: Bengali text tokens (for duration alignment)

### Loss Components
1. **Unit Cross-Entropy**: Predict correct discrete speech units
2. **Duration Loss** (optional): Predict how many units per text token

### Architecture
- **Frozen**: Speech encoder (already fine-tuned in Phase 7)
- **Trainable**: T2U encoder + T2U decoder (NAR transformer)

## Expected Results

- **ASR-BLEU recovery**: From ~0-5 (Phase 6) → 15-25+ (Phase 8)
- **ASR-ChrF recovery**: From ~0-10 (Phase 6) → 30-45+ (Phase 8)
- **Audio quality**: Intelligible Bengali speech output
- **Text quality**: Maintained from Phase 7 (BLEU ~40-50, ChrF ~60-70)

---

## Implementation Notes

### Cell Structure
1. **Cells 1-23**: Setup (identical to Phase 7) - platform detection, I/O helpers, dataset loading
2. **Cell 24**: Load Phase 7 fine-tuned model
3. **Cell 25**: Load or extract unit labels (reuse Phase 7 cache)
4. **Cell 26**: T2U-specific data preparation
5. **Cell 27**: T2U loss functions (NAR-specific)
6. **Cell 28**: Training loop (T2U only, speech encoder frozen)
7. **Cell 29**: Loss curve plotting
8. **Cell 30**: Save Phase 8 model
9. **Cell 31**: Full S2ST benchmark with ASR-BLEU

### Key Differences from Phase 7
- **No DoRA/LoRA**: Direct parameter updates on T2U
- **Frozen speech encoder**: Only T2U trains
- **Unit-level supervision**: Cross-entropy on discrete units, not text tokens
- **Shorter training**: ~500-1000 steps (T2U is smaller than full model)

---

## Quick Start

1. **Ensure Phase 7 is complete**: `phase7_dora_merged` model must exist in Drive
2. **Run all setup cells** (1-23): Same as Phase 7
3. **Run Phase 8 cells** (24-31): T2U training + benchmark
4. **Check ASR-BLEU**: Should see significant audio quality recovery

---

## Troubleshooting

### "Unit cache not found"
- Phase 7 Cell 6 must have run successfully
- Check `{CKPT_DIR}/unit_labels_cache.pt` exists
- If missing, re-run Phase 7 Cell 6 to extract units

### "T2U forward pass fails"
- Verify unit sequence lengths: min 3, max ~500
- Check unit vocab size matches model config
- Ensure text decoder output is passed to T2U encoder

### "Audio output still silent"
- Check T2U loss is decreasing (should drop from ~8 to ~2-3)
- Verify vocoder is not replaced with NoOp
- Run benchmark with `save_n=4` to save audio clips

### "Out of memory"
- Reduce `BATCH_SIZE` from 2 to 1
- Increase `GRAD_ACCUM` from 4 to 8
- Use `torch.cuda.empty_cache()` between batches

---

## Next Steps After Phase 8

1. **Final benchmark**: Run full S2ST evaluation on test set
2. **Compare all phases**: Plot Phase 0 → Phase 8 progression
3. **Paper table**: Generate LaTeX table with all metrics
4. **Audio samples**: Save 10-20 audio clips for qualitative evaluation
5. **Model export**: Save final compressed model for deployment

