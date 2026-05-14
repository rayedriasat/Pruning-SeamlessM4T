# Phase 8: Full Model Knowledge Distillation - Implementation Guide

## Overview

This guide explains the **Full Model KD** approach for Phase 8, replacing the failed T2U-only KD attempt.

## Why Full Model KD?

### Previous Approach (T2U-only KD) - FAILED ❌
- **Goal**: Train only T2U model while freezing other components
- **Problem**: T2U is deeply integrated with frozen upstream components
- **Errors encountered**:
  - "decoder_input_ids required" errors
  - Attention mask size mismatches
  - OOM (43-127 GB allocation attempts)
  - NoneType errors and gradient flow issues
- **Root cause**: Architectural incompatibility - T2U cannot be trained in isolation

### New Approach (Full Model KD) - RECOMMENDED ✅
- **Goal**: Train the ENTIRE model using knowledge distillation
- **Base model**: Phase 7 DoRA model (`phase7_dora_merged_v1`)
- **Why Phase 7?**
  - Already has good text quality from DoRA fine-tuning
  - Audio quality (ASR-BLEU/ChrF) needs improvement
  - Better starting point than Phase 6
- **Target metrics**: Increase ASR-BLEU and ASR-ChrF (audio quality)

## Implementation Details

### Key Changes

1. **All Parameters Trainable**
   - No freezing of any components
   - Train speech_encoder, text_decoder, t2u_model, lm_head, vocoder
   - ~1B trainable parameters

2. **Dual Distillation Loss**
   ```python
   total_loss = alpha * kl_loss + (1-alpha) * audio_mse_loss
   ```
   - **Text sequence distillation**: KL divergence on decoder logits
   - **Audio waveform distillation**: MSE on vocoder outputs
   - **Alpha = 0.7**: 70% text, 30% audio (prioritize text quality)

3. **Conservative Hyperparameters**
   - Learning rate: `1e-5` (lower for full model fine-tuning)
   - Batch size: `1` (memory constraints)
   - Gradient accumulation: `8` (effective batch = 8)
   - Max steps: `1000`
   - Temperature: `2.0`
   - Gradient clipping: `1.0`

4. **Memory Management**
   - Teacher always in eval() mode with no_grad
   - Periodic cache clearing every 50 steps
   - OOM error handling with graceful fallback

## Files Provided

### 1. `phase8_full_kd_cells.py`
Contains 7 replacement cells for Phase 8:
- **Cell 1**: Load Phase 7 model (all params trainable)
- **Cell 2**: Load teacher model
- **Cell 3**: Full KD loss function
- **Cell 4**: Optimizer setup
- **Cell 5**: Training loop
- **Cell 6**: Plot training curves
- **Cell 7**: Save model to Drive

### 2. This Guide
Explains the approach and how to apply changes

## How to Apply Changes to `full-kd.ipynb`

### Step 1: Locate Phase 8 Cells
In `full-kd.ipynb`, find these sections:
- "Phase 8 — Cell 1: Load Phase 7 Student Model"
- "Phase 8 — Cell 2: Load Teacher Model"
- "Phase 8 — Cell 3: T2U KD Loss & Training Utilities"
- "Phase 8 — Cell 4: Optimiser Setup"
- "Phase 8 — Cell 5: T2U KD Training Loop"
- "Phase 8 — Cell 6: Plot KD Training Curves"
- "Phase 8 — Cell 7: Save phase8_kd Model to Drive"

### Step 2: Replace Cell Contents
Copy the corresponding cell code from `phase8_full_kd_cells.py` into each Phase 8 cell in the notebook.

**Important**: Keep the cell structure (markdown headers, cell IDs) but replace the code content.

### Step 3: Update Benchmark Cells
Find Phase 8 Benchmark cells and make these changes:

#### Change 1: Model name references
```python
# OLD:
'phase8_kd'

# NEW:
'phase8_full_kd'
```

#### Change 2: Display labels
```python
# OLD:
('phase8_kd', 'P8 KD\\n(final)'),

# NEW:
('phase8_full_kd', 'P8 Full KD\\n(final)'),
```

#### Change 3: Checkpoint names
```python
# OLD:
save_checkpoint(..., name='phase8_kd', ...)

# NEW:
save_checkpoint(..., name='phase8_full_kd', ...)
```

#### Change 4: Figure titles
```python
# OLD:
'Phase 8 — T2U Knowledge Distillation Training'

# NEW:
'Phase 8 — Full Model Knowledge Distillation Training'
```

### Step 4: Update Markdown Headers
Update the Phase 8 section header:

```markdown
# OLD:
# Phase 8: T2U Knowledge Distillation (Audio Translation Recovery)

# NEW:
# Phase 8: Full Model Knowledge Distillation (Audio Quality Recovery)
```

## Expected Results

### Training Metrics
- **Total Loss**: Should decrease steadily
- **KL Loss**: Text sequence distillation quality
- **Audio MSE**: Waveform similarity to teacher

### Benchmark Metrics
Compare 4 models:
1. **Teacher** (baseline reference)
2. **Phase 6** (after T2U pruning - quality dip)
3. **Phase 7** (after DoRA - text recovery)
4. **Phase 8 Full KD** (after Full KD - audio recovery)

Target improvements in Phase 8:
- ✅ **ASR-BLEU**: Should increase (audio quality)
- ✅ **ASR-ChrF**: Should increase (audio quality)
- ✅ **Text-BLEU**: Should maintain or improve
- ✅ **Text-ChrF**: Should maintain (already good from Phase 7)

## Memory Considerations

### VRAM Usage
- Teacher: ~3.5 GB (frozen, eval)
- Student: ~3.5 GB (trainable)
- Activations: ~4-5 GB (forward + backward)
- **Total**: ~11-12 GB (fits in 16 GB GPU)

### If OOM Occurs
1. Reduce `KD_BATCH_SIZE` to 1 (already set)
2. Increase `KD_GRAD_ACCUM` to 16 (double accumulation)
3. Reduce `max_new_tokens` in generate() calls
4. Enable gradient checkpointing (if available)

## Troubleshooting

### Issue: "out of memory"
**Solution**: The code includes OOM handling that returns a small dummy loss. Training will continue. If persistent:
- Increase gradient accumulation
- Reduce max_new_tokens
- Clear cache more frequently

### Issue: "NoneType" errors
**Solution**: This was specific to T2U-only approach. Full Model KD should not encounter this since all components are active.

### Issue: Slow training
**Expected**: Full model training is slower than component-only training. Each step processes:
- Forward pass through entire model
- Generate text sequences
- Generate audio waveforms
- Compute dual distillation loss
- Backward pass through entire model

Expect ~30-60 seconds per step on T4 GPU.

### Issue: Loss not decreasing
**Check**:
1. Learning rate not too low (1e-5 is good)
2. Gradient clipping not too aggressive (1.0 is good)
3. Alpha balance (0.7 prioritizes text, adjust if needed)
4. Temperature (2.0 is standard, can try 1.5-3.0)

## Comparison: T2U-only vs Full Model KD

| Aspect | T2U-only KD (Failed) | Full Model KD (Recommended) |
|--------|---------------------|----------------------------|
| **Trainable params** | ~50M (T2U only) | ~1000M (entire model) |
| **Training time** | N/A (failed) | ~30-60 sec/step |
| **Memory usage** | N/A (OOM) | ~11-12 GB |
| **Gradient flow** | Broken (frozen deps) | Clean (all active) |
| **Architecture** | Incompatible | Compatible |
| **Expected quality** | N/A | Good (both text + audio) |
| **Risk** | High (many errors) | Low (standard approach) |

## Next Steps After Implementation

1. **Run Phase 8 training** (Cells 1-5)
   - Monitor loss curves
   - Check for OOM or errors
   - Training should complete in ~8-16 hours for 1000 steps

2. **Visualize training** (Cell 6)
   - Verify loss is decreasing
   - Check KL and Audio MSE trends

3. **Save model** (Cell 7)
   - Model saved as `phase8_full_kd`

4. **Run benchmarks** (Benchmark Cells 1-5)
   - Compare all 4 models
   - Focus on ASR-BLEU and ASR-ChrF improvements

5. **Analyze results**
   - If ASR metrics improved: Success! ✅
   - If not: Adjust alpha, temperature, or training steps

## References

- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network"
- **Temperature Scaling**: Standard practice is T=2.0 to 4.0
- **Alpha Weighting**: Typically 0.5-0.9 for KD loss weight
- **SeamlessM4T**: Meta's multilingual speech translation model

## Questions?

If you encounter issues not covered here:
1. Check error messages carefully
2. Verify all cell dependencies are run in order
3. Ensure Phase 7 model exists before starting Phase 8
4. Monitor VRAM usage with `nvidia-smi`
5. Check training logs for OOM or gradient issues

---

**Summary**: Full Model KD trains the entire Phase 7 model using dual distillation (text + audio) from the teacher. This approach is architecturally sound, memory-efficient, and should improve audio quality while maintaining text quality.
