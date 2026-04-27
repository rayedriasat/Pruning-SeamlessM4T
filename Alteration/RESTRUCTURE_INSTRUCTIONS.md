# Instructions for Restructuring the Notebook

## Overview
The notebook `pragmata copy.ipynb` needs to be restructured to replace Phases 4-6 (CIF connector approach) with new Phases 4-6 (additional pruning approach).

## What to Change

### Current Structure (to be replaced):
- **Phase 4:** Text Decoder Removal + CIF Connector + Speaker Adapter
- **Phase 5:** KD Target Extraction from Teacher
- **Phase 6a:** CIF Connector + Speaker Adapter Feature KD Training
- **Phase 6b:** End-to-End Fine-tuning with DoRA

### New Structure (replacement):
- **Phase 4:** Speech Encoder Additional Pruning (16L → 12L)
- **Phase 5:** Text Decoder Pruning (24L → 12L)
- **Phase 6:** DoRA Fine-tuning (Optional Recovery)

## Step-by-Step Instructions

### Option 1: Manual Editing (Recommended for Jupyter Notebook)

1. **Open the notebook** in Jupyter/JupyterLab/Kaggle

2. **Locate Phase 4-6 sections** (search for markdown cells with "## Phase 4", "## Phase 5", "## Phase 6")

3. **Delete all cells** from Phase 4 start to Phase 6 end (before Phase 7)

4. **Insert new cells** using the code from `NEW_PHASES_4_5_6.md`:
   - Copy Phase 4 markdown + code cells
   - Copy Phase 5 markdown + code cells
   - Copy Phase 6 markdown + code cells

5. **Update Phase 7** to reference the new final model:
   - Change `load_textless_model_from_drive('phase6b_textless')` 
   - To: `load_model_from_drive('phase6_dora_merged')`

6. **Save the notebook**

### Option 2: Python Script Approach

If you prefer automated editing, I can create a Python script that:
1. Reads the notebook JSON
2. Identifies Phase 4-6 cells
3. Replaces them with new cells
4. Saves the modified notebook

Let me know if you want this approach.

## Key Changes Summary

### Removed Concepts:
- ❌ CIF Connector (not viable without removing text decoder)
- ❌ Speaker Adapter (voice cloning out of scope)
- ❌ Textless architecture (keeping text decoder)
- ❌ KD target extraction from teacher
- ❌ Feature distillation training

### Added Concepts:
- ✅ Additional speech encoder pruning (16L → 12L)
- ✅ Text decoder pruning (24L → 12L)
- ✅ BI-guided iterative greedy for decoder
- ✅ DoRA fine-tuning for quality recovery
- ✅ Balanced encoder + decoder pruning

## Expected Results

### Model Size:
- **Before (Phase 3):** ~542M params (Enc16L + Dec24L + T2U 4+4L)
- **After (Phase 6):** ~818M params (Enc12L + Dec12L + T2U 4+4L)
- **Reduction from teacher:** 55% (1805M → 818M)

### Quality Target:
- Maintain ASR-ChrF > 35 across all language pairs
- RTF < 0.15 (faster than V1 baseline)
- Balanced quality across 5 languages

## Files Reference

- **NEW_PHASES_4_5_6.md** - Complete code for new phases
- **PLAN.md** - Original plan (for reference, now outdated)
- **pragmata copy.ipynb** - Notebook to be modified

## Testing After Changes

After restructuring, test the pipeline:

```python
# Quick test sequence
model_p3, processor = load_model_from_drive('phase3_t2u_laco')
print_model_breakdown(model_p3, 'Phase 3 Starting Point')

# Verify Phase 3 has:
# - Speech Encoder: 16 layers
# - Text Decoder: 24 layers
# - T2U: 4+4 layers

# Then run Phase 4-6 cells sequentially
```

## Notes

1. **Checkpoint compatibility:** Old Phase 4-6 checkpoints are incompatible with new phases. Delete them:
   ```bash
   rm -rf checkpoints/phase4*.pt
   rm -rf checkpoints/phase5*.pt
   rm -rf checkpoints/phase6*.pt
   ```

2. **Model files:** Old phase4-6 model files on Drive are also incompatible. They will be replaced by new training.

3. **Evaluation samples:** The `eval_samples` and `ft_samples` remain unchanged and compatible.

4. **Helper functions:** All existing helper functions (BI computation, iterative pruning, etc.) are reused with minor adaptations for decoder.

## Questions?

If you encounter issues:
1. Check that Phase 3 model loads correctly
2. Verify speech encoder has 16 layers (not 24)
3. Verify text decoder has 24 layers (not pruned yet)
4. Ensure all helper functions are defined before Phase 4
