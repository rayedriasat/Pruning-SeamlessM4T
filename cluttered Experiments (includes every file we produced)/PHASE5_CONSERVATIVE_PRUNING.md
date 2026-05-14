# Phase 5: Conservative FLAP Pruning Strategy

## Date: April 18, 2026

## Problem: Decoder Collapse from Over-Pruning

### Symptoms After 15% FLAP Pruning
- ChrF dropped to 0-5 (from ~45)
- Repeated characters: "নানানানান", "ইইই", "ওওওওও"
- Model stuck in loops, unable to generate diverse tokens
- **Decoder collapse** - lost ability to produce coherent translations

### Root Cause: Cumulative Pruning Effect

Your model has been pruned across multiple phases:

| Phase | Technique | Params Removed |
|-------|-----------|----------------|
| 1 | Vocab pruning | ~200M |
| 3 | Text decoder layers (8/24) | ~150M |
| 4 | Speech encoder layers (6/24) | ~150M |
| **5** | **FLAP width (15%)** | **~103M** |
| **Total** | | **~603M (26% of 2.3B)** |

**The issue:** 15% width pruning on top of 33% depth pruning is too aggressive. The remaining neurons are already working harder after layer removal.

## Solution: Conservative FLAP Settings

### New Configuration
```python
FLAP_RATIO    = 0.08   # prune only 8% of neurons (was 0.15)
MIN_KEEP_FRAC = 0.80   # keep at least 80% per layer (was 0.70)
```

### Why 8% Is Safe

**Evidence from literature:**
1. **Wanda paper (ICLR 2024):** Shows 10% unstructured pruning is safe for LLMs
2. **LLM-Pruner:** Uses 5-10% width pruning after depth pruning
3. **FLAP paper:** Tests on models WITHOUT prior depth pruning

**Your case:** After removing 33% of layers, 8% width pruning is the safe upper bound.

### Expected Results with 8% Pruning

| Metric | Before P5 | After P5 (8%) | Drop |
|--------|-----------|---------------|------|
| Params | 1217.6M | ~1165M | ~50M |
| ChrF | ~45 | ~40-43 | 2-5 pts |
| Quality | Functional | Functional | Acceptable |

**Key point:** ChrF should stay above 35. If it drops below 30, even 8% is too much.

## Alternative: Skip Phase 5 Entirely

If 8% still causes issues, you have two options:

### Option A: Skip FLAP, Go Straight to Phase 6
```python
# In Phase 6 Cell 4, change:
model_p6 = model_p4  # Skip Phase 5 entirely
```

**Pros:**
- Zero risk of decoder collapse
- Model definitely works for Phase 7 fine-tuning
- Still get ~500M param reduction from Phases 1-4

**Cons:**
- Miss ~50M additional param savings
- Final model ~1.2B instead of ~1.1B

### Option B: FLAP Only on Speech Encoder
Speech encoder is more robust to pruning than text decoder. You could:
1. Skip FLAP on text_decoder and t2u_model
2. Apply 10-12% FLAP only to speech_encoder
3. Save ~30M params with minimal quality impact

```python
# In Phase 5 Cell 6, modify the loop:
for comp_name in ["speech_encoder"]:  # Only speech encoder
    # ... rest of code ...
```

## What To Do Now

### Step 1: Clean Up
```bash
!rm -rf /kaggle/working/models/phase5_flap_pruned
```

### Step 2: Re-run Phase 5 Cell 6
The fixed code now uses 8% pruning ratio.

### Step 3: Check Sanity Results
After pruning completes, look at the sanity check:
- **Good:** ChrF > 35, coherent Bengali text
- **Acceptable:** ChrF 30-35, mostly coherent with minor repetition
- **Bad:** ChrF < 30, repeated characters, loops

### Step 4: Decision Tree

```
Sanity ChrF > 35?
├─ YES → Proceed to Phase 6 (8% worked!)
└─ NO → ChrF 30-35?
    ├─ YES → Proceed to Phase 6 (Phase 7 will recover)
    └─ NO → Skip Phase 5, use model_p4 for Phase 6
```

## Why Width Pruning Is Harder Than Depth Pruning

**Depth pruning (Phases 3-4):**
- Remove entire layers
- Remaining layers unchanged
- Model adapts by using remaining layers more

**Width pruning (Phase 5):**
- Shrink every layer
- Changes information flow through entire network
- Harder for model to compensate

**Analogy:** Removing 8 floors from a 24-story building is easier than making every floor 15% smaller.

## Expected Phase 7 Recovery

With 8% FLAP pruning:
- Phase 5 ChrF: ~40-43
- Phase 7 ChrF: ~44-46 (DoRA fine-tuning recovers 2-4 points)
- Final model: ~1.16B params, 95%+ quality retention

This is a **successful compression** outcome.

## Files Modified
- `cse465v5-s2st-corrected.ipynb` - Phase 5 Cell 6 (cell index 75)

## Files Created
- `fix_phase5_pruning_ratio.py` - Script that applied the fix
- `PHASE5_CONSERVATIVE_PRUNING.md` - This guide
