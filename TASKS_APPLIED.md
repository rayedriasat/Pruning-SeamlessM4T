# Tasks Applied to seamless-final.ipynb

## ✅ Task 1: Checkpoint Key Fixes (COMPLETE)
All 6 locations fixed to use consistent key names:
- `'cif_state'` → `'cif_connector'`
- `'spk_state'` → `'speaker_adapter'`

## ✅ Task 2: Phase 6a Extended Training Configuration (COMPLETE)

### Changes Applied:

1. **MAX_STEPS increased**: 5000 → 10000
   - Line ~5336: `MAX_STEPS_P6A = 10000  # EXTENDED from 5000`

2. **Connector LR increased**: 1e-4 → 2e-4
   - Line ~5360: `'lr': 2e-4` for cif_connector parameters

3. **Loss weights adjusted**:
   - Cosine: 0.50 → 0.40 (lower for better convergence)
   - MSE: 0.20 (unchanged)
   - Quantity: 0.25 → 0.30 (higher for better signal)
   - Speaker reg: 0.05 → 0.10 (higher)
   - Line ~5577-5580

4. **Print statement updated** to reflect new configuration
   - Line ~5429-5431

### Expected Results:
- Cosine loss should converge to < 0.10 by step 10000
- Better quantity prediction accuracy
- More stable training with adjusted weights

## ⏳ Task 3: Phase 6b Textless Architecture Rewrite (NOT YET APPLIED)

### What Needs to Be Done:
Replace Phase 6b cells (Cell 8, 9, 10) with code from `Alteration/phase6_fixes.py` lines 268-520:

1. **Cell 8 (DoRA Setup)**: Apply DoRA only to speech_encoder + t2u_model (NOT text_decoder)
2. **Cell 9 (Training Loop)**: Train with unit CE loss, real speech encoder forward every step
3. **Cell 10 (Merge and Save)**: Merge DoRA adapters and save with correct keys

### Why This Is Important:
The current Phase 6b code was copied from `only-p7-dora.ipynb` which trains models WITH text_decoder. The textless model has NO text_decoder (surgically removed in Phase 4), so the current code will fail.

### Next Steps:
Apply the Phase 6b fixes from `phase6_fixes.py` to complete the transformation.

---

**Status Summary:**
- ✅ Task 1: Complete (checkpoint keys fixed)
- ✅ Task 2: Complete (extended training config applied)
- ⏳ Task 3: Pending (Phase 6b textless rewrite needed)

**Date:** 2026-04-25
**File:** `Alteration/seamless-final.ipynb`
