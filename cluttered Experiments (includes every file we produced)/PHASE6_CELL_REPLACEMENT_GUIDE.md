# Phase 6 Cell Replacement Guide

## Quick Reference: What to Replace

### Phase 6a: Replace 1 Cell

**Find this cell** (around line 5321 in seamless-final.ipynb):
```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 6a: CIF Connector + Speaker Adapter Training (FIXED v4)              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Hyperparameters ────────────────────────────────────────────────────────────
MAX_STEPS_P6A = 5000  # ← OLD VALUE
BATCH_SIZE    = 8
...
print('\n✓ Phase 6a training complete!')
```

**Replace with** (from phase6_fixes.py, lines 28-265):
```python
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6A: EXTENDED TRAINING (CELL 1 - Replace your Phase 6a training cell)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hyperparameters (IMPROVED) ────────────────────────────────────────────────
MAX_STEPS_P6A = 10000  # ← NEW VALUE (DOUBLED)
BATCH_SIZE    = 8
...
print(f'  Target: < 0.10 ({"CONVERGED" if feat_log_6a[-1] < 0.10 else "needs more training"})')
```

---

### Phase 6b: Replace 3 Cells

#### Cell 1: DoRA Setup (around line 5803)

**Find this cell**:
```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8 — Phase 6b: DoRA E2E Fine-tuning — CORRECTED                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from peft import LoraConfig, get_peft_model

print('Loading Phase 6a model for DoRA fine-tuning...')
model_6b = model_6a
...
```

**Replace with** (from phase6_fixes.py, lines 268-350):
```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 6b: DoRA E2E Fine-tuning (FIXED FOR TEXTLESS MODEL)                  ║
# ║  - Apply DoRA to speech_encoder + t2u_model ONLY (no text_decoder)          ║
# ║  - Train with unit CE loss (T2U generates units, not text)                   ║
# ║  - Based on working only-p7-dora.ipynb but adapted for textless arch         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print('Loading Phase 6a model for Phase 6b DoRA fine-tuning...')
model_6b = model_6a  # Already in memory with trained CIF + speaker adapter
...
```

#### Cell 2: Training Loop (around line 5895)

**Find this cell**:
```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9 — Phase 6b Training Loop                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MAX_STEPS_E2E = 2500
...
print('Phase 6b training complete.')
```

**Replace with** (from phase6_fixes.py, lines 353-490):
```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 6b Training Loop (UNIT CE LOSS - for textless model)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MAX_STEPS_E2E = 2500
BATCH_ACCUM   = 4
...
print('Phase 6b training complete.')
```

#### Cell 3: Merge and Save (around line 6079)

**Find this cell**:
```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10 — Phase 6b: Merge DoRA + Save Final Model                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print('Merging DoRA adapters into base weights...')
...
```

**Replace with** (from phase6_fixes.py, lines 493-520):
```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 6b: Merge DoRA + Save Final Model                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print('Merging DoRA adapters into base weights...')
model_6b.speech_encoder = model_6b.speech_encoder.merge_and_unload()
model_6b.t2u_model      = model_6b.t2u_model.merge_and_unload()
...
print('\n✓ Final ~673M textless model saved to Drive.')
```

---

## Visual Summary

```
seamless-final.ipynb
│
├─ Phase 6a (1 cell to replace)
│  └─ Training cell (~240 lines)
│     ├─ OLD: MAX_STEPS=5000, LR=1e-4, cosine_weight=0.50
│     └─ NEW: MAX_STEPS=10000, LR=2e-4, cosine_weight=0.40
│
└─ Phase 6b (3 cells to replace)
   ├─ Cell 1: DoRA setup (~80 lines)
   │  ├─ OLD: Applies DoRA to text_decoder (doesn't exist!)
   │  └─ NEW: Applies DoRA to speech_encoder + t2u_model only
   │
   ├─ Cell 2: Training loop (~140 lines)
   │  ├─ OLD: Uses cached embeddings
   │  └─ NEW: Real speech encoder forward every step
   │
   └─ Cell 3: Merge and save (~30 lines)
      ├─ OLD: Standard merge
      └─ NEW: Merge speech_encoder + t2u_model separately
```

---

## Key Differences Summary

### Phase 6a Changes:
| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| MAX_STEPS | 5000 | 10000 | Need more training for convergence |
| Connector LR | 1e-4 | 2e-4 | Faster learning |
| Cosine weight | 0.50 | 0.40 | Reduce emphasis on cosine |
| Qty weight | 0.25 | 0.30 | Increase emphasis on quantity |

### Phase 6b Changes:
| Aspect | Old Approach | New Approach | Reason |
|--------|--------------|--------------|--------|
| DoRA scope | text_decoder + t2u | speech_encoder + t2u | text_decoder doesn't exist |
| Loss function | Text CE | Unit CE | Textless model generates units |
| Encoder forward | Cached embeddings | Real forward pass | DoRA needs gradients |
| Multi-GPU layout | Standard | Custom for textless | Optimize for architecture |

---

## Verification Checklist

After replacing cells, verify:

- [ ] Phase 6a cell has `MAX_STEPS_P6A = 10000`
- [ ] Phase 6a cell has connector LR `2e-4`
- [ ] Phase 6a cell has loss weights `0.40, 0.20, 0.30, 0.10`
- [ ] Phase 6b Cell 1 applies DoRA to `speech_encoder` and `t2u_model` only
- [ ] Phase 6b Cell 1 does NOT mention `text_decoder`
- [ ] Phase 6b Cell 2 has real speech encoder forward pass
- [ ] Phase 6b Cell 2 uses `unit_loss` (not `text_loss`)
- [ ] Phase 6b Cell 3 merges both `speech_encoder` and `t2u_model`

---

## Copy-Paste Workflow

1. Open `Alteration/phase6_fixes.py` in one window
2. Open `Alteration/seamless-final.ipynb` in another window
3. For Phase 6a:
   - Find the training cell (search for "Phase 6a: CIF Connector")
   - Select entire cell
   - Copy lines 28-265 from phase6_fixes.py
   - Paste to replace
4. For Phase 6b:
   - Find Cell 8 (search for "CELL 8 — Phase 6b")
   - Copy lines 268-350 from phase6_fixes.py
   - Paste to replace
   - Find Cell 9 (search for "CELL 9 — Phase 6b")
   - Copy lines 353-490 from phase6_fixes.py
   - Paste to replace
   - Find Cell 10 (search for "CELL 10 — Phase 6b")
   - Copy lines 493-520 from phase6_fixes.py
   - Paste to replace
5. Save notebook
6. Run cells in order

---

## Line Number Reference

In `phase6_fixes.py`:
- **Phase 6a code**: Lines 28-265 (238 lines)
- **Phase 6b Cell 1**: Lines 268-350 (83 lines)
- **Phase 6b Cell 2**: Lines 353-490 (138 lines)
- **Phase 6b Cell 3**: Lines 493-520 (28 lines)

In `seamless-final.ipynb` (approximate):
- **Phase 6a cell**: Around line 5321
- **Phase 6b Cell 8**: Around line 5803
- **Phase 6b Cell 9**: Around line 5895
- **Phase 6b Cell 10**: Around line 6079

---

## Done!

After replacing these 4 cells (1 for Phase 6a, 3 for Phase 6b), your notebook will be ready to:
1. Resume Phase 6a training from step 5000 → 10000
2. Run Phase 6b DoRA training correctly on the textless model
3. Merge adapters and save the final ~673M model
4. Proceed to Phase 7 benchmark
