# Phase 6a Training Divergence - Root Cause & Solution

## **TL;DR**

Your Phase 6a training is diverging because the **basic CIF connector is architecturally too weak** to replace the 867M text decoder. The learning rate is NOT the problem.

**Solution**: Replace basic CIF (2-layer, 5M params) with **EnhancedCIFConnector** (6-layer + cross-attention, 15M params).

---

## **Training Log Analysis**

```
Step 2950/5000 | loss=5.78  | cos=0.4695 | qty_err=4.6 | fired=35 vs tgt=32
Step 3000/5000 | loss=9.99  | cos=0.4732 | qty_err=5.6 | fired=43 vs tgt=36
[VAL]          | loss=12.28 | cos=0.5054 | qty_err=7.5
Step 3050/5000 | loss=10.38 | cos=0.4880 | qty_err=6.0 | fired=49 vs tgt=37
Step 3100/5000 | loss=15.83 | cos=0.4633 | qty_err=6.1 | fired=25 vs tgt=32
Step 3500/5000 | loss=7.00  | cos=0.4793 | qty_err=5.9 | fired=42 vs tgt=50
[VAL]          | loss=15.15 | cos=0.4821 | qty_err=7.8
```

### **Key Observations:**

1. **Cosine loss stuck at 0.47-0.52** (target: <0.15)
   - This means CIF output is only ~50% similar to text decoder output
   - No improvement over 3500 steps - model is NOT learning

2. **Loss is diverging** (5.78 → 15.83 → 15.15)
   - Not converging, just oscillating
   - Validation loss WORSE than training loss

3. **Quantity prediction is reasonable** (fired ≈ target ± 10)
   - CIF firing mechanism works
   - But the FEATURES are wrong

4. **Learning rate is fine** (1.97e-04)
   - Not too high (no explosion)
   - Not too low (updates are happening)
   - Problem is NOT optimization

---

## **Root Cause: Architectural Capacity Mismatch**

### **What You're Trying to Do:**

Replace the **text decoder** (867M params, 24 layers) with **CIF connector** (5M params, 2 layers).

### **Why Basic CIF Cannot Work:**

| Component | Text Decoder | Basic CIF | Gap |
|---|---|---|---|
| **Layers** | 24 transformer layers | 2 refiner layers | **12× fewer** |
| **Parameters** | 867M | 5M | **174× fewer** |
| **Cross-attention** | ✓ To speech encoder | ✗ None | **Missing** |
| **Self-attention** | ✓ 24 layers | ✓ 2 layers | **Weak** |
| **Capacity** | Can learn complex language-specific representations | Can only do basic feature smoothing | **Insufficient** |

### **The Critical Missing Piece: Cross-Attention**

The text decoder has **cross-attention to the speech encoder** in every layer. This allows it to:
- Align target language generation with source speech features
- Attend to relevant acoustic frames for each output token
- Learn language-specific acoustic-to-semantic mappings

**Basic CIF has NO cross-attention.** It only sees the speech encoder output once, then tries to compress it. It cannot learn the rich alignment patterns the text decoder uses.

---

## **Why Cosine Loss is Stuck at 0.49**

Cosine similarity of 0.49 means:
- CIF output and text decoder output are **roughly 50% aligned**
- This is barely better than random (0.0 would be orthogonal, 1.0 would be identical)
- The model has learned SOME basic patterns but cannot go further

**Why it's stuck:**
1. 2-layer refiner cannot learn 24-layer decoder's representation space
2. No cross-attention means CIF cannot align features properly
3. Model is at its **architectural capacity limit**

**Analogy**: You're trying to compress a 4K movie (text decoder) into a thumbnail (basic CIF). No amount of training will make the thumbnail contain all the movie's information.

---

## **Solution: Enhanced CIF Connector**

### **Architecture:**

```
Speech Encoder Output [B, T_frames, 1024]
    │
    ├─→ CIF Weight Predictor (unchanged)
    │       │
    │       ▼
    │   Weights [B, T_frames] → CIF Firing → [B, T_cif, 1024]
    │
    ├─→ Language Embedding [B, 1, 1024]
    │       │
    │       ▼
    │   Add to CIF features
    │
    ├─→ Cross-Attention Layers (NEW - 2 layers)
    │       │  Query: CIF features
    │       │  Key/Value: Speech encoder output
    │       │  → Mimics text decoder's encoder attention
    │       ▼
    │   Cross-attended features [B, T_cif, 1024]
    │
    ├─→ Refiner Transformer (6 layers, not 2)
    │       │  Self-attention + FFN
    │       ▼
    │   Refined features [B, T_cif, 1024]
    │
    └─→ Output Projection
            │
            ▼
        Final output [B, T_cif, 1024] → matches text decoder output space
```

### **Key Improvements:**

1. **6-layer refiner** (vs 2-layer)
   - 3× more capacity to learn feature transformations
   - Can approximate deeper text decoder representations

2. **2-layer cross-attention to speech encoder** (NEW)
   - Mimics text decoder's encoder-decoder attention
   - Allows CIF to align target features with source acoustics
   - **This is the critical missing piece**

3. **~15M params** (vs 5M basic)
   - Still 58× smaller than 867M text decoder
   - But enough capacity to learn the essential patterns

### **Expected Performance:**

| Metric | Basic CIF (current) | Enhanced CIF (expected) |
|---|---|---|
| Cosine loss @ step 1500 | 0.49 (stuck) | <0.30 (improving) |
| Cosine loss @ step 3000 | 0.49 (stuck) | <0.20 (good) |
| Cosine loss @ step 5000 | 0.49 (stuck) | <0.15 (target) |
| Validation loss | Diverging | Converging |

---

## **Implementation Steps**

### **1. Load Current Model**

```python
# Load your phase6a model (or restart from phase4 if phase6a is too corrupted)
model, processor = load_textless_model_from_drive('phase4_textless_pretrain')
model = _consolidate_to_single_gpu(model)
```

### **2. Replace CIF Connector**

```python
from enhanced_cif_fix import replace_cif_with_enhanced

# Replace basic CIF with enhanced version
model = replace_cif_with_enhanced(model, device='cuda:0')

# Verify
print(f"CIF params: {sum(p.numel() for p in model.cif_connector.parameters())/1e6:.1f}M")
```

### **3. Retrain Phase 6a with Enhanced CIF**

```python
# Freeze everything except CIF connector
for p in model.parameters():
    p.requires_grad_(False)
for p in model.cif_connector.parameters():
    p.requires_grad_(True)

# Optimizer - slightly lower LR for larger model
optimizer = torch.optim.AdamW(
    model.cif_connector.parameters(),
    lr=1e-4,  # vs 2e-4 before
    betas=(0.9, 0.98),
    weight_decay=0.01
)

# Warmup scheduler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=500)
cosine = CosineAnnealingLR(optimizer, T_max=4500, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[500])

# Training loop (same as before, but with enhanced CIF)
MAX_STEPS = 5000
for step in range(MAX_STEPS):
    # ... (same training code)
    
    # Loss: feature matching ONLY (no quantity loss in Phase 6a)
    loss = 0.70 * cosine_loss + 0.30 * mse_loss
    
    # Monitor quantity for debugging, but don't train on it
    if step % 100 == 0:
        print(f"Step {step} | cos={cosine_loss:.4f} | mse={mse_loss:.4f} | "
              f"qty_pred={qty_pred.mean():.1f} vs target={n_tokens:.1f}")
```

### **4. Success Criteria**

**After 1500 steps:**
- Cosine loss < 0.30 (vs 0.49 stuck)
- Validation loss decreasing

**After 3000 steps:**
- Cosine loss < 0.20
- MSE loss < 0.002

**After 5000 steps:**
- Cosine loss < 0.15 (target)
- Ready for Phase 6c (unit CE loss training)

---

## **Why This Will Work**

### **1. Grounded in Published Research**

- **S2UT (Lee et al., ACL 2022)**: Showed speech→units without text is viable
- **SeamlessExpressive (Meta 2023)**: Used similar connector architecture with cross-attention
- **CIF (Dong & Xu, ICASSP 2020)**: Proven length compression mechanism

### **2. Architectural Parity**

Enhanced CIF has the **same key components** as text decoder:
- ✓ Cross-attention to encoder (NEW)
- ✓ Multi-layer self-attention (6 layers)
- ✓ Language conditioning
- ✓ Sufficient parameter capacity (~15M)

### **3. Proven in Similar Systems**

UnitY2 (Meta's T2U model) uses a similar connector architecture:
- Encoder output → length adapter → decoder input
- Cross-attention between adapter and encoder
- Works at scale (trained on millions of samples)

---

## **What NOT to Do**

❌ **Don't just lower the learning rate**
- Problem is architectural, not optimization
- LR 1.97e-04 is already reasonable

❌ **Don't remove quantity loss entirely**
- You already did this (correct for Phase 6a)
- Quantity will be trained in Phase 6c with unit CE loss

❌ **Don't add more training data yet**
- 1600 samples is enough to see if architecture works
- If enhanced CIF converges, THEN add more data

❌ **Don't train for more steps with basic CIF**
- It's stuck at 0.49 - more steps won't help
- You're hitting architectural capacity limit

---

## **Next Steps**

1. **Implement enhanced CIF** (code provided in `enhanced_cif_fix.py`)
2. **Replace in model** (use `replace_cif_with_enhanced()`)
3. **Retrain Phase 6a** (5000 steps, ~4-6 hours on T4×2)
4. **Monitor cosine loss** - should drop below 0.30 by step 1500
5. **If successful** (cos <0.15), proceed to Phase 6c (unit CE loss)

---

## **Expected Timeline**

| Phase | Duration | Goal |
|---|---|---|
| Enhanced CIF implementation | 30 min | Add code to notebook |
| Phase 6a retraining | 4-6 hours | Cosine loss <0.15 |
| Phase 6c (unit CE loss) | 4-6 hours | Train T2U decoder |
| Phase 6d (speaker adapter) | 2-3 hours | Voice cloning |
| Phase 7 (full benchmark) | 3-4 hours | Final evaluation |
| **Total** | **~15-20 hours** | Complete textless model |

---

## **References**

- **PLAN.md**: Original architecture plan (Section 7, Phase 6a)
- **PLAN2.md**: Detailed diagnosis and corrected approach
- **enhanced_cif_fix.py**: Implementation of EnhancedCIFConnector
- **S2UT paper**: Lee et al., "Direct Speech-to-Speech Translation with Discrete Units", ACL 2022
- **SeamlessExpressive**: Meta, arXiv:2312.05187v1
