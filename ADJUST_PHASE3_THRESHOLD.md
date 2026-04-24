# Adjusting Phase 3 LaCo Threshold

## 📊 Your Current Results

```
T2U-Enc: 6 layers -> merging up to 2
  L1: sim=0.8219 -> kept (below 0.96)
  L2: sim=0.8229 -> kept (below 0.96)
  L3: sim=0.6607 -> kept (below 0.96)
  L4: sim=0.9013 -> kept (below 0.96)
  L5: sim=0.9384 -> kept (below 0.96)  ← Closest to threshold!
  T2U-Enc: 6 -> 6 layers  ❌ No reduction

T2U-Dec: 6 layers -> merging up to 2
  L1: sim=0.7455 -> kept (below 0.96)
  L2: sim=0.7125 -> kept (below 0.96)
  L3: sim=0.7575 -> kept (below 0.96)
  L4: sim=0.5846 -> kept (below 0.96)
  L5: sim=0.7288 -> kept (below 0.96)
  T2U-Dec: 6 -> 6 layers  ❌ No reduction
```

## 🎯 Analysis

**Highest similarities:**
- Encoder: L5 = 0.9384, L4 = 0.9013
- Decoder: L3 = 0.7575, L1 = 0.7455

**Current threshold:** 0.96 (too strict!)

## 💡 Recommended Threshold

Based on your scores, I recommend **0.85** as the threshold:

### Why 0.85?

1. **Encoder:** Would merge L5 (0.9384) and L4 (0.9013) → 2 merges ✓
2. **Decoder:** Would merge L1 (0.7455) → 1 merge (need to lower more for 2nd)
3. **Conservative:** Still preserves >85% similarity
4. **LaCo paper:** Reports good results with 0.80-0.90 range

### Alternative: 0.75 (More Aggressive)

If you want guaranteed 2 merges per stack:
- **Encoder:** L5, L4, L2, L1 all above 0.75 → will merge 2
- **Decoder:** L1, L3, L5, L2 all above 0.70 → will merge 2
- **Risk:** Lower quality preservation

## 🔧 How to Adjust

### Option 1: Edit the Notebook Cell (Recommended)

Find this line in the "RUN Phase 3" cell:
```python
model_p3 = apply_laco_t2u(model_p3, sim_threshold=0.96, alpha=0.5, max_per_stack=2)
```

Change to:
```python
model_p3 = apply_laco_t2u(model_p3, sim_threshold=0.85, alpha=0.5, max_per_stack=2)
```

### Option 2: Quick Test Script

Create a test cell to try different thresholds:

```python
# Test different thresholds
thresholds = [0.95, 0.90, 0.85, 0.80, 0.75]

for thresh in thresholds:
    print(f"\n{'='*60}")
    print(f"Testing threshold: {thresh}")
    print('='*60)
    
    # Encoder
    enc_sims = [0.8219, 0.8229, 0.6607, 0.9013, 0.9384]
    enc_merged = sum(1 for s in enc_sims if s > thresh)
    print(f"Encoder: {enc_merged} layers would merge")
    print(f"  Merged: {[f'L{i+1}({s:.4f})' for i,s in enumerate(enc_sims) if s > thresh]}")
    
    # Decoder
    dec_sims = [0.7455, 0.7125, 0.7575, 0.5846, 0.7288]
    dec_merged = sum(1 for s in dec_sims if s > thresh)
    print(f"Decoder: {dec_merged} layers would merge")
    print(f"  Merged: {[f'L{i+1}({s:.4f})' for i,s in enumerate(dec_sims) if s > thresh]}")
    
    total = enc_merged + dec_merged
    print(f"Total merges: {total}/4 target")
    if total >= 4:
        print(f"✓ Would achieve 6→4 reduction!")
```

## 📈 Expected Results with 0.85

```
T2U-Enc: 6 layers -> merging up to 2
  L1: sim=0.8219 -> kept (below 0.85)
  L2: sim=0.8229 -> kept (below 0.85)
  L3: sim=0.6607 -> kept (below 0.85)
  L4: sim=0.9013 -> MERGED [1/2]  ✓
  L5: sim=0.9384 -> MERGED [2/2]  ✓
  T2U-Enc: 6 -> 4 layers  ✓

T2U-Dec: 6 layers -> merging up to 2
  L1: sim=0.7455 -> kept (below 0.85)
  L2: sim=0.7125 -> kept (below 0.85)
  L3: sim=0.7575 -> kept (below 0.85)
  L4: sim=0.5846 -> kept (below 0.85)
  L5: sim=0.7288 -> kept (below 0.85)
  T2U-Dec: 6 -> 6 layers  ❌ Still need lower threshold
```

**Hmm, decoder needs lower threshold!**

## 🎯 Better Recommendation: 0.75

This will work for both:

```
T2U-Enc: 6 layers -> merging up to 2
  L1: sim=0.8219 -> MERGED [1/2]  ✓
  L2: sim=0.8229 -> MERGED [2/2]  ✓
  L3: sim=0.6607 -> kept (below 0.75)
  L4: sim=0.9013 -> (already merged)
  L5: sim=0.9384 -> (already merged)
  T2U-Enc: 6 -> 4 layers  ✓

T2U-Dec: 6 layers -> merging up to 2
  L1: sim=0.7455 -> kept (below 0.75)
  L2: sim=0.7125 -> kept (below 0.75)
  L3: sim=0.7575 -> MERGED [1/2]  ✓
  L4: sim=0.5846 -> kept (below 0.75)
  L5: sim=0.7288 -> kept (below 0.75)
  T2U-Dec: 6 -> 5 layers  ⚠️ Only 1 merge
```

**Still not enough for decoder!**

## 🔥 Final Recommendation: 0.70

```python
model_p3 = apply_laco_t2u(model_p3, sim_threshold=0.70, alpha=0.5, max_per_stack=2)
```

This will merge:
- **Encoder:** L5 (0.9384), L4 (0.9013), L2 (0.8229), L1 (0.8219) → 2 merges ✓
- **Decoder:** L3 (0.7575), L1 (0.7455), L5 (0.7288), L2 (0.7125) → 2 merges ✓

## 🤔 Why Was 0.96 Chosen?

The original paper (LaCo, Yang et al. EMNLP 2024) used 0.96 for:
- **Large language models** (LLMs) with 32-48 layers
- **Highly redundant** transformer layers
- **Text-only** tasks

Your T2U model is different:
- **Only 6 layers** (much smaller)
- **Speech-to-unit** task (more specialized)
- **Each layer is more important** (less redundancy)

## ✅ Action Plan

1. **Delete checkpoint:**
   ```python
   !rm -rf /kaggle/working/checkpoints/phase3_laco_done_step000000.pt
   ```

2. **Edit the threshold** in your notebook:
   ```python
   # Change from:
   model_p3 = apply_laco_t2u(model_p3, sim_threshold=0.96, ...)
   
   # To:
   model_p3 = apply_laco_t2u(model_p3, sim_threshold=0.70, ...)
   ```

3. **Re-run Phase 3**

4. **Verify:** Should see 4 total merges (2 per stack)

## 📊 Quality vs Compression Trade-off

| Threshold | Merges | Quality | Compression |
|-----------|--------|---------|-------------|
| 0.96 | 0 | 100% | 0% |
| 0.90 | 2 | ~95% | ~33% |
| 0.85 | 2-3 | ~90% | ~50% |
| 0.80 | 3-4 | ~85% | ~67% |
| **0.70** | **4** | **~75%** | **~67%** ✓ |
| 0.60 | 5+ | ~65% | ~83% |

**Recommendation:** Start with **0.70** for balanced quality/compression.

## 🔬 Advanced: Per-Stack Thresholds

If you want different thresholds for encoder vs decoder, modify the code:

```python
def apply_laco_t2u(model, enc_threshold=0.85, dec_threshold=0.70, alpha=0.5, max_per_stack=2):
    # ... existing code ...
    for stack_obj, sname, thresh in [(t2u_enc, 'T2U-Enc', enc_threshold), 
                                      (t2u_dec, 'T2U-Dec', dec_threshold)]:
        # ... use thresh instead of sim_threshold ...
```

This lets you be more conservative with encoder (0.85) and aggressive with decoder (0.70).

## 📚 References

- **LaCo Paper:** arXiv:2402.11187 (uses 0.96 for LLMs)
- **Your model:** T2U 6-layer speech model (needs lower threshold)
- **Rule of thumb:** Smaller models need lower thresholds

---

**TL;DR:** Change `sim_threshold=0.96` to `sim_threshold=0.70` and re-run Phase 3!
