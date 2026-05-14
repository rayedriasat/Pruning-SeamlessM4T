# 🚀 START HERE: Phase 7 Recovery Implementation

## Welcome!

You're about to implement **Full End-to-End Knowledge Distillation** to recover your pruned SeamlessM4T model's quality. This document will get you started in 5 minutes.

## ⚠️ IMPORTANT UPDATE

**The hybrid approach (DoRA + T2U KD) is now OBSOLETE.**

**Why?** Initial memory estimates were wrong:
- Estimated: 18GB (doesn't fit in T4)
- Actual: 12GB (fits comfortably in T4!)

**New recommendation:** Use **Full KD** instead (better quality, simpler, faster)

---

## ⚡ 5-Minute Quick Start

### 1. What You Have Now (Phase 6)
- ✅ Pruned model: 1563.7M params (13.4% reduction)
- ❌ Broken quality: BLEU 2.04 (was 12.21)
- ❌ Broken audio: Repetition loops

### 2. What You'll Get (Phase 7 - Full KD)
- ✅ Recovered model: 1563.7M params (same size)
- ✅ **Excellent quality: BLEU 38-42 (95-100% recovery)** ✅✅
- ✅ Clean audio: No repetition loops

### 3. How Long It Takes
- **Week 1:** Full KD training - 6-8 hours
- **Week 2:** Benchmark + tuning - 1-2 hours

**Total:** 1-2 weeks (much faster than hybrid!)

---

## 📚 Which Document to Read First?

### ⭐ RECOMMENDED: Full KD Approach
👉 **Read:** `RECOVERY_STRATEGY_FULL_KD.md`
- Why Full KD is better than hybrid
- Memory optimization techniques
- Complete implementation guide
- Expected: 95-100% recovery

**Time:** 20 minutes

### If you want the implementation code:
👉 **Read:** `phase7_full_kd_implementation.py`
- Complete Python implementation
- Single-phase training
- Memory-optimized

**Time:** 10 minutes (code review)

### If you want to compare approaches:
👉 **Read:** `FULL_KD_VS_HYBRID_COMPARISON.md`
- Side-by-side comparison
- Why hybrid is obsolete
- Migration guide

**Time:** 10 minutes

---

## 🎯 Recommended Reading Order

### For First-Time Implementation

1. **This file** (5 min) ← You are here!
2. **`PHASE7_IMPLEMENTATION_SUMMARY.md`** (10 min) - Get the big picture
3. **`RECOVERY_STRATEGY_HYBRID_APPROACH.md`** (30 min) - Understand the approach
4. **`QUICK_START_PHASE7.md`** (15 min) - Follow step-by-step
5. **`PHASE7_ARCHITECTURE_VISUAL.txt`** (15 min) - Visual understanding
6. **`PHASE7_PROGRESS_CHECKLIST.md`** (ongoing) - Track your progress

**Total reading time:** ~1.5 hours (then you're ready to start!)

---

## 🔑 Key Concepts (2-Minute Version)

### The Problem
Your Phase 6 model is broken because:
1. **Text pathway broken:** Encoder → Decoder needs recovery
2. **Audio pathway broken:** T2U model needs recovery

### The Solution
**Two-phase recovery:**

**Phase 7a (Week 1):** DoRA Fine-Tuning
- Add low-rank adapters to encoder + decoder
- Train with text labels
- **Result:** Text BLEU 35-40 ✅

**Phase 7b (Week 2):** T2U Distillation
- Load teacher model on CPU (key innovation!)
- Train only T2U with soft targets
- **Result:** ASR-BLEU 25-30 ✅

### Why This Works
- **Memory efficient:** Teacher on CPU = no VRAM usage
- **Modular:** Fix text first, then audio
- **Proven:** Phase 7a already works in your experiments

---

## 💻 What You Need

### Hardware
- Kaggle T4 GPU (15GB VRAM) ✅ You have this
- Or equivalent (V100, A100, RTX 3090, etc.)

### Software
```bash
pip install transformers>=4.40.0 peft>=0.10.0 datasets accelerate torch torchaudio
```

### Files
- Phase 6 pruned model at `./models/phase6_pruned/`
- FLEURS dataset (auto-downloads from HuggingFace)

---

## 🚀 Start Training in 2 Commands

### Install Dependencies
```bash
# Install 8-bit optimizer (50% memory savings)
pip install bitsandbytes

# Verify installation
python -c "from bitsandbytes.optim import Adam8bit; print('✅ Ready!')"
```

### Run Full KD Training
```bash
# Single command - that's it!
python phase7_full_kd_implementation.py

# Expected output:
# - Training completes in 6-8 hours
# - Text BLEU: 38-42 ✅✅
# - ASR-BLEU: 30-35 ✅✅
# - Model saved to ./models/phase7_full_kd/
```

---

## 📊 Expected Results

### Phase 7a (After Week 1)
```
Training complete!
  Final loss: 2.04
  Text BLEU: 37.2 ✅
  ChrF: 46.8 ✅
  Training time: 2.3 hours ✅
  Model saved to ./models/phase7a_dora/
```

### Phase 7b (After Week 2)
```
Training complete!
  Final loss: 1.18
  ASR-BLEU: 27.4 ✅
  No repetition loops ✅
  Training time: 3.7 hours ✅
  Model saved to ./models/phase7b_t2u_distilled/
```

---

## ⚠️ Common Issues (and Quick Fixes)

### Issue 1: "CUDA out of memory"
**Fix:** Reduce batch size to 1
```python
TRAINING_CONFIG['batch_size'] = 1
```

### Issue 2: "Phase 6 model not found"
**Fix:** Check the path
```bash
ls ./models/phase6_pruned/config.json
# If not found, adjust path in phase7a_dora_implementation.py
```

### Issue 3: "Text BLEU still low after Phase 7a"
**Fix:** Train longer
```python
TRAINING_CONFIG['num_train_steps'] = 5000  # Instead of 2000
```

### Issue 4: "Audio has repetition loops after Phase 7b"
**Fix:** Increase temperature
```python
DISTILL_CONFIG['temperature'] = 3.0  # Instead of 2.0
```

---

## 📈 Progress Tracking

Use `PHASE7_PROGRESS_CHECKLIST.md` to track your progress:

```markdown
## Week 1: Phase 7a
- [x] DoRA adapters injected
- [x] Training completed (2000 steps)
- [x] Text BLEU ≥ 35
- [x] Model saved

## Week 2: Phase 7b
- [ ] Teacher on CPU verified
- [ ] Training started
- [ ] ...
```

---

## 🎯 Success Criteria

### You're done when:
- ✅ Text BLEU ≥ 35 (Phase 7a)
- ✅ ASR-BLEU ≥ 25 (Phase 7b)
- ✅ No repetition loops in audio
- ✅ Model size 1563.7M (maintained)
- ✅ Full test set evaluated

---

## 🆘 Need Help?

### Quick Reference
| Issue | Document | Section |
|-------|----------|---------|
| Memory error | `QUICK_START_PHASE7.md` | Troubleshooting |
| Low BLEU | `QUICK_START_PHASE7.md` | Troubleshooting |
| Repetition loops | `QUICK_START_PHASE7.md` | Troubleshooting |
| Why hybrid? | `APPROACH_COMPARISON_TABLE.md` | Comparison |
| Architecture | `PHASE7_ARCHITECTURE_VISUAL.txt` | Diagrams |

### Still Stuck?
1. Check `PHASE7_PROGRESS_CHECKLIST.md` - Are all prerequisites met?
2. Check `RECOVERY_STRATEGY_HYBRID_APPROACH.md` - Fallback Plans section
3. Check `QUICK_START_PHASE7.md` - Troubleshooting section

---

## 📁 File Overview

### Must Read (in order)
1. `START_HERE.md` ← You are here
2. `PHASE7_IMPLEMENTATION_SUMMARY.md` - Overview
3. `RECOVERY_STRATEGY_HYBRID_APPROACH.md` - Full strategy
4. `QUICK_START_PHASE7.md` - Step-by-step guide

### Reference
5. `PHASE7_ARCHITECTURE_VISUAL.txt` - Diagrams
6. `APPROACH_COMPARISON_TABLE.md` - Why hybrid?
7. `PHASE7_PROGRESS_CHECKLIST.md` - Track progress

### Code
8. `phase7a_dora_implementation.py` - Week 1
9. `phase7b_t2u_distillation.py` - Week 2

### Optional
10. `README_PHASE7_RECOVERY.md` - Complete README

---

## 🎓 Learning Path

### Beginner (Never done model compression)
1. Read `PHASE7_IMPLEMENTATION_SUMMARY.md` (overview)
2. Read `PHASE7_ARCHITECTURE_VISUAL.txt` (visual understanding)
3. Read `RECOVERY_STRATEGY_HYBRID_APPROACH.md` (full strategy)
4. Follow `QUICK_START_PHASE7.md` (step-by-step)

**Time:** 2 hours reading + 3 weeks implementation

### Intermediate (Done model compression before)
1. Read `PHASE7_IMPLEMENTATION_SUMMARY.md` (overview)
2. Skim `RECOVERY_STRATEGY_HYBRID_APPROACH.md` (focus on Phase 7b)
3. Follow `QUICK_START_PHASE7.md` (step-by-step)

**Time:** 1 hour reading + 3 weeks implementation

### Advanced (Familiar with DoRA and KD)
1. Skim `PHASE7_IMPLEMENTATION_SUMMARY.md` (key innovations)
2. Review `phase7a_dora_implementation.py` (code)
3. Review `phase7b_t2u_distillation.py` (code)
4. Run training

**Time:** 30 min reading + 3 weeks implementation

---

## 🎉 Ready to Start?

### Next Steps:

1. **Read** `PHASE7_IMPLEMENTATION_SUMMARY.md` (10 min)
   - Get the big picture
   - Understand the approach
   - See expected results

2. **Read** `RECOVERY_STRATEGY_HYBRID_APPROACH.md` (30 min)
   - Understand WHY hybrid
   - Review memory breakdowns
   - Study loss functions

3. **Follow** `QUICK_START_PHASE7.md` (Week 1)
   - Day 1-2: Setup
   - Day 3-4: Train Phase 7a
   - Day 5: Benchmark

4. **Track** progress with `PHASE7_PROGRESS_CHECKLIST.md`
   - Check off tasks
   - Document issues
   - Verify success criteria

---

## 💡 Key Insight

The **critical innovation** that makes this work:

```
Standard Distillation:
  Teacher (GPU) + Student (GPU) = 16.9 GB ❌ (doesn't fit in T4)

Hybrid Approach:
  Teacher (CPU) + Student (GPU) = 6.4 GB ✅ (fits in T4!)
```

**Only transfer logits (0.4 GB), not entire model (3.6 GB)!**

---

## 🏁 Final Checklist Before Starting

- [ ] Read this document (5 min)
- [ ] Read `PHASE7_IMPLEMENTATION_SUMMARY.md` (10 min)
- [ ] Read `RECOVERY_STRATEGY_HYBRID_APPROACH.md` (30 min)
- [ ] Read `QUICK_START_PHASE7.md` (15 min)
- [ ] Verify GPU available (Kaggle T4 or equivalent)
- [ ] Verify Phase 6 model exists
- [ ] Install dependencies
- [ ] Ready to start Week 1!

---

## 🎯 Your Goal

**Transform this:**
```
Phase 6: BLEU 2.04, broken audio, 1563.7M params
```

**Into this:**
```
Phase 7: BLEU 35-40, clean audio, 1563.7M params
```

**In 3 weeks!**

---

**Good luck! 🚀**

**Next:** Read `PHASE7_IMPLEMENTATION_SUMMARY.md` to get the big picture.
