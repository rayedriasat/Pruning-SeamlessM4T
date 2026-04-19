# Phase 7 Recovery Strategy: Complete Implementation Guide

## 🎯 Overview

This repository contains a complete implementation of the **Hybrid Recovery Strategy** (DoRA + T2U Distillation) to recover your pruned SeamlessM4T model's quality from Phase 6.

**Current State:** BLEU 2.04 (Phase 6, 13.4% size reduction)  
**Target:** BLEU 35-40 (Phase 7, maintaining 13.4% reduction)  
**Approach:** Two-phase recovery (text → audio)  
**Timeline:** 2-3 weeks  
**Platform:** Kaggle T4 (15GB VRAM)

---

## 📁 Files in This Repository

### 📚 Documentation (Read First)

1. **`RECOVERY_STRATEGY_HYBRID_APPROACH.md`** ⭐ **START HERE**
   - Complete technical explanation
   - Why NOT pure knowledge distillation
   - Detailed loss functions and architecture
   - Memory breakdowns and expected results
   - Fallback plans and risk mitigation

2. **`QUICK_START_PHASE7.md`** ⭐ **STEP-BY-STEP GUIDE**
   - Day-by-day timeline (3 weeks)
   - Setup instructions
   - Training commands
   - Benchmark procedures
   - Troubleshooting guide

3. **`PHASE7_IMPLEMENTATION_SUMMARY.md`**
   - High-level overview
   - File structure explanation
   - Key innovations
   - Success criteria

4. **`PHASE7_ARCHITECTURE_VISUAL.txt`**
   - Visual diagrams of architecture
   - Memory layout diagrams
   - Loss computation flowcharts
   - Training flow comparisons

5. **`APPROACH_COMPARISON_TABLE.md`**
   - Comparison of 4 recovery approaches
   - Why hybrid is best
   - Risk assessment
   - Decision matrix

6. **`PHASE7_PROGRESS_CHECKLIST.md`**
   - Detailed checklist for tracking progress
   - Week-by-week tasks
   - Success criteria verification
   - Notes section for issues

### 💻 Implementation Code

7. **`phase7a_dora_implementation.py`** (Week 1)
   - DoRA fine-tuning for text recovery
   - Target: BLEU 35-40
   - Time: 2-3 hours
   - Memory: 8-10 GB VRAM

8. **`phase7b_t2u_distillation.py`** (Week 2)
   - T2U distillation for audio recovery
   - Target: ASR-BLEU 25-30
   - Time: 3-4 hours
   - Memory: 6-7 GB VRAM

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install transformers>=4.40.0 peft>=0.10.0 datasets accelerate torch torchaudio

# Verify Phase 6 model exists
ls ./models/phase6_pruned/config.json
```

### Week 1: Phase 7a (DoRA)

```bash
# Run DoRA fine-tuning
python phase7a_dora_implementation.py

# Expected output:
# - Training completes in 2-3 hours
# - Final loss ~2.0
# - Model saved to ./models/phase7a_dora/
# - Text BLEU 35-40
```

### Week 2: Phase 7b (T2U Distillation)

```bash
# Run T2U distillation
python phase7b_t2u_distillation.py

# Expected output:
# - Training completes in 3-4 hours
# - Final loss ~1.2
# - Model saved to ./models/phase7b_t2u_distilled/
# - ASR-BLEU 25-30
```

### Week 3: Final Benchmark

```python
# Evaluate on full test set
from transformers import SeamlessM4Tv2Model, AutoProcessor

model = SeamlessM4Tv2Model.from_pretrained('./models/phase7b_t2u_distilled')
processor = AutoProcessor.from_pretrained('./models/phase7b_t2u_distilled')

# Run your benchmark code from Phase 6
results = benchmark_asr_bleu(model, processor, test_samples)
print(f"Final BLEU: {results['bleu']:.2f}")
```

---

## 📖 Reading Order

### For First-Time Readers

1. **Start here:** `RECOVERY_STRATEGY_HYBRID_APPROACH.md`
   - Read sections: "Why NOT Pure KD?", "Phase 7a", "Phase 7b"
   - Understand the memory breakdown
   - Review the key innovation (teacher on CPU)

2. **Then read:** `QUICK_START_PHASE7.md`
   - Follow day-by-day instructions
   - Set up your environment
   - Prepare for Week 1

3. **Reference:** `PHASE7_ARCHITECTURE_VISUAL.txt`
   - Visual understanding of architecture
   - Memory layout diagrams
   - Loss computation flowcharts

4. **Track progress:** `PHASE7_PROGRESS_CHECKLIST.md`
   - Check off tasks as you complete them
   - Verify success criteria
   - Document issues

### For Quick Reference

- **Memory issues?** → `RECOVERY_STRATEGY_HYBRID_APPROACH.md` (Fallback Plans)
- **Training issues?** → `QUICK_START_PHASE7.md` (Troubleshooting)
- **Architecture questions?** → `PHASE7_ARCHITECTURE_VISUAL.txt`
- **Why hybrid?** → `APPROACH_COMPARISON_TABLE.md`

---

## 🎯 The Hybrid Approach Explained

### Phase 7a: DoRA Fine-Tuning (Week 1)

**Goal:** Recover text translation quality

```
English Audio → [Speech Encoder + DoRA] → [Text Decoder + DoRA] → Bengali Text
                      ↑ trainable                ↑ trainable
                      
Loss: Cross-Entropy(predicted_text, reference_text)
Result: Text BLEU 35-40 (90-95% recovery)
```

**Key Features:**
- Only ~50M trainable params (3% of model)
- Proven to work in your previous experiments
- Memory efficient (8-10 GB VRAM)
- Fast training (2-3 hours)

### Phase 7b: T2U Distillation (Week 2)

**Goal:** Recover audio output quality

```
Teacher (CPU):  Audio → Encoder → Decoder → T2U → Unit Logits (soft targets)
                                                         ↓
                                                    Transfer to GPU
                                                         ↓
Student (GPU):  Audio → Encoder → Decoder → T2U → Unit Logits
                        (frozen)   (frozen)   ↑
                                          trainable
                                              
Loss: KL_div(student_logits, teacher_logits, temperature=2.0)
Result: ASR-BLEU 25-30 (80-85% recovery)
```

**Key Innovation:**
- Teacher on CPU (no VRAM usage!)
- Only T2U trains (~86M params, 5.5%)
- Soft targets richer than hard units
- Memory efficient (6-7 GB VRAM)

---

## 💡 Key Innovations

### 1. DoRA for Encoder Recovery
**Standard:** Full fine-tuning (393M params)  
**Ours:** DoRA adapters (50M params, 3%)  
**Benefit:** 8× fewer trainable params → faster, more stable

### 2. On-the-Fly T2U Distillation
**Standard:** Pre-extract units, train NAR decoder  
**Ours:** Teacher generates logits on-the-fly  
**Benefit:** No extraction overhead, richer soft targets

### 3. CPU-GPU Split
**Standard:** Both models on GPU → OOM  
**Ours:** Teacher on CPU, student on GPU  
**Benefit:** Fits in T4 (15GB VRAM)

---

## 📊 Expected Results

### Phase 7a (DoRA)
| Metric | Before | After | Recovery |
|--------|--------|-------|----------|
| Text BLEU | 2.04 | 35-40 | 90-95% |
| ChrF | 20.74 | 45-48 | 90-95% |
| Training time | - | 2-3 hours | - |

### Phase 7b (T2U KD)
| Metric | Before | After | Recovery |
|--------|--------|-------|----------|
| ASR-BLEU | <10 | 25-30 | 80-85% |
| Audio quality | Repetition loops | Clean speech | ✅ |
| Training time | - | 3-4 hours | - |

### Overall
| Metric | Baseline | Phase 6 | Phase 7 | Recovery |
|--------|----------|---------|---------|----------|
| Params | 1805.5M | 1563.7M | 1563.7M | 13.4% reduction |
| Text BLEU | 12.21 | 2.04 | 35-40 | 85-90% |
| ASR-BLEU | ~40 | <10 | 25-30 | 80-85% |
| RTF | 0.24 | 0.28 | <0.24 | Faster |

---

## 🔧 Troubleshooting

### Issue 1: OOM during Phase 7a
**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# Reduce batch size
TRAINING_CONFIG['batch_size'] = 1

# Or use gradient accumulation
TRAINING_CONFIG['gradient_accumulation_steps'] = 2
```

### Issue 2: OOM during Phase 7b
**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# Verify teacher is on CPU
print(next(teacher_model.parameters()).device)  # Should be 'cpu'

# Reduce batch size
DISTILL_CONFIG['batch_size'] = 1
```

### Issue 3: Text BLEU not improving
**Symptoms:** Loss decreases but BLEU stays low

**Solutions:**
1. Increase training steps to 5000
2. Try higher learning rate (1e-4)
3. Check if DoRA adapters are training:
```python
for name, param in model.named_parameters():
    if 'lora' in name and param.requires_grad:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
```

### Issue 4: Audio has repetition loops
**Symptoms:** Output audio sounds like "rererere..."

**Solutions:**
1. Increase temperature to 3.0
2. Increase alpha to 0.9
3. Check if T2U is training:
```python
for name, param in model.t2u_model.named_parameters():
    if param.requires_grad:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
```

---

## 📈 Timeline

| Week | Phase | Task | Time | Checkpoint |
|------|-------|------|------|------------|
| 1 | 7a | Setup + DoRA injection | 1 day | DoRA config verified |
| 1 | 7a | Training (2000 steps) | 2 days | Text BLEU 35-40 |
| 1 | 7a | Benchmark | 1 day | Phase 7a complete |
| 2 | 7b | Setup + teacher-student | 1 day | Memory verified |
| 2 | 7b | Training (2000 steps) | 2 days | ASR-BLEU 25-30 |
| 2 | 7b | Benchmark | 1 day | Phase 7b complete |
| 3 | Tuning | Hyperparameter sweep | 2 days | Optimal config found |
| 3 | Final | Extended training | 2 days | Quality maximized |
| 3 | Final | Full benchmark | 1 day | Paper-ready results |

---

## ✅ Success Criteria

### Phase 7a Complete
- [ ] DoRA adapters injected successfully
- [ ] Training completed (2000 steps, 2-3 hours)
- [ ] Text BLEU ≥ 35
- [ ] ChrF ≥ 45
- [ ] Model saved to `./models/phase7a_dora`

### Phase 7b Complete
- [ ] Teacher on CPU, student on GPU
- [ ] Training completed (2000 steps, 3-4 hours)
- [ ] ASR-BLEU ≥ 25
- [ ] No repetition loops
- [ ] Model saved to `./models/phase7b_t2u_distilled`

### Final Benchmark Complete
- [ ] Full FLEURS test set evaluated (647 samples)
- [ ] Combined BLEU ≥ 30
- [ ] Model size 1563.7M (13.4% reduction maintained)
- [ ] RTF < baseline (0.24)
- [ ] Paper table generated

---

## 🤝 Contributing

If you encounter issues or have improvements:

1. Document the issue in `PHASE7_PROGRESS_CHECKLIST.md` (Notes section)
2. Try the troubleshooting steps in `QUICK_START_PHASE7.md`
3. Check the fallback plans in `RECOVERY_STRATEGY_HYBRID_APPROACH.md`
4. If you find a solution, document it for future reference

---

## 📚 References

### DoRA (Phase 7a)
- Liu et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation"
- Your proven implementation: `only-p7-cse465v5-s2st-corrected.ipynb`

### Knowledge Distillation (Phase 7b)
- Hinton et al. (2015). "Distilling the Knowledge in a Neural Network"
- Sanh et al. (2019). "DistilBERT" (temperature scaling)

### SeamlessM4T Architecture
- Barrault et al. (2023). "SeamlessM4T: Massively Multilingual & Multimodal Machine Translation"

---

## 🎓 Citation

If you use this recovery strategy in your research, please cite:

```bibtex
@misc{seamlessm4t_recovery_2026,
  title={Hybrid Recovery Strategy for Pruned Speech Translation Models},
  author={Your Name},
  year={2026},
  note={DoRA fine-tuning + on-the-fly T2U distillation}
}
```

---

## 📞 Support

If you need help:

1. **Read the docs:** Start with `RECOVERY_STRATEGY_HYBRID_APPROACH.md`
2. **Check troubleshooting:** See `QUICK_START_PHASE7.md` (Troubleshooting section)
3. **Review checklist:** Use `PHASE7_PROGRESS_CHECKLIST.md` to track progress
4. **Verify environment:** Ensure prerequisites are met

---

## 🎉 Next Steps

1. **Read** `RECOVERY_STRATEGY_HYBRID_APPROACH.md` (30 min)
2. **Follow** `QUICK_START_PHASE7.md` (3 weeks)
3. **Track** progress with `PHASE7_PROGRESS_CHECKLIST.md`
4. **Run** `phase7a_dora_implementation.py` (Week 1)
5. **Run** `phase7b_t2u_distillation.py` (Week 2)
6. **Benchmark** on full test set (Week 3)
7. **Write** paper with results

**Good luck! 🚀**

---

## 📝 License

This implementation is provided as-is for research purposes. The underlying SeamlessM4T model is subject to Meta's license terms.

---

## 🙏 Acknowledgments

- Meta AI for SeamlessM4T
- HuggingFace for Transformers and PEFT
- Google for FLEURS dataset
- Kaggle for free GPU access

---

**Last Updated:** April 19, 2026  
**Version:** 1.0  
**Status:** Ready for implementation
