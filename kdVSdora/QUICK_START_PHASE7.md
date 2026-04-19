# Quick Start Guide: Phase 7 Recovery

## Overview

This guide walks you through implementing the hybrid recovery strategy (DoRA + T2U Distillation) to recover your pruned SeamlessM4T model's quality.

**Current state:** BLEU 2.04 (Phase 6)  
**Target:** BLEU 35-40 (Phase 7)  
**Time:** 2-3 weeks  

---

## Prerequisites

### Files You Need
- ✅ `RECOVERY_STRATEGY_HYBRID_APPROACH.md` - Full strategy document
- ✅ `phase7a_dora_implementation.py` - DoRA fine-tuning code
- ✅ `phase7b_t2u_distillation.py` - T2U distillation code
- ✅ Phase 6 pruned model checkpoint

### Environment
- Kaggle T4 GPU (15GB VRAM) or equivalent
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.40+
- PEFT 0.10+

---

## Step-by-Step Instructions

### Week 1: Phase 7a (DoRA Fine-Tuning)

#### Day 1-2: Setup

1. **Install dependencies**
```bash
pip install transformers>=4.40.0 peft>=0.10.0 datasets accelerate
```

2. **Verify Phase 6 model exists**
```python
import os
assert os.path.exists('./models/phase6_pruned/config.json'), \
    "Phase 6 model not found! Run Phase 6 first."
```

3. **Test DoRA injection**
```python
from phase7a_dora_implementation import inject_dora
from transformers import SeamlessM4Tv2Model

model = SeamlessM4Tv2Model.from_pretrained('./models/phase6_pruned')
model_with_dora = inject_dora(model)

# Should print: "Trainable params: ~50M, Trainable %: ~3%"
```

#### Day 3-4: Training

4. **Run Phase 7a training**
```bash
python phase7a_dora_implementation.py
```

Expected output:
```
Phase 7a: DoRA Fine-Tuning for Text Recovery
============================================
[1/5] Loading Phase 6 pruned model...
[2/5] Injecting DoRA adapters...
  Trainable params: 47.3M
  Total params: 1563.7M
  Trainable %: 3.02%
[3/5] Loading training data...
  Loaded 2554 training pairs.
[4/5] Training...
  Steps: 2000
  Batch size: 2
  Learning rate: 5e-05
  Warmup steps: 200

Step 50/2000  Loss=8.2341  LR=1.25e-05
Step 100/2000  Loss=6.8912  LR=2.50e-05
...
Step 2000/2000  Loss=2.1456  LR=2.50e-07

[5/5] Saving final model...
Model saved to ./models/phase7a_dora

Phase 7a complete! Next: Run Phase 7b (T2U distillation)
```

**Training time:** 2-3 hours

#### Day 5: Benchmark

5. **Evaluate Phase 7a results**
```python
from transformers import SeamlessM4Tv2Model, AutoProcessor
from datasets import load_dataset
import sacrebleu

# Load model
model = SeamlessM4Tv2Model.from_pretrained('./models/phase7a_dora')
processor = AutoProcessor.from_pretrained('./models/phase7a_dora')

# Load test data
test_ds = load_dataset('google/fleurs', 'en_us', split='test[:20]')

# Run inference + evaluate
# (Use your existing benchmark code from Phase 6)
```

**Success criteria:**
- ✅ Text BLEU ≥ 35 (target: 35-40)
- ✅ No OOM errors
- ✅ Training completed in <3 hours

---

### Week 2: Phase 7b (T2U Distillation)

#### Day 1-2: Setup

6. **Verify Phase 7a model exists**
```python
assert os.path.exists('./models/phase7a_dora/config.json'), \
    "Phase 7a model not found! Run Phase 7a first."
```

7. **Test teacher-student setup**
```python
import torch
from transformers import SeamlessM4Tv2Model

# Teacher on CPU
teacher = SeamlessM4Tv2Model.from_pretrained(
    'facebook/seamless-m4t-v2-large',
    torch_dtype=torch.float32,
).to('cpu')

# Student on GPU
student = SeamlessM4Tv2Model.from_pretrained(
    './models/phase7a_dora',
    torch_dtype=torch.float16,
).to('cuda')

print(f"Teacher device: {next(teacher.parameters()).device}")  # cpu
print(f"Student device: {next(student.parameters()).device}")  # cuda:0
```

#### Day 3-4: Training

8. **Run Phase 7b training**
```bash
python phase7b_t2u_distillation.py
```

Expected output:
```
Phase 7b: T2U Knowledge Distillation for Audio Recovery
=======================================================
[1/5] Loading teacher model (CPU)...
Teacher model loaded on CPU
[2/5] Loading Phase 7a student model (GPU)...
Student model loaded on GPU
Trainable parameters:
  T2U params: 86.4M
  Total params: 1563.7M
  Trainable %: 5.52%
[3/5] Loading training data...
  Loaded 2554 training samples.
[4/5] Training...
  Steps: 2000
  Temperature: 2.0
  Alpha: 0.7

Step 50/2000  Total=3.4521  Soft=4.1234  Hard=2.1234  LR=5.00e-05
Step 100/2000  Total=2.9876  Soft=3.5432  Hard=1.8765  LR=1.00e-04
...
Step 2000/2000  Total=1.2345  Soft=1.4567  Hard=0.8901  LR=2.50e-07

[5/5] Saving final model...
Model saved to ./models/phase7b_t2u_distilled

Phase 7b complete! Next: Run final benchmark
```

**Training time:** 3-4 hours

#### Day 5: Benchmark

9. **Evaluate Phase 7b results (ASR-BLEU)**
```python
import whisper
import soundfile as sf
import sacrebleu

# Load Whisper for ASR
whisper_model = whisper.load_model('medium')

# Load your model
model = SeamlessM4Tv2Model.from_pretrained('./models/phase7b_t2u_distilled')

# Run S2ST inference
for sample in test_samples:
    # Generate audio
    audio_output = model.generate(...)
    
    # Transcribe with Whisper
    asr_text = whisper_model.transcribe(audio_output)['text']
    
    # Compare with reference
    bleu = sacrebleu.sentence_bleu(asr_text, [sample['reference']])
```

**Success criteria:**
- ✅ ASR-BLEU ≥ 25 (target: 25-30)
- ✅ No repetition loops in audio
- ✅ Training completed in <4 hours

---

### Week 3: Tuning + Final Benchmark

#### Day 1-2: Hyperparameter Tuning

10. **If results are below target, tune hyperparameters**

**For low text BLEU (<35):**
```python
# Increase Phase 7a training steps
TRAINING_CONFIG['num_train_steps'] = 5000

# Or increase learning rate
TRAINING_CONFIG['learning_rate'] = 1e-4
```

**For low ASR-BLEU (<25):**
```python
# Increase temperature (softer targets)
DISTILL_CONFIG['temperature'] = 3.0

# Or increase alpha (more weight on soft targets)
DISTILL_CONFIG['alpha'] = 0.9
```

#### Day 3-4: Extended Training (if needed)

11. **Resume from checkpoint and train longer**
```python
# Load checkpoint
checkpoint = torch.load('./checkpoints/phase7b/phase7b_step002000.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue training
train_phase7b(model, teacher, processor, train_samples, 
              config={'num_train_steps': 3000})  # Additional 1000 steps
```

#### Day 5: Final Benchmark

12. **Run full benchmark on FLEURS test set**
```python
# Evaluate on all 647 test samples (not just 20)
test_samples = load_fleurs_parquet(split='test', max_samples=647)

results = benchmark_asr_bleu(
    model, 
    processor, 
    test_samples, 
    label='phase7b_final'
)

print(f"Final ASR-BLEU: {results['asr_bleu']:.2f}")
print(f"Final params: {results['params_M']:.1f}M")
print(f"Final RTF: {results['avg_rtf']:.4f}")
```

---

## Troubleshooting

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
# Reduce batch size
DISTILL_CONFIG['batch_size'] = 1

# Or move teacher to CPU (already done!)
# Or use gradient checkpointing
student_model.gradient_checkpointing_enable()
```

### Issue 3: Text BLEU not improving
**Symptoms:** Loss decreases but BLEU stays low

**Solutions:**
1. Check if DoRA adapters are actually training:
```python
for name, param in model.named_parameters():
    if 'lora' in name and param.requires_grad:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
```

2. Increase training steps to 5000
3. Try higher learning rate (1e-4)

### Issue 4: Audio has repetition loops
**Symptoms:** Output audio sounds like "rererere..."

**Solutions:**
1. Increase temperature to 3.0 (softer targets)
2. Increase alpha to 0.9 (more weight on soft targets)
3. Check if T2U is actually training:
```python
for name, param in model.t2u_model.named_parameters():
    if param.requires_grad:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
```

### Issue 5: Training too slow
**Symptoms:** >5 hours for 2000 steps

**Solutions:**
1. Reduce batch size (paradoxically faster due to less memory overhead)
2. Use mixed precision training:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Expected Timeline

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

## Success Checklist

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

## Next Steps After Completion

1. **Generate paper figures**
   - Loss curves (Phase 7a + 7b)
   - BLEU comparison (baseline → Phase 6 → Phase 7)
   - Size-quality tradeoff scatter plot

2. **Write results section**
   - Phase 7a: "DoRA fine-tuning recovered text quality to 90% of baseline..."
   - Phase 7b: "T2U distillation recovered audio quality to 85% of baseline..."

3. **Prepare for submission**
   - Upload models to HuggingFace Hub
   - Create demo notebook
   - Write README with usage instructions

---

## Questions?

If you get stuck:
1. Read `RECOVERY_STRATEGY_HYBRID_APPROACH.md` for detailed explanations
2. Check the troubleshooting section above
3. Verify your environment matches prerequisites
4. Check GPU memory: `torch.cuda.memory_summary()`

Good luck! 🚀
