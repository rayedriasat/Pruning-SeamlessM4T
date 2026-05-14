# SeamlessM4T Recovery Strategy: Hybrid DoRA + T2U Distillation

## Executive Summary

**Recommended Approach:** Hybrid (DoRA + T2U Distillation)  
**Target:** Recover from Phase 6 degradation (BLEU 2.04 → target 35-40)  
**Timeline:** 2-3 weeks  
**Memory:** Fits in Kaggle T4 (15GB VRAM)

---

## Problem Analysis

### Current State (After Phase 6)
- **Model:** 1563.7M params (13.4% reduction from baseline)
- **BLEU:** 2.04 (−10.2 from baseline 12.21)
- **ChrF:** 20.74 (−27.4 from baseline 48.12)
- **Issue:** 500 training steps insufficient for 393.2M trainable encoder params

### Root Cause
1. **Text pathway partially recovered** (encoder → text decoder)
2. **Audio pathway completely broken** (encoder → text decoder → T2U → vocoder)
3. **T2U receives zero gradient** during S2TT cross-entropy training
4. **Repetition loops** in audio output ("rererere" problem)

---

## Why NOT Pure Knowledge Distillation?

### Memory Constraints
```
Teacher (full model):     1805.5M params × fp16 = 3.6 GB
Student (pruned model):   1563.7M params × fp16 = 3.1 GB
Optimizer states:         2× student params    = 6.2 GB
Activations + gradients:                       = 4.0 GB
────────────────────────────────────────────────────────
Total:                                        ≈ 16.9 GB
```
**Result:** Exceeds T4 VRAM (15GB) → OOM errors

### Computational Cost
- Teacher inference on every batch (2.3B params forward pass)
- Slower training (2-3× longer per step)
- Unnecessary for text recovery (DoRA already proven to work)

---

## Hybrid Approach: DoRA + T2U Distillation

### Phase 7a: DoRA Fine-Tuning (Text Recovery)

**Target:** Speech encoder + Text decoder  
**Loss:** S2TT Cross-Entropy  
**Proven:** Your `only-p7-cse465v5-s2st-corrected.ipynb` already works

#### Configuration
```python
# DoRA injection (rank-8 decomposition)
from peft import LoraConfig, get_peft_model

dora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        # Speech encoder
        "speech_encoder.encoder.layers.*.self_attn.q_proj",
        "speech_encoder.encoder.layers.*.self_attn.v_proj",
        "speech_encoder.encoder.layers.*.ffn.fc1",
        "speech_encoder.encoder.layers.*.ffn.fc2",
        # Text decoder
        "text_decoder.layers.*.self_attn.q_proj",
        "text_decoder.layers.*.self_attn.v_proj",
        "text_decoder.layers.*.encoder_attn.q_proj",
        "text_decoder.layers.*.encoder_attn.v_proj",
        "text_decoder.layers.*.ffn.fc1",
        "text_decoder.layers.*.ffn.fc2",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    use_dora=True,  # Enable DoRA (magnitude + direction decomposition)
)

model_p7a = get_peft_model(pruned_model, dora_config)
```

#### Training Loop
```python
# Phase 7a: S2TT Cross-Entropy Loss
optimizer = torch.optim.AdamW(model_p7a.parameters(), lr=5e-5)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=200, num_training_steps=2000
)

for step in range(2000):
    # English audio → Bengali text labels
    outputs = model_p7a(
        input_features=audio_features,
        labels=bengali_text_ids,
    )
    loss = outputs.loss  # Cross-entropy on text tokens
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_p7a.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

#### Expected Results
- **Text BLEU:** 35-40 (90-95% recovery from baseline 12.21)
- **Training time:** 2-3 hours (2000 steps)
- **Memory:** 8-10 GB (fits comfortably in T4)

---

### Phase 7b: T2U Distillation (Audio Recovery)

**Key Innovation:** On-the-fly teacher logit generation (no pre-extraction)

#### Why This Works
1. **Teacher on CPU, student on GPU** → memory efficient
2. **Soft targets from teacher** → richer supervision than hard units
3. **Only T2U trains** → speech encoder already recovered in Phase 7a
4. **Logit-level distillation** → captures teacher's uncertainty

#### Architecture
```
┌─────────────────────────────────────────────────────────┐
│  TEACHER (CPU, frozen)                                  │
│  ├─ speech_encoder (frozen)                             │
│  ├─ text_decoder (frozen)                               │
│  └─ t2u_model → unit_logits [B, T, 10082]               │
└─────────────────────────────────────────────────────────┘
                        │
                        │ (soft targets)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STUDENT (GPU)                                          │
│  ├─ speech_encoder (frozen, from Phase 7a)              │
│  ├─ text_decoder (frozen, from Phase 7a)                │
│  └─ t2u_model (trainable) → unit_logits [B, T, 10082]   │
└─────────────────────────────────────────────────────────┘
```

#### Loss Function
```python
def compute_t2u_distillation_loss(
    student_model,
    teacher_model,
    audio_features,
    temperature=2.0,
    alpha=0.7,  # Weight for distillation loss
):
    """
    Compute T2U distillation loss with on-the-fly teacher inference.
    
    Args:
        student_model: Pruned model (GPU)
        teacher_model: Full model (CPU)
        audio_features: Input audio [B, T, 80]
        temperature: Softmax temperature for distillation
        alpha: Weight for soft targets (1-alpha for hard targets)
    """
    # ── Teacher inference (CPU) ────────────────────────────────
    with torch.no_grad():
        teacher_model.eval()
        
        # Move input to CPU
        audio_cpu = audio_features.cpu()
        
        # Teacher forward pass
        teacher_enc_out = teacher_model.speech_encoder(
            input_features=audio_cpu
        )
        teacher_enc_hidden = teacher_enc_out.last_hidden_state
        
        # Teacher text decoder (get hidden states for T2U)
        teacher_dec_out = teacher_model.text_decoder(
            encoder_hidden_states=teacher_enc_hidden,
            decoder_input_ids=torch.zeros(
                (audio_cpu.shape[0], 1), 
                dtype=torch.long, 
                device='cpu'
            ),
        )
        teacher_text_hidden = teacher_dec_out.last_hidden_state
        
        # Teacher T2U logits
        teacher_t2u_out = teacher_model.t2u_model(
            inputs_embeds=teacher_text_hidden,
            return_dict=True,
        )
        teacher_logits = teacher_t2u_out.logits  # [B, T, 10082]
        
        # Move teacher logits to GPU
        teacher_logits = teacher_logits.to('cuda')
    
    # ── Student inference (GPU) ────────────────────────────────
    student_model.train()
    
    # Student forward pass (encoder + decoder already recovered)
    student_enc_out = student_model.speech_encoder(
        input_features=audio_features
    )
    student_enc_hidden = student_enc_out.last_hidden_state
    
    student_dec_out = student_model.text_decoder(
        encoder_hidden_states=student_enc_hidden,
        decoder_input_ids=torch.zeros(
            (audio_features.shape[0], 1), 
            dtype=torch.long, 
            device='cuda'
        ),
    )
    student_text_hidden = student_dec_out.last_hidden_state
    
    # Student T2U logits
    student_t2u_out = student_model.t2u_model(
        inputs_embeds=student_text_hidden,
        return_dict=True,
    )
    student_logits = student_t2u_out.logits  # [B, T, 10082]
    
    # ── Distillation loss ──────────────────────────────────────
    # Soft targets (KL divergence with temperature scaling)
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_pred = F.log_softmax(student_logits / temperature, dim=-1)
    
    distill_loss = F.kl_div(
        soft_pred, 
        soft_targets, 
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard targets (optional, for stability)
    hard_targets = teacher_logits.argmax(dim=-1)
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        hard_targets.view(-1),
        ignore_index=-100,
    )
    
    # Combined loss
    loss = alpha * distill_loss + (1 - alpha) * hard_loss
    
    return loss
```

#### Training Loop
```python
# Phase 7b: T2U Distillation
# Load teacher model on CPU
teacher_model = SeamlessM4Tv2Model.from_pretrained(
    'facebook/seamless-m4t-v2-large',
    torch_dtype=torch.float32,  # CPU uses fp32
).to('cpu')
teacher_model.eval()

# Load student model (from Phase 7a) on GPU
student_model = load_phase7a_model().to('cuda')

# Freeze encoder + decoder (already recovered)
for param in student_model.speech_encoder.parameters():
    param.requires_grad = False
for param in student_model.text_decoder.parameters():
    param.requires_grad = False

# Only T2U is trainable
for param in student_model.t2u_model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    student_model.t2u_model.parameters(), 
    lr=1e-4
)

for step in range(2000):
    audio_features = next(dataloader)
    
    loss = compute_t2u_distillation_loss(
        student_model=student_model,
        teacher_model=teacher_model,
        audio_features=audio_features,
        temperature=2.0,
        alpha=0.7,
    )
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        student_model.t2u_model.parameters(), 
        1.0
    )
    optimizer.step()
    optimizer.zero_grad()
    
    if step % 100 == 0:
        print(f"Step {step}/2000  Loss={loss.item():.4f}")
```

#### Memory Breakdown
```
Teacher (CPU):            3.6 GB (CPU RAM, not VRAM)
Student encoder (GPU):    0.8 GB (frozen, inference mode)
Student decoder (GPU):    1.7 GB (frozen, inference mode)
Student T2U (GPU):        0.5 GB (trainable)
Optimizer states:         1.0 GB (only T2U params)
Activations + gradients:  2.0 GB
Teacher logits (GPU):     0.4 GB (transferred from CPU)
────────────────────────────────────────────────────────
Total VRAM:              ≈ 6.4 GB (fits in T4!)
```

#### Expected Results
- **ASR-BLEU:** 25-30 (80-85% recovery)
- **Audio quality:** No more repetition loops
- **Training time:** 3-4 hours (2000 steps)

---

## Implementation Timeline

### Week 1: Phase 7a (DoRA)
- **Day 1-2:** Set up DoRA config, verify target modules
- **Day 3-4:** Train 2000 steps, monitor text BLEU
- **Day 5:** Benchmark on FLEURS test set
- **Checkpoint:** Text BLEU 35-40 achieved

### Week 2: Phase 7b (T2U KD)
- **Day 1-2:** Implement on-the-fly distillation loss
- **Day 3-4:** Train 2000 steps, monitor ASR-BLEU
- **Day 5:** Benchmark audio quality
- **Checkpoint:** ASR-BLEU 25-30 achieved

### Week 3: Tuning + Final Benchmark
- **Day 1-2:** Hyperparameter sweep (temperature, alpha)
- **Day 3-4:** Extended training if needed (up to 5000 steps)
- **Day 5:** Final benchmark on full FLEURS test set

---

## Success Metrics

### Phase 7a (DoRA)
- ✅ Text BLEU ≥ 35 (90% recovery)
- ✅ ChrF ≥ 45 (90% recovery)
- ✅ No OOM errors
- ✅ Training completes in <3 hours

### Phase 7b (T2U KD)
- ✅ ASR-BLEU ≥ 25 (80% recovery)
- ✅ No repetition loops in audio
- ✅ RTF ≤ 0.30 (faster than baseline)
- ✅ Training completes in <4 hours

### Overall
- ✅ Combined BLEU ≥ 30 (85% recovery)
- ✅ Model size 1563.7M (13.4% reduction maintained)
- ✅ Inference speed improvement (RTF < baseline)

---

## Fallback Plans

### If Phase 7a Fails (Text Recovery)
**Symptoms:** Text BLEU < 30 after 2000 steps

**Solutions:**
1. **Increase training steps** → 5000 steps
2. **Higher learning rate** → 1e-4 (from 5e-5)
3. **Reduce DoRA rank** → r=4 (faster convergence)
4. **Curriculum learning** → Start with short samples

### If Phase 7b Fails (Audio Recovery)
**Symptoms:** ASR-BLEU < 20 or repetition loops persist

**Solutions:**
1. **Increase temperature** → 3.0 (softer targets)
2. **Adjust alpha** → 0.9 (more weight on soft targets)
3. **Pre-extract units** → Fall back to NAR-specific training
4. **Two-stage T2U** → Train encoder first, then decoder

### If Memory Issues Occur
**Symptoms:** OOM during Phase 7b

**Solutions:**
1. **Reduce batch size** → 1 (from 2)
2. **Gradient accumulation** → Accumulate 4 steps
3. **Teacher in fp16** → Move teacher to GPU in fp16
4. **Checkpoint activations** → Use gradient checkpointing

---

## Code Structure

### File Organization
```
cse465v7-recovery/
├── phase7a_dora.py          # DoRA fine-tuning script
├── phase7b_t2u_kd.py        # T2U distillation script
├── loss_functions.py        # Distillation loss implementations
├── data_utils.py            # FLEURS data loading
├── eval_utils.py            # ASR-BLEU evaluation
└── checkpoints/
    ├── phase7a_step2000.pt  # DoRA checkpoint
    └── phase7b_step2000.pt  # T2U KD checkpoint
```

### Notebook Cells
```
Cell 1-11:  Setup (imports, paths, rclone)
Cell 12:    Load Phase 6 pruned model
Cell 13:    Phase 7a DoRA injection
Cell 14:    Phase 7a training loop (2000 steps)
Cell 15:    Phase 7a benchmark (text BLEU)
Cell 16:    Load teacher model (CPU)
Cell 17:    Phase 7b distillation loss
Cell 18:    Phase 7b training loop (2000 steps)
Cell 19:    Phase 7b benchmark (ASR-BLEU)
Cell 20:    Final comparison table
```

---

## Risk Mitigation

### Risk 1: Teacher-Student Mismatch
**Issue:** Pruned encoder produces different distributions than full encoder

**Mitigation:**
- Phase 7a recovers encoder first → reduces mismatch
- Temperature scaling smooths distributions
- Soft targets are more robust than hard units

### Risk 2: T2U Overfitting
**Issue:** T2U memorizes teacher outputs without generalizing

**Mitigation:**
- Use validation set for early stopping
- Monitor ASR-BLEU on held-out samples
- Reduce alpha if overfitting detected

### Risk 3: Kaggle Time Limits
**Issue:** Training exceeds 9-hour Kaggle session

**Mitigation:**
- Save checkpoints every 200 steps
- Resume from checkpoint in new session
- Use rclone to sync checkpoints to Drive

---

## Comparison with Alternatives

### vs. Pure DoRA (No T2U Training)
- ❌ T2U remains broken → audio output fails
- ❌ ASR-BLEU stays low (<10)
- ✅ Faster (only 2000 steps)

### vs. Pure KD (End-to-End)
- ❌ OOM on T4 (16.9 GB > 15 GB)
- ❌ Slower (2-3× per step)
- ✅ Potentially higher quality (if memory allowed)

### vs. NAR-Specific Training
- ❌ Requires unit extraction (slow, lossy)
- ❌ Hard targets less informative than soft
- ✅ Simpler implementation

### Hybrid (Recommended)
- ✅ Fits in T4 memory
- ✅ Proven text recovery (DoRA)
- ✅ Innovative audio recovery (on-the-fly KD)
- ✅ Modular (can tune each phase independently)

---

## References

### DoRA (Phase 7a)
- Liu et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation"
- Your proven implementation: `only-p7-cse465v5-s2st-corrected.ipynb`

### Knowledge Distillation (Phase 7b)
- Hinton et al. (2015). "Distilling the Knowledge in a Neural Network"
- Sanh et al. (2019). "DistilBERT" (temperature scaling)

### SeamlessM4T Architecture
- Barrault et al. (2023). "SeamlessM4T: Massively Multilingual & Multimodal Machine Translation"

---

## Next Steps

1. **Read this document carefully** → Understand the approach
2. **Review Phase 7a code** → Verify DoRA config
3. **Run Phase 7a** → Achieve text recovery first
4. **Implement Phase 7b** → Add T2U distillation
5. **Benchmark** → Compare against baseline

**Start with Phase 7a** since you've already proven it works. The innovation is Phase 7b's on-the-fly distillation approach.

---

## Questions?

If you encounter issues:
1. Check memory usage: `torch.cuda.memory_summary()`
2. Verify gradients: `print(model.t2u_model.parameters()[0].grad)`
3. Monitor losses: Plot training curves
4. Compare outputs: Listen to audio samples

Good luck! 🚀
