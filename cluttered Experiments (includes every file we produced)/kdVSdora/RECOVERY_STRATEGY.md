# Phase 7 Recovery Strategy: KD vs DoRA Fine-tuning

## Current Situation Analysis

### Model State (model_p6)
- **Text Decoder**: 10 layers removed (14 remaining from 24)
- **Speech Encoder**: 8 layers removed (16 remaining from 24)
- **T2U Model**: 2 layers removed per stack (4+4 remaining from 6+6)
- **Total Compression**: ~2300M → ~1600M parameters (~30% reduction)
- **Quality Drop**: Significant degradation in both text and audio translation

### Previous Recovery Attempts
1. **DoRA Fine-tuning (only-p7-cse465v5-s2st-corrected.ipynb)**
   - ✅ Preserved audio output (used better model_p6)
   - ❌ Only recovered text translation quality
   - ❌ T2U layers received zero gradient (no audio quality recovery)
   - ⚠️ Requires additional NAR-specific Phase 8

2. **DoRA Fine-tuning (only-p7p8-cse465v5.ipynb)**
   - ❌ Broken audio output
   - ✅ Faster loss reduction
   - ❌ Still doesn't train T2U effectively
   - ⚠️ Same NAR-specific Phase 8 requirement

---

## Strategy Comparison

### Option A: Knowledge Distillation (KD)

#### Architecture
```
Teacher: facebook/seamless-m4t-v2-large (2.3B params, full model)
Student: model_p6 (1.6B params, pruned model)
```

#### Advantages
1. **End-to-End Training**: All components (speech encoder, text decoder, T2U) receive gradients simultaneously
2. **Implicit T2U Recovery**: Teacher's audio output provides supervision signal for T2U layers
3. **Single-Phase Solution**: No need for separate NAR-specific training
4. **Better Generalization**: Learns from teacher's soft targets, not just hard labels
5. **Proven for S2ST**: KD has been successfully applied to speech translation models (e.g., CCSRD, IWSLT 2023)

#### Distillation Losses
```python
# 1. Text Decoder KD (intermediate representations)
L_text = KL(student_text_logits || teacher_text_logits) + CE(student_text, labels)

# 2. Speech Encoder KD (hidden states)
L_speech = MSE(student_speech_hidden, teacher_speech_hidden)

# 3. T2U KD (unit predictions) ← KEY FOR AUDIO RECOVERY
L_t2u = KL(student_unit_logits || teacher_unit_logits)

# 4. Audio Reconstruction (optional, if feasible)
L_audio = L1(student_audio, teacher_audio)

Total Loss = α*L_text + β*L_speech + γ*L_t2u + δ*L_audio
```

#### Disadvantages
1. **Computational Cost**: Requires loading both teacher and student (memory-intensive)
2. **Implementation Complexity**: Need to extract intermediate outputs from teacher
3. **Hyperparameter Tuning**: Multiple loss weights (α, β, γ, δ) to balance
4. **Slower Training**: Forward pass through both models per batch

---

### Option B: DoRA Fine-tuning + NAR-specific T2U Training

#### Phase 7a: DoRA Fine-tuning (S2TT)
- Target: Speech encoder + Text decoder
- Loss: Cross-entropy on text tokens
- Duration: ~2000 steps
- Result: Recovers text translation quality

#### Phase 7b: NAR-specific T2U Training
- Target: T2U encoder + decoder
- Loss: Cross-entropy on discrete unit sequences
- Requires: Extracting unit labels from target audio using teacher's unit extractor
- Duration: ~1000-1500 steps
- Result: Recovers audio translation quality

#### Advantages
1. **Memory Efficient**: Only student model in memory during training
2. **Proven Approach**: Your only-p7-cse465v5-s2st-corrected.ipynb showed text recovery works
3. **Modular**: Can debug text and audio recovery separately
4. **Lower GPU Requirements**: Can run on T4 (15GB VRAM)

#### Disadvantages
1. **Two-Phase Training**: More complex pipeline
2. **Suboptimal for T2U**: T2U trains on extracted units, not end-to-end
3. **Unit Extraction Overhead**: Need to run teacher inference on all training audio
4. **No Cross-Component Learning**: Speech encoder and T2U don't learn together

---

## Recommended Strategy: **Hybrid Approach**

### Why Hybrid?
- **Kaggle Constraints**: T4 GPU (15GB VRAM) cannot fit teacher + student simultaneously for full KD
- **Best of Both Worlds**: Use DoRA for text recovery (proven), then lightweight KD for T2U

### Implementation Plan

#### Phase 7a: DoRA Fine-tuning (Text Recovery)
**Duration**: 2000 steps | **Target**: Speech encoder + Text decoder

```python
# LoRA/DoRA config
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    use_dora=True,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'],
    # Exclude T2U from this phase
    modules_to_save=[]
)

# Loss: S2TT cross-entropy
loss = F.cross_entropy(student_text_logits, text_labels)
```

**Expected Result**: Text BLEU/ChrF recovers to ~90-95% of Phase 4 baseline

---

#### Phase 7b: T2U Distillation (Audio Recovery)
**Duration**: 1500 steps | **Target**: T2U encoder + decoder only

**Key Innovation**: Use teacher's **unit predictions** as soft targets (not extracted units)

```python
# Freeze speech encoder + text decoder (already recovered)
for param in model.speech_encoder.parameters():
    param.requires_grad = False
for param in model.text_decoder.parameters():
    param.requires_grad = False

# Only train T2U
for param in model.t2u_model.parameters():
    param.requires_grad = True

# Loss: KD on unit logits (no need to extract units!)
with torch.no_grad():
    teacher_unit_logits = teacher.generate(
        audio, return_intermediate_token_ids=True
    ).unit_logits  # [B, T, vocab_unit]

student_unit_logits = student.t2u_model(
    text_tokens, ...
).logits

# Soft distillation loss
loss_kd = F.kl_div(
    F.log_softmax(student_unit_logits / temperature, dim=-1),
    F.softmax(teacher_unit_logits / temperature, dim=-1),
    reduction='batchmean'
) * (temperature ** 2)

# Optional: Hard label loss (if you have target audio)
loss_ce = F.cross_entropy(student_unit_logits, target_units)

loss = 0.7 * loss_kd + 0.3 * loss_ce
```

**Why This Works**:
1. Teacher generates unit sequences on-the-fly (no pre-extraction needed)
2. Soft targets from teacher provide richer supervision than hard units
3. Only T2U trains → fits in T4 memory (teacher on CPU, student on GPU)
4. Speech encoder already recovered in Phase 7a → provides good input to T2U

---

### Memory-Efficient Implementation

#### Teacher on CPU, Student on GPU
```python
# Load teacher on CPU (inference only)
teacher = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
    'facebook/seamless-m4t-v2-large',
    torch_dtype=torch.float16,
    device_map='cpu'  # Keep on CPU
)
teacher.eval()

# Student on GPU (training)
student = load_model_from_drive('phase6_t2u_iter_pruned')
student = student.to('cuda:0')
student.train()

# During training
with torch.no_grad():
    # Teacher inference on CPU (slow but memory-safe)
    teacher_out = teacher.generate(audio.cpu(), ...)
    teacher_logits = teacher_out.unit_logits.to('cuda:0')

# Student forward on GPU (fast)
student_out = student.t2u_model(...)
loss = kl_div(student_out.logits, teacher_logits)
```

#### Batch Size Optimization
- Phase 7a (DoRA): Batch size 2-4 (text decoder is large)
- Phase 7b (T2U KD): Batch size 1-2 (teacher inference is slow)

---

## Expected Results

### After Phase 7a (DoRA)
- **Text BLEU**: 35-40 (from ~25 in Phase 6)
- **Text ChrF**: 50-55 (from ~40 in Phase 6)
- **Audio Quality**: Still broken (T2U not trained yet)

### After Phase 7b (T2U KD)
- **Text BLEU**: 35-40 (maintained)
- **Text ChrF**: 50-55 (maintained)
- **ASR-BLEU**: 25-30 (from ~5-10 in Phase 6)
- **ASR-ChrF**: 40-45 (from ~15-20 in Phase 6)
- **Audio Quality**: Intelligible, ~80-85% of baseline

---

## Alternative: Full KD (If You Have A100/V100)

If you can access a GPU with 40GB+ VRAM:

```python
# Both models on GPU
teacher = load_teacher().to('cuda:0')  # 2.3B params
student = load_student().to('cuda:0')  # 1.6B params

# Multi-level distillation
loss = (
    0.3 * kl_div(student_text_logits, teacher_text_logits) +
    0.2 * mse(student_speech_hidden, teacher_speech_hidden) +
    0.4 * kl_div(student_unit_logits, teacher_unit_logits) +
    0.1 * ce(student_text, labels)
)
```

**Training**: 3000-4000 steps, batch size 4-8

**Expected Result**: Better than hybrid (90-95% baseline recovery)

---

## Final Recommendation

### For Kaggle T4 GPU: **Hybrid Approach (Phase 7a + 7b)**

**Rationale**:
1. ✅ Fits in 15GB VRAM
2. ✅ Proven text recovery (Phase 7a)
3. ✅ Novel T2U distillation (Phase 7b) avoids unit extraction overhead
4. ✅ End-to-end audio recovery without NAR-specific complexity
5. ✅ Total training time: ~6-8 hours (manageable in Kaggle session)

### Implementation Priority
1. **Week 1**: Implement Phase 7a (DoRA) → Verify text recovery
2. **Week 2**: Implement Phase 7b (T2U KD) → Verify audio recovery
3. **Week 3**: Hyperparameter tuning + final benchmark

---

## Code Structure

```
phase7_recovery.ipynb
├── Cell 1-10: Setup (reuse from seamless-cse465v5.ipynb)
├── Cell 11: Load model_p6 from Kaggle dataset
├── Cell 12-15: Phase 7a - DoRA Fine-tuning
│   ├── LoRA injection
│   ├── S2TT training loop
│   ├── Text benchmark
│   └── Save phase7a_dora_merged
├── Cell 16-20: Phase 7b - T2U Distillation
│   ├── Load teacher (CPU)
│   ├── Freeze speech encoder + text decoder
│   ├── T2U KD training loop
│   ├── S2ST benchmark (with ASR-BLEU)
│   └── Save phase7b_final
└── Cell 21-25: Final evaluation + comparison plots
```

---

## Risk Mitigation

### If Phase 7b Fails (T2U KD doesn't work)
**Fallback**: Extract units from teacher and train with hard labels (original NAR approach)

```python
# Pre-extract units (run once)
unit_cache = []
for audio in train_data:
    with torch.no_grad():
        units = teacher.generate(audio, ...).unit_sequences
    unit_cache.append(units)

# Train T2U with hard labels
loss = F.cross_entropy(student_unit_logits, cached_units)
```

### If Memory Issues Persist
**Fallback**: Use gradient checkpointing + mixed precision

```python
student.gradient_checkpointing_enable()
scaler = torch.cuda.amp.GradScaler()
```

---

## Success Metrics

### Minimum Acceptable Recovery (Phase 7 Complete)
- Text BLEU: ≥35 (baseline: 42)
- Text ChrF: ≥50 (baseline: 58)
- ASR-BLEU: ≥25 (baseline: 38)
- ASR-ChrF: ≥40 (baseline: 52)
- **Compression**: 30% parameter reduction maintained
- **Speedup**: 1.5-2x RTF improvement maintained

### Stretch Goal
- Text BLEU: ≥38 (90% of baseline)
- ASR-BLEU: ≥32 (85% of baseline)

---

## Conclusion

**Choose Hybrid Approach** for your Kaggle T4 environment. It balances:
- ✅ Memory efficiency
- ✅ Training time
- ✅ Implementation complexity
- ✅ Expected quality recovery

The key innovation is **Phase 7b's on-the-fly unit distillation**, which avoids the unit extraction overhead while providing richer supervision than hard labels.

Start with Phase 7a to verify text recovery, then proceed to Phase 7b for audio recovery. This modular approach allows debugging each component independently.
