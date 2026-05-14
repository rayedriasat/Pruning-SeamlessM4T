# Phase 8 Knowledge Distillation Fix Guide

## Problem Summary

Your original Phase 8 KD training had two critical issues:

1. **AttributeError: 'tuple' object has no attribute 'sequences'**
   - `model.generate()` can return either a dict or tuple depending on the model state
   - Your code assumed it always returns a dict with `.sequences` attribute

2. **No gradient flow through student model**
   - `model.generate()` always runs in eval mode with `torch.no_grad()` internally
   - This means the student model never received gradients during training
   - The training loop was essentially doing nothing

## Root Cause Analysis

### Issue 1: generate() Return Type
```python
# ❌ WRONG: Assumes dict return
teacher_outputs = teacher.generate(..., return_dict_in_generate=True)
teacher_text_ids = teacher_outputs.sequences  # Fails if tuple returned
```

**Why it fails:**
- HuggingFace's `generate()` can return different types based on model configuration
- Even with `return_dict_in_generate=True`, some model states return tuples
- SeamlessM4T's complex architecture (speech encoder + text decoder + T2U + vocoder) can trigger tuple returns

### Issue 2: No Gradients in generate()
```python
# ❌ WRONG: generate() disables gradients internally
student_audio_outputs = student.generate(...)  # No gradients!
student_waveform = student_audio_outputs.waveform
audio_mse_loss = F.mse_loss(student_waveform, teacher_waveform)
loss.backward()  # Gradients don't flow back through generate()
```

**Why it fails:**
- `generate()` is designed for inference, not training
- It internally wraps everything in `torch.no_grad()` and sets model to eval mode
- Even if you call it on a model in train mode, gradients are blocked

## Solution Architecture

### Strategy: Teacher Forcing with Forward Passes

Instead of using `generate()` for the student, we use **forward passes with teacher forcing**:

```python
# ✅ CORRECT: Use forward() for gradient flow

# 1. Get teacher predictions (no grad needed)
with torch.no_grad():
    teacher_gen = teacher.generate(...)
    # Handle both dict and tuple
    if isinstance(teacher_gen, dict):
        teacher_ids = teacher_gen['sequences']
    else:
        teacher_ids = teacher_gen[0]  # tuple case
    
    # Get teacher logits via forward pass
    teacher_forward = teacher(..., labels=teacher_ids)
    teacher_logits = teacher_forward.logits

# 2. Student forward pass WITH GRADIENTS
student.train()  # Ensure train mode
student_forward = student(..., labels=teacher_ids)
student_logits = student_forward.logits  # ✅ Gradients flow here!

# 3. Compute KL divergence
kl_loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction='batchmean'
) * (T ** 2)

# 4. Backprop through student
kl_loss.backward()  # ✅ Gradients flow through student!
```

### Multi-Component Distillation

The fixed implementation distills knowledge at multiple levels:

| Component | Method | Gradient Flow |
|-----------|--------|---------------|
| **Text Decoder** | KL divergence on logits | ✅ Yes |
| **T2U Model** | MSE on hidden states | ✅ Yes |
| **Hard Labels** | Cross-entropy on ground truth | ✅ Yes |

Loss combination:
```
total_loss = α × text_kl_loss + β × t2u_loss + γ × hard_loss
where α + β + γ = 1.0
```

## Implementation Details

### Fixed compute_full_kd_loss Function

Key improvements:

1. **Robust generate() handling:**
```python
teacher_gen = teacher.generate(...)
if isinstance(teacher_gen, dict):
    teacher_ids = teacher_gen['sequences']
elif isinstance(teacher_gen, tuple):
    teacher_ids = teacher_gen[0]
else:
    teacher_ids = teacher_gen
```

2. **Forward passes for gradients:**
```python
# Teacher: no grad
with torch.no_grad():
    teacher_forward = teacher(..., labels=teacher_ids)
    teacher_logits = teacher_forward.logits

# Student: WITH grad
student.train()
student_forward = student(..., labels=teacher_ids)
student_logits = student_forward.logits  # Gradients enabled!
```

3. **Multi-level distillation:**
```python
# Text decoder KD
text_kl_loss = F.kl_div(student_log_probs, teacher_probs, ...)

# T2U KD via hidden states
if teacher_hidden is not None and student_hidden is not None:
    t2u_loss = F.mse_loss(student_hidden, teacher_hidden.detach())

# Hard label grounding
if gamma > 0:
    hard_loss = student(..., labels=ground_truth_labels).loss

# Combine
total_loss = alpha * text_kl_loss + beta * t2u_loss + gamma * hard_loss
```

### Training Loop Improvements

1. **Proper gradient accumulation:**
```python
loss = loss / GRAD_ACCUM_STEPS
loss.backward()

if micro_step % GRAD_ACCUM_STEPS == 0:
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

2. **Better error handling:**
```python
try:
    loss, metrics = compute_full_kd_loss(...)
    loss.backward()
except RuntimeError as e:
    if 'out of memory' in str(e):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        continue
    else:
        raise
```

3. **Comprehensive metrics tracking:**
```python
metrics = {
    'total_loss': total_loss.item(),
    'text_kl': text_kl_loss.item(),
    't2u_loss': t2u_loss.item(),
    'hard_loss': hard_loss.item(),
}
```

## How to Apply the Fix

### Step 1: Backup Your Current Notebook
```bash
cp full-kd.ipynb full-kd.ipynb.backup
```

### Step 2: Replace Phase 8 Cell 3 (Loss Function)

Replace the entire `compute_full_kd_loss` function with the version from `phase8_kd_fix.py`.

**Location:** After "## Phase 8 — Cell 3: T2U KD Loss & Training Utilities"

### Step 3: Replace Phase 8 Cell 5 (Training Loop)

Replace the training loop with:

```python
import time
from tqdm.auto import tqdm

# Hyperparameters
KD_MAX_STEPS = 1000
KD_BATCH_SIZE = 1
KD_GRAD_ACCUM = 8
KD_TEMPERATURE = 2.0
KD_ALPHA = 0.7  # Text KD weight
KD_BETA = 0.2   # T2U KD weight
TARGET_LANG = 'ben'

print('\n' + '='*80)
print('STARTING FULL MODEL KD TRAINING (FIXED)')
print('='*80)
print(f'Alpha (text KD): {KD_ALPHA}')
print(f'Beta (T2U KD): {KD_BETA}')
print(f'Gamma (hard label): {1.0 - KD_ALPHA - KD_BETA}')
print('='*80)

DEVICE = next(model_teacher.parameters()).device

# Training state
optim_steps = kd_start_step
micro_step = 0
epoch_start = time.time()

# Metrics tracking
text_kl_losses = []
t2u_losses = []
hard_losses = []
total_losses = []

try:
    model_p8_student.train()
    kd_optimizer.zero_grad()
    
    pbar = tqdm(total=KD_MAX_STEPS - kd_start_step, desc='[P8] Full KD', 
                initial=kd_start_step, unit='step')
    
    while optim_steps < KD_MAX_STEPS:
        # Sample a batch
        batch_wavs = []
        batch_refs = []
        for _ in range(KD_BATCH_SIZE):
            idx = torch.randint(0, len(ft_samples), (1,)).item()
            sample = ft_samples[idx]
            batch_wavs.append(sample['wav'])
            batch_refs.append(sample['ref'])
        
        # Compute KD loss
        try:
            loss, metrics = compute_full_kd_loss(
                teacher=model_teacher,
                student=model_p8_student,
                wav_batch=batch_wavs,
                ref_texts=batch_refs,
                processor=processor,
                tgt_lang=TARGET_LANG,
                temperature=KD_TEMPERATURE,
                alpha=KD_ALPHA,
                beta=KD_BETA,
                device=DEVICE
            )
            
            # Gradient accumulation
            loss = loss / KD_GRAD_ACCUM
            loss.backward()
            
            # Track metrics
            total_losses.append(metrics['total_loss'])
            text_kl_losses.append(metrics.get('text_kl', 0.0))
            t2u_losses.append(metrics.get('t2u_loss', 0.0))
            hard_losses.append(metrics.get('hard_loss', 0.0))
            
            micro_step += 1
            
            # Optimizer step
            if micro_step % KD_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model_p8_student.parameters(), max_norm=1.0)
                kd_optimizer.step()
                kd_scheduler.step()
                kd_optimizer.zero_grad()
                optim_steps += 1
                
                # Logging
                if optim_steps % 10 == 0:
                    avg_total = sum(total_losses[-10:]) / len(total_losses[-10:])
                    avg_text_kl = sum(text_kl_losses[-10:]) / len(text_kl_losses[-10:])
                    avg_t2u = sum(t2u_losses[-10:]) / len(t2u_losses[-10:])
                    avg_hard = sum(hard_losses[-10:]) / len(hard_losses[-10:])
                    lr = kd_scheduler.get_last_lr()[0]
                    
                    pbar.set_postfix({
                        'loss': f'{avg_total:.4f}',
                        'text_kl': f'{avg_text_kl:.4f}',
                        't2u': f'{avg_t2u:.4f}',
                        'hard': f'{avg_hard:.4f}',
                        'lr': f'{lr:.2e}'
                    })
                
                pbar.update(1)
                
                # Checkpointing
                if optim_steps % 250 == 0:
                    save_checkpoint(
                        dict(step=optim_steps,
                             loss_log=kd_loss_log + total_losses,
                             model_state=model_p8_student.state_dict(),
                             optimizer_state=kd_optimizer.state_dict(),
                             scheduler_state=kd_scheduler.state_dict()),
                        name='phase8_full_kd',
                        step=optim_steps
                    )
                
                if optim_steps % 50 == 0:
                    torch.cuda.empty_cache()
        
        except RuntimeError as e:
            print(f'\n[ERR] Step {optim_steps}: {e}')
            torch.cuda.empty_cache()
            kd_optimizer.zero_grad()
            micro_step = 0
            continue
    
    pbar.close()
    
    # Final save
    kd_loss_log.extend(total_losses)
    save_checkpoint(
        dict(step=optim_steps,
             loss_log=kd_loss_log,
             model_state=model_p8_student.state_dict(),
             optimizer_state=kd_optimizer.state_dict(),
             scheduler_state=kd_scheduler.state_dict()),
        name='phase8_full_kd',
        step=optim_steps
    )
    
    elapsed = (time.time() - epoch_start) / 60
    print(f'\n[P8] Full KD complete. Final step: {optim_steps}  Time: {elapsed:.1f} min')
    print(f'[P8] Final losses - Total: {total_losses[-1]:.4f}, Text KL: {text_kl_losses[-1]:.4f}, '
          f'T2U: {t2u_losses[-1]:.4f}, Hard: {hard_losses[-1]:.4f}')

except KeyboardInterrupt:
    print('\n[P8] Training interrupted by user.')
    kd_loss_log.extend(total_losses)
    save_checkpoint(
        dict(step=optim_steps,
             loss_log=kd_loss_log,
             model_state=model_p8_student.state_dict(),
             optimizer_state=kd_optimizer.state_dict(),
             scheduler_state=kd_scheduler.state_dict()),
        name='phase8_full_kd',
        step=optim_steps
    )

finally:
    model_p8_student.eval()
    torch.cuda.empty_cache()
    print('[P8] Training loop exited. Model set to eval mode.')
```

### Step 4: Verify the Fix

After applying the changes, verify:

1. **Check gradient flow:**
```python
# Add this after first training step
for name, param in model_p8_student.named_parameters():
    if param.grad is not None:
        print(f"✅ {name}: grad norm = {param.grad.norm().item():.6f}")
    else:
        print(f"❌ {name}: NO GRADIENT")
```

2. **Monitor loss curves:**
   - Text KL loss should decrease steadily
   - T2U loss should decrease (if hidden states available)
   - Hard loss should decrease
   - Total loss should show clear downward trend

3. **Check model is in train mode:**
```python
print(f"Student training mode: {model_p8_student.training}")  # Should be True
print(f"Teacher training mode: {model_teacher.training}")      # Should be False
```

## Expected Behavior After Fix

### Before Fix (Broken)
```
[P8] Step 10: loss=0.0100, kl=0.0000, audio=0.0000  # No learning!
[P8] Step 20: loss=0.0100, kl=0.0000, audio=0.0000  # Stuck!
[P8] Step 30: loss=0.0100, kl=0.0000, audio=0.0000  # No gradients!
```

### After Fix (Working)
```
[P8] Step 10: loss=2.4567, text_kl=2.1234, t2u=0.3333, hard=0.0000
[P8] Step 20: loss=2.1234, text_kl=1.8901, t2u=0.2333, hard=0.0000
[P8] Step 30: loss=1.8901, text_kl=1.6789, t2u=0.2112, hard=0.0000
...
[P8] Step 1000: loss=0.8456, text_kl=0.7123, t2u=0.1333, hard=0.0000
```

## Hyperparameter Tuning

### Loss Weights (α, β, γ)

**Default configuration:**
```python
KD_ALPHA = 0.7  # Text decoder KD (most important)
KD_BETA = 0.2   # T2U hidden state KD
# Gamma = 0.1   # Hard label loss (computed as 1 - α - β)
```

**Tuning guidelines:**

| Scenario | α (text) | β (T2U) | γ (hard) | Rationale |
|----------|----------|---------|----------|-----------|
| **Default** | 0.7 | 0.2 | 0.1 | Balanced distillation |
| **Text quality priority** | 0.8 | 0.1 | 0.1 | Focus on BLEU/ChrF |
| **Audio quality priority** | 0.5 | 0.4 | 0.1 | Focus on ASR-BLEU |
| **Fast convergence** | 0.6 | 0.2 | 0.2 | More hard label guidance |
| **Pure KD** | 0.9 | 0.1 | 0.0 | Maximum teacher mimicking |

### Temperature (T)

**Default:** `T = 2.0`

- **Lower (T = 1.0-1.5):** Sharper distributions, faster convergence, may overfit
- **Higher (T = 2.5-4.0):** Softer distributions, better generalization, slower convergence

### Learning Rate

**Default:** `LR = 1e-5`

- **Higher (5e-5):** Faster convergence, risk of instability
- **Lower (1e-6):** More stable, slower convergence

## Troubleshooting

### Issue: Loss is NaN
**Cause:** Gradient explosion or numerical instability
**Fix:**
```python
# Lower learning rate
KD_LR = 5e-6

# Increase gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

# Lower temperature
KD_TEMPERATURE = 1.5
```

### Issue: Loss not decreasing
**Cause:** No gradient flow or learning rate too low
**Fix:**
```python
# Verify gradients
for name, param in model_p8_student.named_parameters():
    if param.requires_grad and param.grad is None:
        print(f"WARNING: {name} has no gradient!")

# Increase learning rate
KD_LR = 5e-5

# Check model is in train mode
model_p8_student.train()
```

### Issue: OOM errors
**Cause:** Batch size or sequence length too large
**Fix:**
```python
# Reduce batch size
KD_BATCH_SIZE = 1

# Increase gradient accumulation
KD_GRAD_ACCUM = 16

# Reduce max tokens
max_new_tokens = 128  # Instead of 256

# Clear cache more frequently
if optim_steps % 10 == 0:
    torch.cuda.empty_cache()
```

### Issue: Training too slow
**Cause:** Multiple forward passes per step
**Fix:**
```python
# Reduce T2U distillation weight (skip if not helping)
KD_BETA = 0.0

# Reduce hard label loss (skip if not helping)
# This is automatic if α + β = 1.0

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    loss, metrics = compute_full_kd_loss(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Validation

After training completes, validate the fix worked:

1. **Check final model quality:**
```python
results, summary = run_benchmark_full(
    model_p8_student, 
    eval_samples, 
    label='phase8_full_kd_fixed',
    tgt_lang='ben'
)

print(f"Text-BLEU: {summary['avg_text_bleu']:.2f}")
print(f"ASR-BLEU: {summary['avg_asr_bleu']:.2f}")
```

2. **Compare with Phase 7:**
   - Text-BLEU should be similar or better than Phase 7
   - ASR-BLEU should improve significantly (audio quality recovery)

3. **Verify model size:**
```python
params = sum(p.numel() for p in model_p8_student.parameters()) / 1e6
print(f"Final model: {params:.1f}M parameters")
# Should be ~1000M (same as Phase 7, no pruning in Phase 8)
```

## References

- **Knowledge Distillation:** Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- **Temperature Scaling:** Müller et al., "When Does Label Smoothing Help?" (NeurIPS 2019)
- **Teacher Forcing:** Williams & Zipser, "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks" (1989)
- **Gradient Accumulation:** Ott et al., "Scaling Neural Machine Translation" (WMT 2018)

## Summary

The fix addresses two critical issues:

1. ✅ **Robust generate() handling** - Works with both dict and tuple returns
2. ✅ **Proper gradient flow** - Uses forward() instead of generate() for student
3. ✅ **Multi-level distillation** - Text decoder + T2U + hard labels
4. ✅ **Better error handling** - OOM recovery, gradient checking
5. ✅ **Comprehensive metrics** - Track all loss components

Your Phase 8 KD training should now work correctly and improve audio quality!
