# Phase 8 KD Training Fix Instructions

## Problem
The original `compute_full_kd_loss` function had two critical errors:

1. **Type Error**: Token IDs were `float16` tensors, but embeddings require `long` (int64)
   ```
   Expected tensor for 'indices' to have Long or Int; but got torch.cuda.HalfTensor
   ```

2. **Architecture Error**: Using `model()` forward pass which doesn't use T2U model
   ```
   This is the same forward method as SeamlessM4Tv2ForSpeechToText. 
   It doesn't use self.t2u_model.
   ```

## Solution

### Step 1: Replace the function in your notebook

In **Phase 8 — Cell 3** (T2U KD Loss & Training Utilities), replace the entire `compute_full_kd_loss` function with the fixed version from `phase8_kd_loss_fixed.py`.

### Step 2: Key changes made

```python
# OLD (BROKEN):
labels = text_inputs['input_ids'].to(device)  # ❌ Wrong dtype
teacher_text_ids = teacher_gen.sequences      # ❌ Wrong dtype

student_forward = student(                     # ❌ Doesn't use T2U
    **inputs,
    labels=teacher_text_ids,
)

# NEW (FIXED):
labels = text_inputs['input_ids'].to(device).long()  # ✅ Correct dtype
teacher_text_ids = teacher_gen.sequences.long()      # ✅ Correct dtype

# ✅ Direct access to text_decoder (teacher-forced)
text_decoder_out = student.model.text_decoder(
    input_ids=decoder_input_ids,
    encoder_hidden_states=encoder_hidden,
)

# ✅ Direct access to T2U model (teacher-forced)
t2u_decoder_out = student.t2u_model.model.decoder(
    input_ids=t2u_decoder_input,
    encoder_hidden_states=t2u_encoder_out.last_hidden_state,
)
```

### Step 3: Update the training loop metrics

In **Phase 8 — Cell 5** (Training Loop), update the metric names:

```python
# OLD:
'text_kl': f'{avg_text_kl:.4f}',

# NEW:
'text_ce': f'{avg_text_ce:.4f}',
```

Also update the tracking lists:

```python
# OLD:
text_kl_losses = []

# NEW:
text_ce_losses = []
```

### Step 4: Update Cell 6 (Plotting)

Change the plot labels:

```python
# OLD:
ax.set_title('Text Sequence Distillation')
ax.set_ylabel('KL Loss')

# NEW:
ax.set_title('Text Decoder CE Loss')
ax.set_ylabel('CE Loss')
```

## Why This Works

### 1. Type Conversion
- PyTorch embeddings require integer indices (`torch.long` = `int64`)
- The model uses `float16` for weights, but indices must be integers
- `.long()` converts any tensor to `int64` dtype

### 2. Teacher-Forced Training
- `model.generate()` uses sampling/beam search (non-differentiable)
- We need gradients for KD, so we use teacher-forced forward passes
- Teacher provides the "correct" sequence, student learns to predict it

### 3. Direct Component Access
- `student.model.text_decoder(...)` - Text translation
- `student.t2u_model.model.decoder(...)` - Unit prediction for vocoder
- Both are trained simultaneously with proper gradient flow

## Expected Behavior After Fix

```
[P8] Full KD:   0%|          | 0/1000 [00:00<?, ?step/s]
[P8] Full KD:   1%|▏         | 10/1000 [00:15<25:30, loss=2.3456, text_ce=1.8234, t2u=0.4123, hard=0.1099, lr=1.00e-05]
[P8] Full KD:   2%|▎         | 20/1000 [00:30<24:45, loss=2.1234, text_ce=1.6543, t2u=0.3891, hard=0.0800, lr=9.98e-06]
...
```

## Verification Checklist

- [ ] No more "Expected Long but got HalfTensor" errors
- [ ] No more "doesn't use self.t2u_model" warnings
- [ ] Training progresses with decreasing loss
- [ ] Both `text_ce` and `t2u_loss` are non-zero
- [ ] Checkpoints save successfully every 250 steps

## Troubleshooting

### If you still get type errors:
```python
# Add explicit conversion everywhere:
teacher_text_ids = teacher_gen.sequences.long()
teacher_unit_ids = teacher_gen.unit_sequences.long() if hasattr(...) else None
labels = text_inputs['input_ids'].to(device).long()
```

### If T2U loss is always 0:
- Check that teacher generates `unit_sequences`:
  ```python
  print(f"Has units: {hasattr(teacher_gen, 'unit_sequences')}")
  print(f"Units shape: {teacher_gen.unit_sequences.shape if hasattr(...) else 'None'}")
  ```

### If OOM errors occur:
- Reduce `KD_BATCH_SIZE` from 1 to 1 (already minimal)
- Increase `KD_GRAD_ACCUM` from 8 to 16 (slower but less memory)
- Use shorter audio samples (filter `ft_samples` by duration)

## Next Steps

After training completes:
1. Run **Phase 8 — Cell 6** to plot training curves
2. Run **Phase 8 — Cell 7** to save the model
3. Run **Phase 8 — Benchmark Cells** to evaluate quality improvement
