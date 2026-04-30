# Quick Fix Reference - Vocab Mismatch

## Problem
```
Teacher cache tokens: [256022, 178607, 247676, ...] ← 256K vocab
Student model vocab:  0 to 22,766                  ← 22K vocab
Result: CUDA assertion error
```

## Solution Decision Tree

```
Step 1: Check if you have vocab remapping
├─ Run: hasattr(model_student, '_vocab_remap_to_old')
│
├─ YES → You can use Option 1 OR Option 2
│         Recommendation: Use Option 2 (better quality)
│
└─ NO → You MUST use Option 2
```

---

## Option 1: Token Remapping (5 minutes)

**When to use**: If you have `_vocab_remap_to_old` and need to start training ASAP

**Steps**:
1. Update `text_recovery_step` function with remapping logic
2. Test with one sample
3. Start training

**Code to add to text_recovery_step**:
```python
# Inside use_teacher_text block:
old_to_new = {old_id: new_id for new_id, old_id in enumerate(model_student._vocab_remap_to_old)}
remapped_seq = teacher_seq.clone()
for i in range(len(remapped_seq)):
    old_id = int(remapped_seq[i].item())
    remapped_seq[i] = old_to_new.get(old_id, 1)  # 1 = <unk>
labels = remapped_seq.unsqueeze(0).to(student_device)
```

---

## Option 2: Rebuild Cache (40 minutes) ⭐ RECOMMENDED

**When to use**: Always (best quality), or if Option 1 not available

**Steps**:
1. Clear old cache files: `rm {CKPT_DIR}/phase6_teacher_cache_*`
2. Set `model_teacher = model_student`
3. Rebuild cache: `build_or_load_phase6_cache('train', ft_samples, ...)`
4. Start training

**Why better**:
- ✓ Student learns from its own output distribution
- ✓ No information loss from unmapped tokens
- ✓ Simpler code (no remapping)
- ✓ More accurate for 22K vocab model

---

## Test Your Fix

```python
# Get a sample
sample, cache_entry = phase6_pick_training_pair(max_audio_sec=20, balanced=True)

# Check token range
teacher_seq = cache_entry['teacher_text_sequences']
print(f"Token range: [{teacher_seq.min()}, {teacher_seq.max()}]")
print(f"Model vocab: {model_student.config.vocab_size}")

# Test forward pass
loss = text_recovery_step(sample, cache_entry, use_teacher_text=True)
print(f"✓ Success! Loss: {loss.item():.4f}")
```

---

## Files to Read

1. **APPLY_VOCAB_FIX_NOW.md** - Complete step-by-step guide
2. **SOLUTION_VOCAB_MISMATCH.md** - Detailed explanation
3. **FINAL_FIX_text_recovery_step.py** - Updated function (Option 1)

---

## Quick Commands

### Check remapping availability:
```python
hasattr(model_student, '_vocab_remap_to_old')
```

### Clear cache (Option 2):
```python
import glob
for f in glob.glob(f'{CKPT_DIR}/phase6_teacher_cache_*'):
    os.remove(f)
```

### Set student as teacher (Option 2):
```python
model_teacher = model_student
model_teacher.eval()
```

### Rebuild cache (Option 2):
```python
phase6_cache_manifest = build_or_load_phase6_cache('train', ft_samples, shard_size=PHASE6_CACHE_SHARD_SIZE)
```

---

## Expected Result

After fix:
```
[6b1] step   50/300 | loss=2.3456 | KD=50% | lr=1.00e-04
[6b1] step  100/300 | loss=2.1234 | KD=50% | lr=9.50e-05
```

No more CUDA errors! ✓
