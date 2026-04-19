# Pre-Training Checklist for Phase 8 Full Model KD

## ✅ Completed Tasks

- [x] **Notebook Updated**: `full-kd.ipynb` updated with Full Model KD
- [x] **7 Training Cells**: All Phase 8 cells (1-7) replaced
- [x] **4 Benchmark Cells**: All updated with `phase8_full_kd`
- [x] **Documentation**: 6 comprehensive guides created
- [x] **Verification**: Update verified successfully

## 📋 Before Starting Training

### 1. Environment Setup

**Check these before running Phase 8:**

```python
# Run this in a notebook cell to verify prerequisites
import os
import torch

print("Environment Check:")
print(f"  Platform: {'Kaggle' if os.path.exists('/kaggle/working') else 'Colab'}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  PyTorch version: {torch.__version__}")
```

**Requirements:**
- [ ] GPU with 16 GB VRAM (minimum 12 GB)
- [ ] CUDA available
- [ ] PyTorch with CUDA support
- [ ] Transformers library installed

### 2. Model Prerequisites

**Phase 7 model must exist:**

```python
# Run this to check if Phase 7 model exists
import os

MODEL_DIR = '/kaggle/working/models' if os.path.exists('/kaggle/working') else '/content/drive/MyDrive/seamV5/models'
phase7_path = f'{MODEL_DIR}/phase7_dora_merged_v1'

if os.path.exists(phase7_path):
    files = os.listdir(phase7_path)
    print(f"✓ Phase 7 model found: {len(files)} files")
    print(f"  Path: {phase7_path}")
    
    # Check for essential files
    essential = ['config.json', 'pytorch_model.bin', 'processor_config.json']
    for f in essential:
        exists = f in files or any(f.startswith(f.split('.')[0]) for f in files)
        print(f"  {'✓' if exists else '✗'} {f}")
else:
    print(f"✗ Phase 7 model NOT found at {phase7_path}")
    print("  You must complete Phase 7 before starting Phase 8")
```

**Requirements:**
- [ ] `phase7_dora_merged_v1` model exists
- [ ] Model has config.json
- [ ] Model has weights (pytorch_model.bin or model.safetensors)
- [ ] Model has processor_config.json

### 3. Data Prerequisites

**Training data must be loaded:**

```python
# Run this to check if training data is ready
if 'ft_samples' in dir():
    print(f"✓ Training data loaded: {len(ft_samples)} samples")
else:
    print("✗ Training data not loaded")
    print("  Run Cell 24 to load FLEURS training data")

if 'eval_samples' in dir():
    print(f"✓ Eval data loaded: {len(eval_samples)} samples")
else:
    print("✗ Eval data not loaded")
    print("  Run Cell 23 to load FLEURS eval data")
```

**Requirements:**
- [ ] `ft_samples` loaded (training data)
- [ ] `eval_samples` loaded (evaluation data)
- [ ] Both have > 0 samples

### 4. Storage Space

**Check available disk space:**

```python
# Run this to check disk space
import shutil

def check_space(path):
    total, used, free = shutil.disk_usage(path)
    print(f"  Total: {total / 1e9:.1f} GB")
    print(f"  Used:  {used / 1e9:.1f} GB")
    print(f"  Free:  {free / 1e9:.1f} GB")
    return free / 1e9

work_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '/content/drive/MyDrive/seamV5'
print(f"Disk space at {work_dir}:")
free_gb = check_space(work_dir)

if free_gb > 5:
    print(f"✓ Sufficient space ({free_gb:.1f} GB free)")
else:
    print(f"✗ Low space ({free_gb:.1f} GB free)")
    print("  Need at least 5 GB for checkpoints and model")
```

**Requirements:**
- [ ] At least 5 GB free space
- [ ] Drive mounted (Colab) or rclone configured (Kaggle)

### 5. Previous Cells Executed

**These setup cells must be run first:**

- [ ] Cell 1-8: Setup and configuration
- [ ] Cell 11: I/O helpers (save/load functions)
- [ ] Cell 13: Core utilities and benchmark functions
- [ ] Cell 14: MMS-ASR setup
- [ ] Cell 20: load_base_model() function
- [ ] Cell 23: Load eval_samples
- [ ] Cell 24: Load ft_samples (training data)

**Quick check:**

```python
# Run this to verify essential functions exist
required_functions = [
    'load_base_model',
    'load_model_from_drive',
    'save_model_to_drive',
    'run_benchmark_full',
    'compute_full_kd_loss',  # This will be defined in Phase 8 Cell 3
]

for func in required_functions[:-1]:  # Skip the last one (defined in Phase 8)
    if func in dir():
        print(f"✓ {func} defined")
    else:
        print(f"✗ {func} NOT defined - run setup cells")
```

## 🚀 Ready to Start Training

Once all checkboxes above are checked, you're ready to run Phase 8!

### Training Sequence

**Run these cells in order:**

1. **Phase 8 Cell 1** - Load student model (~2 min)
   - Loads `phase7_dora_merged_v1`
   - Sets all parameters trainable
   - Verifies ~1B trainable params

2. **Phase 8 Cell 2** - Load teacher model (~3 min)
   - Loads `facebook/seamless-m4t-v2-large`
   - Sets to eval mode (frozen)
   - Verifies ~2.3B params

3. **Phase 8 Cell 3** - Define KD loss (~1 sec)
   - Defines `compute_full_kd_loss()` function
   - Sets up dual distillation (text + audio)

4. **Phase 8 Cell 4** - Setup optimizer (~1 sec)
   - Creates AdamW optimizer
   - Sets up cosine scheduler
   - Checks for existing checkpoints

5. **Phase 8 Cell 5** - Run training ⏱️ **8-16 HOURS**
   - Trains for 1000 steps
   - Saves checkpoints every 250 steps
   - Shows progress bar with loss metrics

6. **Phase 8 Cell 6** - Plot curves (~5 sec)
   - Generates 3 training plots
   - Saves figure to Drive

7. **Phase 8 Cell 7** - Save model (~2 min)
   - Saves `phase8_full_kd` to Drive
   - Syncs config with architecture

### During Training (Cell 5)

**What to expect:**

```
[P8] Full KD: 100%|████████| 1000/1000 [8:23:45<00:00, 30.23s/step, 
              loss=0.1234, kl=0.0890, audio=0.0344, lr=9.5e-06]
```

**Metrics:**
- `loss`: Total KD loss (should decrease)
- `kl`: Text distillation loss (should decrease)
- `audio`: Waveform MSE loss (should decrease)
- `lr`: Learning rate (cosine decay)

**Checkpoints:**
- Saved every 250 steps
- Location: `checkpoints/phase8_full_kd_step*.pt`
- Automatically synced to Drive (Kaggle)

**If interrupted:**
- Training can resume from last checkpoint
- Re-run Cell 5 to continue

### After Training

**Run benchmark cells:**

1. **Benchmark Cell 1** - Define function (~1 sec)
2. **Benchmark Cell 2** - Evaluate 4 models (~10 min)
   - Teacher, Phase 6, Phase 7, Phase 8
3. **Benchmark Cell 3** - Comparison plot (~5 sec)
4. **Benchmark Cell 4** - Radar chart (~5 sec)
5. **Benchmark Cell 5** - Summary table (~1 sec)

## 📊 Expected Results

### Training Metrics (Cell 6)

**Loss curves should show:**
- Total loss: Decreasing trend
- KL loss: Decreasing (text quality)
- Audio MSE: Decreasing (audio quality)

**If loss is flat after 100 steps:**
- Try increasing learning rate to 3e-5
- Or adjust alpha to 0.5 (more audio focus)

### Benchmark Metrics (Cell 5)

**Compared to Phase 7:**

| Metric | Phase 7 | Phase 8 Target | Change |
|--------|---------|----------------|--------|
| ASR-BLEU | Baseline | +2 to +5 | ↑ Better |
| ASR-ChrF | Baseline | +3 to +7 | ↑ Better |
| Text-BLEU | Good | Maintain | → Same |
| Text-ChrF | Good | Maintain | → Same |

**Success criteria:**
- ✅ ASR-BLEU increased (audio quality improved)
- ✅ ASR-ChrF increased (audio quality improved)
- ✅ Text-BLEU maintained (text quality preserved)
- ✅ Text-ChrF maintained (text quality preserved)

## 🔧 Troubleshooting

### Issue: Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. Code handles automatically (returns dummy loss, continues)
2. If persistent, edit Cell 4:
   ```python
   KD_GRAD_ACCUM = 16  # Was 8, now 16 (slower but less memory)
   ```

### Issue: Training Too Slow

**Symptoms:**
- More than 60 seconds per step
- Estimated time > 20 hours

**Solutions:**
1. This is normal for full model training
2. To speed up (at cost of quality):
   ```python
   KD_MAX_STEPS = 500  # Was 1000, now 500 (half training)
   ```

### Issue: Loss Not Decreasing

**Symptoms:**
- Loss flat after 100+ steps
- No improvement in metrics

**Solutions:**
1. Check after 100 steps (early steps can be noisy)
2. Try higher learning rate:
   ```python
   KD_LR = 3e-5  # Was 1e-5, now 3x higher
   ```
3. Or adjust alpha:
   ```python
   KD_ALPHA = 0.5  # Was 0.7, now more audio focus
   ```

### Issue: Checkpoint Not Saving

**Symptoms:**
```
[ckpt] WARNING: rclone push failed
```

**Solutions:**
1. **Kaggle**: Check rclone config (Cell 5)
2. **Colab**: Check Drive mounted (Cell 2)
3. Check disk space (see Storage Space section)

### Issue: Phase 7 Model Not Found

**Symptoms:**
```
[model] Path not found or empty: .../phase7_dora_merged_v1
```

**Solutions:**
1. You must complete Phase 7 first
2. Or download Phase 7 model from Drive
3. Verify path in Cell 1 of Phase 8

## 📞 Support

If you encounter issues not covered here:

1. **Check error message** - Most are self-explanatory
2. **Read implementation guide** - `PHASE8_FULL_KD_IMPLEMENTATION_GUIDE.md`
3. **Check quick start** - `QUICK_START_GUIDE.md`
4. **Verify prerequisites** - Run all checks in this document

## ✨ Final Checklist

Before clicking "Run" on Phase 8 Cell 5:

- [ ] All environment checks passed
- [ ] Phase 7 model exists and loads
- [ ] Training data loaded (ft_samples)
- [ ] Eval data loaded (eval_samples)
- [ ] At least 5 GB free space
- [ ] All setup cells (1-24) executed
- [ ] Phase 8 Cells 1-4 executed successfully
- [ ] GPU VRAM usage < 4 GB (room for training)

**If all checked:** You're ready! Run Phase 8 Cell 5 to start training. 🚀

**If any unchecked:** Complete the missing prerequisites first.

---

**Good luck with the training!** The Full Model KD approach is architecturally sound and should complete successfully. Expect 8-16 hours for 1000 steps, with checkpoints every 250 steps for safety.
