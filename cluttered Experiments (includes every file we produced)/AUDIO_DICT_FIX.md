# Audio Dictionary Fix - sample_id_to_audio

## Problem

Phase 6a training failed with:
```python
NameError: name 'sample_id_to_audio' is not defined
```

## Root Cause

The Phase 6a training loop needs to look up audio waveforms by sample ID to run the speech encoder on real audio. However, the `sample_id_to_audio` dictionary was never created.

The KD data (from Phase 5) stores:
- Sample ID (`'id'`)
- T2U input embeddings (`'t2u_input'`)
- Speaker embeddings (`'spk_emb'`)
- Unit IDs (`'unit_ids'`)

But **NOT** the actual audio waveforms. The audio needs to be looked up from the original training samples.

## The Fix

Added code to create the `sample_id_to_audio` dictionary at the beginning of Phase 6a training:

```python
# ── Create audio lookup dictionary ────────────────────────────────────
# Map sample IDs to audio waveforms for training
print("Creating sample_id_to_audio dictionary...")
sample_id_to_audio = {}

# Add from ft_samples (training data)
if "ft_samples" in globals() and ft_samples is not None:
    for s in ft_samples:
        if "id" in s and "wav" in s:
            sample_id_to_audio[s["id"]] = s["wav"]
    print(f"  Added {len(sample_id_to_audio)} samples from ft_samples")

# Add from all_train_samples if available
if "all_train_samples" in globals() and all_train_samples is not None:
    for pair_key, samples in all_train_samples.items():
        for s in samples:
            if "id" in s and "wav" in s:
                sample_id_to_audio[s["id"]] = s["wav"]
    print(f"  Total samples in dictionary: {len(sample_id_to_audio)}")

# Fallback: create from kd_data if it has wav field
if len(sample_id_to_audio) == 0:
    print("  Warning: ft_samples not found, trying to reconstruct from eval_samples...")
    if "eval_samples" in globals() and eval_samples is not None:
        for s in eval_samples:
            if "id" in s and "wav" in s:
                sample_id_to_audio[s["id"]] = s["wav"]
        print(f"  Added {len(sample_id_to_audio)} samples from eval_samples")

if len(sample_id_to_audio) == 0:
    raise RuntimeError("Could not create sample_id_to_audio dictionary. "
                       "Make sure ft_samples or eval_samples is loaded.")

print(f"✓ sample_id_to_audio ready with {len(sample_id_to_audio)} samples")
```

## How It Works

The dictionary is built from three possible sources (in order of priority):

1. **ft_samples** (primary) - Training samples loaded from `ft_samples.pt`
2. **all_train_samples** (secondary) - If available, adds more samples
3. **eval_samples** (fallback) - Evaluation samples if training samples not found

The dictionary maps:
```python
sample_id_to_audio = {
    'sample_001': numpy.array([...]),  # audio waveform
    'sample_002': numpy.array([...]),
    ...
}
```

## Files Modified

- **Alteration/seamless-final.ipynb** (Cell 75: Phase 6a training)
- **Backup**: `Alteration/seamless-final.ipynb.backup_before_audio_dict_fix`

## Expected Output

When you run Phase 6a training, you should now see:

```
Creating sample_id_to_audio dictionary...
  Added 1600 samples from ft_samples
✓ sample_id_to_audio ready with 1600 samples
Valid KD samples for Phase 6a: 1600 / 1600
Audio lookup: 1600 samples
```

## Verification

Before starting training, you can verify the dictionary was created:

```python
# Check if dictionary exists
print(f"sample_id_to_audio has {len(sample_id_to_audio)} samples")

# Check a sample
sample_id = list(sample_id_to_audio.keys())[0]
audio_wav = sample_id_to_audio[sample_id]
print(f"Sample {sample_id}: audio shape = {audio_wav.shape}")
```

Expected output:
```
sample_id_to_audio has 1600 samples
Sample fleurs_eng_001: audio shape = (48000,)
```

## Why This Was Needed

Phase 6a training runs the **real speech encoder** on actual audio every step:

```python
# Real audio
audio_wav = sample_id_to_audio.get(sample['id'])

# Real speech encoder forward (frozen)
with torch.no_grad():
    enc_out = model_6a.speech_encoder(
        input_features=inp_f,
        attention_mask=attn_m
    ).last_hidden_state.float()

# CIF connector forward
cif_out = model_6a.cif_connector(enc_out, tgt_lang_id=lang_id)
```

This is critical because:
- The CIF connector needs to learn from **real encoder outputs**, not cached embeddings
- The encoder outputs have different distributions than cached embeddings
- Training on real audio ensures the CIF learns the correct input distribution

## Summary

✓ **Problem**: `sample_id_to_audio` not defined  
✓ **Fix**: Added dictionary creation from `ft_samples`  
✓ **Location**: Cell 75 (Phase 6a training)  
✓ **Backup**: Created before applying fix  

The training should now start without errors! 🎉
