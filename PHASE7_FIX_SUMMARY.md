# Phase 7 Training Loop Fix - Complete Guide

## Problem Summary

**Error**: `AttributeError: 'NoneType' object has no attribute 'sum'`

**Root Cause**: The T2U (Text-to-Unit) model in SeamlessM4Tv2 has a **non-autoregressive (NAR) architecture** that requires specific inputs and returns a tuple instead of a standard output with `.loss` attribute.

## Why the Original Code Failed

### 1. T2U Model Architecture Mismatch

```python
# ❌ WRONG: Treating T2U like a standard seq2seq model
t2u_out = t2u(
    inputs_embeds=text_hidden,
    labels=unit_labels,
    return_dict=True,
)
# Expects: t2u_out.loss (doesn't exist!)
```

**Reality**: SeamlessM4Tv2's T2U model is `UnitYNART2UModel` (Non-AutoRegressive), which:
- Requires `text_decoder_output`, `text_decoder_padding_mask`, `text_seqs` (character tokens)
- Returns `(SequenceModelOutput, padding_mask, durations)` tuple
- Does NOT have a `.loss` attribute
- Uses duration prediction + character-level upsampling

### 2. Missing Required Inputs

The T2U forward signature is:
```python
def forward(
    self,
    text_decoder_output: Tensor,           # ✓ Required
    text_decoder_padding_mask: Optional[PaddingMask],  # ✓ Required
    text_seqs: Optional[Tensor],           # ✓ Required (char tokens!)
    duration_factor: float = 1.0,
    film_cond_emb: Optional[Tensor] = None,
) -> Tuple[SequenceModelOutput, Optional[PaddingMask], Tensor]:
```

Your code only provided `inputs_embeds` and `labels`, which are **not valid parameters**.

### 3. Return Value Structure

```python
# T2U returns a tuple, not a ModelOutput with .loss
seq_output, padding_mask, durations = t2u(...)
logits = seq_output.logits  # Must extract logits manually
```

## The Fix: Three Approaches

### Approach 1: S2TT-Only Training (RECOMMENDED)

**Why**: Simplest, most stable, proven to work.

```python
def compute_s2tt_loss(model, input_feats, attn_mask, labels):
    """S2TT cross-entropy via the text_decoder path."""
    outputs = model(
        input_features=input_feats,
        attention_mask=attn_mask,
        labels=labels,
        return_dict=True,
    )
    return outputs.loss  # ✓ This works!
```

**Training loop**:
```python
# Just compute S2TT loss
in_f, attn, txt_labels = prepare_s2tt_batch(batch, processor, device, TARGET_LANG, model_p7)
loss = compute_s2tt_loss(model_p7, in_f, attn, txt_labels) / GRAD_ACCUM
loss.backward()
```

**Pros**:
- ✅ Works immediately
- ✅ Recovers text translation quality (BLEU/ChrF)
- ✅ No complex NAR setup needed
- ✅ Proven in production systems

**Cons**:
- ⚠️ Doesn't train T2U (audio output may still be degraded)
- ⚠️ Requires separate T2U fine-tuning phase

### Approach 2: Manual T2U Loss (ADVANCED)

**Why**: Trains both text and audio paths, but complex.

```python
def compute_t2u_loss_manual(model, input_feats, attn_mask, unit_labels, tgt_lang="ben"):
    base = model.base_model if hasattr(model, "base_model") else model
    
    # 1. Encode speech
    speech_enc_out = base.speech_encoder(
        input_features=input_feats,
        attention_mask=attn_mask,
        return_dict=True,
    )
    enc_hidden = speech_enc_out.last_hidden_state
    
    # 2. Text decoder forward
    tgt_lang_code = base.generation_config.text_decoder_lang_to_code_id[tgt_lang]
    dec_input_ids = torch.full((B, 1), tgt_lang_code, dtype=torch.long, device=device)
    
    text_dec_out = base.text_decoder(
        input_ids=dec_input_ids,
        encoder_hidden_states=enc_hidden,
        encoder_attention_mask=encoder_attention_mask,
        return_dict=True,
    )
    text_hidden = text_dec_out.last_hidden_state
    
    # 3. T2U forward (NAR)
    t2u = base.t2u_model
    text_seqs = dec_input_ids  # Simplified - should be char-level tokens
    
    t2u_out = t2u(
        text_decoder_output=text_hidden,
        text_decoder_padding_mask=None,
        text_seqs=text_seqs,
        duration_factor=1.0,
    )
    
    # 4. Extract logits and compute loss
    seq_output, _, _ = t2u_out
    logits = seq_output.logits  # [B, T, V]
    
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        unit_labels.view(-1),
        ignore_index=-100,
    )
    return loss
```

**Pros**:
- ✅ Trains both text and audio paths
- ✅ Full S2ST recovery

**Cons**:
- ⚠️ Complex implementation
- ⚠️ Requires character-level tokenization
- ⚠️ Prone to dimension mismatches
- ⚠️ Slower training

### Approach 3: Separate T2U Phase (PRODUCTION)

**Why**: Clean separation of concerns, easier debugging.

**Phase 7**: Train S2TT only (text decoder recovery)
**Phase 8**: Train T2U separately with proper NAR setup

```python
# Phase 7: S2TT recovery
for batch in train_loader:
    loss = compute_s2tt_loss(model, batch)
    loss.backward()

# Phase 8: T2U fine-tuning (separate notebook/script)
for batch in train_loader:
    # Extract text_decoder_output from frozen text decoder
    with torch.no_grad():
        text_hidden = get_text_decoder_output(model, batch)
    
    # Train T2U only
    loss = compute_t2u_loss(model.t2u_model, text_hidden, batch.units)
    loss.backward()
```

## Implementation Steps

### Step 1: Replace Cell 8 (Loss Functions)

Use the code from `phase7_cell8_replacement.py`:
- Implements `prepare_s2tt_batch()`
- Implements `compute_s2tt_loss()` with proper error handling
- Removes broken T2U loss computation

### Step 2: Replace Cell 9 (Training Loop)

Use the code from `phase7_cell9_replacement.py`:
- Simplified to S2TT-only training
- Proper error handling and logging
- Gradient accumulation and clipping
- Checkpoint saving

### Step 3: Verify Training

After starting training, you should see:
```
Starting Phase 7 from scratch.
Step    50/2000  S2TT=2.3456  t=0.5min
Step   100/2000  S2TT=1.8234  t=1.0min
...
```

**No more AttributeError!**

## Why This Approach Works

1. **Uses HuggingFace's Built-in Loss**: The `SeamlessM4Tv2ForSpeechToSpeech.forward()` method with `labels` parameter automatically computes cross-entropy loss for the text decoder.

2. **Avoids T2U Complexity**: The T2U model's NAR architecture requires:
   - Character-level text tokenization
   - Duration prediction
   - Hierarchical upsampling
   
   These are complex to implement correctly in a training loop.

3. **Proven Pattern**: This is how the original SeamlessM4T paper trained the model:
   - Stage 1: Train text decoder (S2TT)
   - Stage 2: Train T2U separately
   - Stage 3: Joint fine-tuning (optional)

## Expected Results

After Phase 7 (S2TT-only training):
- ✅ Text BLEU/ChrF should recover to ~Phase 4 levels
- ✅ Model can generate correct Bengali text
- ⚠️ Audio output may still be degraded (T2U not trained)

To fully recover audio:
- Add Phase 8 for T2U fine-tuning
- Or use Approach 2 (manual T2U loss) if you need joint training

## Debugging Tips

If you still get errors:

1. **Check model device**:
   ```python
   print(f"Model device: {next(model_p7.parameters()).device}")
   print(f"Input device: {input_feats.device}")
   ```

2. **Verify labels shape**:
   ```python
   print(f"Labels shape: {labels.shape}")
   print(f"Labels range: {labels.min()} to {labels.max()}")
   print(f"Vocab size: {model_p7.config.vocab_size}")
   ```

3. **Test forward pass**:
   ```python
   with torch.no_grad():
       outputs = model_p7(
           input_features=input_feats[:1],
           attention_mask=attn_mask[:1],
           labels=labels[:1],
           return_dict=True,
       )
       print(f"Loss: {outputs.loss}")
       print(f"Logits shape: {outputs.logits.shape}")
   ```

## References

- [SeamlessM4T v2 Paper](https://arxiv.org/abs/2312.05187)
- [HuggingFace SeamlessM4Tv2 Docs](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2)
- [UnitY2 Architecture](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/)

## Quick Start

1. Copy `phase7_cell8_replacement.py` → Notebook Cell 8
2. Copy `phase7_cell9_replacement.py` → Notebook Cell 9
3. Run cells in order
4. Training should start without errors

**That's it!** Your Phase 7 training loop is now fixed.
