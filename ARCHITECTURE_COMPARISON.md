# Architecture Comparison: Full vs Textless Model

## Visual Architecture Diagrams

### Full SeamlessM4T Model (only-p7-dora.ipynb)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL SEAMLESSM4T MODEL                       │
│                         (~2.3B params)                          │
└─────────────────────────────────────────────────────────────────┘

INPUT MODALITIES:
┌──────────────┐                    ┌──────────────┐
│  Text Input  │                    │ Audio Input  │
│   (tokens)   │                    │  (waveform)  │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       ▼                                   ▼
┌──────────────┐                    ┌──────────────┐
│Text Encoder  │                    │Speech Encoder│
│  24 layers   │                    │  24 layers   │
│   ~350M      │                    │   ~600M      │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       └───────────────┬───────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Text Decoder   │  ← DoRA APPLIED HERE
              │   24 layers    │
              │    ~500M       │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │   T2U Model    │  ← DoRA APPLIED HERE
              │  enc:6 dec:6   │
              │    ~220M       │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │    Vocoder     │
              │    ~50M        │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │ Audio Output   │
              └────────────────┘

TRAINING:
- Loss: Text CE (text_decoder generates text tokens)
- DoRA: Applied to text_decoder + t2u_model
- Input: Text or audio → text_decoder → T2U → vocoder → audio
```

---

### Textless Model (seamless-final.ipynb - YOUR MODEL)

```
┌─────────────────────────────────────────────────────────────────┐
│                      TEXTLESS MODEL                             │
│                       (~673M params)                            │
│                                                                 │
│  KEY DIFFERENCE: NO text encoder/decoder!                      │
│  Speech → CIF Connector → T2U → Vocoder → Speech              │
└─────────────────────────────────────────────────────────────────┘

INPUT MODALITY:
┌──────────────┐
│ Audio Input  │
│  (waveform)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Speech Encoder│  ← DoRA APPLIED HERE (Phase 6b)
│  16 layers   │     (pruned from 24 in Phase 4)
│   ~442M      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│CIF Connector │  ← CUSTOM COMPONENT (replaces text_decoder)
│  + Quantity  │     Trained in Phase 6a
│  Predictor   │     Learns speech→unit alignment
│    ~15M      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Speaker       │  ← CUSTOM COMPONENT
│Adapter       │     Trained in Phase 6a
│    ~1M       │     Projects speaker embeddings
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  T2U Model   │  ← DoRA APPLIED HERE (Phase 6b)
│ enc:6 dec:6  │
│   ~220M      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Vocoder    │
│    ~50M      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Audio Output  │
└──────────────┘

TRAINING:
- Loss: Unit CE (T2U generates unit tokens, not text)
- DoRA: Applied to speech_encoder + t2u_model (NOT text_decoder)
- Input: Audio → speech_encoder → CIF → T2U → vocoder → audio
```

---

## Component-by-Component Comparison

### Text Encoder
```
Full Model:     ✓ EXISTS (24 layers, ~350M params)
                  Purpose: Encode text input
                  
Textless Model: ✗ REMOVED in Phase 2
                  Reason: Speech-to-speech only, no text input needed
```

### Text Decoder
```
Full Model:     ✓ EXISTS (24 layers, ~500M params)
                  Purpose: Generate text tokens
                  DoRA: APPLIED HERE
                  
Textless Model: ✗ REMOVED in Phase 3
                  Replaced by: CIF Connector (~15M params)
                  Reason: Direct speech→unit mapping, skip text bottleneck
```

### Speech Encoder
```
Full Model:     ✓ EXISTS (24 layers, ~600M params)
                  DoRA: NOT applied (kept frozen)
                  
Textless Model: ✓ EXISTS (16 layers, ~442M params)
                  Pruned: Phase 4 removed 8 layers
                  DoRA: APPLIED HERE in Phase 6b ✓
```

### CIF Connector
```
Full Model:     ✗ DOES NOT EXIST
                  
Textless Model: ✓ CUSTOM COMPONENT (~15M params)
                  Purpose: Speech encoder → T2U input alignment
                  Features:
                  - Continuous Integrate-and-Fire mechanism
                  - Quantity predictor (predicts output length)
                  - Trained in Phase 6a with feature KD
```

### Speaker Adapter
```
Full Model:     ✗ DOES NOT EXIST
                  
Textless Model: ✓ CUSTOM COMPONENT (~1M params)
                  Purpose: Project speaker embeddings
                  Trained: Phase 6a alongside CIF connector
```

### T2U Model
```
Full Model:     ✓ EXISTS (enc:6 dec:6, ~220M params)
                  Input: Text decoder output
                  DoRA: APPLIED HERE
                  
Textless Model: ✓ EXISTS (enc:6 dec:6, ~220M params)
                  Input: CIF connector output
                  DoRA: APPLIED HERE in Phase 6b ✓
```

### Vocoder
```
Full Model:     ✓ EXISTS (~50M params)
                  Input: T2U units
                  
Textless Model: ✓ EXISTS (~50M params)
                  Input: T2U units
                  (Same as full model)
```

---

## Data Flow Comparison

### Full Model (Text-Based):
```
Audio Input
    ↓
Speech Encoder (24 layers)
    ↓
Text Decoder (24 layers) ← Generates TEXT tokens
    ↓                       Loss: Text CE
T2U Model (enc:6 dec:6)  ← Converts text→units
    ↓
Vocoder
    ↓
Audio Output
```

### Textless Model (Direct Speech-to-Speech):
```
Audio Input
    ↓
Speech Encoder (16 layers) ← Pruned, DoRA applied
    ↓
CIF Connector (~15M)       ← Replaces text decoder
    ↓                         Learns alignment directly
    ├─ Quantity Predictor     Loss: Cosine + MSE + Qty
    └─ Speaker Adapter
    ↓
T2U Model (enc:6 dec:6)    ← Generates UNIT tokens
    ↓                         Loss: Unit CE
Vocoder
    ↓
Audio Output
```

---

## Why Phase 6b Code Needs Different Approach

### Full Model Training (only-p7-dora.ipynb):
```python
# Apply DoRA to text_decoder (exists in full model)
model.text_decoder = get_peft_model(model.text_decoder, lora_cfg)  ✓

# Apply DoRA to T2U
model.t2u_model = get_peft_model(model.t2u_model, lora_cfg)  ✓

# Training loop
text_out = model.text_decoder(
    inputs_embeds=encoder_out,
    labels=text_ids)  # Text labels
loss = text_out.loss  # Text CE loss
```

### Textless Model Training (seamless-final.ipynb - FIXED):
```python
# Apply DoRA to speech_encoder (not text_decoder!)
model.speech_encoder = get_peft_model(model.speech_encoder, lora_cfg)  ✓

# Apply DoRA to T2U
model.t2u_model = get_peft_model(model.t2u_model, lora_cfg)  ✓

# Training loop
enc_out = model.speech_encoder(input_features=audio)  # Real forward pass
connector_out = model.cif_connector(enc_out)  # CIF alignment
t2u_out = model.t2u_model(
    inputs_embeds=connector_out,
    labels=unit_ids)  # Unit labels (not text!)
loss = t2u_out.loss  # Unit CE loss
```

---

## Parameter Count Breakdown

### Full Model (~2.3B total):
```
Component           Params      Percentage
─────────────────────────────────────────
Text Encoder        ~350M       15.2%
Speech Encoder      ~600M       26.1%
Text Decoder        ~500M       21.7%  ← DoRA applied
T2U Model           ~220M        9.6%  ← DoRA applied
Vocoder             ~50M         2.2%
Other               ~580M       25.2%
─────────────────────────────────────────
TOTAL               ~2300M      100%
```

### Textless Model (~673M total):
```
Component           Params      Percentage
─────────────────────────────────────────
Speech Encoder      ~442M       65.7%  ← DoRA applied (Phase 6b)
CIF Connector       ~15M         2.2%  ← Trained (Phase 6a)
Speaker Adapter     ~1M          0.1%  ← Trained (Phase 6a)
T2U Model           ~220M       32.7%  ← DoRA applied (Phase 6b)
Vocoder             ~50M         7.4%
Other               ~5M          0.7%
─────────────────────────────────────────
TOTAL               ~673M       100%

COMPRESSION: 2300M → 673M = 70.7% reduction
```

---

## Training Differences Summary

| Aspect | Full Model | Textless Model |
|--------|------------|----------------|
| **Input** | Text or Audio | Audio only |
| **Intermediate** | Text tokens | Unit tokens directly |
| **Text Decoder** | Exists, DoRA applied | Removed, replaced by CIF |
| **Speech Encoder** | Frozen | DoRA applied |
| **Loss Function** | Text CE | Unit CE |
| **Training Data** | Text labels | Unit labels |
| **Forward Pass** | Can use cached | Must be real (for DoRA gradients) |
| **Alignment** | Implicit in decoder | Explicit in CIF connector |

---

## Why This Matters for Phase 6b

### Problem:
Original Phase 6b code was copied from `only-p7-dora.ipynb` which trains the **FULL MODEL**:
```python
# This line FAILS on textless model:
model.text_decoder = get_peft_model(model.text_decoder, lora_cfg)
# AttributeError: 'SeamlessM4Tv2Model' object has no attribute 'text_decoder'
```

### Solution:
Apply DoRA only to components that **EXIST** in textless model:
```python
# These lines WORK on textless model:
model.speech_encoder = get_peft_model(model.speech_encoder, lora_cfg)  ✓
model.t2u_model = get_peft_model(model.t2u_model, lora_cfg)  ✓
```

---

## Verification Commands

### Check Model Architecture:
```python
# Full model has these:
assert hasattr(full_model, 'text_encoder')   # ✓
assert hasattr(full_model, 'text_decoder')   # ✓
assert hasattr(full_model, 'speech_encoder') # ✓
assert hasattr(full_model, 't2u_model')      # ✓

# Textless model has these:
assert not hasattr(textless_model, 'text_encoder')   # ✓ Removed
assert not hasattr(textless_model, 'text_decoder')   # ✓ Removed
assert hasattr(textless_model, 'speech_encoder')     # ✓ Exists (pruned)
assert hasattr(textless_model, 't2u_model')          # ✓ Exists
assert hasattr(textless_model, 'cif_connector')      # ✓ Custom component
assert hasattr(textless_model, 'speaker_adapter')    # ✓ Custom component
```

### Check DoRA Application:
```python
# Full model DoRA:
print(full_model.text_decoder)  # Should show LoRA layers
print(full_model.t2u_model)     # Should show LoRA layers

# Textless model DoRA:
print(textless_model.speech_encoder)  # Should show LoRA layers
print(textless_model.t2u_model)       # Should show LoRA layers
```

---

## Key Takeaway

**You cannot blindly copy training code between different model architectures!**

The full model and textless model have fundamentally different architectures:
- Full model: Text-based intermediate representation
- Textless model: Direct speech-to-speech with custom alignment

Phase 6b fix adapts the DoRA training approach to work with the textless architecture by:
1. Applying DoRA to correct components (speech_encoder + t2u_model)
2. Using correct loss function (unit CE, not text CE)
3. Using correct data labels (unit_ids, not text_ids)
4. Running real forward passes (not cached embeddings)

This is why the fix is a complete rewrite, not just a small patch!
