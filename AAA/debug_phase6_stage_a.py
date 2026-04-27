# ══════════════════════════════════════════════════════════════════════════════
# DEBUG CELLS FOR PHASE 6 STAGE A - T2U Training Issue
# Error: 'NoneType' object has no attribute 'sum'
# ══════════════════════════════════════════════════════════════════════════════

# ── DEBUG 1: Check if models are loaded correctly ────────────────────────────
print("="*80)
print("DEBUG 1: Model Loading Check")
print("="*80)

print(f"\nStudent model type: {type(model_student)}")
print(f"Teacher model type: {type(model_teacher)}")
print(f"Processor type: {type(processor)}")

# Check T2U model structure
print(f"\nStudent has t2u_model: {hasattr(model_student, 't2u_model')}")
if hasattr(model_student, 't2u_model'):
    print(f"T2U model type: {type(model_student.t2u_model)}")
    print(f"T2U has model attr: {hasattr(model_student.t2u_model, 'model')}")
    if hasattr(model_student.t2u_model, 'model'):
        print(f"T2U.model has encoder: {hasattr(model_student.t2u_model.model, 'encoder')}")
        print(f"T2U.model has decoder: {hasattr(model_student.t2u_model.model, 'decoder')}")

print(f"\nTeacher has t2u_model: {hasattr(model_teacher, 't2u_model')}")

# Check speech encoder
print(f"\nStudent has speech_encoder: {hasattr(model_student, 'speech_encoder')}")
print(f"Teacher has speech_encoder: {hasattr(model_teacher, 'speech_encoder')}")


# ── DEBUG 2: Test single sample processing ───────────────────────────────────
print("\n" + "="*80)
print("DEBUG 2: Single Sample Processing")
print("="*80)

# Get a test sample
test_sample = ft_samples[0]
print(f"\nTest sample keys: {test_sample.keys()}")
print(f"Sample ID: {test_sample.get('id', 'N/A')}")
print(f"Source lang: {test_sample.get('src_lang', 'N/A')}")
print(f"Target lang: {test_sample.get('tgt_lang', 'N/A')}")
print(f"Audio shape: {test_sample['wav'].shape if hasattr(test_sample['wav'], 'shape') else len(test_sample['wav'])}")
print(f"Reference text: {test_sample.get('ref', 'N/A')[:50]}...")


# ── DEBUG 3: Test processor input preparation ────────────────────────────────
print("\n" + "="*80)
print("DEBUG 3: Processor Input Preparation")
print("="*80)

try:
    inputs = processor(audio=test_sample['wav'], sampling_rate=16000, return_tensors='pt')
    print(f"\nProcessor output keys: {inputs.keys()}")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        else:
            print(f"  {k}: type={type(v)}")
    
    # Move to device
    inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
    print(f"\nInputs moved to device: {device}")
    
except Exception as e:
    print(f"\n❌ ERROR in processor: {e}")
    import traceback
    traceback.print_exc()


# ── DEBUG 4: Test teacher generation ──────────────────────────────────────────
print("\n" + "="*80)
print("DEBUG 4: Teacher Model Generation")
print("="*80)

try:
    model_teacher.eval()
    with torch.no_grad():
        teacher_out = model_teacher.generate(
            **inputs_gpu, 
            tgt_lang=test_sample['tgt_lang'],
            return_intermediate_token_ids=True
        )
    
    print(f"\nTeacher output type: {type(teacher_out)}")
    print(f"Teacher output attributes: {dir(teacher_out)}")
    
    # Check for unit_sequences
    if hasattr(teacher_out, 'unit_sequences'):
        teacher_units = teacher_out.unit_sequences
        print(f"\n✓ teacher_out.unit_sequences exists")
        print(f"  Type: {type(teacher_units)}")
        if teacher_units is not None:
            print(f"  Shape: {teacher_units.shape}")
            print(f"  Dtype: {teacher_units.dtype}")
            print(f"  Device: {teacher_units.device}")
            print(f"  Sample values: {teacher_units[0, :10] if teacher_units.numel() > 0 else 'empty'}")
        else:
            print(f"  ❌ unit_sequences is None!")
    else:
        print(f"\n❌ teacher_out does NOT have 'unit_sequences' attribute")
        print(f"Available attributes: {[a for a in dir(teacher_out) if not a.startswith('_')]}")
    
    # Check for sequences (text tokens)
    if hasattr(teacher_out, 'sequences'):
        print(f"\n✓ teacher_out.sequences exists")
        print(f"  Shape: {teacher_out.sequences.shape}")
        print(f"  Sample: {teacher_out.sequences[0, :10]}")
    
    # Check for waveform
    if hasattr(teacher_out, 'waveform'):
        print(f"\n✓ teacher_out.waveform exists")
        if teacher_out.waveform is not None:
            print(f"  Shape: {teacher_out.waveform.shape}")
        else:
            print(f"  waveform is None")
    
except Exception as e:
    print(f"\n❌ ERROR in teacher generation: {e}")
    import traceback
    traceback.print_exc()


# ── DEBUG 5: Test student speech encoder ──────────────────────────────────────
print("\n" + "="*80)
print("DEBUG 5: Student Speech Encoder Forward Pass")
print("="*80)

try:
    model_student.eval()
    with torch.no_grad():
        enc_out = model_student.speech_encoder(**inputs_gpu)
    
    print(f"\nSpeech encoder output type: {type(enc_out)}")
    
    if hasattr(enc_out, 'last_hidden_state'):
        print(f"✓ enc_out.last_hidden_state exists")
        print(f"  Shape: {enc_out.last_hidden_state.shape}")
        print(f"  Dtype: {enc_out.last_hidden_state.dtype}")
        print(f"  Device: {enc_out.last_hidden_state.device}")
    else:
        print(f"❌ enc_out does NOT have 'last_hidden_state'")
        print(f"Available attributes: {[a for a in dir(enc_out) if not a.startswith('_')]}")
    
except Exception as e:
    print(f"\n❌ ERROR in speech encoder: {e}")
    import traceback
    traceback.print_exc()


# ── DEBUG 6: Test T2U forward pass with dummy labels ──────────────────────────
print("\n" + "="*80)
print("DEBUG 6: T2U Model Forward Pass")
print("="*80)

try:
    # Get encoder output
    with torch.no_grad():
        enc_out = model_student.speech_encoder(**inputs_gpu).last_hidden_state
    
    print(f"\nEncoder output shape: {enc_out.shape}")
    
    # Try T2U forward with None labels first
    print("\nTest 1: T2U forward with labels=None")
    try:
        t2u_out_none = model_student.t2u_model(
            inputs_embeds=enc_out,
            labels=None,
            tgt_lang=test_sample['tgt_lang']
        )
        print(f"✓ T2U forward with None labels succeeded")
        print(f"  Output type: {type(t2u_out_none)}")
        print(f"  Has loss: {hasattr(t2u_out_none, 'loss')}")
        if hasattr(t2u_out_none, 'loss'):
            print(f"  Loss value: {t2u_out_none.loss}")
    except Exception as e:
        print(f"❌ T2U forward with None labels failed: {e}")
    
    # Try with dummy labels
    print("\nTest 2: T2U forward with dummy labels")
    try:
        # Create dummy unit labels (typical T2U vocab size is ~10000)
        dummy_labels = torch.randint(0, 1000, (1, 50), device=device)
        print(f"  Dummy labels shape: {dummy_labels.shape}")
        
        t2u_out_dummy = model_student.t2u_model(
            inputs_embeds=enc_out,
            labels=dummy_labels,
            tgt_lang=test_sample['tgt_lang']
        )
        print(f"✓ T2U forward with dummy labels succeeded")
        print(f"  Has loss: {hasattr(t2u_out_dummy, 'loss')}")
        if hasattr(t2u_out_dummy, 'loss'):
            print(f"  Loss value: {t2u_out_dummy.loss}")
            print(f"  Loss is None: {t2u_out_dummy.loss is None}")
    except Exception as e:
        print(f"❌ T2U forward with dummy labels failed: {e}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"\n❌ ERROR in T2U testing: {e}")
    import traceback
    traceback.print_exc()


# ── DEBUG 7: Inspect T2U model signature ──────────────────────────────────────
print("\n" + "="*80)
print("DEBUG 7: T2U Model Method Signatures")
print("="*80)

import inspect

if hasattr(model_student, 't2u_model'):
    t2u = model_student.t2u_model
    
    # Check forward method
    if hasattr(t2u, 'forward'):
        sig = inspect.signature(t2u.forward)
        print(f"\nT2U forward signature:")
        print(f"  {sig}")
        print(f"\nParameters:")
        for param_name, param in sig.parameters.items():
            print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'no annotation'}")
    
    # Check __call__ method
    if hasattr(t2u, '__call__'):
        print(f"\nT2U is callable: {callable(t2u)}")
    
    # Check model class
    print(f"\nT2U model class: {t2u.__class__.__name__}")
    print(f"T2U model module: {t2u.__class__.__module__}")


# ── DEBUG 8: Check if teacher actually produces unit_sequences ───────────────
print("\n" + "="*80)
print("DEBUG 8: Teacher Model Output Investigation")
print("="*80)

try:
    # Try different generation parameters
    print("\nTest 1: Standard generation")
    with torch.no_grad():
        out1 = model_teacher.generate(**inputs_gpu, tgt_lang=test_sample['tgt_lang'])
    print(f"  Output type: {type(out1)}")
    print(f"  Has unit_sequences: {hasattr(out1, 'unit_sequences')}")
    
    print("\nTest 2: Generation with return_intermediate_token_ids=True")
    with torch.no_grad():
        out2 = model_teacher.generate(
            **inputs_gpu, 
            tgt_lang=test_sample['tgt_lang'],
            return_intermediate_token_ids=True
        )
    print(f"  Output type: {type(out2)}")
    print(f"  Has unit_sequences: {hasattr(out2, 'unit_sequences')}")
    
    print("\nTest 3: Check generation_config")
    if hasattr(model_teacher, 'generation_config'):
        gen_cfg = model_teacher.generation_config
        print(f"  Generation config: {gen_cfg}")
        print(f"  Config attributes: {[a for a in dir(gen_cfg) if not a.startswith('_')][:20]}")
    
    print("\nTest 4: Direct model forward pass")
    with torch.no_grad():
        # Try calling model directly
        out3 = model_teacher(**inputs_gpu, tgt_lang=test_sample['tgt_lang'])
    print(f"  Output type: {type(out3)}")
    print(f"  Output attributes: {[a for a in dir(out3) if not a.startswith('_')][:20]}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
print("\nSummary of findings will help identify the root cause.")
print("Look for:")
print("  1. Whether teacher_out.unit_sequences exists and is not None")
print("  2. Whether T2U forward accepts the labels format")
print("  3. Whether the loss computation is failing")
