# ══════════════════════════════════════════════════════════════════════════════
# DEBUG: Training Loop Issue - Detailed Step-by-Step
# ══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("TRAINING LOOP DEBUG - Step by Step")
print("="*80)

# Simulate one training step with detailed logging
sample = ft_samples[random.randint(0, len(ft_samples)-1)]

print(f"\n1. Sample loaded:")
print(f"   ID: {sample.get('id', 'N/A')}")
print(f"   Target lang: {sample['tgt_lang']}")
print(f"   Audio length: {len(sample['wav'])/16000:.2f}s")

# Step 1: Process input
print(f"\n2. Processing input...")
try:
    inputs = processor(audio=sample['wav'], sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"   ✓ Input processed, keys: {list(inputs.keys())}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    raise

# Step 2: Get teacher units
print(f"\n3. Getting teacher's unit sequence...")
try:
    with torch.no_grad():
        teacher_out = model_teacher.generate(
            **inputs, tgt_lang=sample['tgt_lang'],
            return_intermediate_token_ids=True)
        
    print(f"   Teacher output type: {type(teacher_out)}")
    print(f"   Has unit_sequences: {hasattr(teacher_out, 'unit_sequences')}")
    
    if hasattr(teacher_out, 'unit_sequences'):
        teacher_units = teacher_out.unit_sequences
        print(f"   teacher_units type: {type(teacher_units)}")
        print(f"   teacher_units is None: {teacher_units is None}")
        
        if teacher_units is not None:
            print(f"   ✓ teacher_units shape: {teacher_units.shape}")
            print(f"   ✓ teacher_units device: {teacher_units.device}")
            print(f"   ✓ teacher_units dtype: {teacher_units.dtype}")
            print(f"   ✓ Sample values: {teacher_units[0, :5]}")
        else:
            print(f"   ❌ teacher_units is None!")
            print(f"   This is the root cause of the error!")
            
            # Check what else is in teacher_out
            print(f"\n   Available in teacher_out:")
            for attr in dir(teacher_out):
                if not attr.startswith('_'):
                    val = getattr(teacher_out, attr)
                    if isinstance(val, torch.Tensor):
                        print(f"     - {attr}: Tensor {val.shape}")
                    elif val is not None and not callable(val):
                        print(f"     - {attr}: {type(val)}")
    else:
        print(f"   ❌ teacher_out has no 'unit_sequences' attribute!")
        print(f"   Available attributes: {[a for a in dir(teacher_out) if not a.startswith('_')]}")
        teacher_units = None
        
except Exception as e:
    print(f"   ❌ ERROR in teacher generation: {e}")
    import traceback
    traceback.print_exc()
    raise

# Step 3: Get student encoder output
print(f"\n4. Getting student encoder output...")
try:
    with torch.no_grad():
        enc_out = model_student.speech_encoder(**inputs).last_hidden_state
    print(f"   ✓ Encoder output shape: {enc_out.shape}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    raise

# Step 4: T2U forward pass
print(f"\n5. T2U forward pass...")
try:
    print(f"   Labels (teacher_units): {teacher_units}")
    
    if teacher_units is not None:
        print(f"   Moving teacher_units to device...")
        labels_gpu = teacher_units.to(device)
        print(f"   ✓ Labels on device: {labels_gpu.device}")
    else:
        print(f"   ⚠️  teacher_units is None, using None as labels")
        labels_gpu = None
    
    print(f"   Calling t2u_model.forward...")
    t2u_out = model_student.t2u_model(
        inputs_embeds=enc_out,
        labels=labels_gpu,
        tgt_lang=sample['tgt_lang']
    )
    
    print(f"   ✓ T2U forward succeeded")
    print(f"   Output type: {type(t2u_out)}")
    print(f"   Has loss: {hasattr(t2u_out, 'loss')}")
    
    if hasattr(t2u_out, 'loss'):
        loss = t2u_out.loss
        print(f"   Loss type: {type(loss)}")
        print(f"   Loss is None: {loss is None}")
        
        if loss is not None:
            print(f"   ✓ Loss value: {loss.item()}")
        else:
            print(f"   ❌ Loss is None!")
    else:
        print(f"   ❌ t2u_out has no 'loss' attribute")
        loss = torch.tensor(0.0).to(device)
        print(f"   Using fallback loss: {loss}")
        
except Exception as e:
    print(f"   ❌ ERROR in T2U forward: {e}")
    import traceback
    traceback.print_exc()
    
    # Try to identify which line causes the error
    print(f"\n   Detailed error analysis:")
    print(f"   - enc_out is None: {enc_out is None}")
    print(f"   - labels_gpu is None: {labels_gpu is None if 'labels_gpu' in locals() else 'not defined'}")
    print(f"   - model_student.t2u_model is None: {model_student.t2u_model is None}")
    raise

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)


# ── ALTERNATIVE: Check if we need to use a different approach ────────────────
print("\n" + "="*80)
print("ALTERNATIVE APPROACH: Check T2U Training Methods")
print("="*80)

print("\nOption 1: Check if model has a specific T2U training method")
if hasattr(model_student, 't2u_model'):
    t2u_methods = [m for m in dir(model_student.t2u_model) if not m.startswith('_')]
    print(f"T2U model methods: {t2u_methods[:30]}")

print("\nOption 2: Check SeamlessM4T documentation for T2U training")
print("The model might need:")
print("  - Text token IDs as input (not audio)")
print("  - Different forward signature")
print("  - Separate T2U training pipeline")

print("\nOption 3: Check if we should train on text→units instead of audio→units")
print("SeamlessM4T T2U model typically expects:")
print("  - Input: text decoder hidden states OR text token embeddings")
print("  - Output: unit sequences")
print("  - NOT direct audio→units")

print("\n" + "="*80)
