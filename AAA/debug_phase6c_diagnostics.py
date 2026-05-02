# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6C DIAGNOSTIC CELLS — Insert after Phase 6C setup, before training
# ═══════════════════════════════════════════════════════════════════════════════

# ── DIAGNOSTIC 1: Verify Phase 6A cache vocab consistency ─────────────────────
print("=" * 80)
print("DIAGNOSTIC 1: Phase 6A Cache Vocab Consistency")
print("=" * 80)

# Load a few cache entries
test_keys = list(phase6_cache_index.keys())[:5]
print(f"\nTesting {len(test_keys)} cache entries...\n")

for key in test_keys:
    entry = phase6_get_cache_entry(key)
    teacher_seq = entry['teacher_text_sequences']
    
    # Check if IDs are in student vocab range
    max_id = teacher_seq.max().item()
    student_vocab_size = model_student.text_decoder.get_base_model().embed_tokens.num_embeddings
    
    print(f"Sample: {key[:50]}")
    print(f"  teacher_text_sequences shape: {teacher_seq.shape}")
    print(f"  max ID in sequence: {max_id}")
    print(f"  student vocab size: {student_vocab_size}")
    
    if max_id >= student_vocab_size:
        print(f"  ❌ VOCAB MISMATCH: max_id {max_id} >= student_vocab {student_vocab_size}")
        print(f"     This means cache contains OLD 256K vocab IDs!")
    else:
        print(f"  ✓ IDs are within student vocab range")
    
    # Try to decode with processor (will fail if IDs are wrong)
    try:
        # Processor expects OLD vocab IDs
        decoded = processor.batch_decode(teacher_seq.unsqueeze(0), skip_special_tokens=True)[0]
        print(f"  processor decode: '{decoded[:60]}'")
    except Exception as e:
        print(f"  ❌ processor decode FAILED: {e}")
    
    print()

print("\n" + "=" * 80)
print("CONCLUSION:")
print("If max_id >= student_vocab_size, then Phase 6A cache was built with")
print("OLD vocab IDs and needs remapping before use in Phase 6C.")
print("=" * 80)


# ── DIAGNOSTIC 2: Test T2U conditioning path with frozen text_decoder ─────────
print("\n" + "=" * 80)
print("DIAGNOSTIC 2: T2U Conditioning Path — Dropout Corruption Test")
print("=" * 80)

# Pick a sample
sample, cache_entry = phase6_pick_training_pair(max_audio_sec=12, balanced=True)
teacher_text_sequences = cache_entry['teacher_text_sequences'].unsqueeze(0)

audio_inputs = phase6_prepare_audio_inputs(sample, student_device)

print(f"\nSample: {sample['id']}")
print(f"teacher_text_sequences shape: {teacher_text_sequences.shape}")
print(f"teacher_text_sequences (first 10 IDs): {teacher_text_sequences[0, :10].tolist()}")

# Test 1: Run conditioning in TRAIN mode (current broken behavior)
model_student.train()
print("\n--- Test 1: Conditioning with model_student.train() ---")
with torch.no_grad():
    cond_train = build_t2u_conditioning_from_sequences(
        model_student,
        input_features=audio_inputs['input_features'],
        attention_mask=audio_inputs.get('attention_mask'),
        text_sequences=teacher_text_sequences.to(student_device),
    )
    t2u_embeds_train = cond_train['t2u_input_embeds']
    print(f"t2u_input_embeds shape: {t2u_embeds_train.shape}")
    print(f"t2u_input_embeds mean: {t2u_embeds_train.mean().item():.6f}")
    print(f"t2u_input_embeds std:  {t2u_embeds_train.std().item():.6f}")

# Test 2: Run conditioning in EVAL mode (correct behavior)
model_student.eval()
print("\n--- Test 2: Conditioning with model_student.eval() ---")
with torch.no_grad():
    cond_eval = build_t2u_conditioning_from_sequences(
        model_student,
        input_features=audio_inputs['input_features'],
        attention_mask=audio_inputs.get('attention_mask'),
        text_sequences=teacher_text_sequences.to(student_device),
    )
    t2u_embeds_eval = cond_eval['t2u_input_embeds']
    print(f"t2u_input_embeds shape: {t2u_embeds_eval.shape}")
    print(f"t2u_input_embeds mean: {t2u_embeds_eval.mean().item():.6f}")
    print(f"t2u_input_embeds std:  {t2u_embeds_eval.std().item():.6f}")

# Compare
diff = (t2u_embeds_train - t2u_embeds_eval).abs().mean().item()
print(f"\n--- Comparison ---")
print(f"Absolute difference between train/eval embeddings: {diff:.6f}")

if diff > 0.01:
    print("❌ DROPOUT CORRUPTION DETECTED!")
    print("   The frozen text_decoder is in train mode during conditioning,")
    print("   causing dropout to corrupt t2u_input_embeds even inside no_grad().")
    print("   FIX: Set text_decoder.eval() before running it in conditioning.")
else:
    print("✓ No significant difference — dropout is not the issue.")

model_student.train()  # restore training mode

print("=" * 80)


# ── DIAGNOSTIC 3: Compare teacher vs student T2U output lengths ───────────────
print("\n" + "=" * 80)
print("DIAGNOSTIC 3: Teacher vs Student T2U Output Length Mismatch")
print("=" * 80)

# Use same sample from above
audio_inputs_teacher = {k: v.to(teacher_device) for k, v in audio_inputs.items()}

print("\n--- Teacher T2U Forward ---")
with torch.no_grad():
    teacher_cond = build_t2u_conditioning_from_sequences(
        model_teacher,
        input_features=audio_inputs_teacher['input_features'],
        attention_mask=audio_inputs_teacher.get('attention_mask'),
        text_sequences=teacher_text_sequences.to(teacher_device),
    )
    teacher_t2u = model_teacher.t2u_model(
        inputs_embeds=teacher_cond['t2u_input_embeds'],
        attention_mask=teacher_cond['t2u_attention_mask'],
        char_input_ids=teacher_cond['t2u_char_input_ids'],
        char_count_per_id=teacher_cond['t2u_char_count_per_id'],
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    teacher_len = teacher_t2u.padding_mask.sum(1).item()
    print(f"Teacher T2U output length: {teacher_len}")
    print(f"Teacher T2U logits shape: {teacher_t2u.last_hidden_state.shape}")

print("\n--- Student T2U Forward ---")
model_student.eval()
with torch.no_grad():
    student_cond = build_t2u_conditioning_from_sequences(
        model_student,
        input_features=audio_inputs['input_features'],
        attention_mask=audio_inputs.get('attention_mask'),
        text_sequences=teacher_text_sequences.to(student_device),
    )
    student_t2u = model_student.t2u_model(
        inputs_embeds=student_cond['t2u_input_embeds'],
        attention_mask=student_cond['t2u_attention_mask'],
        char_input_ids=student_cond['t2u_char_input_ids'],
        char_count_per_id=student_cond['t2u_char_count_per_id'],
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    student_len = student_t2u.padding_mask.sum(1).item()
    print(f"Student T2U output length: {student_len}")
    print(f"Student T2U logits shape: {student_t2u.last_hidden_state.shape}")

print(f"\n--- Length Comparison ---")
print(f"Teacher length: {teacher_len}")
print(f"Student length: {student_len}")
print(f"Difference: {abs(teacher_len - student_len)}")
print(f"Overlap length: {min(teacher_len, student_len)}")

if abs(teacher_len - student_len) > teacher_len * 0.3:
    print("⚠️  Large length mismatch (>30%)!")
    print("   This is expected due to pruned T2U decoder (4 layers vs 6).")
    print("   The overlap-based KD loss should handle this, but verify it's working.")
else:
    print("✓ Length mismatch is reasonable.")

model_student.train()

print("=" * 80)


# ── DIAGNOSTIC 4: Verify overlap-based KD loss computation ────────────────────
print("\n" + "=" * 80)
print("DIAGNOSTIC 4: Overlap-Based KD Loss Sanity Check")
print("=" * 80)

# Use outputs from Diagnostic 3
teacher_t2u.last_hidden_state = teacher_t2u.last_hidden_state.to(student_device)
teacher_t2u.padding_mask = teacher_t2u.padding_mask.to(student_device)

print("\n--- Computing KD Losses ---")
try:
    t2u_soft, t2u_hard, t2u_len = t2u_overlap_losses(student_t2u, teacher_t2u)
    
    print(f"t2u_soft (KL divergence): {t2u_soft.item():.4f}")
    print(f"t2u_hard (hard CE):       {t2u_hard.item():.4f}")
    print(f"t2u_len (length loss):    {t2u_len.item():.4f}")
    
    loss = 0.60 * t2u_soft + 0.30 * t2u_hard + 0.10 * t2u_len
    print(f"\nCombined loss: {loss.item():.4f}")
    
    # Sanity checks
    if t2u_soft.item() > 100 or t2u_hard.item() > 100:
        print("❌ LOSS EXPLOSION: KD losses are abnormally high!")
        print("   This suggests the student T2U is producing garbage logits.")
    elif t2u_soft.item() < 0.01 and t2u_hard.item() < 0.01:
        print("⚠️  Losses are suspiciously low — student might be copying teacher exactly.")
    else:
        print("✓ Loss magnitudes look reasonable.")
    
except Exception as e:
    print(f"❌ KD loss computation FAILED: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)


# ── DIAGNOSTIC 5: Check if student T2U weights are actually updating ──────────
print("\n" + "=" * 80)
print("DIAGNOSTIC 5: T2U Weight Update Verification")
print("=" * 80)

# Snapshot current weights
t2u_weight_snapshot = {}
for name, param in model_student.t2u_model.named_parameters():
    if param.requires_grad:
        t2u_weight_snapshot[name] = param.data.clone()

print(f"Captured {len(t2u_weight_snapshot)} trainable T2U parameters")

# Run one training step
print("\n--- Running 1 training step ---")
model_student.train()
optimizer_test = torch.optim.AdamW(
    [p for p in model_student.t2u_model.parameters() if p.requires_grad],
    lr=8e-5
)

sample, cache_entry = phase6_pick_training_pair(max_audio_sec=12, balanced=True)
teacher_text_sequences = cache_entry['teacher_text_sequences'].unsqueeze(0)
audio_inputs_student = phase6_prepare_audio_inputs(sample, student_device)
audio_inputs_teacher = {k: v.to(teacher_device) for k, v in audio_inputs_student.items()}

with torch.no_grad():
    teacher_cond = build_t2u_conditioning_from_sequences(
        model_teacher,
        input_features=audio_inputs_teacher['input_features'],
        attention_mask=audio_inputs_teacher.get('attention_mask'),
        text_sequences=teacher_text_sequences.to(teacher_device),
    )
    teacher_t2u = model_teacher.t2u_model(
        inputs_embeds=teacher_cond['t2u_input_embeds'],
        attention_mask=teacher_cond['t2u_attention_mask'],
        char_input_ids=teacher_cond['t2u_char_input_ids'],
        char_count_per_id=teacher_cond['t2u_char_count_per_id'],
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

student_cond = build_t2u_conditioning_from_sequences(
    model_student,
    input_features=audio_inputs_student['input_features'],
    attention_mask=audio_inputs_student.get('attention_mask'),
    text_sequences=teacher_text_sequences.to(student_device),
)
student_t2u = model_student.t2u_model(
    inputs_embeds=student_cond['t2u_input_embeds'],
    attention_mask=student_cond['t2u_attention_mask'],
    char_input_ids=student_cond['t2u_char_input_ids'],
    char_count_per_id=student_cond['t2u_char_count_per_id'],
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
)

teacher_t2u.last_hidden_state = teacher_t2u.last_hidden_state.to(student_device)
teacher_t2u.padding_mask = teacher_t2u.padding_mask.to(student_device)

t2u_soft, t2u_hard, t2u_len = t2u_overlap_losses(student_t2u, teacher_t2u)
loss = 0.60 * t2u_soft + 0.30 * t2u_hard + 0.10 * t2u_len

optimizer_test.zero_grad()
loss.backward()
optimizer_test.step()

print(f"Loss: {loss.item():.4f}")

# Check if weights changed
print("\n--- Checking weight updates ---")
n_updated = 0
n_unchanged = 0
max_change = 0.0

for name, param in model_student.t2u_model.named_parameters():
    if param.requires_grad and name in t2u_weight_snapshot:
        old_weight = t2u_weight_snapshot[name]
        change = (param.data - old_weight).abs().max().item()
        max_change = max(max_change, change)
        
        if change > 1e-8:
            n_updated += 1
        else:
            n_unchanged += 1

print(f"Parameters updated: {n_updated}")
print(f"Parameters unchanged: {n_unchanged}")
print(f"Max weight change: {max_change:.2e}")

if n_updated == 0:
    print("❌ NO WEIGHTS UPDATED! Gradients are not flowing to T2U.")
elif max_change < 1e-6:
    print("⚠️  Weight changes are extremely small — learning rate might be too low.")
else:
    print("✓ Weights are updating normally.")

print("=" * 80)


# ── DIAGNOSTIC 6: Verify teacher cache text strings are correct ───────────────
print("\n" + "=" * 80)
print("DIAGNOSTIC 6: Phase 6A Cache Text String Verification")
print("=" * 80)

# This checks if the cache poisoning fix from planning.md was applied
print("\nChecking if teacher_text_str matches teacher_text_sequences...\n")

_new_to_old_tensor = model_student._vocab_remap_to_old.to('cpu')

for key in test_keys[:3]:
    entry = phase6_get_cache_entry(key)
    seq = entry['teacher_text_sequences']
    stored_str = entry['teacher_text_str']
    
    # Correct decode: remap NEW IDs -> OLD IDs, then use processor
    old_ids = _new_to_old_tensor[seq]
    correct_str = processor.batch_decode(old_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
    
    # Poisoned decode: feed NEW IDs directly to processor (buggy)
    try:
        poisoned_str = processor.batch_decode(seq.unsqueeze(0), skip_special_tokens=True)[0].strip()
    except:
        poisoned_str = "[DECODE FAILED]"
    
    print(f"Sample: {key[:50]}")
    print(f"  stored:   '{stored_str[:60]}'")
    print(f"  correct:  '{correct_str[:60]}'")
    print(f"  poisoned: '{poisoned_str[:60]}'")
    
    if stored_str == poisoned_str and stored_str != correct_str:
        print(f"  ❌ CACHE POISONED: stored string matches buggy decode!")
    elif stored_str == correct_str:
        print(f"  ✓ Cache string is correct")
    else:
        print(f"  ⚠️  Ambiguous — strings don't match either pattern")
    print()

print("=" * 80)
print("If cache is poisoned, teacher_text_str is wrong but teacher_text_sequences")
print("and teacher_unit_sequences are still correct. You can regenerate text_str")
print("without re-running teacher inference.")
print("=" * 80)
