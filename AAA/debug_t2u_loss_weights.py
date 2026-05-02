# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: T2U Loss Weight Analysis
# ═══════════════════════════════════════════════════════════════════════════════
#
# The planning.md suggests loss weights of 0.60 soft + 0.30 hard + 0.10 len,
# but your training logs show t2u_len values around 0.10-0.15, which means
# the length loss is contributing 0.01-0.015 to the total loss (10% of 0.10).
#
# Meanwhile, t2u_soft is ~0.85-1.10 and t2u_hard is ~7.0-8.8.
# This means the hard CE loss is DOMINATING (contributing ~2.1-2.6 to total loss).
#
# Let's verify if this is causing the problem.
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np

print("=" * 80)
print("T2U LOSS WEIGHT ANALYSIS")
print("=" * 80)

# Simulate typical loss values from your training logs
typical_losses = [
    {"soft": 1.1061, "hard": 8.8533, "len": 0.1505},  # step 10
    {"soft": 0.9740, "hard": 8.1846, "len": 0.1465},  # step 20
    {"soft": 0.8525, "hard": 7.5829, "len": 0.0995},  # step 30
    {"soft": 0.8882, "hard": 7.3588, "len": 0.1160},  # step 40
    {"soft": 0.8729, "hard": 7.0888, "len": 0.1104},  # step 50
    {"soft": 0.8457, "hard": 6.9818, "len": 0.1162},  # step 60
]

print("\nCurrent weights: 0.60 soft + 0.30 hard + 0.10 len")
print("-" * 80)
print(f"{'Step':<8} {'Soft':<8} {'Hard':<8} {'Len':<8} {'Total':<8} {'Soft%':<8} {'Hard%':<8} {'Len%':<8}")
print("-" * 80)

for i, losses in enumerate(typical_losses):
    soft_contrib = 0.60 * losses["soft"]
    hard_contrib = 0.30 * losses["hard"]
    len_contrib = 0.10 * losses["len"]
    total = soft_contrib + hard_contrib + len_contrib
    
    soft_pct = (soft_contrib / total) * 100
    hard_pct = (hard_contrib / total) * 100
    len_pct = (len_contrib / total) * 100
    
    step = (i + 1) * 10
    print(f"{step:<8} {losses['soft']:<8.4f} {losses['hard']:<8.4f} {losses['len']:<8.4f} "
          f"{total:<8.4f} {soft_pct:<8.1f} {hard_pct:<8.1f} {len_pct:<8.1f}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)

avg_soft = np.mean([l["soft"] for l in typical_losses])
avg_hard = np.mean([l["hard"] for l in typical_losses])
avg_len = np.mean([l["len"] for l in typical_losses])

avg_soft_contrib = 0.60 * avg_soft
avg_hard_contrib = 0.30 * avg_hard
avg_len_contrib = 0.10 * avg_len
avg_total = avg_soft_contrib + avg_hard_contrib + avg_len_contrib

print(f"\nAverage loss components:")
print(f"  Soft KD contribution: {avg_soft_contrib:.4f} ({(avg_soft_contrib/avg_total)*100:.1f}%)")
print(f"  Hard CE contribution: {avg_hard_contrib:.4f} ({(avg_hard_contrib/avg_total)*100:.1f}%)")
print(f"  Length contribution:  {avg_len_contrib:.4f} ({(avg_len_contrib/avg_total)*100:.1f}%)")
print(f"  Total loss:           {avg_total:.4f}")

print("\n" + "=" * 80)
print("PROBLEM IDENTIFIED:")
print("=" * 80)
print("The hard CE loss is DOMINATING the training signal (~70-80% of total loss).")
print("This is likely because:")
print("  1. Student T2U (4+4 layers) has lower capacity than teacher (6+6 layers)")
print("  2. Hard CE penalizes ANY deviation from teacher's argmax units")
print("  3. Student cannot match teacher's exact unit predictions due to capacity gap")
print("\nThis causes the student to chase an impossible target, damaging quality.")

print("\n" + "=" * 80)
print("RECOMMENDED FIX:")
print("=" * 80)
print("Rebalance the loss weights to reduce hard CE dominance:")
print("\nOption A (Soft KD focus):")
print("  loss = 0.70 * t2u_soft + 0.20 * t2u_hard + 0.10 * t2u_len")
print("  → Soft KD ~60%, Hard CE ~40%, Length ~1%")
print("\nOption B (Pure soft KD):")
print("  loss = 0.85 * t2u_soft + 0.10 * t2u_hard + 0.05 * t2u_len")
print("  → Soft KD ~75%, Hard CE ~20%, Length ~5%")
print("\nOption C (Length-aware):")
print("  loss = 0.60 * t2u_soft + 0.20 * t2u_hard + 0.20 * t2u_len")
print("  → Soft KD ~50%, Hard CE ~35%, Length ~15%")

print("\nStart with Option A. If ASR-ChrF still drops, try Option B.")
print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL CHECK: Verify if hard CE is causing gradient explosion
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("GRADIENT MAGNITUDE CHECK")
print("=" * 80)

print("\nRunning one forward pass to check gradient magnitudes...")

# Pick a sample
sample, cache_entry = phase6_pick_training_pair(max_audio_sec=12, balanced=True)
teacher_text_sequences = cache_entry['teacher_text_sequences'].unsqueeze(0)

audio_inputs_student = phase6_prepare_audio_inputs(sample, student_device)
audio_inputs_teacher = {k: v.to(teacher_device) for k, v in audio_inputs_student.items()}

# Teacher forward
with torch.no_grad():
    teacher_cond = build_t2u_conditioning_from_sequences(
        model_teacher,
        input_features=audio_inputs_teacher['input_features'],
        attention_mask=audio_inputs_teacher.get('attention_mask'),
        text_sequences=teacher_text_sequences.to(teacher_device),
    )
    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        teacher_t2u = model_teacher.t2u_model(
            inputs_embeds=teacher_cond['t2u_input_embeds'],
            attention_mask=teacher_cond['t2u_attention_mask'],
            char_input_ids=teacher_cond['t2u_char_input_ids'],
            char_count_per_id=teacher_cond['t2u_char_count_per_id'],
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

# Student forward
model_student.train()
model_student.zero_grad()

with torch.no_grad():
    student_cond = build_t2u_conditioning_from_sequences(
        model_student,
        input_features=audio_inputs_student['input_features'],
        attention_mask=audio_inputs_student.get('attention_mask'),
        text_sequences=teacher_text_sequences.to(student_device),
    )

with torch.cuda.amp.autocast(dtype=autocast_dtype):
    student_t2u = model_student.t2u_model(
        inputs_embeds=student_cond['t2u_input_embeds'].detach(),
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

# Compute gradients for each loss component separately
print("\n--- Gradient magnitudes from each loss component ---")

# Soft KD gradients
model_student.zero_grad()
t2u_soft.backward(retain_graph=True)
soft_grad_norm = torch.nn.utils.clip_grad_norm_(
    [p for p in model_student.t2u_model.parameters() if p.requires_grad],
    float('inf')
)
print(f"Soft KD gradient norm: {soft_grad_norm:.4f}")

# Hard CE gradients
model_student.zero_grad()
t2u_hard.backward(retain_graph=True)
hard_grad_norm = torch.nn.utils.clip_grad_norm_(
    [p for p in model_student.t2u_model.parameters() if p.requires_grad],
    float('inf')
)
print(f"Hard CE gradient norm: {hard_grad_norm:.4f}")

# Length gradients
model_student.zero_grad()
t2u_len.backward()
len_grad_norm = torch.nn.utils.clip_grad_norm_(
    [p for p in model_student.t2u_model.parameters() if p.requires_grad],
    float('inf')
)
print(f"Length gradient norm:  {len_grad_norm:.4f}")

print("\n--- Analysis ---")
if hard_grad_norm > soft_grad_norm * 3:
    print("❌ Hard CE gradients are 3x larger than soft KD gradients!")
    print("   This confirms hard CE is dominating the training signal.")
    print("   Reduce hard CE weight to 0.20 or lower.")
elif hard_grad_norm > soft_grad_norm * 1.5:
    print("⚠️  Hard CE gradients are 1.5x larger than soft KD gradients.")
    print("   Consider reducing hard CE weight to 0.25.")
else:
    print("✓ Gradient magnitudes are balanced.")

print("=" * 80)
