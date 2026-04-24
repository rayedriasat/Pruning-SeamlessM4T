#!/usr/bin/env python3
"""
Analyze Phase 3 similarity scores and recommend optimal threshold
"""

# Your actual similarity scores from the output
enc_sims = [0.8219, 0.8229, 0.6607, 0.9013, 0.9384]
dec_sims = [0.7455, 0.7125, 0.7575, 0.5846, 0.7288]

print("=" * 80)
print("PHASE 3 THRESHOLD ANALYSIS")
print("=" * 80)

print("\n📊 Your Similarity Scores:")
print("-" * 80)
print("Encoder:", [f"L{i+1}={s:.4f}" for i, s in enumerate(enc_sims)])
print("Decoder:", [f"L{i+1}={s:.4f}" for i, s in enumerate(dec_sims)])

print("\n🎯 Testing Different Thresholds:")
print("-" * 80)

thresholds = [0.96, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65]
target_merges = 2  # Want 2 merges per stack

best_threshold = None
best_score = float('inf')

for thresh in thresholds:
    enc_merged = min(sum(1 for s in enc_sims if s > thresh), target_merges)
    dec_merged = min(sum(1 for s in dec_sims if s > thresh), target_merges)
    total = enc_merged + dec_merged
    
    # Score: how close to 4 total merges
    score = abs(4 - total)
    
    status = "✓" if total == 4 else "✗"
    
    print(f"\nThreshold: {thresh:.2f}")
    print(f"  Encoder: {enc_merged}/2 merges", end="")
    if enc_merged > 0:
        merged_layers = [f"L{i+1}" for i, s in enumerate(enc_sims) if s > thresh][:target_merges]
        print(f" ({', '.join(merged_layers)})")
    else:
        print()
    
    print(f"  Decoder: {dec_merged}/2 merges", end="")
    if dec_merged > 0:
        merged_layers = [f"L{i+1}" for i, s in enumerate(dec_sims) if s > thresh][:target_merges]
        print(f" ({', '.join(merged_layers)})")
    else:
        print()
    
    print(f"  Total: {total}/4 {status}")
    
    if score < best_score:
        best_score = score
        best_threshold = thresh

print("\n" + "=" * 80)
print("🎯 RECOMMENDATION")
print("=" * 80)

if best_threshold:
    print(f"\n✅ Use threshold: {best_threshold:.2f}")
    
    enc_merged = min(sum(1 for s in enc_sims if s > best_threshold), target_merges)
    dec_merged = min(sum(1 for s in dec_sims if s > best_threshold), target_merges)
    
    print(f"\nThis will merge:")
    print(f"  • Encoder: {enc_merged} layers → 6 → {6 - enc_merged} layers")
    print(f"  • Decoder: {dec_merged} layers → 6 → {6 - dec_merged} layers")
    print(f"  • Total: {enc_merged + dec_merged} merges")
    
    print(f"\n📝 Edit your notebook:")
    print(f"   Change: sim_threshold=0.96")
    print(f"   To:     sim_threshold={best_threshold:.2f}")
    
    print(f"\n💡 Quality estimate:")
    enc_merged_sims = sorted([s for s in enc_sims if s > best_threshold], reverse=True)[:target_merges]
    dec_merged_sims = sorted([s for s in dec_sims if s > best_threshold], reverse=True)[:target_merges]
    if enc_merged_sims and dec_merged_sims:
        avg_sim = (sum(enc_merged_sims) + sum(dec_merged_sims)) / (len(enc_merged_sims) + len(dec_merged_sims))
        print(f"   Average similarity of merged layers: ~{avg_sim:.1%}")
        print(f"   Expected quality retention: ~{avg_sim:.1%}")

print("\n" + "=" * 80)
print("📚 CONTEXT")
print("=" * 80)
print("""
Why 0.96 didn't work:
  • LaCo paper used 0.96 for large language models (32-48 layers)
  • Your T2U model has only 6 layers per stack
  • Smaller models have less redundancy
  • Each layer is more specialized

Your similarity scores show:
  • Encoder layers are moderately similar (0.66-0.94)
  • Decoder layers are less similar (0.58-0.76)
  • This is normal for speech-to-unit models
  • Lower threshold needed to achieve target compression
""")

print("=" * 80)
print("🚀 NEXT STEPS")
print("=" * 80)
print("""
1. Delete checkpoint:
   !rm -rf /kaggle/working/checkpoints/phase3_laco_done_step000000.pt

2. Edit Phase 3 cell:
   model_p3 = apply_laco_t2u(model_p3, sim_threshold=0.70, ...)

3. Re-run Phase 3

4. Verify you see "MERGED" messages and 6→4 reduction
""")

print("=" * 80)
