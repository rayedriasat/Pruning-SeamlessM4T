# cell 83
# ── Phase 7 Cell 3: Inject DoRA adapters ─────────────────────────────────────
#
# DoRA hyperparameters:
#   r=16          → rank; balances expressivity vs parameters. 8 is too low
#                   for recovery (the pruned model lost capacity); 32 uses too
#                   much memory. 16 is the sweet spot.
#   lora_alpha=32 → effective scale = alpha/r = 2.0. Standard 2× rule.
#   use_rslora    → rank-stabilized scaling (alpha/√r). Use if r>16.
#   use_dora=True → activates Weight-Decomposed LoRA (ICML 2024 Oral).
#   target_modules → text_decoder + t2u_model projections only.
#   modules_to_save → embed_tokens and layer norms: these are low-param but
#                     critical for the pruned model to adapt vocabulary and
#                     activation distributions.

LORA_R     = 16
LORA_ALPHA = 32     # scale = alpha/r = 2.0
LORA_DROP  = 0.05

lora_cfg = LoraConfig(
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROP,
    bias           = 'none',
    use_dora       = True,          # ← DoRA (ICML 2024 Oral) instead of plain LoRA
    target_modules = targets,
    # task_type intentionally omitted — SeamlessM4T is encoder-decoder S2ST,
    # not a standard HF task type. PEFT handles it in "feature extraction" mode.
)

model_p7 = get_peft_model(model_p6, lora_cfg)
model_p7.print_trainable_parameters()

# Move to single GPU (same as Phase 4)
model_p7 = _consolidate_to_single_gpu(model_p7)
model_p7.train()