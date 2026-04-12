# cell 79
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 7 — Recovery Fine-tuning with DoRA (ICML 2024 Oral)                 ║
# ║  Model: model_p6 (1.058B pruned SeamlessM4T v2)                             ║
# ║                                                                              ║
# ║  WHY DoRA over plain LoRA / QLoRA?                                          ║
# ║  ─────────────────────────────────────────────────────────────────────────  ║
# ║  • LoRA (Hu et al. 2021): adds low-rank ΔW = B·A to frozen weights.        ║
# ║    Simple, fast, ~1-3% trainable params. But has a systematic accuracy gap  ║
# ║    vs full fine-tuning, especially problematic for a pruned model that has  ║
# ║    already lost capacity and needs to regain quality.                        ║
# ║                                                                              ║
# ║  • QLoRA (Dettmers et al. 2023): loads base model in 4-bit NF4, trains     ║
# ║    LoRA adapters in bf16. Great for memory-constrained single-GPU but       ║
# ║    dequantization overhead + accuracy gap make it suboptimal here since      ║
# ║    model_p6 is only 1B and already fits in a Kaggle T4/P100 in bf16.        ║
# ║                                                                              ║
# ║  • DoRA (Liu et al. ICML 2024 Oral): decomposes W into magnitude ‖W‖ and  ║
# ║    direction W/‖W‖. LoRA updates direction; a learnable scalar updates      ║
# ║    magnitude. This is the key insight: LoRA can only rotate OR scale a      ║
# ║    weight matrix together, whereas DoRA decouples them — matching the        ║
# ║    richer update pattern of full fine-tuning. DoRA consistently outperforms ║
# ║    LoRA on LLM, VLM, and VL-BART (multimodal) tasks.  Zero additional      ║
# ║    inference cost — adapters merge back into base weights.                  ║
# ║    use_dora=True is a single flag in HuggingFace PEFT LoraConfig.           ║
# ║                                                                              ║
# ║  TARGET MODULES — SeamlessM4T v2 anatomy:                                  ║
# ║  ─────────────────────────────────────────────────────────────────────────  ║
# ║  Speech encoder (Conformer w2v-BERT 2.0): acoustic features — mostly        ║
# ║  fixed; we do NOT target it. Adapting it risks corrupting speech            ║
# ║  representations that took 4.5M hours to pre-train.                         ║
# ║                                                                              ║
# ║  Text decoder (24→16 layers after Phase 3): directly responsible for        ║
# ║  translation quality → PRIMARY target for DoRA. Projections:                ║
# ║    q_proj, k_proj, v_proj, out_proj  (self-attn + cross-attn)               ║
# ║    fc1, fc2  (FFN, optional, add if budget allows)                          ║
# ║                                                                              ║
# ║  T2U encoder+decoder (3+3 layers after Phase 6): converts text tokens to   ║
# ║  discrete acoustic units — secondary target. After pruning it may have      ║
# ║  lost unit-prediction accuracy. Same projections as above.                   ║
# ║                                                                              ║
# ║  LOSS: S2TT cross-entropy (text decoder output only).                       ║
# ║  Rationale: ChrF and BLEU measure text quality. S2TT loss is 3-5× faster   ║
# ║  per step than full S2ST (no T2U forward or vocoder). We recover the text   ║
# ║  decoder first; T2U quality largely follows.                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Phase 7 Cell 1: Install & imports ────────────────────────────────────────
subprocess.run(['pip', 'install', '-q', 'peft>=0.10.0'], check=True)

import gc, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset as ld

print('PEFT + torch ready.')
print(f'GPU: {torch.cuda.get_device_name(0)}  '
      f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')