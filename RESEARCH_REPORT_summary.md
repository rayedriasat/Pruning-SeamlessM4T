Here's a summary of the report:

---

## Summary: Structured Compression of SeamlessM4T v2 Large (EN→BN Speech Translation)

**Goal:** Compress Meta's `seamless-m4t-v2-large` model (1,805 M parameters) for English→Bengali speech-to-speech translation, while running on a single consumer-grade GPU (Kaggle T4, ~16 GB VRAM).

### What They Did

The researchers applied a multi-phase compression pipeline:

1. **Vocabulary Pruning (Phase 1):** Trimmed the tokenizer vocabulary from 256,102 tokens down to 20,425 (only languages relevant to the task), saving ~241 M parameters.
2. **Text Encoder Removal (Phase 2):** Found this was a no-op — the encoder isn't used for speech-to-speech tasks at all.
3. **Depth Pruning (Phases 3, 4, 6):** Iteratively removed the least-important layers from the text decoder (10 of 24 layers removed), speech encoder (8 of 24 removed), and T2U model (2 encoder + 2 decoder layers removed), guided by ChrF quality scores.
4. **Width Pruning (Phase 5):** Attempted FLAP-based pruning but abandoned it after catastrophic quality collapse.
5. **DoRA Fine-tuning (Phase 7):** Used weight-decomposed low-rank adaptation on 1,449 training pairs to recover translation quality lost during pruning.
6. **Knowledge Distillation (Phase 8):** Still in progress at the time of writing.

### Final Results

| Metric | Baseline | Final (P7) | Change |
|--------|----------|------------|--------|
| Parameters | 1,805 M | 1,039 M | **−42.4%** |
| ChrF (quality) | 50.52 | 45.14 | −10.6% |
| BLEU | 11.63 | 10.20 | −12.3% |
| RTF (speed) | 0.268 | 0.113 | **2.37× faster** |

### Key Findings

- **Big wins from vocabulary pruning** — eliminating 92% of unused tokens was the single largest parameter saving with minimal quality loss.
- **Depth pruning is effective but asymmetric** — pruning optimized for EN→BN inadvertently damaged the reverse direction (BN→EN), because upper decoder layers are disproportionately important for that direction.
- **DoRA recovered most of the quality** — fine-tuning brought ChrF from ~40 back up to 45, nearly closing the gap to baseline.
- **Width pruning failed** — FLAP caused catastrophic, unrecoverable quality collapse and was abandoned.
- **Bidirectional use requires joint optimization** — future work recommends using combined directional metrics and protecting "direction-critical" layers before pruning.