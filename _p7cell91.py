# cell 91
# ── Phase 7 Cell 9: Full benchmark ───────────────────────────────────────────
p7b = load_latest_checkpoint('phase7_benchmark')
if p7b:
    p7_results, p7_summary = p7b['results'], p7b['summary']
    print(f'Loaded P7 benchmark: BLEU={p7_summary["avg_bleu"]:.2f}  '
          f'ChrF={p7_summary["avg_chrf"]:.2f}')
else:
    p7_results, p7_summary = run_benchmark(
        model_p7_merged, eval_samples, label='P7_DoRA', save_n=4)
    save_checkpoint(dict(results=p7_results, summary=p7_summary),
                    name='phase7_benchmark', step=0)

# ── Recovery delta against p4 and p6 baselines ───────────────────────────────
p4b = load_latest_checkpoint('phase4_benchmark')
p6b = load_latest_checkpoint('phase6_benchmark')
p4_chrf = p4b['summary']['avg_chrf'] if p4b else 0.0
p6_chrf = p6b['summary']['avg_chrf'] if p6b else 0.0
p7_chrf = p7_summary['avg_chrf']

print(f'\n{"="*55}')
print(f'  Phase 4 (before T2U pruning) ChrF : {p4_chrf:.2f}')
print(f'  Phase 6 (after T2U pruning)  ChrF : {p6_chrf:.2f}  '
      f'(drop: {p4_chrf - p6_chrf:.2f})')
print(f'  Phase 7 (DoRA fine-tuned)    ChrF : {p7_chrf:.2f}  '
      f'(recovery: +{p7_chrf - p6_chrf:.2f})')
print(f'{"="*55}')

store_summary(p7_summary)
plot_phase_comparison()
plot_size_vs_quality()


# ── Phase 7 Cell 10: Load Phase 7 from Drive (future sessions) ───────────────
# Use this cell at the start of a new session instead of rerunning training.
#
# IMPORTANT: Phase 7 saved a MERGED model — no adapter overhead at inference.
# Loading is identical to loading any other phase model.
#
# def load_p7():
#     model_p7_merged, processor = load_model_from_drive('phase7_dora_merged')
#     sync_model_config(model_p7_merged)
#     custom = load_latest_checkpoint('_custom_state')
#     if custom and '_vocab_remap_to_old' in custom:
#         model_p7_merged._vocab_remap_to_old = custom['_vocab_remap_to_old']
#     return model_p7_merged, processor
#
# If you want to reload without merging (e.g., to continue training):
#
# def load_p7_with_adapter():
#     from peft import PeftModel
#     # Start from model_p6 (already pruned architecture)
#     base = model_p6
#     model_p7 = PeftModel.from_pretrained(
#         base, f'{MODEL_DIR}/phase7_dora_adapter')
#     return model_p7