# cell 90
# ── Phase 7 Cell 8: Merge DoRA adapters into base model ──────────────────────
# After training, merge the DoRA adapter weights back into the base model.
# This eliminates ALL inference overhead — the merged model is identical in
# speed to model_p6.  Only do this once training is complete.

print('Merging DoRA adapters into base model...')
model_p7_merged = model_p7.merge_and_unload()
model_p7_merged.eval()
gc.collect(); torch.cuda.empty_cache()
print('Merge complete. model_p7_merged has no adapter overhead.')

# Sync config after merge (DoRA merge can affect hidden sizes in config)
sync_model_config(model_p7_merged)

# Save merged model to Drive for next session
save_model_to_drive(model_p7_merged, processor, 'phase7_dora_merged')
print_model_breakdown(model_p7_merged, 'After Phase 7: DoRA Fine-tuned & Merged')