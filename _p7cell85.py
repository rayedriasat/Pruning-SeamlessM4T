# cell 85
# ── Phase 7 Cell 5: S2TT loss function ───────────────────────────────────────
# We train with S2TT cross-entropy: given speech → predict the Bengali text.
# This is 3-5× faster per step than full S2ST (no T2U/vocoder forward pass)
# and directly optimises the ChrF/BLEU metric we measure at benchmark time.
#
# Implementation: use model.generate() labels trick via teacher-forcing
# with processor's tokenizer to get target token IDs, then run a forward
# pass that returns text_decoder logits, compute cross-entropy.

def _model_device(mdl):
    return next(mdl.parameters()).device

def prepare_s2tt_batch(batch, processor, device, tgt_lang):
    audios = [s['wav'] for s in batch]
    targets = [s['ref'] for s in batch]

    # Processor handles task tokens (<S2TT>) to prevent CUDA asserts
    encoded = processor(
        audio=audios,
        sampling_rate=16000,
        text_target=targets,
        tgt_lang=tgt_lang,
        return_tensors="pt",
        padding=True
    )

    input_feats = encoded['input_features'].to(device)
    attn_mask = encoded['attention_mask'].to(device)
    
    # FIX: Check if 'labels' exists; if not, use 'input_ids' from text_target
    if 'labels' in encoded:
        labels = encoded['labels'].to(device)
    elif 'input_ids' in encoded:
        labels = encoded['input_ids'].to(device)
    else:
        # Fallback for older transformers versions
        labels = encoded['decoder_input_ids'].to(device)

    # Standard: mask padding tokens so they don't contribute to loss
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return input_feats, attn_mask, labels

def compute_s2tt_loss(model, input_feats, attn_mask, labels, tgt_lang):
    outputs = model(
        input_features=input_feats,
        attention_mask=attn_mask,
        labels=labels,
        tgt_lang=tgt_lang,
        return_dict=True
    )
    return outputs.loss