# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 FIX: ASR-ChrF for Direction, ASR-BLEU for Benchmarking
# ═══════════════════════════════════════════════════════════════════════════════
#
# ROOT CAUSE OF BUG:
#   quick_eval_chrf() uses run_s2t_only(), which bypasses the T2U model entirely.
#   T2U layers only affect the audio output path (text tokens → speech units → audio).
#   Text-only evaluation shows no difference when T2U layers are removed because
#   the text decoder path is unchanged.
#
# FIX:
#   - Direction Selection: ASR-ChrF (determines which layer to remove)
#   - Benchmarking: ASR-BLEU (computed separately for final reporting)
#
#   This evaluates the ACTUAL S2ST audio output by:
#     1. Running full S2ST pipeline (generates audio)
#     2. Transcribing output audio with MMS-ASR (Bengali)
#     3. Computing ChrF for direction selection (which layer to prune)
#     4. Computing BLEU for benchmarking (final quality metric)
#
# USAGE:
#   Replace Phase 6 Cell 3 in your notebook with this code.
#
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import numpy as np


def quick_eval_asr_chrf(model, samples, tgt_lang='ben', max_eval=10):
    """
    Fast S2ST quality evaluation using ASR-ChrF for direction selection.
    
    This is the CORRECT metric for T2U pruning because:
      - T2U layers only affect the audio output path (text → units → audio)
      - Text-only metrics (quick_eval_chrf) bypass T2U entirely
      - ASR-ChrF measures whether the output audio contains correct words
    
    Returns: avg_asr_chrf (used for pruning direction selection)
    """
    _ensure_mms_loaded()  # Load MMS-ASR model if not already loaded
    
    scores = []
    for s in samples[:max_eval]:
        try:
            # Full S2ST: generates both text and audio
            pred_text, out_wav = run_s2st(model, s['wav'], tgt_lang=tgt_lang)
            
            # ASR-ChrF: transcribe output audio, compare to reference
            if out_wav is not None and len(out_wav) > 1600:
                _, asr_chrf = compute_asr_chrf(
                    out_wav, s['ref'], sr=model.config.sampling_rate)
                scores.append(asr_chrf)
            else:
                # No audio output = catastrophic failure
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    
    return float(np.mean(scores)) if scores else 0.0


def benchmark_asr_bleu(model, samples, tgt_lang='ben', max_eval=10):
    """
    Compute ASR-BLEU for benchmarking purposes only.
    
    This metric is NOT used for pruning decisions, only for final reporting.
    ASR-ChrF is used for direction selection during pruning.
    
    Returns: avg_asr_bleu
    """
    _ensure_mms_loaded()  # Load MMS-ASR model if not already loaded
    
    scores = []
    for s in samples[:max_eval]:
        try:
            # Full S2ST: generates both text and audio
            pred_text, out_wav = run_s2st(model, s['wav'], tgt_lang=tgt_lang)
            
            # ASR-BLEU: transcribe output audio, compare to reference
            if out_wav is not None and len(out_wav) > 1600:
                asr_bleu, _ = compute_asr_bleu(
                    out_wav, s['ref'], sr=model.config.sampling_rate)
                scores.append(asr_bleu)
            else:
                # No audio output = catastrophic failure
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    
    return float(np.mean(scores)) if scores else 0.0


def iterative_prune_t2u_stack(model, stack_parent, layers_attr,
                               stack_name, samples, n_remove,
                               tgt_lang='ben', max_eval=10,
                               ckpt_name=None):
    """
    Iterative greedy pruning for one T2U layer stack.
    
    KEY FIX: Uses ASR-ChrF scoring for direction selection (not text-only ChrF).
    T2U layers only affect the audio output path, so we must evaluate
    the actual S2ST audio quality using MMS-ASR transcription.
    
    Direction Selection: ASR-ChrF (which layer to remove)
    Benchmarking: ASR-BLEU (computed separately for final reporting)
    
    Selection Strategy: Remove the layer whose removal causes LEAST ChrF degradation
                       (i.e., keeps the highest ASR-ChrF).
    """
    if ckpt_name is None:
        ckpt_name = f'phase6_{stack_name.replace(".", "_").replace(" ", "_")}_pruning'

    current = list(getattr(stack_parent, layers_attr))
    orig_indices = list(range(len(current)))
    n_total_orig = len(current)

    # Clamp n_remove so at least 2 layers always remain
    if n_total_orig - n_remove < 2:
        n_remove = max(0, n_total_orig - 2)
        print(f'  Clamped n_remove to {n_remove} (keeping minimum 2 layers)')

    if n_remove == 0:
        print(f'  {stack_name}: nothing to remove.')
        return [], []

    print(f'  {stack_name}: {n_total_orig} layers, removing {n_remove} (all eligible)')
    print(f'  Direction Selection: ASR-ChrF (via MMS-ASR Bengali transcription)')
    print(f'  Benchmarking: ASR-BLEU (computed separately)')

    removed, log = [], []

    # ── Resume from checkpoint ──
    partial = load_latest_checkpoint(ckpt_name)
    if partial and partial.get('removed'):
        removed = list(partial['removed'])
        log = partial.get('log', [])
        for r in removed:
            if r in orig_indices:
                pos = orig_indices.index(r)
                current.pop(pos)
                orig_indices.pop(pos)
        setattr(stack_parent, layers_attr, nn.ModuleList(current))
        print(f'  Resuming: already removed {removed}, {len(current)} layers remain')

    # ── Baseline Metrics ──
    baseline_chrf = quick_eval_asr_chrf(model, samples, tgt_lang, max_eval)
    baseline_bleu = benchmark_asr_bleu(model, samples, tgt_lang, max_eval)
    print(f'  Baseline ASR-ChrF: {baseline_chrf:.2f} (direction metric)')
    print(f'  Baseline ASR-BLEU: {baseline_bleu:.2f} (benchmark metric)')

    start_iter = len(removed)
    for it in range(start_iter, n_remove):
        eligible = list(range(len(current)))

        if not eligible:
            print(f'  WARNING: No layers left to prune. Stopping.')
            break

        print(f'\n  Iter {it+1}/{n_remove} ({len(current)} layers remain, '
              f'all {len(eligible)} eligible)')

        scores = {}
        for pos in eligible:
            temp = current[:pos] + current[pos+1:]
            setattr(stack_parent, layers_attr, nn.ModuleList(temp))
            
            # Evaluate ASR-ChrF with this layer removed (for direction selection)
            chrf_sc = quick_eval_asr_chrf(model, samples, tgt_lang, max_eval)
            scores[pos] = (orig_indices[pos], chrf_sc)
            print(f'    Remove L{orig_indices[pos]:>2} -> ASR-ChrF={chrf_sc:.2f}')
        
        # Restore full current stack before committing the best removal
        setattr(stack_parent, layers_attr, nn.ModuleList(current))

        # Pick the removal that keeps ASR-ChrF highest (direction selection)
        best_pos = max(scores, key=lambda k: scores[k][1])
        best_orig, best_chrf = scores[best_pos]
        
        # Now compute ASR-BLEU for the selected removal (benchmarking only)
        temp = current[:best_pos] + current[best_pos+1:]
        setattr(stack_parent, layers_attr, nn.ModuleList(temp))
        best_bleu = benchmark_asr_bleu(model, samples, tgt_lang, max_eval)
        
        # Commit the removal
        current.pop(best_pos)
        orig_indices.pop(best_pos)
        setattr(stack_parent, layers_attr, nn.ModuleList(current))
        removed.append(best_orig)
        
        log.append(dict(
            iter=it+1,
            removed=best_orig,
            asr_chrf=best_chrf,
            asr_bleu=best_bleu,
            remaining=len(current)
        ))
        
        print(f'  -> Removed L{best_orig} (ASR-ChrF={best_chrf:.2f}, '
              f'ASR-BLEU={best_bleu:.2f}, {len(current)} layers remain)')

        # Save progress after every removal
        save_checkpoint(dict(removed=removed, log=log), name=ckpt_name, step=0)
        print(f'  [ckpt] Progress saved ({it+1}/{n_remove} iterations done)')
        torch.cuda.empty_cache()

    return removed, log


print('iterative_prune_t2u_stack() — ASR-ChrF for direction, ASR-BLEU for benchmarking.')
print('  Direction Selection: ASR-ChrF (determines which layer to remove)')
print('  Benchmarking: ASR-BLEU (computed separately for final reporting)')
print('  This correctly evaluates T2U layer impact on audio output quality.')
