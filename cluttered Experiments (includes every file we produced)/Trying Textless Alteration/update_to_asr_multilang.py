#!/usr/bin/env python3
"""
Update seamless-final.ipynb to:
1. Use ASR-based metrics (ASR-ChrF, ASR-BLEU) instead of text-based
2. Support all 5 languages: EN, BN, HI, ZH, AR
3. Bidirectional translation: En→X and X→En
4. Update all benchmarks, training, and evaluation to be multilingual
"""

import json
import re

def read_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_notebook(path, nb):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def update_cell_source(cell, old_pattern, new_source):
    """Replace cell source matching pattern with new source"""
    if cell['cell_type'] != 'code':
        return False
    
    source_text = ''.join(cell['source'])
    if re.search(old_pattern, source_text, re.DOTALL):
        cell['source'] = new_source if isinstance(new_source, list) else [new_source]
        return True
    return False

def main():
    nb_path = 'Alteration/seamless-final.ipynb'
    nb = read_notebook(nb_path)
    
    updates_made = 0
    
    # Update 1: Replace eval_samples loading with multilingual version
    for cell in nb['cells']:
        if update_cell_source(cell, 
            r"# ── Load EN→BN eval samples.*?print\(f'Loaded \{len\(eval_samples\)\} eval samples\.'\)",
            [
                "# ── Load Multilingual Eval Samples: En→X and X→En (all 5 languages) ─────────\n",
                "# PLAN.md Section 5: 5 languages — EN, BN, ZH, AR, HI\n",
                "N_EVAL_PER_PAIR = 25\n",
                "EVAL_LANG_PAIRS = [\n",
                "    ('eng', 'ben'), ('ben', 'eng'),  # English ↔ Bengali\n",
                "    ('eng', 'cmn'), ('cmn', 'eng'),  # English ↔ Mandarin\n",
                "    ('eng', 'arb'), ('arb', 'eng'),  # English ↔ Arabic\n",
                "    ('eng', 'hin'), ('hin', 'eng'),  # English ↔ Hindi\n",
                "]\n",
                "\n",
                "eval_samples = []  # Unified multilingual eval set\n",
                "for src_m4t, tgt_m4t in EVAL_LANG_PAIRS:\n",
                "    src_fleurs = M4T_FLEURS_MAP.get(src_m4t, src_m4t)\n",
                "    tgt_fleurs = M4T_FLEURS_MAP.get(tgt_m4t, tgt_m4t)\n",
                "    \n",
                "    print(f'\\nLoading {src_m4t}→{tgt_m4t} ({src_fleurs}→{tgt_fleurs}) [test]...')\n",
                "    ds_src, ds_tgt = load_fleurs_from_drive(src_fleurs, tgt_fleurs, split='test')\n",
                "    if ds_src is None or ds_tgt is None:\n",
                "        print('  [Cache miss] Downloading...')\n",
                "        ds_src, ds_tgt = load_fleurs_parallel(src_fleurs, tgt_fleurs, split='test', n_workers=8)\n",
                "        push_fleurs_to_drive()\n",
                "    \n",
                "    df_src = ds_src.to_pandas() if hasattr(ds_src,'to_pandas') else pd.DataFrame(ds_src)\n",
                "    df_tgt = ds_tgt.to_pandas() if hasattr(ds_tgt,'to_pandas') else pd.DataFrame(ds_tgt)\n",
                "    \n",
                "    src_dedup = (df_src[['id','transcription','audio']].drop_duplicates('id',keep='first')\n",
                "                 .rename(columns={'transcription':'src_text','audio':'src_audio'}))\n",
                "    tgt_dedup = (df_tgt[['id','transcription','audio']].drop_duplicates('id',keep='first')\n",
                "                 .rename(columns={'transcription':'tgt_text','audio':'tgt_audio'}))\n",
                "    merged = (pd.merge(src_dedup, tgt_dedup, on='id', how='inner')\n",
                "              .sort_values('id').reset_index(drop=True).head(N_EVAL_PER_PAIR))\n",
                "    \n",
                "    for _, row in merged.iterrows():\n",
                "        eval_samples.append({\n",
                "            'id': f\"{src_m4t}2{tgt_m4t}_{row['id']}\",\n",
                "            'src_lang': src_m4t,\n",
                "            'tgt_lang': tgt_m4t,\n",
                "            'wav': _load_wav(row['src_audio']),\n",
                "            'ref': row['tgt_text'],\n",
                "            'src_text': row['src_text'],\n",
                "        })\n",
                "    \n",
                "    del df_src, df_tgt, src_dedup, tgt_dedup, merged, ds_src, ds_tgt\n",
                "    gc.collect()\n",
                "    print(f'  Added {len([s for s in eval_samples if s[\"src_lang\"]==src_m4t and s[\"tgt_lang\"]==tgt_m4t])} samples')\n",
                "\n",
                "print(f'\\n✓ Loaded {len(eval_samples)} multilingual eval samples across {len(EVAL_LANG_PAIRS)} pairs')\n",
                "print(f'  Pairs: {[(s,t) for s,t in EVAL_LANG_PAIRS]}')\n"
            ]):
            updates_made += 1
            print("✓ Updated eval_samples loading to multilingual")
    
    # Update 2: Replace ft_samples loading with multilingual version
    for cell in nb['cells']:
        if update_cell_source(cell,
            r"# ── Load EN→BN training samples.*?print\(f'Loaded \{len\(ft_samples\)\} training samples\.'\)",
            [
                "# ── Load Multilingual Training Samples: En→X and X→En (all 5 languages) ─────\n",
                "N_TRAIN_PER_PAIR = 200  # 200 samples per direction = 1600 total\n",
                "\n",
                "ft_samples = []  # Unified multilingual training set\n",
                "for src_m4t, tgt_m4t in EVAL_LANG_PAIRS:\n",
                "    src_fleurs = M4T_FLEURS_MAP.get(src_m4t, src_m4t)\n",
                "    tgt_fleurs = M4T_FLEURS_MAP.get(tgt_m4t, tgt_m4t)\n",
                "    \n",
                "    print(f'\\nLoading {src_m4t}→{tgt_m4t} training data...')\n",
                "    src_ds, tgt_ds = load_fleurs_from_drive(src_fleurs, tgt_fleurs, split='train')\n",
                "    if src_ds is None or tgt_ds is None:\n",
                "        print('  [Cache miss] Downloading...')\n",
                "        src_ds, tgt_ds = load_fleurs_parallel(src_fleurs, tgt_fleurs, split='train', n_workers=8)\n",
                "        push_fleurs_to_drive()\n",
                "    \n",
                "    df_src_train = src_ds.to_pandas() if hasattr(src_ds,'to_pandas') else pd.DataFrame(src_ds)\n",
                "    df_tgt_train = tgt_ds.to_pandas() if hasattr(tgt_ds,'to_pandas') else pd.DataFrame(tgt_ds)\n",
                "    \n",
                "    src_tr = (df_src_train[['id','audio']].drop_duplicates('id',keep='first')\n",
                "              .rename(columns={'audio':'src_audio'}))\n",
                "    tgt_tr = (df_tgt_train[['id','transcription','audio']].drop_duplicates('id',keep='first')\n",
                "              .rename(columns={'transcription':'tgt_text','audio':'tgt_audio'}))\n",
                "    merged_train = pd.merge(src_tr, tgt_tr, on='id', how='inner').reset_index(drop=True)\n",
                "    merged_train = merged_train[merged_train['tgt_text'].str.strip().str.len() > 0]\n",
                "    merged_train = merged_train.head(N_TRAIN_PER_PAIR)\n",
                "    \n",
                "    for _, row in merged_train.iterrows():\n",
                "        ft_samples.append({\n",
                "            'id': f\"{src_m4t}2{tgt_m4t}_{row['id']}\",\n",
                "            'src_lang': src_m4t,\n",
                "            'tgt_lang': tgt_m4t,\n",
                "            'wav': _load_wav(row['src_audio']),\n",
                "            'ref': row['tgt_text'],\n",
                "        })\n",
                "    \n",
                "    del df_src_train, df_tgt_train, src_tr, tgt_tr, merged_train, src_ds, tgt_ds\n",
                "    gc.collect()\n",
                "    print(f'  Added {len([s for s in ft_samples if s[\"src_lang\"]==src_m4t and s[\"tgt_lang\"]==tgt_m4t])} samples')\n",
                "\n",
                "print(f'\\n✓ Loaded {len(ft_samples)} multilingual training samples across {len(EVAL_LANG_PAIRS)} pairs')\n"
            ]):
            updates_made += 1
            print("✓ Updated ft_samples loading to multilingual")
    
    # Update 3: Replace run_benchmark function with ASR version
    for cell in nb['cells']:
        if update_cell_source(cell,
            r"def run_benchmark\(mdl, samples, label=.*?\n    return results, summary",
            [
                "def run_benchmark_asr(mdl, samples, label='model', save_n=2):\n",
                "    \"\"\"ASR-based benchmark: translate audio → ASR transcribe → compute ASR-ChrF/BLEU.\"\"\"\n",
                "    print(f'\\n{\"=\"*60}\\n  BENCHMARK (ASR): {label}  Samples:{len(samples)}\\n{\"=\"*60}')\n",
                "    gpu_mem()\n",
                "    results = []\n",
                "    \n",
                "    # Group samples by language pair for organized output\n",
                "    from collections import defaultdict\n",
                "    by_pair = defaultdict(list)\n",
                "    for s in samples:\n",
                "        by_pair[f\"{s['src_lang']}→{s['tgt_lang']}\"].append(s)\n",
                "    \n",
                "    for pair_key, pair_samples in by_pair.items():\n",
                "        print(f'\\n  === {pair_key} ({len(pair_samples)} samples) ===')\n",
                "        for i, s in enumerate(pair_samples):\n",
                "            try:\n",
                "                dur = len(s['wav']) / 16000\n",
                "                t0  = time.time()\n",
                "                \n",
                "                # Run S2ST translation\n",
                "                _, wav_out = run_s2st_legacy(mdl, s['wav'], tgt_lang=s['tgt_lang'])\n",
                "                \n",
                "                # ASR transcribe output audio\n",
                "                pred = asr_transcribe(wav_out, s['tgt_lang'])\n",
                "                \n",
                "                rtf  = (time.time() - t0) / dur\n",
                "                bleu = compute_bleu(pred, s['ref'])\n",
                "                chrf = compute_chrf(pred, s['ref'])\n",
                "                \n",
                "                print(f'  [{i+1:>2}/{len(pair_samples)}] ASR-BLEU={bleu:5.1f} ASR-ChrF={chrf:5.1f} RTF={rtf:.3f}')\n",
                "                print(f'              pred: {pred[:80]}')\n",
                "                \n",
                "                if save_n > 0 and i < save_n:\n",
                "                    play(s['wav'], 16000, label=f'{label}_{pair_key}_s{i+1}in.wav')\n",
                "                    save_audio(s['wav'], 16000, f'{label}_{pair_key}_s{i+1}in.wav')\n",
                "                    play(wav_out, 16000, label=f'{label}_{pair_key}_s{i+1}out.wav')\n",
                "                    save_audio(wav_out, 16000, f'{label}_{pair_key}_s{i+1}out.wav')\n",
                "                \n",
                "                results.append(dict(\n",
                "                    id=s['id'], src_lang=s['src_lang'], tgt_lang=s['tgt_lang'],\n",
                "                    bleu=bleu, chrf=chrf, rtf=rtf, pred=pred, ref=s['ref']))\n",
                "            except Exception as e:\n",
                "                import traceback; traceback.print_exc()\n",
                "                results.append(dict(\n",
                "                    id=s['id'], src_lang=s.get('src_lang','?'), tgt_lang=s.get('tgt_lang','?'),\n",
                "                    bleu=0, chrf=0, rtf=float('nan'), pred='', ref=s.get('ref','')))\n",
                "    \n",
                "    valid = [r for r in results if not math.isnan(r['rtf'])]\n",
                "    summary = dict(\n",
                "        label=label, n=len(valid),\n",
                "        avg_bleu=float(np.mean([r['bleu'] for r in valid])) if valid else 0,\n",
                "        avg_chrf=float(np.mean([r['chrf'] for r in valid])) if valid else 0,\n",
                "        avg_rtf =float(np.mean([r['rtf']  for r in valid])) if valid else 0,\n",
                "        params_M=count_params(mdl)\n",
                "    )\n",
                "    \n",
                "    # Per-pair breakdown\n",
                "    print(f'\\n  === Summary by Language Pair ===')\n",
                "    for pair_key in by_pair.keys():\n",
                "        pair_res = [r for r in valid if f\"{r['src_lang']}→{r['tgt_lang']}\" == pair_key]\n",
                "        if pair_res:\n",
                "            avg_chrf_pair = np.mean([r['chrf'] for r in pair_res])\n",
                "            avg_bleu_pair = np.mean([r['bleu'] for r in pair_res])\n",
                "            print(f'  {pair_key:<12} ASR-ChrF={avg_chrf_pair:5.2f}  ASR-BLEU={avg_bleu_pair:5.2f}')\n",
                "    \n",
                "    print(f'\\n  Overall: ASR-BLEU={summary[\"avg_bleu\"]:.2f} ASR-ChrF={summary[\"avg_chrf\"]:.2f}'\n",
                "          f' RTF={summary[\"avg_rtf\"]:.4f} Params={summary[\"params_M\"]:.1f}M')\n",
                "    return results, summary\n",
                "\n",
                "# Alias for backward compatibility\n",
                "run_benchmark = run_benchmark_asr\n"
            ]):
            updates_made += 1
            print("✓ Updated run_benchmark to ASR version")
    
    # Update 4: Update quick_eval_chrf to ASR version
    for cell in nb['cells']:
        if update_cell_source(cell,
            r"def quick_eval_chrf\(mdl, samples.*?\n    return float\(np\.mean\(scores\)\)",
            [
                "def quick_eval_chrf(mdl, samples, tgt_lang='ben', max_samples=10):\n",
                "    \"\"\"Quick ASR-ChrF evaluation for pruning decisions.\"\"\"\n",
                "    scores = []\n",
                "    for s in samples[:max_samples]:\n",
                "        try:\n",
                "            # Get target language from sample if available\n",
                "            tgt = s.get('tgt_lang', tgt_lang)\n",
                "            _, wav_out = run_s2st_legacy(mdl, s['wav'], tgt_lang=tgt)\n",
                "            pred = asr_transcribe(wav_out, tgt)\n",
                "            scores.append(compute_chrf(pred, s['ref']))\n",
                "        except:\n",
                "            scores.append(0.0)\n",
                "    return float(np.mean(scores))\n"
            ]):
            updates_made += 1
            print("✓ Updated quick_eval_chrf to ASR version")
    
    # Update 5: Update iterative_enc_prune to use multilingual samples
    for cell in nb['cells']:
        source_text = ''.join(cell.get('source', []))
        if 'def iterative_enc_prune(' in source_text and 'baseline = quick_eval_chrf' in source_text:
            # Update the baseline calculation to use multilingual samples
            new_source = source_text.replace(
                'baseline = quick_eval_chrf(mdl, samples, tgt_lang, max_eval)',
                '# Use multilingual samples for baseline\n    baseline = quick_eval_chrf(mdl, samples, max_eval=max_eval)'
            )
            new_source = new_source.replace(
                'sc = quick_eval_chrf(mdl, samples, tgt_lang, max_eval)',
                'sc = quick_eval_chrf(mdl, samples, max_eval=max_eval)'
            )
            cell['source'] = [new_source]
            updates_made += 1
            print("✓ Updated iterative_enc_prune to use multilingual samples")
    
    # Update 6: Update Phase 0 benchmark call
    for cell in nb['cells']:
        if update_cell_source(cell,
            r"p0_results, p0_summary = run_benchmark\(\s*model_v1, eval_samples, label='P0_V1_Baseline', tgt_lang='ben'",
            [
                "    p0_results, p0_summary = run_benchmark_asr(\n",
                "        model_v1, eval_samples, label='P0_V1_Baseline', save_n=2)\n"
            ]):
            updates_made += 1
            print("✓ Updated Phase 0 benchmark call")
    
    # Update 7: Update Phase 2 enc pruning call
    for cell in nb['cells']:
        source_text = ''.join(cell.get('source', []))
        if 'removed_enc, p2_log = iterative_enc_prune(' in source_text:
            new_source = source_text.replace(
                "iterative_enc_prune(\n        model_p2, eval_samples, N_ENC_REMOVE, 'ben'",
                "iterative_enc_prune(\n        model_p2, eval_samples, N_ENC_REMOVE"
            )
            cell['source'] = [new_source]
            updates_made += 1
            print("✓ Updated Phase 2 enc pruning call")
    
    # Update 8: Remove multilang_eval section (now integrated into eval_samples)
    for i, cell in enumerate(nb['cells']):
        source_text = ''.join(cell.get('source', []))
        if '# ── Multilingual eval samples: 10 per lang for Phase 7 benchmark' in source_text:
            cell['source'] = [
                "# ── Multilingual eval samples now integrated into eval_samples ──────────────\n",
                "# All 5 languages (EN, BN, HI, ZH, AR) with bidirectional pairs are loaded above\n",
                "print(f'Multilingual eval ready: {len(eval_samples)} samples across {len(EVAL_LANG_PAIRS)} pairs')\n",
                "print(f'Language pairs: {EVAL_LANG_PAIRS}')\n"
            ]
            updates_made += 1
            print("✓ Removed redundant multilang_eval section")
    
    # Update 9: Update KD extraction to use multilingual samples
    for cell in nb['cells']:
        source_text = ''.join(cell.get('source', []))
        if 'all_train_samples = {' in source_text and 'eng2ben' in source_text:
            new_source = source_text.replace(
                "all_train_samples = {'eng2ben': ft_samples[:200]}",
                "# Build multilingual train set from ft_samples (already multilingual)\n    all_train_samples = {}\n    for src_m4t, tgt_m4t in EVAL_LANG_PAIRS:\n        pair_key = f'{src_m4t}2{tgt_m4t}'\n        pair_samples = [s for s in ft_samples if s['src_lang']==src_m4t and s['tgt_lang']==tgt_m4t]\n        all_train_samples[pair_key] = pair_samples[:200]"
            )
            new_source = new_source.replace(
                "PAIRS = [('eng','ben')]   # primary; extend with more lang pairs if Drive has data",
                "PAIRS = EVAL_LANG_PAIRS  # All 5 language pairs, bidirectional"
            )
            cell['source'] = [new_source]
            updates_made += 1
            print("✓ Updated KD extraction to use multilingual samples")
    
    # Save updated notebook
    write_notebook(nb_path, nb)
    print(f"\n✅ Successfully made {updates_made} updates to {nb_path}")
    print("\nKey changes:")
    print("  • eval_samples: Now loads all 5 languages bidirectionally (8 pairs)")
    print("  • ft_samples: Now loads training data for all 8 language pairs")
    print("  • run_benchmark: Now uses ASR-ChrF and ASR-BLEU metrics")
    print("  • quick_eval_chrf: Now uses ASR transcription")
    print("  • All training/pruning: Now uses multilingual samples")
    print("\nNext: Run the notebook to verify all changes work correctly")

if __name__ == '__main__':
    main()
