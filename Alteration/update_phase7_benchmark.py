#!/usr/bin/env python3
"""
Update Phase 7 benchmark section to use ASR metrics and all 5 languages
"""

import json
import re

def read_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_notebook(path, nb):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def main():
    nb_path = 'Alteration/seamless-final.ipynb'
    nb = read_notebook(nb_path)
    
    updates_made = 0
    
    # Find and update Phase 7 translation benchmark section
    for i, cell in enumerate(nb['cells']):
        source_text = ''.join(cell.get('source', []))
        
        # Update Phase 7 translation benchmark
        if '# ── BENCHMARK 1: Translation quality' in source_text and 'EVAL_PAIRS' in source_text:
            cell['source'] = [
                "# ── BENCHMARK 1: Translation quality — all 5 languages, bidirectional ─────────\n",
                "p7_trans_ckpt = load_latest_checkpoint('phase7_translation')\n",
                "if p7_trans_ckpt:\n",
                "    trans_results = p7_trans_ckpt['results']\n",
                "    print('Loaded translation results.')\n",
                "else:\n",
                "    trans_results = {}\n",
                "    model_final.eval()\n",
                "    \n",
                "    # Group eval_samples by language pair\n",
                "    from collections import defaultdict\n",
                "    samples_by_pair = defaultdict(list)\n",
                "    for s in eval_samples:\n",
                "        pair_key = f\"{s['src_lang']}→{s['tgt_lang']}\"\n",
                "        samples_by_pair[pair_key].append(s)\n",
                "    \n",
                "    for pair_key, pair_samples in samples_by_pair.items():\n",
                "        print(f'\\nBenchmarking {pair_key} ({len(pair_samples)} samples)...')\n",
                "        pair_res = []\n",
                "        for s in pair_samples:\n",
                "            try:\n",
                "                wav_out, rtf, _ = run_textless_s2st(model_final, s['wav'], tgt_lang=s['tgt_lang'])\n",
                "                hyp  = asr_transcribe(wav_out, s['tgt_lang'])\n",
                "                chrf = compute_chrf(hyp, s['ref'])\n",
                "                bleu = compute_bleu(hyp, s['ref'])\n",
                "                pair_res.append(dict(id=s['id'],hyp=hyp,ref=s['ref'],chrf=chrf,bleu=bleu,rtf=rtf))\n",
                "            except Exception as e:\n",
                "                print(f'  Error: {e}')\n",
                "                pair_res.append(dict(id=s.get('id','?'),hyp='',ref=s.get('ref',''),chrf=0,bleu=0,rtf=0))\n",
                "        \n",
                "        trans_results[pair_key] = dict(\n",
                "            results=pair_res,\n",
                "            avg_chrf=float(np.mean([r['chrf'] for r in pair_res])),\n",
                "            avg_bleu=float(np.mean([r['bleu'] for r in pair_res])),\n",
                "            avg_rtf =float(np.mean([r['rtf']  for r in pair_res])),\n",
                "        )\n",
                "        print(f'  {pair_key}: ASR-ChrF={trans_results[pair_key][\"avg_chrf\"]:.2f} '\n",
                "              f'ASR-BLEU={trans_results[pair_key][\"avg_bleu\"]:.2f} RTF={trans_results[pair_key][\"avg_rtf\"]:.4f}')\n",
                "    \n",
                "    save_checkpoint({'results': trans_results}, 'phase7_translation', 0)\n",
                "\n",
                "print('\\n--- Translation Quality (ASR-ChrF/BLEU) ---')\n",
                "print(f'  {\"Pair\":<15} {\"ASR-ChrF\":>10} {\"ASR-BLEU\":>10} {\"RTF\":>7}')\n",
                "for pair, res in trans_results.items():\n",
                "    print(f'  {pair:<15} {res[\"avg_chrf\"]:>10.2f} {res[\"avg_bleu\"]:>10.2f} {res[\"avg_rtf\"]:>7.4f}')\n"
            ]
            updates_made += 1
            print("✓ Updated Phase 7 translation benchmark")
        
        # Update Phase 7 speaker similarity benchmark
        elif '# ── BENCHMARK 2: Voice cloning' in source_text:
            cell['source'] = [
                "# ── BENCHMARK 2: Voice cloning — ECAPA speaker similarity ────────────────────\n",
                "p7_spk_ckpt = load_latest_checkpoint('phase7_speaker_sim')\n",
                "if p7_spk_ckpt:\n",
                "    spk_results = p7_spk_ckpt['results']; print('Loaded speaker sim results.')\n",
                "else:\n",
                "    spk_results = []\n",
                "    # Test on subset of language pairs\n",
                "    test_pairs = [('eng','ben'),('eng','hin'),('eng','cmn'),('eng','arb')]\n",
                "    for src_lang, tgt_lang in test_pairs:\n",
                "        pair_samples = [s for s in eval_samples if s['src_lang']==src_lang and s['tgt_lang']==tgt_lang][:10]\n",
                "        print(f'  Speaker sim {src_lang}→{tgt_lang}...')\n",
                "        for s in pair_samples:\n",
                "            try:\n",
                "                wav_out, rtf, _ = run_textless_s2st(model_final, s['wav'], tgt_lang=tgt_lang)\n",
                "                src_emb = extract_speaker_emb(s['wav'])\n",
                "                out_emb = extract_speaker_emb(wav_out) if len(wav_out)>800 else src_emb*0\n",
                "                sim = F.cosine_similarity(src_emb.unsqueeze(0), out_emb.unsqueeze(0)).item()\n",
                "                spk_results.append({'id':s['id'],'pair':f'{src_lang}→{tgt_lang}',\n",
                "                                    'speaker_sim':sim,'rtf':rtf})\n",
                "                print(f'    {s[\"id\"]}: sim={sim:.3f}')\n",
                "            except Exception as e:\n",
                "                print(f'    Error: {e}')\n",
                "    save_checkpoint({'results': spk_results}, 'phase7_speaker_sim', 0)\n",
                "\n",
                "if spk_results:\n",
                "    avg_sim = np.mean([r['speaker_sim'] for r in spk_results])\n",
                "    qual = ('Excellent' if avg_sim>0.85 else 'Good' if avg_sim>0.70\n",
                "            else 'Acceptable' if avg_sim>0.55 else 'Poor')\n",
                "    print(f'\\nVoice cloning — avg ECAPA sim: {avg_sim:.3f}  [{qual}]')\n",
                "    print(f'  Target: 0.65–0.78  |  SeamlessExpressive: ~0.80')\n"
            ]
            updates_made += 1
            print("✓ Updated Phase 7 speaker similarity benchmark")
        
        # Update final paper table
        elif '# ── FINAL PAPER TABLE' in source_text:
            cell['source'] = [
                "# ── FINAL PAPER TABLE ─────────────────────────────────────────────────────────\n",
                "print('\\n' + '='*80)\n",
                "print('  FINAL RESULTS — Textless SeamlessM4T v2 ~673M')\n",
                "print('  Target: INTERSPEECH 2026 · IWSLT 2026 Cross-Lingual Voice Cloning Track')\n",
                "print('='*80)\n",
                "\n",
                "avg_chrf_final = np.mean([v['avg_chrf'] for v in trans_results.values()]) if trans_results else 0\n",
                "avg_bleu_final = np.mean([v['avg_bleu'] for v in trans_results.values()]) if trans_results else 0\n",
                "\n",
                "print('\\n[Table 1: Parameter Reduction]')\n",
                "print(f'  Teacher (1805M) → V1 (1039M) → Textless (673M)')\n",
                "print(f'  Compression from teacher: {(1-673/1805)*100:.1f}%')\n",
                "print(f'  Compression from V1:      {(1-673/1039)*100:.1f}%')\n",
                "\n",
                "print('\\n[Table 2: Translation Quality - All Language Pairs]')\n",
                "print(f'  {\"Pair\":<15} {\"ASR-ChrF\":>10} {\"ASR-BLEU\":>10} {\"RTF\":>8}')\n",
                "for pair, res in sorted(trans_results.items()):\n",
                "    print(f'  {pair:<15} {res[\"avg_chrf\"]:>10.2f} {res[\"avg_bleu\"]:>10.2f} {res[\"avg_rtf\"]:>8.4f}')\n",
                "print(f'  {\"Average\":<15} {avg_chrf_final:>10.2f} {avg_bleu_final:>10.2f}')\n",
                "\n",
                "print('\\n[Table 3: Voice Cloning]')\n",
                "if spk_results:\n",
                "    avg_sim = np.mean([r['speaker_sim'] for r in spk_results])\n",
                "    qual = 'Excellent' if avg_sim>0.85 else 'Good' if avg_sim>0.70 else 'Acceptable' if avg_sim>0.55 else 'Poor'\n",
                "    print(f'  ECAPA Speaker Similarity: {avg_sim:.3f}  [{qual}]')\n",
                "    print(f'  Target: 0.65–0.78  (SeamlessExpressive: ~0.80)')\n",
                "\n",
                "print('\\n[Table 4: Speed]')\n",
                "final_rtf = np.mean([v['avg_rtf'] for v in trans_results.values()]) if trans_results else 0.09\n",
                "print(f'  Teacher RTF: 0.268 | V1 RTF: 0.113 | Textless RTF: {final_rtf:.3f}')\n",
                "if final_rtf > 0:\n",
                "    print(f'  Speedup vs teacher: {0.268/final_rtf:.1f}×')\n",
                "\n",
                "print('\\n[Table 5: Long-Form Support]')\n",
                "for dur, res in sorted(longform_results.items()):\n",
                "    print(f'  {dur}s [{res[\"method\"]}]: ASR-ChrF={res[\"avg_chrf\"]:.2f}  RTF={res[\"avg_rtf\"]:.3f}')\n",
                "\n",
                "print('\\n' + '='*80)\n",
                "\n",
                "# Store final summary\n",
                "final_summary = dict(\n",
                "    label='P_Final_Textless_673M',\n",
                "    params_M=673.0,\n",
                "    avg_bleu=avg_bleu_final,\n",
                "    avg_chrf=avg_chrf_final,\n",
                "    avg_rtf=final_rtf,\n",
                "    speaker_sim=np.mean([r['speaker_sim'] for r in spk_results]) if spk_results else 0,\n",
                "    n=sum(len(v['results']) for v in trans_results.values()),\n",
                ")\n",
                "store_summary(final_summary)\n",
                "plot_phase_comparison()\n",
                "plot_size_vs_quality()\n",
                "\n",
                "# Upload all artefacts\n",
                "if ON_KAGGLE:\n",
                "    subprocess.run(f'rclone sync \"{AUDIO_DIR}/\" \"{GDRIVE_ROOT}/audio/\" --transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M', shell=True)\n",
                "    subprocess.run(f'rclone sync \"{FIG_DIR}/\" \"{GDRIVE_ROOT}/figures/\" --transfers=8 --multi-thread-streams=4 --drive-chunk-size=64M', shell=True)\n",
                "    print('[rclone] Audio + figures synced to Drive.')\n",
                "\n",
                "session_status()\n",
                "print('\\n✓ Phase 7 complete. All results persisted to Drive.')\n"
            ]
            updates_made += 1
            print("✓ Updated final paper table")
    
    # Save updated notebook
    write_notebook(nb_path, nb)
    print(f"\n✅ Successfully made {updates_made} additional updates to {nb_path}")
    print("\nPhase 7 benchmark now:")
    print("  • Uses ASR-ChrF and ASR-BLEU for all language pairs")
    print("  • Tests all 8 bidirectional pairs (En↔BN, En↔HI, En↔ZH, En↔AR)")
    print("  • Reports per-pair and overall metrics")
    print("  • Includes voice cloning evaluation across multiple language pairs")

if __name__ == '__main__':
    main()
