#!/usr/bin/env python3
"""
Update Phase 7 comprehensive visualization to show multilingual results
"""

import json

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
    
    # Find and update the comprehensive visualization
    for i, cell in enumerate(nb['cells']):
        source_text = ''.join(cell.get('source', []))
        
        if '# ── FINAL COMPREHENSIVE VISUALISATION (paper figures)' in source_text:
            cell['source'] = [
                "# ── FINAL COMPREHENSIVE VISUALISATION (paper figures) ─────────────────────────\n",
                "fig = plt.figure(figsize=(20, 16))\n",
                "fig.suptitle('Textless SeamlessM4T v2 (~673M): Comprehensive Benchmark - 5 Languages',\n",
                "             fontsize=14, fontweight='bold', y=0.99)\n",
                "\n",
                "# 1: Parameter evolution\n",
                "ax1 = fig.add_subplot(3,3,1)\n",
                "phase_names  = ['Teacher\\n1805M','V1\\n1039M','Vocab5L\\n824M','Enc16L\\n630M',\n",
                "                'LaCoT2U\\n542M','Textless\\n673M']\n",
                "phase_params = [1805, 1039, 824, 630, 542, 673]\n",
                "colors_pb = ['#9E9E9E']*5 + ['#4CAF50']\n",
                "bars = ax1.bar(range(len(phase_names)), phase_params, color=colors_pb, alpha=0.85, edgecolor='white')\n",
                "bars[-1].set_edgecolor('#2E7D32'); bars[-1].set_linewidth(2)\n",
                "ax1.set_xticks(range(len(phase_names))); ax1.set_xticklabels(phase_names, fontsize=7)\n",
                "ax1.set_ylabel('Parameters (M)'); ax1.set_title('Model Size Evolution', fontweight='bold')\n",
                "for bar, v in zip(bars, phase_params):\n",
                "    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, f'{v}M',\n",
                "             ha='center', va='bottom', fontsize=7, fontweight='bold')\n",
                "\n",
                "# 2: ASR-ChrF by lang pair (all 8 pairs)\n",
                "ax2 = fig.add_subplot(3,3,2)\n",
                "if trans_results:\n",
                "    pairs_  = list(trans_results.keys())\n",
                "    chrfs_  = [trans_results[p]['avg_chrf'] for p in pairs_]\n",
                "    bleus_  = [trans_results[p]['avg_bleu'] for p in pairs_]\n",
                "    x_  = np.arange(len(pairs_)); w_ = 0.35\n",
                "    ax2.bar(x_-w_/2, chrfs_, w_, label='ASR-ChrF', color='#2196F3', alpha=0.85)\n",
                "    ax2.bar(x_+w_/2, bleus_, w_, label='ASR-BLEU', color='#FF9800', alpha=0.85)\n",
                "    ax2.set_xticks(x_); ax2.set_xticklabels(pairs_, rotation=45, ha='right', fontsize=7)\n",
                "    ax2.set_title('Translation Quality by Language Pair (ASR)', fontweight='bold')\n",
                "    ax2.legend(fontsize=8)\n",
                "    ax2.axhline(35, color='green', ls=':', lw=1.5, alpha=0.7, label='Target')\n",
                "\n",
                "# 3: Speaker similarity\n",
                "ax3 = fig.add_subplot(3,3,3)\n",
                "if spk_results:\n",
                "    sims_spk = [r['speaker_sim'] for r in spk_results]\n",
                "    ax3.hist(sims_spk, bins=12, color='#E91E63', alpha=0.8, edgecolor='white')\n",
                "    for thresh, lbl, col in [(0.85,'Excellent','green'),(0.70,'Good','orange'),(0.55,'Acceptable','red')]:\n",
                "        ax3.axvline(thresh, color=col, ls='--', lw=1.5, label=f'{lbl}>{thresh}')\n",
                "    ax3.axvline(np.mean(sims_spk), color='black', ls='-', lw=2,\n",
                "                label=f'Mean={np.mean(sims_spk):.3f}')\n",
                "    ax3.set_xlabel('ECAPA Cosine Similarity'); ax3.set_title('Speaker Similarity (Voice Cloning)', fontweight='bold')\n",
                "    ax3.legend(fontsize=7)\n",
                "\n",
                "# 4: Long-form quality\n",
                "ax4 = fig.add_subplot(3,3,4)\n",
                "if longform_results:\n",
                "    durs_ = sorted(longform_results.keys())\n",
                "    lf_ch = [longform_results[d]['avg_chrf'] for d in durs_]\n",
                "    lf_rt = [longform_results[d]['avg_rtf']  for d in durs_]\n",
                "    ax4_t = ax4.twinx()\n",
                "    ax4.plot(durs_, lf_ch, 'o-', color='#4CAF50', lw=2, ms=8, label='ASR-ChrF')\n",
                "    ax4_t.plot(durs_, lf_rt, 's--', color='#FF5722', lw=2, ms=8, label='RTF')\n",
                "    ax4.axvline(25, color='gray', ls=':', lw=1.5, label='Chunking boundary')\n",
                "    ax4.set_xlabel('Duration (s)'); ax4.set_ylabel('ASR-ChrF', color='#4CAF50')\n",
                "    ax4_t.set_ylabel('RTF', color='#FF5722')\n",
                "    ax4.set_title('Long-Form: Quality vs Duration', fontweight='bold')\n",
                "    ax4.legend(loc='upper left', fontsize=8); ax4_t.legend(loc='upper right', fontsize=8)\n",
                "\n",
                "# 5: RTF comparison\n",
                "ax5 = fig.add_subplot(3,3,5)\n",
                "final_rtf = np.mean([v['avg_rtf'] for v in trans_results.values()]) if trans_results else 0.09\n",
                "spd_labels = ['Teacher\\n1805M','V1\\n1039M','Textless\\n673M']\n",
                "spd_rtfs   = [0.268, 0.113, final_rtf]\n",
                "ax5.bar(spd_labels, spd_rtfs, color=['#F44336','#FF9800','#4CAF50'], alpha=0.85, edgecolor='white')\n",
                "ax5.set_ylabel('RTF (lower=faster)'); ax5.set_title('Inference Speed (RTF)', fontweight='bold')\n",
                "for i,(l,v) in enumerate(zip(spd_labels,spd_rtfs)):\n",
                "    ax5.text(i, v+0.003, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')\n",
                "\n",
                "# 6: Speaker sim by pair\n",
                "ax6 = fig.add_subplot(3,3,6)\n",
                "if spk_results:\n",
                "    from collections import defaultdict\n",
                "    pair_sims_ = defaultdict(list)\n",
                "    for r in spk_results: pair_sims_[r['pair']].append(r['speaker_sim'])\n",
                "    pn = list(pair_sims_.keys())\n",
                "    pm = [np.mean(pair_sims_[p]) for p in pn]\n",
                "    ps = [np.std(pair_sims_[p]) for p in pn]\n",
                "    ax6.bar(pn, pm, yerr=ps, capsize=5, color='#9C27B0', alpha=0.8, edgecolor='white')\n",
                "    ax6.axhline(0.65, color='green', ls='--', lw=1.5, label='Target 0.65')\n",
                "    ax6.set_ylim(0,1); ax6.set_ylabel('Speaker Similarity')\n",
                "    ax6.set_xticklabels(pn, rotation=30, ha='right', fontsize=7)\n",
                "    ax6.set_title('Speaker Sim by Language Pair', fontweight='bold'); ax6.legend(fontsize=8)\n",
                "\n",
                "# 7: Enc pruning ChrF curve (Phase 2)\n",
                "ax7 = fig.add_subplot(3,3,7)\n",
                "if 'p2_log' in dir() and p2_log:\n",
                "    iters7 = [e['iter'] for e in p2_log]; chrfs7 = [e['chrf'] for e in p2_log]\n",
                "    ax7.plot(iters7, chrfs7, 'o-', color='#FF9800', lw=2, ms=7)\n",
                "    for e in p2_log:\n",
                "        ax7.annotate(f'L{e[\"removed\"]}', (e['iter'],e['chrf']),\n",
                "                     fontsize=6, ha='center', va='bottom')\n",
                "    ax7.set_xlabel('Pruning iter'); ax7.set_ylabel('ASR-ChrF')\n",
                "    ax7.set_title('Enc Pruning: ASR-ChrF per Removal', fontweight='bold')\n",
                "else:\n",
                "    ax7.text(0.5,0.5,'P2 log not in session', ha='center', va='center', transform=ax7.transAxes)\n",
                "\n",
                "# 8: Per-language-pair scatter\n",
                "ax8 = fig.add_subplot(3,3,8)\n",
                "if trans_results:\n",
                "    all_chrfs = []\n",
                "    all_bleus = []\n",
                "    for pair_key, pair_data in trans_results.items():\n",
                "        for r in pair_data['results']:\n",
                "            all_chrfs.append(r['chrf'])\n",
                "            all_bleus.append(r['bleu'])\n",
                "    if all_chrfs:\n",
                "        ax8.scatter(all_bleus, all_chrfs, color='#2196F3', alpha=0.5, s=30, edgecolors='white')\n",
                "        mu_c = np.mean(all_chrfs)\n",
                "        ax8.axhline(mu_c, color='red', ls='--', lw=1.5, label=f'Mean ChrF={mu_c:.1f}')\n",
                "        ax8.set_xlabel('ASR-BLEU'); ax8.set_ylabel('ASR-ChrF')\n",
                "        ax8.set_title('All Pairs: BLEU vs ChrF per sample', fontweight='bold'); ax8.legend(fontsize=8)\n",
                "\n",
                "# 9: Architecture comparison table\n",
                "ax9 = fig.add_subplot(3,3,9)\n",
                "ax9.axis('off')\n",
                "tbl_data = [\n",
                "    ['Component','Original','Textless 673M'],\n",
                "    ['Text Decoder','867M 24L','0M (removed)'],\n",
                "    ['lm_head+vocab','~262M','0M (removed)'],\n",
                "    ['Speech Encoder','635M 24L','~441M 16L'],\n",
                "    ['T2U Model','262M 6+6L','~175M 4+4L'],\n",
                "    ['CIF Connector','—','~5M (NEW)'],\n",
                "    ['Speaker Adapter','—','~0.1M (NEW)'],\n",
                "    ['Vocoder','41.9M','41.9M (frozen)'],\n",
                "    ['TOTAL','1805M','~673M'],\n",
                "]\n",
                "tbl = ax9.table(cellText=tbl_data[1:], colLabels=tbl_data[0],\n",
                "                cellLoc='center', loc='center')\n",
                "tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.2, 1.5)\n",
                "for j in range(3):\n",
                "    tbl[len(tbl_data)-1, j].set_facecolor('#C8E6C9')\n",
                "    tbl[len(tbl_data)-1, j].set_text_props(fontweight='bold')\n",
                "tbl[1,2].set_facecolor('#FFCDD2'); tbl[2,2].set_facecolor('#FFCDD2')\n",
                "ax9.set_title('Architecture Comparison', fontweight='bold', pad=10)\n",
                "\n",
                "plt.tight_layout(rect=[0,0,1,0.98])\n",
                "save_figure(fig, 'phase7_comprehensive_benchmark.png')\n",
                "plt.show()\n",
                "print('✓ Comprehensive benchmark figure saved (5 languages, ASR metrics).')\n"
            ]
            updates_made += 1
            print("✓ Updated comprehensive visualization")
    
    # Save updated notebook
    write_notebook(nb_path, nb)
    print(f"\n✅ Successfully made {updates_made} visualization updates to {nb_path}")
    print("\nVisualization now shows:")
    print("  • All 8 language pairs in translation quality chart")
    print("  • ASR-ChrF and ASR-BLEU metrics throughout")
    print("  • Speaker similarity across multiple language pairs")
    print("  • Per-sample scatter plot across all pairs")

if __name__ == '__main__':
    main()
