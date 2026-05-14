"""
QUICK INTEGRATION SNIPPET
Copy-paste this entire cell into seamless-final.ipynb after the existing summary functions
This is a condensed version for immediate use
"""

# ============================================================================
# ENHANCED TRACKING - Insert after ALL_SUMMARIES definition
# ============================================================================

def _load_detailed_summaries_from_drive():
    ckpt = load_latest_checkpoint('all_detailed_summaries')
    if ckpt and 'detailed_summaries' in ckpt:
        return {s['label']: s for s in ckpt['detailed_summaries']}
    return {}

ALL_DETAILED_SUMMARIES = _load_detailed_summaries_from_drive()
print(f'Loaded {len(ALL_DETAILED_SUMMARIES)} detailed summaries')

def store_detailed_summary(s):
    label = s['label']
    ALL_DETAILED_SUMMARIES[label] = s.copy()
    save_checkpoint({'detailed_summaries': list(ALL_DETAILED_SUMMARIES.values())}, 
                    'all_detailed_summaries', 0)
    print(f'[detailed] Stored {label}')

def compute_detailed_summary(results, label, params_M):
    from collections import defaultdict
    by_pair = defaultdict(list)
    for r in results:
        if not math.isnan(r.get('rtf', float('nan'))):
            by_pair[f"{r['src_lang']}→{r['tgt_lang']}"].append(r)
    
    pair_stats = {}
    for pair_key, pair_results in by_pair.items():
        pair_stats[pair_key] = {
            'n_samples': len(pair_results),
            'avg_bleu': float(np.mean([r['bleu'] for r in pair_results])),
            'avg_chrf': float(np.mean([r['chrf'] for r in pair_results])),
            'avg_rtf': float(np.mean([r['rtf'] for r in pair_results])),
            'std_chrf': float(np.std([r['chrf'] for r in pair_results])),
        }
    
    valid = [r for r in results if not math.isnan(r.get('rtf', float('nan')))]
    by_src = defaultdict(list)
    by_tgt = defaultdict(list)
    for r in valid:
        by_src[r['src_lang']].append(r)
        by_tgt[r['tgt_lang']].append(r)
    
    return {
        'label': label, 'params_M': params_M, 'n_total': len(valid),
        'avg_bleu': float(np.mean([r['bleu'] for r in valid])),
        'avg_chrf': float(np.mean([r['chrf'] for r in valid])),
        'avg_rtf': float(np.mean([r['rtf'] for r in valid])),
        'std_chrf': float(np.std([r['chrf'] for r in valid])),
        'pair_stats': pair_stats,
        'by_src_lang': {lang: {
            'n_samples': len(rs),
            'avg_chrf': float(np.mean([r['chrf'] for r in rs])),
            'avg_bleu': float(np.mean([r['bleu'] for r in rs])),
        } for lang, rs in by_src.items()},
        'by_tgt_lang': {lang: {
            'n_samples': len(rs),
            'avg_chrf': float(np.mean([r['chrf'] for r in rs])),
            'avg_bleu': float(np.mean([r['bleu'] for r in rs])),
        } for lang, rs in by_tgt.items()},
    }

def plot_detailed_phase_comparison(save_name='detailed_comparison.png'):
    summaries = sorted(ALL_DETAILED_SUMMARIES.values(), key=lambda s: s['label'])
    if not summaries: return
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Detailed Phase Comparison: Per-Language Breakdown', fontsize=14, fontweight='bold')
    
    labels = [s['label'] for s in summaries]
    
    # Panel 1: Overall ChrF/BLEU
    ax1 = plt.subplot(3, 3, 1)
    chrfs = [s['avg_chrf'] for s in summaries]
    bleus = [s['avg_bleu'] for s in summaries]
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, chrfs, 0.4, label='ChrF', color='#4CAF50', alpha=0.85)
    ax1.bar(x + 0.2, bleus, 0.4, label='BLEU', color='#2196F3', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax1.set_title('Overall Quality', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Panel 2: Per-pair for latest phase
    if summaries:
        latest = summaries[-1]
        pair_stats = latest.get('pair_stats', {})
        if pair_stats:
            ax2 = plt.subplot(3, 3, 2)
            pairs = sorted(pair_stats.keys())
            chrf_vals = [pair_stats[p]['avg_chrf'] for p in pairs]
            ax2.barh(pairs, chrf_vals, color='#4CAF50', alpha=0.85)
            ax2.set_xlabel('ASR-ChrF')
            ax2.set_title(f'{latest["label"]}: ChrF by Pair', fontweight='bold', fontsize=9)
            for i, (p, v) in enumerate(zip(pairs, chrf_vals)):
                ax2.text(v, i, f'{v:.1f}', va='center', ha='left', fontsize=6)
            ax2.grid(alpha=0.3, axis='x')
            
            # Panel 3: BLEU by pair
            ax3 = plt.subplot(3, 3, 3)
            bleu_vals = [pair_stats[p]['avg_bleu'] for p in pairs]
            ax3.barh(pairs, bleu_vals, color='#2196F3', alpha=0.85)
            ax3.set_xlabel('ASR-BLEU')
            ax3.set_title(f'{latest["label"]}: BLEU by Pair', fontweight='bold', fontsize=9)
            for i, (p, v) in enumerate(zip(pairs, bleu_vals)):
                ax3.text(v, i, f'{v:.1f}', va='center', ha='left', fontsize=6)
            ax3.grid(alpha=0.3, axis='x')
    
    # Panel 4: Source language trends
    ax4 = plt.subplot(3, 3, 4)
    if summaries and 'by_src_lang' in summaries[-1]:
        src_langs = sorted(summaries[-1]['by_src_lang'].keys())
        for src in src_langs:
            src_chrfs = []
            for s in summaries:
                if 'by_src_lang' in s and src in s['by_src_lang']:
                    src_chrfs.append(s['by_src_lang'][src]['avg_chrf'])
                else:
                    src_chrfs.append(None)
            valid_x = [i for i, v in enumerate(src_chrfs) if v is not None]
            valid_y = [v for v in src_chrfs if v is not None]
            if valid_y:
                ax4.plot(valid_x, valid_y, 'o-', label=src.upper(), lw=2, ms=5)
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=30, ha='right', fontsize=6)
        ax4.set_ylabel('ASR-ChrF')
        ax4.set_title('Source Language Trends', fontweight='bold', fontsize=9)
        ax4.legend(fontsize=6, ncol=2)
        ax4.grid(alpha=0.3)
    
    # Panel 5: Target language trends
    ax5 = plt.subplot(3, 3, 5)
    if summaries and 'by_tgt_lang' in summaries[-1]:
        tgt_langs = sorted(summaries[-1]['by_tgt_lang'].keys())
        for tgt in tgt_langs:
            tgt_chrfs = []
            for s in summaries:
                if 'by_tgt_lang' in s and tgt in s['by_tgt_lang']:
                    tgt_chrfs.append(s['by_tgt_lang'][tgt]['avg_chrf'])
                else:
                    tgt_chrfs.append(None)
            valid_x = [i for i, v in enumerate(tgt_chrfs) if v is not None]
            valid_y = [v for v in tgt_chrfs if v is not None]
            if valid_y:
                ax5.plot(valid_x, valid_y, 's-', label=tgt.upper(), lw=2, ms=5)
        ax5.set_xticks(range(len(labels)))
        ax5.set_xticklabels(labels, rotation=30, ha='right', fontsize=6)
        ax5.set_ylabel('ASR-ChrF')
        ax5.set_title('Target Language Trends', fontweight='bold', fontsize=9)
        ax5.legend(fontsize=6, ncol=2)
        ax5.grid(alpha=0.3)
    
    # Panel 6: Params vs Quality
    ax6 = plt.subplot(3, 3, 6)
    params = [s['params_M'] for s in summaries]
    ax6.scatter(params, chrfs, s=100, c='#4CAF50', marker='o', label='ChrF', zorder=5)
    ax6.scatter(params, bleus, s=100, c='#2196F3', marker='s', label='BLEU', zorder=5)
    ax6.set_xlabel('Parameters (M)')
    ax6.set_ylabel('Score')
    ax6.set_title('Size vs Quality', fontweight='bold', fontsize=9)
    ax6.legend(fontsize=7)
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()

def print_detailed_summary_table(phase_label=None):
    summaries = sorted(ALL_DETAILED_SUMMARIES.values(), key=lambda s: s['label'])
    if not summaries: return
    summary = next((s for s in summaries if s['label'] == phase_label), summaries[-1]) if phase_label else summaries[-1]
    
    print(f'\n{"="*80}\n  {summary["label"]} - {summary["params_M"]:.1f}M params\n{"="*80}')
    print(f'Overall: ChrF={summary["avg_chrf"]:.2f}±{summary.get("std_chrf",0):.2f}  '
          f'BLEU={summary["avg_bleu"]:.2f}  RTF={summary["avg_rtf"]:.4f}')
    
    pair_stats = summary.get('pair_stats', {})
    if pair_stats:
        print(f'\nPer-Pair ({len(pair_stats)} pairs):')
        print(f'  {"Pair":<15} {"N":>4} {"ChrF":>8} {"BLEU":>8} {"RTF":>8}')
        for pair in sorted(pair_stats.keys()):
            s = pair_stats[pair]
            print(f'  {pair:<15} {s["n_samples"]:>4} {s["avg_chrf"]:>8.2f} '
                  f'{s["avg_bleu"]:>8.2f} {s["avg_rtf"]:>8.4f}')
    
    by_src = summary.get('by_src_lang', {})
    if by_src:
        print(f'\nBy Source Language:')
        for lang in sorted(by_src.keys()):
            s = by_src[lang]
            print(f'  {lang.upper():>6}: ChrF={s["avg_chrf"]:>6.2f}  BLEU={s["avg_bleu"]:>6.2f}  (n={s["n_samples"]})')
    
    by_tgt = summary.get('by_tgt_lang', {})
    if by_tgt:
        print(f'\nBy Target Language:')
        for lang in sorted(by_tgt.keys()):
            s = by_tgt[lang]
            print(f'  {lang.upper():>6}: ChrF={s["avg_chrf"]:>6.2f}  BLEU={s["avg_bleu"]:>6.2f}  (n={s["n_samples"]})')
    print('='*80)

print('✓ Enhanced tracking loaded: store_detailed_summary(), compute_detailed_summary(), plot_detailed_phase_comparison(), print_detailed_summary_table()')
