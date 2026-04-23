"""
Enhanced Benchmark Tracking for Seamless-Final.ipynb
Adds per-language detailed summaries and comprehensive visualizations

Insert this code after the existing summary functions in the notebook
"""

# ============================================================================
# ENHANCED SUMMARY TRACKING WITH PER-LANGUAGE BREAKDOWN
# ============================================================================

def _load_detailed_summaries_from_drive():
    """Load detailed per-language summaries from checkpoint"""
    ckpt = load_latest_checkpoint('all_detailed_summaries')
    if ckpt and 'detailed_summaries' in ckpt:
        return {s['label']: s for s in ckpt['detailed_summaries']}
    return {}

# Global storage for detailed summaries
ALL_DETAILED_SUMMARIES = _load_detailed_summaries_from_drive()
print(f'Loaded {len(ALL_DETAILED_SUMMARIES)} detailed summaries: {list(ALL_DETAILED_SUMMARIES.keys())}')

def store_detailed_summary(s):
    """Store detailed per-language summary with full breakdown"""
    label = s['label']
    ALL_DETAILED_SUMMARIES[label] = s.copy()
    save_checkpoint({'detailed_summaries': list(ALL_DETAILED_SUMMARIES.values())}, 
                    'all_detailed_summaries', 0)
    print(f'[detailed_summary] Stored {label} ({len(ALL_DETAILED_SUMMARIES)} total)')

def get_detailed_summaries():
    """Get all detailed summaries sorted by label"""
    return sorted(ALL_DETAILED_SUMMARIES.values(), key=lambda s: s['label'])

def compute_detailed_summary(results, label, params_M):
    """
    Compute comprehensive per-language summary from benchmark results
    
    Args:
        results: list of dicts with keys: id, src_lang, tgt_lang, bleu, chrf, rtf, pred, ref
        label: phase label (e.g., 'P0_V1_Baseline')
        params_M: model parameters in millions
    
    Returns:
        dict with overall + per-language-pair metrics
    """
    from collections import defaultdict
    
    # Group by language pair
    by_pair = defaultdict(list)
    for r in results:
        if not math.isnan(r.get('rtf', float('nan'))):
            pair_key = f"{r['src_lang']}→{r['tgt_lang']}"
            by_pair[pair_key].append(r)
    
    # Compute per-pair stats
    pair_stats = {}
    for pair_key, pair_results in by_pair.items():
        pair_stats[pair_key] = {
            'n_samples': len(pair_results),
            'avg_bleu': float(np.mean([r['bleu'] for r in pair_results])),
            'avg_chrf': float(np.mean([r['chrf'] for r in pair_results])),
            'avg_rtf': float(np.mean([r['rtf'] for r in pair_results])),
            'std_bleu': float(np.std([r['bleu'] for r in pair_results])),
            'std_chrf': float(np.std([r['chrf'] for r in pair_results])),
            'std_rtf': float(np.std([r['rtf'] for r in pair_results])),
            'min_bleu': float(np.min([r['bleu'] for r in pair_results])),
            'max_bleu': float(np.max([r['bleu'] for r in pair_results])),
            'min_chrf': float(np.min([r['chrf'] for r in pair_results])),
            'max_chrf': float(np.max([r['chrf'] for r in pair_results])),
        }
    
    # Overall stats
    valid = [r for r in results if not math.isnan(r.get('rtf', float('nan')))]
    
    detailed_summary = {
        'label': label,
        'params_M': params_M,
        'n_total': len(valid),
        'n_pairs': len(by_pair),
        
        # Overall metrics
        'avg_bleu': float(np.mean([r['bleu'] for r in valid])),
        'avg_chrf': float(np.mean([r['chrf'] for r in valid])),
        'avg_rtf': float(np.mean([r['rtf'] for r in valid])),
        'std_bleu': float(np.std([r['bleu'] for r in valid])),
        'std_chrf': float(np.std([r['chrf'] for r in valid])),
        'std_rtf': float(np.std([r['rtf'] for r in valid])),
        
        # Per-pair breakdown
        'pair_stats': pair_stats,
        
        # Language-level aggregation (src→* and *→tgt)
        'by_src_lang': {},
        'by_tgt_lang': {},
    }
    
    # Aggregate by source language
    by_src = defaultdict(list)
    for r in valid:
        by_src[r['src_lang']].append(r)
    for src_lang, src_results in by_src.items():
        detailed_summary['by_src_lang'][src_lang] = {
            'n_samples': len(src_results),
            'avg_bleu': float(np.mean([r['bleu'] for r in src_results])),
            'avg_chrf': float(np.mean([r['chrf'] for r in src_results])),
            'avg_rtf': float(np.mean([r['rtf'] for r in src_results])),
        }
    
    # Aggregate by target language
    by_tgt = defaultdict(list)
    for r in valid:
        by_tgt[r['tgt_lang']].append(r)
    for tgt_lang, tgt_results in by_tgt.items():
        detailed_summary['by_tgt_lang'][tgt_lang] = {
            'n_samples': len(tgt_results),
            'avg_bleu': float(np.mean([r['bleu'] for r in tgt_results])),
            'avg_chrf': float(np.mean([r['chrf'] for r in tgt_results])),
            'avg_rtf': float(np.mean([r['rtf'] for r in tgt_results])),
        }
    
    return detailed_summary


# ============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ============================================================================

def plot_detailed_phase_comparison(save_name='detailed_phase_comparison.png'):
    """
    Comprehensive multi-panel comparison showing:
    - Overall metrics across phases
    - Per-language-pair breakdown for each phase
    - Language-specific trends
    """
    summaries = get_detailed_summaries()
    if not summaries:
        print('No detailed summaries available yet.')
        return
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Detailed Phase Comparison: Per-Language Breakdown', 
                 fontsize=16, fontweight='bold')
    
    # Panel 1: Overall metrics evolution
    ax1 = plt.subplot(3, 3, 1)
    labels = [s['label'] for s in summaries]
    chrfs = [s['avg_chrf'] for s in summaries]
    bleus = [s['avg_bleu'] for s in summaries]
    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w/2, chrfs, w, label='ASR-ChrF', color='#4CAF50', alpha=0.85)
    ax1.bar(x + w/2, bleus, w, label='ASR-BLEU', color='#2196F3', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax1.set_title('Overall Quality Evolution', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Panel 2: Parameters vs Quality
    ax2 = plt.subplot(3, 3, 2)
    params = [s['params_M'] for s in summaries]
    ax2.scatter(params, chrfs, s=120, c='#4CAF50', marker='o', label='ChrF', zorder=5)
    ax2.scatter(params, bleus, s=120, c='#2196F3', marker='s', label='BLEU', zorder=5)
    for i, lbl in enumerate(labels):
        ax2.annotate(lbl, (params[i], chrfs[i]), fontsize=7, 
                    xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Parameters (M)')
    ax2.set_ylabel('Score')
    ax2.set_title('Size vs Quality Trade-off', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Panel 3: RTF evolution
    ax3 = plt.subplot(3, 3, 3)
    rtfs = [s['avg_rtf'] for s in summaries]
    bars = ax3.bar(labels, rtfs, color='#FF9800', alpha=0.85, edgecolor='white')
    ax3.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax3.set_ylabel('RTF (lower=faster)')
    ax3.set_title('Inference Speed Evolution', fontweight='bold')
    for bar, v in zip(bars, rtfs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax3.grid(alpha=0.3)
    
    # Panel 4-6: Per-language-pair heatmaps for latest phase
    if summaries:
        latest = summaries[-1]
        pair_stats = latest.get('pair_stats', {})
        
        if pair_stats:
            pairs = sorted(pair_stats.keys())
            
            # Panel 4: ChrF heatmap
            ax4 = plt.subplot(3, 3, 4)
            chrf_vals = [pair_stats[p]['avg_chrf'] for p in pairs]
            bars4 = ax4.barh(pairs, chrf_vals, color='#4CAF50', alpha=0.85)
            ax4.set_xlabel('ASR-ChrF')
            ax4.set_title(f'{latest["label"]}: ChrF by Pair', fontweight='bold')
            for bar, v in zip(bars4, chrf_vals):
                ax4.text(v, bar.get_y() + bar.get_height()/2, f'{v:.1f}',
                        va='center', ha='left', fontsize=7)
            ax4.grid(alpha=0.3, axis='x')
            
            # Panel 5: BLEU heatmap
            ax5 = plt.subplot(3, 3, 5)
            bleu_vals = [pair_stats[p]['avg_bleu'] for p in pairs]
            bars5 = ax5.barh(pairs, bleu_vals, color='#2196F3', alpha=0.85)
            ax5.set_xlabel('ASR-BLEU')
            ax5.set_title(f'{latest["label"]}: BLEU by Pair', fontweight='bold')
            for bar, v in zip(bars5, bleu_vals):
                ax5.text(v, bar.get_y() + bar.get_height()/2, f'{v:.1f}',
                        va='center', ha='left', fontsize=7)
            ax5.grid(alpha=0.3, axis='x')
            
            # Panel 6: RTF by pair
            ax6 = plt.subplot(3, 3, 6)
            rtf_vals = [pair_stats[p]['avg_rtf'] for p in pairs]
            bars6 = ax6.barh(pairs, rtf_vals, color='#FF9800', alpha=0.85)
            ax6.set_xlabel('RTF')
            ax6.set_title(f'{latest["label"]}: Speed by Pair', fontweight='bold')
            for bar, v in zip(bars6, rtf_vals):
                ax6.text(v, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
                        va='center', ha='left', fontsize=7)
            ax6.grid(alpha=0.3, axis='x')
    
    # Panel 7: Source language performance across phases
    ax7 = plt.subplot(3, 3, 7)
    if summaries and 'by_src_lang' in summaries[-1]:
        src_langs = sorted(summaries[-1]['by_src_lang'].keys())
        for src in src_langs:
            src_chrfs = []
            for s in summaries:
                if 'by_src_lang' in s and src in s['by_src_lang']:
                    src_chrfs.append(s['by_src_lang'][src]['avg_chrf'])
                else:
                    src_chrfs.append(None)
            # Plot only non-None values
            valid_x = [i for i, v in enumerate(src_chrfs) if v is not None]
            valid_y = [v for v in src_chrfs if v is not None]
            if valid_y:
                ax7.plot(valid_x, valid_y, 'o-', label=src.upper(), lw=2, ms=6)
        ax7.set_xticks(range(len(labels)))
        ax7.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
        ax7.set_ylabel('ASR-ChrF')
        ax7.set_title('Source Language Trends', fontweight='bold')
        ax7.legend(fontsize=7, ncol=2)
        ax7.grid(alpha=0.3)
    
    # Panel 8: Target language performance across phases
    ax8 = plt.subplot(3, 3, 8)
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
                ax8.plot(valid_x, valid_y, 's-', label=tgt.upper(), lw=2, ms=6)
        ax8.set_xticks(range(len(labels)))
        ax8.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
        ax8.set_ylabel('ASR-ChrF')
        ax8.set_title('Target Language Trends', fontweight='bold')
        ax8.legend(fontsize=7, ncol=2)
        ax8.grid(alpha=0.3)
    
    # Panel 9: Variance analysis
    ax9 = plt.subplot(3, 3, 9)
    if summaries:
        std_chrfs = [s.get('std_chrf', 0) for s in summaries]
        std_bleus = [s.get('std_bleu', 0) for s in summaries]
        x = np.arange(len(labels))
        w = 0.35
        ax9.bar(x - w/2, std_chrfs, w, label='ChrF σ', color='#4CAF50', alpha=0.6)
        ax9.bar(x + w/2, std_bleus, w, label='BLEU σ', color='#2196F3', alpha=0.6)
        ax9.set_xticks(x)
        ax9.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax9.set_ylabel('Standard Deviation')
        ax9.set_title('Quality Variance Across Samples', fontweight='bold')
        ax9.legend()
        ax9.grid(alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
    print(f'✓ Detailed phase comparison saved: {save_name}')


def plot_language_pair_matrix(phase_label=None, save_name='language_pair_matrix.png'):
    """
    Create a matrix visualization showing all language pair scores
    for a specific phase (or latest if None)
    """
    summaries = get_detailed_summaries()
    if not summaries:
        print('No detailed summaries available.')
        return
    
    # Select phase
    if phase_label:
        summary = next((s for s in summaries if s['label'] == phase_label), None)
        if not summary:
            print(f'Phase {phase_label} not found.')
            return
    else:
        summary = summaries[-1]
    
    pair_stats = summary.get('pair_stats', {})
    if not pair_stats:
        print('No pair stats available.')
        return
    
    # Extract unique source and target languages
    pairs = list(pair_stats.keys())
    src_langs = sorted(set(p.split('→')[0] for p in pairs))
    tgt_langs = sorted(set(p.split('→')[1] for p in pairs))
    
    # Create matrices
    chrf_matrix = np.zeros((len(src_langs), len(tgt_langs)))
    bleu_matrix = np.zeros((len(src_langs), len(tgt_langs)))
    
    for pair, stats in pair_stats.items():
        src, tgt = pair.split('→')
        i = src_langs.index(src)
        j = tgt_langs.index(tgt)
        chrf_matrix[i, j] = stats['avg_chrf']
        bleu_matrix[i, j] = stats['avg_bleu']
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{summary["label"]}: Language Pair Performance Matrix',
                 fontsize=14, fontweight='bold')
    
    # ChrF heatmap
    im1 = ax1.imshow(chrf_matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(range(len(tgt_langs)))
    ax1.set_yticks(range(len(src_langs)))
    ax1.set_xticklabels([l.upper() for l in tgt_langs], fontsize=10)
    ax1.set_yticklabels([l.upper() for l in src_langs], fontsize=10)
    ax1.set_xlabel('Target Language', fontweight='bold')
    ax1.set_ylabel('Source Language', fontweight='bold')
    ax1.set_title('ASR-ChrF Scores', fontweight='bold')
    
    # Add values to cells
    for i in range(len(src_langs)):
        for j in range(len(tgt_langs)):
            val = chrf_matrix[i, j]
            if val > 0:
                ax1.text(j, i, f'{val:.1f}', ha='center', va='center',
                        color='white' if val > 50 else 'black', fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, label='ChrF Score')
    
    # BLEU heatmap
    im2 = ax2.imshow(bleu_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(range(len(tgt_langs)))
    ax2.set_yticks(range(len(src_langs)))
    ax2.set_xticklabels([l.upper() for l in tgt_langs], fontsize=10)
    ax2.set_yticklabels([l.upper() for l in src_langs], fontsize=10)
    ax2.set_xlabel('Target Language', fontweight='bold')
    ax2.set_ylabel('Source Language', fontweight='bold')
    ax2.set_title('ASR-BLEU Scores', fontweight='bold')
    
    for i in range(len(src_langs)):
        for j in range(len(tgt_langs)):
            val = bleu_matrix[i, j]
            if val > 0:
                ax2.text(j, i, f'{val:.1f}', ha='center', va='center',
                        color='white' if val > 50 else 'black', fontweight='bold')
    
    plt.colorbar(im2, ax=ax2, label='BLEU Score')
    
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
    print(f'✓ Language pair matrix saved: {save_name}')


def print_detailed_summary_table(phase_label=None):
    """Print comprehensive text table of per-language results"""
    summaries = get_detailed_summaries()
    if not summaries:
        print('No detailed summaries available.')
        return
    
    if phase_label:
        summary = next((s for s in summaries if s['label'] == phase_label), None)
        if not summary:
            print(f'Phase {phase_label} not found.')
            return
    else:
        summary = summaries[-1]
    
    print('\n' + '='*90)
    print(f'  DETAILED RESULTS: {summary["label"]}')
    print(f'  Parameters: {summary["params_M"]:.1f}M | Total Samples: {summary["n_total"]}')
    print('='*90)
    
    # Overall stats
    print(f'\n[Overall Metrics]')
    print(f'  ASR-ChrF: {summary["avg_chrf"]:.2f} ± {summary.get("std_chrf", 0):.2f}')
    print(f'  ASR-BLEU: {summary["avg_bleu"]:.2f} ± {summary.get("std_bleu", 0):.2f}')
    print(f'  RTF:      {summary["avg_rtf"]:.4f} ± {summary.get("std_rtf", 0):.4f}')
    
    # Per-pair breakdown
    pair_stats = summary.get('pair_stats', {})
    if pair_stats:
        print(f'\n[Per-Language-Pair Breakdown] ({len(pair_stats)} pairs)')
        print(f'  {"Pair":<15} {"N":>4} {"ChrF":>8} {"BLEU":>8} {"RTF":>8} {"ChrF σ":>8}')
        print('  ' + '-'*70)
        for pair in sorted(pair_stats.keys()):
            stats = pair_stats[pair]
            print(f'  {pair:<15} {stats["n_samples"]:>4} '
                  f'{stats["avg_chrf"]:>8.2f} {stats["avg_bleu"]:>8.2f} '
                  f'{stats["avg_rtf"]:>8.4f} {stats["std_chrf"]:>8.2f}')
    
    # By source language
    by_src = summary.get('by_src_lang', {})
    if by_src:
        print(f'\n[By Source Language]')
        print(f'  {"Lang":>6} {"N":>4} {"ChrF":>8} {"BLEU":>8} {"RTF":>8}')
        print('  ' + '-'*40)
        for lang in sorted(by_src.keys()):
            stats = by_src[lang]
            print(f'  {lang.upper():>6} {stats["n_samples"]:>4} '
                  f'{stats["avg_chrf"]:>8.2f} {stats["avg_bleu"]:>8.2f} '
                  f'{stats["avg_rtf"]:>8.4f}')
    
    # By target language
    by_tgt = summary.get('by_tgt_lang', {})
    if by_tgt:
        print(f'\n[By Target Language]')
        print(f'  {"Lang":>6} {"N":>4} {"ChrF":>8} {"BLEU":>8} {"RTF":>8}')
        print('  ' + '-'*40)
        for lang in sorted(by_tgt.keys()):
            stats = by_tgt[lang]
            print(f'  {lang.upper():>6} {stats["n_samples"]:>4} '
                  f'{stats["avg_chrf"]:>8.2f} {stats["avg_bleu"]:>8.2f} '
                  f'{stats["avg_rtf"]:>8.4f}')
    
    print('='*90 + '\n')


# ============================================================================
# USAGE EXAMPLE - Replace existing benchmark calls with this pattern
# ============================================================================

def run_benchmark_with_detailed_tracking(mdl, samples, label='model', save_n=4):
    """
    Enhanced benchmark that captures both simple and detailed summaries
    
    Usage:
        results, summary, detailed_summary = run_benchmark_with_detailed_tracking(
            model_p0, eval_samples, label='P0_V1_Baseline', save_n=4)
        store_summary(summary)
        store_detailed_summary(detailed_summary)
    """
    # Run standard benchmark
    results, summary = run_benchmark_asr(mdl, samples, label, save_n)
    
    # Compute detailed summary
    detailed_summary = compute_detailed_summary(results, label, summary['params_M'])
    
    return results, summary, detailed_summary


print('\n✓ Enhanced benchmark tracking functions loaded.')
print('  - store_detailed_summary()')
print('  - compute_detailed_summary()')
print('  - plot_detailed_phase_comparison()')
print('  - plot_language_pair_matrix()')
print('  - print_detailed_summary_table()')
print('  - run_benchmark_with_detailed_tracking()')
