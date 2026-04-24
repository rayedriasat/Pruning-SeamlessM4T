#!/usr/bin/env python3
"""
Fix plotting functions in seamless-final.ipynb to show all phases correctly
"""

import json
import sys

# Read the notebook
with open('Alteration/seamless-final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New improved plotting functions
new_plot_phase_comparison = '''def plot_phase_comparison(summaries=None, save_name='phase_comparison.png'):
    data = summaries or get_summaries()
    if not data: 
        print('No summaries yet.'); 
        return
    
    # Sort by label to ensure consistent ordering
    data = sorted(data, key=lambda s: s['label'])
    labels = [s['label'] for s in data]
    
    print(f'Plotting {len(data)} phases: {labels}')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Textless S2ST Compression Pipeline: Phase Comparison',
                 fontsize=15, fontweight='bold')
    metrics = [('avg_bleu', 'ASR-BLEU (higher=better)', '#2196F3'),
               ('avg_chrf', 'ASR-ChrF (higher=better)', '#4CAF50'),
               ('avg_rtf',  'RTF (lower=faster)',        '#FF9800'),
               ('params_M', 'Parameters (M)',            '#9C27B0')]
    
    for ax, (key, title, color) in zip(axes.flat, metrics):
        vals = [s.get(key, 0) for s in data]
        x_pos = range(len(labels))
        bars = ax.bar(x_pos, vals, color=color, alpha=0.85, edgecolor='white', width=0.7)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, v in zip(bars, vals):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height, 
                       f'{v:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()'''

new_plot_detailed = '''def plot_detailed_phase_comparison(save_name='detailed_comparison.png'):
    summaries = sorted(ALL_DETAILED_SUMMARIES.values(), key=lambda s: s['label'])
    if not summaries: 
        print('No detailed summaries yet.')
        return
    
    print(f'Plotting detailed comparison for {len(summaries)} phases: {[s["label"] for s in summaries]}')
    
    fig = plt.figure(figsize=(20, 14))
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
    
    # Panel 2: Per-pair ChrF comparison across ALL phases (vertical bars, grouped by language pair)
    ax2 = plt.subplot(3, 3, 2)
    
    # Collect all unique language pairs across all phases
    all_pairs = set()
    for s in summaries:
        if 'pair_stats' in s:
            all_pairs.update(s['pair_stats'].keys())
    all_pairs = sorted(all_pairs)
    
    if all_pairs:
        n_pairs = len(all_pairs)
        n_phases = len(summaries)
        bar_width = 0.8 / n_phases
        x_pos = np.arange(n_pairs)
        
        for phase_idx, s in enumerate(summaries):
            pair_stats = s.get('pair_stats', {})
            chrf_vals = [pair_stats.get(pair, {}).get('avg_chrf', 0) for pair in all_pairs]
            offset = (phase_idx - n_phases/2 + 0.5) * bar_width
            ax2.bar(x_pos + offset, chrf_vals, bar_width, 
                   label=s['label'], alpha=0.85)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(all_pairs, rotation=45, ha='right', fontsize=6)
        ax2.set_ylabel('ASR-ChrF')
        ax2.set_title('ChrF by Language Pair (All Phases)', fontweight='bold', fontsize=9)
        ax2.legend(fontsize=6, ncol=2)
        ax2.grid(alpha=0.3, axis='y')
    
    # Panel 3: BLEU by pair for all phases
    ax3 = plt.subplot(3, 3, 3)
    if all_pairs:
        for phase_idx, s in enumerate(summaries):
            pair_stats = s.get('pair_stats', {})
            bleu_vals = [pair_stats.get(pair, {}).get('avg_bleu', 0) for pair in all_pairs]
            offset = (phase_idx - n_phases/2 + 0.5) * bar_width
            ax3.bar(x_pos + offset, bleu_vals, bar_width, 
                   label=s['label'], alpha=0.85)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(all_pairs, rotation=45, ha='right', fontsize=6)
        ax3.set_ylabel('ASR-BLEU')
        ax3.set_title('BLEU by Language Pair (All Phases)', fontweight='bold', fontsize=9)
        ax3.legend(fontsize=6, ncol=2)
        ax3.grid(alpha=0.3, axis='y')
    
    # Panel 4: Source language trends
    ax4 = plt.subplot(3, 3, 4)
    if summaries and 'by_src_lang' in summaries[0]:
        # Get all source languages
        all_src_langs = set()
        for s in summaries:
            if 'by_src_lang' in s:
                all_src_langs.update(s['by_src_lang'].keys())
        all_src_langs = sorted(all_src_langs)
        
        for src in all_src_langs:
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
    if summaries and 'by_tgt_lang' in summaries[0]:
        all_tgt_langs = set()
        for s in summaries:
            if 'by_tgt_lang' in s:
                all_tgt_langs.update(s['by_tgt_lang'].keys())
        all_tgt_langs = sorted(all_tgt_langs)
        
        for tgt in all_tgt_langs:
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
    for i, lbl in enumerate(labels):
        ax6.annotate(lbl, (params[i], chrfs[i]), fontsize=6, xytext=(3,3),
                    textcoords='offset points')
    ax6.set_xlabel('Parameters (M)')
    ax6.set_ylabel('Score')
    ax6.set_title('Size vs Quality', fontweight='bold', fontsize=9)
    ax6.legend(fontsize=7)
    ax6.grid(alpha=0.3)
    
    # Panel 7: Speaker sim by pair (if available)
    ax7 = plt.subplot(3, 3, 7)
    ax7.text(0.5, 0.5, 'Reserved for\\nSpeaker Similarity', 
            ha='center', va='center', transform=ax7.transAxes, fontsize=10)
    ax7.axis('off')
    
    # Panel 8: RTF comparison
    ax8 = plt.subplot(3, 3, 8)
    rtfs = [s['avg_rtf'] for s in summaries]
    bars = ax8.bar(range(len(labels)), rtfs, color='#FF9800', alpha=0.85, edgecolor='white')
    ax8.set_xticks(range(len(labels)))
    ax8.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax8.set_ylabel('RTF (lower=faster)')
    ax8.set_title('Inference Speed', fontweight='bold', fontsize=9)
    ax8.grid(alpha=0.3, axis='y')
    for bar, v in zip(bars, rtfs):
        if v > 0:
            ax8.text(bar.get_x()+bar.get_width()/2, bar.get_height(), 
                    f'{v:.3f}', ha='center', va='bottom', fontsize=6)
    
    # Panel 9: Summary table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    table_data = [['Phase', 'Params(M)', 'ChrF', 'BLEU', 'RTF']]
    for s in summaries:
        table_data.append([
            s['label'][:12],
            f"{s['params_M']:.0f}",
            f"{s['avg_chrf']:.1f}",
            f"{s['avg_bleu']:.1f}",
            f"{s['avg_rtf']:.3f}"
        ])
    tbl = ax9.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.5)
    ax9.set_title('Summary Table', fontweight='bold', fontsize=9, pad=10)
    
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
    print(f'✓ Detailed comparison plotted for {len(summaries)} phases')'''

# Find and replace the functions in the notebook cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Replace plot_phase_comparison
        if 'def plot_phase_comparison(' in source:
            # Find the function and replace it
            lines = source.split('\n')
            new_lines = []
            in_function = False
            indent_level = 0
            
            for line in lines:
                if 'def plot_phase_comparison(' in line:
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                    new_lines.extend(new_plot_phase_comparison.split('\n'))
                    continue
                
                if in_function:
                    current_indent = len(line) - len(line.lstrip())
                    # Check if we've exited the function
                    if line.strip() and current_indent <= indent_level and not line.strip().startswith('#'):
                        in_function = False
                        new_lines.append(line)
                    elif not line.strip():  # Keep blank lines within function
                        continue
                else:
                    new_lines.append(line)
            
            cell['source'] = '\n'.join(new_lines)
        
        # Replace plot_detailed_phase_comparison
        if 'def plot_detailed_phase_comparison(' in source:
            lines = source.split('\n')
            new_lines = []
            in_function = False
            indent_level = 0
            
            for line in lines:
                if 'def plot_detailed_phase_comparison(' in line:
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                    new_lines.extend(new_plot_detailed.split('\n'))
                    continue
                
                if in_function:
                    current_indent = len(line) - len(line.lstrip())
                    if line.strip() and current_indent <= indent_level and not line.strip().startswith('#'):
                        in_function = False
                        new_lines.append(line)
                    elif not line.strip():
                        continue
                else:
                    new_lines.append(line)
            
            cell['source'] = '\n'.join(new_lines)

# Write the updated notebook
with open('Alteration/seamless-final.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Fixed plotting functions in seamless-final.ipynb")
print("  - plot_phase_comparison: Now shows ALL phases with debug output")
print("  - plot_detailed_phase_comparison: Panel 2 now shows vertical bars grouped by language pair across all phases")
