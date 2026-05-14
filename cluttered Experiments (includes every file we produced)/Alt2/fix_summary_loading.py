"""
Fix for Phase Comparison Plots Not Showing All Phases

PROBLEM:
- ALL_SUMMARIES and ALL_DETAILED_SUMMARIES are only loaded once at notebook startup
- If you skip Phase 1-3 cells and jump to Phase 4, plots only show Phase 4 data
- The checkpoint files exist in Drive but aren't being reloaded

SOLUTION:
Add this cell BEFORE each benchmark cell to reload all summaries from checkpoints
"""

# ══════════════════════════════════════════════════════════════════════════════
# RELOAD ALL SUMMARIES FROM CHECKPOINT
# Add this cell BEFORE your Phase 4 (or any phase) benchmark cell
# ══════════════════════════════════════════════════════════════════════════════

print('='*70)
print('  RELOADING ALL SUMMARIES FROM CHECKPOINT')
print('='*70)

# Reload basic summaries
ALL_SUMMARIES = _load_summaries_from_drive()
print(f'\n✓ Loaded {len(ALL_SUMMARIES)} basic summaries:')
for label in sorted(ALL_SUMMARIES.keys()):
    s = ALL_SUMMARIES[label]
    print(f'  {label:<25} ChrF={s.get("avg_chrf",0):>6.2f}  Params={s.get("params_M",0):>6.1f}M')

# Reload detailed summaries
ALL_DETAILED_SUMMARIES = _load_detailed_summaries_from_drive()
print(f'\n✓ Loaded {len(ALL_DETAILED_SUMMARIES)} detailed summaries:')
for label in sorted(ALL_DETAILED_SUMMARIES.keys()):
    print(f'  {label}')

print('\n' + '='*70)
print('  NOW SAFE TO RUN BENCHMARK + PLOTTING')
print('='*70)


# ══════════════════════════════════════════════════════════════════════════════
# BETTER SOLUTION: Modify plotting functions to auto-reload
# Replace your existing plot_phase_comparison() and plot_detailed_phase_comparison()
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase_comparison_fixed(summaries=None, save_name='phase_comparison.png'):
    """
    Fixed version that always reloads from checkpoint before plotting.
    This ensures we have ALL phases, not just the current session's phases.
    """
    global ALL_SUMMARIES
    
    # CRITICAL: Reload from checkpoint to get all phases
    print('[plot] Reloading summaries from checkpoint...')
    ALL_SUMMARIES = _load_summaries_from_drive()
    
    data = summaries or get_summaries()
    if not data: 
        print('No summaries yet.'); 
        return
    
    # Sort by label to ensure consistent ordering
    data = sorted(data, key=lambda s: s['label'])
    labels = [s['label'] for s in data]
    
    print(f'[plot] Plotting {len(data)} phases: {labels}')
    
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
    plt.show()


def plot_detailed_phase_comparison_fixed(save_name='detailed_comparison.png'):
    """
    Fixed version that always reloads from checkpoint before plotting.
    """
    global ALL_DETAILED_SUMMARIES
    
    # CRITICAL: Reload from checkpoint to get all phases
    print('[plot] Reloading detailed summaries from checkpoint...')
    ALL_DETAILED_SUMMARIES = _load_detailed_summaries_from_drive()
    
    summaries = sorted(ALL_DETAILED_SUMMARIES.values(), key=lambda s: s['label'])
    if not summaries: 
        print('No detailed summaries yet.')
        return
    
    print(f'[plot] Plotting detailed comparison for {len(summaries)} phases: {[s["label"] for s in summaries]}')
    
    # ... rest of your plotting code ...


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: Check what's in your checkpoint files
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_checkpoint_summaries():
    """
    Run this to see what's actually saved in your checkpoint files.
    """
    print('\n' + '='*70)
    print('  CHECKPOINT DIAGNOSTIC')
    print('='*70)
    
    # Check basic summaries checkpoint
    ckpt = load_latest_checkpoint('all_summaries')
    if ckpt and 'summaries' in ckpt:
        print(f'\n✓ all_summaries checkpoint found:')
        print(f'  Contains {len(ckpt["summaries"])} summaries')
        for s in ckpt['summaries']:
            print(f'    {s["label"]:<25} ChrF={s.get("avg_chrf",0):>6.2f}  '
                  f'BLEU={s.get("avg_bleu",0):>6.2f}  Params={s.get("params_M",0):>6.1f}M')
    else:
        print('\n✗ all_summaries checkpoint NOT FOUND or empty')
    
    # Check detailed summaries checkpoint
    ckpt_detailed = load_latest_checkpoint('all_detailed_summaries')
    if ckpt_detailed and 'detailed_summaries' in ckpt_detailed:
        print(f'\n✓ all_detailed_summaries checkpoint found:')
        print(f'  Contains {len(ckpt_detailed["detailed_summaries"])} detailed summaries')
        for s in ckpt_detailed['detailed_summaries']:
            print(f'    {s["label"]:<25} {s.get("n_total",0)} samples')
    else:
        print('\n✗ all_detailed_summaries checkpoint NOT FOUND or empty')
    
    # Check individual phase benchmarks
    print(f'\n✓ Individual phase benchmarks:')
    for phase_name in ['phase0_benchmark', 'phase1_benchmark', 'phase2_benchmark', 
                       'phase3_benchmark', 'phase4_benchmark', 'phase5_benchmark']:
        ckpt = load_latest_checkpoint(phase_name)
        if ckpt:
            label = ckpt.get('summary', {}).get('label', '?')
            chrf = ckpt.get('summary', {}).get('avg_chrf', 0)
            print(f'    {phase_name:<25} → {label:<20} ChrF={chrf:>6.2f}')
        else:
            print(f'    {phase_name:<25} → NOT FOUND')
    
    print('\n' + '='*70)


# ══════════════════════════════════════════════════════════════════════════════
# USAGE INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════

print("""
USAGE:

1. RUN DIAGNOSTIC FIRST:
   diagnose_checkpoint_summaries()
   
   This will show you what's actually in your checkpoint files.

2. IF CHECKPOINTS EXIST BUT PLOTS ARE EMPTY:
   Add this cell BEFORE your Phase 4 benchmark:
   
   ALL_SUMMARIES = _load_summaries_from_drive()
   ALL_DETAILED_SUMMARIES = _load_detailed_summaries_from_drive()
   print(f'Reloaded {len(ALL_SUMMARIES)} summaries')

3. BETTER: Replace your plotting functions with the _fixed versions above.
   They auto-reload from checkpoint every time they're called.

4. IF CHECKPOINTS ARE EMPTY:
   You need to re-run Phase 1-3 benchmark cells to regenerate the data.
   The store_summary() and store_detailed_summary() calls will save them.
""")
