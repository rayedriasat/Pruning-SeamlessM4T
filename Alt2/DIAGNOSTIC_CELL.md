# DIAGNOSTIC: Why Are My Plots Only Showing Phase 4?

## Quick Answer

Your `ALL_SUMMARIES` and `ALL_DETAILED_SUMMARIES` dictionaries are only loaded **once** at notebook startup. If you:

1. Run Phase 1-3 benchmarks → summaries saved to checkpoint ✅
2. Restart kernel or skip to Phase 4 → `ALL_SUMMARIES` is empty at startup ❌
3. Run Phase 4 benchmark → only Phase 4 data in memory ❌
4. Plot → only shows Phase 4 ❌

## The Fix

### Option 1: Quick Fix (Add Before Phase 4 Benchmark)

```python
# ══════════════════════════════════════════════════════════════════════════════
# RELOAD ALL SUMMARIES FROM CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

print('Reloading all summaries from checkpoint...')
ALL_SUMMARIES = _load_summaries_from_drive()
ALL_DETAILED_SUMMARIES = _load_detailed_summaries_from_drive()

print(f'✓ Loaded {len(ALL_SUMMARIES)} summaries: {list(ALL_SUMMARIES.keys())}')
print(f'✓ Loaded {len(ALL_DETAILED_SUMMARIES)} detailed summaries')

# Now run your Phase 4 benchmark...
```

### Option 2: Permanent Fix (Modify Plotting Functions)

Replace your `plot_phase_comparison()` function with this:

```python
def plot_phase_comparison(summaries=None, save_name='phase_comparison.png'):
    # CRITICAL: Always reload from checkpoint to ensure we have ALL phases
    global ALL_SUMMARIES
    ALL_SUMMARIES = _load_summaries_from_drive()
    
    data = summaries or get_summaries()
    if not data: 
        print('No summaries yet.'); 
        return
    
    # ... rest of your existing code ...
```

Do the same for `plot_detailed_phase_comparison()`:

```python
def plot_detailed_phase_comparison(save_name='detailed_comparison.png'):
    # CRITICAL: Always reload from checkpoint
    global ALL_DETAILED_SUMMARIES
    ALL_DETAILED_SUMMARIES = _load_detailed_summaries_from_drive()
    
    summaries = sorted(ALL_DETAILED_SUMMARIES.values(), key=lambda s: s['label'])
    # ... rest of your existing code ...
```

## Diagnostic: Check What's Actually Saved

Run this to see what's in your checkpoint files:

```python
# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: What's in my checkpoints?
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('  CHECKPOINT DIAGNOSTIC')
print('='*70)

# Check basic summaries
ckpt = load_latest_checkpoint('all_summaries')
if ckpt and 'summaries' in ckpt:
    print(f'\n✓ all_summaries checkpoint: {len(ckpt["summaries"])} summaries')
    for s in ckpt['summaries']:
        print(f'  {s["label"]:<25} ChrF={s.get("avg_chrf",0):>6.2f}  '
              f'Params={s.get("params_M",0):>6.1f}M')
else:
    print('\n✗ all_summaries checkpoint NOT FOUND')

# Check detailed summaries
ckpt_detailed = load_latest_checkpoint('all_detailed_summaries')
if ckpt_detailed and 'detailed_summaries' in ckpt_detailed:
    print(f'\n✓ all_detailed_summaries checkpoint: {len(ckpt_detailed["detailed_summaries"])} summaries')
    for s in ckpt_detailed['detailed_summaries']:
        print(f'  {s["label"]:<25} {s.get("n_total",0)} samples')
else:
    print('\n✗ all_detailed_summaries checkpoint NOT FOUND')

# Check individual phase benchmarks
print(f'\n✓ Individual phase benchmarks:')
for phase in ['phase0_benchmark', 'phase1_benchmark', 'phase2_benchmark', 
              'phase3_benchmark', 'phase4_benchmark']:
    ckpt = load_latest_checkpoint(phase)
    if ckpt and 'summary' in ckpt:
        label = ckpt['summary'].get('label', '?')
        chrf = ckpt['summary'].get('avg_chrf', 0)
        print(f'  {phase:<25} → {label:<20} ChrF={chrf:>6.2f}')
    else:
        print(f'  {phase:<25} → NOT FOUND')

print('\n' + '='*70)
```

## What Your Checkpoint Files Contain

Your checkpoint files (`all_summaries_step000000.pt` and `all_detailed_summaries_step000000.pt`) contain:

```python
# all_summaries_step000000.pt
{
    'summaries': [
        {'label': 'P0_V1_Baseline', 'avg_chrf': 35.2, 'avg_bleu': 10.5, 'params_M': 1039, ...},
        {'label': 'P1_Vocab5L', 'avg_chrf': 35.1, 'avg_bleu': 10.4, 'params_M': 824, ...},
        {'label': 'P2_Enc16L', 'avg_chrf': 34.8, 'avg_bleu': 10.2, 'params_M': 630, ...},
        # ... etc
    ]
}

# all_detailed_summaries_step000000.pt
{
    'detailed_summaries': [
        {'label': 'P0_V1_Baseline', 'pair_stats': {...}, 'by_src_lang': {...}, ...},
        # ... etc
    ]
}
```

These are **cumulative** — each time you call `store_summary()` or `store_detailed_summary()`, it:
1. Loads existing checkpoint
2. Updates the dictionary
3. Saves back to checkpoint

So your data **is** in Drive, it's just not being loaded into memory when you skip to Phase 4.

## Why This Happens

Look at your setup cell:

```python
ALL_SUMMARIES: dict = _load_summaries_from_drive()
print(f'Loaded {len(ALL_SUMMARIES)} existing summaries: {list(ALL_SUMMARIES.keys())}')
```

This runs **once** at notebook startup. If you:
- Restart kernel → `ALL_SUMMARIES` reloads from checkpoint ✅
- Skip Phase 1-3 cells → `ALL_SUMMARIES` stays as loaded at startup ❌

## The Solution

Add reload logic before plotting:

```python
# Before Phase 4 benchmark
ALL_SUMMARIES = _load_summaries_from_drive()
ALL_DETAILED_SUMMARIES = _load_detailed_summaries_from_drive()

# Now run benchmark and plot
p4_results, p4_summary = run_benchmark_asr(...)
store_summary(p4_summary)
plot_phase_comparison()  # ← Now shows ALL phases
```

Or modify the plotting functions to auto-reload (better long-term solution).
