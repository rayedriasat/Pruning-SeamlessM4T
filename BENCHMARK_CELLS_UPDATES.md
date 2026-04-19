# Phase 8 Benchmark Cells - Quick Update Reference

## Overview
This document shows the exact changes needed in Phase 8 Benchmark cells to work with Full Model KD.

## Changes Required

### Change 1: Model Name
**Find and replace throughout all benchmark cells:**

```python
# OLD:
'phase8_kd'

# NEW:
'phase8_full_kd'
```

**Affected locations:**
- Checkpoint loading: `load_latest_checkpoint('phase8_kd')`
- Model saving: `save_checkpoint(..., name='phase8_kd', ...)`
- Model loading: `load_model_from_drive('phase8_kd', ...)`
- Dictionary keys: `p8_bench_summaries['phase8_kd']`
- Figure filenames: `'phase8_kd_training_curves.png'`

### Change 2: Display Labels

**In benchmark comparison lists:**

```python
# OLD:
MODEL_LABELS = [
    ('teacher', 'Teacher\\n(baseline)'),
    ('phase6_t2u_iter_pruned', 'P6 T2U\\nPruned'),
    ('phase7_dora_merged_v1',  'P7 DoRA\\nMerged'),
    ('phase8_kd',              'P8 KD\\n(final)'),
]

# NEW:
MODEL_LABELS = [
    ('teacher', 'Teacher\\n(baseline)'),
    ('phase6_t2u_iter_pruned', 'P6 T2U\\nPruned'),
    ('phase7_dora_merged_v1',  'P7 DoRA\\nMerged'),
    ('phase8_full_kd',         'P8 Full KD\\n(final)'),
]
```

### Change 3: Color Mappings

```python
# OLD:
MODEL_COLORS = {
    'teacher':                '#FFA726',
    'phase6_t2u_iter_pruned': '#FF7043',
    'phase7_dora_merged_v1':  '#42A5F5',
    'phase8_kd':              '#66BB6A',
}

# NEW:
MODEL_COLORS = {
    'teacher':                '#FFA726',
    'phase6_t2u_iter_pruned': '#FF7043',
    'phase7_dora_merged_v1':  '#42A5F5',
    'phase8_full_kd':         '#66BB6A',
}
```

### Change 4: Figure Titles

**Phase 8 Benchmark Cell 3 (4-model comparison):**

```python
# OLD:
fig.suptitle(
    'Phase 8 — 4-Model Quality Comparison\\n'
    '(Teacher  ·  P6 Pruned  ·  P7 DoRA  ·  P8 KD Final)',
    fontsize=14, fontweight='bold', y=1.01
)

# NEW:
fig.suptitle(
    'Phase 8 — 4-Model Quality Comparison\\n'
    '(Teacher  ·  P6 Pruned  ·  P7 DoRA  ·  P8 Full KD Final)',
    fontsize=14, fontweight='bold', y=1.01
)
```

### Change 5: Figure Filenames

```python
# OLD:
save_figure(fig, 'phase8_4model_comparison.png')
save_figure(fig, 'phase8_radar_comparison.png')
csv_path = f'{FIG_DIR}/phase8_benchmark_summary.csv'

# NEW:
save_figure(fig, 'phase8_full_kd_4model_comparison.png')
save_figure(fig, 'phase8_full_kd_radar_comparison.png')
csv_path = f'{FIG_DIR}/phase8_full_kd_benchmark_summary.csv'
```

### Change 6: Print Statements

```python
# OLD:
print('[P8 BENCH] 4/4 — phase8_kd  (final student)')
print('[P8] Comparison figure saved → phase8_4model_comparison.png')
print('[P8] Radar chart saved → phase8_radar_comparison.png')

# NEW:
print('[P8 BENCH] 4/4 — phase8_full_kd  (final student)')
print('[P8] Comparison figure saved → phase8_full_kd_4model_comparison.png')
print('[P8] Radar chart saved → phase8_full_kd_radar_comparison.png')
```

### Change 7: Benchmark Cell Comments

**Phase 8 Benchmark Cell 2:**

```python
# OLD:
# ── 4. phase8_kd (final student) ──────────────────────────────────────────────

# NEW:
# ── 4. phase8_full_kd (final student) ─────────────────────────────────────────
```

## Complete Cell-by-Cell Updates

### Phase 8 Benchmark Cell 1: run_benchmark_full()
**No changes needed** - this is a utility function

### Phase 8 Benchmark Cell 2: Evaluate all four models

```python
# ── 4. phase8_full_kd (final student) ─────────────────────────────────────────
print('\n' + '='*60)
print('[P8 BENCH] 4/4 — phase8_full_kd  (final student)')
print('='*60)
model_p8_student.eval()
_, summ = run_benchmark_full(model_p8_student, eval_samples,
                              label='phase8_full_kd', tgt_lang=TARGET_LANG, save_n=2)
p8_bench_summaries['phase8_full_kd'] = summ
store_summary({**summ, 'label': 'P8_FullKD_Final'})

print('\n[P8] All benchmarks complete.')
```

### Phase 8 Benchmark Cell 3: 4-metric comparison figure

```python
MODEL_LABELS = [
    ('teacher', 'Teacher\\n(baseline)'),
    ('phase6_t2u_iter_pruned', 'P6 T2U\\nPruned'),
    ('phase7_dora_merged_v1',  'P7 DoRA\\nMerged'),
    ('phase8_full_kd',         'P8 Full KD\\n(final)'),
]

MODEL_COLORS = {
    'teacher':                '#FFA726',
    'phase6_t2u_iter_pruned': '#FF7043',
    'phase7_dora_merged_v1':  '#42A5F5',
    'phase8_full_kd':         '#66BB6A',
}

# ... rest of plotting code ...

fig.suptitle(
    'Phase 8 — 4-Model Quality Comparison\\n'
    '(Teacher  ·  P6 Pruned  ·  P7 DoRA  ·  P8 Full KD Final)',
    fontsize=14, fontweight='bold', y=1.01
)

# ... rest of plotting code ...

save_figure(fig, 'phase8_full_kd_4model_comparison.png')
plt.show()
print('[P8] Comparison figure saved → phase8_full_kd_4model_comparison.png')
```

### Phase 8 Benchmark Cell 4: Radar chart

```python
for key, label in [
    ('teacher', 'Teacher'),
    ('phase6_t2u_iter_pruned', 'P6 T2U Pruned'),
    ('phase7_dora_merged_v1',  'P7 DoRA Merged'),
    ('phase8_full_kd',         'P8 Full KD Final'),
]:
    # ... plotting code ...

save_figure(fig, 'phase8_full_kd_radar_comparison.png')
plt.show()
print('[P8] Radar chart saved → phase8_full_kd_radar_comparison.png')
```

### Phase 8 Benchmark Cell 5: Numeric summary table

```python
for k, label in [
    ('teacher', 'Teacher'),
    ('phase6_t2u_iter_pruned', 'P6 T2U Pruned'),
    ('phase7_dora_merged_v1',  'P7 DoRA Merged'),
    ('phase8_full_kd',         'P8 Full KD Final'),
]:
    # ... table building code ...

csv_path = f'{FIG_DIR}/phase8_full_kd_benchmark_summary.csv'
df_results.to_csv(csv_path, index=False)
if ON_KAGGLE:
    _rclone_push(csv_path, 'figures')
print(f'\n[P8] CSV saved → {csv_path}')
```

## Search and Replace Strategy

For efficiency, use these search-replace operations in your editor:

1. **Search**: `'phase8_kd'` → **Replace**: `'phase8_full_kd'`
2. **Search**: `"phase8_kd"` → **Replace**: `"phase8_full_kd"`
3. **Search**: `P8 KD` → **Replace**: `P8 Full KD`
4. **Search**: `phase8_4model` → **Replace**: `phase8_full_kd_4model`
5. **Search**: `phase8_radar` → **Replace**: `phase8_full_kd_radar`
6. **Search**: `phase8_benchmark_summary` → **Replace**: `phase8_full_kd_benchmark_summary`

## Verification Checklist

After making changes, verify:

- [ ] All `'phase8_kd'` references changed to `'phase8_full_kd'`
- [ ] All display labels updated to include "Full KD"
- [ ] All figure filenames include `full_kd` prefix
- [ ] All print statements updated
- [ ] Model color mappings updated
- [ ] CSV filename updated
- [ ] No references to old T2U-only approach remain

## Testing

After updates, test by:

1. Running Phase 8 Benchmark Cell 1 (utility function)
2. Running Phase 8 Benchmark Cell 2 (model evaluation)
3. Checking that `p8_bench_summaries['phase8_full_kd']` exists
4. Running visualization cells (3-5)
5. Verifying all figures saved with correct names

---

**Note**: These changes are purely cosmetic/naming updates. The benchmark logic remains identical - we're just updating references from the old T2U-only approach to the new Full Model KD approach.
