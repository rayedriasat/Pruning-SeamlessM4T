#!/usr/bin/env python3
"""
Verification script to check that enhancements were applied correctly
"""

import json

print("="*70)
print("  VERIFICATION: Enhanced Tracking in seamless-final.ipynb")
print("="*70)

# Load the notebook
with open('Alteration/seamless-final.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"\n✓ Notebook loaded: {len(notebook['cells'])} cells")

# Check for enhanced tracking cell
enhanced_cell_found = False
guide_cell_found = False
updated_benchmarks = []

for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Check for enhanced tracking functions
        if 'compute_detailed_summary' in source and 'def compute_detailed_summary' in source:
            enhanced_cell_found = True
            print(f"✓ Enhanced tracking cell found at index {idx}")
        
        # Check for updated benchmark cells
        if 'compute_detailed_summary(' in source and 'store_detailed_summary(' in source:
            if 'P0_V1_Baseline' in source:
                updated_benchmarks.append(('Phase 0', idx))
            elif 'P1_Vocab5L' in source:
                updated_benchmarks.append(('Phase 1', idx))
            elif 'P2_Enc16L' in source:
                updated_benchmarks.append(('Phase 2', idx))
            elif 'P3_LaCoT2U' in source:
                updated_benchmarks.append(('Phase 3', idx))
    
    elif cell['cell_type'] == 'markdown':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'Enhanced Per-Language Tracking Enabled' in source:
            guide_cell_found = True
            print(f"✓ Guide cell found at index {idx}")

print(f"\n{'='*70}")
print("  RESULTS")
print(f"{'='*70}")

if enhanced_cell_found:
    print("✓ Enhanced tracking functions: PRESENT")
else:
    print("✗ Enhanced tracking functions: MISSING")

if guide_cell_found:
    print("✓ Guide cell: PRESENT")
else:
    print("✗ Guide cell: MISSING")

print(f"\n✓ Updated benchmark cells: {len(updated_benchmarks)}/4")
for phase, idx in updated_benchmarks:
    print(f"  ✓ {phase} (cell {idx})")

missing_phases = []
expected = ['Phase 0', 'Phase 1', 'Phase 2', 'Phase 3']
found_phases = [p for p, _ in updated_benchmarks]
for phase in expected:
    if phase not in found_phases:
        missing_phases.append(phase)

if missing_phases:
    print(f"\n⚠ Missing updates:")
    for phase in missing_phases:
        print(f"  ✗ {phase}")

print(f"\n{'='*70}")
if enhanced_cell_found and len(updated_benchmarks) >= 4:
    print("  STATUS: ✅ ALL ENHANCEMENTS APPLIED SUCCESSFULLY")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("  1. Open Alteration/seamless-final.ipynb in Jupyter/Kaggle")
    print("  2. Run cells to generate enhanced visualizations")
    print("  3. Check for 'detailed_comparison.png' in figures/")
    print("  4. Verify 'all_detailed_summaries_step000000.pt' in checkpoints/")
else:
    print("  STATUS: ⚠ INCOMPLETE - Some enhancements missing")
    print(f"{'='*70}")
    print("\nPlease review the notebook manually.")

print(f"\n{'='*70}")
print("  BACKUP")
print(f"{'='*70}")

import os
if os.path.exists('Alteration/seamless-final.ipynb.backup'):
    backup_size = os.path.getsize('Alteration/seamless-final.ipynb.backup') / 1024 / 1024
    print(f"✓ Backup exists: seamless-final.ipynb.backup ({backup_size:.1f} MB)")
    print("  To restore: cp seamless-final.ipynb.backup seamless-final.ipynb")
else:
    print("✗ No backup found")

print(f"\n{'='*70}\n")
