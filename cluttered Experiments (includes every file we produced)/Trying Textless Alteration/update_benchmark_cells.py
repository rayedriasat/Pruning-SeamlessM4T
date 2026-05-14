#!/usr/bin/env python3
"""
Script to update benchmark cells in the enhanced notebook
Adds detailed summary tracking to all phase benchmark cells
"""

import json
import re

# Read the enhanced notebook
with open('Alteration/seamless-final-enhanced.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Loaded notebook with {len(notebook['cells'])} cells")

def get_cell_source(cell):
    """Get cell source as string"""
    if isinstance(cell['source'], list):
        return ''.join(cell['source'])
    return cell['source']

def set_cell_source(cell, source):
    """Set cell source from string"""
    cell['source'] = source

# Find and update Phase 0 benchmark cell
for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code':
        continue
    
    source = get_cell_source(cell)
    
    # Phase 0 benchmark
    if 'p0_bench = load_latest_checkpoint' in source and 'phase0_benchmark' in source:
        print(f"Updating Phase 0 benchmark cell at index {idx}")
        
        # Replace the cell with enhanced version
        new_source = """p0_bench = load_latest_checkpoint('phase0_benchmark')
if p0_bench and p0_bench.get('summary', {}).get('avg_bleu', 0) > 0:
    p0_results = p0_bench['results']
    p0_summary = p0_bench['summary']
    p0_detailed = p0_bench.get('detailed_summary')
    print('Loaded Phase 0 benchmark results from checkpoint.')
    
    # Recompute detailed if missing
    if not p0_detailed:
        p0_detailed = compute_detailed_summary(p0_results, 'P0_V1_Baseline', p0_summary['params_M'])
else:
    p0_results, p0_summary = run_benchmark_asr(
        model_v1, eval_samples, label='P0_V1_Baseline', save_n=4)
    p0_detailed = compute_detailed_summary(p0_results, 'P0_V1_Baseline', p0_summary['params_M'])
    save_checkpoint(dict(
        results=p0_results, 
        summary=p0_summary,
        detailed_summary=p0_detailed
    ), 'phase0_benchmark', 0)

store_summary(p0_summary)
store_detailed_summary(p0_detailed)
print_detailed_summary_table('P0_V1_Baseline')
plot_phase_comparison()
plot_detailed_phase_comparison()
"""
        set_cell_source(cell, new_source)
    
    # Phase 1 benchmark
    elif 'p1_bench = load_latest_checkpoint' in source and 'phase1_benchmark' in source:
        print(f"Updating Phase 1 benchmark cell at index {idx}")
        
        new_source = """p1_bench = load_latest_checkpoint('phase1_benchmark')
if p1_bench and p1_bench.get('summary', {}).get('avg_bleu', 0) > 0:
    p1_results = p1_bench['results']
    p1_summary = p1_bench['summary']
    p1_detailed = p1_bench.get('detailed_summary')
    if not p1_detailed:
        p1_detailed = compute_detailed_summary(p1_results, 'P1_Vocab5L', p1_summary['params_M'])
else:
    p1_results, p1_summary = run_benchmark_asr(
        model_p1, eval_samples, label='P1_Vocab5L', save_n=4)
    p1_detailed = compute_detailed_summary(p1_results, 'P1_Vocab5L', p1_summary['params_M'])
    save_checkpoint(dict(
        results=p1_results, 
        summary=p1_summary,
        detailed_summary=p1_detailed
    ), 'phase1_benchmark', 0)

store_summary(p1_summary)
store_detailed_summary(p1_detailed)
print_detailed_summary_table('P1_Vocab5L')
plot_phase_comparison()
plot_detailed_phase_comparison()
"""
        set_cell_source(cell, new_source)
    
    # Phase 2 benchmark
    elif 'p2_bench = load_latest_checkpoint' in source and 'phase2_benchmark' in source:
        print(f"Updating Phase 2 benchmark cell at index {idx}")
        
        new_source = """p2_bench = load_latest_checkpoint('phase2_benchmark')
if p2_bench:
    p2_results = p2_bench['results']
    p2_summary = p2_bench['summary']
    p2_detailed = p2_bench.get('detailed_summary')
    if not p2_detailed:
        p2_detailed = compute_detailed_summary(p2_results, 'P2_Enc16L', p2_summary['params_M'])
else:
    p2_results, p2_summary = run_benchmark_asr(model_p2, eval_samples, 'P2_Enc16L', save_n=4)
    p2_detailed = compute_detailed_summary(p2_results, 'P2_Enc16L', p2_summary['params_M'])
    save_checkpoint(dict(
        results=p2_results, 
        summary=p2_summary,
        detailed_summary=p2_detailed
    ), 'phase2_benchmark', 0)

store_summary(p2_summary)
store_detailed_summary(p2_detailed)
print_detailed_summary_table('P2_Enc16L')
plot_phase_comparison()
plot_detailed_phase_comparison()
"""
        set_cell_source(cell, new_source)
    
    # Phase 3 benchmark
    elif 'p3_bench = load_latest_checkpoint' in source and 'phase3_benchmark' in source:
        print(f"Updating Phase 3 benchmark cell at index {idx}")
        
        new_source = """p3_bench = load_latest_checkpoint('phase3_benchmark')
if p3_bench:
    p3_results = p3_bench['results']
    p3_summary = p3_bench['summary']
    p3_detailed = p3_bench.get('detailed_summary')
    if not p3_detailed:
        p3_detailed = compute_detailed_summary(p3_results, 'P3_LaCoT2U', p3_summary['params_M'])
else:
    p3_results, p3_summary = run_benchmark_asr(
        model_p3, eval_samples, 'P3_LaCoT2U', save_n=4)
    p3_detailed = compute_detailed_summary(p3_results, 'P3_LaCoT2U', p3_summary['params_M'])
    save_checkpoint(dict(
        results=p3_results, 
        summary=p3_summary,
        detailed_summary=p3_detailed
    ), 'phase3_benchmark', 0)

store_summary(p3_summary)
store_detailed_summary(p3_detailed)
print_detailed_summary_table('P3_LaCoT2U')
plot_phase_comparison()
plot_detailed_phase_comparison()
"""
        set_cell_source(cell, new_source)

# Save the updated notebook
with open('Alteration/seamless-final-enhanced.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"\n✓ Updated benchmark cells in enhanced notebook")
print(f"  File: Alteration/seamless-final-enhanced.ipynb")
print(f"\nEnhancements applied:")
print(f"  ✓ Phase 0 benchmark - detailed tracking added")
print(f"  ✓ Phase 1 benchmark - detailed tracking added")
print(f"  ✓ Phase 2 benchmark - detailed tracking added")
print(f"  ✓ Phase 3 benchmark - detailed tracking added")
print(f"\nReady to use! Open the enhanced notebook and run cells.")
