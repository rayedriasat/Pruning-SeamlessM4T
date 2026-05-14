#!/usr/bin/env python3
"""
Update full-kd.ipynb with Full Model KD implementation for Phase 8
"""

import json
import re

def main():
    # Read the notebook
    print("Reading full-kd.ipynb...")
    with open('full-kd.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Read the new cell implementations
    print("Reading phase8_full_kd_cells.py...")
    with open('phase8_full_kd_cells.py', 'r', encoding='utf-8') as f:
        new_code = f.read()
    
    # Extract individual cells from the new code
    cell_pattern = r'# ={70,}\n# Phase 8 — Cell (\d+):([^\n]+)\n# ={70,}\n\n(.*?)(?=\n# ={70,}\n# Phase 8 — Cell \d+:|# ={70,}\n# NOTES FOR BENCHMARK CELLS|$)'
    cell_matches = re.findall(cell_pattern, new_code, re.DOTALL)
    
    new_cells = {}
    for cell_num, title, code in cell_matches:
        # Clean up the code
        code = code.strip()
        # Remove the docstring/markdown comment at the start if present
        if code.startswith('"""'):
            parts = code.split('"""', 2)
            if len(parts) >= 3:
                code = parts[2].strip()
        new_cells[int(cell_num)] = {
            'title': title.strip(),
            'code': code
        }
    
    print(f"Extracted {len(new_cells)} new cells: {list(new_cells.keys())}")
    
    # Find and update Phase 8 cells in the notebook
    updated_count = 0
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Check if this is a Phase 8 cell header
            match = re.search(r'## Phase 8 — Cell (\d+):', source)
            if match:
                cell_num = int(match.group(1))
                print(f"Found Phase 8 Cell {cell_num} header at index {i}")
                
                # The code cell should be next
                if i + 1 < len(nb['cells']) and nb['cells'][i + 1]['cell_type'] == 'code':
                    if cell_num in new_cells:
                        # Update the code cell
                        new_code_text = new_cells[cell_num]['code']
                        new_code_lines = new_code_text.split('\n')
                        
                        # Format as notebook expects (list of strings with \n)
                        if len(new_code_lines) > 0:
                            nb['cells'][i + 1]['source'] = [line + '\n' for line in new_code_lines[:-1]] + [new_code_lines[-1]]
                        else:
                            nb['cells'][i + 1]['source'] = [new_code_text]
                        
                        # Clear outputs
                        nb['cells'][i + 1]['outputs'] = []
                        nb['cells'][i + 1]['execution_count'] = None
                        
                        updated_count += 1
                        print(f"  ✓ Updated Phase 8 Cell {cell_num} code ({len(new_code_lines)} lines)")
    
    print(f"\nUpdated {updated_count} Phase 8 training cells")
    
    # Now update benchmark cells - replace 'phase8_kd' with 'phase8_full_kd'
    benchmark_updates = 0
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Check if this is a Phase 8 benchmark cell
            if 'Phase 8 Benchmark Cell' in source or ('phase8_kd' in source and 'p8_bench_summaries' in source):
                # Replace phase8_kd with phase8_full_kd
                updated_source = source.replace("'phase8_kd'", "'phase8_full_kd'")
                updated_source = updated_source.replace('"phase8_kd"', '"phase8_full_kd"')
                updated_source = updated_source.replace('P8 KD', 'P8 Full KD')
                updated_source = updated_source.replace('phase8_4model', 'phase8_full_kd_4model')
                updated_source = updated_source.replace('phase8_radar', 'phase8_full_kd_radar')
                updated_source = updated_source.replace('phase8_benchmark_summary', 'phase8_full_kd_benchmark_summary')
                updated_source = updated_source.replace('phase8_kd_training_curves', 'phase8_full_kd_training_curves')
                
                if updated_source != source:
                    # Convert back to list format
                    lines = updated_source.split('\n')
                    if len(lines) > 0:
                        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
                    else:
                        cell['source'] = [updated_source]
                    
                    cell['outputs'] = []
                    cell['execution_count'] = None
                    benchmark_updates += 1
                    print(f"  ✓ Updated benchmark cell at index {i}")
    
    print(f"\nUpdated {benchmark_updates} benchmark cells")
    
    # Update the main Phase 8 markdown header
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if '# Phase 8: T2U Knowledge Distillation (Audio Translation Recovery)' in source:
                updated_source = source.replace(
                    '# Phase 8: T2U Knowledge Distillation (Audio Translation Recovery)',
                    '# Phase 8: Full Model Knowledge Distillation (Audio Quality Recovery)'
                )
                updated_source = updated_source.replace(
                    'T2U sub-model of',
                    'entire model from'
                )
                updated_source = updated_source.replace(
                    'All Phase 7 DoRA-recovered components are **frozen**; only the\npruned T2U encoder + decoder are updated.',
                    'All model components are **trainable** (speech encoder, text decoder, T2U model, etc.).'
                )
                updated_source = updated_source.replace(
                    '| `t2u_model`      | 🔥 **Trained via KD** |',
                    '| `t2u_model`      | 🔥 **Trained via Full KD** |'
                )
                
                lines = updated_source.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
                print(f"  ✓ Updated Phase 8 main header at index {i}")
                break
    
    # Write back the notebook
    print("\nWriting updated notebook...")
    with open('full-kd.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully updated full-kd.ipynb!")
    print(f"{'='*60}")
    print(f"Changes made:")
    print(f"  • {updated_count} Phase 8 training cells updated")
    print(f"  • {benchmark_updates} benchmark cells updated")
    print(f"  • Model name: phase8_kd → phase8_full_kd")
    print(f"  • Approach: T2U-only KD → Full Model KD")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Open full-kd.ipynb in Jupyter/Kaggle/Colab")
    print(f"  2. Run Phase 8 cells 1-7 to train")
    print(f"  3. Run Phase 8 Benchmark cells to evaluate")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
