#!/usr/bin/env python3
"""
Extract and analyze Phase 6a training code from the notebook
"""
import json

# Read the notebook
with open('Alteration/seamless-final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Phase 6a cells
phase6a_cells = []
in_phase6a = False

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if 'Phase 6a: CIF Connector' in source:
            in_phase6a = True
            print("="*80)
            print("FOUND PHASE 6A MARKDOWN")
            print("="*80)
            print(source[:500])
            print()
    
    if in_phase6a and cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Look for key training components
        if any(keyword in source for keyword in [
            'MAX_STEPS_P6A', 'optimizer_6a', 'loss =', 'cos_loss', 
            'CIFConnector', 'threshold='
        ]):
            phase6a_cells.append(source)
            
            # Print key sections
            if 'MAX_STEPS_P6A' in source:
                print("\n" + "="*80)
                print("TRAINING HYPERPARAMETERS")
                print("="*80)
                for line in source.split('\n')[:20]:
                    if line.strip():
                        print(line)
            
            if 'threshold=' in source and 'CIFConnector' in source:
                print("\n" + "="*80)
                print("CIF CONNECTOR DEFINITION")
                print("="*80)
                lines = source.split('\n')
                for i, line in enumerate(lines):
                    if 'def __init__' in line:
                        for j in range(i, min(i+10, len(lines))):
                            print(lines[j])
                        break
            
            if 'loss =' in source and 'cos_loss' in source:
                print("\n" + "="*80)
                print("LOSS COMPUTATION")
                print("="*80)
                lines = source.split('\n')
                for i, line in enumerate(lines):
                    if 'loss =' in line and 'cos_loss' in line:
                        # Print context around loss computation
                        start = max(0, i-5)
                        end = min(len(lines), i+3)
                        for j in range(start, end):
                            print(lines[j])
                        break
            
            if 'optimizer_6a = ' in source:
                print("\n" + "="*80)
                print("OPTIMIZER CONFIGURATION")
                print("="*80)
                lines = source.split('\n')
                for i, line in enumerate(lines):
                    if 'optimizer_6a' in line:
                        for j in range(i, min(i+10, len(lines))):
                            print(lines[j])
                            if '], betas=' in lines[j]:
                                break
                        break
        
        # Stop after Phase 6b starts
        if 'Phase 6b' in source or 'DoRA' in source:
            break

print("\n" + "="*80)
print(f"TOTAL PHASE 6A CODE CELLS FOUND: {len(phase6a_cells)}")
print("="*80)
