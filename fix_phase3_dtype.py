#!/usr/bin/env python3
"""
Fix Phase 3 dtype mismatch issue
The model is in float16 but calibration tensors are float32
"""

import json
import sys

NOTEBOOK_PATH = "Alteration/seamless-final.ipynb"

# The fixed _cosine_sim_layers function with dtype handling
FIXED_FUNCTION = '''def _cosine_sim_layers(merged, orig_j, calib_tensors, device):
    """Measure output similarity between merged and original layer_j."""    
    orig_j = orig_j.to(device).eval()
    merged = merged.to(device).eval()
    
    # Get the dtype from the model layers
    model_dtype = next(orig_j.parameters()).dtype
    
    sims = []
    for x in calib_tensors[:5]:
        if x is None: continue
        # Convert calibration tensor to match model dtype
        x = x.to(device=device, dtype=model_dtype)
        with torch.no_grad():
            try:
                # T2U layers expect (hidden_states, attention_mask=None, ...)
                o = orig_j(x, attention_mask=None)
                o = o[0] if isinstance(o, tuple) else o
                m = merged(x, attention_mask=None)
                m = m[0] if isinstance(m, tuple) else m
                # Compute cosine similarity on flattened tensors
                sim = F.cosine_similarity(o.reshape(-1), m.reshape(-1), dim=0).item()
                sims.append(sim)
            except Exception as e:
                # Debug: print what went wrong
                print(f' [sim_err: {str(e)[:50]}]', end='')
                pass
    return float(np.mean(sims)) if sims else 0.0'''

def fix_notebook():
    print(f"Reading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    fixed_count = 0
    
    # Find and fix the cell containing _cosine_sim_layers
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            if '_cosine_sim_layers' in source and 'def _cosine_sim_layers' in source:
                print(f"Found cell with _cosine_sim_layers function")
                
                # Find the function and replace it
                lines = source.split('\n')
                new_lines = []
                in_function = False
                skip_until_next_def = False
                
                for i, line in enumerate(lines):
                    if 'def _cosine_sim_layers' in line:
                        in_function = True
                        skip_until_next_def = True
                        # Add the fixed function
                        new_lines.extend(FIXED_FUNCTION.split('\n'))
                        continue
                    
                    if skip_until_next_def:
                        # Skip old function lines until we hit the next def or end
                        if line.strip().startswith('def ') and '_cosine_sim_layers' not in line:
                            skip_until_next_def = False
                            new_lines.append(line)
                        elif i == len(lines) - 1:  # Last line
                            skip_until_next_def = False
                        continue
                    
                    new_lines.append(line)
                
                # Update the cell source
                cell['source'] = '\n'.join(new_lines)
                fixed_count += 1
                print(f"✓ Fixed _cosine_sim_layers function with dtype handling")
    
    if fixed_count == 0:
        print("ERROR: Could not find the function to fix!")
        return False
    
    # Save the fixed notebook
    print(f"Writing fixed notebook to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"\n✅ Successfully fixed {fixed_count} cell(s)")
    print("\nWhat was fixed:")
    print("  1. Added dtype detection from model parameters")
    print("  2. Convert calibration tensors to match model dtype (float16)")
    print("  3. Line: x = x.to(device=device, dtype=model_dtype)")
    print("\nThe issue: Model is in float16 but calibration tensors were float32")
    print("This caused: 'expected scalar type Float but found Half' errors")
    
    return True

if __name__ == '__main__':
    success = fix_notebook()
    sys.exit(0 if success else 1)
