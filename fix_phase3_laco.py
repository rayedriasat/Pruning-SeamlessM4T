#!/usr/bin/env python3
"""
Fix Phase 3 LaCo RDSC merge code in seamless-final.ipynb
Issue: _cosine_sim_layers returns 0.0 because T2U layers need proper arguments
"""

import json
import sys

NOTEBOOK_PATH = "Alteration/seamless-final.ipynb"

# The fixed _cosine_sim_layers function
FIXED_FUNCTION = '''def _cosine_sim_layers(merged, orig_j, calib_tensors, device):
    """Measure output similarity between merged and original layer_j."""    
    orig_j = orig_j.to(device).eval(); merged = merged.to(device).eval()
    sims = []
    for x in calib_tensors[:5]:
        if x is None: continue
        x = x.to(device)
        with torch.no_grad():
            try:
                # T2U layers expect (hidden_states, attention_mask=None, ...)
                # Pass as positional arg to match layer signature
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

# Pattern to find in the old code
OLD_PATTERN = "o = orig_j(x);   o = o[0] if isinstance(o,tuple) else o"

def fix_notebook():
    print(f"Reading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    fixed_count = 0
    
    # Find and fix the cell containing _cosine_sim_layers
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            if OLD_PATTERN in source and '_cosine_sim_layers' in source:
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
                print(f"✓ Fixed _cosine_sim_layers function")
    
    if fixed_count == 0:
        print("ERROR: Could not find the function to fix!")
        return False
    
    # Save the fixed notebook
    backup_path = NOTEBOOK_PATH + '.backup'
    print(f"Creating backup at {backup_path}...")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Writing fixed notebook to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"\n✅ Successfully fixed {fixed_count} cell(s)")
    print("\nWhat was fixed:")
    print("  1. Added attention_mask=None parameter to layer calls")
    print("  2. Added debug output for exceptions")
    print("  3. Improved error handling")
    print("\nThe issue: T2U encoder/decoder layers expect attention_mask parameter,")
    print("but the old code was calling them with just the hidden states.")
    print("This caused silent exceptions, resulting in sim=0.0000")
    
    return True

if __name__ == '__main__':
    success = fix_notebook()
    sys.exit(0 if success else 1)
