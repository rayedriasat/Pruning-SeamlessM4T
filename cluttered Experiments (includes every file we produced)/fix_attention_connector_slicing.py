#!/usr/bin/env python3
"""
Fix AttentionConnector type conversion bug in Phase 6a
Converts max_len tensor to int before using it for slicing query_embed
"""

import json
import sys

def fix_attention_connector(notebook_path):
    """Fix the AttentionConnector slicing bug in the notebook"""
    
    print(f"Loading notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    fixed_count = 0
    
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Look for the AttentionConnector class definition
        if 'class AttentionConnector' in source and 'def forward' in source:
            print("\n✓ Found AttentionConnector class")
            
            # Check if the bug exists (max_len used directly in slicing without int())
            if 'queries = self.query_embed[:max_len]' in source:
                print("  → Found buggy line: queries = self.query_embed[:max_len]")
                
                # Fix: Convert max_len to int before slicing
                source = source.replace(
                    'queries = self.query_embed[:max_len].unsqueeze(0).expand(B, -1, -1)',
                    'queries = self.query_embed[:int(max_len)].unsqueeze(0).expand(B, -1, -1)'
                )
                
                # Update the cell source
                cell['source'] = source.split('\n') if '\n' in source else [source]
                fixed_count += 1
                print("  ✓ FIXED: Added int() conversion")
            
            elif 'queries = self.query_embed[:int(max_len)]' in source:
                print("  ✓ Already fixed (int() conversion present)")
            
            else:
                # Check if there's a different pattern
                if ':max_len]' in source:
                    print("  ⚠ WARNING: Found ':max_len]' pattern but not in expected location")
                    print("  → Manual inspection recommended")
    
    if fixed_count > 0:
        # Save backup
        backup_path = notebook_path + '.backup'
        print(f"\n📦 Creating backup: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        # Save fixed notebook
        print(f"💾 Saving fixed notebook: {notebook_path}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        print(f"\n✅ SUCCESS: Fixed {fixed_count} cell(s)")
        print("\nThe fix converts max_len tensor to int before slicing:")
        print("  BEFORE: queries = self.query_embed[:max_len]")
        print("  AFTER:  queries = self.query_embed[:int(max_len)]")
        print("\nThis resolves: TypeError: slice indices must be integers or None or have an __index__ method")
        
    else:
        print("\n⚠ No fixes applied - either already fixed or pattern not found")
    
    return fixed_count

if __name__ == '__main__':
    notebook_path = 'Alteration/seamless-final.ipynb'
    
    try:
        fixed = fix_attention_connector(notebook_path)
        sys.exit(0 if fixed > 0 else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
