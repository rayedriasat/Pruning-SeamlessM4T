#!/usr/bin/env python3
"""
Complete verification that both Phase 3 fixes are applied
"""

import json

NOTEBOOK_PATH = "Alteration/seamless-final.ipynb"

def verify_all_fixes():
    print("=" * 80)
    print("PHASE 3 COMPLETE FIX VERIFICATION")
    print("=" * 80)
    
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the cell with _cosine_sim_layers
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            if 'def _cosine_sim_layers' in source:
                print("\n✓ Found _cosine_sim_layers function")
                
                # Check for both fixes
                has_attention_mask = 'attention_mask=None' in source
                has_dtype_fix = 'model_dtype' in source
                has_dtype_conversion = 'dtype=model_dtype' in source
                
                print("\n" + "─" * 80)
                print("FIX STATUS:")
                print("─" * 80)
                
                if has_attention_mask:
                    print("✅ Fix 1: attention_mask parameter - APPLIED")
                else:
                    print("❌ Fix 1: attention_mask parameter - MISSING")
                
                if has_dtype_fix and has_dtype_conversion:
                    print("✅ Fix 2: dtype conversion - APPLIED")
                else:
                    print("❌ Fix 2: dtype conversion - MISSING")
                
                print("\n" + "─" * 80)
                print("KEY LINES IN FUNCTION:")
                print("─" * 80)
                
                lines = source.split('\n')
                for line in lines:
                    if 'model_dtype' in line and '=' in line:
                        print(f"  {line.strip()}")
                    elif 'dtype=model_dtype' in line:
                        print(f"  {line.strip()}")
                    elif 'attention_mask=None' in line:
                        print(f"  {line.strip()}")
                
                print("\n" + "─" * 80)
                
                if has_attention_mask and has_dtype_fix and has_dtype_conversion:
                    print("✅ ALL FIXES APPLIED - READY TO USE!")
                    print("─" * 80)
                    print("\nNEXT STEPS:")
                    print("  1. Delete checkpoint: rm checkpoints/phase3_laco_done_step000000.pt")
                    print("  2. Re-run Phase 3 cells in your notebook")
                    print("  3. Verify you see non-zero similarity scores")
                    print("\nEXPECTED OUTPUT:")
                    print("  T2U-Enc: 6 layers -> merging up to 2")
                    print("    L1: sim=0.9234 -> MERGED [1/2]")
                    print("    L2: sim=0.9567 -> MERGED [2/2]")
                    print("    ...")
                    print("    T2U-Enc: 6 -> 4 layers")
                else:
                    print("⚠️  INCOMPLETE - Some fixes missing")
                    print("─" * 80)
                    if not has_attention_mask:
                        print("\n  Missing: attention_mask parameter")
                        print("  Run: python fix_phase3_laco.py")
                    if not (has_dtype_fix and has_dtype_conversion):
                        print("\n  Missing: dtype conversion")
                        print("  Run: python fix_phase3_dtype.py")
                
                print("\n" + "=" * 80)
                return has_attention_mask and has_dtype_fix and has_dtype_conversion
    
    print("❌ Could not find _cosine_sim_layers function")
    print("=" * 80)
    return False

if __name__ == '__main__':
    import sys
    success = verify_all_fixes()
    sys.exit(0 if success else 1)
