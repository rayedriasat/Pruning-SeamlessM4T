#!/usr/bin/env python3
"""
Verify that the Phase 6 fix was applied correctly to pragmata-recovery.ipynb
"""

import json
import sys

def verify_fix():
    notebook_path = 'AAA/pragmata-recovery.ipynb'
    
    print("=" * 70)
    print("Phase 6 Fix Verification")
    print("=" * 70)
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"❌ Error loading notebook: {e}")
        return False
    
    print(f"\n✓ Loaded notebook with {len(notebook['cells'])} cells")
    
    # Check for the fixed text_recovery_step
    fix1_found = False
    fix1_pattern = "cache_entry['teacher_text_sequences'].unsqueeze(0).to(student_device)"
    
    # Check for cache validation
    fix2_found = False
    fix2_pattern = "if teacher_text_sequences.numel() == 0:"
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = cell.get('source', [])
        if isinstance(source, list):
            source_str = ''.join(source)
        else:
            source_str = source
        
        if fix1_pattern in source_str and not fix1_found:
            fix1_found = True
            print(f"\n✓ Fix 1 found in cell {i}: text_recovery_step uses pre-tokenized sequences")
        
        if fix2_pattern in source_str and not fix2_found:
            fix2_found = True
            print(f"✓ Fix 2 found in cell {i}: Cache validation added")
    
    print("\n" + "=" * 70)
    print("Verification Results:")
    print("=" * 70)
    
    if fix1_found and fix2_found:
        print("✓ All fixes verified successfully!")
        print("\nYour notebook is ready to use. Next steps:")
        print("1. Upload AAA/pragmata-recovery.ipynb to Kaggle")
        print("2. Restart the Kaggle kernel")
        print("3. Run all cells up to Phase 6")
        print("4. Training should proceed without CUDA errors")
        return True
    else:
        print("❌ Some fixes are missing:")
        if not fix1_found:
            print("  - text_recovery_step fix not found")
        if not fix2_found:
            print("  - Cache validation fix not found")
        print("\nPlease run: python apply_phase6_fix.py")
        return False

if __name__ == '__main__':
    success = verify_fix()
    sys.exit(0 if success else 1)
