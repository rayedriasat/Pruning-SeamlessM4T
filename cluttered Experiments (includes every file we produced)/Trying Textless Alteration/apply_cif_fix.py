"""
Quick script to apply CIF over-firing fix to seamless-final.ipynb

This script will:
1. Update CIF threshold from 0.50 to 0.95
2. Update weight scaling from 1.0× to 0.8×
3. Update loss weights (rebalanced)
4. Update learning rates
5. Create a backup before modifying

Usage:
    python apply_cif_fix.py
"""

import json
import shutil
from pathlib import Path

def apply_cif_fix(notebook_path='./seamless-final.ipynb'):
    """Apply the CIF over-firing fix to the notebook."""
    
    nb_path = Path(notebook_path)
    if not nb_path.exists():
        print(f"ERROR: Notebook not found: {notebook_path}")
        return False
    
    # Create backup
    backup_path = nb_path.with_suffix('.ipynb.backup_before_cif_fix')
    shutil.copy2(nb_path, backup_path)
    print(f"✓ Backup created: {backup_path}")
    
    # Load notebook
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = 0
    
    # Fix 1: Update CIF threshold in class definition
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Fix CIF threshold
            if 'def __init__(self, d_model=1024, n_refiner_layers=2, n_langs=45, threshold=0.50)' in source:
                cell['source'] = [line.replace('threshold=0.50', 'threshold=0.95') 
                                 for line in cell['source']]
                changes_made += 1
                print("✓ Fixed CIF threshold: 0.50 → 0.95")
            
            # Fix weight scaling
            if 'alpha  = raw_w / w_sum * qty_pred.unsqueeze(1)' in source:
                cell['source'] = [line.replace(
                    'alpha  = raw_w / w_sum * qty_pred.unsqueeze(1)',
                    'alpha  = raw_w / w_sum * (0.8 * qty_pred.unsqueeze(1))  # FIXED: gentler scaling'
                ) for line in cell['source']]
                changes_made += 1
                print("✓ Fixed weight scaling: 1.0× → 0.8×")
            
            # Fix residual threshold
            if 'if acc_w > 1e-6:' in source:
                cell['source'] = [line.replace('if acc_w > 1e-6:', 'if acc_w > 0.05:  # FIXED')
                                 for line in cell['source']]
                changes_made += 1
                print("✓ Fixed residual threshold: 1e-6 → 0.05")
            
            # Fix final fire threshold
            if 'if acc_w > 0.1:' in source and 'Fire remaining' in source:
                cell['source'] = [line.replace('if acc_w > 0.1:', 'if acc_w > 0.3:  # FIXED')
                                 for line in cell['source']]
                changes_made += 1
                print("✓ Fixed final fire threshold: 0.1 → 0.3")
            
            # Fix optimizer LR
            if "'lr': 3e-4" in source and 'cif_connector' in source:
                cell['source'] = [line.replace("'lr': 3e-4", "'lr': 2e-4  # FIXED")
                                 for line in cell['source']]
                changes_made += 1
                print("✓ Fixed connector LR: 3e-4 → 2e-4")
            
            # Fix loss weights
            if 'loss = (0.30 * cos_loss' in source:
                # Find and replace the entire loss computation
                new_loss = '''        # FIXED LOSS WEIGHTS (rebalanced)
        loss = (0.25 * cos_loss +      # REDUCED from 0.30 (was dominating)
                0.40 * mse_loss +      # KEPT (magnitude alignment is critical)
                0.35 * qty_loss +      # INCREASED from 0.25 (qty needs more signal)
                0.00 * spk_reg)        # REMOVED (not needed in Phase 6a)'''
                
                # Replace old loss computation
                source_str = ''.join(cell['source'])
                if 'loss = (0.30 * cos_loss' in source_str:
                    # Find the loss computation block
                    lines = cell['source']
                    new_lines = []
                    skip_until_newline = False
                    
                    for i, line in enumerate(lines):
                        if 'loss = (0.30 * cos_loss' in line:
                            new_lines.append(new_loss + '\n')
                            skip_until_newline = True
                        elif skip_until_newline:
                            if line.strip() == '' or (not line.strip().startswith('0.') and ')' in line):
                                skip_until_newline = False
                                if ')' not in new_loss:
                                    new_lines.append(line)
                        else:
                            new_lines.append(line)
                    
                    cell['source'] = new_lines
                    changes_made += 1
                    print("✓ Fixed loss weights: rebalanced for quantity learning")
    
    if changes_made == 0:
        print("\n⚠ WARNING: No changes were made. The notebook may already be fixed,")
        print("  or the code structure has changed. Please apply fixes manually.")
        return False
    
    # Save modified notebook
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\n✓ Applied {changes_made} fixes to {notebook_path}")
    print(f"✓ Backup saved to {backup_path}")
    print("\nNEXT STEPS:")
    print("1. Delete old checkpoints: rm checkpoints/phase6a_connector_step*.pt")
    print("2. Restart Phase 6a training from step 0")
    print("3. Monitor: fired tokens should match target ± 3 tokens")
    print("4. Expect: quantity error < 3 tokens by step 1500")
    
    return True


def print_manual_fix_instructions():
    """Print manual fix instructions if automatic fix fails."""
    print("\n" + "="*80)
    print("  MANUAL FIX INSTRUCTIONS")
    print("="*80)
    print()
    print("If automatic fix failed, apply these changes manually:")
    print()
    print("1. CIF Connector Class (__init__):")
    print("   FIND:    threshold=0.50")
    print("   REPLACE: threshold=0.95")
    print()
    print("2. CIF Connector Forward (weight scaling):")
    print("   FIND:    alpha = raw_w / w_sum * qty_pred.unsqueeze(1)")
    print("   REPLACE: alpha = raw_w / w_sum * (0.8 * qty_pred.unsqueeze(1))")
    print()
    print("3. Phase 6a Optimizer:")
    print("   FIND:    'lr': 3e-4  (for cif_connector)")
    print("   REPLACE: 'lr': 2e-4")
    print()
    print("4. Phase 6a Loss Computation:")
    print("   FIND:    loss = (0.30 * cos_loss + 0.40 * mse_loss + 0.25 * qty_loss + ...)")
    print("   REPLACE: loss = (0.25 * cos_loss + 0.40 * mse_loss + 0.35 * qty_loss + 0.00 * spk_reg)")
    print()
    print("5. Residual Threshold:")
    print("   FIND:    if acc_w > 1e-6:")
    print("   REPLACE: if acc_w > 0.05:")
    print()
    print("6. Final Fire Threshold:")
    print("   FIND:    if acc_w > 0.1:  (in 'Fire remaining' section)")
    print("   REPLACE: if acc_w > 0.3:")
    print()
    print("="*80)


if __name__ == "__main__":
    print("="*80)
    print("  CIF OVER-FIRING FIX - AUTOMATIC APPLICATION")
    print("="*80)
    print()
    
    success = apply_cif_fix()
    
    if not success:
        print_manual_fix_instructions()
    else:
        print("\n✅ FIX APPLIED SUCCESSFULLY!")
        print()
        print("VERIFICATION:")
        print("  Run this in notebook to verify:")
        print("  >>> print(f'CIF threshold: {model_6a.cif_connector.threshold}')")
        print("  Expected output: CIF threshold: 0.95")
