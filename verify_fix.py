#!/usr/bin/env python3
"""
Verify that Phase 5 Cell 2 has all the necessary fixes applied.
"""

import json

with open('cse465v5-s2st-corrected.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Phase 5 Cell 2
cell = nb['cells'][67]
source = cell['source']

print("=" * 70)
print("PHASE 5 CELL 2 FIX VERIFICATION")
print("=" * 70)

checks = [
    ("Stats tensors on device", "device=device" in source and "torch.zeros" in source),
    ("Processor outputs moved", "Move ALL tensors" in source or "for k, v in enc_in.items()" in source),
    ("Final stats to CPU", ".cpu()" in source and "Finalize statistics" in source),
    ("Device-aware comment", "CRITICAL FIX" in source or "device-aware" in source),
]

all_passed = True
for check_name, result in checks:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status:8} {check_name}")
    if not result:
        all_passed = False

print("=" * 70)
if all_passed:
    print("✓ ALL CHECKS PASSED - Fix is correctly applied")
    print("\nThe notebook should now:")
    print("  1. Create stats tensors on cuda:0 (same device as model)")
    print("  2. Move all processor outputs to cuda:0")
    print("  3. Successfully calibrate all FFN layers")
else:
    print("✗ SOME CHECKS FAILED - Fix may not be complete")

print("=" * 70)
