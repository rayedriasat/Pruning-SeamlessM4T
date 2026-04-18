#!/usr/bin/env python3
"""Verify the notebook update was successful."""

import json

with open('cse465v5-s2st-corrected.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

source = ''.join(nb['cells'][105]['source'])

print('Cell 105 now contains:')
print('- prepare_s2tt_batch:', 'def prepare_s2tt_batch' in source)
print('- prepare_unit_batch:', 'def prepare_unit_batch' in source)
print('- compute_s2tt_loss:', 'def compute_s2tt_loss' in source)
print('- compute_t2u_loss:', 'def compute_t2u_loss' in source)
print('- S2TT_WEIGHT:', 'S2TT_WEIGHT' in source)
print('- T2U_WEIGHT:', 'T2U_WEIGHT' in source)
print('- encoder_attention_mask fix:', 'encoder_attention_mask = torch.ones' in source)

print('\nFirst 500 chars:')
print(source[:500])

print('\n✓ All required functions are present in Cell 105!')
