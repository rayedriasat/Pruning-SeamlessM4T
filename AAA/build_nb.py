# Cell 34 still has the old patterns — replacement didn't take. Check why.
import json
nb = json.load(open('./pm_fixed.ipynb'))
src = ''.join(nb['cells'][34]['source'])
# Find the offending lines
for line in src.split('\n'):
    if 'labels=teacher_text_ids' in line or 'return_char_input_ids' in line:
        print(repr(line))