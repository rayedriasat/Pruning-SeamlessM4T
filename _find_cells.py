import json

with open('cse465-approach2v3-compression.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
for idx in [7, 13, 21, 26, 27]:
    c = cells[idx]
    src = ''.join(c.get('source', []))
    ct = c['cell_type']
    print(f"=== Cell {idx} ({ct}) ===")
    print(src)
    print()
