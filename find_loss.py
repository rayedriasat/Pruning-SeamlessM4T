import json

with open('Alteration/seamless-final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'cos_loss' in source and 'mse_loss' in source and 'qty_loss' in source and 'loss =' in source:
            print(f"\n{'='*80}\nCELL {i} - LOSS COMPUTATION\n{'='*80}")
            lines = source.split('\n')
            for j, line in enumerate(lines):
                if 'loss =' in line and ('cos' in line or 'mse' in line):
                    start = max(0, j-10)
                    end = min(len(lines), j+5)
                    for k in range(start, end):
                        print(f"{k:4d}: {lines[k]}")
                    print()
                    break
