import json

with open('Alteration/seamless-final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '# ADJUSTED LOSS WEIGHTS' in source or ('loss =' in source and '0.40' in source and 'cos_loss' in source):
            print(f"\n{'='*80}\nCELL {i} - FULL LOSS SECTION\n{'='*80}")
            lines = source.split('\n')
            for j, line in enumerate(lines):
                if 'ADJUSTED LOSS WEIGHTS' in line or ('loss =' in line and 'cos_loss' in line):
                    start = max(0, j-5)
                    end = min(len(lines), j+15)
                    for k in range(start, end):
                        print(f"{k:4d}: {lines[k]}")
                    print()
                    break
