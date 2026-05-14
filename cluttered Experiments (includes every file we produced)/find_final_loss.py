import json

with open('Alteration/seamless-final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        # Look for the final loss aggregation
        if 'torch.stack(batch_cos_loss).mean()' in source or 'Average losses across batch' in source:
            print(f"\n{'='*80}\nCELL {i} - FINAL LOSS AGGREGATION\n{'='*80}")
            lines = source.split('\n')
            for j, line in enumerate(lines):
                if 'Average losses' in line or 'torch.stack(batch_cos_loss)' in line:
                    start = max(0, j-5)
                    end = min(len(lines), j+25)
                    for k in range(start, end):
                        print(f"{k:4d}: {lines[k]}")
                    print()
                    break
