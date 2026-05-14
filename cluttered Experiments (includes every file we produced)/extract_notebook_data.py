import json
import sys

def extract_notebook_outputs(notebook_path):
    """Extract cell outputs from a Jupyter notebook"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    results = []
    for i, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type', '')
        source = ''.join(cell.get('source', []))
        
        # Extract outputs
        if cell_type == 'code':
            outputs = cell.get('outputs', [])
            output_text = []
            for output in outputs:
                if 'text' in output:
                    output_text.append(''.join(output['text']))
                elif 'data' in output and 'text/plain' in output['data']:
                    output_text.append(''.join(output['data']['text/plain']))
            
            if output_text or ('phase' in source.lower() or 'benchmark' in source.lower() or 'bleu' in source.lower() or 'chrf' in source.lower()):
                results.append({
                    'cell_num': i,
                    'source_preview': source[:200] if len(source) > 200 else source,
                    'outputs': '\n'.join(output_text)[:2000] if output_text else 'No output'
                })
    
    return results

if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    notebook = sys.argv[1] if len(sys.argv) > 1 else 'seamless-cse465v5.ipynb'
    results = extract_notebook_outputs(notebook)
    
    print(f"=== Extracted from {notebook} ===\n")
    for r in results:
        print(f"\n--- Cell {r['cell_num']} ---")
        print(f"Source: {r['source_preview']}")
        print(f"Output: {r['outputs'][:500]}")
        print("-" * 80)
