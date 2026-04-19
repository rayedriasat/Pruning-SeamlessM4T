#!/usr/bin/env python3
"""Extract key information from Jupyter notebooks for the research report."""

import json
import re
from pathlib import Path

def extract_markdown_cells(notebook_path):
    """Extract all markdown cells from a notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    markdown_cells = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            content = ''.join(cell['source'])
            markdown_cells.append((i, content))
    
    return markdown_cells

def extract_phase_headers(notebook_path):
    """Extract phase headers and their cell indices."""
    markdown_cells = extract_markdown_cells(notebook_path)
    phases = []
    
    for idx, content in markdown_cells:
        # Look for phase headers
        if re.search(r'(PHASE|Phase)\s+\d+', content, re.IGNORECASE):
            # Extract first few lines
            lines = content.split('\n')[:5]
            phases.append({
                'cell_index': idx,
                'header': '\n'.join(lines)
            })
    
    return phases

def extract_benchmark_tables(notebook_path):
    """Extract benchmark result tables from markdown cells."""
    markdown_cells = extract_markdown_cells(notebook_path)
    tables = []
    
    for idx, content in markdown_cells:
        # Look for tables with benchmark data
        if '|' in content and any(keyword in content.lower() for keyword in ['bleu', 'chrf', 'params', 'benchmark']):
            tables.append({
                'cell_index': idx,
                'content': content
            })
    
    return tables

def main():
    notebooks = ['seamless-cse465v5.ipynb', 'only-p7-dora.ipynb']
    
    for nb_path in notebooks:
        if not Path(nb_path).exists():
            print(f"Skipping {nb_path} - not found")
            continue
            
        print(f"\n{'='*80}")
        print(f"NOTEBOOK: {nb_path}")
        print('='*80)
        
        # Extract phases
        print("\n--- PHASE STRUCTURE ---")
        phases = extract_phase_headers(nb_path)
        for phase in phases:
            print(f"\nCell {phase['cell_index']}:")
            print(phase['header'])
            print('-' * 40)
        
        # Extract benchmark tables
        print("\n--- BENCHMARK TABLES ---")
        tables = extract_benchmark_tables(nb_path)
        for table in tables[:5]:  # Limit to first 5 tables
            print(f"\nCell {table['cell_index']}:")
            print(table['content'][:500])  # First 500 chars
            print('-' * 40)

if __name__ == '__main__':
    main()
