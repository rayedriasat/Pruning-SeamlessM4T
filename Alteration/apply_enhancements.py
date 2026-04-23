#!/usr/bin/env python3
"""
Script to programmatically add enhanced tracking to seamless-final.ipynb
Reads the notebook JSON, inserts new cells, and saves the updated version
"""

import json
import sys

# Read the enhanced tracking code
with open('Alteration/quick_integration_snippet.py', 'r', encoding='utf-8') as f:
    enhanced_tracking_code = f.read()

# Read the notebook
with open('Alteration/seamless-final.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Loaded notebook with {len(notebook['cells'])} cells")

# Find the cell with ALL_SUMMARIES definition
summary_cell_idx = None
for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'ALL_SUMMARIES' in source and '_load_summaries_from_drive' in source:
            summary_cell_idx = idx
            print(f"Found summary cell at index {idx}")
            break

if summary_cell_idx is None:
    print("ERROR: Could not find summary cell")
    sys.exit(1)

# Create new cell with enhanced tracking
new_cell = {
    "cell_type": "code",
    "source": enhanced_tracking_code,
    "metadata": {
        "trusted": True
    },
    "outputs": [],
    "execution_count": None
}

# Insert after the summary cell
notebook['cells'].insert(summary_cell_idx + 1, new_cell)
print(f"Inserted enhanced tracking cell at index {summary_cell_idx + 1}")

# Now update benchmark cells to use detailed tracking
# We'll add helper text to guide manual updates since full automation is complex

update_guide_cell = {
    "cell_type": "markdown",
    "source": [
        "## 📊 Enhanced Per-Language Tracking Enabled\n\n",
        "**New functions available:**\n",
        "- `compute_detailed_summary(results, label, params_M)` - Extract per-language metrics\n",
        "- `store_detailed_summary(summary)` - Save to checkpoint\n",
        "- `plot_detailed_phase_comparison()` - 9-panel visualization\n",
        "- `print_detailed_summary_table(phase_label)` - Text output\n\n",
        "**To use in benchmark cells:**\n",
        "```python\n",
        "# After running benchmark\n",
        "p0_results, p0_summary = run_benchmark_asr(model, samples, 'P0_Label', save_n=4)\n",
        "p0_detailed = compute_detailed_summary(p0_results, 'P0_Label', p0_summary['params_M'])\n\n",
        "# Save both\n",
        "save_checkpoint({\n",
        "    'results': p0_results,\n",
        "    'summary': p0_summary,\n",
        "    'detailed_summary': p0_detailed  # NEW\n",
        "}, 'phase0_benchmark', 0)\n\n",
        "store_summary(p0_summary)\n",
        "store_detailed_summary(p0_detailed)  # NEW\n",
        "print_detailed_summary_table('P0_Label')  # NEW\n",
        "plot_detailed_phase_comparison()  # NEW\n",
        "```\n",
        "\n",
        "**All per-language data now preserved in checkpoints!**"
    ],
    "metadata": {}
}

notebook['cells'].insert(summary_cell_idx + 2, update_guide_cell)
print(f"Inserted guide cell at index {summary_cell_idx + 2}")

# Save the updated notebook
output_path = 'Alteration/seamless-final-enhanced.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"\n✓ Enhanced notebook saved to: {output_path}")
print(f"  Total cells: {len(notebook['cells'])}")
print(f"\nNext steps:")
print(f"  1. Review the new cells in the notebook")
print(f"  2. Update benchmark cells to call compute_detailed_summary()")
print(f"  3. Add plot_detailed_phase_comparison() calls after benchmarks")
print(f"  4. Run the notebook to generate enhanced visualizations")
