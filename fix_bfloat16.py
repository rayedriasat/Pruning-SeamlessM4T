import json

# Load the notebook
with open('cse465v6-s2st-optimised.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix the run_s2st function
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    if 'waveform = out.waveform.cpu().numpy().squeeze()' in source:
        # Replace the problematic line
        old_line = '            waveform = out.waveform.cpu().numpy().squeeze() if out.waveform is not None else np.zeros(16000)'
        new_line = '            waveform = out.waveform.cpu().float().numpy().squeeze() if out.waveform is not None else np.zeros(16000)'
        
        new_source = source.replace(old_line, new_line)
        
        # Split into lines for notebook format
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        # Remove trailing newline from last line
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
        
        print(f"Fixed BFloat16 issue in cell {i}")
        print(f"Changed: waveform = out.waveform.cpu().numpy().squeeze()")
        print(f"To:      waveform = out.waveform.cpu().float().numpy().squeeze()")
        break

# Save the modified notebook
with open('cse465v6-s2st-optimised.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nNotebook fixed successfully!")
print("\nThe fix converts BFloat16 tensors to Float32 before converting to numpy.")
print("This is necessary because NumPy doesn't support BFloat16 dtype.")
