import json

def notebook_to_text(ipynb_path, output_txt_path):
    # Load the notebook
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook.get("cells", [])

    with open(output_txt_path, 'w', encoding='utf-8') as out:
        for i, cell in enumerate(cells, start=1):
            out.write(f"Cell{i}:\n")

            # Get cell content
            source = cell.get("source", [])

            # Join lines (they are usually a list of strings)
            if isinstance(source, list):
                out.write("".join(source))
            else:
                out.write(source)

            out.write("\n\n")  # spacing between cells

    print(f"Saved to {output_txt_path}")


# Example usage
notebook_to_text("./bengali-seamless-notebook.ipynb", "bengali-seamless-notebook.ipynb.txt")