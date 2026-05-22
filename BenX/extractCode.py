import json
import re


def _extract_output_text(output: dict) -> str:
    """Return a markdown-friendly string for a single cell output entry."""
    output_type = output.get("output_type", "")

    if output_type in ("stream",):
        text = output.get("text", [])
        content = "".join(text) if isinstance(text, list) else text
        stream_name = output.get("name", "stdout")
        return f"**[{stream_name}]**\n```\n{content.rstrip()}\n```\n"

    if output_type in ("execute_result", "display_data"):
        data = output.get("data", {})
        parts = []

        # Prefer plain text; fall back to other mime types
        if "text/plain" in data:
            plain = data["text/plain"]
            content = "".join(plain) if isinstance(plain, list) else plain
            parts.append(f"```\n{content.rstrip()}\n```")

        if "text/html" in data:
            html = data["text/html"]
            content = "".join(html) if isinstance(html, list) else html
            # Strip tags for a readable summary
            stripped = re.sub(r"<[^>]+>", "", content).strip()
            if stripped:
                parts.append(f"*[HTML output — plain text preview]*\n```\n{stripped}\n```")

        if "image/png" in data:
            parts.append("*[Image output — PNG]*")

        if "image/jpeg" in data:
            parts.append("*[Image output — JPEG]*")

        return "\n".join(parts) + "\n" if parts else ""

    if output_type == "error":
        ename = output.get("ename", "Error")
        evalue = output.get("evalue", "")
        traceback_lines = output.get("traceback", [])
        # Strip ANSI escape codes from traceback
        ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
        clean_tb = "\n".join(
            ansi_escape.sub("", line) for line in traceback_lines
        )
        return f"**[ERROR]** `{ename}: {evalue}`\n```\n{clean_tb}\n```\n"

    return ""


def notebook_to_markdown(ipynb_path: str, output_md_path: str) -> None:
    """Convert a Jupyter notebook to a well-structured Markdown file,
    including both cell source code and cell outputs."""

    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    cells = notebook.get("cells", [])
    kernel_name = (
        notebook.get("metadata", {})
        .get("kernelspec", {})
        .get("display_name", "unknown")
    )

    with open(output_md_path, "w", encoding="utf-8") as out:
        # Notebook header
        out.write(f"# Notebook: `{ipynb_path}`\n\n")
        out.write(f"> Kernel: **{kernel_name}** | Total cells: **{len(cells)}**\n\n")
        out.write("---\n\n")

        for i, cell in enumerate(cells, start=1):
            cell_type = cell.get("cell_type", "unknown")
            execution_count = cell.get("execution_count", None)

            # --- Section header ---
            ec_str = f" (execution #{execution_count})" if execution_count is not None else ""
            out.write(f"## Cell {i} — `{cell_type}`{ec_str}\n\n")

            # --- Source ---
            source = cell.get("source", [])
            code = "".join(source) if isinstance(source, list) else source

            if cell_type == "markdown":
                # Render markdown cells as a block-quote so they stand out
                quoted = "\n".join(f"> {line}" for line in code.splitlines())
                out.write(f"{quoted}\n\n")
            else:
                # Detect language for syntax highlighting
                lang = (
                    notebook.get("metadata", {})
                    .get("kernelspec", {})
                    .get("language", "python")
                )
                out.write(f"```{lang}\n{code}\n```\n\n")

            # --- Outputs ---
            outputs = cell.get("outputs", [])
            if outputs:
                out.write("### Output\n\n")
                for output in outputs:
                    rendered = _extract_output_text(output)
                    if rendered:
                        out.write(rendered + "\n")

            out.write("---\n\n")

    print(f"Saved to {output_md_path}")


# Example usage
notebook_to_markdown("./bengali-seamless-notebook.ipynb", "bengali-seamless-notebook.ipynb.md")