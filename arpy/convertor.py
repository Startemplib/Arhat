################################################### Convertor ###################################################

import re
import pandas as pd
from pathlib import Path

######## table Markdown -> Excel (table_mdTexcel) ########
### markdown_file      str | Path      The Markdown file to parse.
### excel_file         str | Path      Output .xlsx path for extracted tables.
### readme_file        str | Path      Output .txt path describing tables & marked blocks.
### mark_keyword       str             Case-insensitive keyword to collect blocks (in title path).
### ignore_br          bool            If True, drop lines containing "<br>".
### sheet_prefix       str             Optional prefix for sheet names.
### table_regex        str | None      Custom regex for GitHub-flavored MD tables; default is sensible.
### ensure_trailing_nl bool            Ensure file content ends with a newline before parsing.
### return_df          bool            Also return DataFrames in-memory (may be large).


def table_mdTexcel(
    markdown_file,
    excel_file="Data.xlsx",
    readme_file="Readme.txt",
    mark_keyword="mark",
    ignore_br=True,
    sheet_prefix="",
    table_regex=None,
    ensure_trailing_nl=True,
    return_df=False,
):
    """
    Extract all Markdown tables and export them to an Excel workbook; also
    write a README-like summary and collect blocks whose title path contains
    `mark_keyword`.

    Notes
    -----
    * Headings define hierarchical "title paths" like "H1 > H2 > H3".
    * Sheet names are derived from title paths plus a table index.
    * Table rows are right-padded to match the header width.

    Returns
    -------
    dict
        {
          "excel_path": Path,
          "readme_path": Path,
          "tables_written": int,
          "marked_blocks": list[str],
          "sheet_names": list[str],
          "dataframes": Optional[dict[str, pd.DataFrame]]
        }
    """
    markdown_file = Path(markdown_file)
    excel_file = Path(excel_file)
    readme_file = Path(readme_file)

    # --- read content
    content = markdown_file.read_text(encoding="utf-8")
    if ensure_trailing_nl and not content.endswith("\n"):
        content += "\n"
    content += "# OVER\n"  # end sentinel (safe no-op as a heading)

    # --- heading splitter
    title_pat = re.compile(r"^(#+)\s*(.+)$")
    lines = content.splitlines()
    sections = []
    current_h = []
    current_block = {"title_hierarchy": "", "content": []}

    for line in lines:
        m = title_pat.match(line)
        if m:
            if current_block["content"]:
                sections.append(current_block)
                current_block = {"title_hierarchy": "", "content": []}
            lvl = len(m.group(1))
            title = m.group(2).strip()
            while len(current_h) >= lvl:
                current_h.pop()
            current_h.append(title)
            current_block["title_hierarchy"] = " > ".join(current_h)
        else:
            if not (ignore_br and "<br>" in line):
                current_block["content"].append(line)

    if current_block["content"]:
        sections.append(current_block)

    # --- table regex (GitHub-flavored)
    table_pattern = re.compile(
        table_regex or r"(\|.+?\|\n\|[-:| ]+\|\n(?:\|.*?\|\n)+)",
        flags=re.DOTALL,
    )

    # --- export
    writer = pd.ExcelWriter(excel_file, engine="xlsxwriter")
    sheet_names = []
    table_descriptions = []
    marked_blocks = []
    dfs_out = {} if return_df else None

    def _sanitize_name(name: str) -> str:
        name = name.replace(">", "-").replace(" ", "")
        if sheet_prefix:
            name = f"{sheet_prefix}{name}"
        # Excel sheet name constraints
        bad = set(r"[]:*?/\\")
        name = "".join(ch for ch in name if ch not in bad)
        return name[:31] or "Sheet"

    tables_written = 0
    mk = (mark_keyword or "").lower()

    for idx, sec in enumerate(sections, start=1):
        block_text = "\n".join(sec["content"])
        title_path = sec["title_hierarchy"]

        if mk and mk in title_path.lower():
            marked_blocks.append(
                f"Title: {title_path}\nContent:\n{block_text.strip()}\n"
            )

        tables = table_pattern.findall(block_text)
        for t_i, table_md in enumerate(tables, start=1):
            rows = table_md.strip().split("\n")
            headers = [c.strip() for c in rows[0].split("|") if c.strip()]
            data_rows = [
                [c.strip() for c in r.split("|") if c.strip()] for r in rows[2:]
            ]
            # broadcast/pad
            width = len(headers)
            data_rows = [
                row + [""] * (width - len(row)) if len(row) < width else row
                for row in data_rows
            ]

            df = pd.DataFrame(data_rows, columns=headers)

            base = _sanitize_name(title_path)
            sheet = f"{base}_T{t_i}"
            if len(sheet) > 31:
                sheet = f"Block_{idx}_T{t_i}"
            # guard collisions
            original_sheet = sheet
            k = 1
            while sheet in sheet_names:
                suffix = f"_{k}"
                sheet = (original_sheet[: 31 - len(suffix)]) + suffix
                k += 1

            df.to_excel(writer, sheet_name=sheet, index=False)
            tables_written += 1
            sheet_names.append(sheet)
            table_descriptions.append(
                f"{sheet}: extracted from '{title_path}' with columns: {', '.join(headers)}"
            )
            if return_df:
                dfs_out[sheet] = df

    writer.close()

    # --- README / notes
    with readme_file.open("w", encoding="utf-8") as f:
        f.write("Extracted table structures:\n")
        f.write("\n".join(table_descriptions) if table_descriptions else "(none)")
        f.write("\n\nMarked blocks:\n")
        f.write("\n---\n".join(marked_blocks) if marked_blocks else "(none)")

    return {
        "excel_path": excel_file,
        "readme_path": readme_file,
        "tables_written": tables_written,
        "marked_blocks": marked_blocks,
        "sheet_names": sheet_names,
        "dataframes": dfs_out,
    }
