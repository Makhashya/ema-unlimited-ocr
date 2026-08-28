r"""Merge the per-image *_table.md files into one CSV.

Reads every markdown table produced by md_to_table.py and writes a single
spreadsheet with one row per equipment unit across all images.

    output_table_Mechanical_input\*.md
        -> output_csv_Mechanical_input\Mechanical_input.csv

Usage:
    python tables_to_csv.py --table_dir output_table_Mechanical_input
    python tables_to_csv.py --table_dir output_table_Mechanical_input --out all.csv
"""

import argparse
import csv
import os
import re
import sys

SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
# split on pipes that are not backslash-escaped
CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def csv_path_for(table_dir: str) -> str:
    """output_table_Mechanical_input
       -> output_csv_Mechanical_input\\Mechanical_input.csv (beside the folder)
    """
    p = os.path.abspath(table_dir)
    name = os.path.basename(p)
    base = name[len("output_table_"):] if name.startswith("output_table_") else name
    return os.path.join(os.path.dirname(p), f"output_csv_{base}", f"{base}.csv")


def split_row(line: str):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip().replace("\\|", "|") for c in CELL_SPLIT_RE.split(s)]


def is_separator(cells) -> bool:
    non_empty = [c for c in cells if c]
    return bool(non_empty) and all(SEP_CELL_RE.match(c) for c in non_empty)


def parse_table(text: str):
    """Return (header, rows) from the first markdown table in the text."""
    header, rows = None, []
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = split_row(line)
        if header is None:
            header = cells
            continue
        if is_separator(cells):
            continue
        if any(c for c in cells):          # skip fully blank rows
            rows.append(cells)
    return header, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table_dir", required=True,
                    help="folder of *_table.md files, e.g. output_table_ujju")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: creates 'output_csv_<name>' "
                         "beside the table folder and writes '<name>.csv' inside)")
    ap.add_argument("--encoding", default="utf-8-sig",
                    help="CSV encoding; utf-8-sig (default) opens cleanly in Excel")
    args = ap.parse_args()

    if not os.path.isdir(args.table_dir):
        sys.exit(f"not a folder: {args.table_dir}")

    files = sorted((f for f in os.listdir(args.table_dir) if f.lower().endswith(".md")),
                   key=natural_key)
    if not files:
        sys.exit(f"no .md files found in {args.table_dir}")

    out_path = args.out or csv_path_for(args.table_dir)

    columns, records, empty = [], [], []
    for name in files:
        with open(os.path.join(args.table_dir, name), encoding="utf-8") as f:
            header, rows = parse_table(f.read())
        if not header:
            print(f"  {name}: no table found, skipped")
            continue
        for col in header:                  # union, in first-seen order
            if col and col not in columns:
                columns.append(col)
        if not rows:
            empty.append(name)
            continue
        for cells in rows:
            # tolerate a row that is short or long relative to its header
            record = {h: (cells[i] if i < len(cells) else "")
                      for i, h in enumerate(header) if h}
            record["__src__"] = name
            records.append(record)

    if not columns:
        sys.exit("no tables could be parsed")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="", encoding=args.encoding) as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    print(f"{len(records)} row(s) from {len(files) - len(empty)} table(s) "
          f"-> {out_path}")
    if empty:
        print(f"no data rows in: {', '.join(empty)}")


if __name__ == "__main__":
    main()
