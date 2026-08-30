r"""Single-file pipeline: equipment nameplate images -> markdown -> CSV.

Everything runs through the already-authenticated `claude` CLI (Claude Code),
so no API key is needed. For each image the CLI is called three times:

  1. transcribe   the image is read by Claude's vision and every character of
                  text on the nameplate is saved to  <stem>_text.md
  2. tabulate     the transcription is turned into one markdown equipment-
                  schedule table
  3. verify       Claude re-opens the ORIGINAL image, checks every cell of the
                  draft table against what is actually printed, corrects any
                  mistakes, and fills in blanks it can genuinely read or
                  derive; the verified table is saved to  <stem>_table.md
                  (skip this pass with --no-verify)

Finally all per-image tables are merged into one CSV with the columns:

  Image | Tag # | Make | Model Number | Serial Number | Refrigerant Type |
  Manufacture Date | Approximate Tonnage | MCA/MOCP | Voltage/Phase |
  Condition  (G/M/P) | Comments

Outputs land beside the input folder, matching the project layout:

    Mechanical_input\                                (your images)
    Output_Mechanical_input\
        output_text_Mechanical_input\   *_text.md
        output_table_Mechanical_input\  *_table.md
        output_csv_Mechanical_input\    Mechanical_input.csv

Usage:
    python equipment_pipeline.py --image_dir Mechanical_input
    python equipment_pipeline.py --image_dir Mechanical_input --model claude-sonnet-5
    python equipment_pipeline.py --image_dir Mechanical_input --skip-existing
    python equipment_pipeline.py --image_dir Mechanical_input --no-verify
    python equipment_pipeline.py --image_dir Mechanical_input --limit 2   # smoke test
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

COLUMNS = ("Tag #", "Make", "Model Number", "Serial Number", "Refrigerant Type",
           "Manufacture Date", "Approximate Tonnage", "MCA/MOCP",
           "Voltage/Phase", "Condition  (G/M/P)", "Comments")

TRANSCRIBE_PROMPT = """\
You are a meticulous transcription engine for equipment nameplate photos.
Read the image file you are given and transcribe ALL text visible in it,
exactly as printed.

Rules:
- Preserve every character of model numbers, serial numbers, part numbers,
  and electrical ratings exactly as shown. Never "correct", normalize, or
  guess a character. If a character is truly unreadable, write ? in its place.
- Keep each label with its value on one line, in the form "MODEL NO: <the
  characters printed on the plate>". Only ever write values you can actually
  see in the image.
- Transcribe printed, stamped, embossed, and handwritten text, including
  stickers and secondary labels.
- Output plain text only: no commentary, no descriptions of the image, no
  markdown code fences, no "Here is the transcription".
- If the image contains no readable text, output nothing."""

TABLE_PROMPT = """\
You extract HVAC equipment nameplate data from OCR text and return it as a \
single markdown table. You output the table and nothing else -- no preamble, \
no explanation, no closing remarks, no code fences.

The table has exactly these columns, in this order:

Tag # | Make | Model Number | Serial Number | Refrigerant Type | Manufacture \
Date | Approximate Tonnage | MCA/MOCP | Voltage/Phase | Condition  (G/M/P) | \
Comments

Rules:

- One row per physical unit found in the text. If the text describes a single \
nameplate, emit exactly one row.
- Begin and end every row, including the header and separator, with a pipe.
- Tag #: use the equipment tag if the text gives one (e.g. CU5, AH2, RTU-3). \
If there is no tag, write the unit-type abbreviation followed by ( no tag ), \
e.g. "CU ( no tag )". Use these abbreviations: AH for an air handler, air \
handling unit, furnace, or indoor evaporator/cased coil; CU for a condensing \
unit or heat pump outdoor unit; RTU for a packaged rooftop unit; FC for a fan \
coil; B for a boiler; P for a pump; EF for an exhaust fan.
- Manufacture Date: the year. If it is not stated outright but the serial \
number encodes it in the manufacturer's scheme, decode it -- e.g. Lennox \
serial 1519F31081 -> 2019 (week 15 of 2019), Ducane serial 1911G22987 -> \
2011. These are examples, not the only supported makes: apply the same idea \
to any manufacturer whose serial or tag convention you know (Trane, Carrier, \
York, Goodman, Rheem, smaller specialty makers, ...), including serials \
whose leading digits plainly read as a week-year, month-year, or year (e.g. \
02-02-... -> 2002). Fill the year whenever the decoding is clear; leave it \
blank only when the serial is genuinely ambiguous.
- Approximate Tonnage: whole tons, derived from the capacity digits in the \
model number when present -- e.g. -036- or 036 -> 3, 048 -> 4, 060 -> 5. \
Give the number only, with no unit.
- Refrigerant Type: copy the designation as printed, e.g. HFC-410A, HCFC-22.
- MCA/MOCP: combine as "MCA/MOCP" values separated by a slash, e.g. 18.6/30.
- Voltage/Phase: as printed, e.g. 208/230/1.
- Condition (G/M/P): only fill this in if the text actually assesses the \
unit's condition. Otherwise leave it blank.
- Leave any cell blank when the source text does not supply the value. Never \
invent, guess, or infer a value that is not supported by the text, and never \
write placeholders like N/A, unknown, or --.
- Preserve the OCR's exact characters for model and serial numbers. Do not \
"correct" them.

Example row (a Lennox air handler with no tag on it):

| AH ( no tag ) | Lennox | CX35-36A-6F-20 | 1519F31081 | HFC-410A | 2019 | 3 \
|  |  |  |  |

If the text contains no equipment nameplate data at all, output the table \
header and separator row with no data rows."""

VERIFY_PROMPT = TABLE_PROMPT + """

You are on a VERIFICATION pass. You are given the path to the original \
nameplate photo, the raw transcription of it, and a draft table that was \
built from that transcription. Use the Read tool to open the image and look \
at the nameplate yourself, then:

- Check every filled cell of the draft against what is actually printed in \
the image. Fix any cell the image contradicts -- especially model and serial \
numbers, where the draft may contain OCR character errors (0/O, 1/I, 5/S, \
8/B) or ? placeholders you can now resolve by looking closely at the plate.
- Hunt for missing information: for every blank cell, search the image \
(including stickers, secondary labels, and stamped text) for the value. Fill \
the cell only when you can genuinely read the value in the image or derive \
it under the rules above (year from the serial number, tonnage from the \
model number). A cell whose value simply is not present must stay blank.
- Do not drop or add rows unless the image clearly shows a different number \
of physical units than the draft has rows.
- Keep exactly the same columns, in the same order.

You may also use OUTSIDE INFORMATION -- your own knowledge of manufacturers, \
and the WebSearch/WebFetch tools -- to complete the verified plate data, \
under strict conditions:

- Use it to decode what the plate already gives you: the manufacturer's \
serial-number date scheme, the capacity/tonnage digits in the model number, \
the refrigerant a specific model line shipped with, or the full company name \
behind a logo or abbreviation on the plate.
- Only accept an outside fact when it is specific to THIS make and exact \
model/serial as read from the image, and the answer is unambiguous. If a \
search returns conflicting schemes, a different model variant, or a "similar" \
model, do not use it -- leave the cell as the plate supports.
- Never copy ratings (MCA, MOCP, voltage, refrigerant) from a web listing \
into a cell the plate itself contradicts; the plate always wins.
- Cells for values that are neither on the plate nor confidently derivable \
stay blank.

Output ONLY the corrected markdown table -- no commentary, no explanation of \
what you changed, no code fences."""

FENCE_RE = re.compile(r"^\s*```[^\n]*\n(.*?)\n\s*```\s*$", re.DOTALL)
SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")   # pipes that are not backslash-escaped

# Tools the CLI must never use in any stage. Read is allowed only when a
# stage needs to open the image; WebSearch/WebFetch only in the verify stage.
BLOCKED_TOOLS = ["Bash", "Write", "Edit", "NotebookEdit", "Glob", "Grep",
                 "Task", "TodoWrite"]


# --------------------------------------------------------------------------
# claude CLI plumbing
# --------------------------------------------------------------------------

def find_claude(explicit: str = None) -> str:
    """Locate the claude CLI.

    A terminal opened before Claude Code was installed has a stale PATH, so
    fall back to the usual install locations before giving up.
    """
    for cand in (explicit, os.environ.get("CLAUDE_CLI")):
        if cand:
            if os.path.isfile(cand):
                return cand
            found = shutil.which(cand)
            if found:
                return found
            sys.exit(f"claude CLI not found at: {cand}")

    found = shutil.which("claude")
    if found:
        return found

    home = os.path.expanduser("~")
    for cand in (
        os.path.join(home, ".local", "bin", "claude.exe"),
        os.path.join(home, ".local", "bin", "claude"),
        os.path.join(os.environ.get("APPDATA", ""), "npm", "claude.cmd"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "claude",
                     "claude.exe"),
    ):
        if cand and os.path.isfile(cand):
            return cand

    sys.exit(
        "`claude` CLI not found on PATH.\n"
        "  - If Claude Code is installed, open a NEW terminal (this one has a\n"
        "    stale PATH) and try again.\n"
        "  - Or point at it directly:  --claude \"C:\\path\\to\\claude.exe\"\n"
        "  - Or set the CLAUDE_CLI environment variable."
    )


def strip_fence(text: str) -> str:
    m = FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text.strip()


def run_claude(claude: str, prompt: str, system_prompt: str, model: str,
               timeout: int, read_dir: str = None, web: bool = False) -> str:
    """One prompt -> one text response through `claude -p`.

    read_dir, when given, allows the Read tool (and grants access to that
    directory) so Claude's vision can open the image file named in the prompt.
    web additionally allows WebSearch/WebFetch, for the verify stage to look
    up manufacturer serial/model schemes. Everything else stays disallowed.
    """
    allowed, blocked = [], list(BLOCKED_TOOLS)
    if read_dir:
        allowed.append("Read")
    else:
        blocked.append("Read")
    if web:
        allowed += ["WebSearch", "WebFetch"]
    else:
        blocked += ["WebSearch", "WebFetch"]

    cmd = [claude, "-p",
           "--system-prompt", system_prompt,
           "--model", model,
           "--output-format", "text",
           "--no-session-persistence"]
    if read_dir:
        cmd += ["--add-dir", read_dir]
    if allowed:
        cmd += ["--allowed-tools", *allowed]
    cmd += ["--disallowed-tools", *blocked]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "").strip()[:500])
    return strip_fence(proc.stdout)


# --------------------------------------------------------------------------
# stage 1: image -> transcription markdown
# --------------------------------------------------------------------------

def transcribe_image(claude: str, image_path: str, model: str,
                     timeout: int) -> str:
    image_path = os.path.abspath(image_path)
    prompt = (f"Read the image file at {image_path} and transcribe all text "
              f"in it, following your transcription rules.")
    return run_claude(claude, prompt, TRANSCRIBE_PROMPT, model, timeout,
                      read_dir=os.path.dirname(image_path))


# --------------------------------------------------------------------------
# stage 2: transcription -> markdown table
# --------------------------------------------------------------------------

def add_image_column(table: str, image: str) -> str:
    """Prepend an 'Image' column as the first column of a markdown table.

    Done here rather than in the prompt so the value is exact on every row.
    """
    out, header_done = [], False
    for line in table.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            out.append(line)
            continue
        cells = s.strip("|").split("|")
        non_empty = [c.strip() for c in cells if c.strip()]
        if not header_done:
            cells.insert(0, " Image ")
            header_done = True
        elif non_empty and all(SEP_CELL_RE.match(c) for c in non_empty):
            cells.insert(0, "---")
        else:
            cells.insert(0, f" {image} ")
        out.append("|" + "|".join(cells) + "|")
    return "\n".join(out)


def extract_table(claude: str, transcription: str, model: str,
                  timeout: int) -> str:
    return run_claude(claude, transcription, TABLE_PROMPT, model, timeout)


# --------------------------------------------------------------------------
# stage 3: re-check the table against the original image
# --------------------------------------------------------------------------

def verify_table(claude: str, image_path: str, transcription: str,
                 draft_table: str, model: str, timeout: int) -> str:
    """Second look: correct and complete the draft table from the image.

    Falls back to the draft when the verifier returns something that no
    longer parses as a table, so a bad verification pass can never lose data.
    """
    image_path = os.path.abspath(image_path)
    prompt = (f"Original nameplate photo: {image_path}\n"
              f"(open it with the Read tool)\n\n"
              f"Raw transcription of the photo:\n\n{transcription}\n\n"
              f"Draft table to verify and complete:\n\n{draft_table}")
    checked = run_claude(claude, prompt, VERIFY_PROMPT, model, timeout,
                         read_dir=os.path.dirname(image_path), web=True)
    header, _ = parse_table(checked)
    return checked if header else draft_table


# --------------------------------------------------------------------------
# stage 3: markdown tables -> one CSV
# --------------------------------------------------------------------------

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


def tables_to_csv(table_dir: str, out_path: str, encoding: str = "utf-8-sig"):
    files = sorted((f for f in os.listdir(table_dir)
                    if f.lower().endswith(".md")), key=natural_key)
    columns, records, empty = ["Image", *COLUMNS], [], []
    for name in files:
        with open(os.path.join(table_dir, name), encoding="utf-8") as f:
            header, rows = parse_table(f.read())
        if not header:
            print(f"  {name}: no table found, skipped")
            continue
        for col in header:                 # tolerate extra columns from Claude
            if col and col not in columns:
                columns.append(col)
        if not rows:
            empty.append(name)
            continue
        for cells in rows:
            # tolerate a row that is short or long relative to its header
            records.append({h: (cells[i] if i < len(cells) else "")
                            for i, h in enumerate(header) if h})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding=encoding) as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    print(f"{len(records)} row(s) from {len(files) - len(empty)} table(s) "
          f"-> {out_path}")
    if empty:
        print(f"no data rows in: {', '.join(empty)}")


# --------------------------------------------------------------------------
# glue
# --------------------------------------------------------------------------

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def paths_for(image_dir: str):
    """Every output path, from the input folder alone (project convention)."""
    src = os.path.abspath(image_dir)
    name = os.path.basename(src)
    root = os.path.join(os.path.dirname(src), f"Output_{name}")
    return {
        "name": name,
        "images": src,
        "root": root,
        "text": os.path.join(root, f"output_text_{name}"),
        "table": os.path.join(root, f"output_table_{name}"),
        "csv": os.path.join(root, f"output_csv_{name}", f"{name}.csv"),
    }


def main():
    ap = argparse.ArgumentParser(
        description="nameplate images -> markdown -> equipment-schedule CSV, "
                    "all through the claude CLI")
    ap.add_argument("--image_dir", required=True, help="folder of input images")
    ap.add_argument("--model", default="claude-opus-5",
                    help="Claude model for both stages (default: claude-opus-5)")
    ap.add_argument("--claude", default=None,
                    help="path to the claude CLI if it is not on PATH")
    ap.add_argument("--timeout", type=int, default=600,
                    help="seconds per CLI call (default: 600)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the verification pass that re-checks each "
                         "table against the original image (faster, cheaper)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse *_text.md / *_table.md files that already exist")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N images (0 = all)")
    args = ap.parse_args()

    if not os.path.isdir(args.image_dir):
        sys.exit(f"not a folder: {args.image_dir}")
    claude = find_claude(args.claude)

    images = sorted((f for f in os.listdir(args.image_dir)
                     if os.path.splitext(f)[1].lower() in EXTS),
                    key=natural_key)
    if not images:
        sys.exit(f"no images found in {args.image_dir}")
    if args.limit > 0:
        images = images[:args.limit]

    p = paths_for(args.image_dir)
    os.makedirs(p["text"], exist_ok=True)
    os.makedirs(p["table"], exist_ok=True)
    print(f"input : {p['images']}  ({len(images)} image(s))")
    print(f"output: {p['root']}")
    print(f"model : {args.model}")

    t0 = time.perf_counter()
    failed = []

    # Map each image to its output stems, disambiguating stems that collide
    # (ujju.png and ujju.jpg would otherwise both claim ujju_text.md).
    jobs, taken = [], {}
    for name in images:
        stem = os.path.splitext(name)[0]
        n = taken.get(stem.lower(), 0) + 1
        taken[stem.lower()] = n
        suffix = "" if n == 1 else f"_{n}"
        jobs.append((name,
                     os.path.join(p["text"], f"{stem}{suffix}_text.md"),
                     os.path.join(p["table"], f"{stem}{suffix}_table.md")))

    for i, (name, text_path, table_path) in enumerate(jobs, 1):
        t = time.perf_counter()
        try:
            # -- stage 1: transcribe -----------------------------------------
            if args.skip_existing and os.path.exists(text_path):
                text = open(text_path, encoding="utf-8").read().strip()
                note = "text reused"
            else:
                text = transcribe_image(
                    claude, os.path.join(p["images"], name),
                    args.model, args.timeout).strip()
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text + "\n")
                note = f"{len(text)} chars"
            if not text:
                print(f"[{i}/{len(jobs)}] {name}  no text found, skipped")
                continue

            # -- stage 2: tabulate -------------------------------------------
            if args.skip_existing and os.path.exists(table_path):
                note += ", table reused"
            else:
                table = extract_table(claude, text, args.model, args.timeout)

                # -- stage 3: re-check against the original image ------------
                _, draft_rows = parse_table(table)
                if not args.no_verify and draft_rows:
                    table = verify_table(
                        claude, os.path.join(p["images"], name), text, table,
                        args.model, args.timeout)
                    note += ", verified"

                table = add_image_column(table, name)
                with open(table_path, "w", encoding="utf-8") as f:
                    f.write(f"# {name}\n\n{table}\n")
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
            print(f"[{i}/{len(jobs)}] {name}  FAILED: {e}")
            failed.append(name)
            continue
        print(f"[{i}/{len(jobs)}] {name}  {time.perf_counter()-t:.1f}s  "
              f"{note} -> {os.path.basename(table_path)}")

    # -- stage 3: merge every table (including earlier runs) into the CSV ----
    print()
    tables_to_csv(p["table"], p["csv"])

    print(f"\nDONE — {len(jobs) - len(failed)}/{len(jobs)} image(s) in "
          f"{time.perf_counter()-t0:.1f}s")
    if failed:
        print(f"failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
