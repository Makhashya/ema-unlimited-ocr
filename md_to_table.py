r"""Turn OCR'd nameplate text into equipment schedule tables.

Reads the per-image .md files produced by ocr_to_md.py and asks Claude -- via
the already-authenticated `claude` CLI, so no API key is needed -- to build one
markdown table per file.

    output_text_ujju\image1_text.md  ->  output_table_ujju\image1_table.md

Usage:
    python md_to_table.py --md_dir output_text_ujju
    python md_to_table.py --md_dir output_text_ujju --skip-existing
    python md_to_table.py --md_dir output_text_ujju --model claude-sonnet-5
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

COLUMNS = ("Tag #", "Make", "Model Number", "Serial Number", "Refrigerant Type",
           "Manufacture Date", "Approximate Tonnage", "MCA/MOCP",
           "Voltage/Phase", "Condition  (G/M/P)", "Comments")

SYSTEM_PROMPT = """\
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
number encodes it in a standard manufacturer scheme, decode it -- e.g. Lennox \
serial 1519F31081 -> 2019 (week 15 of 2019), Ducane serial 1911G22987 -> 2011.
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

If the text contains no equipment nameplate data at all, output the table \
header and separator row with no data rows."""

FENCE_RE = re.compile(r"^\s*```[^\n]*\n(.*?)\n\s*```\s*$", re.DOTALL)
SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
NO_TOOLS = ["Bash", "Read", "Write", "Edit", "NotebookEdit", "Glob", "Grep",
            "WebFetch", "WebSearch", "Task", "TodoWrite"]


def output_dir_for(md_dir: str) -> str:
    """output_text_ujju -> output_table_ujju, beside the input folder."""
    p = os.path.abspath(md_dir)
    name = os.path.basename(p)
    if name.startswith("output_text_"):
        out = "output_table_" + name[len("output_text_"):]
    else:
        out = "output_table_" + name
    return os.path.join(os.path.dirname(p), out)


def source_stem(md_name: str) -> str:
    """image1_text.md -> image1"""
    stem = os.path.splitext(md_name)[0]
    return stem[:-len("_text")] if stem.endswith("_text") else stem


def table_name_for(md_name: str) -> str:
    """image1_text.md -> image1_table.md"""
    return f"{source_stem(md_name)}_table.md"


def image_name_for(md_name: str, image_dir: str = None) -> str:
    """image1_text.md -> image1.jpg when the image is findable, else image1."""
    stem = source_stem(md_name)
    if image_dir and os.path.isdir(image_dir):
        for f in sorted(os.listdir(image_dir)):
            if os.path.splitext(f)[0] == stem:
                return f
    return stem


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


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


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
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "claude", "claude.exe"),
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


def run_claude(claude: str, document: str, model: str, timeout: int) -> str:
    cmd = [claude, "-p",
           "--system-prompt", SYSTEM_PROMPT,
           "--model", model,
           "--output-format", "text",
           "--no-session-persistence",
           "--disallowed-tools", *NO_TOOLS]
    proc = subprocess.run(cmd, input=document, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "").strip()[:500])
    return strip_fence(proc.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md_dir", required=True,
                    help="folder of OCR .md files, e.g. output_text_ujju")
    ap.add_argument("--output_dir", default=None,
                    help="default: 'output_table_<name>' beside the input folder")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--claude", default=None,
                    help="path to the claude CLI if it is not on PATH")
    ap.add_argument("--image_dir", default=None,
                    help="original image folder; lets the Image column show the "
                         "real filename (image1.jpg) instead of just the stem")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per file")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    claude = find_claude(args.claude)

    files = sorted((f for f in os.listdir(args.md_dir) if f.lower().endswith(".md")),
                   key=natural_key)
    if not files:
        sys.exit(f"no .md files found in {args.md_dir}")

    out_dir = args.output_dir or output_dir_for(args.md_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"model: {args.model}")
    print(f"output folder: {out_dir}")

    jobs = [(f, os.path.join(out_dir, table_name_for(f))) for f in files]
    if args.skip_existing:
        pending = [(f, p) for f, p in jobs if not os.path.exists(p)]
        if len(pending) < len(jobs):
            print(f"skipping {len(jobs) - len(pending)} file(s) already done")
        jobs = pending
    if not jobs:
        print("nothing to do")
        return

    t0 = time.perf_counter()
    failed = []
    for i, (name, out_path) in enumerate(jobs, 1):
        text = open(os.path.join(args.md_dir, name), encoding="utf-8").read().strip()
        if not text:
            print(f"[{i}/{len(jobs)}] {name}  empty, skipped")
            continue
        t = time.perf_counter()
        try:
            table = run_claude(claude, text, args.model, args.timeout)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"[{i}/{len(jobs)}] {name}  FAILED: {e}")
            failed.append(name)
            continue
        image = image_name_for(name, args.image_dir)
        table = add_image_column(table, image)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# {image}\n\n{table}\n")
        print(f"[{i}/{len(jobs)}] {name}  {time.perf_counter()-t:.1f}s "
              f"-> {os.path.basename(out_path)}")

    print(f"\n{len(jobs)-len(failed)}/{len(jobs)} tables in "
          f"{time.perf_counter()-t0:.1f}s -> {os.path.abspath(out_dir)}")
    if failed:
        print(f"failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
