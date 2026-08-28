r"""Clean up image filenames: strip spaces (and optionally renumber them).

Spaces in filenames break command lines, and the OCR pipeline derives every
downstream name from the image name, so a space propagates all the way to the
.md, .csv and .xlsx files.

    python rename_images.py --image_dir Mechanical_input --dry-run
    python rename_images.py --image_dir Mechanical_input
    python rename_images.py --image_dir Mechanical_input --replace ""
    python rename_images.py --image_dir Mechanical_input --sequential image

By default "Roof Top Unit 3.jpg" -> "Roof_Top_Unit_3.jpg". Use --replace ""
to delete spaces instead of replacing them, or --sequential to renumber
everything as image1.jpg, image2.jpg, ...

Every run writes rename_map.csv next to the images so a rename can be undone.
"""

import argparse
import csv
import os
import re
import sys

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# characters beyond spaces that tend to cause trouble in shells and URLs
STRICT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def clean_stem(stem: str, replace: str, strict: bool) -> str:
    s = stem.strip()
    s = re.sub(r"\s+", replace, s)          # collapse runs of whitespace
    if strict:
        s = STRICT_RE.sub(replace or "_", s)
    if replace:                              # avoid doubled separators
        s = re.sub(re.escape(replace) + r"{2,}", replace, s)
        s = s.strip(replace)
    return s or "image"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--replace", default="_",
                    help="what to put in place of spaces (default: '_')")
    ap.add_argument("--delete", action="store_true",
                    help="delete spaces instead of replacing them "
                         "(PowerShell drops an empty --replace \"\", so use this)")
    ap.add_argument("--sequential", metavar="PREFIX", default=None,
                    help="renumber as PREFIX1, PREFIX2, ... instead of cleaning names")
    ap.add_argument("--strict", action="store_true",
                    help="also replace characters other than A-Z a-z 0-9 . _ -")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would change, rename nothing")
    args = ap.parse_args()

    if not os.path.isdir(args.image_dir):
        sys.exit(f"not a folder: {args.image_dir}")
    replace = "" if args.delete else args.replace

    files = sorted((f for f in os.listdir(args.image_dir)
                    if os.path.splitext(f)[1].lower() in EXTS), key=natural_key)
    if not files:
        sys.exit(f"no images found in {args.image_dir}")

    # Desired name for each file, before collision handling
    width = len(str(len(files)))
    desired = []
    for i, name in enumerate(files, 1):
        stem, ext = os.path.splitext(name)
        ext = ext.strip().lower()
        new_stem = (f"{args.sequential}{i:0{width}d}" if args.sequential
                    else clean_stem(stem, replace, args.strict))
        desired.append((name, f"{new_stem}{ext}"))

    # Files that are already correct keep their name — reserve those first so a
    # renamed file never displaces one that needed no change.
    taken = {a.lower() for a, b in desired if a == b}
    planned = []
    for old, want in desired:
        if old == want:
            planned.append((old, want))
            continue
        candidate, n = want, 1
        while candidate.lower() in taken:
            n += 1
            stem, ext = os.path.splitext(want)
            candidate = f"{stem}_{n}{ext}"
        taken.add(candidate.lower())
        planned.append((old, candidate))

    changes = [(a, b) for a, b in planned if a != b]
    if not changes:
        print(f"{len(files)} image(s), nothing to rename")
        return

    width_old = max(len(a) for a, _ in changes)
    for old, new in changes:
        print(f"  {old.ljust(width_old)}  ->  {new}")
    print(f"\n{len(changes)} of {len(files)} image(s) would be renamed"
          if args.dry_run else f"\nrenaming {len(changes)} of {len(files)} image(s)")
    if args.dry_run:
        return

    # Two phases via temporary names, so a swap (a->b while b->a) cannot clobber.
    tmp = []
    for i, (old, new) in enumerate(changes):
        t = os.path.join(args.image_dir, f".renaming_{i}.tmp")
        os.rename(os.path.join(args.image_dir, old), t)
        tmp.append((t, old, new))
    for t, old, new in tmp:
        os.rename(t, os.path.join(args.image_dir, new))

    log = os.path.join(args.image_dir, "rename_map.csv")
    exists = os.path.isfile(log)
    with open(log, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["old_name", "new_name"])
        w.writerows((old, new) for _, old, new in tmp)

    print(f"done — mapping appended to {log}")


if __name__ == "__main__":
    main()
