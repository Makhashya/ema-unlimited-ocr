r"""Folder of nameplate photos -> ONLY the schedule fields -> Excel (API).

The lean extractor, API edition: every image in the given folder is
processed one after another with ONE model call each, asking directly for
the fields the equipment schedule needs. No transcription files, no markdown
tables -- the model answers in JSON and everything lands in one formatted
Excel sheet (and on the console). Photos of the same unit (same serial) are
merged into one row that lists every image name. No photo is worked on
for longer than --photo-timeout (default 5 minutes): the call -- including
any retries -- is abandoned at that point and the run moves on.

The endpoint comes from .env / flags exactly like equipment_pipeline_api.py
(template: .env.example). CLI-claude twin: image_to_excel_cli.py.

Usage:
    python image_to_excel_api.py --image_dir Mechanical_input
    python image_to_excel_api.py --image_dir Mechanical_input --provider anthropic
    python image_to_excel_api.py --image_dir Mechanical_input --limit 2   # smoke test
"""

import argparse
import json
import os
import re
import sys
import time

import requests

from equipment_pipeline_api import (
    COLUMNS, EXTS, PROVIDERS, ApiError, PhotoTimeout,
    chat, dedupe_records, image_to_data_uri, load_env_file, natural_key,
    strip_fence, write_excel,
)

PHOTO_TIMEOUT = 300.0     # hard cap per image, seconds (5 minutes)

FIELDS_PROMPT = """\
You read equipment nameplate photos (HVAC and similar) and return ONLY the \
fields an equipment schedule needs. Output strict JSON and nothing else -- \
no prose, no explanation, no code fences: a JSON array with one object per \
physical unit shown, or [] if the image contains no equipment information. \
Every object has exactly these keys (use "" for a value the image does not \
supply):

["Tag #", "Make", "Model Number", "Serial Number", "Refrigerant Type", \
"Manufacture Date", "Approximate Tonnage", "MCA/MOCP", "Voltage/Phase", \
"Condition  (G/M/P)", "Comments"]

Rules:
- Tag #: the printed equipment tag (e.g. CU5, AH2, RTU-3); if none, the \
unit-type abbreviation plus " ( no tag )" -- AH air handler/furnace/coil, \
CU condensing or heat pump outdoor unit, RTU packaged rooftop, FC fan coil, \
B boiler, P pump, EF exhaust fan.
- Model Number / Serial Number: the exact characters printed on the plate; \
write ? for a truly unreadable character; never guess or "correct".
- Manufacture Date: the year -- one of the MOST IMPORTANT fields. Use a \
date printed anywhere on the plate; otherwise decode the serial number via \
the manufacturer's scheme (Lennox 1519F31081 -> 2019, Ducane 1911G22987 -> \
2011, and any maker whose convention you know, including leading week-year, \
month-year, or year digits such as 02-02-... -> 2002). Leave "" only when \
the serial is genuinely ambiguous.
- Approximate Tonnage: whole tons from the capacity digits in the model \
number (036 -> 3, 048 -> 4, 060 -> 5); the number only.
- Refrigerant Type: as printed, e.g. HFC-410A, HCFC-22.
- MCA/MOCP: both values separated by a slash, e.g. "18.6/30".
- Voltage/Phase: as printed, e.g. 208/230/1.
- Condition  (G/M/P): fill only if a condition assessment is actually \
written on or with the unit; otherwise "".
- Comments: other useful facts from the plate (equipment type, factory \
charge, listing marks). Supporting equipment (a disconnect / safety switch, \
unit heater, exhaust fan) also gets an object -- name its type here.
- A unit with ANY readable data (a make, model, serial, or rating) gets an \
object; return [] only for an image with no equipment information at all."""


def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


_CANON = {_norm_key(c): c for c in COLUMNS}


def parse_fields(raw: str):
    """Model output -> list of {canonical column: string value} dicts."""
    raw = strip_fence(raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # salvage the first JSON array or object in the text
        m = re.search(r"\[.*\]|\{.*\}", raw, re.DOTALL)
        if not m:
            raise ApiError(f"model did not return JSON: {raw[:200]!r}")
        data = json.loads(m.group(0))
    if isinstance(data, dict):
        data = [data]
    rows = []
    for d in data:
        if not isinstance(d, dict):
            continue
        row = {c: "" for c in COLUMNS}
        for k, v in d.items():
            c = _CANON.get(_norm_key(k))
            if c and v is not None:
                row[c] = str(v).strip()
        if any(row.values()):
            rows.append(row)
    return rows


def extract_image(api, image_path: str, max_side: int,
                  photo_timeout: float = None):
    """One photo -> list of field dicts, never running past its time budget.

    The budget is api["deadline"] (a perf_counter value set by a caller
    that manages its own per-photo clock, e.g. app.py) or, failing that,
    photo_timeout seconds from now. When it runs out, chat() raises
    PhotoTimeout instead of starting or retrying a request, and the
    request itself is capped at the remaining budget.
    """
    deadline = api.get("deadline")
    if deadline is None and photo_timeout and photo_timeout > 0:
        deadline = time.perf_counter() + photo_timeout
    uri = image_to_data_uri(image_path, max_side)
    content = [
        {"type": "image_url", "image_url": {"url": uri}},
        {"type": "text", "text": "Extract the equipment-schedule fields "
                                 "from this photo, following your rules."},
    ]
    raw = chat(api["session"], api["base_url"], api["headers"], api["model"],
               FIELDS_PROMPT, content, api["timeout"], api["max_tokens"],
               deadline=deadline)
    return parse_fields(raw)


def main():
    ap = argparse.ArgumentParser(
        description="one direct API call per photo in a folder: just the "
                    "equipment-schedule fields, straight into an Excel sheet")
    ap.add_argument("--image_dir", required=True,
                    help="folder of nameplate images")
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default=None,
                    help="API preset (default: VLM_PROVIDER from .env, "
                         "else openai)")
    ap.add_argument("--model", default=None,
                    help="model name (default: the provider's preset / .env)")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--out", default=None,
                    help="output .xlsx (default: <folder>_equipment.xlsx "
                         "beside the input folder)")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="seconds per model call (default: 300)")
    ap.add_argument("--photo-timeout", type=float, default=PHOTO_TIMEOUT,
                    help="hard cap per image including retries; a photo "
                         "still unfinished after this is skipped "
                         "(default: 300 = 5 minutes, 0 disables)")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-side", type=int, default=3000,
                    help="downscale images above this many pixels on the "
                         "longest side (default: 3000)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N images (0 = all)")
    args = ap.parse_args()

    if not os.path.isdir(args.image_dir):
        sys.exit(f"not a folder: {args.image_dir}")
    images = sorted((f for f in os.listdir(args.image_dir)
                     if os.path.splitext(f)[1].lower() in EXTS),
                    key=natural_key)
    if not images:
        sys.exit(f"no images found in {args.image_dir}")
    if args.limit > 0:
        images = images[:args.limit]

    load_env_file()
    provider = (args.provider or os.environ.get("VLM_PROVIDER", "").lower()
                or "openai")
    if provider not in PROVIDERS:
        ap.error(f"VLM_PROVIDER must be one of {', '.join(sorted(PROVIDERS))}")

    def env_default(name):
        # .env endpoint settings only apply when --provider wasn't given
        return None if args.provider else os.environ.get(name)

    preset_url, preset_env, preset_model = PROVIDERS[provider]
    base_url = (args.base_url or env_default("VLM_BASE_URL")
                or preset_url or "").rstrip("/")
    model = args.model or env_default("VLM_MODEL") or preset_model
    if not base_url or not model:
        ap.error("--base-url and --model are required with the custom "
                 "provider")
    key_env = args.api_key_env or env_default("VLM_API_KEY_ENV") or preset_env
    api_key = os.environ.get(key_env, "")
    if not api_key and provider != "custom":
        sys.exit(f"no API key: set {key_env} (template: .env.example)")
    api = {"session": requests.Session(), "base_url": base_url,
           "headers": ({"Authorization": f"Bearer {api_key}"}
                       if api_key else {}),
           "model": model, "timeout": args.timeout,
           "max_tokens": args.max_tokens}

    src = os.path.abspath(args.image_dir)
    out = args.out or os.path.join(os.path.dirname(src),
                                   f"{os.path.basename(src)}_equipment.xlsx")
    print(f"input  : {src}  ({len(images)} image(s))")
    print(f"backend: {provider} / {model}")
    print(f"output : {out}")

    t0 = time.perf_counter()
    records, failed, timed_out, no_data = [], [], [], []
    for i, name in enumerate(images, 1):
        t = time.perf_counter()
        try:
            rows = extract_image(api, os.path.join(src, name), args.max_side,
                                 args.photo_timeout)
        except PhotoTimeout:
            print(f"[{i}/{len(images)}] {name}  TIMED OUT after "
                  f"{time.perf_counter()-t:.0f}s (limit "
                  f"{args.photo_timeout:.0f}s), skipped")
            timed_out.append(name)
            continue
        except (ApiError, OSError) as e:
            print(f"[{i}/{len(images)}] {name}  FAILED: {e}")
            failed.append(name)
            continue
        if not rows:
            print(f"[{i}/{len(images)}] {name}  "
                  f"{time.perf_counter()-t:.1f}s  no equipment data")
            no_data.append(name)
            continue
        print(f"[{i}/{len(images)}] {name}  {time.perf_counter()-t:.1f}s")
        for row in rows:
            row["Image"] = name
            records.append(row)
            for c in COLUMNS:
                if row.get(c):
                    print(f"    {c:<22}: {row[c]}")

    if records:
        columns = ["Image", *COLUMNS]
        unique = dedupe_records(records, columns)
        written = write_excel(unique, columns, out)
        dups = len(records) - len(unique)
        print(f"\n{len(unique)} unit(s) -> {written}"
              + (f"  ({dups} duplicate photo row(s) merged)" if dups else ""))
    else:
        print("\nno equipment data extracted; no Excel written")

    print(f"done — {len(images) - len(failed) - len(timed_out)}/"
          f"{len(images)} image(s) in {time.perf_counter()-t0:.1f}s")
    if no_data:
        print(f"skipped (no equipment data): {', '.join(no_data)}")
    if timed_out:
        print(f"timed out (> {args.photo_timeout:.0f}s per photo): "
              f"{', '.join(timed_out)}")
    if failed:
        print(f"failed: {', '.join(failed)}")
    if failed or timed_out:
        sys.exit(1)


if __name__ == "__main__":
    main()
