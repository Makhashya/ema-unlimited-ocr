r"""Run the whole image -> Excel pipeline with one command.

    python run_pipeline.py --image_dir Mechanical_input
    python run_pipeline.py --image_dir Mechanical_input --model claude-sonnet-5
    python run_pipeline.py --list-models

Chains the four stages and writes everything under Output_<input folder name>:

    Mechanical_input\                              (your images)
    Output_Mechanical_input\
        output_text_Mechanical_input\   *_text.md    <- ocr_to_md.py (GPU) or vlm_to_md.py (API)
        output_table_Mechanical_input\  *_table.md   <- md_to_table.py  (Claude CLI)
        output_csv_Mechanical_input\    <name>.csv   <- tables_to_csv.py
        output_excel_Mechanical_input\  <name>.xlsx  <- csv_to_excel.py

The stages run in sequence, not in parallel: each one consumes the previous
stage's output. Only --model affects the Claude step. The OCR step is local
by default; --ocr-backend api sends the images to a cloud vision model
instead (vlm_to_md.py -- see --provider / --ocr-model / --base-url):

    python run_pipeline.py --image_dir Mechanical_input --ocr-backend api --provider anthropic
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# The Claude model used by md_to_table.py. Anything the `claude` CLI accepts
# works -- these are the current ones, cheapest last.
MODELS = [
    ("claude-opus-5",    "default; best extraction accuracy"),
    ("claude-fable-5",   "most capable, slowest and priciest"),
    ("claude-sonnet-5",  "good quality, noticeably cheaper than Opus"),
    ("claude-opus-4-8",  "previous Opus generation"),
    ("claude-sonnet-4-6", "previous Sonnet generation"),
    ("claude-haiku-4-5", "fastest and cheapest; verify accuracy before bulk use"),
    ("opus",             "alias -> latest Opus"),
    ("sonnet",           "alias -> latest Sonnet"),
    ("haiku",            "alias -> latest Haiku"),
]


def print_models():
    print("Models for the table-extraction step (--model):\n")
    width = max(len(m) for m, _ in MODELS)
    for name, note in MODELS:
        print(f"  {name.ljust(width)}   {note}")


def paths_for(image_dir: str):
    """Every path the pipeline uses, from the input folder alone."""
    src = os.path.abspath(image_dir)
    name = os.path.basename(src)
    root = os.path.join(os.path.dirname(src), f"Output_{name}")
    return {
        "name": name,
        "images": src,
        "root": root,
        "text": os.path.join(root, f"output_text_{name}"),
        "table": os.path.join(root, f"output_table_{name}"),
        "csv_dir": os.path.join(root, f"output_csv_{name}"),
        "csv": os.path.join(root, f"output_csv_{name}", f"{name}.csv"),
        "excel": os.path.join(root, f"output_excel_{name}", f"{name}.xlsx"),
    }


def run(step: str, script: str, argv, python: str) -> float:
    cmd = [python, os.path.join(HERE, script)] + argv
    print(f"\n{'=' * 70}\n{step}: {script}\n{'=' * 70}")
    t = time.perf_counter()
    proc = subprocess.run(cmd)
    dt = time.perf_counter() - t
    if proc.returncode != 0:
        sys.exit(f"\n{script} failed (exit {proc.returncode}) after {dt:.1f}s — "
                 f"pipeline stopped.")
    return dt


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="run with --list-models to see the available models")
    ap.add_argument("--image_dir", help="folder of input images")
    ap.add_argument("--model", default="claude-opus-5",
                    help="Claude model for the table step (default: claude-opus-5)")
    ap.add_argument("--list-models", action="store_true",
                    help="print the model list and exit")
    ap.add_argument("--mode", choices=["gundam", "base"], default="gundam",
                    help="OCR mode (default: gundam; local backend only)")
    ap.add_argument("--raw", action="store_true",
                    help="keep <|det|> layout markers in the OCR text "
                         "(local backend only)")
    ap.add_argument("--ocr-backend", choices=["local", "api"], default="local",
                    help="how to read the images: 'local' runs the GPU OCR "
                         "model (ocr_to_md.py), 'api' calls a cloud vision "
                         "model (vlm_to_md.py) (default: local)")
    ap.add_argument("--provider", choices=["openai", "anthropic", "gemini",
                                           "custom"], default=None,
                    help="API preset for --ocr-backend api "
                         "(default: VLM_PROVIDER from .env, else openai)")
    ap.add_argument("--base-url", default=None,
                    help="API base URL override (required with --provider custom)")
    ap.add_argument("--ocr-model", default=None,
                    help="vision model for --ocr-backend api "
                         "(default: the provider's preset)")
    ap.add_argument("--api-key-env", default=None,
                    help="env var holding the API key "
                         "(default: the provider's usual variable)")
    ap.add_argument("--ocr-concurrency", type=int, default=1,
                    help="parallel API requests for --ocr-backend api (default: 1)")
    ap.add_argument("--ocr-max-side", type=int, default=None,
                    help="downscale photos to this many pixels on the longest "
                         "side before sending to the API; smaller is faster "
                         "and avoids gateway timeouts (default: vlm_to_md.py's "
                         "3000; try 1280 if you see HTTP 504)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse OCR and table files that already exist")
    ap.add_argument("--claude", default=None, help="path to the claude CLI")
    ap.add_argument("--ocr-timeout", type=float, default=300.0,
                    help="seconds per image before OCR saves the partial text "
                         "and moves on (default: 300 = 5 min; 0 disables)")
    args = ap.parse_args()

    if args.list_models:
        print_models()
        return
    if not args.image_dir:
        ap.error("--image_dir is required (or use --list-models)")
    if not os.path.isdir(args.image_dir):
        sys.exit(f"not a folder: {args.image_dir}")

    known = {m for m, _ in MODELS}
    if args.model not in known:
        print(f"note: '{args.model}' is not in the known list; passing it to the "
              f"claude CLI anyway. Use --list-models to see the list.")

    p = paths_for(args.image_dir)
    print(f"input   : {p['images']}")
    print(f"output  : {p['root']}")
    print(f"model   : {args.model}")
    if args.ocr_backend == "api":
        print(f"ocr     : api ({args.provider or 'provider from .env/default'}"
              f" / {args.ocr_model or 'preset model'})")
        if args.raw or args.mode != "gundam":
            print("note: --raw/--mode only apply to the local backend; ignored")
    else:
        print("ocr     : local (ocr_to_md.py)")

    python = sys.executable
    times = {}

    if args.ocr_backend == "api":
        ocr = ["--image_dir", p["images"], "--output_dir", p["text"],
               "--timeout", str(args.ocr_timeout),
               "--concurrency", str(args.ocr_concurrency)]
        if args.provider:
            ocr += ["--provider", args.provider]
        if args.base_url:
            ocr += ["--base-url", args.base_url]
        if args.ocr_model:
            ocr += ["--model", args.ocr_model]
        if args.api_key_env:
            ocr += ["--api-key-env", args.api_key_env]
        if args.ocr_max_side is not None:
            ocr += ["--max-side", str(args.ocr_max_side)]
        if args.skip_existing:
            ocr.append("--skip-existing")
        ocr_script = "vlm_to_md.py"
    else:
        ocr = ["--image_dir", p["images"], "--output_dir", p["text"],
               "--mode", args.mode, "--timeout", str(args.ocr_timeout)]
        if not args.raw:
            ocr.append("--clean")
        if args.skip_existing:
            ocr.append("--skip-existing")
        ocr_script = "ocr_to_md.py"
    times["1 OCR"] = run("STEP 1/4  images -> text", ocr_script, ocr, python)

    tbl = ["--md_dir", p["text"], "--image_dir", p["images"],
           "--output_dir", p["table"], "--model", args.model]
    if args.claude:
        tbl += ["--claude", args.claude]
    if args.skip_existing:
        tbl.append("--skip-existing")
    times["2 tables"] = run("STEP 2/4  text -> tables", "md_to_table.py", tbl, python)

    times["3 csv"] = run("STEP 3/4  tables -> csv", "tables_to_csv.py",
                         ["--table_dir", p["table"], "--out", p["csv"]], python)

    times["4 excel"] = run("STEP 4/4  csv -> excel", "csv_to_excel.py",
                           ["--csv", p["csv"], "--out", p["excel"]], python)

    print(f"\n{'=' * 70}\nDONE — {sum(times.values()):.1f}s total")
    for k, v in times.items():
        print(f"  {k:<10} {v:6.1f}s")
    print(f"\noutputs under {p['root']}:")
    for label, path in (("text ", p["text"]), ("tables", p["table"]),
                        ("csv  ", p["csv"]), ("excel", p["excel"])):
        mark = "" if os.path.exists(path) else "   (missing)"
        print(f"  {label}  {os.path.relpath(path, p['root'])}{mark}")


if __name__ == "__main__":
    main()
