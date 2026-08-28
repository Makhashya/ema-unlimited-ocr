r"""OCR a folder of images into one markdown file per image.

Loads the model once and loops model.infer() per image (measured faster and
flatter in VRAM than infer_multi -- see bench.py). Uses eval_mode=True, which
returns the text and writes no side artifacts, so the only files produced are
the markdown ones.

Each image yields <output_dir>/<image name>_text.md -- ujju.png -> ujju_text.md.

Everything for one input folder lives under a single 'Output_<name>' folder
created beside it, so .\Mechanical_input produces:

    Output_Mechanical_input\output_text_Mechanical_input\   (this script)
    Output_Mechanical_input\output_table_Mechanical_input\  (md_to_table.py)
    Output_Mechanical_input\output_csv_Mechanical_input\    (tables_to_csv.py)
    Output_Mechanical_input\output_excel_Mechanical_input\  (csv_to_excel.py)

    python ocr_to_md.py --image_dir .\Mechanical_input
    python ocr_to_md.py --image_dir .\Mechanical_input --clean
    python ocr_to_md.py --image_dir .\Mechanical_input --output_dir D:\elsewhere
"""

import argparse
import os
import re
import shutil
import tempfile
import time

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.generation.stopping_criteria import (StoppingCriteria,
                                                       StoppingCriteriaList)

MODEL_NAME = os.environ.get("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR")
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DET_RE = re.compile(r"<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)", re.DOTALL)


def remove_det(raw: str) -> str:
    """Strip <|det|>type [bbox]<|/det|> markers (from the repo README)."""
    blocks = []
    cur = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = DET_RE.match(line)
        if m:
            category, content = m.group(1).strip(), m.group(2).strip()
            if category == "image":
                continue
            if cur is not None:
                blocks.append(cur)
            cur = [content] if content else []
            continue
        if cur is None:
            cur = []
        cur.append(line)
    if cur is not None:
        blocks.append(cur)
    return "\n\n".join("\n".join(b) for b in blocks).strip()


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


class TimeLimit(StoppingCriteria):
    """Stop decoding once `limit` seconds have passed.

    Stopping generation (rather than killing the call) means generate() returns
    normally and infer() still decodes the tokens produced so far, so a slow
    page yields partial text instead of nothing.
    """

    def __init__(self, limit: float):
        self.limit = limit
        self.start = time.monotonic()
        self.hit = False

    def __call__(self, input_ids, scores, **kwargs):
        over = time.monotonic() - self.start > self.limit
        if over:
            self.hit = True
        return torch.full((input_ids.shape[0],), over,
                          dtype=torch.bool, device=input_ids.device)


def attach_time_limit(model, limit: float) -> dict:
    """Make every model.infer() call stop after `limit` seconds.

    infer() calls self.generate() internally with no way to pass stopping
    criteria, so shadow the bound method with one that injects ours. Returns a
    dict whose 'criteria' key holds the most recent TimeLimit.
    """
    state = {"criteria": None}
    original = model.generate

    def generate_with_limit(*a, **kw):
        criteria = TimeLimit(limit)
        state["criteria"] = criteria
        existing = kw.get("stopping_criteria") or StoppingCriteriaList()
        existing.append(criteria)
        kw["stopping_criteria"] = existing
        return original(*a, **kw)

    model.generate = generate_with_limit
    return state


def default_output_dir(image_dir: str) -> str:
    """Mechanical_input -> Output_Mechanical_input\\output_text_Mechanical_input

    The 'Output_<name>' folder is created beside the input folder and holds
    every stage of the pipeline. The later scripts each place their output
    beside their own input, so they land inside 'Output_<name>' too.

    abspath() normalises a trailing separator and '.', so 'imgs\\' and 'imgs'
    both yield the same result.
    """
    p = os.path.abspath(image_dir)
    name = os.path.basename(p)
    return os.path.join(os.path.dirname(p), f"Output_{name}", f"output_text_{name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True, help="folder of images")
    ap.add_argument("--output_dir", default=None,
                    help="folder for the per-image .md files "
                         "(default: 'output_text_<input folder name>' beside the input folder)")
    ap.add_argument("--mode", choices=["gundam", "base"], default="gundam")
    ap.add_argument("--clean", action="store_true",
                    help="strip <|det|> layout markers, leaving plain markdown")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave images whose .md already exists untouched")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="seconds to spend on one image before saving the "
                         "partial text and moving on (default: 300 = 5 min; "
                         "0 disables)")
    args = ap.parse_args()

    files = sorted((f for f in os.listdir(args.image_dir)
                    if os.path.splitext(f)[1].lower() in EXTS), key=natural_key)
    if not files:
        raise SystemExit(f"no images found in {args.image_dir}")

    out_dir = args.output_dir or default_output_dir(args.image_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"output folder: {out_dir}")

    # Map each image to <stem>_text.md, disambiguating stems that collide
    # (ujju.png and ujju.jpg would otherwise both claim ujju_text.md).
    jobs, taken = [], {}
    for name in files:
        stem = os.path.splitext(name)[0]
        n = taken.get(stem.lower(), 0) + 1
        taken[stem.lower()] = n
        suffix = "" if n == 1 else f"_{n}"
        jobs.append((name, os.path.join(out_dir, f"{stem}{suffix}_text.md")))

    if args.skip_existing:
        pending = [(n, p) for n, p in jobs if not os.path.exists(p)]
        if len(pending) < len(jobs):
            print(f"skipping {len(jobs) - len(pending)} image(s) already done")
        jobs = pending
    if not jobs:
        print("nothing to do")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                      use_safetensors=True,
                                      dtype=torch.bfloat16).eval().cuda()

    limit_state = attach_time_limit(model, args.timeout) if args.timeout > 0 else None
    if limit_state:
        print(f"per-image time limit: {args.timeout:.0f}s")

    gundam = args.mode == "gundam"
    t0 = time.perf_counter()
    truncated = []

    # infer() unconditionally mkdirs output_path and output_path/images even in
    # eval_mode, so hand it a scratch dir and delete it afterwards.
    scratch = tempfile.mkdtemp(prefix="uocr_")
    try:
        for i, (name, md_path) in enumerate(jobs, 1):
            t = time.perf_counter()
            text = model.infer(
                tokenizer,
                prompt="<image>document parsing.",
                image_file=os.path.join(args.image_dir, name),
                output_path=scratch,
                base_size=1024,
                image_size=640 if gundam else 1024,
                crop_mode=gundam,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=128,
                eval_mode=True,      # return text, write nothing
            ) or ""
            if args.clean:
                text = remove_det(text)
            hit = bool(limit_state and limit_state["criteria"]
                       and limit_state["criteria"].hit)
            body = text.strip()
            if hit:
                truncated.append(name)
                body += (f"\n\n<!-- TRUNCATED: OCR stopped after "
                         f"{args.timeout:.0f}s; this page is incomplete -->")
            # write as we go, so an interrupted run keeps what it finished
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(body + "\n")
            flag = "  TIMEOUT, saved partial" if hit else ""
            print(f"[{i}/{len(jobs)}] {name}  {time.perf_counter()-t:.1f}s  "
                  f"{len(text)} chars -> {os.path.basename(md_path)}{flag}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n{len(jobs)} images in {time.perf_counter()-t0:.1f}s "
          f"-> {os.path.abspath(out_dir)}")
    if truncated:
        print(f"hit the {args.timeout:.0f}s limit (partial text saved): "
              f"{', '.join(truncated)}")


if __name__ == "__main__":
    main()
