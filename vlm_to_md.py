r"""Transcribe a folder of images into one markdown file per image using a
cloud vision-language model over an OpenAI-compatible chat-completions API.

Drop-in replacement for the stage-1 OCR script (ocr_to_md.py): it writes the
same <stem>_text.md files into the same output folder, so md_to_table.py and
the rest of the pipeline need no changes. Needs no GPU and never imports
torch/transformers -- only requests (+ Pillow when an image must be
re-encoded).

The API key is read from an environment variable, never from the command line.
Keys and a default endpoint can live in a `.env` file beside this script
(copy .env.example to .env and fill it in); real environment variables win
over .env values, and command-line flags win over both.

    openai     https://api.openai.com/v1                            OPENAI_API_KEY
    anthropic  https://api.anthropic.com/v1  (OpenAI compat layer)  ANTHROPIC_API_KEY
    gemini     https://generativelanguage.googleapis.com/v1beta/openai  GEMINI_API_KEY
    custom     --base-url (e.g. a local vLLM server)                API_KEY (optional)

    python vlm_to_md.py --image_dir .\Mechanical_input --provider anthropic
    python vlm_to_md.py --image_dir .\Mechanical_input --provider openai --model gpt-4o
    python vlm_to_md.py --image_dir .\Mechanical_input --provider custom \
        --base-url http://127.0.0.1:8000/v1 --model my-vlm
"""

import argparse
import base64
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Formats every provider accepts as-is; anything else is re-encoded to JPEG.
RAW_OK = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
          ".webp": "image/webp"}

MAX_RAW_BYTES = 15 * 1024 * 1024   # stay under ~20 MB request limits

PROVIDERS = {
    # name: (base_url, key env var, default model)
    "openai":    ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY",
                  "claude-opus-5"),
    "gemini":    ("https://generativelanguage.googleapis.com/v1beta/openai",
                  "GEMINI_API_KEY", "gemini-2.5-flash"),
    "custom":    (None, "API_KEY", None),
}

MAX_RETRIES = 5                    # same policy as infer.py
RETRYABLE = {429, 500, 502, 503, 504, 529}

TRANSCRIBE_PROMPT = """\
You are a meticulous transcription engine for equipment nameplate photos.
Transcribe ALL text visible in the image, exactly as printed.

Rules:
- Preserve every character of model numbers, serial numbers, part numbers,
  and electrical ratings exactly as shown. Never "correct", normalize, or
  guess a character. If a character is truly unreadable, write ? in its place.
- Keep each label with its value on one line, e.g. "MODEL NO: 4TTR4036L1000A".
- Transcribe printed, stamped, embossed, and handwritten text, including
  stickers and secondary labels.
- Output plain text only: no commentary, no descriptions of the image, no
  markdown code fences, no "Here is the transcription".
- If the image contains no readable text, output nothing.
"""


def load_env_file():
    """Load KEY=VALUE lines from .env beside this script into os.environ.

    Values already set in the real environment are left alone, so exported
    variables always win over the file.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and value:
                os.environ.setdefault(key, value)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def default_output_dir(image_dir: str) -> str:
    """Mechanical_input -> Output_Mechanical_input\\output_text_Mechanical_input"""
    p = os.path.abspath(image_dir)
    name = os.path.basename(p)
    return os.path.join(os.path.dirname(p), f"Output_{name}", f"output_text_{name}")


def image_to_data_uri(path: str, max_side: int) -> str:
    """Base64 data URI for the image, re-encoding only when needed.

    Re-encode to JPEG when the format isn't universally accepted (bmp/tiff),
    the longest side exceeds max_side, or the file is too big for one request.
    """
    ext = os.path.splitext(path)[1].lower()
    mime = RAW_OK.get(ext)
    needs_reencode = mime is None or os.path.getsize(path) > MAX_RAW_BYTES

    if not needs_reencode and max_side > 0:
        from PIL import Image
        with Image.open(path) as im:
            needs_reencode = max(im.size) > max_side

    if needs_reencode:
        import io
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            if max_side > 0 and max(im.size) > max_side:
                scale = max_side / max(im.size)
                im = im.resize((round(im.width * scale), round(im.height * scale)),
                               Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=92)
        data, mime = buf.getvalue(), "image/jpeg"
    else:
        with open(path, "rb") as f:
            data = f.read()

    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


class ApiError(RuntimeError):
    pass


def transcribe(session, base_url, headers, model, data_uri,
               timeout, max_tokens) -> str:
    """One image -> transcribed text, with retry/backoff on transient errors."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TRANSCRIBE_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Transcribe all text in this image."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.post(f"{base_url}/chat/completions", json=payload,
                                headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"] or ""
            body = resp.text[:300]
            if resp.status_code not in RETRYABLE:
                # wrong model, bad key, oversized request: retrying won't help
                raise ApiError(f"HTTP {resp.status_code}: {body}")
            last = f"HTTP {resp.status_code}: {body}"
            wait = 3 * (attempt + 1)
            if resp.status_code == 429 and resp.headers.get("Retry-After"):
                try:
                    wait = min(float(resp.headers["Retry-After"]), 60.0)
                except ValueError:
                    pass
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
            wait = 3 * (attempt + 1)
        if attempt < MAX_RETRIES - 1:
            time.sleep(wait)
    raise ApiError(f"gave up after {MAX_RETRIES} attempts: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True, help="folder of images")
    ap.add_argument("--output_dir", default=None,
                    help="folder for the per-image .md files "
                         "(default: 'output_text_<input folder name>' beside the input folder)")
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default=None,
                    help="which API preset to use "
                         "(default: VLM_PROVIDER from .env, else openai)")
    ap.add_argument("--base-url", default=None,
                    help="override the API base URL (required for --provider custom)")
    ap.add_argument("--model", default=None,
                    help="model name (default: the provider's preset)")
    ap.add_argument("--api-key-env", default=None,
                    help="name of the env var holding the API key "
                         "(default: the provider's usual variable)")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-attempt HTTP timeout in seconds (default: 300)")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="response token cap per image (default: 4096)")
    ap.add_argument("--max-side", type=int, default=3000,
                    help="downscale images whose longest side exceeds this "
                         "many pixels (default: 3000; 0 disables)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel requests (default: 1)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave images whose .md already exists untouched")
    args = ap.parse_args()

    load_env_file()

    provider = args.provider or os.environ.get("VLM_PROVIDER", "").lower() or "openai"
    if provider not in PROVIDERS:
        ap.error(f"VLM_PROVIDER in .env must be one of "
                 f"{', '.join(sorted(PROVIDERS))} (got '{provider}')")

    # The VLM_* endpoint settings in .env describe one endpoint together with
    # VLM_PROVIDER, so they only apply when the provider wasn't chosen on the
    # command line -- an explicit --provider uses that provider's presets.
    def env_default(name):
        return None if args.provider else os.environ.get(name)

    preset_url, preset_env, preset_model = PROVIDERS[provider]
    base_url = (args.base_url or env_default("VLM_BASE_URL")
                or preset_url or "").rstrip("/")
    if not base_url:
        ap.error("--base-url (or VLM_BASE_URL in .env) is required "
                 "with the custom provider")
    model = args.model or env_default("VLM_MODEL") or preset_model
    if not model:
        ap.error("--model (or VLM_MODEL in .env) is required "
                 "with the custom provider")

    key_env = args.api_key_env or env_default("VLM_API_KEY_ENV") or preset_env
    api_key = os.environ.get(key_env, "")
    if not api_key and provider != "custom":
        sys.exit(f"no API key: set {key_env} in the environment or in "
                 f"{os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')} "
                 f"(template: .env.example)")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    files = sorted((f for f in os.listdir(args.image_dir)
                    if os.path.splitext(f)[1].lower() in EXTS), key=natural_key)
    if not files:
        raise SystemExit(f"no images found in {args.image_dir}")

    out_dir = args.output_dir or default_output_dir(args.image_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"output folder: {out_dir}")
    print(f"api: {provider}  model: {model}  ({base_url})")

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

    session = requests.Session()
    t0 = time.perf_counter()
    failed = []

    def work(i, name, md_path):
        t = time.perf_counter()
        uri = image_to_data_uri(os.path.join(args.image_dir, name), args.max_side)
        text = transcribe(session, base_url, headers, model, uri,
                          args.timeout, args.max_tokens)
        # write as we go, so an interrupted run keeps what it finished
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        flag = "  EMPTY response" if not text.strip() else ""
        print(f"[{i}/{len(jobs)}] {name}  {time.perf_counter()-t:.1f}s  "
              f"{len(text)} chars -> {os.path.basename(md_path)}{flag}")

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(work, i, name, path): name
                       for i, (name, path) in enumerate(jobs, 1)}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    fut.result()
                except (ApiError, OSError) as e:
                    print(f"{name}  FAILED: {e}")
                    failed.append(name)
    else:
        for i, (name, md_path) in enumerate(jobs, 1):
            try:
                work(i, name, md_path)
            except (ApiError, OSError) as e:
                print(f"[{i}/{len(jobs)}] {name}  FAILED: {e}")
                failed.append(name)

    print(f"\n{len(jobs) - len(failed)} of {len(jobs)} images in "
          f"{time.perf_counter()-t0:.1f}s -> {os.path.abspath(out_dir)}")
    if failed:
        print(f"failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
