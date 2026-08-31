r"""Backend-agnostic extraction core shared by worker.py and app_jobs.py.

The per-photo logic of app.py lifted out of the Streamlit script so a
background process can run it: build the model runner from a plain
settings dict, push one photo through either mode, and turn the results
into DataFrames / Excel. Nothing here imports Streamlit. The pipelines
themselves (equipment_pipeline*.py, image_to_excel_*.py) are untouched.

Modes
  pipeline  transcribe -> tabulate -> (verify)   three model calls per photo
  direct    one call per photo straight into the schedule fields

Backends
  api       OpenAI-compatible endpoint (equipment_pipeline_api.py)
  cli       the locally installed `claude` command (equipment_pipeline.py)
"""

import io
import os
import subprocess
import time
import zipfile

import pandas as pd
import requests

import equipment_pipeline as cli_pipe
import image_to_excel_api as direct_api
import image_to_excel_cli as direct_cli
from equipment_pipeline_api import (
    COLUMNS, EXTS, PROVIDERS, ApiError, PhotoTimeout,
    build_excel_workbook, dedupe_records, parse_table,
    transcribe_image, extract_table, verify_table, write_excel,
)

DEFAULT_SETTINGS = {
    "mode": "pipeline",           # "pipeline" | "direct"
    "backend": "api",             # "api" | "cli"
    "cli_model": "claude-opus-5",
    "claude_path": "",
    "provider": "",               # "" = .env default
    "model_override": "",
    "base_url_override": "",
    "key_env_override": "",
    "do_verify": True,
    "photo_timeout": 300,         # seconds per photo, 0 = no budget
    "call_timeout": 600,
    "max_side": 3000,
    "max_tokens": 16384,
}


def describe(settings: dict) -> str:
    """Short human label of a settings dict (no endpoint resolution)."""
    s = {**DEFAULT_SETTINGS, **settings}
    if s["mode"] == "direct":
        mode = "Direct fields"
    else:
        mode = "Full pipeline" + (" + verify" if s["do_verify"] else "")
    if s["backend"] == "cli":
        backend = f"Claude CLI / {s['cli_model'] or 'claude-opus-5'}"
    else:
        backend = (s["provider"] or "API (.env)") + (
            f" / {s['model_override']}" if s["model_override"] else "")
    return f"{mode} | {backend}"


# --------------------------------------------------------------------------
# API endpoint from settings + .env (same resolution rules as the pipelines)
# --------------------------------------------------------------------------

def build_api(provider, model_override, base_url_override, key_env_override,
              timeout, max_tokens):
    """Return the api dict the pipeline stages expect, or (None, error)."""
    env_provider = not provider          # "" means: use .env / preset defaults
    if env_provider:
        provider = os.environ.get("VLM_PROVIDER", "").lower() or "openai"
    if provider not in PROVIDERS:
        return None, (f"VLM_PROVIDER in .env must be one of "
                      f"{', '.join(sorted(PROVIDERS))} (got '{provider}')")

    def env_default(name):
        return os.environ.get(name) if env_provider else None

    preset_url, preset_env, preset_model = PROVIDERS[provider]
    base_url = (base_url_override or env_default("VLM_BASE_URL")
                or preset_url or "").rstrip("/")
    if not base_url:
        return None, "a base URL is required with the custom provider"
    model = model_override or env_default("VLM_MODEL") or preset_model
    if not model:
        return None, "a model name is required with the custom provider"

    key_env = key_env_override or env_default("VLM_API_KEY_ENV") or preset_env
    api_key = os.environ.get(key_env, "")
    if not api_key and provider != "custom":
        return None, (f"no API key: set {key_env} in the environment or in "
                      f".env (template: .env.example)")

    return {
        "session": requests.Session(),
        "base_url": base_url,
        "headers": {"Authorization": f"Bearer {api_key}"} if api_key else {},
        "model": model,
        "timeout": timeout,
        "max_tokens": max_tokens,
        "provider": provider,
    }, None


# --------------------------------------------------------------------------
# runner: the stage calls wired to the chosen backend
# --------------------------------------------------------------------------

def make_runner(settings: dict):
    """(runner, None) or (None, error) for a settings dict.

    runner["deadline"] is set per photo by set_photo_deadline(); left()
    reads it live so every model call is capped by the photo's remaining
    budget.
    """
    s = {**DEFAULT_SETTINGS, **settings}
    call_timeout = float(s["call_timeout"])
    state = {"deadline": None}

    def left():
        if state["deadline"] is None:
            return call_timeout
        return max(0.01, min(call_timeout,
                             state["deadline"] - time.perf_counter()))

    if s["backend"] == "cli":
        try:
            claude = cli_pipe.find_claude((s["claude_path"] or "").strip()
                                          or None)
        except SystemExit as e:
            return None, str(e)
        mdl = (s["cli_model"] or "").strip() or "claude-opus-5"
        runner = {
            "label": f"claude CLI / {mdl}",
            "mode": s["mode"],
            "photo_timeout": float(s["photo_timeout"]),
            "state": state,
            "extract": lambda path: direct_cli.extract_image(
                claude, path, mdl, max(1, int(left()))),
            "transcribe": lambda path: cli_pipe.transcribe_image(
                claude, path, mdl, left()),
            "tabulate": lambda text: cli_pipe.extract_table(
                claude, text, mdl, left()),
            "verify": lambda path, text, table: cli_pipe.verify_table(
                claude, path, text, table, mdl, left()),
            "timeout_exc": (subprocess.TimeoutExpired,),
            "fail_exc": (RuntimeError, OSError),
        }
        return runner, None

    api, err = build_api(s["provider"], (s["model_override"] or "").strip(),
                         (s["base_url_override"] or "").strip(),
                         (s["key_env_override"] or "").strip(),
                         call_timeout, int(s["max_tokens"]))
    if err:
        return None, err
    max_side = int(s["max_side"])
    runner = {
        "label": f"{api['provider']} / {api['model']}",
        "mode": s["mode"],
        "photo_timeout": float(s["photo_timeout"]),
        "state": state,
        "extract": lambda path: direct_api.extract_image(api, path, max_side),
        "transcribe": lambda path: transcribe_image(api, path, max_side),
        "tabulate": lambda text: extract_table(api, text),
        "verify": lambda path, text, table: verify_table(
            api, path, text, table, max_side),
        "timeout_exc": (PhotoTimeout,),
        "fail_exc": (ApiError, OSError),
    }
    api["deadline"] = None
    runner["set_deadline"] = lambda d: api.__setitem__("deadline", d)
    return runner, None


def set_photo_deadline(runner) -> None:
    pt = runner["photo_timeout"]
    d = (time.perf_counter() + pt) if pt > 0 else None
    runner["state"]["deadline"] = d
    if "set_deadline" in runner:
        runner["set_deadline"](d)


# --------------------------------------------------------------------------
# per-photo extraction (never raises)
# --------------------------------------------------------------------------

def _blank(name):
    return {"image": name, "text": None, "table": None, "rows": None,
            "verified": False, "timed_out": False, "error": None}


def run_photo(runner, image_path, name, do_verify, stage_cb=None):
    """Full pipeline: transcribe -> tabulate -> (verify)."""
    say = stage_cb or (lambda msg: None)
    res = _blank(name)
    try:
        say("transcribing")
        res["text"] = runner["transcribe"](image_path).strip()
        if not res["text"]:
            return res
        say("building the table")
        res["table"] = runner["tabulate"](res["text"])
        _, draft_rows = parse_table(res["table"])
        if do_verify and draft_rows:
            say("verifying against the photo")
            try:
                res["table"] = runner["verify"](image_path, res["text"],
                                                res["table"])
                res["verified"] = True
            except runner["timeout_exc"]:
                res["timed_out"] = True   # unverified draft table kept
    except runner["timeout_exc"]:
        res["timed_out"] = True
    except runner["fail_exc"] as e:
        res["error"] = str(e)
    return res


def run_photo_direct(runner, image_path, name, stage_cb=None):
    """Direct mode: one call per photo straight into schedule fields."""
    say = stage_cb or (lambda msg: None)
    res = _blank(name)
    try:
        say("extracting the schedule fields")
        res["rows"] = runner["extract"](image_path)
    except runner["timeout_exc"]:
        res["timed_out"] = True
    except runner["fail_exc"] + (ValueError,) as e:
        res["error"] = str(e)
    return res


def run_one(runner, settings, image_path, name, stage_cb=None):
    """Dispatch on mode; sets the photo deadline first."""
    s = {**DEFAULT_SETTINGS, **settings}
    set_photo_deadline(runner)
    if s["mode"] == "direct":
        return run_photo_direct(runner, image_path, name, stage_cb)
    return run_photo(runner, image_path, name, s["do_verify"], stage_cb)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def gather_images(files):
    """[(name, bytes)] uploads -> ([(unique name, bytes)], [warnings]).

    ZIPs are expanded (base names only, any nesting); name collisions get a
    _2/_3... suffix so each photo keeps its own results.
    """
    raw, warnings = [], []
    for fname, data in files or []:
        if fname.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for info in z.infolist():
                        if info.is_dir():
                            continue
                        base = os.path.basename(info.filename)
                        if (base and not base.startswith(".")
                                and os.path.splitext(base)[1].lower() in EXTS):
                            raw.append((base, z.read(info)))
            except zipfile.BadZipFile:
                warnings.append(f"{fname} is not a valid ZIP file, skipped.")
        elif os.path.splitext(fname)[1].lower() in EXTS:
            raw.append((fname, data))
        else:
            warnings.append(f"{fname}: unsupported file type, skipped.")

    images, taken = [], {}
    for name, data in raw:
        stem, ext = os.path.splitext(name)
        n = taken.get(name.lower(), 0) + 1
        taken[name.lower()] = n
        images.append((name if n == 1 else f"{stem}_{n}{ext}", data))
    return images, warnings


# --------------------------------------------------------------------------
# results -> tables / files
# --------------------------------------------------------------------------

def table_to_df(table_md: str, image_name: str) -> pd.DataFrame:
    header, rows = parse_table(table_md)
    if not header:
        return pd.DataFrame(columns=["Image", *COLUMNS])
    rows = [r[:len(header)] + [""] * (len(header) - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=header)
    df.insert(0, "Image", image_name)
    return df


def rows_to_df(rows, image_name: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS).fillna("")
    df.insert(0, "Image", image_name)
    return df


def result_df(res) -> pd.DataFrame:
    if res.get("rows") is not None:
        return rows_to_df(res["rows"], res["image"])
    return table_to_df(res["table"], res["image"])


def has_rows(res) -> bool:
    if res.get("rows") is not None:
        return bool(res["rows"])
    return bool(res["table"]) and bool(parse_table(res["table"])[1])


def ok_results(results):
    return [r for r in results if r.get("table") or r.get("rows")]


def results_frame(results) -> pd.DataFrame:
    """One row per extracted unit, Image column first (not deduplicated)."""
    dfs = [result_df(r) for r in ok_results(results)]
    if not dfs:
        return pd.DataFrame(columns=["Image", *COLUMNS])
    return pd.concat(dfs, ignore_index=True).fillna("")


def summarize(results) -> dict:
    ok = ok_results(results)
    return {
        "ok": len(ok),
        "verified": sum(1 for r in ok if r.get("verified")),
        "timed_out": [r["image"] for r in results if r.get("timed_out")],
        "failed": [(r["image"], r.get("error")) for r in results
                   if r.get("error")],
        "no_data": [r["image"] for r in results
                    if not r.get("error") and not r.get("timed_out")
                    and not r.get("text") and not r.get("rows")],
        "no_rows": [r["image"] for r in ok if not has_rows(r)],
    }


def write_outputs(results, xlsx_path: str, csv_path: str) -> dict:
    """Write result.csv (per photo) and result.xlsx (units merged).

    Returns {"xlsx", "csv", "rows", "units", "merged"} or {} when there is
    nothing to write yet.
    """
    df = results_frame(results)
    if df.empty:
        return {}
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    columns = list(df.columns)
    unique = dedupe_records(df.to_dict("records"), columns)
    written = write_excel(unique, columns, xlsx_path)
    return {"xlsx": written, "csv": csv_path, "rows": len(df),
            "units": len(unique), "merged": len(df) - len(unique)}
