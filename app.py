r"""Streamlit web UI: upload nameplate photos, get the equipment list.

Accepts a single image, multiple images, or ZIP archive(s) containing images
-- everything is flattened into one photo list and extracted into one merged
equipment table with a per-photo progress bar and CSV download.

A thin front end over the direct-fields extractors (image_to_excel_api.py /
image_to_excel_cli.py): one model call per photo that answers straight in
the schedule fields. The prompts and plumbing are imported from them, so the
web app and the scripts always behave identically. The three-stage
transcribe -> tabulate -> verify pipeline is no longer offered here (the
run_photo() path below is kept but disabled; DIRECT_ONLY = True).

Two backends, chosen in the sidebar:

  API key      equipment_pipeline_api.py -- OpenAI-compatible endpoint,
               keys/defaults from .env (template: .env.example)
  Claude CLI   equipment_pipeline.py -- the locally installed, already-
               authenticated `claude` command; no API key needed

Run it with:

    streamlit run app.py
"""

import hmac
import io
import os
import shutil
import subprocess
import tempfile
import time
import zipfile

import pandas as pd
import requests
import streamlit as st

import equipment_pipeline as cli_pipe
import image_to_excel_api as direct_api
import image_to_excel_cli as direct_cli
from equipment_pipeline_api import (
    COLUMNS, EXTS, PROVIDERS, ApiError, PhotoTimeout,
    build_excel_workbook, dedupe_records, load_env_file,
    transcribe_image, extract_table, verify_table, parse_table,
)

st.set_page_config(page_title="EMA Equipment Extractor", page_icon="🔧",
                   layout="wide")

load_env_file()


def load_streamlit_secrets():
    """Hosted deployments keep keys in st.secrets, not .env.

    Streamlit Community Cloud (and a local .streamlit/secrets.toml) expose
    settings through st.secrets; copy the string entries into the
    environment so the pipelines' .env-style lookups (OPENAI_API_KEY,
    VLM_PROVIDER, ...) work unchanged. Real environment variables win.
    """
    try:
        items = list(st.secrets.items())
    except Exception:                       # no secrets file: local use
        return
    for key, value in items:
        if isinstance(value, str) and value:
            os.environ.setdefault(key, value)


load_streamlit_secrets()


def require_password():
    """Gate the page behind APP_PASSWORD when it is set.

    Without APP_PASSWORD (environment or secrets) the app is open, which is
    fine on a private machine; set it before exposing the app publicly.
    """
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected or st.session_state.get("authed"):
        return
    st.title("🔧 Equipment List Extractor")
    with st.form("login"):
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if hmac.compare_digest(password, expected):
                st.session_state["authed"] = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()


require_password()

# The Claude CLI backend only makes sense where `claude` is installed and
# signed in (a developer machine); hosted deployments get the API backend.
try:
    cli_pipe.find_claude(None)
    CLI_AVAILABLE = True
except SystemExit:
    CLI_AVAILABLE = False

# The full transcribe -> tabulate -> verify pipeline is switched off in the
# web UI; every photo goes through the direct-fields extractor instead.
DIRECT_ONLY = True


# --------------------------------------------------------------------------
# API endpoint from sidebar + .env (same resolution rules as the pipelines)
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
        # the VLM_* endpoint settings describe one endpoint together with
        # VLM_PROVIDER, so they only apply when the provider wasn't overridden
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
# helpers
# --------------------------------------------------------------------------

def gather_images(uploaded_files):
    """Flatten the uploads into [(unique name, bytes)], expanding ZIPs.

    ZIP entries keep only their base name; images inside nested folders are
    found regardless of the archive's layout. Name collisions across files
    and archives get a _2/_3... suffix so each photo keeps its own results.
    """
    raw = []
    for uf in uploaded_files or []:
        if uf.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(uf.getvalue())) as z:
                    for info in z.infolist():
                        if info.is_dir():
                            continue
                        base = os.path.basename(info.filename)
                        if (base and not base.startswith(".")
                                and os.path.splitext(base)[1].lower() in EXTS):
                            raw.append((base, z.read(info)))
            except zipfile.BadZipFile:
                st.error(f"{uf.name} is not a valid ZIP file, skipping it.")
        else:
            raw.append((uf.name, uf.getvalue()))

    images, taken = [], {}
    for name, data in raw:
        stem, ext = os.path.splitext(name)
        n = taken.get(name.lower(), 0) + 1
        taken[name.lower()] = n
        images.append((name if n == 1 else f"{stem}_{n}{ext}", data))
    return images


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
    """One result (either mode) -> DataFrame with the Image column first."""
    if res["rows"] is not None:
        return rows_to_df(res["rows"], res["image"])
    return table_to_df(res["table"], res["image"])


def has_rows(res) -> bool:
    if res["rows"] is not None:
        return bool(res["rows"])
    return bool(res["table"]) and bool(parse_table(res["table"])[1])


def run_photo_direct(runner, image_path, name, stage_cb=None):
    """Direct mode: one call per photo straight into schedule fields.

    Same result shape as run_photo but with "rows" (list of field dicts)
    filled instead of text/table. Never raises.
    """
    say = stage_cb or (lambda msg: None)
    res = {"image": name, "text": None, "table": None, "rows": None,
           "verified": False, "timed_out": False, "error": None}
    try:
        say("extracting the schedule fields")
        res["rows"] = runner["extract"](image_path)
    except runner["timeout_exc"]:
        res["timed_out"] = True
    except runner["fail_exc"] + (ValueError,) as e:
        res["error"] = str(e)
    return res


def run_photo(runner, image_path, name, do_verify, stage_cb=None):
    """Extract one photo through the three stages; never raises.

    Returns {image, text, table, verified, timed_out, error}. A timeout in
    the verify stage keeps the unverified draft table, matching the batch
    pipelines. stage_cb, when given, is told which stage is starting.
    """
    say = stage_cb or (lambda msg: None)
    res = {"image": name, "text": None, "table": None, "rows": None,
           "verified": False, "timed_out": False, "error": None}
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


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    if DIRECT_ONLY:
        direct = True
        st.caption("Mode: direct fields -- one model call per photo, "
                   "straight into the schedule columns.")
    else:
        mode = st.radio(
            "Mode",
            ["Full pipeline", "Direct fields"],
            help="Full pipeline: transcribe the plate, build the table, "
                 "then re-check it against the photo (three model calls "
                 "per photo, most accurate). Direct fields: one model call "
                 "per photo that answers straight in the schedule fields.")
        direct = mode == "Direct fields"
    if CLI_AVAILABLE:
        backend = st.radio(
            "Backend",
            ["API key", "Claude CLI"],
            help="API key: OpenAI-compatible endpoint configured via .env / "
                 "secrets / the fields below (equipment_pipeline_api.py). "
                 "Claude CLI: the locally installed, already-authenticated "
                 "`claude` command (equipment_pipeline.py) -- no API key "
                 "needed.")
    else:
        backend = "API key"
    use_cli = backend == "Claude CLI"
    if use_cli:
        cli_model = st.text_input("Claude model", value="claude-opus-5")
    else:
        provider = st.selectbox(
            "Provider",
            ["", *sorted(PROVIDERS)],
            format_func=lambda v: v if v else ".env default",
            help="Blank uses VLM_PROVIDER / VLM_MODEL / VLM_BASE_URL from "
                 ".env (falling back to openai). Picking a provider uses its "
                 "preset endpoint and key variable.")
        model_override = st.text_input(
            "Model (blank = preset / .env)", value="")
    if direct:
        do_verify = False
    else:
        do_verify = st.toggle(
            "Verify against the image", value=True,
            help="Extra pass that re-reads each photo to fix OCR errors and "
                 "fill missing cells. Slower but more accurate.")
    photo_timeout = st.number_input(
        "Photo time budget (seconds)", min_value=0, value=300, step=30,
        help="If one photo exceeds this, whatever was extracted so far is "
             "kept and the run moves to the next photo. 0 disables the "
             "budget.")
    with st.expander("Advanced"):
        call_timeout = st.number_input(
            "Timeout per model call (s)", min_value=10, value=600, step=30)
        if use_cli:
            claude_path = st.text_input(
                "Path to the claude CLI (blank = auto-detect)", value="")
        else:
            base_url_override = st.text_input("Base URL override", value="")
            key_env_override = st.text_input(
                "API key env var override", value="",
                help="Name of the environment variable holding the key")
            max_side = st.number_input(
                "Max image side (px, 0 = no downscale)",
                min_value=0, value=3000, step=500)
            max_tokens = st.number_input(
                "Max response tokens", min_value=256, value=16384, step=256)


def make_runner():
    """Wire the three stage calls to the chosen backend, or (None, error).

    The runner's "deadline" entry is set per photo; left() reads it live so
    every model call is capped by that photo's remaining budget.
    """
    state = {"deadline": None}

    def left():
        if state["deadline"] is None:
            return float(call_timeout)
        return max(0.01, min(float(call_timeout),
                             state["deadline"] - time.perf_counter()))

    if use_cli:
        try:
            claude = cli_pipe.find_claude(claude_path.strip() or None)
        except SystemExit as e:
            return None, str(e)
        mdl = cli_model.strip() or "claude-opus-5"
        return {
            "label": f"claude CLI / {mdl}",
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
        }, None

    api, err = build_api(provider, model_override.strip(),
                         base_url_override.strip(), key_env_override.strip(),
                         float(call_timeout), int(max_tokens))
    if err:
        return None, err
    runner = {
        "label": f"{api['provider']} / {api['model']}",
        "state": state,
        "extract": lambda path: direct_api.extract_image(
            api, path, int(max_side)),
        "transcribe": lambda path: transcribe_image(api, path, int(max_side)),
        "tabulate": lambda text: extract_table(api, text),
        "verify": lambda path, text, table: verify_table(
            api, path, text, table, int(max_side)),
        "timeout_exc": (PhotoTimeout,),
        "fail_exc": (ApiError, OSError),
    }
    # the API stage functions read the deadline from the api dict; share the
    # same mutable state so per-photo updates reach both
    api["deadline"] = None
    runner["set_deadline"] = lambda d: api.__setitem__("deadline", d)
    return runner, None


def set_photo_deadline(runner):
    d = (time.perf_counter() + photo_timeout) if photo_timeout > 0 else None
    runner["state"]["deadline"] = d
    if "set_deadline" in runner:
        runner["set_deadline"](d)


# --------------------------------------------------------------------------
# main page
# --------------------------------------------------------------------------

st.title("🔧 Equipment List Extractor")
st.caption("Upload one photo, many photos, or a ZIP of a whole folder of "
           "equipment nameplates (HVAC and similar). Each photo is read "
           "straight into an equipment-schedule row; everything lands in "
           "one table with Excel and CSV downloads.")

# The uploader can't be cleared programmatically, so "New upload" swaps its
# widget key: the fresh key mounts an empty uploader and drops the old files.
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0


def reset_uploads():
    st.session_state["uploader_key"] += 1
    st.session_state.pop("batch", None)


uploaded_files = st.file_uploader(
    "Nameplate photo(s) or ZIP folder",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "zip"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}")

images = gather_images(uploaded_files)
batch_key = tuple(name for name, _ in images)

if images:
    single = len(images) == 1
    if single:
        left_col, right_col = st.columns([1, 2])
        with left_col:
            st.image(images[0][1], caption=images[0][0], width="stretch")
        run_col = right_col
    else:
        st.caption(f"{len(images)} photo(s) queued")
        with st.expander("Photos in this batch"):
            st.write("\n".join(f"- {name}" for name, _ in images))
        run_col = st.container()

    with run_col:
        extract_col, new_col = st.columns([3, 1])
        with new_col:
            st.button("🔄 New upload", on_click=reset_uploads,
                      help="Clear these photos and results to start over")
        label = ("Extract equipment list" if single else
                 f"Extract equipment list ({len(images)} photos)")
        if extract_col.button(label, type="primary"):
            runner, err = make_runner()
            if err:
                st.error(err)
                st.stop()

            # the stage functions read images from disk
            tmp_dir = tempfile.mkdtemp(prefix="ema_upload_")
            results = []
            try:
                if single:
                    name, data = images[0]
                    image_path = os.path.join(tmp_dir, name)
                    with open(image_path, "wb") as f:
                        f.write(data)
                    set_photo_deadline(runner)
                    with st.status(f"Extracting with {runner['label']} ...",
                                   expanded=True) as status:
                        say = lambda msg: st.write(msg.capitalize() + " ...")
                        res = (run_photo_direct(runner, image_path, name,
                                                stage_cb=say) if direct else
                               run_photo(runner, image_path, name, do_verify,
                                         stage_cb=say))
                        status.update(
                            label="Done" if not res["error"]
                            else "Extraction failed",
                            state="error" if res["error"] else "complete",
                            expanded=False)
                    results.append(res)
                else:
                    progress = st.progress(
                        0.0, text=f"Extracting {len(images)} photos with "
                                  f"{runner['label']} ...")
                    for idx, (name, data) in enumerate(images):
                        image_path = os.path.join(tmp_dir, name)
                        with open(image_path, "wb") as f:
                            f.write(data)
                        progress.progress(
                            idx / len(images),
                            text=f"[{idx + 1}/{len(images)}] {name} ...")
                        set_photo_deadline(runner)
                        results.append(
                            run_photo_direct(runner, image_path, name)
                            if direct else
                            run_photo(runner, image_path, name, do_verify))
                        os.remove(image_path)   # keep the temp dir small
                    progress.progress(
                        1.0, text=f"Done -- {len(images)} photo(s) processed")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            st.session_state["batch"] = {"key": batch_key, "items": results}

    # ---- results (kept in session state so downloads don't re-run) --------
    batch = st.session_state.get("batch")
    if batch and batch["key"] == batch_key:
        items = batch["items"]
        ok = [r for r in items if r["table"] or r["rows"]]
        verified = [r for r in ok if r["verified"]]
        timed = [r["image"] for r in items if r["timed_out"]]
        failed = [(r["image"], r["error"]) for r in items if r["error"]]
        # pipeline mode: nothing transcribed; direct mode: model returned []
        no_text = [r["image"] for r in items
                   if not r["error"] and not r["timed_out"]
                   and not r["text"] and not r["rows"]]

        if ok:
            st.success(f"Extracted {len(ok)} of {len(items)} photo(s)"
                       + (f" ({len(verified)} verified against the image)"
                          if verified else "") + ".")
        if timed:
            st.warning("Photo time budget hit (partial results kept): "
                       + ", ".join(timed))
        if no_text:
            st.info("Skipped -- no equipment nameplate data found: "
                    + ", ".join(no_text))
        no_rows = [r["image"] for r in ok if not has_rows(r)]
        if no_rows:
            st.info("Skipped -- text found but no equipment data in it: "
                    + ", ".join(no_rows))
        if failed:
            with st.expander(f"{len(failed)} photo(s) failed", expanded=True):
                for name, error in failed:
                    st.error(f"{name}: {error}")

        dfs = [result_df(r) for r in ok]
        # concat gap-fills mismatched columns with NaN; keep cells as strings
        df = (pd.concat(dfs, ignore_index=True).fillna("") if dfs
              else pd.DataFrame(columns=["Image", *COLUMNS]))
        if not df.empty:
            st.subheader("Equipment list")
            st.dataframe(df, width="stretch", hide_index=True)
            csv_stem = (os.path.splitext(items[0]["image"])[0]
                        if len(items) == 1 else "equipment_list")
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{csv_stem}.csv", mime="text/csv")
            # Excel merges rows for the same physical unit (same serial /
            # identical data) and lists all its photos in the Image cell
            unique = dedupe_records(df.to_dict("records"), list(df.columns))
            xbuf = io.BytesIO()
            build_excel_workbook(unique, list(df.columns)).save(xbuf)
            dups = len(df) - len(unique)
            st.download_button(
                "Download Excel"
                + (f" ({dups} duplicate photo row(s) merged)" if dups else ""),
                xbuf.getvalue(), file_name=f"{csv_stem}.xlsx",
                mime="application/vnd.openxmlformats-officedocument."
                     "spreadsheetml.sheet")
            if len(items) == 1 and items[0]["table"]:
                st.download_button(
                    "Download markdown table",
                    (items[0]["table"] + "\n").encode("utf-8"),
                    file_name=f"{csv_stem}_table.md", mime="text/markdown")
        elif ok == [] and not failed and not no_text and not timed:
            st.info("No equipment nameplate data was found.")

        if len(items) == 1 and items[0]["text"]:
            with st.expander("Raw nameplate transcription"):
                st.text(items[0]["text"])
else:
    st.session_state.pop("batch", None)
