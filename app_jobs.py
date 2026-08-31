r"""Streamlit web UI with persistent jobs: queue, close the tab, come back.

    streamlit run app_jobs.py

Each upload (photos or ZIPs) becomes a job on disk under ./jobs (see
job_store.py). A detached worker process (worker.py) runs the queue; this
page only reads the job folder, so:

  - closing the browser or restarting Streamlit does not stop a running
    extraction; reopen the page and the progress bar picks up where it is
  - every finished job keeps its Excel / CSV for download until you delete
    the job -- run a second ZIP and both stay listed

Settings in the sidebar are captured per job at queue time. The previous
single-session UI is app.py; both use the same pipelines.
"""

import os
import time

import pandas as pd
import streamlit as st

import extract_core as core
import job_store as js
from equipment_pipeline_api import PROVIDERS, load_env_file

st.set_page_config(page_title="EMA Equipment Extractor", page_icon="🔧",
                   layout="wide")
load_env_file()

STATE_BADGE = {
    "queued": ("⏳ Queued", "secondary"),
    "running": ("⚙️ Running", "primary"),
    "done": ("✅ Done", "primary"),
    "failed": ("❌ Failed", "primary"),
    "cancelled": ("⏹ Cancelled", "secondary"),
}


def fmt_ts(ts):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else ""


def fmt_dur(seconds):
    if seconds is None:
        return ""
    seconds = int(seconds)
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 \
        else f"{seconds}s"


# --------------------------------------------------------------------------
# sidebar -> settings dict captured with each job
# --------------------------------------------------------------------------

settings = dict(core.DEFAULT_SETTINGS)
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Mode", ["Full pipeline", "Direct fields"],
        help="Full pipeline: transcribe the plate, build the table, then "
             "re-check it against the photo (three model calls per photo, "
             "most accurate). Direct fields: one model call per photo that "
             "answers straight in the schedule fields (about 3x faster).")
    direct = mode == "Direct fields"
    settings["mode"] = "direct" if direct else "pipeline"
    backend = st.radio(
        "Backend", ["API key", "Claude CLI"],
        help="API key: OpenAI-compatible endpoint configured via .env / the "
             "fields below. Claude CLI: the locally installed, already-"
             "authenticated `claude` command -- no API key needed.")
    use_cli = backend == "Claude CLI"
    settings["backend"] = "cli" if use_cli else "api"
    if use_cli:
        settings["cli_model"] = st.text_input("Claude model",
                                              value="claude-opus-5")
    else:
        settings["provider"] = st.selectbox(
            "Provider", ["", *sorted(PROVIDERS)],
            format_func=lambda v: v if v else ".env default",
            help="Blank uses VLM_PROVIDER / VLM_MODEL / VLM_BASE_URL from "
                 ".env (falling back to openai).")
        settings["model_override"] = st.text_input(
            "Model (blank = preset / .env)", value="")
    if direct:
        settings["do_verify"] = False
    else:
        settings["do_verify"] = st.toggle(
            "Verify against the image", value=True,
            help="Extra pass that re-reads each photo to fix OCR errors and "
                 "fill missing cells. Slower but more accurate.")
    settings["photo_timeout"] = st.number_input(
        "Photo time budget (seconds)", min_value=0, value=300, step=30,
        help="A photo still unfinished after this is skipped (partial "
             "results kept) and the job moves on. 0 disables the budget.")
    with st.expander("Advanced"):
        settings["call_timeout"] = st.number_input(
            "Timeout per model call (s)", min_value=10, value=600, step=30)
        if use_cli:
            settings["claude_path"] = st.text_input(
                "Path to the claude CLI (blank = auto-detect)", value="")
        else:
            settings["base_url_override"] = st.text_input(
                "Base URL override", value="")
            settings["key_env_override"] = st.text_input(
                "API key env var override", value="",
                help="Name of the environment variable holding the key")
            settings["max_side"] = st.number_input(
                "Max image side (px, 0 = no downscale)",
                min_value=0, value=3000, step=500)
            settings["max_tokens"] = st.number_input(
                "Max response tokens", min_value=256, value=16384, step=256)
    # number_input returns numpy/ints; keep the dict JSON-friendly
    for k in ("photo_timeout", "call_timeout", "max_side", "max_tokens"):
        settings[k] = int(settings[k])

    st.divider()
    info = js.worker_info()
    if js.worker_alive():
        st.caption(f"🟢 Worker running (pid {info['pid']}, since "
                   f"{fmt_ts(info.get('started'))})")
    else:
        st.caption("⚪ Worker idle -- starts automatically when a job is "
                   "queued")
        if st.button("Start worker now"):
            js.ensure_worker()
            time.sleep(1.5)
            st.rerun()
    st.caption(f"Jobs folder: `{js.JOBS_DIR}`")


# --------------------------------------------------------------------------
# new job
# --------------------------------------------------------------------------

st.title("🔧 Equipment List Extractor")
st.caption("Upload nameplate photos or a ZIP, queue the extraction, and come "
           "back any time: jobs keep running in the background and finished "
           "Excel files stay here for download.")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

uploaded = st.file_uploader(
    "Nameplate photo(s) or ZIP folder(s)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "zip"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}")

images, warnings = core.gather_images(
    [(f.name, f.getvalue()) for f in uploaded or []])
for w in warnings:
    st.warning(w)

if images:
    if len(uploaded) == 1:
        default_name = os.path.splitext(uploaded[0].name)[0]
    else:
        default_name = f"batch {time.strftime('%Y-%m-%d %H:%M')}"
    name_col, btn_col = st.columns([3, 1])
    job_name = name_col.text_input("Job name", value=default_name)
    with btn_col:
        st.write("")            # align with the text input
        queue = st.button(f"Queue extraction ({len(images)} photos)",
                          type="primary", width="stretch")
    with st.expander("Photos in this upload"):
        st.write("\n".join(f"- {n}" for n, _ in images))
    if queue:
        job_id = js.new_job(job_name.strip() or default_name, images,
                            settings)
        started = js.ensure_worker()
        st.session_state["uploader_key"] += 1
        st.toast(f"Queued '{job_name}' ({len(images)} photos)"
                 + (" -- worker started" if started else ""))
        time.sleep(0.5)
        st.rerun()


# --------------------------------------------------------------------------
# jobs list (auto-refreshes while anything is queued / running)
# --------------------------------------------------------------------------

def render_job(job: dict) -> None:
    jid, state = job["id"], job["state"]
    counts = job.get("counts") or {}
    outputs = job.get("outputs") or {}
    total, done = job.get("total", 0), job.get("done", 0)

    with st.container(border=True):
        head, meta, badge = st.columns([5, 4, 2])
        head.markdown(f"**{job['name']}**")
        head.caption(f"{total} photo(s) · {core.describe(job['settings'])}")
        meta.caption(f"Queued {fmt_ts(job['created'])}")
        if job.get("started"):
            end = job.get("finished") or time.time()
            meta.caption(f"{'Ran' if job.get('finished') else 'Running'} "
                         f"{fmt_dur(end - job['started'])}")
        label, _ = STATE_BADGE.get(state, (state, "secondary"))
        badge.markdown(f"### {label}")

        if state == "queued":
            st.info("Waiting for the worker to pick this job up ...")
        elif state == "running":
            frac = done / total if total else 0.0
            current = job.get("current")
            st.progress(frac, text=f"{done}/{total} photos done"
                        + (f" -- working on {current}" if current else ""))
        elif state == "failed":
            st.error(job.get("error") or "unknown error")
        elif state == "cancelled":
            st.warning(f"Cancelled after {done}/{total} photos.")
        elif state == "done":
            msg = (f"Extracted data from {counts.get('ok', 0)} of {total} "
                   f"photo(s)")
            if counts.get("verified"):
                msg += f" ({counts['verified']} verified against the image)"
            if outputs:
                msg += (f" -- {outputs['units']} unit(s) in Excel"
                        + (f", {outputs['merged']} duplicate photo row(s) "
                           f"merged" if outputs.get("merged") else ""))
            st.success(msg + ".")

        if counts.get("timed_out"):
            st.warning("Photo time budget hit (partial results kept): "
                       + ", ".join(counts["timed_out"]))
        if counts.get("no_data"):
            st.info("No equipment nameplate data found: "
                    + ", ".join(counts["no_data"]))
        if counts.get("no_rows"):
            st.info("Text found but no equipment data in it: "
                    + ", ".join(counts["no_rows"]))
        if counts.get("failed"):
            with st.expander(f"{len(counts['failed'])} photo(s) failed",
                             expanded=state == "done"):
                for name, error in counts["failed"]:
                    st.error(f"{name}: {error}")

        # ---- actions -----------------------------------------------------
        cols = st.columns([1.2, 1.2, 1.4, 4])
        stem = "".join(c if c.isalnum() or c in "-_ " else "_"
                       for c in job["name"]).strip() or jid
        partial = "" if state == "done" else " (partial)"
        xlsx, csv = outputs.get("xlsx"), outputs.get("csv")
        if xlsx and os.path.exists(xlsx):
            with open(xlsx, "rb") as f:
                cols[0].download_button(
                    f"⬇ Excel{partial}", f.read(), file_name=f"{stem}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument."
                         "spreadsheetml.sheet", key=f"xlsx_{jid}",
                    width="stretch")
        if csv and os.path.exists(csv):
            with open(csv, "rb") as f:
                cols[1].download_button(
                    f"⬇ CSV{partial}", f.read(), file_name=f"{stem}.csv",
                    mime="text/csv", key=f"csv_{jid}", width="stretch")
        if state in js.ACTIVE_STATES:
            if cols[2].button("⏹ Cancel", key=f"cancel_{jid}",
                              width="stretch"):
                js.request_cancel(jid)
                st.rerun(scope="app")
        else:
            with cols[2].popover("🗑 Delete", width="stretch"):
                st.write("Removes the photos and the results of this job.")
                if st.button("Delete permanently", key=f"del_{jid}",
                             type="primary"):
                    js.delete_job(jid)
                    st.rerun(scope="app")

        if csv and os.path.exists(csv):
            with st.expander("Preview equipment list"):
                try:
                    df = pd.read_csv(csv, dtype=str, keep_default_na=False,
                                     encoding="utf-8-sig")
                    st.dataframe(df, width="stretch", hide_index=True)
                except (OSError, ValueError) as e:
                    st.caption(f"(preview unavailable: {e})")


_jobs_now = js.list_jobs()
_active = any(j["state"] in js.ACTIVE_STATES for j in _jobs_now)


@st.fragment(run_every=3 if _active else None)
def jobs_section():
    jobs = js.list_jobs()
    active = [j for j in jobs if j["state"] in js.ACTIVE_STATES]
    if _active and not active:
        st.rerun(scope="app")      # everything finished: stop polling
    st.subheader(f"Jobs ({len(jobs)})")
    if not jobs:
        st.info("No jobs yet. Upload photos above and queue an extraction.")
        return
    if active and not js.worker_alive():
        c1, c2 = st.columns([4, 1])
        c1.warning("Jobs are waiting but no worker is running.")
        if c2.button("Start worker", type="primary", width="stretch"):
            js.ensure_worker()
            time.sleep(1.5)
            st.rerun(scope="app")
    for job in jobs:
        render_job(job)


jobs_section()
