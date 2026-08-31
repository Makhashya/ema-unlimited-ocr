r"""On-disk job store shared by the web UI (app_jobs.py) and the worker.

Every extraction is a folder under JOBS_DIR (./jobs beside this file, or
$EMA_JOBS_DIR), so jobs survive closing the browser tab, restarting
Streamlit, or rebooting -- the page just re-reads the folder:

    jobs/
      worker.json           {"pid", "started", "heartbeat"}  -- worker liveness
      worker.log            stdout/stderr of a worker started by the UI
      <job_id>/
        status.json         metadata, settings, progress, state
        input/              the uploaded photos (ZIPs already expanded)
        results.json        one entry per finished photo (resume point)
        result.xlsx         merged equipment list (partial while running)
        result.csv          one row per photo (partial while running)

Job states:  queued -> running -> done | failed | cancelled
A job left "running" by a worker that died is re-queued on the next worker
start and resumes after the last finished photo.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import uuid

JOBS_DIR = os.environ.get("EMA_JOBS_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "jobs")
WORKER_FILE = os.path.join(JOBS_DIR, "worker.json")
WORKER_LOG = os.path.join(JOBS_DIR, "worker.log")
WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "worker.py")
HEARTBEAT_STALE = 20.0          # seconds without a heartbeat = worker dead

ACTIVE_STATES = ("queued", "running")


# --------------------------------------------------------------------------
# low-level JSON helpers (atomic write, tolerant read)
# --------------------------------------------------------------------------

def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:        # reader has it open (Windows)
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def _read_json(path: str, default=None):
    for attempt in range(5):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.05 * (attempt + 1))   # mid-replace; retry
    return default


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------

def job_dir(job_id: str) -> str:
    return os.path.join(JOBS_DIR, job_id)


def input_dir(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "input")


def status_path(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "status.json")


def results_path(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "results.json")


def output_path(job_id: str, ext: str) -> str:
    return os.path.join(job_dir(job_id), f"result.{ext}")


def new_job(name: str, images, settings: dict) -> str:
    """Create a queued job from [(file name, bytes)] and return its id."""
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    os.makedirs(input_dir(job_id))
    for fname, data in images:
        with open(os.path.join(input_dir(job_id), fname), "wb") as f:
            f.write(data)
    status = {
        "id": job_id,
        "name": name or job_id,
        "created": time.time(),
        "settings": settings,
        "state": "queued",
        "total": len(images),
        "done": 0,
        "current": None,
        "started": None,
        "finished": None,
        "error": None,
        "cancel": False,
        "counts": {},
        "outputs": {},
        "worker_pid": None,
    }
    write_status(job_id, status)
    return job_id


def read_status(job_id: str):
    return _read_json(status_path(job_id))


def write_status(job_id: str, status: dict) -> None:
    _write_json(status_path(job_id), status)


def update_status(job_id: str, **fields):
    status = read_status(job_id)
    if status is None:
        return None
    status.update(fields)
    write_status(job_id, status)
    return status


def read_results(job_id: str):
    return _read_json(results_path(job_id), default=[])


def write_results(job_id: str, results) -> None:
    _write_json(results_path(job_id), results)


def list_jobs():
    """All jobs' status dicts, newest first (folders without one skipped)."""
    if not os.path.isdir(JOBS_DIR):
        return []
    jobs = []
    for entry in os.listdir(JOBS_DIR):
        st = _read_json(os.path.join(JOBS_DIR, entry, "status.json"))
        if st and st.get("id"):
            jobs.append(st)
    jobs.sort(key=lambda s: s.get("created", 0), reverse=True)
    return jobs


def request_cancel(job_id: str) -> None:
    status = read_status(job_id)
    if status and status["state"] in ACTIVE_STATES:
        if status["state"] == "queued":
            status["state"] = "cancelled"
            status["finished"] = time.time()
        status["cancel"] = True
        write_status(job_id, status)


def delete_job(job_id: str) -> None:
    shutil.rmtree(job_dir(job_id), ignore_errors=True)


# --------------------------------------------------------------------------
# worker liveness / launch
# --------------------------------------------------------------------------

def write_heartbeat(started: float) -> None:
    _write_json(WORKER_FILE, {"pid": os.getpid(), "started": started,
                              "heartbeat": time.time()})


def worker_info():
    return _read_json(WORKER_FILE)


def worker_alive() -> bool:
    info = worker_info()
    return bool(info) and (time.time() - info.get("heartbeat", 0)
                           < HEARTBEAT_STALE)


def ensure_worker() -> bool:
    """Start a detached worker process unless one is alive. True if started.

    The worker outlives the Streamlit process that launched it, so closing
    the tab or stopping the app does not interrupt a running extraction.
    Its output goes to jobs/worker.log.
    """
    if worker_alive():
        return False
    os.makedirs(JOBS_DIR, exist_ok=True)
    log = open(WORKER_LOG, "a", encoding="utf-8")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                               | getattr(subprocess, "DETACHED_PROCESS", 8))
    else:
        kw["start_new_session"] = True
    subprocess.Popen([sys.executable, WORKER_SCRIPT],
                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                     cwd=os.path.dirname(WORKER_SCRIPT), close_fds=True, **kw)
    log.close()
    return True
