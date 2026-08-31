r"""Background job runner for the persistent web UI (app_jobs.py).

    python worker.py

Polls the job store (job_store.py) for queued jobs and runs them one at a
time, writing progress after every photo so the web page can show it live
and a job can resume where it stopped if the worker is restarted. The UI
starts a worker automatically when one is not alive, as a detached
process, so closing the browser tab or stopping Streamlit never interrupts
an extraction. Output goes to the console (or jobs/worker.log when started
by the UI). The worker exits after 30 idle minutes; the UI restarts it on
demand.
"""

import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import extract_core as core          # noqa: E402
import job_store as js               # noqa: E402
from equipment_pipeline_api import EXTS, load_env_file, natural_key  # noqa

POLL_SECONDS = 2.0
IDLE_EXIT_SECONDS = 30 * 60
HEARTBEAT_SECONDS = 5.0


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def heartbeat_loop(started: float, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            js.write_heartbeat(started)
        except OSError:
            pass
        stop.wait(HEARTBEAT_SECONDS)


def recover_stale() -> None:
    """Jobs left 'running' by a dead worker go back to the queue."""
    for st in js.list_jobs():
        if st["state"] == "running":
            log(f"[{st['id']}] was running under a previous worker; "
                f"re-queued (resumes after photo {st.get('done', 0)})")
            js.update_status(st["id"], state="queued", current=None,
                             worker_pid=None)


def next_job():
    queued = [s for s in js.list_jobs() if s["state"] == "queued"]
    queued.sort(key=lambda s: s.get("created", 0))
    return queued[0] if queued else None


def write_outputs(job_id: str, results) -> dict:
    try:
        return core.write_outputs(results, js.output_path(job_id, "xlsx"),
                                  js.output_path(job_id, "csv"))
    except Exception as e:                      # noqa: BLE001
        log(f"[{job_id}] could not write outputs: {e}")
        return {}


def finish(job_id: str, results, state: str, error: str = None) -> None:
    outputs = write_outputs(job_id, results)
    counts = core.summarize(results)
    js.update_status(job_id, state=state, finished=time.time(), current=None,
                     done=len(results), counts=counts, outputs=outputs,
                     error=error)
    log(f"[{job_id}] {state}: {counts['ok']} photo(s) with data, "
        f"{len(counts['timed_out'])} timed out, {len(counts['failed'])} "
        f"failed, {len(counts['no_data'])} without data"
        + (f" -> {outputs['xlsx']}" if outputs else ""))


def run_job(status: dict) -> None:
    job_id = status["id"]
    settings = status.get("settings") or {}
    results = js.read_results(job_id)
    done_names = {r["image"] for r in results}
    names = sorted((f for f in os.listdir(js.input_dir(job_id))
                    if os.path.splitext(f)[1].lower() in EXTS),
                   key=natural_key)

    runner, err = core.make_runner(settings)
    if err:
        log(f"[{job_id}] cannot start: {err}")
        finish(job_id, results, "failed", error=err)
        return

    js.update_status(job_id, state="running", worker_pid=os.getpid(),
                     started=status.get("started") or time.time(),
                     total=len(names), done=len(results), current=None,
                     error=None)
    log(f"[{job_id}] {status['name']}: {len(names)} photo(s) via "
        f"{runner['label']}, {core.describe(settings)}"
        + (f", {len(done_names)} already done" if done_names else ""))

    for name in names:
        if name in done_names:
            continue
        current = js.read_status(job_id)
        if current is None:
            log(f"[{job_id}] deleted while running; stopping")
            return
        if current.get("cancel"):
            finish(job_id, results, "cancelled")
            return
        js.update_status(job_id, current=name)
        t = time.perf_counter()
        res = core.run_one(runner, settings,
                           os.path.join(js.input_dir(job_id), name), name)
        results.append(res)
        js.write_results(job_id, results)
        outputs = write_outputs(job_id, results)
        js.update_status(job_id, done=len(results), current=None,
                         counts=core.summarize(results), outputs=outputs)
        tag = ("FAILED: " + res["error"] if res["error"] else
               "timed out" if res["timed_out"] else
               "no data" if not core.has_rows(res) else "ok")
        log(f"[{job_id}] [{len(results)}/{len(names)}] {name}  "
            f"{time.perf_counter() - t:.1f}s  {tag}")

    finish(job_id, results, "done")


def main() -> None:
    if js.worker_alive():
        log("another worker is alive; exiting")
        return
    load_env_file()
    os.makedirs(js.JOBS_DIR, exist_ok=True)
    started = time.time()
    stop = threading.Event()
    js.write_heartbeat(started)
    threading.Thread(target=heartbeat_loop, args=(started, stop),
                     daemon=True).start()
    log(f"worker {os.getpid()} started; jobs in {js.JOBS_DIR}")
    recover_stale()

    idle_since = time.time()
    try:
        while True:
            job = next_job()
            if job is None:
                if time.time() - idle_since > IDLE_EXIT_SECONDS:
                    log("idle for 30 minutes; exiting (the UI restarts a "
                        "worker when a job is queued)")
                    break
                time.sleep(POLL_SECONDS)
                continue
            try:
                run_job(job)
            except Exception:                   # noqa: BLE001
                log(f"[{job['id']}] crashed:\n{traceback.format_exc()}")
                js.update_status(job["id"], state="failed",
                                 finished=time.time(), current=None,
                                 error=traceback.format_exc().strip()
                                 .splitlines()[-1])
            idle_since = time.time()
    finally:
        stop.set()
        try:
            os.remove(js.WORKER_FILE)
        except OSError:
            pass
        log("worker stopped")


if __name__ == "__main__":
    main()
