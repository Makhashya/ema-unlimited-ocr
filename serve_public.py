r"""Serve a web UI from this PC on a public HTTPS URL.

    python serve_public.py              # app.py
    python serve_public.py app_jobs.py  # the persistent-jobs app
    run_server.bat [app]                # same, double-clickable

Starts Streamlit on a local port (bound to 127.0.0.1 only) and a Cloudflare
quick tunnel to it, prints the public https://....trycloudflare.com URL
(also written to public_url.txt), and stops both on Ctrl+C. Everything runs
on this machine, so every backend installed here works -- including the
Claude CLI. The URL changes each start; a fixed one needs a named Cloudflare
tunnel (see DEPLOY.md).

Needs: cloudflared (winget install Cloudflare.cloudflared) and APP_PASSWORD
in .env, so the URL is not open to everyone. Pass --open to skip the
password check deliberately.
"""

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from equipment_pipeline_api import load_env_file       # noqa: E402

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
CLOUDFLARED_CANDIDATES = [
    r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
    r"C:\Program Files\cloudflared\cloudflared.exe",
]


def find_cloudflared():
    found = shutil.which("cloudflared")
    if found:
        return found
    for cand in CLOUDFLARED_CANDIDATES:
        if os.path.isfile(cand):
            return cand
    return None


def port_free(port: int) -> bool:
    # Something already answering on the port (Windows lets a 127.0.0.1
    # bind coexist with another process's 0.0.0.0 bind, so test both ways).
    with socket.socket() as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return False
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def pick_port(start: int) -> int:
    port = start
    while not port_free(port):
        port += 1
    return port


def pump(stream, on_line):
    for line in iter(stream.readline, ""):
        on_line(line.rstrip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("app", nargs="?", default="app.py",
                    help="Streamlit script to serve (default: app.py)")
    ap.add_argument("--port", type=int, default=8501,
                    help="local port to try first (default: 8501; the next "
                         "free one is used if it is busy)")
    ap.add_argument("--open", action="store_true",
                    help="allow serving without APP_PASSWORD (public!)")
    args = ap.parse_args()

    app = os.path.join(ROOT, args.app)
    if not os.path.isfile(app):
        sys.exit(f"app not found: {app}")
    cf = find_cloudflared()
    if not cf:
        sys.exit("cloudflared not found. Install it with:\n"
                 "    winget install Cloudflare.cloudflared\n"
                 "then open a new terminal and run this again.")
    load_env_file()
    if not os.environ.get("APP_PASSWORD") and not args.open:
        sys.exit("APP_PASSWORD is not set in .env -- the public URL would be "
                 "open to anyone. Add a line  APP_PASSWORD=...  to .env "
                 "(or pass --open to serve without a password).")

    port = pick_port(args.port)
    if port != args.port:
        print(f"port {args.port} is busy; using {port}")

    print(f"starting Streamlit: {args.app} on http://127.0.0.1:{port}",
          flush=True)
    web = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app,
         "--server.port", str(port), "--server.address", "127.0.0.1",
         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=ROOT)

    print("starting Cloudflare tunnel ...", flush=True)
    tunnel = subprocess.Popen(
        [cf, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")

    url_file = os.path.join(ROOT, "public_url.txt")
    state = {"url": None}
    start = time.time()

    def on_line(line):
        m = URL_RE.search(line)
        if m and not state["url"]:
            state["url"] = m.group(0)
            with open(url_file, "w", encoding="utf-8") as f:
                f.write(state["url"] + "\n")
            print("\n" + "=" * 64)
            print(f"  PUBLIC URL:  {state['url']}")
            print(f"  (also saved to {url_file})")
            print("  Keep this window open. Ctrl+C stops the server.")
            print("=" * 64 + "\n")
        elif "error" in line.lower() or "failed" in line.lower():
            print("cloudflared:", line)

    threading.Thread(target=pump, args=(tunnel.stdout, on_line),
                     daemon=True).start()

    try:
        while True:
            if web.poll() is not None:
                print(f"Streamlit exited with code {web.returncode}")
                break
            if tunnel.poll() is not None:
                print(f"cloudflared exited with code {tunnel.returncode}")
                break
            if not state["url"] and time.time() - start > 40:
                print("no public URL after 40 s -- check your internet "
                      "connection; still waiting ...")
                start = time.time()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        for proc in (tunnel, web):
            if proc.poll() is None:
                proc.terminate()
        for proc in (tunnel, web):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            os.remove(url_file)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
