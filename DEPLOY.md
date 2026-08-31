# Hosting the web app

`app.py` runs on **Streamlit Community Cloud** (free, HTTPS, deploys from
this GitHub repo). Only the API backend is available when hosted; the
Claude CLI backend is hidden automatically because `claude` is not
installed there.

## One-time setup

1. Go to <https://share.streamlit.io> and sign in with the GitHub account
   that owns `Makhashya/ema-unlimited-ocr`.
2. **Create app** -> *Deploy a public app from GitHub*:
   - Repository: `Makhashya/ema-unlimited-ocr`
   - Branch: `equipment-pipeline` (or `main` once merged)
   - Main file path: `web/streamlit_app.py`
   - App URL: pick a subdomain, e.g. `ema-equipment`
   - Advanced settings -> Python version: **3.12**
3. Advanced settings -> **Secrets**: paste the contents of
   `.streamlit/secrets.toml.example` with real values. At minimum:
   ```toml
   APP_PASSWORD = "a long password"
   VLM_PROVIDER = "openai"          # or anthropic / gemini
   OPENAI_API_KEY = "sk-..."        # the key for that provider
   ```
4. **Deploy**. First build takes a few minutes (installs
   `web/requirements.txt`). The app is then live at
   `https://<subdomain>.streamlit.app`.

## Access control

- `APP_PASSWORD` gates the whole page; share it only with the people who
  should upload photos. Without it the app is open to anyone with the URL.
- For tighter control, keep the repo private and in the app's
  **Settings -> Sharing** restrict viewers to specific email addresses;
  Community Cloud then requires those viewers to sign in.

## Updating

Push to the deployed branch; Community Cloud redeploys automatically.
Secrets are edited in the app's Settings -> Secrets (no redeploy needed;
the app picks them up on the next rerun).

## Limits to know

- Community Cloud gives ~1 GB RAM and sleeps the app after inactivity;
  the first visit afterwards takes ~30 s to wake.
- Uploads are capped at 500 MB (`.streamlit/config.toml`). Photos are held
  in a temp folder only while extraction runs and are deleted afterwards.
- Results live in the browser session: download the Excel before closing
  the tab. The persistent-jobs UI (`app_jobs.py` + `worker.py`) needs a
  host with a disk and background processes (a VM or container), not
  Community Cloud.

## Running the same thing locally

```
pip install -r web/requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in
streamlit run app.py
```
