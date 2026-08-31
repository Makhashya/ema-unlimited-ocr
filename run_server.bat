@echo off
rem Serve the web UI from this PC on a public URL (see serve_public.py).
rem   run_server.bat               -> app.py
rem   run_server.bat app_jobs.py   -> the persistent-jobs app
cd /d "%~dp0"
python serve_public.py %*
echo.
pause
