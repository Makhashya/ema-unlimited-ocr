@echo off
rem Serve the web UI from this PC on a public URL (see serve_public.py).
rem   run_server.bat            -> app_jobs.py (persistent jobs)
rem   run_server.bat app.py     -> the single-session app
cd /d "%~dp0"
python serve_public.py %*
echo.
pause
