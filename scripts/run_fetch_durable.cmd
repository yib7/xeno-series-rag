@echo off
REM Wrapper invoked by the XenoRagHtmlFetch scheduled task every 2 min.
REM run_fetch_durable.py self-locks (PID lockfile), so overlapping triggers no-op cleanly.
cd /d "C:\Users\creep\Downloads\Claude_Work_Zone\xeno_series_rag"
".venv\Scripts\python.exe" -m scripts.run_fetch_durable >> "data\raw\durable.log" 2>&1
