@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Open http://127.0.0.1:8000 in your browser.
echo Create an account, then log in. Keep this window open.
".venv\Scripts\python.exe" run.py
pause
exit /b
:failed
echo.
echo Setup failed. Copy the error above and share it for help.
echo This app requires a supported Python version; Python 3.12 is recommended.
pause
exit /b 1
