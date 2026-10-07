@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_with_gmail.ps1"
pause
exit /b
:failed
echo Setup failed. Copy the error above for help. Python 3.12 is recommended.
pause
exit /b 1
