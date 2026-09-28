@echo off
rem VIGHNAX one-click launcher (Windows): creates .venv on the first run, installs the requirements,
rem then starts the dashboard with that environment's Python and opens it in the browser.
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [VIGHNAX] First run: creating a Python environment in .venv ...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if not exist "%PY%" (
        echo [VIGHNAX] Could not create .venv. Install Python 3.10 or newer from python.org and run this again.
        pause
        exit /b 1
    )
)

rem (Re)install when requirements.txt differs from the copy saved after the last successful install.
fc /b requirements.txt .venv\vighnax-requirements.txt >nul 2>&1
if errorlevel 1 (
    echo [VIGHNAX] Installing the requirements - the first time this takes a few minutes ...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [VIGHNAX] Installing the requirements failed - see the messages above.
        pause
        exit /b 1
    )
    copy /y requirements.txt .venv\vighnax-requirements.txt >nul
)

"%PY%" server.py --open %*
pause
