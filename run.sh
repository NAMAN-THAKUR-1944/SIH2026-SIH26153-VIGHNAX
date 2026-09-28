#!/usr/bin/env sh
# VIGHNAX one-click launcher (Linux / macOS): creates .venv on the first run, installs the requirements,
# then starts the dashboard with that environment's Python and opens it in the browser.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

if [ ! -x "$PY" ]; then
    echo "[VIGHNAX] First run: creating a Python environment in .venv ..."
    python3 -m venv .venv || { echo "[VIGHNAX] Could not create .venv - install Python 3.10 or newer (with venv) and run this again."; exit 1; }
fi

# (Re)install when requirements.txt differs from the copy saved after the last successful install.
if ! cmp -s requirements.txt .venv/vighnax-requirements.txt; then
    echo "[VIGHNAX] Installing the requirements - the first time this takes a few minutes ..."
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    "$PY" -m pip install -r requirements.txt
    cp requirements.txt .venv/vighnax-requirements.txt
fi

exec "$PY" server.py --open "$@"
