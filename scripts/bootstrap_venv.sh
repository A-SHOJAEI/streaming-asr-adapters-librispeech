#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python3 not found" >&2
  exit 1
fi

# Create venv without pip (assume ensurepip may be missing)
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv --without-pip "$VENV_DIR"
fi

# Bootstrap pip inside the venv if missing
if [[ ! -x "$VENV_DIR/bin/pip" ]]; then
  GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"
  TMP_GETPIP="${VENV_DIR}/get-pip.py"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$GET_PIP_URL" -o "$TMP_GETPIP"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$TMP_GETPIP" "$GET_PIP_URL"
  else
    "$PYTHON_BIN" - <<'PY'
import urllib.request
url = 'https://bootstrap.pypa.io/get-pip.py'
out = '.venv/get-pip.py'
urllib.request.urlretrieve(url, out)
print('Downloaded', url, '->', out)
PY
  fi

  "$VENV_DIR/bin/python" "$TMP_GETPIP" --no-warn-script-location
fi

# Install deps (only via venv pip; never touch system interpreter)
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install -r requirements.txt
