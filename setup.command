#!/bin/zsh
set -e
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="$(command -v python3.13 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Install Python 3.13, then run setup.command again."
  read -r "?Press Enter to close."
  exit 1
fi
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$PROJECT_DIR/.venv"
fi
"$PROJECT_DIR/.venv/bin/python" -m pip install -r src/vision/requirements-lock.txt
"$PROJECT_DIR/.venv/bin/python" -m src.demo_bundle
"$PROJECT_DIR/.venv/bin/python" -m src.vision prepare
echo "Setup complete. Starting CV Office."
"$PROJECT_DIR/.venv/bin/python" -m src.app --open
