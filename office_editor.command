#!/bin/zsh
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  echo "Python environment missing. See src/vision/README.md for setup."
  read -r "?Press Enter to close."
  exit 1
fi
"$PROJECT_DIR/.venv/bin/python" -m src.app --open
STATUS=$?
if [[ $STATUS -ne 0 ]]; then
  echo "Could not start editor. If port 8765 is busy, open http://127.0.0.1:8765 in your browser."
  read -r "?Press Enter to close."
fi
exit $STATUS
