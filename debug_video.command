#!/bin/zsh
# Double-click on macOS: choose a video; process locally; open annotated result.
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  echo "Python environment not found. See src/vision/README.md for setup."
  read -r "?Press Enter to close."
  exit 1
fi
"$PROJECT_DIR/.venv/bin/python" -m src.vision debug --choose --open
STATUS=$?
if [[ $STATUS -ne 0 ]]; then
  echo "Debug run failed. Read the error above."
  read -r "?Press Enter to close."
fi
exit $STATUS
