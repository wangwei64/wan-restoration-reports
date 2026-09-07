#!/usr/bin/env bash
# Select an existing environment automatically, then forward the prompt unchanged.
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${WAN_RESTORE_PYTHON:-}" ]]; then
  PYTHON_BIN="$WAN_RESTORE_PYTHON"
elif [[ -x "$PACKAGE_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$PACKAGE_DIR/.venv/bin/python"
elif [[ -x "$PACKAGE_DIR/../envs/wan-downscaler-full/bin/python" ]]; then
  PYTHON_BIN="$PACKAGE_DIR/../envs/wan-downscaler-full/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi
exec "$PYTHON_BIN" "$PACKAGE_DIR/run.py" "$@"
