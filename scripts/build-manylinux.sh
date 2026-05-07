#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CIBW_BUILD="${CIBW_BUILD:-cp39-*}"
export CIBW_CONTAINER_ENGINE="${CIBW_CONTAINER_ENGINE:-podman}"

PYTHON_BIN="${PYTHON:-python}"
if ! "$PYTHON_BIN" -c "import cibuildwheel" >/dev/null 2>&1 && [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

"$PYTHON_BIN" -m cibuildwheel --platform linux --output-dir wheelhouse
