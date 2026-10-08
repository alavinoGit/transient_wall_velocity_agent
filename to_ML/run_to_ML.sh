#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf '%s\n' 'Run this script from inside the repository.' >&2
  exit 2
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$PWD" != "$REPO_ROOT" || "$SCRIPT_DIR" != "$REPO_ROOT/to_ML" ]]; then
  printf '%s\n' 'Run this script from the repository root.' >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"
DEPS_DIR="$SCRIPT_DIR/.deps"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$SCRIPT_DIR/.mplconfig"
export PYTEST_ADDOPTS="--basetemp=$SCRIPT_DIR/.pytest-tmp -p no:cacheprovider ${PYTEST_ADDOPTS:-}"
mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/.python-cache"
export PYTHONPYCACHEPREFIX="$SCRIPT_DIR/.python-cache"
export PYTHONPATH="$DEPS_DIR${PYTHONPATH:+:$PYTHONPATH}"

cleanup() {
  rm -rf -- "$DEPS_DIR" "$SCRIPT_DIR/.python-cache" "$SCRIPT_DIR/.mplconfig" "$SCRIPT_DIR/.pytest-tmp"
}
trap cleanup EXIT

LOG_FILE="$SCRIPT_DIR/logs/postprocess.log"
exec > >(tee "$LOG_FILE") 2>&1

printf 'Postprocessing started: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if ! "$PYTHON" -c 'import numpy, pyarrow, yaml, pytest, matplotlib' >/dev/null 2>&1; then
  printf '%s\n' 'Installing package requirements into to_ML/.deps.'
  mkdir -p "$DEPS_DIR"
  "$PYTHON" -m pip install --disable-pip-version-check --no-cache-dir --upgrade --target "$DEPS_DIR" -r "$SCRIPT_DIR/requirements.txt"
fi

"$PYTHON" "$SCRIPT_DIR/scripts/build_ml_dataset.py" \
  --input "$REPO_ROOT/results/transient_velocity_profiles.csv" \
  --config "$SCRIPT_DIR/config/postprocess_config.yaml" \
  --output-root "$SCRIPT_DIR"

"$PYTHON" -m pytest -q "$SCRIPT_DIR/tests/test_build_ml_dataset.py"
printf '{"status":"PASS","command":"python -m pytest -q to_ML/tests/test_build_ml_dataset.py"}\n' > "$SCRIPT_DIR/validation/test_result.json"
"$PYTHON" "$SCRIPT_DIR/scripts/validate_ml_dataset.py" \
  --input "$REPO_ROOT/results/transient_velocity_profiles.csv" \
  --config "$SCRIPT_DIR/config/postprocess_config.yaml" \
  --package-root "$SCRIPT_DIR"

printf '\nGenerated package files:\n'
find "$SCRIPT_DIR/data" "$SCRIPT_DIR/validation" "$SCRIPT_DIR/logs" -maxdepth 1 -type f -printf '%p\n' | sort
printf '\nFINAL STATUS: PASS\n'
