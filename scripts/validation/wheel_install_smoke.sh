#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_DEPS="${ROBOT_SF_WHEEL_INSTALL_SMOKE_DEPS:-loguru numba matplotlib}"
REPORT_FILE="output/validation/wheel_install_smoke_report.json"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WHEEL_INPUT="${1:-${WHEEL_GLOB:-}}"
WHEEL_PATH="$(python3 - "${REPO_ROOT}" "${WHEEL_INPUT}" <<'PY'
import glob
import os
import sys

repo_root, wheel_input = sys.argv[1:]
if wheel_input:
    wheel_pattern = wheel_input if os.path.isabs(wheel_input) else os.path.join(repo_root, wheel_input)
else:
    wheel_pattern = os.path.join(repo_root, "dist", "*.whl")

matches = glob.glob(wheel_pattern)
if matches:
    print(max(matches, key=os.path.getmtime))
elif os.path.isfile(wheel_pattern):
    print(wheel_pattern)
PY
)"

if [[ -z "${WHEEL_PATH}" ]]; then
  echo "No wheel file found to validate."
  echo "Set WHEEL_GLOB or pass a wheel path relative to the repository root: $0 <path-to-wheel>"
  exit 1
fi

if [[ ! -f "${WHEEL_PATH}" ]]; then
  echo "Wheel not found: ${WHEEL_PATH}"
  exit 1
fi

WORK_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

VENV_DIR="${WORK_DIR}/smoke-venv"
PIP_BIN="${VENV_DIR}/bin/pip"
PYTHON_BIN="${VENV_DIR}/bin/python"

python3 -m venv "${VENV_DIR}"

echo "Installing wheel (no dependency resolution) in clean temp venv: ${WHEEL_PATH}"
"${PIP_BIN}" install --no-cache-dir --no-deps "${WHEEL_PATH}"

if [[ -n "${BOOTSTRAP_DEPS}" ]]; then
  # shellcheck disable=SC2206
  bootstrap_deps=(${BOOTSTRAP_DEPS})
  echo "Installing bootstrap smoke deps: ${BOOTSTRAP_DEPS}"
  "${PIP_BIN}" install --no-cache-dir "${bootstrap_deps[@]}"
fi

mkdir -p "${REPO_ROOT}/output/validation"
python_check_output="$(cd /tmp && PYTHONPATH= PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -c "import robot_sf; print(robot_sf.__file__)")"

if [[ -z "${python_check_output}" ]]; then
  echo "Wheel smoke import validation produced no module file output."
  exit 1
fi

"${PYTHON_BIN}" - "${REPO_ROOT}/${REPORT_FILE}" "${WHEEL_PATH}" "${python_check_output}" "${BOOTSTRAP_DEPS}" <<'PY'
import json
import sys
from pathlib import Path

report_path, wheel, module_file, bootstrap_deps = sys.argv[1:]
Path(report_path).write_text(
    json.dumps(
        {
            "wheel": wheel,
            "status": "passed",
            "install_mode": "wheel_no_deps_with_bootstrap_import_deps",
            "module_file": module_file,
            "command": "import robot_sf from clean wheel install",
            "bootstrap_deps": bootstrap_deps,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "Wheel install smoke passed. Report: ${REPO_ROOT}/${REPORT_FILE}"
