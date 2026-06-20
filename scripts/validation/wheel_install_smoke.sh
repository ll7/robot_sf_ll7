#!/usr/bin/env bash
set -euo pipefail

EXTRAS_SMOKE="${ROBOT_SF_WHEEL_INSTALL_SMOKE_EXTRAS:-progress analysis analytics viz}"
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

echo "Installing wheel with dependency resolution in clean temp venv: ${WHEEL_PATH}"
"${PIP_BIN}" install --no-cache-dir "${WHEEL_PATH}"

mkdir -p "${REPO_ROOT}/output/validation"
python_check_output="$(
  cd /tmp && PYTHONPATH= PYTHONNOUSERSITE=1 "${PYTHON_BIN}" <<'PY'
import json

import numpy as np

import robot_sf
from robot_sf.gym_env.environment_factory import make_crowd_sim_env

env = make_crowd_sim_env(seed=123)
try:
    obs, info = env.reset(seed=123)
    next_obs, reward, terminated, truncated, next_info = env.step()
finally:
    env.close()

print(
    json.dumps(
        {
            "module_file": robot_sf.__file__,
            "env_factory": "make_crowd_sim_env",
            "reset_positions_shape": list(np.asarray(obs["positions"]).shape),
            "step_positions_shape": list(np.asarray(next_obs["positions"]).shape),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "map_id": info.get("map_id"),
            "step_map_id": next_info.get("map_id"),
        }
    )
)
PY
)"

if [[ -z "${python_check_output}" ]]; then
  echo "Wheel runtime smoke validation produced no output."
  exit 1
fi

extras_status_json="[]"
if [[ -n "${EXTRAS_SMOKE}" ]]; then
  extras_status_file="${WORK_DIR}/extras-status.jsonl"
  # shellcheck disable=SC2206
  extras=(${EXTRAS_SMOKE})
  for extra in "${extras[@]}"; do
    extra_venv="${WORK_DIR}/extra-${extra}-venv"
    extra_pip="${extra_venv}/bin/pip"
    extra_python="${extra_venv}/bin/python"
    python3 -m venv "${extra_venv}"
    echo "Installing optional extra independently: ${extra}"
    "${extra_pip}" install --no-cache-dir "${WHEEL_PATH}[${extra}]"
    cd /tmp && PYTHONPATH= PYTHONNOUSERSITE=1 "${extra_python}" - "${extra}" >>"${extras_status_file}" <<'PY'
import json
import sys

import robot_sf

extra = sys.argv[1]
print(json.dumps({"extra": extra, "status": "passed", "module_file": robot_sf.__file__}))
PY
  done
  extras_status_json="$("${PYTHON_BIN}" - "${extras_status_file}" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
if not status_path.exists():
    print("[]")
else:
    print(json.dumps([json.loads(line) for line in status_path.read_text().splitlines() if line]))
PY
)"
fi

"${PYTHON_BIN}" - "${REPO_ROOT}/${REPORT_FILE}" "${WHEEL_PATH}" "${python_check_output}" "${EXTRAS_SMOKE}" "${extras_status_json}" <<'PY'
import json
import sys
from pathlib import Path

report_path, wheel, runtime_smoke_json, extras_smoke, extras_status_json = sys.argv[1:]
runtime_smoke = json.loads(runtime_smoke_json)
Path(report_path).write_text(
    json.dumps(
        {
            "wheel": wheel,
            "status": "passed",
            "install_mode": "wheel_with_dependency_resolution",
            "module_file": runtime_smoke["module_file"],
            "command": "import robot_sf; make_crowd_sim_env().reset(); env.step()",
            "runtime_smoke": runtime_smoke,
            "extras_smoke": extras_smoke.split(),
            "extras": json.loads(extras_status_json),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "Wheel install smoke passed. Report: ${REPO_ROOT}/${REPORT_FILE}"
