#!/usr/bin/env bash
set -euo pipefail

# Preserve an explicit empty value so focused runs can exercise the base wheel
# without reinstalling the optional feature stacks. Unset keeps the CI-friendly
# default smoke matrix.
EXTRAS_SMOKE="${ROBOT_SF_WHEEL_INSTALL_SMOKE_EXTRAS-progress analytics viz}"
REPORT_FILE="${ROBOT_SF_WHEEL_INSTALL_SMOKE_REPORT:-output/validation/wheel_install_smoke_report.json}"

# The smoke is deliberately self-contained. CI sets these values too, but a
# maintainer running this script directly must get the same quiet, headless
# behavior without remembering workflow-only environment variables.
export SDL_VIDEODRIVER="dummy"
export MPLBACKEND="Agg"
export PYGAME_HIDE_SUPPORT_PROMPT="1"
export PYTHONNOUSERSITE="1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VALIDATION_DIR="${REPO_ROOT}/output/validation"
if [[ "${REPORT_FILE}" = /* ]]; then
  REPORT_PATH="${REPORT_FILE}"
else
  REPORT_PATH="${REPO_ROOT}/${REPORT_FILE}"
fi
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

mkdir -p "${VALIDATION_DIR}" "$(dirname "${REPORT_PATH}")"
WORK_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

VENV_DIR="${WORK_DIR}/smoke-venv"
PIP_BIN="${VENV_DIR}/bin/pip"
PYTHON_BIN="${VENV_DIR}/bin/python"

create_venv() {
  local target="$1"
  # uv is the repository's pinned environment tool and can seed pip into an
  # isolated environment even when the host's python3 is a relocatable build
  # without ensurepip. Keep a standard-library fallback for installations
  # that intentionally provide Python but not uv.
  if command -v uv >/dev/null 2>&1; then
    UV_NO_PROJECT=1 uv venv --seed --python "${ROBOT_SF_WHEEL_INSTALL_SMOKE_PYTHON:-python3}" "${target}" >&2
  else
    python3 -m venv "${target}" >&2
  fi
}

create_venv "${VENV_DIR}"

echo "Installing wheel with dependency resolution in clean temp venv: ${WHEEL_PATH}" >&2
"${PIP_BIN}" install --no-cache-dir "${WHEEL_PATH}" >&2

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
  echo "Wheel runtime smoke validation produced no output." >&2
  exit 1
fi

# Discover the entry-point roster from the installed distribution itself, then
# invoke every advertised command from an unrelated directory. This catches a
# package that works only because the source checkout happens to be importable.
console_scripts_path="${WORK_DIR}/console-scripts.json"
console_probes_path="${WORK_DIR}/console-script-probes.json"
PYTHONPATH= PYTHONNOUSERSITE=1 "${PYTHON_BIN}" - "${console_scripts_path}" <<'PY'
import importlib.metadata as metadata
import json
import sys
from pathlib import Path


def _normalise(name: str) -> str:
    """Normalise a distribution name according to Python package-name rules."""
    return name.replace("_", "-").replace(".", "-").lower()


output_path = Path(sys.argv[1])
distribution = next(
    (
        candidate
        for candidate in metadata.distributions()
        if _normalise(candidate.metadata.get("Name", "")) == "robot-sf"
    ),
    None,
)
if distribution is None:
    raise SystemExit("installed robot_sf distribution metadata was not found")

entries = sorted(
    (
        {"name": entry.name, "value": entry.value}
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    ),
    key=lambda entry: entry["name"],
)
if not entries:
    raise SystemExit("installed robot_sf distribution advertises no console scripts")

output_path.write_text(
    json.dumps(
        {
            "schema": "robot_sf_wheel_console_scripts.v1",
            "distribution": distribution.metadata.get("Name"),
            "version": distribution.version,
            "entries": entries,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

PYTHONPATH= PYTHONNOUSERSITE=1 "${PYTHON_BIN}" - "${console_scripts_path}" "${console_probes_path}" "${WORK_DIR}/probe-cwd" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path


roster_path, probes_path, probe_cwd = map(Path, sys.argv[1:])
probe_cwd.mkdir(parents=True, exist_ok=True)
roster = json.loads(roster_path.read_text(encoding="utf-8"))
probe_env = os.environ.copy()
probe_env.update(
    {
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
        "SDL_VIDEODRIVER": "dummy",
        "MPLBACKEND": "Agg",
        "PYGAME_HIDE_SUPPORT_PROMPT": "1",
    }
)
bin_dir = Path(sys.executable).parent
rows = []
for entry in roster["entries"]:
    name = str(entry["name"])
    command_path = bin_dir / name
    row = {
        "name": name,
        "value": str(entry["value"]),
        "command": [str(command_path), "--help"],
        "status": "failed",
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    if not command_path.is_file():
        row["error"] = "installed console-script executable is missing"
        rows.append(row)
        continue
    try:
        completed = subprocess.run(
            [str(command_path), "--help"],
            cwd=probe_cwd,
            env=probe_env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        row["error"] = "--help probe timed out after 60 seconds"
        row["stdout"] = (exc.stdout or "")[-4000:]
        row["stderr"] = (exc.stderr or "")[-4000:]
    except OSError as exc:
        row["error"] = f"could not execute installed console script: {exc}"
    else:
        row["returncode"] = completed.returncode
        row["stdout"] = completed.stdout[-4000:]
        row["stderr"] = completed.stderr[-4000:]
        if completed.returncode == 0:
            row["status"] = "passed"
    rows.append(row)

payload = {
    "schema": "robot_sf_wheel_console_script_probes.v1",
    "probe_cwd": str(probe_cwd),
    "entries": rows,
    "passed": sum(row["status"] == "passed" for row in rows),
    "failed": sum(row["status"] != "passed" for row in rows),
}
payload["ok"] = payload["failed"] == 0 and payload["passed"] == len(rows)
probes_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

extras_status_json="[]"
if [[ -n "${EXTRAS_SMOKE}" ]]; then
  extras_status_file="${WORK_DIR}/extras-status.jsonl"
  # shellcheck disable=SC2206
  extras=(${EXTRAS_SMOKE})
  for extra in "${extras[@]}"; do
    extra_venv="${WORK_DIR}/extra-${extra}-venv"
    extra_pip="${extra_venv}/bin/pip"
    extra_python="${extra_venv}/bin/python"
    install_log="${VALIDATION_DIR}/wheel-extra-${extra}-install.log"
    probe_log="${VALIDATION_DIR}/wheel-extra-${extra}-probe.log"
    create_venv "${extra_venv}"
    echo "Installing optional extra independently: ${extra}" >&2
    if ! "${extra_pip}" install --no-cache-dir "${WHEEL_PATH}[${extra}]" \
      >"${install_log}" 2>&1; then
      echo "Optional extra installation failed: ${extra}" >&2
      tail -n 200 "${install_log}" >&2
      "${PYTHON_BIN}" - "${REPORT_PATH}" "${WHEEL_PATH}" "${extra}" \
        "${install_log}" <<'PY'
import json
import sys
from pathlib import Path

report_path, wheel, extra, install_log = sys.argv[1:]
log_path = Path(install_log)
log_tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
Path(report_path).write_text(
    json.dumps(
        {
            "wheel": wheel,
            "status": "failed",
            "stage": "extra_install",
            "extra": extra,
            "install_log": str(log_path),
            "install_log_tail": log_tail,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
      exit 1
    fi
    if ! (
      cd /tmp
      PYTHONPATH= PYTHONNOUSERSITE=1 "${extra_python}" - "${extra}" >>"${extras_status_file}" <<'PY'
import importlib
import json
import sys

import robot_sf

extra = sys.argv[1]
feature_modules = {
    "progress": ["tqdm"],
    "analytics": ["duckdb", "pyarrow"],
    "viz": [
        "pygame",
        "matplotlib",
        "PIL",
        "moviepy",
        "seaborn",
        "robot_sf.render.sim_view",
    ],
    "maps": [
        "osmnx",
        "geopandas",
        "pyproj",
        "robot_sf.nav.osm_map_builder",
    ],
    "benchmark": [
        "pandas",
        "scipy",
        "robot_sf.benchmark.aggregate",
    ],
    "training": [
        "stable_baselines3",
        "torch",
        "sklearn",
        "optuna",
        "tensorboard",
        "wandb",
        "optuna_dashboard",
    ],
    # The all-extra install is itself the dependency-resolution assertion. Probe
    # representative modules from every new feature group so a self-referential
    # meta-extra cannot pass while silently omitting one of the split stacks.
    "all": [
        "pygame",
        "matplotlib",
        "osmnx",
        "geopandas",
        "pandas",
        "scipy",
        "stable_baselines3",
        "torch",
        "robot_sf.render.sim_view",
        "robot_sf.nav.osm_map_builder",
        "robot_sf.benchmark.aggregate",
    ],
}
modules = feature_modules.get(extra, [])
for module_name in modules:
    importlib.import_module(module_name)

print(
    json.dumps(
        {
            "extra": extra,
            "status": "passed",
            "module_file": robot_sf.__file__,
            "feature_modules": modules,
        }
    )
)
PY
    ) 2>"${probe_log}"; then
      echo "Optional extra feature probe failed: ${extra}" >&2
      cat "${probe_log}" >&2
      exit 1
    fi
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

"${PYTHON_BIN}" - "${REPORT_PATH}" "${WHEEL_PATH}" "${REPO_ROOT}" \
  "${python_check_output}" "${EXTRAS_SMOKE}" "${extras_status_json}" \
  "${console_scripts_path}" "${console_probes_path}" <<'PY'
import json
import sys
from pathlib import Path

(
    report_path,
    wheel,
    repo_root,
    runtime_smoke_json,
    extras_smoke,
    extras_status_json,
    console_scripts_path,
    console_probes_path,
) = sys.argv[1:]
runtime_smoke = json.loads(runtime_smoke_json)
console_scripts = json.loads(Path(console_scripts_path).read_text(encoding="utf-8"))
console_probes = json.loads(Path(console_probes_path).read_text(encoding="utf-8"))
repo_path = Path(repo_root).resolve()
module_file = Path(runtime_smoke["module_file"]).resolve()
source_checkout = module_file == repo_path or repo_path in module_file.parents
report = {
    "wheel": wheel,
    "status": "passed" if console_probes.get("ok") and not source_checkout else "failed",
    "install_mode": "wheel_with_dependency_resolution",
    "module_file": str(module_file),
    "source_checkout_import": source_checkout,
    "command": "import robot_sf; make_crowd_sim_env().reset(); env.step()",
    "runtime_smoke": runtime_smoke,
    "extras_smoke": extras_smoke.split(),
    "extras": json.loads(extras_status_json),
    "console_scripts": console_scripts["entries"],
    "console_script_probes": console_probes["entries"],
    "console_scripts_passed": console_probes["passed"],
    "console_scripts_failed": console_probes["failed"] + int(source_checkout),
}
Path(report_path).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, indent=2, sort_keys=True))
if report["status"] != "passed":
    raise SystemExit(1)
PY

echo "Wheel install smoke report: ${REPORT_PATH}" >&2
