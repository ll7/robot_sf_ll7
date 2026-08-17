#!/usr/bin/env bash
set -euo pipefail

# Remote named preflight for the issue-6642 launcher. This is intentionally
# separate from the packet builder: submit_and_record.sh runs it from the
# public worktree before sbatch, without private durable-results environment
# variables and without consulting or mutating the queue.

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "${PROJECT_ROOT}" ]]; then
  echo "[radius-sweep-preflight] unable to resolve the public worktree" >&2
  exit 2
fi

PYTHON_BIN=${CAMERA_READY_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[radius-sweep-preflight] Python executable is missing or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

OUTPUT_ROOT=${RADIUS_SWEEP_PREFLIGHT_OUTPUT_ROOT:-${PROJECT_ROOT}/output/issue_7198_radius_sweep_submission_preflight}
ARM_KEYS=(r0p5 r0p8 r1p0)
ARM_CONFIGS=(
  configs/benchmarks/issue_6642_radius_sweep_arm_0p5m.yaml
  configs/benchmarks/issue_6642_radius_sweep_arm_0p8m.yaml
  configs/benchmarks/issue_6642_radius_sweep_arm_1p0m.yaml
)

for index in "${!ARM_KEYS[@]}"; do
  arm_key=${ARM_KEYS[${index}]}
  arm_config=${ARM_CONFIGS[${index}]}
  campaign_id=issue7198-submission-${arm_key}
  arm_output_root=${OUTPUT_ROOT}/${arm_key}
  arm_config_path=${PROJECT_ROOT}/${arm_config}
  if [[ ! -f "${arm_config_path}" ]]; then
    echo "[radius-sweep-preflight] missing arm config: ${arm_config_path}" >&2
    exit 2
  fi

  "${PYTHON_BIN}" scripts/tools/run_camera_ready_benchmark.py \
    --config "${arm_config}" \
    --output-root "${arm_output_root}" \
    --campaign-id "${campaign_id}" \
    --skip-publication-bundle \
    --mode preflight \
    --checkpoint-preflight-mode enforced_staged \
    --log-level ERROR

  validate_path=${arm_output_root}/${campaign_id}/preflight/validate_config.json
  "${PYTHON_BIN}" - "${validate_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
checkpoint = payload.get("checkpoint_preflight", {})
if payload.get("episodes", 0) not in (None, 0):
    raise SystemExit(f"preflight emitted episodes: {payload.get('episodes')!r}")
if checkpoint.get("mode") != "enforced_staged":
    raise SystemExit(f"checkpoint mode is not enforced_staged: {checkpoint.get('mode')!r}")
if checkpoint.get("stage") is not True or checkpoint.get("submit_safe") is not True:
    raise SystemExit("checkpoint preflight is not staged and submit-safe")
PY

  episode_file=$(find "${arm_output_root}/${campaign_id}" -type f \( -name 'episodes*.jsonl' -o -name 'episodes*.parquet' \) -print -quit)
  if [[ -n "${episode_file}" ]]; then
    echo "[radius-sweep-preflight] production episode file emitted: ${episode_file}" >&2
    exit 2
  fi
done

echo "[radius-sweep-preflight] all three arms passed enforced-staged zero-episode admission"
