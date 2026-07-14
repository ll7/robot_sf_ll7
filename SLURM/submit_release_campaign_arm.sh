#!/usr/bin/env bash
# Per-arm sbatch template for the split release-0.0.3 h600/s30 execution packet
# (see docs/context/release_0_0_3_slurm_execution_packet.md for the full design
# rationale). One sbatch submission runs exactly ONE planner arm's already-generated
# child-campaign command from the plan packet -- it does not invent a new command.
#
# This template intentionally keeps the #SBATCH block minimal and lets the submitter
# pass --job-name, --output, --mem, --cpus-per-task, --time, and --partition on the
# `sbatch` command line so a single file serves all 14 arms with their own memory
# tier and log path. See the "Submission loop" section of the execution-packet doc
# for the exact per-arm sbatch invocations (heavy vs light memory table included).
#
# Required environment (pass via `sbatch --export=...`):
#   ARM_KEY                 planner key exactly as it appears in the split manifest
#                            and in the plan packet's arms[].planner_keys (e.g. "ppo").
# Optional environment:
#   PACKET                  path to the plan packet JSON (default: see below).
#   CHECKPOINT_STAGING_REPORT
#                            path to the checkpoint_staging.json written by
#                            scripts/benchmark/submit_camera_ready_checkpoint_gate.sh
#                            BEFORE this job was submitted. If set, the job refuses to
#                            run unless the report exists and reports submit_safe=true
#                            (issue #4613 contract: stage before sbatch, not inside it).
#   UV_PROJECT_ENVIRONMENT  optional uv-managed venv path override (LiCCA scratch venv).
#
#SBATCH --job-name=rsf-release-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=output/slurm/%j-release-arm.out

set -euo pipefail

# --- Resolve repo root the same way whether invoked via sbatch or manually. ---
PROJECT_ROOT="$(
  git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null \
    || echo "${SLURM_SUBMIT_DIR:-$(pwd)}"
)"
cd "${PROJECT_ROOT}"

ARM_KEY="${ARM_KEY:-}"
PACKET="${PACKET:-output/benchmarks/release_v0_0_3_split_plan/execution_packet.json}"
CHECKPOINT_STAGING_REPORT="${CHECKPOINT_STAGING_REPORT:-}"

if [[ -z "${ARM_KEY}" ]]; then
  echo "error: ARM_KEY is required (sbatch --export=ARM_KEY=<planner_key>,... ...)" >&2
  exit 2
fi
if [[ ! -f "${PACKET}" ]]; then
  echo "error: execution packet not found: ${PACKET}" >&2
  echo "       generate it first with scripts/tools/run_split_camera_ready_campaign.py plan" >&2
  exit 2
fi

echo "== [release-0.0.3] arm=${ARM_KEY} job=${SLURM_JOB_ID:-manual} node=${SLURMD_NODENAME:-local} =="
echo "   packet=${PACKET}"
date

# --- Pre-flight: refuse to run an unstaged learned-policy arm (issue #4613). ---
# The checkpoint gate must run BEFORE sbatch, on the submit/login node, with --stage
# (enforced_staged mode). This is a runtime guard, not the staging step itself.
if [[ -n "${CHECKPOINT_STAGING_REPORT}" ]]; then
  if [[ ! -f "${CHECKPOINT_STAGING_REPORT}" ]]; then
    echo "error: checkpoint staging report missing: ${CHECKPOINT_STAGING_REPORT}" >&2
    echo "       run scripts/benchmark/submit_camera_ready_checkpoint_gate.sh before sbatch." >&2
    exit 2
  fi
  SUBMIT_SAFE="$(python3 -c "
import json, sys
payload = json.load(open('${CHECKPOINT_STAGING_REPORT}'))
print(str(payload.get('submit_safe')).lower())
" 2>/dev/null || echo "false")"
  if [[ "${SUBMIT_SAFE}" != "true" ]]; then
    echo "error: checkpoint staging report reports submit_safe=${SUBMIT_SAFE}; do NOT run." >&2
    echo "       re-run scripts/benchmark/submit_camera_ready_checkpoint_gate.sh --stage." >&2
    exit 2
  fi
  echo "   checkpoint staging: submit_safe=true (${CHECKPOINT_STAGING_REPORT})"
fi

# --- Extract this arm's exact planned command + campaign root from the packet. ---
# The packet's "command" field is the literal argv the `plan` step generated; we do
# not reconstruct or guess it here, only look it up so one script covers all 14 arms.
# `mapfile` (not `read var1 var2`) is required here: the command line itself contains
# spaces, so splitting a single line across two variables would truncate it.
mapfile -t _ARM_LOOKUP < <(python3 -c "
import json, sys
packet = json.load(open('${PACKET}'))
for arm in packet['arms']:
    if list(arm['planner_keys']) == ['${ARM_KEY}']:
        print(arm['command'])
        print(arm['campaign_root'])
        break
else:
    sys.exit('arm not found: ${ARM_KEY}')
")
ARM_COMMAND="${_ARM_LOOKUP[0]:-}"
ARM_CAMPAIGN_ROOT="${_ARM_LOOKUP[1]:-}"

if [[ -z "${ARM_COMMAND:-}" ]]; then
  echo "error: could not resolve command for ARM_KEY=${ARM_KEY} from ${PACKET}" >&2
  exit 2
fi

echo "   campaign_root=${ARM_CAMPAIGN_ROOT}"
echo "   command=${ARM_COMMAND}"

# --- Threading guard (issue: macOS/torch segfault mitigation; harmless + correct
#     thread-limiting on Linux compute nodes too). Each of the campaign's `workers`
#     worker processes (ProcessPoolExecutor, see robot_sf/benchmark/map_runner.py)
#     independently imports torch/tensorflow for learned-policy arms; without these
#     guards each process can additionally spawn N BLAS/OpenMP threads and
#     oversubscribe the allocated CPUs on top of the process-level memory cost. ---
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Headless simulation rendering guards (matches SLURM/feature_extractor_comparison
# templates for this repo).
export DISPLAY=""
export MPLBACKEND="Agg"
export SDL_VIDEODRIVER="dummy"

# --- Module / environment activation. ---
# LiCCA (SLURM/Licca/README.md): purge interactive modules; prefer the project's
# uv-managed environment when present (SLURM/Licca/setup_uv_environment.sh stages
# uv onto scratch), otherwise fall back to the conda/micromamba `robot-sf` env.
if command -v module >/dev/null 2>&1; then
  module purge || true
fi
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  echo "   using UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  echo "   using repo-local .venv"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate robot-sf
  echo "   using conda env 'robot-sf'"
else
  echo "warning: no uv venv or conda env detected; relying on 'uv run' to resolve one" >&2
fi

# --- Per-arm log (in addition to the #SBATCH --output the submitter set). ---
LOG_DIR="output/slurm/logs"
mkdir -p "${LOG_DIR}"
ARM_LOG="${LOG_DIR}/${SLURM_JOB_ID:-manual}-${ARM_KEY}.log"

echo "   log=${ARM_LOG}"
echo "-- running --"

set +e
# shellcheck disable=SC2086
eval ${ARM_COMMAND} 2>&1 | tee "${ARM_LOG}"
STATUS="${PIPESTATUS[0]}"
set -e

echo "-- finished arm=${ARM_KEY} exit=${STATUS} at $(date) --"
# Exit codes mirror scripts/tools/run_camera_ready_benchmark.py:
#   0 benchmark-success, 2 unexpected failure/malformed, 3 accepted-unavailable-only.
exit "${STATUS}"
