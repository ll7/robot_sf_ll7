#!/usr/bin/env bash
#SBATCH --job-name=issue1475-orca-bc
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-issue1475-orca-residual-bc.out
#SBATCH --error=output/slurm/%j-issue1475-orca-residual-bc.err

set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
RESULTS_ROOT=${ISSUE1475_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/issue1475-orca-residual-bc-job-${SLURM_JOB_ID:-local}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/issue1475-orca-residual-bc-${SLURM_JOB_ID:-local}}
LINEAGE_CONFIG=${ISSUE1475_LINEAGE_CONFIG:-configs/training/orca_residual/orca_residual_bc_issue_1428.yaml}
BC_CONFIG=${ISSUE1475_BC_CONFIG:-configs/training/orca_residual/orca_residual_bc_issue_1475_smoke_pretrain.yaml}
TRAINING_ENV_CONFIG=${ISSUE1475_TRAINING_ENV_CONFIG:-configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity.yaml}
SCENARIO_CONFIG=${ISSUE1475_SCENARIO_CONFIG:-configs/scenarios/single/planner_sanity_simple.yaml}
SOURCE_POLICY_ID=${ISSUE1475_SOURCE_POLICY_ID:-ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417}
DATASET_ID=${ISSUE1475_DATASET_ID:-issue_1428_orca_residual_bc_progress_v1_smoke}
POLICY_OUTPUT_ID=${ISSUE1475_POLICY_OUTPUT_ID:-issue_1428_orca_residual_bc_progress_v1_policy_smoke}
EPISODES=${ISSUE1475_EPISODES:-3}
SEEDS=${ISSUE1475_SEEDS:-"111:112:113"}
CANDIDATE=${ISSUE1475_CANDIDATE:-orca_residual_guarded_ppo_progress_v1}
RUN_NOMINAL=${ISSUE1475_RUN_NOMINAL:-0}
LOG_LEVEL=${ISSUE1475_LOG_LEVEL:-WARNING}
RUN_OUTPUT_DIR=""

log() {
  echo "[issue1475-orca-bc] $*"
}

die() {
  echo "[issue1475-orca-bc] $*" >&2
  exit 1
}

cleanup() {
  if [[ -n "${RUN_OUTPUT_DIR}" && -d "${RUN_OUTPUT_DIR}" ]]; then
    mkdir -p "${RESULTS_ROOT}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --partial --prune-empty-dirs "${RUN_OUTPUT_DIR}/" "${RESULTS_ROOT}/" || true
    else
      cp -r "${RUN_OUTPUT_DIR}/." "${RESULTS_ROOT}/" || true
    fi
  fi
}
trap cleanup EXIT

run_in_allocation() {
  if [[ "${ISSUE1475_USE_SRUN:-0}" == "1" ]] && command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    log "Launching with srun on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}."
    srun --cpu_bind=cores --gpus-per-node=1 "$@"
  else
    log "Running directly inside the batch allocation; set ISSUE1475_USE_SRUN=1 to opt into srun."
    "$@"
  fi
}

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  die "PROJECT_ROOT is not inside a git worktree: ${PROJECT_ROOT}"
fi

cd "${PROJECT_ROOT}"

for path in "${LINEAGE_CONFIG}" "${BC_CONFIG}" "${TRAINING_ENV_CONFIG}" "${SCENARIO_CONFIG}"; do
  [[ -f "${path}" ]] || die "Required path not found: ${path}"
done

mkdir -p "${WORKDIR}" "${LOCAL_OUTPUT_ROOT}"
export ROBOT_SF_ARTIFACT_ROOT=${WORKDIR}/results
export UV_PROJECT_ENVIRONMENT=${ISSUE1475_UV_ENV_DIR:-${WORKDIR}/uv_env}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export MPLBACKEND=${MPLBACKEND:-Agg}
export LOGURU_LEVEL=${LOGURU_LEVEL:-${LOG_LEVEL}}
RUN_OUTPUT_DIR=${ROBOT_SF_ARTIFACT_ROOT}
mkdir -p "${ROBOT_SF_ARTIFACT_ROOT}" "$(dirname "${UV_PROJECT_ENVIRONMENT}")"

log "PROJECT_ROOT=${PROJECT_ROOT}"
log "ROBOT_SF_ARTIFACT_ROOT=${ROBOT_SF_ARTIFACT_ROOT}"
log "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
log "RESULTS_ROOT=${RESULTS_ROOT}"
log "LINEAGE_CONFIG=${LINEAGE_CONFIG}"
log "BC_CONFIG=${BC_CONFIG}"
log "TRAINING_ENV_CONFIG=${TRAINING_ENV_CONFIG}"
log "SCENARIO_CONFIG=${SCENARIO_CONFIG}"
log "SOURCE_POLICY_ID=${SOURCE_POLICY_ID}"
log "DATASET_ID=${DATASET_ID}"
log "POLICY_OUTPUT_ID=${POLICY_OUTPUT_ID}"
log "EPISODES=${EPISODES}"
log "SEEDS=${SEEDS}"
log "RUN_NOMINAL=${RUN_NOMINAL}"
log "git_commit=$(git rev-parse HEAD)"

uv sync --group imitation

run_in_allocation \
  uv run python scripts/validation/validate_orca_residual_lineage_packet.py \
  --config "${LINEAGE_CONFIG}" \
  --json

SEEDS_EXPANDED="${SEEDS//,/ }"
SEEDS_EXPANDED="${SEEDS_EXPANDED//:/ }"
# shellcheck disable=SC2206
SEED_ARGS=(${SEEDS_EXPANDED})

run_in_allocation \
  uv run python scripts/training/collect_expert_trajectories.py \
  --dataset-id "${DATASET_ID}" \
  --policy-id "${SOURCE_POLICY_ID}" \
  --episodes "${EPISODES}" \
  --scenario-config "${SCENARIO_CONFIG}" \
  --training-config "${TRAINING_ENV_CONFIG}" \
  --env-config "${BC_CONFIG}" \
  --seeds "${SEED_ARGS[@]}"

run_in_allocation \
  uv run --group imitation python scripts/training/pretrain_from_expert.py \
  --config "${BC_CONFIG}"

MATERIALIZED_DIR="${ROBOT_SF_ARTIFACT_ROOT}/issue1475_materialized_candidate"
MATERIALIZED_JSON="${MATERIALIZED_DIR}/materialized_candidate_manifest.json"
POLICY_MODEL_PATH="${ROBOT_SF_ARTIFACT_ROOT}/benchmarks/expert_policies/${POLICY_OUTPUT_ID}.zip"
run_in_allocation \
  uv run python scripts/tools/materialize_orca_residual_candidate.py \
  --policy-model-id "${POLICY_OUTPUT_ID}" \
  --policy-model-path "${POLICY_MODEL_PATH}" \
  --candidate "${CANDIDATE}" \
  --output-dir "${MATERIALIZED_DIR}" \
  --json

RUNTIME_REGISTRY=$(python - "${MATERIALIZED_JSON}" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["runtime_registry"])
PY
)

SMOKE_OUTPUT="${ROBOT_SF_ARTIFACT_ROOT}/policy_search/${CANDIDATE}/smoke/issue1475_smoke"
run_in_allocation \
  uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate "${CANDIDATE}" \
  --candidate-registry "${RUNTIME_REGISTRY}" \
  --stage smoke \
  --workers 1 \
  --output-dir "${SMOKE_OUTPUT}"

python - "${SMOKE_OUTPUT}/summary.json" <<'PY'
import json
import sys
summary = json.loads(open(sys.argv[1], encoding="utf-8").read())["summary"]
success = float(summary.get("success_rate", 0.0))
collision = float(summary.get("collision_rate", 1.0))
if success < 1.0 or collision > 0.0:
    raise SystemExit(
        f"Smoke gate failed: success_rate={success:.4f} collision_rate={collision:.4f}"
    )
print(f"Smoke gate passed: success_rate={success:.4f} collision_rate={collision:.4f}")
PY

if [[ "${RUN_NOMINAL}" == "1" ]]; then
  NOMINAL_OUTPUT="${ROBOT_SF_ARTIFACT_ROOT}/policy_search/${CANDIDATE}/nominal_sanity/issue1475_nominal"
  run_in_allocation \
    uv run python scripts/validation/run_policy_search_candidate.py \
    --candidate "${CANDIDATE}" \
    --candidate-registry "${RUNTIME_REGISTRY}" \
    --stage nominal_sanity \
    --allow-expensive-stage \
    --workers 2 \
    --output-dir "${NOMINAL_OUTPUT}"
else
  log "Skipping nominal_sanity because ISSUE1475_RUN_NOMINAL is not 1."
fi

log "ORCA-residual BC smoke chain completed. Artifacts will sync to ${RESULTS_ROOT}"
