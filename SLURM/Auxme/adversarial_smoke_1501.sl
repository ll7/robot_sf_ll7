#!/usr/bin/env bash
#SBATCH --job-name=adv1501-smoke
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-adversarial-smoke1501.out

# Run the bounded Issue #1501 crossing/TTC adversarial smoke.
#
# This launcher intentionally executes only the #1571-sharpened first child:
# crossing_ttc with random + optuna_tpe over the frozen goal/orca planner rows.
# The guided_route_search row is recorded as not_available for this family.
set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
OUTPUT_ROOT=${ADVERSARIAL_SMOKE_OUTPUT_ROOT:-${PROJECT_ROOT}/output/adversarial/issue_1501}
LABEL=${ADVERSARIAL_SMOKE_LABEL:-issue1501-crossing-ttc-${SLURM_JOB_ID:-local}}
RESULTS_ROOT=${ADVERSARIAL_SMOKE_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/adversarial-smoke1501-job-${SLURM_JOB_ID:-local}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/adversarial-smoke1501-${SLURM_JOB_ID:-local}}
PYTHON_BIN=${PROJECT_ROOT}/.venv/bin/python
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
SMOKE_BUDGET=${ADVERSARIAL_SMOKE_BUDGET:-32}
SMOKE_SEED=${ADVERSARIAL_SMOKE_SEED:-42}
SMOKE_POLICIES=${ADVERSARIAL_SMOKE_POLICIES:-"goal orca"}
SMOKE_SYNTHETIC=${ADVERSARIAL_SMOKE_SYNTHETIC:-false}
MODULES_AVAILABLE=0
CAMPAIGN_ROOT=""

log() {
  echo "[adv1501] $*"
}

die() {
  echo "[adv1501] $*" >&2
  exit 1
}

ensure_module_command() {
  if command -v module >/dev/null 2>&1; then
    MODULES_AVAILABLE=1
    return 0
  fi

  local init_script
  for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    if [[ -f "${init_script}" ]]; then
      # shellcheck disable=SC1090
      source "${init_script}" >/dev/null 2>&1 || true
    fi
    if command -v module >/dev/null 2>&1; then
      MODULES_AVAILABLE=1
      return 0
    fi
  done

  echo "[adv1501] module command unavailable; continuing without module loads." >&2
  return 0
}

run_in_allocation() {
  if [[ "${ADVERSARIAL_SMOKE_USE_SRUN:-0}" == "1" ]] && command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]] && srun --version >/dev/null 2>&1; then
    log "Launching with srun on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}."
    srun --cpu_bind=cores "$@"
  else
    echo "[adv1501] running directly inside the batch allocation; set ADVERSARIAL_SMOKE_USE_SRUN=1 to opt into srun." >&2
    "$@"
  fi
}

cleanup() {
  if [[ -n "${CAMPAIGN_ROOT}" && -d "${CAMPAIGN_ROOT}" ]]; then
    mkdir -p "${RESULTS_ROOT}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --partial --prune-empty-dirs "${CAMPAIGN_ROOT}/" "${RESULTS_ROOT}/" \
        || echo "[adv1501] Warning: rsync failed to sync artifacts to ${RESULTS_ROOT}" >&2
    else
      cp -r "${CAMPAIGN_ROOT}/." "${RESULTS_ROOT}/" \
        || echo "[adv1501] Warning: cp failed to sync artifacts to ${RESULTS_ROOT}" >&2
    fi
  fi
}
trap cleanup EXIT

case "${SMOKE_SYNTHETIC,,}" in
  1|true|yes|on) SMOKE_SYNTHETIC=true ;;
  0|false|no|off|"") SMOKE_SYNTHETIC=false ;;
  *) die "ADVERSARIAL_SMOKE_SYNTHETIC must be true/false-like, got: ${SMOKE_SYNTHETIC}" ;;
esac

if [[ ! "${SMOKE_BUDGET}" =~ ^[0-9]+$ || "${SMOKE_BUDGET}" -lt 1 ]]; then
  die "ADVERSARIAL_SMOKE_BUDGET must be a positive integer, got: ${SMOKE_BUDGET}"
fi

if [[ ! "${SMOKE_SEED}" =~ ^[0-9]+$ ]]; then
  die "ADVERSARIAL_SMOKE_SEED must be a non-negative integer, got: ${SMOKE_SEED}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  die "Expected repo virtualenv python at ${PYTHON_BIN}"
fi

for required in \
  configs/adversarial/issue_1500_adversarial_comparison_manifest.v1.yaml \
  configs/adversarial/crossing_ttc_space.yaml \
  configs/scenarios/templates/crossing_ttc.yaml \
  scripts/tools/compare_adversarial_samplers.py \
  scripts/tools/curate_adversarial_failure_archive.py
do
  [[ -f "${PROJECT_ROOT}/${required}" ]] || die "Required file missing: ${required}"
done

ensure_module_command

if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    log "Loading module ${mod}"
    module load "${mod}"
  done
  if [[ -n "${CUDA_MODULE}" ]]; then
    log "Loading CUDA module ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
  fi
fi

mkdir -p "${WORKDIR}" "${LOCAL_OUTPUT_ROOT}" "${OUTPUT_ROOT}"
CAMPAIGN_ROOT=${OUTPUT_ROOT}/${LABEL}
mkdir -p "${CAMPAIGN_ROOT}/crossing_ttc"

cd "${PROJECT_ROOT}"

log "Commit: $(git rev-parse HEAD)"
log "Campaign root: ${CAMPAIGN_ROOT}"
log "Results sync root: ${RESULTS_ROOT}"
log "Budget per sampler: ${SMOKE_BUDGET}"
log "Seed: ${SMOKE_SEED}"
log "Policies: ${SMOKE_POLICIES}"
log "Synthetic evaluator: ${SMOKE_SYNTHETIC}"

synthetic_arg=()
if [[ "${SMOKE_SYNTHETIC}" == "true" ]]; then
  synthetic_arg=(--synthetic)
fi

manifest_args=()
for policy in ${SMOKE_POLICIES}; do
  case "${policy}" in
    goal|orca) ;;
    *) die "Unsupported Issue #1501 policy row '${policy}'. Expected goal or orca." ;;
  esac
  policy_root=${CAMPAIGN_ROOT}/crossing_ttc/${policy}
  mkdir -p "${policy_root}"
  log "Running crossing_ttc policy=${policy} samplers=random,optuna"
  run_in_allocation \
    "${PYTHON_BIN}" scripts/tools/compare_adversarial_samplers.py \
      --scenario-template configs/scenarios/templates/crossing_ttc.yaml \
      --search-space configs/adversarial/crossing_ttc_space.yaml \
      --policy "${policy}" \
      --objective worst_case_snqi \
      --output-dir "${policy_root}" \
      --budget "${SMOKE_BUDGET}" \
      --seed "${SMOKE_SEED}" \
      --sampler random \
      --sampler optuna \
      --out-json "${policy_root}/sampler_comparison.json" \
      "${synthetic_arg[@]}"
  manifest_args+=("${policy_root}/random/manifest.json" "${policy_root}/optuna/manifest.json")
done

log "Curating crossing_ttc adversarial failure archive"
run_in_allocation \
  "${PYTHON_BIN}" scripts/tools/curate_adversarial_failure_archive.py \
    "${manifest_args[@]}" \
    --out "${CAMPAIGN_ROOT}/crossing_ttc/archive.json"

log "Writing row-status summary"
GIT_COMMIT=$(git rev-parse HEAD)
"${PYTHON_BIN}" - "${CAMPAIGN_ROOT}" "${SMOKE_BUDGET}" "${SMOKE_SEED}" "${SMOKE_SYNTHETIC}" "${SMOKE_POLICIES}" "${GIT_COMMIT}" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

campaign_root = Path(sys.argv[1])
budget = int(sys.argv[2])
seed = int(sys.argv[3])
synthetic = sys.argv[4].lower() == "true"
policies = sys.argv[5].split()
git_commit = sys.argv[6]

executed_counts: Counter[str] = Counter()
manifest_paths: list[str] = []
rows: list[dict[str, object]] = []
row_status_keys = [
    "valid_non_failure",
    "valid_failure",
    "invalid_candidate",
    "simulation_error",
    "fallback",
    "degraded",
    "not_available",
]

def classify(candidate: dict[str, object]) -> str:
    attribution = candidate.get("failure_attribution")
    if isinstance(attribution, dict):
        primary = str(attribution.get("primary_failure") or "")
        status = str(attribution.get("status") or "")
        if primary == "invalid_candidate" or status == "not_evaluated":
            return "invalid_candidate"
        if primary in {"fallback", "degraded"}:
            return primary
        if primary == "success":
            return "valid_non_failure"
        if primary in {"collision", "near_miss", "timeout", "comfort_violation", "incomplete"}:
            return "valid_failure"
        if primary in {"evaluation_error", "simulation_error"}:
            return "simulation_error"

    certification = candidate.get("certification_status")
    if isinstance(certification, dict) and str(certification.get("status") or "") != "passed":
        return "invalid_candidate"

    if candidate.get("error"):
        return "simulation_error"

    return "simulation_error"

for policy in policies:
    for sampler in ("random", "optuna"):
        manifest_path = campaign_root / "crossing_ttc" / policy / sampler / "manifest.json"
        manifest_paths.append(manifest_path.relative_to(campaign_root).as_posix())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts: Counter[str] = Counter()
        for candidate in manifest.get("candidates", []):
            row_type = classify(candidate)
            counts[row_type] += 1
            executed_counts[row_type] += 1
        rows.append(
            {
                "family": "crossing_ttc",
                "policy": policy,
                "sampler": sampler,
                "availability_status": "available",
                "manifest_path": manifest_path.relative_to(campaign_root).as_posix(),
                "counts": {key: counts[key] for key in row_status_keys},
            }
        )

not_available_row = {
    "family": "crossing_ttc",
    "policy": "all",
    "sampler": "guided_route_search",
    "availability_status": "not_available",
    "reason": (
        "guided_route_search is route-optimization based and is a design exclusion "
        "for the parametric crossing_ttc CandidateSpec family"
    ),
    "counts": {key: 1 if key == "not_available" else 0 for key in row_status_keys},
}
rows.append(not_available_row)
total_counts = Counter(executed_counts)
total_counts["not_available"] += 1

payload = {
    "schema_version": "issue-1501-adversarial-smoke-summary.v1",
    "issue": 1501,
    "parent_issue": 1488,
    "commit": git_commit,
    "family": "crossing_ttc",
    "budget_per_sampler": budget,
    "seed": seed,
    "synthetic": synthetic,
    "executed_policies": policies,
    "executed_samplers": ["random", "optuna"],
    "design_exclusion_samplers": ["guided_route_search"],
    "manifest_paths": manifest_paths,
    "archive_path": (campaign_root / "crossing_ttc" / "archive.json")
    .relative_to(campaign_root)
    .as_posix(),
    "counts": {key: total_counts[key] for key in row_status_keys},
    "rows": rows,
    "non_success_evidence": {
        "fallback": total_counts["fallback"] > 0,
        "degraded": total_counts["degraded"] > 0,
        "valid_failure": total_counts["valid_failure"] > 0,
        "invalid_candidate": total_counts["invalid_candidate"] > 0,
        "simulation_error": executed_counts["simulation_error"] > 0,
        "not_available": total_counts["not_available"] > 0,
    },
}
(campaign_root / "crossing_ttc" / "row_status_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

log "Issue #1501 smoke completed. Campaign root: ${CAMPAIGN_ROOT}"
