#!/usr/bin/env bash
#SBATCH --job-name=adv1502-2fam
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-adversarial-two-family1502.out

# Run the bounded Issue #1502 two-family adversarial comparison.
#
# This launcher consumes the frozen Issue #1500 manifest and the completed #1501
# smoke gate. It keeps family/search-engine design exclusions explicit:
# - crossing_ttc: random + optuna_tpe are available; guided_route_search is not_available.
# - classic_head_on_corridor: guided_route_search is available; random/optuna_tpe are not_available.
set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
OUTPUT_ROOT=${ADVERSARIAL_1502_OUTPUT_ROOT:-${PROJECT_ROOT}/output/adversarial/issue_1502}
LABEL=${ADVERSARIAL_1502_LABEL:-issue1502-two-family-${SLURM_JOB_ID:-local}}
RESULTS_ROOT=${ADVERSARIAL_1502_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/adversarial-two-family1502-job-${SLURM_JOB_ID:-local}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/adversarial-two-family1502-${SLURM_JOB_ID:-local}}
PYTHON_BIN=${PROJECT_ROOT}/.venv/bin/python
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
CROSSING_BUDGET=${ADVERSARIAL_1502_CROSSING_BUDGET:-256}
ROUTE_TRIALS=${ADVERSARIAL_1502_ROUTE_TRIALS:-100}
CROSSING_SEED=${ADVERSARIAL_1502_CROSSING_SEED:-42}
ROUTE_SEED=${ADVERSARIAL_1502_ROUTE_SEED:-123}
CROSSING_POLICIES=${ADVERSARIAL_1502_CROSSING_POLICIES:-"goal orca"}
SYNTHETIC=${ADVERSARIAL_1502_SYNTHETIC:-false}
MODULES_AVAILABLE=0
CAMPAIGN_ROOT=""

if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT=${PROJECT_ROOT}/${OUTPUT_ROOT}
fi
if [[ "${RESULTS_ROOT}" != /* ]]; then
  RESULTS_ROOT=${PROJECT_ROOT}/${RESULTS_ROOT}
fi

log() {
  echo "[adv1502] $*"
}

die() {
  echo "[adv1502] $*" >&2
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

  echo "[adv1502] module command unavailable; continuing without module loads." >&2
  return 0
}

run_in_allocation() {
  if [[ "${ADVERSARIAL_1502_USE_SRUN:-0}" == "1" ]] && command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]] && srun --version >/dev/null 2>&1; then
    log "Launching with srun on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}."
    srun --cpu_bind=cores "$@"
  else
    echo "[adv1502] running directly inside the batch allocation; set ADVERSARIAL_1502_USE_SRUN=1 to opt into srun." >&2
    "$@"
  fi
}

cleanup() {
  if [[ -n "${CAMPAIGN_ROOT}" && -d "${CAMPAIGN_ROOT}" ]]; then
    mkdir -p "${RESULTS_ROOT}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --partial --prune-empty-dirs "${CAMPAIGN_ROOT}/" "${RESULTS_ROOT}/" \
        || echo "[adv1502] Warning: rsync failed to sync artifacts to ${RESULTS_ROOT}" >&2
    else
      cp -r "${CAMPAIGN_ROOT}/." "${RESULTS_ROOT}/" \
        || echo "[adv1502] Warning: cp failed to sync artifacts to ${RESULTS_ROOT}" >&2
    fi
  fi
}
trap cleanup EXIT

case "${SYNTHETIC,,}" in
  1|true|yes|on) SYNTHETIC=true ;;
  0|false|no|off|"") SYNTHETIC=false ;;
  *) die "ADVERSARIAL_1502_SYNTHETIC must be true/false-like, got: ${SYNTHETIC}" ;;
esac

for numeric_pair in \
  "ADVERSARIAL_1502_CROSSING_BUDGET:${CROSSING_BUDGET}" \
  "ADVERSARIAL_1502_ROUTE_TRIALS:${ROUTE_TRIALS}" \
  "ADVERSARIAL_1502_CROSSING_SEED:${CROSSING_SEED}" \
  "ADVERSARIAL_1502_ROUTE_SEED:${ROUTE_SEED}"
do
  name=${numeric_pair%%:*}
  value=${numeric_pair#*:}
  if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -lt 1 ]]; then
    die "${name} must be a positive integer, got: ${value}"
  fi
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  die "Expected repo virtualenv python at ${PYTHON_BIN}"
fi

for required in \
  configs/adversarial/issue_1500_adversarial_comparison_manifest.v1.yaml \
  configs/adversarial/crossing_ttc_space.yaml \
  configs/scenarios/templates/crossing_ttc.yaml \
  configs/adversarial_routes/default.yaml \
  configs/scenarios/classic_interactions.yaml \
  maps/svg_maps/classic_head_on_corridor.svg \
  scripts/tools/compare_adversarial_samplers.py \
  scripts/tools/curate_adversarial_failure_archive.py \
  scripts/tools/generate_adversarial_routes.py
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
mkdir -p "${CAMPAIGN_ROOT}/crossing_ttc" "${CAMPAIGN_ROOT}/classic_head_on_corridor"

cd "${PROJECT_ROOT}"

log "Commit: $(git rev-parse HEAD)"
log "Campaign root: ${CAMPAIGN_ROOT}"
log "Results sync root: ${RESULTS_ROOT}"
log "Crossing/TTC budget per sampler: ${CROSSING_BUDGET}"
log "Classic head-on route trials: ${ROUTE_TRIALS}"
log "Crossing/TTC seed: ${CROSSING_SEED}"
log "Classic head-on route seed: ${ROUTE_SEED}"
log "Crossing/TTC policies: ${CROSSING_POLICIES}"
log "Synthetic crossing evaluator: ${SYNTHETIC}"

synthetic_arg=()
if [[ "${SYNTHETIC}" == "true" ]]; then
  synthetic_arg=(--synthetic)
fi

manifest_args=()
for policy in ${CROSSING_POLICIES}; do
  case "${policy}" in
    goal|orca) ;;
    *) die "Unsupported Issue #1502 crossing policy row '${policy}'. Expected goal or orca." ;;
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
      --budget "${CROSSING_BUDGET}" \
      --seed "${CROSSING_SEED}" \
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

route_config=${WORKDIR}/issue_1502_classic_head_on_corridor.yaml
"${PYTHON_BIN}" - "${PROJECT_ROOT}/configs/adversarial_routes/default.yaml" "${route_config}" "${CAMPAIGN_ROOT}/classic_head_on_corridor/guided_route_search" "${ROUTE_TRIALS}" "${ROUTE_SEED}" "${PROJECT_ROOT}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
output_root = Path(sys.argv[3])
trial_count = int(sys.argv[4])
seed = int(sys.argv[5])
project_root = Path(sys.argv[6])

data = yaml.safe_load(source.read_text(encoding="utf-8"))
data["scenario"]["scenario_file"] = str(project_root / "configs/scenarios/classic_interactions.yaml")
data["scenario"]["scenario_id"] = "classic_head_on_corridor_low"
data["optimization"]["trial_count"] = trial_count
data["optimization"]["seed"] = seed
data["output"]["root"] = str(output_root)
target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

log "Running classic_head_on_corridor guided_route_search"
run_in_allocation \
  "${PYTHON_BIN}" scripts/tools/generate_adversarial_routes.py \
    --config "${route_config}" \
    --scenario-id classic_head_on_corridor_low

log "Writing two-family row-status summary"
GIT_COMMIT=$(git rev-parse HEAD)
"${PYTHON_BIN}" - "${CAMPAIGN_ROOT}" "${CROSSING_BUDGET}" "${ROUTE_TRIALS}" "${CROSSING_SEED}" "${ROUTE_SEED}" "${SYNTHETIC}" "${CROSSING_POLICIES}" "${GIT_COMMIT}" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

campaign_root = Path(sys.argv[1])
crossing_budget = int(sys.argv[2])
route_trials = int(sys.argv[3])
crossing_seed = int(sys.argv[4])
route_seed = int(sys.argv[5])
synthetic = sys.argv[6].lower() == "true"
policies = sys.argv[7].split()
git_commit = sys.argv[8]

executed_counts: Counter[str] = Counter()
manifest_paths: list[str] = []
rows: list[dict[str, object]] = []


def classify(candidate: dict[str, object]) -> str:
    attribution = candidate.get("failure_attribution")
    if isinstance(attribution, dict):
        primary = str(attribution.get("primary_failure") or "")
        status = str(attribution.get("status") or "")
        if primary == "invalid_candidate" or status == "not_evaluated":
            return "invalid_candidate"
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
                "execution_mode": "candidate_spec",
                "manifest_path": manifest_path.relative_to(campaign_root).as_posix(),
                "counts": dict(sorted(counts.items())),
            }
        )

for sampler in ("guided_route_search",):
    rows.append(
        {
            "family": "crossing_ttc",
            "policy": "all",
            "sampler": sampler,
            "availability_status": "not_available",
            "reason": "Route-level guided search is a design exclusion for the parametric crossing_ttc CandidateSpec family.",
            "counts": {"not_available": 1},
        }
    )
    executed_counts["not_available"] += 1

route_summaries = sorted(
    (campaign_root / "classic_head_on_corridor" / "guided_route_search").glob("*/summary.json")
)
if len(route_summaries) != 1:
    raise SystemExit(f"Expected exactly one guided_route_search summary, found {len(route_summaries)}")
route_summary_path = route_summaries[0]
route_summary = json.loads(route_summary_path.read_text(encoding="utf-8"))
valid_trials = int(route_summary["diagnostics"]["valid_trial_count"])
failed_trials = int(route_summary["diagnostics"]["failed_trials"])
rows.append(
    {
        "family": "classic_head_on_corridor",
        "policy": "classic_global_theta_star",
        "sampler": "guided_route_search",
        "availability_status": "available",
        "execution_mode": "route_search",
        "summary_path": route_summary_path.relative_to(campaign_root).as_posix(),
        "route_override_path": route_summary_path.with_name("route_overrides.yaml").relative_to(campaign_root).as_posix(),
        "counts": {
            "valid_route_trial": valid_trials,
            "failed_trial": failed_trials,
        },
        "best_score": route_summary["best_score"],
    }
)
executed_counts["valid_route_trial"] += valid_trials
executed_counts["failed_trial"] += failed_trials

for sampler in ("random", "optuna_tpe"):
    rows.append(
        {
            "family": "classic_head_on_corridor",
            "policy": "all",
            "sampler": sampler,
            "availability_status": "not_available",
            "reason": "Parametric CandidateSpec search is not applicable to route-level head-on corridor optimization.",
            "counts": {"not_available": 1},
        }
    )
    executed_counts["not_available"] += 1

summary = {
    "schema_version": "issue-1502-adversarial-two-family-summary.v1",
    "issue": 1502,
    "parent_issue": 1488,
    "predecessor_issue": 1501,
    "commit": git_commit,
    "families": ["crossing_ttc", "classic_head_on_corridor"],
    "crossing_budget_per_sampler": crossing_budget,
    "classic_head_on_corridor_route_trials": route_trials,
    "crossing_seed": crossing_seed,
    "route_seed": route_seed,
    "synthetic": synthetic,
    "executed_samplers": ["random", "optuna_tpe", "guided_route_search"],
    "counts": dict(sorted(executed_counts.items())),
    "rows": rows,
    "artifact_policy": {
        "benchmark_evidence": False,
        "claim_scope": "bounded two-family development stress comparison",
        "fallback_or_degraded_success": False,
    },
}
(campaign_root / "row_status_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True))
PY

log "Issue #1502 two-family adversarial comparison completed."
