#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SBATCH_MAX_TIME_SCRIPT="$SCRIPT_DIR/sbatch_use_max_time.sh"
PARTITION_STATUS_SCRIPT="$SCRIPT_DIR/auxme_partition_status.sh"

STAGE="nominal_sanity"
PARTITION="a30"
QOS="a30-gpu"
THROTTLE="2"
WORKERS="2"
HORIZON=""
DRY_RUN=0
SHOW_STATUS=1
ALL_IMPLEMENTED=0
CANDIDATES_FILE_OVERRIDE=""
PIN_HEAD=0
REQUIRE_CLEAN_WORKTREE=0
RUN_ID="policy_search_$(date -u +%Y%m%d_%H%M%S)"
EXTRA_SBATCH_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/dev/sbatch_policy_search_sweep.sh [options]

Submit a Slurm array that evaluates every implemented policy-search candidate
registered for one funnel stage.

Options:
  --stage <name>          Funnel stage: smoke, nominal_sanity, stress_slice, full_matrix,
                          leader_collision_slice_h500, full_matrix_h500,
                          robustness_extension
  --partition <name>      Slurm partition (default: a30)
  --qos <name>            Slurm QoS (default: a30-gpu)
  --throttle <n>          Slurm array throttle, e.g. 1 or 2 (default: 2)
  --workers <n>           Runner workers per candidate task (default: 2)
  --horizon <n>           Override stage horizon passed to the candidate runner
  --run-id <id>           Output run id (default: UTC timestamp)
  --all-implemented       Run every implemented candidate at the selected stage,
                          ignoring candidate required_stages
  --candidates-file <p>   Use an explicit newline-delimited candidate file
  --pin-head              Record current HEAD and make each array task fail if
                          the checkout has moved before it starts
  --require-clean-worktree
                          Make each array task fail unless the worktree has no
                          non-ignored changes
  --clean-pinned          Convenience alias for --pin-head --require-clean-worktree
  --sbatch-arg <arg>      Extra sbatch argument
  --no-status             Skip partition status table
  --dry-run               Print resolved submission without submitting
  -h, --help              Show help

Examples:
  scripts/dev/sbatch_policy_search_sweep.sh --stage nominal_sanity --dry-run
  scripts/dev/sbatch_policy_search_sweep.sh --stage full_matrix --throttle 2
  scripts/dev/sbatch_policy_search_sweep.sh --stage full_matrix --horizon 500 --candidates-file candidates.txt
  scripts/dev/sbatch_policy_search_sweep.sh --stage full_matrix_h500 --clean-pinned \
    --candidates-file configs/policy_search/candidate_sets/h500_leader_clean_rerun.txt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --throttle)
      THROTTLE="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --horizon)
      HORIZON="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --all-implemented)
      ALL_IMPLEMENTED=1
      shift
      ;;
    --candidates-file)
      CANDIDATES_FILE_OVERRIDE="$2"
      shift 2
      ;;
    --pin-head)
      PIN_HEAD=1
      shift
      ;;
    --require-clean-worktree)
      REQUIRE_CLEAN_WORKTREE=1
      shift
      ;;
    --clean-pinned)
      PIN_HEAD=1
      REQUIRE_CLEAN_WORKTREE=1
      shift
      ;;
    --sbatch-arg)
      EXTRA_SBATCH_ARGS+=("$2")
      shift 2
      ;;
    --no-status)
      SHOW_STATUS=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      echo "Unexpected positional argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$STAGE" in
  smoke|nominal_sanity|stress_slice|full_matrix|leader_collision_slice_h500|full_matrix_h500|robustness_extension)
    ;;
  *)
    echo "Unsupported stage: $STAGE" >&2
    exit 2
    ;;
esac

mkdir -p "$REPO_ROOT/output/policy_search/sweeps/$RUN_ID" "$REPO_ROOT/output/slurm"
CANDIDATES_FILE="$REPO_ROOT/output/policy_search/sweeps/$RUN_ID/candidates_${STAGE}.txt"

cd "$REPO_ROOT"
if [[ -n "$CANDIDATES_FILE_OVERRIDE" ]]; then
  CANDIDATES_FILE="$(realpath "$CANDIDATES_FILE_OVERRIDE")"
  test -f "$CANDIDATES_FILE"
else
  candidate_args=("--list-candidates" "--stage" "$STAGE")
  if [[ "$ALL_IMPLEMENTED" == "1" ]]; then
    candidate_args+=("--all-implemented")
  fi
  uv run python scripts/tools/summarize_policy_search_portfolio.py "${candidate_args[@]}" \
    > "$CANDIDATES_FILE"
fi

CANDIDATE_COUNT="$(wc -l < "$CANDIDATES_FILE" | tr -d ' ')"
if [[ "$CANDIDATE_COUNT" == "0" ]]; then
  echo "No implemented candidates are registered for stage '$STAGE'." >&2
  exit 2
fi

if [[ "$SHOW_STATUS" == "1" ]]; then
  "$PARTITION_STATUS_SCRIPT"
fi

echo "[policy-search-submit] run_id=$RUN_ID stage=$STAGE candidates=$CANDIDATE_COUNT" >&2
echo "[policy-search-submit] horizon=${HORIZON:-stage-default}" >&2
echo "[policy-search-submit] candidates_file=$CANDIDATES_FILE" >&2

EXPECTED_COMMIT=""
if [[ "$PIN_HEAD" == "1" ]]; then
  EXPECTED_COMMIT="$(git rev-parse HEAD)"
  echo "[policy-search-submit] expected_commit=$EXPECTED_COMMIT" >&2
fi
if [[ "$REQUIRE_CLEAN_WORKTREE" == "1" ]]; then
  echo "[policy-search-submit] require_clean_worktree=1" >&2
fi

wrapper_args=(
  "--partition" "$PARTITION"
  "--qos" "$QOS"
  "--sbatch-arg" "--partition=$PARTITION"
  "--sbatch-arg" "--qos=$QOS"
  "--sbatch-arg" "--array=0-$((CANDIDATE_COUNT - 1))%$THROTTLE"
  "--sbatch-arg" "--job-name=rsf-pol-$STAGE"
  "--sbatch-arg" "--export=ALL,POLICY_SEARCH_STAGE=$STAGE,POLICY_SEARCH_CANDIDATES_FILE=$CANDIDATES_FILE,POLICY_SEARCH_RUN_ID=$RUN_ID,POLICY_SEARCH_WORKERS=$WORKERS,POLICY_SEARCH_HORIZON=$HORIZON,POLICY_SEARCH_EXPECTED_COMMIT=$EXPECTED_COMMIT,POLICY_SEARCH_REQUIRE_CLEAN=$REQUIRE_CLEAN_WORKTREE"
)

if [[ "$DRY_RUN" == "1" ]]; then
  wrapper_args+=("--dry-run")
fi
for arg in "${EXTRA_SBATCH_ARGS[@]}"; do
  wrapper_args+=("--sbatch-arg" "$arg")
done

"$SBATCH_MAX_TIME_SCRIPT" "${wrapper_args[@]}" SLURM/Auxme/policy_search_candidate_stage.sl
