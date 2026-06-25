#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/dev/pr_ready_check.sh [--help | -h]

Runs PR readiness gates: ruff format/fix, pytest, coverage checks,
docstring-todo diff/ratchet checks, and freshness stamp write.

Environment variables:
  BASE_REF            Base ref to compare against for coverage and freshness
                      (default: origin/main)
  PR_READY_MODE       Set to "final" for proof-on-committed-HEAD mode, or
                      "interim" for work-in-progress feedback.
                      Overrides PR_READY_FINAL when set.
  PR_READY_FINAL      Legacy compatibility flag: "1", "true", "yes", or "on"
                      for final mode; "0", "false", "no", "off" for interim.
                      Ignored when PR_READY_MODE is set.
  PR_READY_SKIP_PREFLIGHT  Set to "1" to skip the cheap preflight check for
                      test-collection dependencies (duckdb, pyarrow).
  PR_READY_PR_BODY_FILE  Optional markdown PR body. When set, readiness checks
                      that deferred work has a linked issue or explicit NA.
  PR_READY_REQUIRE_OPEN_FOLLOWUP_ISSUES
                      Set to "0" to skip open-state verification for linked
                      follow-up issues when PR_READY_PR_BODY_FILE is set.
EOF
}

if [ "$#" -gt 0 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

export BASE_REF="${BASE_REF:-origin/main}"
export MIN_COVERAGE="${MIN_COVERAGE:-80}"
export GOAL_COVERAGE="${GOAL_COVERAGE:-100}"
export PR_READY_FINAL="${PR_READY_FINAL:-0}"
export PR_READY_MODE="${PR_READY_MODE:-}"
export PR_READY_SKIP_PREFLIGHT="${PR_READY_SKIP_PREFLIGHT:-0}"
export PR_READY_PR_BODY_FILE="${PR_READY_PR_BODY_FILE:-}"
export PR_READY_REQUIRE_OPEN_FOLLOWUP_ISSUES="${PR_READY_REQUIRE_OPEN_FOLLOWUP_ISSUES:-1}"

worktree_state() {
  if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    printf 'dirty\n'
  else
    printf 'clean\n'
  fi
}

print_compact_worktree_status() {
  local max_lines=30
  local tracked_status
  tracked_status=$(git status --short --untracked-files=no)
  if [[ -n "$tracked_status" ]]; then
    printf 'Tracked or staged changes:\n' >&2
    printf '%s\n' "$tracked_status" | head -n "$max_lines" >&2
    local total_lines
    total_lines=$(printf '%s\n' "$tracked_status" | wc -l)
    if (( total_lines > max_lines )); then
      printf '... truncated (%d more tracked lines)\n' "$((total_lines - max_lines))" >&2
    fi
  fi
  local generated_path
  local generated_present=()
  for generated_path in .venv .opencode node_modules output .pytest_cache __pycache__; do
    if [[ -e "$generated_path" ]]; then
      generated_present+=("$generated_path")
    fi
  done
  if [[ ${#generated_present[@]} -gt 0 ]]; then
    printf 'Generated paths present (contents not enumerated):\n' >&2
    printf '%s\n' "${generated_present[@]}" >&2
  fi
  local porcelain_count
  porcelain_count=$(git status --porcelain --untracked-files=normal | wc -l)
  if (( porcelain_count > 0 )); then
    printf 'Full untracked inventory omitted from this message (%d porcelain lines detected).\n' "$porcelain_count" >&2
    printf 'Run `git status --short --untracked-files=normal` only when a full inventory is needed.\n' >&2
  elif [[ -z "$tracked_status" && ${#generated_present[@]} -eq 0 ]]; then
    printf 'No tracked changes or known generated paths detected.\n' >&2
  fi
}

is_optional_readiness_path() {
  case "$1" in
    robot_sf/benchmark/*|\
    robot_sf/baselines/drl_vo.py|\
    robot_sf/feature_extractor.py|\
    robot_sf/feature_extractors/*|\
    robot_sf/planner/*|\
    robot_sf/training/*|\
    robot_sf_carla_bridge/*|\
    scripts/multi_extractor_training.py|\
    scripts/tools/benchmark_feature_extractors.py|\
    scripts/tools/probe_social_navigation_pyenvs_socialforce_runtime.py|\
    scripts/tools/probe_sonic_model_inference.py|\
    scripts/training/*|\
    tests/benchmark/*|\
    tests/benchmark_full/*|\
    tests/carla_bridge/*|\
    tests/integration/*|\
    tests/planner/*|\
    tests/render/*|\
    tests/training/*|\
    tests/visuals/*|\
    tests/sb3_test.py|\
    tests/tools/test_probe_sonic_model_inference.py|\
    tests/test_baseline_ppo_smoke.py|\
    tests/test_benchmark_visualization_integration.py|\
    tests/test_feature_extractors.py|\
    tests/test_grid_socnav_extractor.py|\
    tests/test_map_runner_ppo.py|\
    tests/test_map_runner_sac.py|\
    tests/test_output_root_migration.py|\
    tests/test_ppo_diagnostics.py|\
    tests/test_predictive_model.py|\
    tests/unit/test_cli_logging_flags.py|\
    tests/unit/test_figure_orchestrator_requirements.py|\
    tests/unit/test_runner_helper_coverage.py|\
    tests/unit/test_runner_video.py)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

pr_ready_mode_lower=$(printf '%s' "$PR_READY_MODE" | tr '[:upper:]' '[:lower:]')
case "$pr_ready_mode_lower" in
  "")
    pr_ready_final_lower=$(printf '%s' "$PR_READY_FINAL" | tr '[:upper:]' '[:lower:]')
    case "$pr_ready_final_lower" in
      1|true|yes|on) pr_ready_final=1 ;;
      0|false|no|off) pr_ready_final=0 ;;
      *)
        printf 'Invalid PR_READY_FINAL value %q (expected 1|0|true|false|yes|no|on|off).\n' "$PR_READY_FINAL" >&2
        exit 2
        ;;
    esac
    ;;
  final) pr_ready_final=1 ;;
  interim) pr_ready_final=0 ;;
  *)
    printf 'Invalid PR_READY_MODE value %q (expected final|interim).\n' "$PR_READY_MODE" >&2
    exit 2
    ;;
esac

if [[ "$pr_ready_final" == "1" && "$(worktree_state)" != "clean" ]]; then
  printf 'Final PR readiness requires a clean non-ignored worktree before validation.\n' >&2
  printf 'Commit or remove local changes, or run without PR_READY_MODE=final for interim feedback.\n' >&2
  print_compact_worktree_status
  exit 2
fi

if [[ "$pr_ready_final" == "1" ]]; then
  preflight_check_test_deps
fi

changed_files=()
core_changed_files=()
optional_changed_files=()
while IFS= read -r changed_file; do
  [[ -z "$changed_file" ]] && continue
  changed_files+=("$changed_file")
  if is_optional_readiness_path "$changed_file"; then
    optional_changed_files+=("$changed_file")
  else
    core_changed_files+=("$changed_file")
  fi
done < <(git diff --name-only --diff-filter=ACMRT "$BASE_REF...HEAD")

if [[ ${#changed_files[@]} -gt 0 ]]; then
  printf 'Changed files vs %s:\n' "$BASE_REF" >&2
  printf '%s\n' "${changed_files[@]}" >&2
fi
if [[ ${#core_changed_files[@]} -gt 0 ]]; then
  printf 'Core-ready changed files:\n' >&2
  printf '%s\n' "${core_changed_files[@]}" >&2
fi
if [[ ${#optional_changed_files[@]} -gt 0 ]]; then
  printf 'Optional-extra changed files requiring the predictive lane:\n' >&2
  printf '%s\n' "${optional_changed_files[@]}" >&2
fi

followup_args=()
if [[ "$pr_ready_final" == "1" ]]; then
  followup_args+=(--require-body)
fi
uv run python "$SCRIPT_DIR/check_pr_followups.py" "${followup_args[@]}"
uv run python "$SCRIPT_DIR/check_fast_results_claim_map.py"
"$SCRIPT_DIR/ruff_fix_format.sh"
printf 'Running core readiness lane.\n' >&2
ROBOT_SF_PYTEST_COVERAGE=1 ROBOT_SF_TEST_LANE=core "$SCRIPT_DIR/run_tests_parallel.sh" --lane core
if [[ ${#optional_changed_files[@]} -gt 0 ]]; then
  printf 'Running optional-extra lane for predictive/optional changed files.\n' >&2
  optional_pytest_addopts="${PYTEST_ADDOPTS:-}"
  if [[ " $optional_pytest_addopts " != *" --cov-append "* ]]; then
    optional_pytest_addopts="${optional_pytest_addopts:+$optional_pytest_addopts }--cov-append"
  fi
  PYTEST_ADDOPTS="$optional_pytest_addopts" ROBOT_SF_PYTEST_COVERAGE=1 ROBOT_SF_TEST_LANE=optional "$SCRIPT_DIR/run_tests_parallel.sh" --lane optional
else
  printf 'No changed files require the optional-extra lane.\n' >&2
fi
"$SCRIPT_DIR/check_changed_coverage.sh"
"$SCRIPT_DIR/check_docstring_todos_diff.sh"
"$SCRIPT_DIR/check_docstring_todos_ratchet.sh"
uv run python "$SCRIPT_DIR/../validation/check_broad_exceptions.py"

freshness_args=(write --base-ref "$BASE_REF")
if [[ "$pr_ready_final" == "1" ]]; then
  freshness_args+=(--require-clean-tree)
elif [[ "$(worktree_state)" != "clean" ]]; then
  printf 'Warning: recording interim PR readiness from a dirty non-ignored worktree.\n' >&2
  printf 'Use PR_READY_MODE=final BASE_REF=%q %q for final committed-HEAD PR proof.\n' "$BASE_REF" "$0" >&2
fi
uv run python "$SCRIPT_DIR/pr_ready_freshness.py" "${freshness_args[@]}"
