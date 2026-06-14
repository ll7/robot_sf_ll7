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

worktree_state() {
  if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    printf 'dirty\n'
  else
    printf 'clean\n'
  fi
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
  git status --short --untracked-files=normal >&2
  exit 2
fi

if [[ "$pr_ready_final" == "1" ]]; then
  preflight_check_test_deps
fi

"$SCRIPT_DIR/ruff_fix_format.sh"
"$SCRIPT_DIR/run_tests_parallel.sh"
"$SCRIPT_DIR/check_changed_coverage.sh"
"$SCRIPT_DIR/check_docstring_todos_diff.sh"
"$SCRIPT_DIR/check_docstring_todos_ratchet.sh"

freshness_args=(write --base-ref "$BASE_REF")
if [[ "$pr_ready_final" == "1" ]]; then
  freshness_args+=(--require-clean-tree)
elif [[ "$(worktree_state)" != "clean" ]]; then
  printf 'Warning: recording interim PR readiness from a dirty non-ignored worktree.\n' >&2
  printf 'Use PR_READY_MODE=final BASE_REF=%q %q for final committed-HEAD PR proof.\n' "$BASE_REF" "$0" >&2
fi
uv run python "$SCRIPT_DIR/pr_ready_freshness.py" "${freshness_args[@]}"
