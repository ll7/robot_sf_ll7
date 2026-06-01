#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

export BASE_REF="${BASE_REF:-origin/main}"
export MIN_COVERAGE="${MIN_COVERAGE:-80}"
export GOAL_COVERAGE="${GOAL_COVERAGE:-100}"
export PR_READY_FINAL="${PR_READY_FINAL:-0}"
export PR_READY_MODE="${PR_READY_MODE:-}"

worktree_state() {
  if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    printf 'dirty\n'
  else
    printf 'clean\n'
  fi
}

check_final_readiness_dependencies() {
  local missing_output
  if ! missing_output="$(uv run python - <<'PY'
from __future__ import annotations

import importlib.util

missing = [
    module_name
    for module_name in ("duckdb", "pyarrow")
    if importlib.util.find_spec(module_name) is None
]
if missing:
    print(", ".join(missing))
    raise SystemExit(1)
PY
  )"; then
    printf 'Final PR readiness requires analytics dependencies: duckdb and pyarrow.\n' >&2
    printf 'Missing or unavailable modules: %s\n' "$missing_output" >&2
    printf 'Run `uv sync --all-extras` in this worktree, then rerun final PR readiness.\n' >&2
    exit 2
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
  check_final_readiness_dependencies
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
