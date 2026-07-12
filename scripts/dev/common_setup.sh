#!/usr/bin/env bash
# shellcheck shell=bash

# Shared repository setup for scripts/dev helpers.
#
# Always resolve from the caller's current checkout. Some wrappers export
# REPO_ROOT before invoking nested tests or fixture repositories, and trusting a
# stale inherited value can make copied helper scripts operate on the outer
# checkout instead of their own repository.
REPO_ROOT="$(git rev-parse --show-toplevel)"
export REPO_ROOT
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.venv/bin/activate"
fi

# Resolve the absolute path to a codex-agent-runs artifact subdirectory under
# the shared Git common directory.  In a linked worktree `.git` is a file, so
# writing to a literal `.git/codex-agent-runs/...` path fails.  This function
# resolves the correct absolute path via `git rev-parse --git-common-dir` and
# prints it.  Callers must `mkdir -p` the result before writing.
#
# Usage:
#   artifact_dir="$(resolve_agent_artifact_dir my-subdir)"
#   mkdir -p "$artifact_dir"
#   echo "data" > "$artifact_dir/result.json"
#
# Returns 0 on success, prints the resolved path.  Falls back to
# output/tmp/<subdir> when git is unavailable.
resolve_agent_artifact_dir() {
  local subdir="${1:?resolve_agent_artifact_dir requires a subdirectory name}"
  local common_dir
  common_dir="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  if [[ -n "$common_dir" ]]; then
    printf '%s\n' "$common_dir/codex-agent-runs/active/$subdir"
  else
    printf '%s\n' "$REPO_ROOT/output/tmp/$subdir"
  fi
}

# Cheap shell preflight: verify that test-collection dependencies (duckdb,
# pyarrow) are importable before expensive pytest collection runs.  Exit 2
# with a concise message on failure so agents see the blocker immediately.
# Set PR_READY_SKIP_PREFLIGHT=1 to bypass this check.
preflight_check_test_deps() {
  if [ "${PR_READY_SKIP_PREFLIGHT:-0}" = "1" ]; then
    return 0
  fi
  local missing
  if ! missing="$(uv run python - <<'PY' 2>&1
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
    printf 'Missing or unavailable modules: %s\n' "$missing" >&2
    printf 'Run `uv sync --all-extras` in this worktree, then rerun final PR readiness.\n' >&2
    exit 2
  fi
}
