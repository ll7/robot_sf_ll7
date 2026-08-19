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

# Verify that a selected worktree environment contains the declared profile before
# a docs-proof entry point invokes `uv run`. A lightweight `uv run` in a fresh
# worktree can create a Python-only `.venv`; later `UV_NO_SYNC=1` consumers then
# fail with an opaque import error. Keep the check import-spec-only and fail with
# the one supported recovery command instead of silently repairing or selecting a
# different environment.
preflight_check_worktree_dependency_profile() {
  local profile="${1:-core}"
  local venv_root
  if [[ -d "$REPO_ROOT/.venv" || -L "$REPO_ROOT/.venv" ]]; then
    venv_root="$REPO_ROOT/.venv"
  elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    venv_root="$VIRTUAL_ENV"
  else
    printf "ERROR: worktree dependency profile '%s' has no initialized virtualenv.\n" "$profile" >&2
    printf 'Run `cd %q && scripts/dev/bootstrap_worktree.sh`, then rerun this command.\n' \
      "$REPO_ROOT" >&2
    return 2
  fi

  if [[ "$venv_root" != /* ]]; then
    venv_root="$REPO_ROOT/$venv_root"
  fi
  if [[ ! -x "$venv_root/bin/python" ]]; then
    printf "ERROR: worktree dependency profile '%s' cannot use incomplete virtualenv: %s\n" \
      "$profile" "$venv_root" >&2
    printf 'Run `cd %q && scripts/dev/bootstrap_worktree.sh`, then rerun this command.\n' \
      "$REPO_ROOT" >&2
    return 2
  fi

  local report
  if ! report="$("$venv_root/bin/python" \
    "$REPO_ROOT/scripts/dev/check_worktree_optional_deps.py" \
    --profile "$profile" 2>&1)"; then
    printf "ERROR: worktree dependency profile '%s' is incomplete in %s.\n" "$profile" "$venv_root" >&2
    printf '%s\n' "$report" >&2
    printf 'Run `cd %q && scripts/dev/bootstrap_worktree.sh`, then rerun this command.\n' \
      "$REPO_ROOT" >&2
    return 2
  fi
}

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
# pyarrow, pandas) are importable before expensive pytest collection runs.  Exit 2
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
    for module_name in ("duckdb", "pyarrow", "pandas")
    if importlib.util.find_spec(module_name) is None
]
if missing:
    print(", ".join(missing))
    raise SystemExit(1)
PY
  )"; then
    printf 'Final PR readiness requires analytics dependencies: duckdb, pyarrow, and pandas.\n' >&2
    printf 'Missing or unavailable modules: %s\n' "$missing" >&2
    printf 'Run `uv sync --all-extras` in this worktree, then rerun final PR readiness.\n' >&2
    exit 2
  fi
}

# The threaded rollout path imports a context manager added to the vendored
# fast-pysf package.  A stale force-included install otherwise fails during
# pytest collection with an opaque ImportError (issue #5665).
preflight_check_fast_pysf() {
  if [ "${PR_READY_SKIP_PREFLIGHT:-0}" = "1" ]; then
    return 0
  fi
  if ! uv run python "$REPO_ROOT/scripts/dev/check_fast_pysf_runtime.py"; then
    printf 'Final PR readiness cannot collect the core suite with this PySocialForce environment.\n' >&2
    printf 'Run `uv sync --all-extras --reinstall-package robot-sf` in this worktree, then rerun readiness.\n' >&2
    exit 2
  fi
}
