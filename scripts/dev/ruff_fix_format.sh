#!/usr/bin/env bash
set -euo pipefail

# Scope-aware ruff fix/format for PR readiness (issue #7710).
#
# With no file arguments, validates the whole checkout (legacy behavior).
# With file arguments, only those files are fixed/formatted, so unrelated
# unformatted files already present on the base ref never dirty a clean PR
# worktree during pr_ready_check.sh.
#
# Note: `ruff check --fix` returns nonzero when violations remain unfixed
# (e.g. preview-rule or docstring rules without autofix), so the check pass
# tolerates failure just like the legacy whole-tree invocation; the final
# `ruff check --statistics` gate still reports the real state.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

if [[ "$#" -gt 0 ]]; then
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    printf 'ERROR: scoped ruff fix/format requires a clean tracked tree (issue #7710).\n' >&2
    printf 'Commit or stash tracked changes before running format-scoped readiness.\n' >&2
    exit 2
  fi
  uv run ruff check --fix "$@" --output-format concise || true
  uv run ruff format "$@"
  uv run ruff check "$@" --statistics
else
  uv run ruff check --fix . --output-format concise || true
  uv run ruff format .
  uv run ruff check . --statistics
fi
