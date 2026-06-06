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
