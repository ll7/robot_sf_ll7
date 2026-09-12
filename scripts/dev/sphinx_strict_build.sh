#!/usr/bin/env bash
# Canonical curated strict Sphinx build (issue #8723).
#
# Builds only the docs/index.rst toctree closure, promotes warnings to errors,
# and fails closed on any warning except cross-references that resolve to an
# existing repository document outside the curated site. The curated source set
# is pinned in docs/sphinx_curated_sources.json; run with --write-manifest after
# reviewing an intentional toctree change.
#
# Usage:
#   scripts/dev/sphinx_strict_build.sh
#   scripts/dev/sphinx_strict_build.sh --builder dummy --json
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,12p' "$0"
  exit 0
fi

exec uv run python scripts/dev/sphinx_curated_build.py "$@"
