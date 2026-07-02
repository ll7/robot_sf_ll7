#!/usr/bin/env bash
set -euo pipefail

# Diff-scoped PR handoff check:
# - docs/context-only diffs include README, INDEX, and catalog paths explicitly,
#   preserving strict docs proof and context-catalog diagnostics.
# - non-docs or mixed code diffs run changed-file proof only, so unrelated
#   historical docs/context/catalog.yaml debt does not block code-only readiness.
# - full evidence catalog hygiene remains explicit via
#   scripts/validation/check_docs_proof_consistency.py --check-evidence-catalog.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

BASE_REF="${BASE_REF:-origin/main}"
DOCS_CONTEXT_PROOF_PATHS=(
  "docs/context/README.md"
  "docs/context/INDEX.md"
  "docs/context/catalog.yaml"
)

path_selected() {
  local needle="$1"
  shift
  local selected
  for selected in "$@"; do
    if [ "$selected" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

changed_files="$(git diff --name-only "${BASE_REF}...HEAD")"
docs_context_only=1

if [ -z "$changed_files" ]; then
  docs_context_only=0
else
  while IFS= read -r file; do
    if [ -z "$file" ]; then
      continue
    fi
    if [[ "$file" != docs/context/* ]]; then
      docs_context_only=0
      break
    fi
  done <<<"$changed_files"
fi

if [ "$docs_context_only" -eq 1 ]; then
  paths=()
  while IFS= read -r file; do
    if [ -z "$file" ]; then
      continue
    fi
    if ! path_selected "$file" "${paths[@]}"; then
      paths+=("$file")
    fi
  done <<<"$changed_files"

  for required_path in "${DOCS_CONTEXT_PROOF_PATHS[@]}"; do
    if ! path_selected "$required_path" "${paths[@]}"; then
      paths+=("$required_path")
    fi
  done

  path_args=()
  for selected in "${paths[@]}"; do
    path_args+=(--path "$selected")
  done

  uv run --active python scripts/validation/check_docs_proof_consistency.py \
    --base "$BASE_REF" "${path_args[@]}"
else
  uv run --active python scripts/validation/check_docs_proof_consistency.py --base "$BASE_REF"
fi
