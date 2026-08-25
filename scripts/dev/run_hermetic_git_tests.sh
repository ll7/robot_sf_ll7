#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_hermetic_git_tests.sh [pytest-args...]

Runs the Git-backed test lane with all ambient Git identity and global/system
configuration disabled, so fixtures must configure their own test-only identity.

The wrapper:
- unsets GIT_AUTHOR_NAME/GIT_AUTHOR_EMAIL/GIT_COMMITTER_NAME/GIT_COMMITTER_EMAIL;
- points GIT_CONFIG_GLOBAL at /dev/null and sets GIT_CONFIG_NOSYSTEM=1;
- forwards any pytest arguments (default: the Git-backed dev/tool modules).

Use this to reproduce clean-runner Git failures locally:
  scripts/dev/run_hermetic_git_tests.sh
  scripts/dev/run_hermetic_git_tests.sh tests/dev/test_worktree_capacity.py -q
EOF
}

if [[ "$#" -gt 0 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

# Remove any ambient author/committer identity so fixtures cannot rely on it.
unset GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL || true

# Disable global and system Git configuration.
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1

# Default lane: the Git-backed dev and tool test modules that create commits.
DEFAULT_TARGETS=(
  tests/dev/test_worktree_capacity.py
  tests/dev/test_check_prepublication_state.py
  tests/dev/test_ci_capacity_preflight.py
  tests/dev/test_optional_import_pr_freshness.py
  tests/dev/test_vcs_version_derivation.py
  tests/tools/test_catalog_evidence.py
  tests/tools/test_check_context_note_freshness.py
  tests/tools/test_plan_context_note_archival.py
  tests/validation/test_pr_contract_check.py
  tests/validation/test_check_docs_proof_consistency.py
  tests/unit/test_snqi_weights_git_sha_provenance.py
  tests/integration/test_git_hook_prevention.py
  tests/test_ci_script_contract.py
  tests/support/test_environment_guards.py
)

# When the caller passes an explicit test file/module target (first non-option
# argument ending in .py or containing ::), forward the full arg list unchanged.
# Otherwise run the default lane and treat every argument as a pytest option.
TARGETS=("${DEFAULT_TARGETS[@]}")
OPTIONS=()
EXPLICIT=0
for arg in "$@"; do
  if [[ "$arg" == *.py || "$arg" == *"::"* ]]; then
    EXPLICIT=1
    break
  fi
done

if [[ "$EXPLICIT" -eq 1 ]]; then
  TARGETS=()
fi

echo "Running Git-backed tests under hermetic identity (GIT_CONFIG_GLOBAL=/dev/null, GIT_CONFIG_NOSYSTEM=1)" >&2
if [[ "$EXPLICIT" -eq 1 ]]; then
  uv run pytest "$@"
else
  uv run pytest "${TARGETS[@]}" "$@"
fi
