#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/local_signoff.sh [options] [<ci-phase> ...]

Run local CI phases on the exact pushed clean HEAD, then post advisory
gh-signoff commit statuses. This script never installs or changes branch
protection.

Default phases:
  lint test

Options:
  --full                 Run all scripts/dev/run_ci_local.sh phases and sign local-pr-ready.
  --context <name>       Sign this context after all phases pass. May repeat.
                         Default: local-<phase> for each selected phase.
  --no-setup             Pass through to scripts/dev/run_ci_local.sh for faster repeat runs.
  --no-auto-install      Do not install basecamp/gh-signoff when missing.
  --dry-run              Run validation and checks, but print instead of signing.
  -h, --help             Show this help.

Examples:
  scripts/dev/local_signoff.sh --no-setup lint test
  scripts/dev/local_signoff.sh --full
  scripts/dev/local_signoff.sh --context local-pr-ready lint test smoke
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

auto_install=true
dry_run=false
full=false
run_setup=true
phases=()
contexts=()

normalize_context() {
  printf "%s" "$1" \
    | tr "[:upper:]_" "[:lower:]-" \
    | sed -E "s/[^a-z0-9./-]+/-/g; s/^-+//; s/-+$//"
}

require_clean_pushed_head() {
  local status
  status="$(git status --porcelain)"
  if [[ -n "$status" ]]; then
    echo "Error: worktree has tracked or untracked changes; refusing to sign HEAD." >&2
    echo "Commit, stash, or clean changes before local signoff." >&2
    return 1
  fi

  local branch
  branch="$(git symbolic-ref --quiet --short HEAD)" || {
    echo "Error: detached HEAD; local signoff requires a branch with a push remote." >&2
    return 1
  }

  local push_ref
  push_ref="$(git rev-parse --abbrev-ref --symbolic-full-name "@{push}" 2>/dev/null)" || {
    echo "Error: branch '$branch' has no push remote configured." >&2
    echo "Push the branch first, then rerun local signoff." >&2
    return 1
  }

  local head_sha push_sha
  head_sha="$(git rev-parse HEAD)"
  push_sha="$(git rev-parse "@{push}")"
  if [[ "$head_sha" != "$push_sha" ]]; then
    echo "Error: HEAD is not pushed to $push_ref." >&2
    echo "Local signoff posts a GitHub status for HEAD, so push this exact commit first." >&2
    return 1
  fi
}

ensure_gh_signoff() {
  if ! command -v gh >/dev/null 2>&1; then
    echo "Error: gh is required for GitHub signoff." >&2
    return 1
  fi

  if gh extension list 2>/dev/null | grep -Eq '(^|[[:space:]])basecamp/gh-signoff([[:space:]]|$)'; then
    return 0
  fi

  if [[ "$auto_install" != true ]]; then
    echo "Error: gh-signoff extension is missing." >&2
    echo "Install with: gh extension install basecamp/gh-signoff" >&2
    return 1
  fi

  echo "Installing gh-signoff extension (advisory statuses only; no branch protection changes)." >&2
  gh extension install basecamp/gh-signoff
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      full=true
      shift
      ;;
    --context)
      if [[ $# -lt 2 ]]; then
        echo "Error: --context requires a value." >&2
        exit 2
      fi
      contexts+=("$(normalize_context "$2")")
      shift 2
      ;;
    --no-setup)
      run_setup=false
      shift
      ;;
    --no-auto-install)
      auto_install=false
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      phases+=("$@")
      break
      ;;
    -*)
      echo "Error: unknown option '$1'." >&2
      show_help >&2
      exit 2
      ;;
    *)
      phases+=("$1")
      shift
      ;;
  esac
done

if [[ "$full" == true ]]; then
  if [[ ${#phases[@]} -gt 0 ]]; then
    echo "Error: --full cannot be combined with explicit phases." >&2
    exit 2
  fi
  mapfile -t phases <("$SCRIPT_DIR/run_ci_local.sh" --list-phases)
  if [[ ${#contexts[@]} -eq 0 ]]; then
    contexts=("local-pr-ready")
  fi
elif [[ ${#phases[@]} -eq 0 ]]; then
  phases=("lint" "test")
fi

if [[ ${#contexts[@]} -eq 0 ]]; then
  for phase in "${phases[@]}"; do
    contexts+=("local-$(normalize_context "$phase")")
  done
fi

require_clean_pushed_head
ensure_gh_signoff

run_args=()
if [[ "$run_setup" != true ]]; then
  run_args+=(--no-setup)
fi
run_args+=("${phases[@]}")

head_before="$(git rev-parse HEAD)"
echo "Running local validation for ${head_before}: scripts/dev/run_ci_local.sh ${run_args[*]}" >&2
"$SCRIPT_DIR/run_ci_local.sh" "${run_args[@]}"

require_clean_pushed_head
head_after="$(git rev-parse HEAD)"
if [[ "$head_before" != "$head_after" ]]; then
  echo "Error: HEAD changed during validation; refusing to sign." >&2
  exit 1
fi

echo "Validation passed for ${head_after}." >&2
echo "Signoff contexts: ${contexts[*]}" >&2
if [[ "$dry_run" == true ]]; then
  echo "Dry run: gh signoff ${contexts[*]}" >&2
  exit 0
fi

gh signoff "${contexts[@]}"
