#!/usr/bin/env bash
# Drop-in for `gh issue view` that transparently bypasses the deprecated
# repository.issue.projectCards GraphQL field (issue #5188).
#
# Usage:
#     scripts/dev/gh_issue_view.sh <number> [--repo OWNER/REPO]
#
# Delegates to `gh_issue_rest.py thread`, which tries native `gh issue view`
# first and falls back to paginated REST only when the projectCards error occurs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: $0 <number> [--repo OWNER/REPO]" >&2
  exit 2
}

REPO=""
NUMBER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      [[ $# -ge 2 ]] || usage
      REPO="$2"
      shift 2
      ;;
    -*)
      echo "Unexpected option: $1" >&2
      usage
      ;;
    *)
      [[ -z "$NUMBER" ]] || usage
      NUMBER="$1"
      shift
      ;;
  esac
done

[[ -n "$NUMBER" ]] || usage

exec uv run python "${SCRIPT_DIR}/gh_issue_rest.py" thread "$NUMBER" \
  ${REPO:+--repo "$REPO"}