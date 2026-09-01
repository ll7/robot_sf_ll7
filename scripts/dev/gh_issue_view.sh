#!/usr/bin/env bash
# Drop-in for `gh issue view` that transparently bypasses the deprecated
# repository.issue.projectCards GraphQL field (issue #5188).
#
# Usage:
#     scripts/dev/gh_issue_view.sh <number> [--repo OWNER/REPO] [--comments]
#
# Without --comments, delegates to `gh_issue_rest.py thread`, which tries native
# `gh issue view` first and falls back to paginated REST only when the
# projectCards error occurs.
#
# With --comments, delegates to `gh_issue_rest.py view --plain --comments`
# which always uses the REST fallback to include comments in the output.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: $0 <number> [--repo OWNER/REPO] [--comments]" >&2
  exit 2
}

REPO=""
NUMBER=""
COMMENTS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      [[ $# -ge 2 ]] || usage
      REPO="$2"
      shift 2
      ;;
    --comments)
      COMMENTS=true
      shift
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

if ! python3 "$SCRIPT_DIR/github_transport_policy.py" check \
  --helper "$SCRIPT_DIR/gh_issue_view.sh" --root "$SCRIPT_DIR/../.." --json >/dev/null; then
  echo "Error: gh_issue_view.sh is not admitted by the GitHub transport policy." >&2
  exit 2
fi

ARGS=()
if [[ -n "$REPO" ]]; then
  ARGS+=("--repo" "$REPO")
fi

if [[ "$COMMENTS" == "true" ]]; then
  exec uv run python "${SCRIPT_DIR}/gh_issue_rest.py" view "$NUMBER" --plain --comments "${ARGS[@]}"
else
  exec uv run python "${SCRIPT_DIR}/gh_issue_rest.py" thread "$NUMBER" "${ARGS[@]}"
fi
