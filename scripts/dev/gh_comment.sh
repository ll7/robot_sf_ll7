#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/dev/gh_comment.sh pr <number> [--repo <owner/repo>] [--body-file <path>]
  scripts/dev/gh_comment.sh issue <number> [--repo <owner/repo>] [--body-file <path>]
  scripts/dev/gh_comment.sh pr --current [--repo <owner/repo>] [--body-file <path>]

Notes:
  - If --body-file is omitted, comment body is read from stdin.
  - Prefer heredoc stdin for multiline comments to avoid literal "\n" escapes.
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

target_type="$1"
shift

if [ "$target_type" != "pr" ] && [ "$target_type" != "issue" ]; then
  echo "Error: target must be 'pr' or 'issue'." >&2
  usage
  exit 2
fi

target_id=""
use_current_pr=false
repo_arg=""
body_file=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo)
      if [ "$#" -lt 2 ]; then
        echo "Error: --repo requires a value." >&2
        exit 2
      fi
      repo_arg="$2"
      shift 2
      ;;
    --body-file)
      if [ "$#" -lt 2 ]; then
        echo "Error: --body-file requires a path." >&2
        exit 2
      fi
      body_file="$2"
      shift 2
      ;;
    --current)
      use_current_pr=true
      shift
      ;;
    -*)
      echo "Error: unknown option '$1'." >&2
      usage
      exit 2
      ;;
    *)
      if [ -n "$target_id" ]; then
        echo "Error: unexpected extra argument '$1'." >&2
        usage
        exit 2
      fi
      target_id="$1"
      shift
      ;;
  esac
done

if [ "$target_type" = "issue" ] && [ "$use_current_pr" = true ]; then
  echo "Error: --current is only supported for 'pr' target type." >&2
  exit 2
fi

if [ "$use_current_pr" = true ]; then
  repo_flag=()
  if [ -n "$repo_arg" ]; then
    repo_flag=(--repo "$repo_arg")
  fi
  target_id="$(gh pr view "${repo_flag[@]}" --json number --jq .number)"
fi

if [ -z "$target_id" ]; then
  echo "Error: missing target number. Provide <number> or use --current for PRs." >&2
  usage
  exit 2
fi

if [ -z "$body_file" ]; then
  body_file="$(mktemp)"
  trap 'rm -f "$body_file"' EXIT
  cat >"$body_file"
fi

if [ ! -f "$body_file" ]; then
  echo "Error: body file '$body_file' does not exist." >&2
  exit 2
fi

if [ ! -s "$body_file" ]; then
  echo "Error: comment body is empty." >&2
  exit 2
fi

repo_flag=()
if [ -n "$repo_arg" ]; then
  repo_flag=(--repo "$repo_arg")
fi

if [ "$target_type" = "pr" ]; then
  gh pr comment "$target_id" "${repo_flag[@]}" --body-file "$body_file"
else
  gh issue comment "$target_id" "${repo_flag[@]}" --body-file "$body_file"
fi
