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
  - Both PR and issue comments use the REST issue-comments endpoint
    (POST repos/<owner>/<repo>/issues/<number>/comments) with REST target
    validation, so publication is independent of GraphQL comment quotas.
EOF
}

if [ "$#" -gt 0 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
  usage
  exit 0
fi

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
    --help|-h)
      usage
      exit 0
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
  branch_name="$(git branch --show-current)"
  if [ -z "$branch_name" ]; then
    echo "Error: could not resolve the current branch for --current." >&2
    exit 1
  fi
  # Review leases use a local branch name distinct from the PR source branch
  # while tracking that source branch. Query the tracked branch when available
  # so --current still finds the PR from an isolated review worktree.
  if upstream_ref="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null)"; then
    branch_name="${upstream_ref#*/}"
  fi
  api_repo="{owner}/{repo}"
  head_owner="{owner}"
  if [ -n "$repo_arg" ]; then
    api_repo="$repo_arg"
    head_owner="${repo_arg%%/*}"
  fi
  if ! target_id="$(gh api "repos/$api_repo/pulls?state=open&head=$head_owner:$branch_name&per_page=100" --jq '.[0].number // empty')"; then
    echo "Error: could not resolve an open PR for branch '$branch_name'." >&2
    exit 1
  fi
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

if [ "$target_type" = "pr" ]; then
  api_repo="{owner}/{repo}"
  if [ -n "$repo_arg" ]; then
    api_repo="$repo_arg"
  fi
  if ! gh api "repos/$api_repo/pulls/$target_id" --silent; then
    echo "Error: PR '$target_id' could not be resolved through the REST API." >&2
    exit 1
  fi
  # Use gh's silent mode so a successful POST with an empty/malformed response
  # body cannot surface as a client-side JSON parse failure (issue #6891).
  gh_api_rc=0
  gh api --method POST "repos/$api_repo/issues/$target_id/comments" \
    --silent -F "body=@$body_file" || gh_api_rc=$?
  exit "$gh_api_rc"
else
  api_repo="{owner}/{repo}"
  if [ -n "$repo_arg" ]; then
    api_repo="$repo_arg"
  fi
  # Mirror the PR path: validate the target through REST before publication,
  # then post through the REST issue-comments endpoint. This keeps the issue
  # path independent of the GraphQL-backed ``gh issue comment`` command, which
  # fails under exhausted GraphQL comment quota even when REST is available.
  if ! gh api "repos/$api_repo/issues/$target_id" --silent; then
    echo "Error: Issue '$target_id' could not be resolved through the REST API." >&2
    exit 1
  fi
  # Use gh's silent mode so a successful POST with an empty/malformed response
  # body cannot surface as a client-side JSON parse failure (issue #6891).
  gh_api_rc=0
  gh api --method POST "repos/$api_repo/issues/$target_id/comments" \
    --silent -F "body=@$body_file" || gh_api_rc=$?
  exit "$gh_api_rc"
fi
