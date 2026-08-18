#!/usr/bin/env bash
# Start a public worktree only after the repository-wide WIP admission check.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/dev/start_worktree.sh --issue NUMBER --path PATH --branch BRANCH [options]

The WIP check runs before `git worktree add`, so a full or unknown queue cannot
create another checkout or environment. It is read-only; the final command is
the ordinary Git worktree creation operation.

Options:
  --issue NUMBER       Owning issue for the proposed lane (required).
  --path PATH          New worktree path (required).
  --branch BRANCH      New branch name (required).
  --source-ref REF    Commit/ref for the worktree (default: origin/main).
  --repo OWNER/REPO   GitHub repository (default: ll7/robot_sf_ll7).
  --remote NAME       Git remote used for claim evidence (default: origin).
  --mode MODE         policy, enforce, or report-only (default: policy).
  --label LABEL       Proposed issue label; repeatable for offline exemptions.
  -h, --help           Show this help.
EOF
}

issue=""
worktree_path=""
branch=""
source_ref="origin/main"
repo="ll7/robot_sf_ll7"
remote="origin"
mode="policy"
labels=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --issue)
      [[ $# -ge 2 ]] || { echo "--issue requires a number" >&2; exit 2; }
      issue="$2"
      shift 2
      ;;
    --path)
      [[ $# -ge 2 ]] || { echo "--path requires a value" >&2; exit 2; }
      worktree_path="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || { echo "--branch requires a value" >&2; exit 2; }
      branch="$2"
      shift 2
      ;;
    --source-ref)
      [[ $# -ge 2 ]] || { echo "--source-ref requires a ref" >&2; exit 2; }
      source_ref="$2"
      shift 2
      ;;
    --repo)
      [[ $# -ge 2 ]] || { echo "--repo requires OWNER/REPO" >&2; exit 2; }
      repo="$2"
      shift 2
      ;;
    --remote)
      [[ $# -ge 2 ]] || { echo "--remote requires a name" >&2; exit 2; }
      remote="$2"
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || { echo "--mode requires policy, enforce, or report-only" >&2; exit 2; }
      mode="$2"
      shift 2
      ;;
    --label)
      [[ $# -ge 2 ]] || { echo "--label requires a name" >&2; exit 2; }
      labels+=("$2")
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$issue" =~ ^[1-9][0-9]*$ || -z "$worktree_path" || -z "$branch" ]]; then
  echo "--issue, --path, and --branch are required" >&2
  usage >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
preflight=(
  "${PYTHON:-python3}"
  "$repo_root/scripts/dev/wip_capacity.py"
  --repo "$repo"
  --remote "$remote"
  --mode "$mode"
  --proposed-issue "$issue"
  --json
)
for label in "${labels[@]}"; do
  preflight+=(--proposed-label "$label")
done

echo "start_worktree: evaluating WIP capacity before creating $worktree_path ..." >&2
"${preflight[@]}"

git -C "$repo_root" worktree add -b "$branch" "$worktree_path" "$source_ref"
