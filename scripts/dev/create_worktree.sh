#!/usr/bin/env bash
# Create a linked worktree only after the target filesystem passes capacity preflight.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
  cat <<'EOF'
Usage: scripts/dev/create_worktree.sh --path PATH --branch BRANCH [options]

Run a read-only capacity check before `git worktree add`.  A failed check exits
before Git creates or partially populates the target directory.

Options:
  --path PATH              New linked-worktree path (its parent must exist).
  --branch BRANCH          New branch name.
  --base REF               Base ref; defaults to origin/main.
  --minimum-free-bytes N   Override ROBOT_SF_WORKTREE_MIN_FREE_BYTES.
  --dry-run                Run the preflight without invoking Git.
  -h, --help               Show this help and exit.

The default threshold is 2 GiB (ROBOT_SF_WORKTREE_MIN_FREE_BYTES).  After
creation, targeted validation should use the main checkout's shared environment:

  scripts/dev/run_worktree_shared_venv.sh -- <command>

Use scripts/dev/bootstrap_worktree.sh only when a worktree-local environment is
explicitly required.  For reclaim guidance, run:

  scripts/dev/check_worktree_capacity.py --inventory --json
EOF
}

worktree_path=""
branch_name=""
base_ref="origin/main"
minimum_free_bytes="${ROBOT_SF_WORKTREE_MIN_FREE_BYTES:-}"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path)
      [[ $# -ge 2 ]] || { echo "--path requires a value" >&2; exit 2; }
      worktree_path="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || { echo "--branch requires a value" >&2; exit 2; }
      branch_name="$2"
      shift 2
      ;;
    --base)
      [[ $# -ge 2 ]] || { echo "--base requires a value" >&2; exit 2; }
      base_ref="$2"
      shift 2
      ;;
    --minimum-free-bytes)
      [[ $# -ge 2 ]] || { echo "--minimum-free-bytes requires a value" >&2; exit 2; }
      minimum_free_bytes="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      show_help >&2
      exit 2
      ;;
  esac
done

if [[ -z "$worktree_path" || -z "$branch_name" ]]; then
  echo "--path and --branch are required" >&2
  show_help >&2
  exit 2
fi

if [[ -e "$worktree_path" || -L "$worktree_path" ]]; then
  echo "refusing to overwrite existing worktree target: $worktree_path" >&2
  exit 2
fi

target_parent="$(dirname -- "$worktree_path")"
if [[ ! -d "$target_parent" || ! -w "$target_parent" ]]; then
  echo "worktree target parent must already exist and be writable: $target_parent" >&2
  echo "Create or choose the parent directory, then rerun this command." >&2
  exit 2
fi

capacity_args=(--path "$worktree_path")
if [[ -n "$minimum_free_bytes" ]]; then
  capacity_args+=(--minimum-free-bytes "$minimum_free_bytes")
fi
python3 "$SCRIPT_DIR/check_worktree_capacity.py" "${capacity_args[@]}"

if [[ "$dry_run" -eq 1 ]]; then
  echo "create_worktree: dry-run passed; git worktree add was not invoked."
  exit 0
fi

git worktree add -b "$branch_name" "$worktree_path" "$base_ref"
echo "create_worktree: created $worktree_path on branch $branch_name from $base_ref"
echo "create_worktree: use scripts/dev/run_worktree_shared_venv.sh for targeted validation."
