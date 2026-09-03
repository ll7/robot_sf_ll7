#!/usr/bin/env bash
# Create a linked worktree only after the target filesystem passes capacity preflight.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

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
  --receipt PATH            Write a delegated-worker receipt after creation.
  --task-id ID              Task identifier for --receipt (delegated mode).
  --dry-run                Run the preflight without invoking Git.
  --exec COMMAND [ARG...]  Run an explicit command from inside the new worktree.
  -h, --help               Show this help and exit.

The default threshold is 2 GiB (ROBOT_SF_WORKTREE_MIN_FREE_BYTES).  After
creation, targeted validation should use the main checkout's shared environment:

  scripts/dev/run_worktree_shared_venv.sh -- <command>

Use scripts/dev/bootstrap_worktree.sh only when a worktree-local environment is
explicitly required.  For reclaim guidance, run:

  scripts/dev/check_worktree_capacity.py --inventory --json

When --exec is supplied, the command is launched in the created worktree even
though this script itself may have been invoked from another checkout.  The
worktree is left in place when the command fails so its diagnostics remain
available for inspection.
EOF
}

worktree_path=""
branch_name=""
base_ref="origin/main"
minimum_free_bytes="${ROBOT_SF_WORKTREE_MIN_FREE_BYTES:-}"
receipt_path=""
task_id=""
dry_run=0
command_args=()

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
    --receipt)
      [[ $# -ge 2 ]] || { echo "--receipt requires a value" >&2; exit 2; }
      receipt_path="$2"
      shift 2
      ;;
    --task-id)
      [[ $# -ge 2 ]] || { echo "--task-id requires a value" >&2; exit 2; }
      task_id="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --exec)
      shift
      if [[ $# -eq 0 ]]; then
        echo "--exec requires a command" >&2
        exit 2
      fi
      command_args=("$@")
      break
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

if [[ -n "$task_id" && -z "$receipt_path" ]] || [[ -n "$receipt_path" && -z "$task_id" ]]; then
  echo "--receipt and --task-id must be supplied together" >&2
  exit 2
fi

if [[ -n "$receipt_path" ]]; then
  receipt_path="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$receipt_path")"
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

# Git derives linked-worktree administrative directory names from the target
# basename. Independent callers with distinct full paths but the same basename
# can therefore race while Git allocates (or prunes) entries under the shared
# common directory. Serialize the complete orphan-recovery/prune/add
# transaction per repository; capacity inspection above remains parallel and
# read-only.
if ! command -v flock >/dev/null 2>&1; then
  echo "create_worktree: flock is required for concurrency-safe worktree creation" >&2
  exit 2
fi
git_common_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
worktree_lock_path="$git_common_dir/robot-sf-create-worktree.lock"
exec {worktree_lock_fd}>"$worktree_lock_path"
if ! flock "$worktree_lock_fd"; then
  echo "create_worktree: failed to acquire repository worktree-creation lock" >&2
  exit 2
fi

# A concurrent creator may have populated this exact target while this process
# waited for the repository lock. Recheck under the lock before any mutation.
if [[ -e "$worktree_path" || -L "$worktree_path" ]]; then
  echo "refusing to overwrite existing worktree target: $worktree_path" >&2
  exit 2
fi

# Recover from a prior interrupted checkout: git can die mid-"Updating files"
# (e.g. SIGPIPE when output is piped through head), leaving the branch ref
# present without a registered worktree. The next retry would otherwise fail
# with a bare "fatal: a branch named '<branch>' already exists".
if git show-ref --verify --quiet "refs/heads/$branch_name"; then
  if ! git worktree list --porcelain | grep -q "^branch refs/heads/$branch_name$"; then
    if git rev-parse --verify --quiet "$branch_name^{commit}" >/dev/null &&
       git merge-base --is-ancestor "$branch_name" "$base_ref" >/dev/null 2>&1; then
      echo "create_worktree: removing orphan branch '$branch_name' (points at base $base_ref)" >&2
      git branch -D "$branch_name"
    else
      echo "create_worktree: orphan branch '$branch_name' does not point at base $base_ref;" >&2
      echo "create_worktree: recover manually with:" >&2
      echo "  git branch -D $branch_name && git worktree prune" >&2
      exit 2
    fi
  fi
fi
git worktree prune

# Avoid automatic upstream-tracking writes to the shared repository config.  A
# linked worktree's branch can be configured explicitly later with
# ``git branch --set-upstream-to``; creation itself must remain safe when
# several workers create worktrees concurrently.
git worktree add --no-track -b "$branch_name" "$worktree_path" "$base_ref"
if [[ -n "$receipt_path" ]]; then
  python3 "$SCRIPT_DIR/worktree_receipt.py" create \
    --worktree "$worktree_path" --task-id "$task_id" --base-ref "$base_ref" --output "$receipt_path"
fi
flock -u "$worktree_lock_fd"
exec {worktree_lock_fd}>&-
echo "create_worktree: created $worktree_path on branch $branch_name from $base_ref"
echo "create_worktree: use scripts/dev/run_worktree_shared_venv.sh for targeted validation."

if [[ "${#command_args[@]}" -gt 0 ]]; then
  echo "create_worktree: executing supplied command in $worktree_path"
  (
    cd -- "$worktree_path"
    if [[ -n "$receipt_path" ]]; then
      python3 "$SCRIPT_DIR/worktree_receipt.py" check --receipt "$receipt_path" --worktree . --json
    fi
    exec "${command_args[@]}"
  )
fi
