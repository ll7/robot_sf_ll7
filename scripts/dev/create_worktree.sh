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
  --mode MODE              Worktree mode: implementation (default) or review.
  --minimum-free-bytes N   Override ROBOT_SF_WORKTREE_MIN_FREE_BYTES.
  --receipt PATH            Write a delegated-worker receipt after creation.
  --task-id ID              Acquire an active-worktree lease for this task.
  --dry-run                Run the preflight without invoking Git.
  --exec COMMAND [ARG...]  Run an explicit command from inside the new worktree.
  -h, --help               Show this help and exit.

The default threshold is 2 GiB (ROBOT_SF_WORKTREE_MIN_FREE_BYTES).  Supplying
--task-id creates a path-scoped ownership lease before the repository worktree
mutation lock is released, so repository-owned cleanup cannot race the task
claim.  --receipt remains optional; when supplied it requires --task-id and
also writes the immutable delegated-worker identity receipt.

After creation, targeted validation should use the main checkout's shared environment:

  scripts/dev/run_worktree_shared_venv.sh -- <command>

Use scripts/dev/bootstrap_worktree.sh only when a worktree-local environment is
explicitly required.  For reclaim guidance, run:

  scripts/dev/check_worktree_capacity.py --inventory --json

When --exec is supplied, the command is launched in the created worktree even
though this script itself may have been invoked from another checkout.  The
worktree is left in place when the command fails so its diagnostics remain
available for inspection.

For --mode review, --exec is launched through the review guard's Linux
Landlock process boundary and fails closed when that boundary is unavailable.
Commands started later must remain descendants of that process to retain the
boundary; use the guard's `run -- ... bash` form for a bounded session.
EOF
}

worktree_path=""
branch_name=""
base_ref="origin/main"
worktree_mode="implementation"
minimum_free_bytes="${ROBOT_SF_WORKTREE_MIN_FREE_BYTES:-}"
receipt_path=""
task_id=""
dry_run=0
command_args=()
# Internal re-entry flag: the portable-lock fallback re-executes this script
# under worktree_creation_lock.py so the critical section runs while a Python
# fcntl holder owns the shared lock file. The inherited lock descriptor is
# validated below; never pass this flag directly.
locked_transaction=0
created_worktree=0
created_branch_sha=""

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
    --mode)
      [[ $# -ge 2 ]] || { echo "--mode requires a value" >&2; exit 2; }
      worktree_mode="$2"
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
    --__locked-transaction)
      locked_transaction=1
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

if [[ "$worktree_mode" != "implementation" && "$worktree_mode" != "review" ]]; then
  echo "--mode must be implementation or review" >&2
  exit 2
fi

if [[ -n "$receipt_path" && -z "$task_id" ]]; then
  echo "--receipt requires --task-id" >&2
  exit 2
fi

if [[ -n "$receipt_path" ]]; then
  receipt_path="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$receipt_path")"
fi

validate_target_preflight() {
  # Validate the target and capacity immediately before a creation mutation.
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
}

if [[ "$locked_transaction" -eq 1 ]]; then
  git_common_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
  worktree_lock_path="$git_common_dir/robot-sf-create-worktree.lock"
  lock_fd="${ROBOT_SF_WORKTREE_LOCK_FD:-}"
  if ! [[ "$lock_fd" =~ ^[0-9]+$ ]]; then
    echo "create_worktree: --__locked-transaction is an internal mode" >&2
    echo "create_worktree: it requires the portable helper's inherited repository lock" >&2
    exit 2
  fi
  if ! python3 "$SCRIPT_DIR/worktree_creation_lock.py" --verify-fd "$worktree_lock_path" "$lock_fd"; then
    echo "create_worktree: --__locked-transaction requires ownership of the repository lock" >&2
    exit 2
  fi
fi

if [[ "$dry_run" -eq 1 ]]; then
  validate_target_preflight
  echo "create_worktree: dry-run passed; git worktree add was not invoked."
  exit 0
fi

git_common_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
worktree_lock_path="$git_common_dir/robot-sf-create-worktree.lock"

# Git derives linked-worktree administrative directory names from the target
# basename. Independent callers with distinct full paths but the same basename
# can therefore race while Git allocates (or prunes) entries under the shared
# common directory. Serialize the complete orphan-recovery/prune/add/lease
# transaction per repository; target and capacity validation run inside that
# transaction so the admission decision matches the mutation it protects.
report_and_exec() {
  echo "create_worktree: created $worktree_path on branch $branch_name from $base_ref"
  if [[ -n "$task_id" ]]; then
    echo "create_worktree: active lease owner/task: $task_id"
    echo "create_worktree: heartbeat with scripts/dev/pr_gate_lease.py heartbeat --worktree '$worktree_path' --extend-hours 2"
    echo "create_worktree: release before teardown with scripts/dev/pr_gate_lease.py release --worktree '$worktree_path'"
  fi
  echo "create_worktree: use scripts/dev/run_worktree_shared_venv.sh for targeted validation."

  if [[ "${#command_args[@]}" -gt 0 ]]; then
    echo "create_worktree: executing supplied command in $worktree_path"
    (
      cd -- "$worktree_path"
      if [[ -n "$receipt_path" ]]; then
        python3 "$SCRIPT_DIR/worktree_receipt.py" check --receipt "$receipt_path" --worktree . --json
      fi
      if [[ "$worktree_mode" == "review" ]]; then
        # Bind the optional first command to the real process boundary. Later
        # commands must remain descendants of this process to retain it.
        exec python3 "$SCRIPT_DIR/review_worktree_guard.py" run \
          --worktree "$worktree_path" -- "${command_args[@]}"
      fi
      exec "${command_args[@]}"
    )
  fi
}

lease_file_for_worktree() {
  python3 - "$worktree_path" "$git_common_dir" <<'PY'
import hashlib
import sys
from pathlib import Path

worktree_path = Path(sys.argv[1]).resolve()
common_dir = Path(sys.argv[2])
digest = hashlib.sha256(str(worktree_path).encode("utf-8")).hexdigest()
print(common_dir / f".pr-gate-lease-{digest}.json")
PY
}

release_task_lease() {
  if [[ -z "$task_id" ]]; then
    return 0
  fi

  if python3 "$SCRIPT_DIR/pr_gate_lease.py" release --worktree "$worktree_path" >/dev/null; then
    return 0
  fi

  # The normal release path is serialized by the inherited lock. If it cannot
  # run, remove only this path-scoped lease file as a last-resort rollback so a
  # failed creation cannot leave an active lease for a deleted worktree.
  local lease_file
  if ! lease_file="$(lease_file_for_worktree)"; then
    return 1
  fi
  rm -f -- "$lease_file"
}

cleanup_failed_creation() {
  local failure_rc="$1"
  local cleanup_failed=0

  if [[ "$created_worktree" -ne 1 ]]; then
    return "$failure_rc"
  fi

  if ! release_task_lease; then
    echo "create_worktree: failed to remove the task lease during rollback" >&2
    cleanup_failed=1
  fi

  if [[ -e "$worktree_path" || -L "$worktree_path" ]]; then
    if ! git worktree remove --force "$worktree_path"; then
      echo "create_worktree: failed to remove worktree during rollback: $worktree_path" >&2
      cleanup_failed=1
    fi
  fi
  if ! git worktree prune; then
    echo "create_worktree: failed to prune worktree metadata during rollback" >&2
    cleanup_failed=1
  fi

  if git show-ref --verify --quiet "refs/heads/$branch_name"; then
    local current_branch_sha
    current_branch_sha="$(git rev-parse --verify "$branch_name^{commit}" 2>/dev/null || true)"
    if [[ -n "$created_branch_sha" && "$current_branch_sha" == "$created_branch_sha" ]] &&
       ! git worktree list --porcelain | grep -q "^branch refs/heads/$branch_name$"; then
      if ! git branch -D "$branch_name"; then
        echo "create_worktree: failed to remove created branch during rollback: $branch_name" >&2
        cleanup_failed=1
      fi
    else
      echo "create_worktree: preserved branch during rollback: $branch_name" >&2
      cleanup_failed=1
    fi
  fi

  if [[ "$cleanup_failed" -ne 0 ]]; then
    echo "create_worktree: rollback was incomplete after creation failure" >&2
  fi
  return "$failure_rc"
}

run_locked_transaction() {
  # Capacity and parent checks must share the same lock as branch cleanup and
  # worktree registration. This applies to both the flock CLI and portable
  # Python backends.
  validate_target_preflight

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
  if git worktree add --no-track -b "$branch_name" "$worktree_path" "$base_ref"; then
    created_worktree=1
    created_branch_sha="$(git rev-parse --verify "$branch_name^{commit}")"
  else
    local worktree_add_rc=$?
    if ! cleanup_failed_creation "$worktree_add_rc"; then
      :
    fi
    return "$worktree_add_rc"
  fi
  if [[ "$worktree_mode" == "review" ]]; then
    review_guard_args=(--worktree "$worktree_path" --mode review)
    # A review candidate may be created from a base that predates this guard.
    # Keep the target clean by using the invoking checkout's tracked helper and
    # hook until the guard itself is present in the selected base.
    if [[ ! -f "$worktree_path/scripts/dev/review_worktree_guard.py" ||
          ! -x "$worktree_path/scripts/dev/git_hooks/pre-push" ]]; then
      review_guard_args+=(--hook-source-root "$SCRIPT_DIR")
    fi
    if python3 "$SCRIPT_DIR/review_worktree_guard.py" configure "${review_guard_args[@]}"; then
      :
    else
      local review_guard_rc=$?
      if ! cleanup_failed_creation "$review_guard_rc"; then
        :
      fi
      return "$review_guard_rc"
    fi
  fi
  if [[ -n "$task_id" ]]; then
    # The lease helper reuses the inherited repository lock. Creating the lease
    # before this transaction releases the lock closes the add->claim cleanup gap.
    if python3 "$SCRIPT_DIR/pr_gate_lease.py" create \
      --worktree "$worktree_path" --gate-id "$task_id" --owner "$task_id"; then
      :
    else
      local lease_create_rc=$?
      echo "create_worktree: task-id lease handoff failed; rolling back creation" >&2
      if ! cleanup_failed_creation "$lease_create_rc"; then
        :
      fi
      return "$lease_create_rc"
    fi
    if python3 "$SCRIPT_DIR/pr_gate_lease.py" is-active --worktree "$worktree_path"; then
      :
    else
      local lease_observation_rc=$?
      echo "create_worktree: task-id lease was not observable after creation; rolling back" >&2
      if ! cleanup_failed_creation "$lease_observation_rc"; then
        :
      fi
      return "$lease_observation_rc"
    fi
  fi
  if [[ -n "$receipt_path" ]]; then
    if python3 "$SCRIPT_DIR/worktree_receipt.py" create \
      --worktree "$worktree_path" --task-id "$task_id" --base-ref "$base_ref" --output "$receipt_path"; then
      :
    else
      local receipt_rc=$?
      echo "create_worktree: receipt handoff failed; rolling back creation" >&2
      if ! cleanup_failed_creation "$receipt_rc"; then
        :
      fi
      return "$receipt_rc"
    fi
  fi
}

if [[ "$locked_transaction" -eq 1 ]]; then
  # Re-entered under worktree_creation_lock.py holding the shared lock file.
  # The outer invocation reports and runs --exec only after this process exits,
  # so arbitrary commands never run while the repository lock is held.
  run_locked_transaction
  exit 0
fi

use_python_lock=0
if [[ -n "${ROBOT_SF_WORKTREE_FORCE_PYTHON_LOCK:-}" ]]; then
  use_python_lock=1
elif ! command -v flock >/dev/null 2>&1; then
  use_python_lock=1
fi

if [[ "$use_python_lock" -eq 0 ]]; then
  exec {worktree_lock_fd}>"$worktree_lock_path"
  if ! flock "$worktree_lock_fd"; then
    echo "create_worktree: failed to acquire repository worktree-creation lock" >&2
    exit 2
  fi
  # Expose the already-held descriptor so the lease helper can verify and reuse
  # this same lock identity instead of opening a second descriptor and deadlocking.
  export ROBOT_SF_WORKTREE_LOCK_FD="$worktree_lock_fd"
  run_locked_transaction
  unset ROBOT_SF_WORKTREE_LOCK_FD
  flock -u "$worktree_lock_fd"
  exec {worktree_lock_fd}>&-
else
  # Portable fallback: fcntl.flock via the helper uses flock(2) on the same
  # lock file identity, so it serializes against flock-CLI holders.
  echo "create_worktree: flock CLI not used; holding portable lock on $worktree_lock_path" >&2
  locked_args=(--__locked-transaction --path "$worktree_path" --branch "$branch_name"
    --base "$base_ref" --mode "$worktree_mode")
  if [[ -n "$minimum_free_bytes" ]]; then
    locked_args+=(--minimum-free-bytes "$minimum_free_bytes")
  fi
  if [[ -n "$task_id" ]]; then
    locked_args+=(--task-id "$task_id")
  fi
  if [[ -n "$receipt_path" ]]; then
    locked_args+=(--receipt "$receipt_path")
  fi
  python_lock_rc=0
  python3 "$SCRIPT_DIR/worktree_creation_lock.py" "$worktree_lock_path" -- \
    "$SCRIPT_DIR/create_worktree.sh" "${locked_args[@]}" || python_lock_rc=$?
  if [[ "$python_lock_rc" -ne 0 ]]; then
    exit "$python_lock_rc"
  fi
fi
report_and_exec
