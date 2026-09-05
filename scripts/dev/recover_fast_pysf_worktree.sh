#!/usr/bin/env bash
# Explicitly create or refresh a worktree-owned environment after the shared
# fast-pysf freshness gate reports a stale installed package.
#
# This helper never refreshes the owning checkout.  It only mutates the
# current linked worktree's ignored .venv, after a repository-scoped lock and
# capacity check, then verifies the installed package before returning.

set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/recover_fast_pysf_worktree.sh

Create or refresh the current linked worktree's .venv with the pinned fast-pysf
package and verify package freshness before a caller runs project code.

This is an explicit recovery operation. It refuses the main checkout, refuses
dirty dependency inputs, serializes recovery per repository with a kernel-backed
lock, and fails closed when the worktree filesystem is below the
ROBOT_SF_WORKTREE_MIN_FREE_BYTES threshold (default: 2 GiB).

The helper is normally invoked through:
  scripts/dev/run_worktree_shared_venv.sh --recover-stale-fast-pysf -- <command>

It uses a worktree-local environment and runs:
  uv sync --all-extras --reinstall-package robot-sf --frozen

The --frozen flag prevents this recovery path from changing dependency locks.
EOF
}

if [[ "$#" -gt 0 ]]; then
  if [[ "$#" -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    show_help
    exit 0
  fi
  echo "recover_fast_pysf_worktree: this helper accepts no arguments" >&2
  show_help >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "recover_fast_pysf_worktree: current directory is not a Git worktree" >&2
  exit 2
}
repo_root="$(cd -- "$repo_root" && pwd -P)" || {
  echo "recover_fast_pysf_worktree: could not resolve the current worktree path" >&2
  exit 2
}
cd -- "$repo_root"

git_common_dir="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" || {
  echo "recover_fast_pysf_worktree: could not resolve the shared Git directory" >&2
  exit 2
}
git_common_dir="$(cd -- "$git_common_dir" && pwd -P)" || {
  echo "recover_fast_pysf_worktree: could not resolve the shared Git directory path" >&2
  exit 2
}
main_repo_root="$(cd -- "$git_common_dir/.." && pwd -P)" || {
  echo "recover_fast_pysf_worktree: could not resolve the owning checkout" >&2
  exit 2
}

# A linked worktree has a .git file and shares the main checkout's common Git
# directory.  Refusing the main checkout is the ownership boundary: all
# package installation stays under the current worktree's .venv.
if [[ "$repo_root" == "$main_repo_root" || "$git_common_dir" == "$repo_root/.git" ]]; then
  echo "recover_fast_pysf_worktree: refusing to mutate the main checkout" >&2
  echo "Run this explicit recovery from a linked worktree; the main .venv is never repaired implicitly." >&2
  exit 2
fi
if [[ ! -f "$repo_root/.git" ]]; then
  echo "recover_fast_pysf_worktree: current checkout is not a registered linked worktree: $repo_root" >&2
  exit 2
fi

checker="$repo_root/scripts/dev/check_fast_pysf_runtime.py"
capacity_checker="$repo_root/scripts/dev/check_worktree_capacity.py"
if [[ ! -f "$checker" || ! -f "$capacity_checker" ]]; then
  echo "recover_fast_pysf_worktree: required freshness or capacity checker is missing" >&2
  exit 2
fi

dependency_inputs=(
  pyproject.toml
  uv.lock
  fast-pysf/pyproject.toml
  fast-pysf/uv.lock
  third_party/python-rvo2
)
dirty_inputs="$(git status --porcelain=v1 -- "${dependency_inputs[@]}")" || {
  echo "recover_fast_pysf_worktree: could not inspect dependency inputs" >&2
  exit 2
}
if [[ -n "$dirty_inputs" ]]; then
  echo "recover_fast_pysf_worktree: refusing to sync dirty dependency inputs in $repo_root" >&2
  printf '%s\n' "$dirty_inputs" >&2
  echo "Commit or preserve the dependency-input changes, then retry the explicit recovery." >&2
  exit 2
fi

local_venv="$repo_root/.venv"
if [[ -L "$local_venv" ]]; then
  echo "recover_fast_pysf_worktree: refusing a symlinked worktree environment: $local_venv" >&2
  exit 2
fi
if [[ -e "$local_venv" && ! -d "$local_venv" ]]; then
  echo "recover_fast_pysf_worktree: worktree environment path is not a directory: $local_venv" >&2
  exit 2
fi

check_local_venv_layout() {
  if [[ ! -e "$local_venv" && ! -L "$local_venv" ]]; then
    return 0
  fi

  local resolved_root
  if ! resolved_root="$(python3 - "$local_venv" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=False))
PY
  )"; then
    echo "recover_fast_pysf_worktree: could not resolve worktree environment ownership: $local_venv" >&2
    return 1
  fi
  if [[ "$resolved_root" != "$local_venv" ]]; then
    echo "recover_fast_pysf_worktree: refusing an environment path that resolves outside the worktree: $local_venv" >&2
    return 1
  fi

  local component resolved_component
  for component in bin lib lib64; do
    if [[ ! -e "$local_venv/$component" && ! -L "$local_venv/$component" ]]; then
      continue
    fi
    if ! resolved_component="$(python3 - "$local_venv/$component" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=False))
PY
    )"; then
      echo "recover_fast_pysf_worktree: could not resolve environment component: $local_venv/$component" >&2
      return 1
    fi
    case "$resolved_component" in
      "$local_venv"/*) ;;
      *)
        echo "recover_fast_pysf_worktree: refusing environment component outside the worktree: $local_venv/$component" >&2
        return 1
        ;;
    esac
  done

  if [[ -e "$local_venv/bin/python" || -L "$local_venv/bin/python" ]]; then
    local resolved_python
    if ! resolved_python="$(python3 - "$local_venv/bin/python" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=False))
PY
    )"; then
      echo "recover_fast_pysf_worktree: could not resolve the worktree Python interpreter" >&2
      return 1
    fi
    case "$resolved_python" in
      "$main_repo_root"/*)
        echo "recover_fast_pysf_worktree: refusing a Python interpreter linked to the owning checkout: $local_venv/bin/python" >&2
        return 1
        ;;
    esac
  fi

  if ! python3 - "$local_venv" "$main_repo_root" <<'PY'
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


venv_root = Path(sys.argv[1]).resolve(strict=False)
main_root = Path(sys.argv[2]).resolve(strict=False)
host_interpreter_name = re.compile(r"python(?:3(?:\.\d+)?)?\Z")


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def fail(message: str) -> None:
    print(f"recover_fast_pysf_worktree: {message}", file=sys.stderr)
    raise SystemExit(1)


def on_walk_error(error: OSError) -> None:
    fail(f"could not inspect worktree environment links: {error}")


for raw_root, directories, files in os.walk(venv_root, followlinks=False, onerror=on_walk_error):
    for name in (*directories, *files):
        link = Path(raw_root) / name
        if not link.is_symlink():
            continue
        try:
            resolved = link.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            fail(f"could not resolve worktree environment link {link}: {error}")

        if is_within(resolved, venv_root):
            continue
        if link.parent == venv_root / "bin" and host_interpreter_name.fullmatch(link.name):
            if not resolved.is_file():
                fail(f"refusing a broken host-interpreter link: {link} -> {resolved}")
            if is_within(resolved, main_root):
                fail(
                    "refusing a host-interpreter link into the owning checkout: "
                    f"{link} -> {resolved}"
                )
            continue
        fail(
            "refusing an environment symlink outside the worktree-local .venv: "
            f"{link} -> {resolved}"
        )
PY
  then
    return 1
  fi
}

if ! check_local_venv_layout; then
  exit 2
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "recover_fast_pysf_worktree: flock is required for concurrency-safe recovery" >&2
  exit 2
fi
lock_path="$git_common_dir/robot-sf-fast-pysf-recovery.lock"
if [[ -L "$lock_path" ]]; then
  echo "recover_fast_pysf_worktree: refusing a symlinked repository recovery lock: $lock_path" >&2
  exit 2
fi
if [[ -e "$lock_path" && ! -f "$lock_path" ]]; then
  echo "recover_fast_pysf_worktree: repository recovery lock is not a regular file: $lock_path" >&2
  exit 2
fi
lock_fd=""
if ! exec {lock_fd}>"$lock_path"; then
  echo "recover_fast_pysf_worktree: could not open repository recovery lock: $lock_path" >&2
  exit 2
fi
if ! flock -n "$lock_fd"; then
  echo "recover_fast_pysf_worktree: another fast-pysf recovery is active for this repository" >&2
  echo "Wait for it to finish, then retry this explicit command." >&2
  echo "Lock: $lock_path" >&2
  exec {lock_fd}>&-
  exit 75
fi

release_lock() {
  flock -u "$lock_fd" 2>/dev/null || true
  exec {lock_fd}>&-
}
trap release_lock EXIT

capacity_report=""
if ! capacity_report="$(python3 "$capacity_checker" --path "$local_venv" 2>&1)"; then
  printf '%s\n' "$capacity_report" >&2
  echo "recover_fast_pysf_worktree: capacity gate blocked recovery before uv started" >&2
  exit 2
fi
printf '%s\n' "$capacity_report" >&2

if ! command -v uv >/dev/null 2>&1; then
  echo "recover_fast_pysf_worktree: uv is required for explicit environment recovery" >&2
  exit 2
fi

sync_needed=1
if [[ -x "$local_venv/bin/python" ]]; then
  existing_report=""
  if existing_report="$(env -u PYTHONPATH "$local_venv/bin/python" "$checker" 2>&1)"; then
    printf '%s\n' "$existing_report" >&2
    echo "recover_fast_pysf_worktree: local environment is already fast-pysf coherent; sync skipped" >&2
    sync_needed=0
  else
    echo "recover_fast_pysf_worktree: existing local environment is not coherent; refreshing it" >&2
    printf '%s\n' "$existing_report" >&2
  fi
fi

if [[ "$sync_needed" -eq 1 ]]; then
  if [[ ! -x "$local_venv/bin/python" ]]; then
    echo "recover_fast_pysf_worktree: creating worktree-local environment: $local_venv" >&2
    if ! env -u UV_NO_SYNC -u VIRTUAL_ENV -u UV_PROJECT \
      UV_PROJECT_ENVIRONMENT="$local_venv" uv venv "$local_venv"; then
      echo "recover_fast_pysf_worktree: uv venv failed; wrapped command was not started" >&2
      exit 2
    fi
  fi

  echo "recover_fast_pysf_worktree: refreshing only $local_venv" >&2
  echo "recover_fast_pysf_worktree: uv sync --all-extras --reinstall-package robot-sf --frozen" >&2
  if ! env -u UV_NO_SYNC -u VIRTUAL_ENV -u UV_PROJECT \
    UV_PROJECT_ENVIRONMENT="$local_venv" uv sync --all-extras --reinstall-package robot-sf --frozen; then
    echo "recover_fast_pysf_worktree: uv sync failed; wrapped command was not started" >&2
    exit 2
  fi
fi

if ! check_local_venv_layout; then
  echo "recover_fast_pysf_worktree: post-sync environment ownership check failed; wrapped command was not started" >&2
  exit 2
fi

final_report=""
if ! final_report="$(env -u PYTHONPATH "$local_venv/bin/python" "$checker" 2>&1)"; then
  echo "recover_fast_pysf_worktree: post-sync fast-pysf freshness check failed" >&2
  printf '%s\n' "$final_report" >&2
  echo "No wrapped command was started because the environment is still mismatched." >&2
  exit 2
fi
printf '%s\n' "$final_report" >&2

remaining_dirty_inputs="$(git status --porcelain=v1 -- "${dependency_inputs[@]}")" || {
  echo "recover_fast_pysf_worktree: could not verify dependency inputs after recovery" >&2
  exit 2
}
if [[ -n "$remaining_dirty_inputs" ]]; then
  echo "recover_fast_pysf_worktree: recovery changed tracked dependency inputs; refusing to continue" >&2
  printf '%s\n' "$remaining_dirty_inputs" >&2
  echo "Inspect and preserve the changes before retrying; no wrapped command was started." >&2
  exit 2
fi

echo "recover_fast_pysf_worktree: verified worktree-owned fast-pysf environment: $local_venv" >&2
