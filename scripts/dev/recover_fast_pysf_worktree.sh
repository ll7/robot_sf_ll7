#!/usr/bin/env bash
# Explicitly create or refresh a worktree-owned environment after the shared
# fast-pysf freshness gate reports a stale installed package.
#
# This helper never refreshes the owning checkout.  It only mutates the
# current linked worktree's ignored .venv, after a repository-scoped lock and
# capacity check, then verifies the installed package and the requested
# dependency profile before returning (issue #8811: recovery results must
# satisfy the wrapper's own profile preflight). Every non-interpreter symlink
# below .venv must remain inside that environment.

set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/recover_fast_pysf_worktree.sh [--profile NAME] [--wait-timeout SECONDS]

Create or refresh the current linked worktree's .venv with the pinned fast-pysf
package and verify package freshness before a caller runs project code.

After syncing, the helper also certifies that the requested dependency import
profile is complete via scripts/dev/check_worktree_optional_deps.py. An
existing environment is refreshed when either the installed fast-pysf package
or the requested profile is incomplete, so a previous interrupted sync cannot
be reported as a successful recovery.

This is an explicit recovery operation. It refuses the main checkout, refuses
dirty dependency inputs, serializes recovery per repository with a kernel-backed
lock, and fails closed when the worktree or effective uv-cache filesystem is below the
ROBOT_SF_WORKTREE_MIN_FREE_BYTES threshold (default: 2 GiB for core and 8 GiB for
named/all-extras profiles).

Options:
  --profile NAME         Dependency import profile the postcondition must certify
                         after sync (default: core). Use all-extras or a named
                         pyproject extra when the caller needs it.
  --wait-timeout SECONDS Maximum seconds to wait for the repository recovery lock when another
                         recovery is active (default: 0 or ROBOT_SF_RECOVERY_LOCK_TIMEOUT_SECONDS).
                         Alias: --timeout SECONDS.
  --non-blocking         Fail immediately with exit code 75 if the recovery lock is held
                         (equivalent to --wait-timeout 0).

Environment:
  ROBOT_SF_RECOVERY_LOCK_TIMEOUT_SECONDS
                         Default timeout in seconds to wait for the repository recovery lock
                         (default: 0).
  ROBOT_SF_WORKTREE_MIN_FREE_BYTES
                         Minimum free bytes required for worktree recovery (default: 2 GiB).
  ROBOT_SF_RECOVERY_MIN_FREE_BYTES
                         Optional profile-specific capacity threshold. When set, this value
                         overrides ROBOT_SF_WORKTREE_MIN_FREE_BYTES for the recovery gate. If
                         neither variable is set, core uses 2 GiB and named/all-extras profiles
                         use 8 GiB to account for larger dependency materialization.
  UV_CACHE_DIR           Explicit uv cache override. The effective cache path is resolved with
                         `uv cache dir` under the same config and environment as the sync;
                         UV_NO_CACHE is therefore checked on its temporary-storage filesystem.
  UV_CONFIG_FILE         Optional uv configuration path honored by `uv cache dir`.
  ROBOT_SF_VENV_SEED_CACHE
                         Directory holding checksum-keyed reusable recovery environments
                         (default: $XDG_CACHE_HOME/robot-sf/worktree-venv-seeds or
                         $HOME/.cache/robot-sf/worktree-venv-seeds). Set to "off" to
                         disable seed publish and restore. A successful recovery publishes
                         its verified worktree .venv as a hardlinked seed; a later recovery
                         with identical dependency inputs restores the seed instead of
                         materializing every package again, then still runs the standard
                         sync (which rebinds the editable install) and verification. Any
                         seed failure falls back to the full-sync path.

The helper is normally invoked through:
  scripts/dev/run_worktree_shared_venv.sh --recover-stale-fast-pysf -- <command>

It uses a worktree-local environment and runs a profile-aware frozen sync:
  core:        uv sync --reinstall-package robot-sf --frozen
  NAME:        uv sync --extra NAME --reinstall-package robot-sf --frozen
  all-extras:  uv sync --all-extras --reinstall-package robot-sf --frozen

The --frozen flag prevents this recovery path from changing dependency locks.
EOF
}

dependency_profile="core"
locked_recovery=0
wait_timeout="${ROBOT_SF_RECOVERY_LOCK_TIMEOUT_SECONDS:-0}"

# Checksum-keyed reusable recovery environments (issue #9338). Seeds live
# outside every checkout so worktree teardown never orphans them; each seed is
# a hardlinked clone, so publishing and restoring cost seconds and only
# megabytes of marginal disk while uv's package cache stays warm.
seed_cache_root="${ROBOT_SF_VENV_SEED_CACHE:-}"
if [[ -z "$seed_cache_root" ]]; then
  if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
    seed_cache_root="$XDG_CACHE_HOME/robot-sf/worktree-venv-seeds"
  elif [[ -n "${HOME:-}" ]]; then
    seed_cache_root="$HOME/.cache/robot-sf/worktree-venv-seeds"
  fi
fi
seed_caching_disabled=0
case "$seed_cache_root" in
  ""|off|OFF|0) seed_caching_disabled=1 ;;
esac

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --profile)
      if [[ "$#" -lt 2 || -z "${2:-}" ]]; then
        echo "recover_fast_pysf_worktree: --profile requires a dependency profile name" >&2
        exit 2
      fi
      dependency_profile="$2"
      shift 2
      ;;
    --wait-timeout|--timeout)
      if [[ "$#" -lt 2 || -z "${2:-}" ]]; then
        echo "recover_fast_pysf_worktree: $1 requires a non-negative integer timeout in seconds" >&2
        exit 2
      fi
      if ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "recover_fast_pysf_worktree: $1 requires a non-negative integer timeout in seconds: $2" >&2
        exit 2
      fi
      wait_timeout="$2"
      shift 2
      ;;
    --wait-timeout=*|--timeout=*)
      val="${1#*=}"
      if ! [[ "$val" =~ ^[0-9]+$ ]]; then
        echo "recover_fast_pysf_worktree: timeout requires a non-negative integer in seconds: $val" >&2
        exit 2
      fi
      wait_timeout="$val"
      shift
      ;;
    --non-blocking)
      wait_timeout=0
      shift
      ;;
    # Internal re-entry flag: the portable-lock fallback re-executes this script
    # under worktree_creation_lock.py so recovery runs while a
    # Python fcntl holder owns the shared lock file. Never pass this directly.
    --__locked-recovery)
      locked_recovery=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "recover_fast_pysf_worktree: unrecognized option: $1" >&2
      show_help >&2
      exit 2
      ;;
  esac
done

if ! [[ "$wait_timeout" =~ ^[0-9]+$ ]]; then
  echo "recover_fast_pysf_worktree: ROBOT_SF_RECOVERY_LOCK_TIMEOUT_SECONDS must be a non-negative integer: $wait_timeout" >&2
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
profile_checker="$repo_root/scripts/dev/check_worktree_optional_deps.py"
capacity_checker="$repo_root/scripts/dev/check_worktree_capacity.py"
if [[ ! -f "$checker" || ! -f "$profile_checker" || ! -f "$capacity_checker" ]]; then
  echo "recover_fast_pysf_worktree: required freshness, profile, or capacity checker is missing" >&2
  exit 2
fi

# Keep accepted profile names owned by the dependency preflight helper. The
# recovery command must reject an unknown profile before it can materialize an
# environment or invoke uv.
if ! python3 - "$profile_checker" "$dependency_profile" <<'PY'
import runpy
import sys

try:
    namespace = runpy.run_path(sys.argv[1])
    profiles = namespace.get("PROFILES")
    profile = sys.argv[2]
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f"unsupported dependency profile: {profile}")
except (OSError, TypeError, ValueError, KeyError) as exc:
    print(f"recover_fast_pysf_worktree: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc
PY
then
  exit 2
fi

sync_args=(sync)
case "$dependency_profile" in
  # ORCA's rvo2 import is provided by the core path dependency, so its
  # optional-import profile does not correspond to a pyproject extra.
  core|orca)
    ;;
  all-extras)
    sync_args+=(--all-extras)
    ;;
  *)
    sync_args+=(--extra "$dependency_profile")
    ;;
esac
sync_args+=(--reinstall-package robot-sf --frozen)

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

recovery_state_file="$local_venv/.robot-sf-recovery-state.json"
recovery_state_active=0
recovery_state_status=""
recovery_started_at=""
recovery_child_pid=""
recovery_child_uses_session=0

write_recovery_state() {
  local status="$1" message="$2"
  if ! mkdir -p "$local_venv"; then
    echo "recover_fast_pysf_worktree: could not create partial-environment state directory: $local_venv" >&2
    return 1
  fi
  if ! python3 - "$recovery_state_file" "$status" "$message" "$dependency_profile" \
    "$repo_root" "$local_venv" "$recovery_started_at" "$$" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "robot_sf.recovery_state.v1",
    "status": sys.argv[2],
    "message": sys.argv[3],
    "dependency_profile": sys.argv[4],
    "worktree": sys.argv[5],
    "environment": sys.argv[6],
    "started_at": int(sys.argv[7]),
    "updated_at": int(time.time()),
    "pid": int(sys.argv[8]),
}
temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
  then
    echo "recover_fast_pysf_worktree: could not write partial-environment state: $recovery_state_file" >&2
    return 1
  fi
  recovery_state_status="$status"
}

arm_recovery_state() {
  recovery_state_active=1
  recovery_started_at="$(date +%s)"
}

begin_recovery_state() {
  arm_recovery_state
  write_recovery_state "in_progress" "profile materialization started; retry after completion or cleanup"
}

record_failed_recovery_start() {
  arm_recovery_state
  write_recovery_state "failed" \
    "recovery could not create the virtual environment; inspect or remove this partial environment before retrying"
}

record_recovery_state() {
  [[ "$recovery_state_active" -eq 1 ]] || return 0
  write_recovery_state "$1" "$2" || true
}

clear_recovery_state() {
  if [[ -e "$recovery_state_file" || -L "$recovery_state_file" ]]; then
    rm -f "$recovery_state_file" || return 1
  fi
  recovery_state_active=0
  recovery_state_status=""
  return 0
}

finalize_recovery_state() {
  local recovery_rc="$1"
  [[ "$recovery_state_active" -eq 1 ]] || return 0
  if [[ "$recovery_state_status" == "in_progress" && "$recovery_rc" -ne 0 ]]; then
    record_recovery_state "failed" \
      "recovery exited with status $recovery_rc; inspect or remove this partial environment before retrying"
  fi
  if [[ "$recovery_state_status" == "failed" || "$recovery_state_status" == "interrupted" ]]; then
    echo "recover_fast_pysf_worktree: partial environment state preserved: $recovery_state_file" >&2
    echo "recover_fast_pysf_worktree: remove that worktree .venv or rerun recovery after inspecting it" >&2
  fi
}

run_recovery_command() {
  recovery_child_uses_session=0
  if command -v setsid >/dev/null 2>&1; then
    recovery_child_uses_session=1
    setsid -- "$@" &
  else
    "$@" &
  fi
  recovery_child_pid=$!

  local command_rc=0
  if wait "$recovery_child_pid"; then
    command_rc=0
  else
    command_rc=$?
  fi
  recovery_child_pid=""
  recovery_child_uses_session=0
  return "$command_rc"
}

terminate_recovery_command() {
  local child_pid="$recovery_child_pid"
  [[ -n "$child_pid" ]] || return 0

  recovery_process_alive() {
    local process_pid="$1" process_state=""
    if ! kill -0 "$process_pid" 2>/dev/null; then
      return 1
    fi
    if command -v ps >/dev/null 2>&1; then
      process_state="$(ps -o stat= -p "$process_pid" 2>/dev/null | tr -d '[:space:]')"
      [[ "$process_state" == Z* ]] && return 1
    fi
    return 0
  }

  recovery_process_group_alive() {
    local process_group_id="$1" process_state=""
    if ! kill -0 -- "-$process_group_id" 2>/dev/null; then
      return 1
    fi
    if command -v ps >/dev/null 2>&1; then
      while read -r process_state; do
        [[ -z "$process_state" || "$process_state" == Z* ]] || return 0
      done < <(ps -o stat= -g "$process_group_id" 2>/dev/null)
      return 1
    fi
    return 0
  }

  recovery_wait_for_exit() {
    local process_pid="$1" attempts=0
    while (( attempts < 100 )); do
      if ! recovery_process_alive "$process_pid" &&
        ! recovery_process_group_alive "$process_pid"; then
        return 0
      fi
      sleep 0.05
      attempts=$((attempts + 1))
    done
    return 1
  }

  if [[ "$recovery_child_uses_session" -eq 1 ]]; then
    kill -TERM -- "-$child_pid" 2>/dev/null || true
  else
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
  if ! recovery_wait_for_exit "$child_pid"; then
    echo "recover_fast_pysf_worktree: recovery child ignored SIGTERM; sending SIGKILL" >&2
    if [[ "$recovery_child_uses_session" -eq 1 ]]; then
      kill -KILL -- "-$child_pid" 2>/dev/null || true
    else
      kill -KILL "$child_pid" 2>/dev/null || true
    fi
    if ! recovery_wait_for_exit "$child_pid"; then
      kill -KILL "$child_pid" 2>/dev/null || true
    fi
  fi
  if ! recovery_process_alive "$child_pid"; then
    wait "$child_pid" 2>/dev/null || true
  else
    echo "recover_fast_pysf_worktree: recovery child cleanup remains unverified" >&2
  fi
  recovery_child_pid=""
  recovery_child_uses_session=0
}

handle_recovery_signal() {
  local signal_name="$1" signal_exit="$2"
  terminate_recovery_command
  if [[ "$recovery_state_active" -eq 1 ]]; then
    record_recovery_state "interrupted" \
      "recovery interrupted by $signal_name; inspect or remove this partial environment before retrying"
  fi
  exit "$signal_exit"
}

trap 'handle_recovery_signal INT 130' INT
trap 'handle_recovery_signal TERM 143' TERM
trap 'handle_recovery_signal HUP 129' HUP

if [[ -f "$recovery_state_file" ]]; then
  echo "recover_fast_pysf_worktree: found prior partial-environment state: $recovery_state_file" >&2
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

  local external_symlink symlink_check_status
  if external_symlink="$(python3 - "$local_venv" "$main_repo_root" <<'PY'
import os
from pathlib import Path
import sys

local_venv = Path(sys.argv[1])
main_checkout = Path(sys.argv[2]).resolve(strict=False)


def is_standard_python_link(path: Path) -> bool:
    relative = path.relative_to(local_venv)
    if not relative.parts or relative.parts[0] != "bin":
        return False
    return path.name == "python" or path.name == "python3" or path.name.startswith("python3.")


def fail_on_walk_error(error: OSError) -> None:
    raise error


try:
    for root, directories, files in os.walk(
        local_venv,
        followlinks=False,
        onerror=fail_on_walk_error,
    ):
        for name in (*directories, *files):
            candidate = Path(root) / name
            if not candidate.is_symlink():
                continue
            try:
                resolved = candidate.resolve(strict=False)
            except (OSError, RuntimeError) as error:
                print(f"could not resolve environment symlink {candidate}: {error}")
                raise SystemExit(3)

            if is_standard_python_link(candidate):
                if not resolved.is_file():
                    print(f"{candidate} -> {resolved}")
                    raise SystemExit(4)
                try:
                    resolved.relative_to(main_checkout)
                except ValueError:
                    continue
                print(f"{candidate} -> {resolved}")
                raise SystemExit(2)

            try:
                resolved.relative_to(local_venv)
            except ValueError:
                print(f"{candidate} -> {resolved}")
                raise SystemExit(1)
except OSError as error:
    print(f"could not inspect worktree environment symlinks: {error}")
    raise SystemExit(3)
PY
)"; then
    :
  else
    symlink_check_status=$?
    case "$symlink_check_status" in
      1)
        echo "recover_fast_pysf_worktree: refusing an environment symlink outside the worktree:" >&2
        printf '%s\n' "$external_symlink" >&2
        ;;
      2)
        echo "recover_fast_pysf_worktree: refusing a host-interpreter link into the owning checkout:" >&2
        printf '%s\n' "$external_symlink" >&2
        ;;
      4)
        echo "recover_fast_pysf_worktree: refusing a broken host-interpreter link:" >&2
        printf '%s\n' "$external_symlink" >&2
        ;;
      *)
        echo "recover_fast_pysf_worktree: could not verify environment symlink ownership:" >&2
        printf '%s\n' "$external_symlink" >&2
        ;;
    esac
    return 1
  fi

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
      "$main_repo_root"|"$main_repo_root"/*)
        echo "recover_fast_pysf_worktree: refusing a host-interpreter link into the owning checkout: $local_venv/bin/python" >&2
        return 1
        ;;
    esac
  fi
}

if ! check_local_venv_layout; then
  exit 2
fi

format_lock_diagnostics() {
  local target_lock="$1"
  local owner_pid="" owner_started="" owner_worktree=""
  if [[ -f "$target_lock" && -r "$target_lock" ]]; then
    while IFS='=' read -r key val || [[ -n "$key" ]]; do
      case "$key" in
        pid) owner_pid="$val" ;;
        started) owner_started="$val" ;;
        worktree) owner_worktree="$val" ;;
      esac
    done < "$target_lock"
  fi

  local status="unknown"
  if [[ -n "$owner_pid" && "$owner_pid" =~ ^[0-9]+$ ]]; then
    if kill -0 "$owner_pid" 2>/dev/null; then
      status="alive (PID $owner_pid running)"
    else
      status="stale (PID $owner_pid not running)"
    fi
  fi

  local elapsed=""
  if [[ -n "$owner_started" && "$owner_started" =~ ^[0-9]+$ ]]; then
    local now
    now="$(date +%s)"
    if [[ "$now" -ge "$owner_started" ]]; then
      elapsed=" ($(( now - owner_started ))s elapsed)"
    fi
  fi

  echo "Lock: $target_lock" >&2
  if [[ -z "$owner_pid" && -z "$owner_started" && -z "$owner_worktree" ]]; then
    echo "Lock owner metadata: none recorded" >&2
  else
    echo "Lock owner metadata:" >&2
    echo "  PID: ${owner_pid:-unknown}" >&2
    echo "  Started: ${owner_started:-unknown}${elapsed}" >&2
    echo "  Worktree: ${owner_worktree:-unknown}" >&2
    echo "  Status: $status" >&2
  fi
}

# Identity inputs mirror exactly what a recovery sync consumes: the manifests
# and lockfiles uv resolves, the vendored rvo2 sources uv builds, and the
# requested dependency profile. Toolchain drift (uv or interpreter updates) is
# intentionally not part of the key; a drifted seed fails the coherence gates
# below and falls back to the full-sync path.
venv_identity_key() {
  [[ "$seed_caching_disabled" -eq 0 ]] || return 1
  cd -- "$repo_root" 2>/dev/null || return 1
  local input
  for input in pyproject.toml uv.lock fast-pysf/pyproject.toml fast-pysf/uv.lock; do
    [[ -f "$input" ]] || return 1
  done
  local rvo2_manifest
  rvo2_manifest="$(git ls-files -s third_party/python-rvo2 2>/dev/null)" || return 1
  [[ -n "$rvo2_manifest" ]] || return 1
  local file_digests
  # Hash only the content digests (never filenames): the key must be identical
  # for every worktree checking out the same dependency inputs.
  file_digests="$(sha256sum pyproject.toml uv.lock \
    fast-pysf/pyproject.toml fast-pysf/uv.lock 2>/dev/null | cut -d' ' -f1)" \
    || return 1
  {
    printf '%s\n' "$file_digests"
    printf 'profile=%s\n' "$dependency_profile"
    printf '%s\n' "$rvo2_manifest"
  } | sha256sum | cut -d' ' -f1
}

seed_receipt_identity() {
  seed_receipt_field "$1" "identity"
}

seed_receipt_field() {
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        value = json.load(stream).get(sys.argv[2], "")
    print(value if isinstance(value, str) else "")
except (OSError, ValueError):
    print("")
PY
}

# Reuse the worktree layout ownership check against a seed directory by
# temporarily retargeting the global it inspects. Seeds must satisfy the same
# symlink-containment contract as worktree environments.
check_seed_venv_layout() {
  local saved_venv="$local_venv"
  local_venv="$1"
  local layout_rc=0
  check_local_venv_layout || layout_rc=$?
  local_venv="$saved_venv"
  return "$layout_rc"
}

# Rewrite absolute source-environment paths in worktree entry-point scripts so
# a restored clone never executes another checkout's interpreter. Only
# NUL-free files under bin/ are rewritten; binaries and symlinks are left
# untouched. The rewritten prefix is the seed's originating venv recorded in
# the receipt, not the seed directory itself: cloned files still reference
# the checkout they were materialized from.
rebind_seed_prefix() {
  local source_venv="$1" target_venv="$2"
  python3 - "$source_venv" "$target_venv" <<'PY'
import os
import stat
import sys
from pathlib import Path

old, new = sys.argv[1], sys.argv[2]
rewritten = 0
bin_dir = Path(new) / "bin"
for child in sorted(bin_dir.iterdir()):
    if child.is_symlink() or not child.is_file():
        continue
    try:
        data = child.read_bytes()
    except OSError:
        continue
    if b"\0" in data or old.encode() not in data:
        continue
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        continue
    updated = text.replace(old, new)
    if updated == text:
        continue
    try:
        mode = stat.S_IMODE(child.stat().st_mode)
        # Break the seed hardlink before writing: an in-place write would
        # mutate the shared seed (and the worktree it was published from).
        child.unlink()
        child.write_bytes(updated.encode())
        os.chmod(child, mode)
    except OSError as exc:
        print(f"could not rebind seed path in {child}: {exc}")
        raise SystemExit(1)
    rewritten += 1
print(f"rebound {rewritten} seed interpreter paths")
PY
}

# Best-effort seed restore. Returns 0 with a cloned worktree .venv ready for
# the standard sync (which rebinds the editable robot-sf install to this
# worktree) and verification below; returns nonzero to take the normal
# full-sync path with the local environment untouched or removed.
restore_seed_venv() {
  [[ "$seed_caching_disabled" -eq 0 ]] || return 1
  local key seed_dir receipt recorded source_venv
  key="$(venv_identity_key)" || return 1
  seed_dir="$seed_cache_root/$key"
  receipt="$seed_dir/seed-receipt.json"
  [[ -f "$receipt" && -x "$seed_dir/bin/python" ]] || return 1
  recorded="$(seed_receipt_identity "$receipt")" || return 1
  [[ -n "$recorded" && "$recorded" == "$key" ]] || return 1
  source_venv="$(seed_receipt_field "$receipt" "source_venv")" || return 1
  [[ -n "$source_venv" ]] || return 1
  if ! check_seed_venv_layout "$seed_dir"; then
    echo "recover_fast_pysf_worktree: seed environment failed ownership verification; using full sync" >&2
    return 1
  fi
  local seed_report
  if ! seed_report="$(env -u PYTHONPATH "$seed_dir/bin/python" "$checker" 2>&1)"; then
    echo "recover_fast_pysf_worktree: seed environment is not fast-pysf coherent; using full sync" >&2
    return 1
  fi
  if ! env -u PYTHONPATH "$seed_dir/bin/python" "$profile_checker" \
    --profile "$dependency_profile" >/dev/null 2>&1; then
    echo "recover_fast_pysf_worktree: seed environment lacks dependency profile '$dependency_profile'; using full sync" >&2
    return 1
  fi
  if [[ -n "$local_venv" && "$local_venv" == "$repo_root/.venv" && -e "$local_venv" ]]; then
    rm -rf "$local_venv" || return 1
  fi
  if ! cp -al "$seed_dir" "$local_venv" 2>/dev/null; then
    echo "recover_fast_pysf_worktree: seed clone failed (cross-filesystem seeds are unsupported); using full sync" >&2
    rm -rf "$local_venv" 2>/dev/null || true
    return 1
  fi
  if ! rebind_seed_prefix "$source_venv" "$local_venv"; then
    rm -rf "$local_venv" 2>/dev/null || true
    return 1
  fi
  printf '%s\n' "$seed_report" >&2
  echo "recover_fast_pysf_worktree: restored checksum-keyed seed environment for identity ${key:0:12}" >&2
  return 0
}

# Best-effort seed publish after a fully verified recovery. Always succeeds
# from the caller's perspective: a publish failure only warns, never fails an
# already-verified recovery.
publish_seed_venv() {
  [[ "$seed_caching_disabled" -eq 0 ]] || return 0
  local key seed_dir staging receipt head_sha uv_version python_version venv_bytes
  key="$(venv_identity_key)" || return 0
  seed_dir="$seed_cache_root/$key"
  [[ -d "$seed_dir" ]] && return 0
  [[ -L "$seed_cache_root" ]] && return 0
  if ! mkdir -p "$seed_cache_root" 2>/dev/null; then
    echo "recover_fast_pysf_worktree: warning: could not create seed cache; skipping seed publish" >&2
    return 0
  fi
  staging="$seed_cache_root/.staging-$key-$$"
  rm -rf "$staging" 2>/dev/null || true
  if ! cp -al "$local_venv" "$staging" 2>/dev/null; then
    echo "recover_fast_pysf_worktree: warning: seed clone failed; skipping seed publish" >&2
    rm -rf "$staging" 2>/dev/null || true
    return 0
  fi
  receipt="$staging/seed-receipt.json"
  head_sha="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  uv_version="$(uv --version 2>/dev/null | head -n 1 || echo unknown)"
  python_version="$("$local_venv/bin/python" --version 2>&1 || echo unknown)"
  venv_bytes="$(du -sb "$local_venv" 2>/dev/null | cut -f1 || echo 0)"
  if ! python3 - "$receipt" "$key" "$repo_root" "$local_venv" "$head_sha" "$dependency_profile" \
    "$uv_version" "$python_version" "$venv_bytes" <<'PY' 2>/dev/null; then
import json
import sys
import time

_, receipt, identity, source, venv, head, profile, uv_version, python_version, size = sys.argv
payload = {
    "identity": identity,
    "source_worktree": source,
    "source_venv": venv,
    "head_sha": head,
    "dependency_profile": profile,
    "uv_version": uv_version.strip(),
    "python_version": python_version.strip(),
    "venv_bytes": size.strip(),
    "created_at": int(time.time()),
}
with open(receipt, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
PY
    echo "recover_fast_pysf_worktree: warning: seed receipt write failed; skipping seed publish" >&2
    rm -rf "$staging" 2>/dev/null || true
    return 0
  fi
  if ! mv "$staging" "$seed_dir" 2>/dev/null; then
    rm -rf "$staging" 2>/dev/null || true
    return 0
  fi
  echo "recover_fast_pysf_worktree: published checksum-keyed seed environment for identity ${key:0:12}" >&2
  return 0
}

lock_path="$git_common_dir/robot-sf-fast-pysf-recovery.lock"
if [[ -L "$lock_path" ]]; then
  echo "recover_fast_pysf_worktree: refusing a symlinked repository recovery lock: $lock_path" >&2
  exit 2
fi
if [[ -e "$lock_path" && ! -f "$lock_path" ]]; then
  echo "recover_fast_pysf_worktree: repository recovery lock is not a regular file: $lock_path" >&2
  exit 2
fi
if [[ "$locked_recovery" -eq 1 ]]; then
  # Re-entered under worktree_creation_lock.py holding the
  # shared lock file; record lock owner metadata and clear on exit.
  printf 'pid=%s\nstarted=%s\nworktree=%s\n' "$$" "$(date +%s)" "$repo_root" > "$lock_path" 2>/dev/null || true
  release_locked_recovery() {
    local recovery_rc="$?"
    finalize_recovery_state "$recovery_rc"
    : > "$lock_path" 2>/dev/null || true
  }
  trap release_locked_recovery EXIT
elif command -v flock >/dev/null 2>&1; then
  lock_fd=""
  if ! exec {lock_fd}<>"$lock_path"; then
    echo "recover_fast_pysf_worktree: could not open repository recovery lock: $lock_path" >&2
    exit 2
  fi
  if [[ "$wait_timeout" -gt 0 ]]; then
    if ! flock -n "$lock_fd"; then
      echo "recover_fast_pysf_worktree: another fast-pysf recovery is active; waiting up to ${wait_timeout}s for lock" >&2
      format_lock_diagnostics "$lock_path"
      if ! flock -w "$wait_timeout" "$lock_fd"; then
        echo "recover_fast_pysf_worktree: timed out waiting for repository fast-pysf recovery lock after ${wait_timeout}s" >&2
        echo "recover_fast_pysf_worktree: another fast-pysf recovery is active for this repository" >&2
        echo "Wait for it to finish, then retry this explicit command." >&2
        format_lock_diagnostics "$lock_path"
        exec {lock_fd}>&-
        exit 75
      fi
    fi
  else
    if ! flock -n "$lock_fd"; then
      echo "recover_fast_pysf_worktree: another fast-pysf recovery is active for this repository" >&2
      echo "Wait for it to finish, then retry this explicit command." >&2
      format_lock_diagnostics "$lock_path"
      exec {lock_fd}>&-
      exit 75
    fi
  fi

  printf 'pid=%s\nstarted=%s\nworktree=%s\n' "$$" "$(date +%s)" "$repo_root" > "$lock_path" 2>/dev/null || true

  release_lock() {
    local recovery_rc="$?"
    finalize_recovery_state "$recovery_rc"
    : > "$lock_path" 2>/dev/null || true
    flock -u "$lock_fd" 2>/dev/null || true
    exec {lock_fd}>&-
  }
  trap release_lock EXIT
else
  # Portable fallback: fcntl.flock via the helper uses flock(2) on the same
  # lock file identity, so it serializes against flock-CLI holders.
  echo "recover_fast_pysf_worktree: flock CLI not used; holding portable lock on $lock_path" >&2
  python_lock_rc=0
  lock_args=()
  if [[ "$wait_timeout" -gt 0 ]]; then
    lock_args+=(--timeout "$wait_timeout")
  else
    lock_args+=(--non-blocking)
  fi
  python3 "$repo_root/scripts/dev/worktree_creation_lock.py" "${lock_args[@]}" "$lock_path" -- \
    "$repo_root/scripts/dev/recover_fast_pysf_worktree.sh" --__locked-recovery \
    --profile "$dependency_profile" || python_lock_rc=$?
  if [[ "$python_lock_rc" -eq 75 ]]; then
    if [[ "$wait_timeout" -gt 0 ]]; then
      echo "recover_fast_pysf_worktree: timed out waiting for repository fast-pysf recovery lock after ${wait_timeout}s" >&2
    fi
    echo "recover_fast_pysf_worktree: another fast-pysf recovery is active for this repository" >&2
    echo "Wait for it to finish, then retry this explicit command." >&2
    format_lock_diagnostics "$lock_path"
    exit 75
  fi
  exit "$python_lock_rc"
fi

capacity_report=""
recovery_minimum_free_bytes="${ROBOT_SF_RECOVERY_MIN_FREE_BYTES:-}"
if [[ -z "$recovery_minimum_free_bytes" && -z "${ROBOT_SF_WORKTREE_MIN_FREE_BYTES:-}" ]]; then
  case "$dependency_profile" in
    core)
      recovery_minimum_free_bytes=$((2 * 1024 * 1024 * 1024))
      ;;
    *)
      recovery_minimum_free_bytes=$((8 * 1024 * 1024 * 1024))
      ;;
  esac
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "recover_fast_pysf_worktree: uv is required for explicit environment recovery" >&2
  exit 2
fi

uv_cache_dir=""
if ! uv_cache_dir="$(
  env -u UV_NO_SYNC -u VIRTUAL_ENV -u UV_PROJECT \
    UV_PROJECT_ENVIRONMENT="$local_venv" uv cache dir --directory "$repo_root" 2>/dev/null
)" || [[ -z "$uv_cache_dir" ]]; then
  echo "recover_fast_pysf_worktree: could not determine the effective uv cache directory" >&2
  echo "uv cache dir failed; inspect UV_CONFIG_FILE, UV_CACHE_DIR, or UV_NO_CACHE, then retry." >&2
  exit 2
fi
if [[ "$uv_cache_dir" != /* ]]; then
  uv_cache_dir="$repo_root/$uv_cache_dir"
fi

for capacity_path in "$local_venv" "$uv_cache_dir"; do
  capacity_args=(--path "$capacity_path")
  if [[ -n "$recovery_minimum_free_bytes" ]]; then
    capacity_args+=(--minimum-free-bytes "$recovery_minimum_free_bytes")
  fi
  echo "recover_fast_pysf_worktree: capacity preflight for dependency profile '$dependency_profile' at $capacity_path" >&2
  if ! capacity_report="$(python3 "$capacity_checker" "${capacity_args[@]}" 2>&1)"; then
    printf '%s\n' "$capacity_report" >&2
    echo "recover_fast_pysf_worktree: capacity gate blocked recovery before materialization at $capacity_path" >&2
    exit 2
  fi
  printf '%s\n' "$capacity_report" >&2
done

sync_needed=1
if [[ -x "$local_venv/bin/python" ]]; then
  existing_report=""
  fast_pysf_coherent=0
  if existing_report="$(env -u PYTHONPATH "$local_venv/bin/python" "$checker" 2>&1)"; then
    fast_pysf_coherent=1
  fi

  # Issue #8811: a profile-incomplete environment is repair-worthy even when
  # fast-pysf is already coherent, so an interrupted sync cannot be certified.
  profile_report=""
  dependency_profile_complete=0
  if profile_report="$(env -u PYTHONPATH "$local_venv/bin/python" "$profile_checker" \
    --profile "$dependency_profile" 2>&1)"; then
    dependency_profile_complete=1
  fi

  if [[ "$fast_pysf_coherent" -eq 1 && "$dependency_profile_complete" -eq 1 ]]; then
    printf '%s\n' "$existing_report" >&2
    echo "recover_fast_pysf_worktree: local environment is already fast-pysf coherent and dependency profile '$dependency_profile' is complete; sync skipped" >&2
    # The environment just passed the same gates a recovery would certify,
    # so share it as a seed for identical worktrees (best-effort, warn-only).
    publish_seed_venv
    sync_needed=0
  elif [[ "$fast_pysf_coherent" -eq 1 ]]; then
    echo "recover_fast_pysf_worktree: existing local environment is missing dependency profile '$dependency_profile'; refreshing it" >&2
    printf '%s\n' "$profile_report" >&2
  else
    echo "recover_fast_pysf_worktree: existing local environment is not coherent; refreshing it" >&2
    printf '%s\n' "$existing_report" >&2
  fi
fi

if [[ "$sync_needed" -eq 1 ]]; then
  # A verified seed for this exact identity skips package materialization;
  # the sync below still rebinds the editable install and re-verifies.
  if restore_seed_venv; then
    echo "recover_fast_pysf_worktree: seed restore will be rebound and verified by the standard sync below" >&2
  fi
  if [[ ! -x "$local_venv/bin/python" ]]; then
    # A previous interrupted attempt may have left only the state marker.  Do
    # not leave that marker in place while uv creates the virtual environment:
    # uv rejects a non-empty directory that is not already a virtualenv.
    if [[ -e "$recovery_state_file" || -L "$recovery_state_file" ]]; then
      rm -f "$recovery_state_file" || {
        echo "recover_fast_pysf_worktree: could not clear the prior partial-environment state" >&2
        exit 2
      }
    fi
    arm_recovery_state
    echo "recover_fast_pysf_worktree: creating worktree-local environment: $local_venv" >&2
    if ! run_recovery_command env -u UV_NO_SYNC -u VIRTUAL_ENV -u UV_PROJECT \
      UV_PROJECT_ENVIRONMENT="$local_venv" uv venv "$local_venv"; then
      if ! record_failed_recovery_start; then
        echo "recover_fast_pysf_worktree: refusing to certify recovery without partial-environment state" >&2
      fi
      echo "recover_fast_pysf_worktree: uv venv failed; wrapped command was not started" >&2
      exit 2
    fi
  fi
  if ! begin_recovery_state; then
    echo "recover_fast_pysf_worktree: refusing to certify recovery without partial-environment state" >&2
    exit 2
  fi

  echo "recover_fast_pysf_worktree: refreshing only $local_venv" >&2
  echo "recover_fast_pysf_worktree: uv ${sync_args[*]}" >&2
  if ! run_recovery_command env -u UV_NO_SYNC -u VIRTUAL_ENV -u UV_PROJECT \
    UV_PROJECT_ENVIRONMENT="$local_venv" uv "${sync_args[@]}"; then
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

# Issue #8811: certify the requested dependency profile as part of the recovery
# postcondition, so callers can rely on a successful recovery satisfying the
# shared wrapper's own profile preflight.
profile_final_report=""
if ! profile_final_report="$(env -u PYTHONPATH "$local_venv/bin/python" "$profile_checker" \
  --profile "$dependency_profile" 2>&1)"; then
  echo "recover_fast_pysf_worktree: post-sync dependency profile '$dependency_profile' is incomplete in $local_venv" >&2
  printf '%s\n' "$profile_final_report" >&2
  echo "No wrapped command was started because the requested dependency profile is incomplete." >&2
  echo "Repair: run 'cd $repo_root && scripts/dev/bootstrap_worktree.sh', then retry." >&2
  exit 2
fi
printf '%s\n' "$profile_final_report" >&2

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

if ! clear_recovery_state; then
  echo "recover_fast_pysf_worktree: could not clear partial-environment state; refusing to certify recovery" >&2
  exit 2
fi

# The recovery is fully verified; share it as a seed for identical worktrees.
# Publish failures only warn and never fail this recovery.
publish_seed_venv

echo "recover_fast_pysf_worktree: verified worktree-owned fast-pysf environment: $local_venv" >&2
echo "recover_fast_pysf_worktree: verified dependency profile '$dependency_profile': $local_venv" >&2
