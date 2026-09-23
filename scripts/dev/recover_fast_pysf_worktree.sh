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
lock, and fails closed when the worktree filesystem is below the
ROBOT_SF_WORKTREE_MIN_FREE_BYTES threshold (default: 2 GiB).

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

It uses a worktree-local environment and runs:
  uv sync --all-extras --reinstall-package robot-sf --frozen

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
    --profile "$dependency_profile" --check-entry-points >/dev/null 2>&1; then
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
  fast_pysf_coherent=0
  if existing_report="$(env -u PYTHONPATH "$local_venv/bin/python" "$checker" 2>&1)"; then
    fast_pysf_coherent=1
  fi

  # Issue #8811: a profile-incomplete environment is repair-worthy even when
  # fast-pysf is already coherent, so an interrupted sync cannot be certified.
  profile_report=""
  dependency_profile_complete=0
  if profile_report="$(env -u PYTHONPATH "$local_venv/bin/python" "$profile_checker" \
    --profile "$dependency_profile" --check-entry-points 2>&1)"; then
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

# Issue #8811: certify the requested dependency profile as part of the recovery
# postcondition, so callers can rely on a successful recovery satisfying the
# shared wrapper's own profile preflight.
# Issue #9591: certify declared entry points match installed package metadata.
profile_final_report=""
if ! profile_final_report="$(env -u PYTHONPATH "$local_venv/bin/python" "$profile_checker" \
  --profile "$dependency_profile" --check-entry-points 2>&1)"; then
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

# The recovery is fully verified; share it as a seed for identical worktrees.
# Publish failures only warn and never fail this recovery.
publish_seed_venv

echo "recover_fast_pysf_worktree: verified worktree-owned fast-pysf environment: $local_venv" >&2
echo "recover_fast_pysf_worktree: verified dependency profile '$dependency_profile': $local_venv" >&2
