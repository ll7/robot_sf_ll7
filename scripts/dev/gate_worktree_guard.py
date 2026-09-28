#!/usr/bin/env python3
"""Deterministic health/recreate guard for PR-gate worktrees.

A PR-gate runs in a linked worktree. During a gate session that worktree can
disappear (stale-worktree reaper, manual `git worktree remove`, sibling
cleanup, filesystem race). When the gate then tries to branch-switch or resolve
conflicts against the now-missing path it fails with an opaque
``CreateProcess ... No such file or directory`` and must recreate/bootstrap the
worktree from scratch, losing important dirty state.

This module makes that failure deterministic and diagnosable:

- ``verify_gate_worktree`` checks the registered path exists *before* the gate
  performs a branch switch or conflict resolution. If the path is missing but a
  live lease still claims it, the guard reports the lease owner (the cleanup
  owner / last gate session) instead of dying opaquely.
- ``recreate_gate_worktree`` rebuilds a missing linked worktree from a surviving
  lease record (branch + owner) and re-registers the lease so a gate can resume
  rather than rebuild cold.

The guard never removes anything. It only inspects, reports, and (optionally)
recreates the worktree that the lease describes.

Usage:
    # Verify the current gate worktree is intact; exit non-zero if missing.
    python scripts/dev/gate_worktree_guard.py verify --path /abs/worktree

    # Read-only preflight before a local Git fetch/merge/rebase/push.
    python scripts/dev/gate_worktree_guard.py preflight --path /abs/worktree --json

    # Same, but recreate from the surviving lease if missing.
    python scripts/dev/gate_worktree_guard.py ensure --path /abs/worktree
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "gate_worktree_guard.v1"
MISSING_STATE_LOSS_BOUNDARY = "dirty_untracked_ignored_state_not_recoverable"
INDEX_LOCK_SCHEMA_VERSION = "gate_worktree_index_lock.v1"
PREFLIGHT_SCHEMA_VERSION = "gate_worktree_preflight.v1"
PROC_ROOT = Path("/proc")


@dataclass(frozen=True, slots=True)
class GateWorktreeHealth:
    """Health result for a single gate worktree path."""

    schema: str
    path: str
    exists: bool
    classification: str  # "healthy", "missing", "errored"
    branch: str | None = None
    head_sha: str | None = None
    lease_owner: str | None = None
    lease_pr_number: int | None = None
    lease_gate_id: str | None = None
    lease_expires_at: str | None = None
    cleanup_owner: str | None = None
    error: str | None = None
    recovery: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GateWorktreeRecreate:
    """Result of a recreate attempt for a missing gate worktree."""

    schema: str
    path: str
    recreated: bool
    branch: str | None = None
    head_sha: str | None = None
    lease_owner: str | None = None
    lease_pr_number: int | None = None
    lease_gate_id: str | None = None
    error: str | None = None
    recovery: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IndexLockInspection:
    """Read-only inspection of a worktree's Git index lock."""

    schema: str
    worktree_path: str
    index_lock_path: str | None
    lock_exists: bool | None
    lock_mtime_utc: str | None
    lock_age_seconds: float | None
    lock_size_bytes: int | None
    dirty_state: str
    owner_classification: str
    owner_pids: list[int] = field(default_factory=list)
    ownership_error: str | None = None
    inspection_error: str | None = None
    next_action: str = ""


def _run_command(
    args: list[str],
    *,
    cwd: str | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Run a command and capture output, tolerating timeouts and OS errors."""
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout="",
            stderr=f"command timed out after {timeout} seconds",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            args=args,
            returncode=127,
            stdout="",
            stderr=str(exc),
        )


def _proc_lock_owners(  # noqa: C901 - scan failures remain separately observable.
    lock_stat: os.stat_result,
    *,
    proc_root: Path | None = None,
) -> tuple[str, list[int], str | None]:
    """Classify lock ownership by checking process file descriptors.

    A complete scan with no matching descriptor proves the lock is orphaned.
    Permission-limited or otherwise incomplete scans stay ambiguous. Processes
    that disappear while being inspected are ignored because their descriptors
    can no longer own the live lock.
    """
    proc_root = PROC_ROOT if proc_root is None else proc_root
    try:
        processes = [entry for entry in proc_root.iterdir() if entry.name.isdecimal()]
    except OSError as exc:
        return "unavailable", [], f"could not enumerate process table: {exc}"
    if not processes:
        return "unavailable", [], "process table contained no inspectable process entries"

    owners: list[int] = []
    uncertainties: list[str] = []
    for process in processes:
        pid = int(process.name)
        try:
            descriptors = list((process / "fd").iterdir())
        except FileNotFoundError:
            # The process exited between enumerating /proc and reading fd/.
            continue
        except OSError as exc:
            uncertainties.append(f"pid {pid} descriptors unavailable: {exc}")
            continue

        for descriptor in descriptors:
            try:
                descriptor_stat = descriptor.stat()
            except FileNotFoundError:
                # The process closed this descriptor while the scan was running.
                continue
            except OSError as exc:
                uncertainties.append(f"pid {pid} descriptor {descriptor.name} unavailable: {exc}")
                continue
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (
                lock_stat.st_dev,
                lock_stat.st_ino,
            ):
                owners.append(pid)
                break

    if owners:
        return "active", sorted(set(owners)), "; ".join(uncertainties[:3]) or None
    if uncertainties:
        return "ambiguous", [], "; ".join(uncertainties[:3])
    return "proven_orphan", [], None


def _index_lock_next_action(owner_classification: str) -> str:
    """Return a safe, non-mutating recovery action for an index-lock state."""
    actions = {
        "absent": "continue_local_refresh",
        "active": "wait_for_owner_to_finish_then_rerun_preflight",
        "proven_orphan": "preserve_lock_and_worktree_then_request_manual_recovery_review",
        "ambiguous": "preserve_lock_and_worktree_then_resolve_ownership_uncertainty",
        "unavailable": "preserve_lock_and_worktree_then_retry_preflight_when_inspection_is_available",
    }
    return actions.get(owner_classification, actions["unavailable"])


def _unavailable_index_lock_inspection(
    path: str | Path,
    *,
    lock_path: str | None = None,
    lock_exists: bool | None = None,
    error: str,
) -> IndexLockInspection:
    """Build an explicit fail-closed result when lock inspection is incomplete."""
    try:
        worktree_path = str(Path(path).resolve())
    except OSError:
        worktree_path = str(path)
    return IndexLockInspection(
        schema=INDEX_LOCK_SCHEMA_VERSION,
        worktree_path=worktree_path,
        index_lock_path=lock_path,
        lock_exists=lock_exists,
        lock_mtime_utc=None,
        lock_age_seconds=None,
        lock_size_bytes=None,
        dirty_state="unavailable",
        owner_classification="unavailable",
        inspection_error=error,
        next_action=_index_lock_next_action("unavailable"),
    )


def inspect_index_lock(path: str | Path) -> IndexLockInspection:  # noqa: C901 - inspection failures stay distinct.
    """Inspect a worktree's exact Git ``index.lock`` without changing it.

    Git resolves ``--git-path index.lock`` so linked worktrees and configured
    index paths use their actual administrative location. Status inspection
    disables Git's optional index refresh, avoiding lock creation as a side
    effect of this preflight.
    """
    try:
        requested_path = Path(path).resolve(strict=True)
        if not requested_path.is_dir():
            raise OSError(f"worktree path is not a directory: {requested_path}")
    except (OSError, RuntimeError, ValueError) as exc:
        return _unavailable_index_lock_inspection(path, error=f"invalid worktree path: {exc}")

    top_level_result = _run_command(
        ["git", "rev-parse", "--show-toplevel"], cwd=str(requested_path)
    )
    top_level_text = top_level_result.stdout.strip()
    if top_level_result.returncode != 0 or not top_level_text:
        detail = top_level_result.stderr.strip() or "git did not return a worktree root"
        return _unavailable_index_lock_inspection(
            requested_path, error=f"could not resolve Git worktree root: {detail}"
        )
    try:
        worktree_path = Path(top_level_text).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        return _unavailable_index_lock_inspection(
            requested_path, error=f"could not resolve Git worktree root: {exc}"
        )

    lock_result = _run_command(
        ["git", "rev-parse", "--git-path", "index.lock"], cwd=str(worktree_path)
    )
    lock_text = lock_result.stdout.strip()
    if lock_result.returncode != 0 or not lock_text:
        detail = lock_result.stderr.strip() or "git did not return an index-lock path"
        return _unavailable_index_lock_inspection(
            worktree_path, error=f"could not resolve Git index-lock path: {detail}"
        )
    lock_path = Path(lock_text)
    if not lock_path.is_absolute():
        lock_path = worktree_path / lock_path
    # Normalize `.` and `..` without following a possible index.lock symlink;
    # lstat below must describe the exact entry Git resolved.
    lock_path = Path(os.path.abspath(lock_path))

    try:
        lock_stat = lock_path.lstat()
    except FileNotFoundError:
        return IndexLockInspection(
            schema=INDEX_LOCK_SCHEMA_VERSION,
            worktree_path=str(worktree_path),
            index_lock_path=str(lock_path),
            lock_exists=False,
            lock_mtime_utc=None,
            lock_age_seconds=None,
            lock_size_bytes=None,
            dirty_state="not_checked",
            owner_classification="absent",
            next_action=_index_lock_next_action("absent"),
        )
    except OSError as exc:
        return _unavailable_index_lock_inspection(
            worktree_path,
            lock_path=str(lock_path),
            error=f"could not inspect Git index-lock path: {exc}",
        )

    mtime_utc = datetime.fromtimestamp(lock_stat.st_mtime, UTC).isoformat().replace("+00:00", "Z")
    dirty_result = _run_command(
        [
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=str(worktree_path),
    )
    if dirty_result.returncode == 0:
        dirty_state = "dirty" if dirty_result.stdout.strip() else "clean"
        dirty_error = None
    else:
        dirty_state = "unavailable"
        dirty_error = dirty_result.stderr.strip() or "git status failed"

    if not stat.S_ISREG(lock_stat.st_mode):
        owner_classification = "ambiguous"
        owner_pids: list[int] = []
        ownership_error = "index-lock path is not a regular file"
    else:
        owner_classification, owner_pids, ownership_error = _proc_lock_owners(lock_stat)

    return IndexLockInspection(
        schema=INDEX_LOCK_SCHEMA_VERSION,
        worktree_path=str(worktree_path),
        index_lock_path=str(lock_path),
        lock_exists=True,
        lock_mtime_utc=mtime_utc,
        lock_age_seconds=round(time.time() - lock_stat.st_mtime, 6),
        lock_size_bytes=lock_stat.st_size,
        dirty_state=dirty_state,
        owner_classification=owner_classification,
        owner_pids=owner_pids,
        ownership_error=ownership_error,
        inspection_error=dirty_error,
        next_action=_index_lock_next_action(owner_classification),
    )


def _preflight_recovery_packet(inspection: IndexLockInspection) -> dict[str, Any]:
    """Serialize the minimum recovery facts for blocked preflight output."""
    return {
        "worktree_path": inspection.worktree_path,
        "index_lock_path": inspection.index_lock_path,
        "lock_exists": inspection.lock_exists,
        "lock_mtime_utc": inspection.lock_mtime_utc,
        "lock_age_seconds": inspection.lock_age_seconds,
        "lock_size_bytes": inspection.lock_size_bytes,
        "dirty_state": inspection.dirty_state,
        "owner_classification": inspection.owner_classification,
        "owner_pids": inspection.owner_pids,
        "ownership_error": inspection.ownership_error,
        "inspection_error": inspection.inspection_error,
        "next_action": inspection.next_action,
    }


def preflight_gate_worktree(path: str | Path) -> dict[str, Any]:
    """Return a reusable fail-closed preflight for local Git mutations.

    The function is read-only. A present lock or any unavailable inspection
    blocks the caller and carries a structured recovery packet.
    """
    health = verify_gate_worktree(path)
    if not health.exists:
        inspection = _unavailable_index_lock_inspection(
            health.path,
            error="worktree path is missing; its index-lock state cannot be inspected",
        )
        inspection = IndexLockInspection(
            **{
                **asdict(inspection),
                "next_action": "restore_or_reselect_the_worktree_then_rerun_preflight",
            }
        )
        return {
            "schema": PREFLIGHT_SCHEMA_VERSION,
            "status": "worktree_missing",
            "worktree_health": asdict(health),
            "index_lock": asdict(inspection),
            "recovery_packet": _preflight_recovery_packet(inspection),
        }

    inspection = inspect_index_lock(path)
    if inspection.lock_exists is True:
        status = "index_lock_present"
        packet: dict[str, Any] | None = _preflight_recovery_packet(inspection)
    elif inspection.lock_exists is False and inspection.owner_classification == "absent":
        status = "ready"
        packet = None
    else:
        status = "inspection_unavailable"
        packet = _preflight_recovery_packet(inspection)
    return {
        "schema": PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "worktree_health": asdict(health),
        "index_lock": asdict(inspection),
        "recovery_packet": packet,
    }


def _owner_label(owner: str | None, pr_number: int | None, gate_id: str | None) -> str | None:
    """Compose a human-readable cleanup-owner label from lease fields."""
    parts: list[str] = []
    if owner:
        parts.append(f"owner={owner}")
    if pr_number is not None:
        parts.append(f"pr=#{pr_number}")
    if gate_id:
        parts.append(f"gate={gate_id}")
    return "; ".join(parts) if parts else None


def _missing_state_recovery(*, branch: str | None, head_sha: str | None) -> dict[str, Any]:
    """Describe the partial recovery boundary for a missing worktree."""
    return {
        "status": "partial",
        "branch_checkout": "available_from_lease" if branch or head_sha else "unknown",
        "local_state_restored": False,
        "loss_boundary": MISSING_STATE_LOSS_BOUNDARY,
        "next_action": "recover_dirty_state_from_backup_or_handoff",
    }


def _load_active_lease_for_path(path: Path) -> Any | None:
    """Load the active (non-expired) lease recorded for a worktree path.

    Returns the PRGateLease if a live lease exists for the path, else None.
    Failures (unreadable lease, import errors) return None so the caller can
    treat the worktree as missing-without-owner rather than crashing.
    """
    try:
        from scripts.dev.pr_gate_lease import lease_path, legacy_lease_path, load_lease
    except ImportError:
        return None

    resolved = path.resolve()
    lease_files = [lease_path(resolved)]
    legacy_file = legacy_lease_path()
    if legacy_file not in lease_files:
        lease_files.append(legacy_file)

    for lease_file in lease_files:
        if not lease_file.exists():
            continue
        try:
            lease = load_lease(lease_file)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if lease is None or lease.is_expired():
            continue
        # The legacy file was shared by older gate processes. Only attribute it
        # when its embedded path identifies this exact missing worktree.
        if lease_file == legacy_file:
            if not lease.worktree_path or Path(lease.worktree_path).resolve() != resolved:
                continue
        return lease
    return None


def verify_gate_worktree(path: str | Path) -> GateWorktreeHealth:
    """Verify a gate worktree path exists before branch switching/conflict resolution.

    Deterministically reports whether the registered path is intact. If it is
    missing but a live lease still claims the path, the lease owner is reported
    as the cleanup owner so the caller can name who held/removed the worktree.

    Returns a :class:`GateWorktreeHealth` describing the result.
    """
    resolved = Path(path).resolve()
    exists = resolved.exists()

    base = {
        "schema": SCHEMA_VERSION,
        "path": str(resolved),
        "exists": exists,
        "branch": None,
        "head_sha": None,
        "lease_owner": None,
        "lease_pr_number": None,
        "lease_gate_id": None,
        "lease_expires_at": None,
        "cleanup_owner": None,
        "error": None,
        "recovery": {},
    }

    if exists:
        return GateWorktreeHealth(
            **base,
            classification="healthy",
        )

    lease = _load_active_lease_for_path(resolved)
    if lease is not None:
        branch = _normalise_branch(getattr(lease, "head_ref", None))
        head_sha = getattr(lease, "head_sha", None)
        base["classification"] = "missing"
        base["branch"] = branch
        base["head_sha"] = head_sha
        base["lease_owner"] = lease.owner
        base["lease_pr_number"] = lease.pr_number
        base["lease_gate_id"] = lease.gate_id
        base["lease_expires_at"] = lease.expires_at
        base["cleanup_owner"] = _owner_label(lease.owner, lease.pr_number, lease.gate_id)
        base["recovery"] = _missing_state_recovery(branch=branch, head_sha=head_sha)
        return GateWorktreeHealth(**base)

    # Missing and no live lease - cannot attribute ownership.
    base["classification"] = "missing"
    base["recovery"] = _missing_state_recovery(branch=None, head_sha=None)
    return GateWorktreeHealth(**base)


def _add_worktree_under_lifecycle_lock(
    resolved: Path, add_args: list[str]
) -> tuple[subprocess.CompletedProcess | None, str | None]:
    """Run a recovery checkout under the repository lifecycle lock."""
    try:
        from scripts.dev.pr_gate_lease import worktree_lifecycle_lock

        # Recovery mutates the shared Git worktree registry just like creation
        # and cleanup. Hold the same lock through `git worktree add` so a
        # concurrent creator or prune cannot invalidate the recovery decision.
        with worktree_lifecycle_lock():
            if resolved.exists():
                return None, "worktree path appeared before recovery; refusing overwrite"
            return _run_command(add_args), None
    except (ImportError, RuntimeError, OSError) as exc:
        return None, f"worktree lifecycle lock failed: {exc}"


def recreate_gate_worktree(  # noqa: PLR0915 - ordered recovery gates stay explicit
    path: str | Path,
    *,
    ttl_hours: float | None = None,
) -> GateWorktreeRecreate:
    """Recreate a missing gate worktree from a surviving lease and re-register it.

    If the worktree path still exists, no recreation is performed (recreated=False,
    healthy). If it is missing but a live lease recorded the branch and owner, the
    linked worktree is recreated with ``git worktree add`` and the lease is
    refreshed so the gate can resume against the original branch.

    Important: this does NOT restore dirty/untracked state that was lost when the
    worktree disappeared. It restores the *branch checkout* so the gate process can
    continue. Callers must treat lost dirty state as unrecoverable and report it.
    """
    resolved = Path(path).resolve()

    base = {
        "schema": SCHEMA_VERSION,
        "path": str(resolved),
        "branch": None,
        "head_sha": None,
        "lease_owner": None,
        "lease_pr_number": None,
        "lease_gate_id": None,
        "error": None,
        "recovery": {},
    }

    if resolved.exists():
        return GateWorktreeRecreate(**base, recreated=False)

    lease = _load_active_lease_for_path(resolved)
    if lease is None:
        base["recreated"] = False
        base["error"] = "worktree missing and no live lease on record; cannot recreate safely"
        base["recovery"] = _missing_state_recovery(branch=None, head_sha=None)
        return GateWorktreeRecreate(**base)

    if ttl_hours is not None and (
        isinstance(ttl_hours, bool)
        or not isinstance(ttl_hours, (int, float))
        or not math.isfinite(ttl_hours)
        or ttl_hours <= 0
    ):
        base["recreated"] = False
        base["lease_owner"] = lease.owner
        base["lease_pr_number"] = lease.pr_number
        base["lease_gate_id"] = lease.gate_id
        base["recovery"] = _missing_state_recovery(
            branch=_normalise_branch(getattr(lease, "head_ref", None)),
            head_sha=getattr(lease, "head_sha", None),
        )
        base["error"] = "ttl_hours must be a positive finite number"
        return GateWorktreeRecreate(**base)

    branch, head_sha = _restore_metadata(lease)
    if not branch and not head_sha:
        base["recreated"] = False
        base["lease_owner"] = lease.owner
        base["lease_pr_number"] = lease.pr_number
        base["lease_gate_id"] = lease.gate_id
        base["recovery"] = _missing_state_recovery(branch=branch, head_sha=head_sha)
        base["error"] = (
            "live lease found but no branch or commit on record; cannot recreate deterministically"
        )
        return GateWorktreeRecreate(**base)

    add_args, add_error = _recovery_add_args(branch, head_sha, resolved)
    if add_error is not None:
        base["recreated"] = False
        base["branch"] = branch
        base["head_sha"] = head_sha
        base["lease_owner"] = lease.owner
        base["lease_pr_number"] = lease.pr_number
        base["lease_gate_id"] = lease.gate_id
        base["recovery"] = _missing_state_recovery(branch=branch, head_sha=head_sha)
        base["error"] = add_error
        return GateWorktreeRecreate(**base)

    result, recovery_error = _add_worktree_under_lifecycle_lock(resolved, add_args)
    if recovery_error is not None:
        base["recreated"] = False
        base["branch"] = branch
        base["head_sha"] = head_sha
        base["lease_owner"] = lease.owner
        base["lease_pr_number"] = lease.pr_number
        base["lease_gate_id"] = lease.gate_id
        base["recovery"] = _missing_state_recovery(branch=branch, head_sha=head_sha)
        base["error"] = recovery_error
        return GateWorktreeRecreate(**base)

    assert result is not None
    if result.returncode != 0:
        base["recreated"] = False
        base["branch"] = branch
        base["head_sha"] = head_sha
        base["lease_owner"] = lease.owner
        base["lease_pr_number"] = lease.pr_number
        base["lease_gate_id"] = lease.gate_id
        base["recovery"] = _missing_state_recovery(branch=branch, head_sha=head_sha)
        base["error"] = f"git worktree add failed: {result.stderr.strip() or result.stdout.strip()}"
        return GateWorktreeRecreate(**base)

    try:
        from scripts.dev.pr_gate_lease import heartbeat

        heartbeat(worktree_path=resolved, extend_hours=ttl_hours)
    except (ImportError, RuntimeError, OSError, TypeError, ValueError) as exc:
        base["recreated"] = False
        base["branch"] = branch
        base["head_sha"] = head_sha
        base["lease_owner"] = lease.owner
        base["lease_pr_number"] = lease.pr_number
        base["lease_gate_id"] = lease.gate_id
        base["recovery"] = _missing_state_recovery(branch=branch, head_sha=head_sha)
        base["error"] = f"worktree recreated but lease refresh failed: {exc}"
        return GateWorktreeRecreate(**base)

    base["recreated"] = True
    base["branch"] = branch
    base["head_sha"] = head_sha
    base["lease_owner"] = lease.owner
    base["lease_pr_number"] = lease.pr_number
    base["lease_gate_id"] = lease.gate_id
    base["recovery"] = _missing_state_recovery(branch=branch, head_sha=head_sha)
    return GateWorktreeRecreate(**base)


def _normalise_branch(branch: str | None) -> str | None:
    """Return a usable short branch name, excluding detached-HEAD markers."""
    if not branch:
        return None
    branch = branch.strip()
    if branch in {"HEAD", "(detached)"}:
        return None
    if branch.startswith("refs/heads/"):
        branch = branch.removeprefix("refs/heads/")
    return branch or None


def _registered_worktree_metadata(path: Path) -> tuple[str | None, str | None]:
    """Recover a branch and commit from Git's worktree registry when it remains."""
    result = _run_command(["git", "worktree", "list", "--porcelain"])
    if result.returncode != 0:
        return None, None

    current_path: str | None = None
    current_sha: str | None = None
    current_branch: str | None = None

    def finish_record() -> tuple[str | None, str | None] | None:
        if current_path is None or Path(current_path).resolve() != path.resolve():
            return None
        return _normalise_branch(current_branch), current_sha

    for line in (*result.stdout.splitlines(), ""):
        if line.startswith("worktree "):
            current_path = line.removeprefix("worktree ").strip()
            current_sha = None
            current_branch = None
        elif line.startswith("HEAD "):
            current_sha = line.removeprefix("HEAD ").strip() or None
        elif line.startswith("branch "):
            current_branch = line.removeprefix("branch ").strip() or None
        elif not line:
            record = finish_record()
            if record is not None:
                return record
            current_path = None
            current_sha = None
            current_branch = None
    return None, None


def _restore_metadata(lease: Any) -> tuple[str | None, str | None]:
    """Select persisted or Git-registered branch/commit metadata for recreation."""
    branch = _normalise_branch(getattr(lease, "head_ref", None))
    head_sha = getattr(lease, "head_sha", None)
    if branch or head_sha:
        return branch, head_sha

    # Older leases may predate persisted head metadata. Git can still retain the
    # worktree registry entry after the directory disappeared, so use it when
    # available; otherwise fail closed instead of rebuilding from origin/main.
    if lease.worktree_path:
        return _registered_worktree_metadata(Path(lease.worktree_path))
    return None, None


def _local_branch_exists(branch: str) -> bool:
    """Return whether a local branch ref exists for the restore operation."""
    result = _run_command(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"])
    return result.returncode == 0


def _recovery_add_args(
    branch: str | None, head_sha: str | None, resolved: Path
) -> tuple[list[str], str | None]:
    """Build a deterministic ``git worktree add`` command for recovery."""
    if branch:
        if _local_branch_exists(branch):
            return ["git", "worktree", "add", "--force", str(resolved), branch], None
        if not head_sha:
            return [], f"leased branch {branch!r} is unavailable and has no commit fallback"
        return [
            "git",
            "worktree",
            "add",
            "--force",
            "-b",
            branch,
            str(resolved),
            head_sha,
        ], None
    return [
        "git",
        "worktree",
        "add",
        "--force",
        "--detach",
        str(resolved),
        head_sha,
    ], None


def _branch_for_path(path: str | Path) -> str | None:
    """Return the checked-out branch for an existing worktree, or None.

    Kept as a small compatibility helper for callers/tests that inspect a live
    worktree; missing-path recreation uses persisted or registry metadata instead.
    """
    result = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(path))
    if result.returncode != 0:
        return None
    return _normalise_branch(result.stdout)


def ensure_gate_worktree(
    path: str | Path,
    *,
    ttl_hours: float | None = None,
) -> tuple[GateWorktreeHealth, GateWorktreeRecreate | None]:
    """Verify a gate worktree; recreate from the surviving lease if missing.

    Returns ``(health, recreate_result)``. ``recreate_result`` is None when the
    worktree was already healthy (no recreate attempted).
    """
    health = verify_gate_worktree(path)
    if health.exists:
        return health, None

    recreate = recreate_gate_worktree(path, ttl_hours=ttl_hours)
    return health, recreate


def _format_health(health: GateWorktreeHealth) -> str:
    """Format a health result as human-readable text."""
    lines = [
        f"Gate Worktree Health (schema: {health.schema})",
        f"  Path: {health.path}",
        f"  Exists: {health.exists}",
        f"  Classification: {health.classification}",
    ]
    if health.branch:
        lines.append(f"  Branch: {health.branch}")
    if health.head_sha:
        lines.append(f"  HEAD: {health.head_sha}")
    if health.cleanup_owner:
        lines.append(f"  Cleanup owner: {health.cleanup_owner}")
    if health.lease_pr_number is not None:
        lines.append(f"  Lease PR: #{health.lease_pr_number}")
    if health.lease_gate_id:
        lines.append(f"  Lease gate: {health.lease_gate_id}")
    if health.lease_expires_at:
        lines.append(f"  Lease expires: {health.lease_expires_at}")
    if health.error:
        lines.append(f"  Error: {health.error}")
    if health.recovery.get("loss_boundary"):
        lines.append(f"  Recovery loss boundary: {health.recovery['loss_boundary']}")
    return "\n".join(lines)


def _format_recreate(recreate: GateWorktreeRecreate) -> str:
    """Format a recreate result as human-readable text."""
    lines = [
        f"Gate Worktree Recreate (schema: {recreate.schema})",
        f"  Path: {recreate.path}",
        f"  Recreated: {recreate.recreated}",
    ]
    if recreate.branch:
        lines.append(f"  Branch: {recreate.branch}")
    if recreate.head_sha:
        lines.append(f"  HEAD: {recreate.head_sha}")
    if recreate.lease_pr_number is not None:
        lines.append(f"  Lease PR: #{recreate.lease_pr_number}")
    if recreate.error:
        lines.append(f"  Error: {recreate.error}")
    if recreate.recovery.get("loss_boundary"):
        lines.append(f"  Recovery loss boundary: {recreate.recovery['loss_boundary']}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify", help="Verify a gate worktree path exists; report owner if missing"
    )
    verify_parser.add_argument(
        "--path", required=True, help="Registered gate worktree path to verify"
    )
    verify_parser.add_argument("--json", action="store_true", help="Emit JSON")

    ensure_parser = subparsers.add_parser(
        "ensure", help="Verify, and recreate from a surviving lease if missing"
    )
    ensure_parser.add_argument(
        "--path", required=True, help="Registered gate worktree path to ensure"
    )
    ensure_parser.add_argument(
        "--ttl-hours",
        type=float,
        default=None,
        help="Lease TTL to (re)apply after recreate, in hours",
    )
    ensure_parser.add_argument("--json", action="store_true", help="Emit JSON")

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Inspect worktree health and index.lock before local Git mutation",
    )
    preflight_parser.add_argument(
        "--path", required=True, help="Git worktree path to inspect without mutation"
    )
    preflight_parser.add_argument("--json", action="store_true", help="Emit JSON")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - CLI status branches fail closed.
    """CLI entry point."""
    args = _parse_args(argv)

    if args.command == "verify":
        health = verify_gate_worktree(args.path)
        if args.json:
            print(json.dumps(asdict(health), indent=2, sort_keys=True))
        else:
            print(_format_health(health))
        return 0 if health.exists else 1

    if args.command == "ensure":
        health, recreate = ensure_gate_worktree(args.path, ttl_hours=args.ttl_hours)
        if args.json:
            payload = {
                "health": asdict(health),
                "recreate": asdict(recreate) if recreate is not None else None,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(_format_health(health))
            if recreate is not None:
                print(_format_recreate(recreate))
        if recreate is not None:
            return 0 if recreate.recreated else 1
        return 0 if health.exists else 1

    if args.command == "preflight":
        payload = preflight_gate_worktree(args.path)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Gate Worktree Preflight (schema: {payload['schema']})")
            print(f"  Path: {payload['worktree_health']['path']}")
            print(f"  Status: {payload['status']}")
            inspection = payload["index_lock"]
            print(f"  Index lock: {inspection['index_lock_path'] or 'unavailable'}")
            print(f"  Owner classification: {inspection['owner_classification']}")
            if payload["recovery_packet"] is not None:
                print(f"  Next action: {payload['recovery_packet']['next_action']}")
        if payload["status"] == "ready":
            return 0
        if payload["status"] == "inspection_unavailable":
            return 2
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
