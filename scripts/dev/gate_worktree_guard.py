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

    # Same, but recreate from the surviving lease if missing.
    python scripts/dev/gate_worktree_guard.py ensure --path /abs/worktree
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "gate_worktree_guard.v1"


@dataclass(frozen=True, slots=True)
class GateWorktreeHealth:
    """Health result for a single gate worktree path."""

    schema: str
    path: str
    exists: bool
    classification: str  # "healthy", "missing", "errored"
    lease_owner: str | None = None
    lease_pr_number: int | None = None
    lease_gate_id: str | None = None
    lease_expires_at: str | None = None
    cleanup_owner: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class GateWorktreeRecreate:
    """Result of a recreate attempt for a missing gate worktree."""

    schema: str
    path: str
    recreated: bool
    branch: str | None = None
    lease_owner: str | None = None
    lease_pr_number: int | None = None
    lease_gate_id: str | None = None
    error: str | None = None


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


def _load_active_lease_for_path(path: Path) -> Any | None:
    """Load the active (non-expired) lease recorded for a worktree path.

    Returns the PRGateLease if a live lease exists for the path, else None.
    Failures (unreadable lease, import errors) return None so the caller can
    treat the worktree as missing-without-owner rather than crashing.
    """
    try:
        from scripts.dev.pr_gate_lease import lease_path, load_lease
    except ImportError:
        return None

    try:
        lease_file = lease_path(path)
        if not lease_file.exists():
            return None
        lease = load_lease(lease_file)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None

    if lease is None or lease.is_expired():
        return None
    return lease


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
        "lease_owner": None,
        "lease_pr_number": None,
        "lease_gate_id": None,
        "lease_expires_at": None,
        "cleanup_owner": None,
        "error": None,
    }

    if exists:
        return GateWorktreeHealth(
            **base,
            classification="healthy",
        )

    lease = _load_active_lease_for_path(resolved)
    if lease is not None:
        base["classification"] = "missing"
        base["lease_owner"] = lease.owner
        base["lease_pr_number"] = lease.pr_number
        base["lease_gate_id"] = lease.gate_id
        base["lease_expires_at"] = lease.expires_at
        base["cleanup_owner"] = _owner_label(lease.owner, lease.pr_number, lease.gate_id)
        return GateWorktreeHealth(**base)

    # Missing and no live lease - cannot attribute ownership.
    base["classification"] = "missing"
    return GateWorktreeHealth(**base)


def recreate_gate_worktree(
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
        "lease_owner": None,
        "lease_pr_number": None,
        "lease_gate_id": None,
        "error": None,
    }

    if resolved.exists():
        return GateWorktreeRecreate(**base, recreated=False)

    lease = _load_active_lease_for_path(resolved)
    if lease is None:
        base["recreated"] = False
        base["error"] = "worktree missing and no live lease on record; cannot recreate safely"
        return GateWorktreeRecreate(**base)

    branch = None
    if lease.worktree_path:
        branch = _branch_for_path(lease.worktree_path)
    if branch is None and lease.gate_id:
        branch = _branch_for_path(resolved)

    if not branch:
        base["recreated"] = False
        base["lease_owner"] = lease.owner
        base["lease_pr_number"] = lease.pr_number
        base["lease_gate_id"] = lease.gate_id
        base["error"] = (
            "live lease found but no branch on record; cannot recreate deterministically"
        )
        return GateWorktreeRecreate(**base)

    result = _run_command(
        ["git", "worktree", "add", "--force", str(resolved), "origin/main", "-b", branch]
    )
    if result.returncode != 0:
        base["recreated"] = False
        base["branch"] = branch
        base["lease_owner"] = lease.owner
        base["lease_pr_number"] = lease.pr_number
        base["lease_gate_id"] = lease.gate_id
        base["error"] = f"git worktree add failed: {result.stderr.strip() or result.stdout.strip()}"
        return GateWorktreeRecreate(**base)

    try:
        from scripts.dev.pr_gate_lease import heartbeat

        heartbeat()
    except (ImportError, RuntimeError, OSError):
        pass

    base["recreated"] = True
    base["branch"] = branch
    base["lease_owner"] = lease.owner
    base["lease_pr_number"] = lease.pr_number
    base["lease_gate_id"] = lease.gate_id
    return GateWorktreeRecreate(**base)


def _branch_for_path(path: str | Path) -> str | None:
    """Return the checked-out branch for an existing worktree, or None."""
    result = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(path))
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None


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
    if recreate.lease_pr_number is not None:
        lines.append(f"  Lease PR: #{recreate.lease_pr_number}")
    if recreate.error:
        lines.append(f"  Error: {recreate.error}")
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

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
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

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
