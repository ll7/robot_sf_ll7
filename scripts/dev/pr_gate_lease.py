#!/usr/bin/env python3
"""PR-gate worktree lease management for protecting active gates from cleanup.

This module provides a lease/heartbeat mechanism so automated worktree pruning
cannot remove a live PR-gate worktree. A lease file marks a worktree as actively
in use by a PR gate session.

Usage:
    # Create a lease (default 2 hour TTL)
    python scripts/dev/pr_gate_lease.py create --pr 5715

    # Refresh an existing lease (heartbeat)
    python scripts/dev/pr_gate_lease.py heartbeat

    # Release a lease explicitly
    python scripts/dev/pr_gate_lease.py release

    # Check if current worktree has an active lease
    python scripts/dev/pr_gate_lease.py status

    # Check if a lease is active (for scripts)
    python scripts/dev/pr_gate_lease.py is-active && echo "active" || echo "not active"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = "pr_gate_lease.v1"
DEFAULT_TTL_HOURS = 2
LEASE_FILENAME = ".pr-gate-lease.json"
LEGACY_LEASE_FILENAME = LEASE_FILENAME


@dataclass(frozen=True, slots=True)
class PRGateLease:
    """A PR-gate worktree lease record."""

    schema: str
    created_at: str  # ISO 8601 timestamp
    expires_at: str  # ISO 8601 timestamp
    pr_number: int | None
    gate_id: str | None
    owner: str | None  # e.g., username, session ID
    last_heartbeat: str  # ISO 8601 timestamp
    worktree_path: str | None = None

    def is_expired(self) -> bool:
        """Check if the lease has expired."""
        expiry = datetime.fromisoformat(self.expires_at)
        return datetime.now(UTC) > expiry

    def time_until_expiry_seconds(self) -> float:
        """Return seconds until lease expires (negative if expired)."""
        expiry = datetime.fromisoformat(self.expires_at)
        delta = expiry - datetime.now(UTC)
        return delta.total_seconds()


def _git_common_dir() -> Path:
    """Get the Git common directory for the current repository."""
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get git common dir: {result.stderr.strip()}")

    common_dir = Path(result.stdout.strip())
    if not common_dir.is_absolute():
        common_dir = Path.cwd() / common_dir
    return common_dir.resolve()


def _repo_root() -> Path:
    """Get the repository root directory."""
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get repo root: {result.stderr.strip()}")
    return Path(result.stdout.strip()).resolve()


def _validate_duration(val: float, name: str) -> None:
    """Validate that a duration is positive, finite, and a valid number."""
    import math

    if val is None:
        raise ValueError(f"{name} cannot be None")
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a numeric value")
    if math.isnan(val) or not math.isfinite(val):
        raise ValueError(f"{name} must be finite")
    if val <= 0:
        raise ValueError(f"{name} must be strictly positive")


def lease_path(worktree_path: Path | None = None) -> Path:
    """Return the path to the lease file in the Git common directory."""
    import hashlib

    if worktree_path is None:
        try:
            worktree_path = _repo_root()
        except RuntimeError:
            worktree_path = Path.cwd()

    resolved = Path(worktree_path).resolve()
    path_bytes = str(resolved).encode("utf-8")
    h = hashlib.sha256(path_bytes).hexdigest()
    filename = f".pr-gate-lease-{h}.json"
    return _git_common_dir() / filename


def legacy_lease_path() -> Path:
    """Return the pre-isolation lease path used by older gate processes."""
    return _git_common_dir() / LEGACY_LEASE_FILENAME


def create_lease(
    pr_number: int | None = None,
    gate_id: str | None = None,
    owner: str | None = None,
    ttl_hours: float = DEFAULT_TTL_HOURS,
) -> PRGateLease:
    """Create a new PR-gate lease."""
    _validate_duration(ttl_hours, "ttl_hours")

    now = datetime.now(UTC)
    from datetime import timedelta

    expires = now + timedelta(hours=ttl_hours)

    try:
        wt_path = str(_repo_root().resolve())
    except RuntimeError:
        wt_path = str(Path.cwd().resolve())

    lease = PRGateLease(
        schema=SCHEMA_VERSION,
        created_at=now.isoformat(),
        expires_at=expires.isoformat(),
        pr_number=pr_number,
        gate_id=gate_id,
        owner=owner,
        last_heartbeat=now.isoformat(),
        worktree_path=wt_path,
    )

    save_lease(lease)
    return lease


def load_lease(path: Path | None = None) -> PRGateLease | None:
    """Load the current lease if it exists."""
    if path is None:
        path = lease_path()
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise TypeError("lease JSON must be an object")
        import inspect

        valid_keys = inspect.signature(PRGateLease).parameters.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        lease = PRGateLease(**filtered_data)
        for timestamp in (lease.created_at, lease.expires_at, lease.last_heartbeat):
            parsed = datetime.fromisoformat(timestamp)
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("lease timestamps must be timezone-aware")
        return lease
    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Invalid lease file: {exc}") from exc


def save_lease(lease: PRGateLease, path: Path | None = None) -> None:
    """Save a lease to disk atomically."""
    import tempfile

    if path is None:
        path = lease_path()

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path_str = tempfile.mkstemp(dir=parent, prefix=".pr-gate-lease-tmp-")
    temp_path = Path(temp_path_str)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(lease), f, indent=2)
            f.write("\n")
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def heartbeat(
    *,
    extend_hours: float | None = None,
) -> PRGateLease:
    """Refresh the heartbeat on an existing lease.

    Args:
        extend_hours: If provided, extend the expiry by this many hours from now.
                     If None, keep the original expiry time.

    Returns:
        The updated lease.

    Raises:
        RuntimeError: If no lease exists.
    """
    if extend_hours is not None:
        _validate_duration(extend_hours, "extend_hours")

    lease = load_lease()
    if lease is None:
        raise RuntimeError("No active lease to refresh")

    now = datetime.now(UTC)
    new_heartbeat = now.isoformat()

    if extend_hours is not None:
        from datetime import timedelta

        new_expiry = now + timedelta(hours=extend_hours)
        expires_at = new_expiry.isoformat()
    else:
        expires_at = lease.expires_at

    try:
        wt_path = lease.worktree_path or str(_repo_root().resolve())
    except RuntimeError:
        wt_path = lease.worktree_path or str(Path.cwd().resolve())

    updated = PRGateLease(
        schema=lease.schema,
        created_at=lease.created_at,
        expires_at=expires_at,
        pr_number=lease.pr_number,
        gate_id=lease.gate_id,
        owner=lease.owner,
        last_heartbeat=new_heartbeat,
        worktree_path=wt_path,
    )
    save_lease(updated)
    return updated


def release_lease() -> bool:
    """Release the current lease by deleting the lease file.

    Returns:
        True if a lease was released, False if no lease existed.
    """
    path = lease_path()
    if path.exists():
        path.unlink()
        return True
    return False


def is_active() -> bool:
    """Check if there is an active (non-expired) lease."""
    lease = load_lease()
    if lease is None:
        return False
    return not lease.is_expired()


def status() -> dict:
    """Get the current lease status."""
    lease = load_lease()
    if lease is None:
        return {"active": False, "reason": "no_lease"}

    if lease.is_expired():
        return {
            "active": False,
            "reason": "expired",
            "lease": asdict(lease),
            "expired_at": lease.expires_at,
        }

    return {
        "active": True,
        "lease": asdict(lease),
        "seconds_until_expiry": lease.time_until_expiry_seconds(),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create subcommand
    create_parser = subparsers.add_parser("create", help="Create a new PR-gate lease")
    create_parser.add_argument("--pr", type=int, help="PR number")
    create_parser.add_argument("--gate-id", type=str, help="Gate identifier")
    create_parser.add_argument("--owner", type=str, help="Owner identifier")
    create_parser.add_argument(
        "--ttl-hours",
        type=float,
        default=DEFAULT_TTL_HOURS,
        help=f"Lease TTL in hours (default: {DEFAULT_TTL_HOURS})",
    )

    # heartbeat subcommand
    hb_parser = subparsers.add_parser("heartbeat", help="Refresh lease heartbeat")
    hb_parser.add_argument(
        "--extend-hours",
        type=float,
        help="Extend expiry by this many hours from now",
    )

    # release subcommand
    subparsers.add_parser("release", help="Release the current lease")

    # status subcommand
    subparsers.add_parser("status", help="Show current lease status")

    # is-active subcommand
    subparsers.add_parser(
        "is-active",
        help="Exit 0 if active lease exists, 1 otherwise",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)

    try:
        if args.command == "create":
            lease = create_lease(
                pr_number=args.pr,
                gate_id=args.gate_id,
                owner=args.owner,
                ttl_hours=args.ttl_hours,
            )
            print(f"Created lease for PR #{lease.pr_number or 'N/A'}")
            print(f"  Gate ID: {lease.gate_id or 'N/A'}")
            print(f"  Owner: {lease.owner or 'N/A'}")
            print(f"  Expires: {lease.expires_at}")
            print(f"  Lease file: {lease_path()}")

        elif args.command == "heartbeat":
            lease = heartbeat(extend_hours=args.extend_hours)
            print(f"Refreshed heartbeat for PR #{lease.pr_number or 'N/A'}")
            print(f"  Expires: {lease.expires_at}")
            print(f"  Last heartbeat: {lease.last_heartbeat}")

        elif args.command == "release":
            if release_lease():
                print("Lease released")
            else:
                print("No active lease to release")

        elif args.command == "status":
            stat = status()
            if stat["active"]:
                lease_data = stat["lease"]
                print(f"Active lease for PR #{lease_data.get('pr_number') or 'N/A'}")
                print(f"  Gate ID: {lease_data.get('gate_id') or 'N/A'}")
                print(f"  Owner: {lease_data.get('owner') or 'N/A'}")
                print(f"  Created: {lease_data['created_at']}")
                print(f"  Expires: {lease_data['expires_at']}")
                print(f"  Last heartbeat: {lease_data['last_heartbeat']}")
                print(f"  Seconds until expiry: {stat['seconds_until_expiry']:.1f}")
            else:
                print(f"No active lease ({stat['reason']})")
                if "lease" in stat:
                    print(f"  Expired at: {stat['expired_at']}")

        elif args.command == "is-active":
            return 0 if is_active() else 1

        return 0

    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
