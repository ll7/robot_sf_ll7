#!/usr/bin/env python3
"""Worktree lease management for active PR gates and delegated tasks.

The historical public name and schema remain ``pr_gate_lease.v1`` for compatibility,
but the path-scoped lease is also the repository's active-worktree ownership marker.
Lease mutation and repository-owned cleanup serialize on the same lifecycle lock used
by ``create_worktree.sh`` so a cleanup check cannot race a new task lease.

Usage:
    # Create a lease for the current worktree (default 2 hour TTL)
    python scripts/dev/pr_gate_lease.py create --pr 5715

    # Create a task lease for an explicitly named worktree from a stable checkout
    python scripts/dev/pr_gate_lease.py create --worktree /path/to/worktree --gate-id issue-8553

    # Refresh an existing lease (heartbeat)
    python scripts/dev/pr_gate_lease.py heartbeat --worktree /path/to/worktree --extend-hours 2

    # Release a lease explicitly
    python scripts/dev/pr_gate_lease.py release --worktree /path/to/worktree

    # Check if a worktree has an active lease
    python scripts/dev/pr_gate_lease.py status --worktree /path/to/worktree
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

SCHEMA_VERSION = "pr_gate_lease.v1"
DEFAULT_TTL_HOURS = 2
LEASE_FILENAME = ".pr-gate-lease.json"
LEGACY_LEASE_FILENAME = LEASE_FILENAME
WORKTREE_LOCK_FILENAME = "robot-sf-create-worktree.lock"
WORKTREE_LOCK_FD_ENV = "ROBOT_SF_WORKTREE_LOCK_FD"


@dataclass(frozen=True, slots=True)
class PRGateLease:
    """A path-scoped active-worktree lease record.

    ``gate_id`` remains the compatibility field for the logical task/run identifier.
    Non-gate delegated worktrees created by ``create_worktree.sh --task-id`` store the
    task id here and in ``owner`` so cleanup diagnostics remain attributable without a
    schema migration.
    """

    schema: str
    created_at: str  # ISO 8601 timestamp
    expires_at: str  # ISO 8601 timestamp
    pr_number: int | None
    gate_id: str | None
    owner: str | None  # e.g., username, session ID, or task ID
    last_heartbeat: str  # ISO 8601 timestamp
    worktree_path: str | None = None
    head_ref: str | None = None  # short branch name; None for detached HEAD
    head_sha: str | None = None  # commit to restore when the branch is unavailable/detached

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
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get repo root: {result.stderr.strip()}")
    return Path(result.stdout.strip()).resolve()


def worktree_lifecycle_lock_path() -> Path:
    """Return the shared repository worktree lifecycle lock path.

    The filename is intentionally unchanged because ``create_worktree.sh`` and the
    portable lock helper already use this exact lock identity.
    """
    return _git_common_dir() / WORKTREE_LOCK_FILENAME


@contextmanager
def worktree_lifecycle_lock() -> Iterator[None]:
    """Serialize lease mutation with worktree creation and guarded cleanup.

    ``create_worktree.sh`` may already hold the lock. Its portable path exposes the
    inherited descriptor through ``ROBOT_SF_WORKTREE_LOCK_FD``; the native ``flock``
    path now does the same. In that case verify ownership and reuse the lock instead
    of trying to acquire a second open-file description and deadlocking.
    """
    lock_path = worktree_lifecycle_lock_path()
    inherited_fd = os.environ.get(WORKTREE_LOCK_FD_ENV)
    if inherited_fd:
        try:
            fd = int(inherited_fd)
            if fd < 0:
                raise ValueError
            if __package__:
                from scripts.dev.worktree_creation_lock import verify_lock_fd
            else:
                # Direct execution must load this checkout's sibling even with
                # Python's safe-path mode or another checkout's editable install.
                # This interpreter-local path never alters the worker environment.
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from worktree_creation_lock import verify_lock_fd

            verify_lock_fd(str(lock_path), fd)
        except (ImportError, OSError, ValueError) as exc:
            raise RuntimeError(f"Invalid inherited worktree lifecycle lock: {exc}") from exc
        yield
        return

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - supported repository hosts are POSIX
        raise RuntimeError(
            "fcntl is unavailable; cannot guard worktree lifecycle mutation"
        ) from exc

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise RuntimeError(f"Failed to hold worktree lifecycle lock {lock_path}: {exc}") from exc


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


def _head_metadata(worktree_path: Path) -> tuple[str | None, str | None]:
    """Return the checked-out branch/ref and commit for a worktree when available."""
    try:
        ref_result = subprocess.run(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None

    head_ref = ref_result.stdout.strip() if ref_result.returncode == 0 else None
    head_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else None
    return head_ref or None, head_sha or None


def lease_path(worktree_path: Path | None = None) -> Path:
    """Return the path-hashed lease file in the Git common directory."""
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


def _resolve_worktree(worktree_path: Path | str | None) -> Path:
    if worktree_path is None:
        return _repo_root().resolve()
    resolved = Path(worktree_path).resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"Worktree path is not an existing directory: {resolved}")
    return resolved


def create_lease(
    pr_number: int | None = None,
    gate_id: str | None = None,
    owner: str | None = None,
    ttl_hours: float = DEFAULT_TTL_HOURS,
    *,
    worktree_path: Path | str | None = None,
) -> PRGateLease:
    """Create a new path-scoped active-worktree lease under the lifecycle lock."""
    _validate_duration(ttl_hours, "ttl_hours")

    with worktree_lifecycle_lock():
        now = datetime.now(UTC)
        from datetime import timedelta

        expires = now + timedelta(hours=ttl_hours)
        wt_root = _resolve_worktree(worktree_path)
        wt_path = str(wt_root)
        head_ref, head_sha = _head_metadata(wt_root)

        lease = PRGateLease(
            schema=SCHEMA_VERSION,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
            pr_number=pr_number,
            gate_id=gate_id,
            owner=owner,
            last_heartbeat=now.isoformat(),
            worktree_path=wt_path,
            head_ref=head_ref,
            head_sha=head_sha,
        )
        save_lease(lease, lease_path(wt_root))
        return lease


def load_lease(path: Path | None = None) -> PRGateLease | None:
    """Load a lease if it exists."""
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
    except (OSError, TypeError, ValueError):
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def heartbeat(
    *,
    extend_hours: float | None = None,
    worktree_path: Path | str | None = None,
) -> PRGateLease:
    """Refresh an existing lease while serialized against cleanup."""
    if extend_hours is not None:
        _validate_duration(extend_hours, "extend_hours")

    with worktree_lifecycle_lock():
        lease_file = lease_path(Path(worktree_path)) if worktree_path is not None else lease_path()
        lease = load_lease(lease_file)
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

        if worktree_path is not None:
            wt_root = Path(worktree_path).resolve()
            wt_path = str(wt_root)
        else:
            try:
                wt_root = (
                    Path(lease.worktree_path).resolve() if lease.worktree_path else _repo_root()
                )
                wt_path = str(wt_root)
            except RuntimeError:
                wt_root = Path.cwd().resolve()
                wt_path = lease.worktree_path or str(wt_root)

        detected_ref, detected_sha = _head_metadata(wt_root) if wt_root.exists() else (None, None)

        updated = PRGateLease(
            schema=lease.schema,
            created_at=lease.created_at,
            expires_at=expires_at,
            pr_number=lease.pr_number,
            gate_id=lease.gate_id,
            owner=lease.owner,
            last_heartbeat=new_heartbeat,
            worktree_path=wt_path,
            head_ref=lease.head_ref or detected_ref,
            head_sha=lease.head_sha or detected_sha,
        )
        save_lease(updated, lease_file)
        return updated


def release_lease(*, worktree_path: Path | str | None = None) -> bool:
    """Release a path-scoped lease while serialized against cleanup."""
    with worktree_lifecycle_lock():
        path = lease_path(Path(worktree_path)) if worktree_path is not None else lease_path()
        if path.exists():
            path.unlink()
            return True
        return False


def is_active(*, worktree_path: Path | str | None = None) -> bool:
    """Check if there is an active (non-expired) lease."""
    path = lease_path(Path(worktree_path)) if worktree_path is not None else lease_path()
    lease = load_lease(path)
    if lease is None:
        return False
    return not lease.is_expired()


def status(*, worktree_path: Path | str | None = None) -> dict:
    """Get lease status for the current or explicitly selected worktree."""
    path = lease_path(Path(worktree_path)) if worktree_path is not None else lease_path()
    lease = load_lease(path)
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


def _add_worktree_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--worktree",
        type=Path,
        help="Explicit worktree path; run from any surviving checkout of the repository",
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create an active-worktree lease")
    create_parser.add_argument("--pr", type=int, help="PR number")
    create_parser.add_argument("--gate-id", type=str, help="Gate/task identifier")
    create_parser.add_argument("--owner", type=str, help="Owner identifier")
    create_parser.add_argument(
        "--ttl-hours",
        type=float,
        default=DEFAULT_TTL_HOURS,
        help=f"Lease TTL in hours (default: {DEFAULT_TTL_HOURS})",
    )
    _add_worktree_argument(create_parser)

    hb_parser = subparsers.add_parser("heartbeat", help="Refresh lease heartbeat")
    hb_parser.add_argument(
        "--extend-hours",
        type=float,
        help="Extend expiry by this many hours from now",
    )
    _add_worktree_argument(hb_parser)

    release_parser = subparsers.add_parser("release", help="Release the selected lease")
    _add_worktree_argument(release_parser)

    status_parser = subparsers.add_parser("status", help="Show selected lease status")
    _add_worktree_argument(status_parser)

    active_parser = subparsers.add_parser(
        "is-active",
        help="Exit 0 if active lease exists, 1 otherwise",
    )
    _add_worktree_argument(active_parser)

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
                worktree_path=args.worktree,
            )
            print(f"Created lease for PR #{lease.pr_number or 'N/A'}")
            print(f"  Gate/task ID: {lease.gate_id or 'N/A'}")
            print(f"  Owner: {lease.owner or 'N/A'}")
            print(f"  Worktree: {lease.worktree_path or 'N/A'}")
            print(f"  Expires: {lease.expires_at}")
            print(
                "  Lease file: "
                f"{lease_path(Path(lease.worktree_path)) if lease.worktree_path else lease_path()}"
            )

        elif args.command == "heartbeat":
            lease = heartbeat(extend_hours=args.extend_hours, worktree_path=args.worktree)
            print(f"Refreshed heartbeat for PR #{lease.pr_number or 'N/A'}")
            print(f"  Gate/task ID: {lease.gate_id or 'N/A'}")
            print(f"  Expires: {lease.expires_at}")
            print(f"  Last heartbeat: {lease.last_heartbeat}")

        elif args.command == "release":
            if release_lease(worktree_path=args.worktree):
                print("Lease released")
            else:
                print("No active lease to release")

        elif args.command == "status":
            stat = status(worktree_path=args.worktree)
            if stat["active"]:
                lease_data = stat["lease"]
                print(f"Active lease for PR #{lease_data.get('pr_number') or 'N/A'}")
                print(f"  Gate/task ID: {lease_data.get('gate_id') or 'N/A'}")
                print(f"  Owner: {lease_data.get('owner') or 'N/A'}")
                print(f"  Worktree: {lease_data.get('worktree_path') or 'N/A'}")
                print(f"  Created: {lease_data['created_at']}")
                print(f"  Expires: {lease_data['expires_at']}")
                print(f"  Last heartbeat: {lease_data['last_heartbeat']}")
                print(f"  Seconds until expiry: {stat['seconds_until_expiry']:.1f}")
            else:
                print(f"No active lease ({stat['reason']})")
                if "lease" in stat:
                    print(f"  Expired at: {stat['expired_at']}")

        elif args.command == "is-active":
            return 0 if is_active(worktree_path=args.worktree) else 1

        return 0

    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
