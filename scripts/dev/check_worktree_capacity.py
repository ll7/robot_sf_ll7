#!/usr/bin/env python3
"""Fail-closed capacity checks and a read-only local reclaim inventory.

The worktree creator calls this module before asking Git to materialize a
checkout.  Keeping the check independent of Git means a low-space refusal
cannot leave a half-created worktree behind.  ``--inventory`` is deliberately
descriptive only: it never deletes, moves, prunes, or mutates a path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_MINIMUM_FREE_BYTES = 2 * 1024**3
RECLAIM_CATEGORIES = {
    "output": "ignored generated output; preserve durable evidence before pruning",
    "uv-cache": "dependency cache; review active uv processes before clearing",
    "worktrees": "Git worktrees; remove only clean, pushed worktrees with git worktree remove",
    "shared-memory": "agent scratch; remove only task-owned, no-longer-running worktrees",
}
_DISK_USAGE = Callable[[str | os.PathLike[str]], shutil._ntuple_diskusage]


@dataclass(frozen=True)
class CapacityResult:
    """The evidence used by a worktree-creation decision."""

    requested_path: str
    filesystem_path: str
    available_bytes: int | None
    minimum_free_bytes: int
    writable: bool
    status: str
    reason: str | None = None

    @property
    def allowed(self) -> bool:
        """Whether the target filesystem satisfies the creation gate."""

        return self.status == "pass"


@dataclass(frozen=True)
class ReclaimEntry:
    """A candidate for manual review, never an automatic deletion target."""

    path: str
    category: str
    exists: bool
    bytes: int | None
    status: str
    guidance: str


def _positive_integer(value: str, *, option: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{option} must be a non-negative integer (got {value!r})") from exc
    if parsed < 0:
        raise ValueError(f"{option} must be a non-negative integer (got {value!r})")
    return parsed


def _minimum_from_environment() -> int:
    raw = os.environ.get("ROBOT_SF_WORKTREE_MIN_FREE_BYTES")
    if raw is None:
        return DEFAULT_MINIMUM_FREE_BYTES
    return _positive_integer(raw, option="ROBOT_SF_WORKTREE_MIN_FREE_BYTES")


def _existing_anchor(path: Path) -> Path:
    """Return an existing directory on the filesystem containing ``path``."""

    candidate = path.expanduser()
    if candidate.exists():
        return candidate if candidate.is_dir() else candidate.parent
    parent = candidate.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent


def inspect_capacity(
    requested_path: Path,
    minimum_free_bytes: int,
    *,
    disk_usage: _DISK_USAGE = shutil.disk_usage,
) -> CapacityResult:
    """Inspect the target filesystem without creating the requested path."""

    requested = requested_path.expanduser()
    anchor = _existing_anchor(requested)
    if not anchor.exists() or not anchor.is_dir():
        return CapacityResult(
            requested_path=str(requested),
            filesystem_path=str(anchor),
            available_bytes=None,
            minimum_free_bytes=minimum_free_bytes,
            writable=False,
            status="error",
            reason="no existing parent directory is available for the target path",
        )

    writable = os.access(anchor, os.W_OK)
    try:
        available_bytes = int(disk_usage(anchor).free)
    except OSError as exc:
        return CapacityResult(
            requested_path=str(requested),
            filesystem_path=str(anchor),
            available_bytes=None,
            minimum_free_bytes=minimum_free_bytes,
            writable=writable,
            status="error",
            reason=f"could not inspect filesystem capacity: {exc}",
        )

    if not writable:
        status = "blocked"
        reason = "target parent is not writable"
    elif available_bytes < minimum_free_bytes:
        status = "blocked"
        reason = "available space is below the worktree safety threshold"
    else:
        status = "pass"
        reason = None

    return CapacityResult(
        requested_path=str(requested),
        filesystem_path=str(anchor),
        available_bytes=available_bytes,
        minimum_free_bytes=minimum_free_bytes,
        writable=writable,
        status=status,
        reason=reason,
    )


def _directory_size_bytes(path: Path) -> int | None:
    """Return a portable, read-only size estimate for one candidate directory."""

    try:
        result = subprocess.run(
            ["du", "-sk", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    fields = result.stdout.split()
    if not fields or not fields[0].isdigit():
        return None
    return int(fields[0]) * 1024


def _shm_candidates(shm_root: Path) -> Iterable[Path]:
    if not shm_root.is_dir():
        return ()
    prefixes = ("issue-", "review-", "codex-", "claude-", "robot-")
    return tuple(
        child
        for child in sorted(shm_root.iterdir())
        if child.is_dir() and child.name.startswith(prefixes)
    )


def reclaim_candidates(
    repo_root: Path, *, shm_root: Path = Path("/dev/shm")
) -> list[tuple[Path, str]]:
    """Return known local reclaim surfaces with conservative classifications."""

    root = repo_root.expanduser().resolve()
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    candidates: list[tuple[Path, str]] = [
        (root / "output", "output"),
        (xdg_cache / "uv", "uv-cache"),
        (root / ".worktrees", "worktrees"),
        (root.parent / f"{root.name}.worktrees", "worktrees"),
    ]
    candidates.extend((path, "shared-memory") for path in _shm_candidates(shm_root))

    seen: set[Path] = set()
    unique: list[tuple[Path, str]] = []
    for path, category in candidates:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((resolved, category))
    return unique


def build_reclaim_inventory(
    repo_root: Path,
    *,
    shm_root: Path = Path("/dev/shm"),
    size_fn: Callable[[Path], int | None] = _directory_size_bytes,
) -> list[ReclaimEntry]:
    """Build a dry-run inventory; this function has no filesystem mutations."""

    inventory: list[ReclaimEntry] = []
    for path, category in reclaim_candidates(repo_root, shm_root=shm_root):
        exists = path.exists()
        inventory.append(
            ReclaimEntry(
                path=str(path),
                category=category,
                exists=exists,
                bytes=size_fn(path) if exists else None,
                status="review" if exists else "absent",
                guidance=RECLAIM_CATEGORIES[category],
            )
        )
    return inventory


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return "unknown"


def _render_text(capacity: CapacityResult, inventory: list[ReclaimEntry]) -> str:
    lines = [
        "Worktree capacity preflight (read-only)",
        f"- target: {capacity.requested_path}",
        f"- filesystem anchor: {capacity.filesystem_path}",
        f"- available: {_human_bytes(capacity.available_bytes)}",
        f"- required: {_human_bytes(capacity.minimum_free_bytes)}",
        f"- verdict: {capacity.status.upper()}",
    ]
    if capacity.reason:
        lines.append(f"- reason: {capacity.reason}")
    if inventory:
        lines.extend(("", "Manual reclaim inventory (dry-run; nothing is deleted):"))
        for entry in inventory:
            size = _human_bytes(entry.bytes) if entry.exists else "absent"
            lines.append(f"- [{entry.status}] {entry.path} ({size}; {entry.category})")
            if entry.exists:
                lines.append(f"  guidance: {entry.guidance}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail closed before worktree creation and inventory safe reclaim candidates.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="target worktree path (the existing parent filesystem is inspected)",
    )
    parser.add_argument(
        "--minimum-free-bytes",
        type=str,
        default=None,
        help=("minimum free bytes required; defaults to ROBOT_SF_WORKTREE_MIN_FREE_BYTES or 2 GiB"),
    )
    parser.add_argument("--inventory", action="store_true", help="show manual reclaim candidates")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--shm-root", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--json", action="store_true", help="emit machine-readable evidence")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the capacity gate and optional read-only reclaim inventory."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        minimum = (
            _positive_integer(args.minimum_free_bytes, option="--minimum-free-bytes")
            if args.minimum_free_bytes is not None
            else _minimum_from_environment()
        )
    except ValueError as exc:
        parser.error(str(exc))

    capacity = inspect_capacity(args.path, minimum)
    inventory = (
        build_reclaim_inventory(args.repo_root, shm_root=args.shm_root) if args.inventory else []
    )
    if args.json:
        payload = {
            "capacity": asdict(capacity),
            "inventory": [asdict(entry) for entry in inventory],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_text(capacity, inventory))
    return 0 if capacity.allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
