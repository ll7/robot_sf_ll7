"""Portable ownership contract for process-isolated pytest temporary roots."""

from __future__ import annotations

import argparse
import errno
import getpass
import hashlib
import os
import shutil
import tempfile
from pathlib import Path


def is_pid_running(pid: int) -> bool:
    """Return whether *pid* still names a live or inaccessible process."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError as err:
        if err.errno == errno.ESRCH:
            return False
        return True


def pytest_worktree_root(
    project_root: Path,
    *,
    temp_base: Path | None = None,
    user: str | None = None,
) -> Path:
    """Return the canonical worktree-owned pytest root on this host."""
    resolved_root = project_root.resolve()
    worktree_hash = hashlib.sha256(str(resolved_root).encode("utf-8")).hexdigest()[:10]
    if user is None:
        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown"
    base = (temp_base or Path(tempfile.gettempdir())).resolve()
    return base / f"pytest-of-{user}" / f"wt-{worktree_hash}"


def pytest_process_root(project_root: Path, pid: int) -> Path:
    """Return the canonical pytest root owned by *pid* in *project_root*."""
    return pytest_worktree_root(project_root) / f"proc-{pid}"


def clean_stale_process_roots(worktree_root: Path, *, current_pid: int | None = None) -> None:
    """Remove only dead ``proc-<pid>`` children from a worktree root."""
    if not worktree_root.exists():
        return
    owner_pid = os.getpid() if current_pid is None else current_pid
    for process_root in worktree_root.glob("proc-*"):
        try:
            pid = int(process_root.name.removeprefix("proc-"))
        except ValueError:
            continue
        if pid != owner_pid and not is_pid_running(pid):
            shutil.rmtree(process_root, ignore_errors=True)


def prepare_process_root(project_root: Path, pid: int) -> Path:
    """Clean stale siblings, create, and return the exact root owned by *pid*."""
    process_root = pytest_process_root(project_root, pid)
    clean_stale_process_roots(process_root.parent, current_pid=pid)
    process_root.mkdir(parents=True, exist_ok=True)
    return process_root


def remove_process_root(project_root: Path, pid: int) -> None:
    """Remove the exact canonical root owned by *pid*, never a caller-provided path."""
    process_root = pytest_process_root(project_root, pid)
    shutil.rmtree(process_root, ignore_errors=True)


def main() -> int:
    """Prepare or clean one exact process root for shell wrappers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "cleanup"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    if args.pid <= 0:
        parser.error("--pid must be positive")
    if args.action == "prepare":
        print(prepare_process_root(args.project_root, args.pid))
    else:
        remove_process_root(args.project_root, args.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
