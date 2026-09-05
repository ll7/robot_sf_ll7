#!/usr/bin/env python3
"""Hold the shared worktree-creation lock while running a child command.

Portable fallback for platforms without the ``flock`` CLI (e.g. macOS).
Both this helper (via :func:`fcntl.flock`, which uses ``flock(2)``) and the
``flock`` CLI lock the same file identity
(``$git_common_dir/robot-sf-create-worktree.lock``), so holders using either
mechanism serialize against each other. The lock is held for exactly the
lifetime of the child process.
"""

from __future__ import annotations

import subprocess
import sys


def _usage() -> str:
    return "usage: worktree_creation_lock.py LOCK_PATH -- COMMAND [ARG ...]"


def run(argv: list[str]) -> int:
    """Acquire an exclusive lock on LOCK_PATH, then run COMMAND."""
    try:
        separator = argv.index("--")
    except ValueError:
        print(_usage(), file=sys.stderr)
        return 2
    lock_path, command = argv[:separator], argv[separator + 1 :]
    if len(lock_path) != 1 or not command:
        print(_usage(), file=sys.stderr)
        return 2
    try:
        import fcntl
    except ImportError:
        print("worktree_creation_lock: fcntl is unavailable on this platform", file=sys.stderr)
        return 2
    try:
        with open(lock_path[0], "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                completed = subprocess.run(command, check=False)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return completed.returncode
    except OSError as exc:
        print(f"worktree_creation_lock: failed to hold {lock_path[0]}: {exc}", file=sys.stderr)
        return 2


def main() -> int:
    """CLI entry point for the portable worktree-creation lock holder."""
    return run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
