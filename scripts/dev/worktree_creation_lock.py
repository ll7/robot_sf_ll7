#!/usr/bin/env python3
"""Hold a shared repository lock while running a child command.

Portable fallback for platforms without the ``flock`` CLI (e.g. macOS).
Both this helper (via :func:`fcntl.flock`, which uses ``flock(2)``) and the
``flock`` CLI lock the same file identity, so holders using either mechanism
serialize against each other. The lock is held for exactly the lifetime of
the child process. With ``--non-blocking``, a contended lock exits 75
instead of waiting, mirroring ``flock -n`` callers.
"""

from __future__ import annotations

import subprocess
import sys


def _usage() -> str:
    return "usage: worktree_creation_lock.py [--non-blocking] LOCK_PATH -- COMMAND [ARG ...]"


def run(argv: list[str]) -> int:
    """Acquire an exclusive lock on LOCK_PATH, then run COMMAND."""
    args = list(argv)
    non_blocking = False
    if args[:1] == ["--non-blocking"]:
        non_blocking = True
        args = args[1:]
    try:
        separator = args.index("--")
    except ValueError:
        print(_usage(), file=sys.stderr)
        return 2
    lock_path, command = args[:separator], args[separator + 1 :]
    if len(lock_path) != 1 or not command:
        print(_usage(), file=sys.stderr)
        return 2
    try:
        import fcntl
    except ImportError:
        print("worktree_creation_lock: fcntl is unavailable on this platform", file=sys.stderr)
        return 2
    flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if non_blocking else 0)
    try:
        with open(lock_path[0], "a+") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), flags)
            except BlockingIOError:
                return 75
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
