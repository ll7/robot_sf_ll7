#!/usr/bin/env python3
"""Hold a shared repository lock while running a child command.

Portable fallback for platforms without the ``flock`` CLI (e.g. macOS).
Both this helper (via :func:`fcntl.flock`, which uses ``flock(2)``) and the
``flock`` CLI lock the same file identity
(``$git_common_dir/robot-sf-create-worktree.lock``), so holders using either
mechanism serialize against each other. With ``--non-blocking``, a contended
lock exits 75 instead of waiting, mirroring ``flock -n`` callers. The lock
descriptor is inherited by the child, so helper termination cannot release the
lock while the child or a descendant that retains that descriptor still has
the mutation process alive.

The lifetime guarantee is intentionally descriptor-based: the helper supervises
the direct child's process group for signal forwarding, while the kernel lock
also remains held through inherited descriptors after a descendant daemonizes.
Commands that deliberately close the inherited descriptor (or otherwise move
work outside that ownership boundary) require the stronger process/OS sandbox
tracked by issue #8343 and are not covered by this helper.
"""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
import time
from typing import TextIO

LOCK_FD_ENV = "ROBOT_SF_WORKTREE_LOCK_FD"
CHILD_TERMINATION_GRACE_SECONDS = 5.0
PROCESS_POLL_INTERVAL_SECONDS = 0.01
FORWARDED_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)


def _usage() -> str:
    return "usage: worktree_creation_lock.py [--non-blocking] LOCK_PATH -- COMMAND [ARG ...]"


def _shell_returncode(returncode: int) -> int:
    """Map a ``Popen`` signal return code to the conventional shell status."""
    return 128 + (-returncode) if returncode < 0 else returncode


def verify_lock_fd(lock_path: str, fd: int) -> None:
    """Verify that *fd* identifies and holds the repository lock file."""
    try:
        import fcntl

        expected = os.stat(lock_path)
        actual = os.fstat(fd)
    except (ImportError, OSError) as exc:
        raise OSError(f"could not inspect inherited lock descriptor: {exc}") from exc
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise OSError("inherited lock descriptor does not identify the repository lock file")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise OSError("inherited lock descriptor does not hold the repository lock") from exc


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether a child process group still exists."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _send_child_signal(child: subprocess.Popen[bytes], signum: int) -> None:
    """Forward *signum* to the child process group, with a direct fallback."""
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            child.send_signal(signum)
        except OSError:
            pass


def _wait_for_child_group(child: subprocess.Popen[bytes], pending_signal: list[int | None]) -> None:
    """Reap the child group, escalating only after a forwarded signal stalls it."""
    termination_deadline: float | None = None
    kill_sent = False
    while True:
        child.poll()
        if not _process_group_exists(child.pid):
            child.wait()
            return

        if pending_signal[0] is not None and not kill_sent:
            if termination_deadline is None:
                termination_deadline = time.monotonic() + CHILD_TERMINATION_GRACE_SECONDS
            elif time.monotonic() >= termination_deadline:
                _send_child_signal(child, signal.SIGKILL)
                kill_sent = True
        time.sleep(PROCESS_POLL_INTERVAL_SECONDS)


def _run_locked_child(lock_file: TextIO, command: list[str]) -> int:
    """Run a child while preserving lock ownership through inherited descriptors."""
    child_ref: list[subprocess.Popen[bytes] | None] = [None]
    pending_signal: list[int | None] = [None]

    def forward_signal(signum: int, _frame: object) -> None:
        """Forward termination to the child before the lock can be released."""
        if pending_signal[0] is None:
            pending_signal[0] = signum
        child = child_ref[0]
        if child is not None and child.poll() is None:
            _send_child_signal(child, signum)

    previous_handlers = {
        signum: signal.signal(signum, forward_signal) for signum in FORWARDED_SIGNALS
    }
    try:
        lock_fd = lock_file.fileno()
        child_environment = os.environ.copy()
        child_environment[LOCK_FD_ENV] = str(lock_fd)
        child = subprocess.Popen(
            command,
            env=child_environment,
            pass_fds=(lock_fd,),
            start_new_session=True,
        )
        child_ref[0] = child
        if pending_signal[0] is not None:
            _send_child_signal(child, pending_signal[0])
        _wait_for_child_group(child, pending_signal)
        returncode = child.returncode
        if returncode is None:
            returncode = child.wait()
        if pending_signal[0] is not None:
            return 128 + pending_signal[0]
        return _shell_returncode(returncode)
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


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
            except OSError as exc:
                if non_blocking and exc.errno in (errno.EACCES, errno.EAGAIN):
                    return 75
                raise
            # Close this descriptor through the context manager instead of
            # calling LOCK_UN explicitly. The child inherits the same open
            # file description; an explicit unlock would release that shared
            # lock even when a descendant still retains the inherited fd.
            return _run_locked_child(lock_file, command)
    except OSError as exc:
        print(f"worktree_creation_lock: failed to hold {lock_path[0]}: {exc}", file=sys.stderr)
        return 2


def _verify_lock_fd_cli(argv: list[str]) -> int:
    """Run the internal inherited-lock descriptor check used by re-entry."""
    if len(argv) != 3 or argv[0] != "--verify-fd":
        print("usage: worktree_creation_lock.py --verify-fd LOCK_PATH FD", file=sys.stderr)
        return 2
    try:
        fd = int(argv[2])
        if fd < 0:
            raise ValueError
        verify_lock_fd(argv[1], fd)
    except (OSError, ValueError) as exc:
        print(f"worktree_creation_lock: {exc}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    """CLI entry point for the portable worktree-creation lock holder."""
    if sys.argv[1:2] == ["--verify-fd"]:
        return _verify_lock_fd_cli(sys.argv[1:])
    return run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
