#!/usr/bin/env python3
"""Hold a shared repository lock while running a child command.

Portable fallback for platforms without the ``flock`` CLI (e.g. macOS).
Both this helper (via :func:`fcntl.flock`, which uses ``flock(2)``) and the
``flock`` CLI lock the same file identity
(``$git_common_dir/robot-sf-create-worktree.lock``), so holders using either
mechanism serialize against each other. With ``--non-blocking`` or ``--timeout``,
a contended lock exits 75 instead of waiting indefinitely, mirroring ``flock -n``
and ``flock -w`` callers. The lock descriptor is inherited by the child, so helper
termination cannot release the lock while the child or a descendant that retains
that descriptor still has the mutation process alive.

The lifetime guarantee is intentionally descriptor-based: the helper supervises
the direct child's process group for signal forwarding, while the kernel lock
also remains held through inherited descriptors after a descendant daemonizes.
Commands that deliberately close the inherited descriptor (or otherwise move
work outside that ownership boundary) require the explicit process/OS boundary
provided by ``review_worktree_guard.py run`` and are not covered by this helper.
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
    return (
        "usage: worktree_creation_lock.py "
        "[--non-blocking] [--timeout SECONDS] [--wait-timeout SECONDS] "
        "LOCK_PATH -- COMMAND [ARG ...]"
    )


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


def _parse_timeout_option(args: list[str]) -> tuple[float, int] | int:
    """Parse a timeout CLI option, returning (timeout, consumed_count) or exit code 2."""
    arg = args[0]
    if arg.startswith(("--timeout=", "--wait-timeout=")):
        raw = arg.split("=", 1)[1]
        consumed = 1
    elif arg in ("--timeout", "--wait-timeout"):
        if len(args) < 2:
            print(_usage(), file=sys.stderr)
            return 2
        raw = args[1]
        consumed = 2
    else:
        return 0.0, 0
    try:
        val = float(raw)
        if val < 0:
            raise ValueError
        return val, consumed
    except ValueError:
        print(f"worktree_creation_lock: invalid timeout: {raw}", file=sys.stderr)
        return 2


def _parse_run_args(
    argv: list[str],
) -> tuple[bool, float | None, str, list[str]] | int:
    """Parse CLI arguments for run(), returning configuration or an exit code."""
    args = list(argv)
    non_blocking = False
    timeout: float | None = None
    while args:
        if args[0] == "--non-blocking":
            non_blocking = True
            args = args[1:]
        elif args[0].startswith(("--timeout", "--wait-timeout")):
            parsed_timeout = _parse_timeout_option(args)
            if isinstance(parsed_timeout, int):
                return parsed_timeout
            timeout, consumed = parsed_timeout
            args = args[consumed:]
        else:
            break
    try:
        separator = args.index("--")
    except ValueError:
        print(_usage(), file=sys.stderr)
        return 2
    lock_path, command = args[:separator], args[separator + 1 :]
    if len(lock_path) != 1 or not command:
        print(_usage(), file=sys.stderr)
        return 2
    return non_blocking, timeout, lock_path[0], command


def _acquire_flock_immediate(fd: int) -> bool:
    """Attempt an immediate exclusive flock; return True if held, False if contended."""
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return False
        raise


def _acquire_flock_with_timeout(fd: int, timeout: float) -> bool:
    """Poll for exclusive flock until acquired or timeout expires."""
    deadline = time.monotonic() + timeout
    while True:
        if _acquire_flock_immediate(fd):
            return True
        now = time.monotonic()
        if now >= deadline:
            return False
        time.sleep(min(0.05, max(0.001, deadline - now)))


def _record_lock_metadata(lock_file: TextIO) -> None:
    """Record owner diagnostics in the lock file."""
    try:
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\nstarted={int(time.time())}\nworktree={os.getcwd()}\n")
        lock_file.flush()
    except OSError:
        pass


def run(argv: list[str]) -> int:
    """Acquire an exclusive lock on LOCK_PATH, then run COMMAND."""
    parsed = _parse_run_args(argv)
    if isinstance(parsed, int):
        return parsed
    non_blocking, timeout, lock_path, command = parsed
    try:
        import fcntl
    except ImportError:
        print("worktree_creation_lock: fcntl is unavailable on this platform", file=sys.stderr)
        return 2
    try:
        with open(lock_path, "a+") as lock_file:
            fd = lock_file.fileno()
            if non_blocking or (timeout is not None and timeout <= 0):
                acquired = _acquire_flock_immediate(fd)
            elif timeout is not None:
                acquired = _acquire_flock_with_timeout(fd, timeout)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX)
                acquired = True

            if not acquired:
                return 75

            _record_lock_metadata(lock_file)
            # Close this descriptor through the context manager instead of
            # calling LOCK_UN explicitly. The child inherits the same open
            # file description; an explicit unlock would release that shared
            # lock even when a descendant still retains the inherited fd.
            return _run_locked_child(lock_file, command)
    except OSError as exc:
        print(f"worktree_creation_lock: failed to hold {lock_path}: {exc}", file=sys.stderr)
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
