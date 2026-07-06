"""Bounded subprocess execution for tests that spawn heavy child processes.

Some tests spawn a *nested* pytest (or another child that imports the full
``robot_sf`` + torch/CUDA stack). On shared GPU/HPC nodes such a child can
deadlock during interpreter/CUDA teardown, or exit while leaving a descendant
that still holds GPU device handles. An unbounded ``subprocess.run(...)`` then
blocks the parent suite *forever* -- even after the visible progress already
reached ``[100%]``. That is the LiCCA post-guard full-suite teardown hang
recorded for issue #4216 (PR #4276): the suite reaches 100%, but a nested pytest
child survives holding device handles and pytest never emits its final summary
or exits.

Two facts make the hang silent on this repo:

* ``pytest-timeout`` is **not** installed, so ``@pytest.mark.timeout(...)`` is a
  no-op decoration -- the intended per-test bound is never enforced.
* the spawning test used ``subprocess.run(...)`` without a ``timeout=`` argument,
  so nothing bounds the nested child.

This helper runs the child in its own session/process group and, on timeout,
terminates the *whole* group (SIGTERM -> SIGKILL) so descendants holding device
handles are reaped too, then raises :class:`NestedProcessTimeout` instead of
blocking. It deliberately mirrors the proven termination pattern in
``scripts/dev/run_compact_validation.py`` (``_terminate_process_group``) but stays
lightweight for the test lane (no artifact writes). If those two implementations
grow further, fold them into one shared primitive.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

# Grace period between the group SIGTERM and the escalated SIGKILL. Kept small so
# a genuinely wedged child cannot extend the teardown budget by much.
DEFAULT_GROUP_KILL_GRACE_SECONDS = 5.0

# POSIX exposes process groups / ``os.killpg``; Windows does not. The repo runs
# its suites on Linux (GitHub ubuntu runners, LiCCA/shared HPC), so the
# process-group path is the norm; the fallback only keeps imports safe elsewhere.
_POSIX = os.name == "posix"


class NestedProcessTimeout(RuntimeError):
    """Raised when a bounded child process exceeds its timeout and is reaped.

    The exception carries the reaped command, the timeout budget, and the
    ``cleanup_status`` describing how the process group was terminated so callers
    can assert on the teardown path instead of hanging.
    """

    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float,
        cleanup_status: str,
    ) -> None:
        """Record the reaped command and how its process group was cleaned up."""
        self.command = list(command)
        self.timeout_seconds = timeout_seconds
        self.cleanup_status = cleanup_status
        super().__init__(
            f"nested process exceeded {timeout_seconds:g}s and was reaped "
            f"({cleanup_status}): {self.command}",
        )


def terminate_process_group(
    process: subprocess.Popen,
    *,
    grace_seconds: float = DEFAULT_GROUP_KILL_GRACE_SECONDS,
) -> str:
    """Terminate a timed-out process and any descendants in its process group.

    Sends ``SIGTERM`` to the whole group, waits ``grace_seconds`` for a clean
    exit, then escalates to ``SIGKILL``. Returns a short status string naming
    which path was taken (mirrors ``run_compact_validation._terminate_process_group``).

    Args:
        process: A ``Popen`` started with ``start_new_session=True`` so its ``pid``
            is also its process-group id.
        grace_seconds: Seconds to wait after ``SIGTERM`` before escalating.
    """
    cleanup_status = "process_group_terminated_and_waited"
    if not _POSIX:
        # No process groups: best-effort terminate the direct child only.
        process.terminate()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return "direct_process_killed_and_waited"
        return "direct_process_terminated_and_waited"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return cleanup_status
    except OSError:
        process.terminate()
        cleanup_status = "direct_process_terminated_and_waited"
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        cleanup_status = cleanup_status.replace("terminated", "killed")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
            cleanup_status = "direct_process_killed_and_waited"
    process.wait()
    return cleanup_status


def _direct_child_pids(parent_pid: int) -> list[int]:
    """Return Linux direct child PIDs for *parent_pid* using procfs."""
    children_path = f"/proc/{parent_pid}/task/{parent_pid}/children"
    try:
        raw_children = open(children_path, encoding="ascii").read().strip()
    except OSError:
        return []
    if not raw_children:
        return []
    return [int(pid_text) for pid_text in raw_children.split()]


def _descendant_pids(parent_pid: int) -> list[int]:
    """Return descendants of *parent_pid*, parents before children."""
    descendants: list[int] = []
    pending = _direct_child_pids(parent_pid)
    while pending:
        pid = pending.pop(0)
        descendants.append(pid)
        pending.extend(_direct_child_pids(pid))
    return descendants


def _cmdline(pid: int) -> str:
    """Return a process command line as a space-separated string."""
    try:
        raw_cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return ""
    return raw_cmdline.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def _pid_alive(pid: int) -> bool:
    """Return whether *pid* still exists."""
    try:
        state = open(f"/proc/{pid}/stat", encoding="ascii").read().split()[2]
    except OSError:
        state = ""
    if state == "Z":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - exists but not owned by this user
        return True
    return True


def _terminate_pids(pids: list[int], *, grace_seconds: float) -> None:
    """Terminate *pids* with SIGTERM, then SIGKILL remaining live processes."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + grace_seconds
    while any(_pid_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.05)

    for pid in pids:
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + max(grace_seconds, 0.5)
    while any(_pid_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.05)


def reap_matching_descendants(
    *,
    parent_pid: int | None = None,
    command_substrings: tuple[str, ...] = ("pytest",),
    grace_seconds: float = DEFAULT_GROUP_KILL_GRACE_SECONDS,
) -> list[int]:
    """Terminate leaked descendant processes matching command substrings.

    This is a fail-closed cleanup for issue #4216's full-suite teardown hang:
    pytest can reach 100% but not exit because a nested pytest child is still
    alive and holding GPU device handles. It intentionally targets only current
    process descendants whose command line matches an expected nested test
    runner, then also terminates their descendants.
    """
    if not _POSIX:
        return []
    if grace_seconds < 0:
        raise ValueError("grace_seconds must be non-negative")

    root_pid = os.getpid() if parent_pid is None else parent_pid
    matching_roots = [
        pid
        for pid in _descendant_pids(root_pid)
        if any(token in _cmdline(pid) for token in command_substrings)
    ]
    if not matching_roots:
        return []

    pids_to_reap: set[int] = set()
    for pid in matching_roots:
        pids_to_reap.add(pid)
        pids_to_reap.update(_descendant_pids(pid))

    reaped = sorted(pids_to_reap)
    _terminate_pids(reaped, grace_seconds=grace_seconds)
    return reaped


def run_bounded_subprocess(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    grace_seconds: float = DEFAULT_GROUP_KILL_GRACE_SECONDS,
) -> subprocess.CompletedProcess:
    """Run *command* with an enforced timeout and process-group reaping.

    Unlike a bare ``subprocess.run(..., timeout=...)`` (which reaps only the direct
    child), on timeout this terminates the entire process group so descendants
    that still hold GPU/device handles are reaped too, then raises
    :class:`NestedProcessTimeout`. This is what keeps a wedged nested pytest child
    from blocking the parent suite's exit after it has reached 100%.

    Output is left inheriting the caller's stdout/stderr (not captured), matching
    the behavior of the tests that spawn nested pytest children.

    Args:
        command: Argv for the child process; must be non-empty.
        timeout_seconds: Positive wall-clock bound for the child.
        cwd: Optional working directory for the child.
        env: Optional environment mapping for the child.
        grace_seconds: Seconds between the group ``SIGTERM`` and ``SIGKILL``.

    Returns:
        A ``CompletedProcess`` with the child's return code on normal completion.

    Raises:
        ValueError: If ``command`` is empty or ``timeout_seconds`` is not positive.
        NestedProcessTimeout: If the child exceeds ``timeout_seconds``.
    """
    command_list = list(command)
    if not command_list:
        raise ValueError("command must not be empty")
    if timeout_seconds is None or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive number")

    process = subprocess.Popen(
        command_list,
        cwd=None if cwd is None else str(cwd),
        env=None if env is None else dict(env),
        # New session => child pid is its process-group id, so descendants it
        # spawns can be reaped as a group on timeout.
        start_new_session=_POSIX,
    )
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        cleanup_status = terminate_process_group(process, grace_seconds=grace_seconds)
        raise NestedProcessTimeout(command_list, timeout_seconds, cleanup_status) from exc
    return subprocess.CompletedProcess(command_list, returncode)
