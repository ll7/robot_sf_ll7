"""Tests for the bounded-subprocess teardown helper (issue #4216).

These are CPU-only and reproduce the mechanism of the LiCCA post-guard hang: a
child that leaves a long-lived descendant. The key assertion is that on timeout
the *whole process group* is reaped, so a descendant that would otherwise hold
device handles (and block the parent's exit after 100%) is terminated too.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from typing import TYPE_CHECKING

import pytest

from tests.support.process_teardown import (
    NestedProcessTimeout,
    reap_matching_descendants,
    run_bounded_subprocess,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="process-group reaping semantics are POSIX-specific",
)


def _pid_alive(pid: int) -> bool:
    """Return whether *pid* still exists (signal 0 probes without delivering)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as stat_file:
            state = stat_file.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        state = ""
    if state == "Z":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - exists but not ours
        return True
    return True


def test_run_bounded_subprocess_returns_for_fast_command() -> None:
    """A fast child completes normally and its return code is reported."""
    result = run_bounded_subprocess(
        [sys.executable, "-c", "raise SystemExit(3)"],
        timeout_seconds=30,
    )
    assert result.returncode == 3


def test_run_bounded_subprocess_reaps_descendant_on_timeout(tmp_path: Path) -> None:
    """On timeout the descendant (grandchild) is reaped with the process group.

    This is the direct regression guard for issue #4216: an unbounded parent that
    leaves a live descendant is exactly the teardown hang. Here the parent spawns
    a long sleeping grandchild, records its PID, then blocks; after the bounded
    timeout fires, the whole group must be gone.
    """
    pid_file = tmp_path / "descendant.pid"
    child_source = textwrap.dedent(
        f"""
        import subprocess, sys, time
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        with open({str(pid_file)!r}, "w", encoding="utf-8") as handle:
            handle.write(str(proc.pid))
        time.sleep(120)
        """,
    )

    with pytest.raises(NestedProcessTimeout) as excinfo:
        run_bounded_subprocess(
            [sys.executable, "-c", child_source],
            timeout_seconds=3,
        )

    assert excinfo.value.timeout_seconds == 3
    assert "terminated" in excinfo.value.cleanup_status or "killed" in excinfo.value.cleanup_status

    # The grandchild PID was written before the parent blocked.
    assert pid_file.exists(), "child did not start its descendant"
    descendant_pid = int(pid_file.read_text(encoding="utf-8").strip())

    # The group SIGTERM/SIGKILL should have reaped the descendant. Poll briefly to
    # allow the OS to finish tearing the process group down.
    deadline = time.monotonic() + 10.0
    while _pid_alive(descendant_pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _pid_alive(descendant_pid):
        # Best-effort cleanup so a failure here does not leak a process.
        try:
            os.kill(descendant_pid, 9)
        except ProcessLookupError:
            pass
        pytest.fail(
            f"descendant pid {descendant_pid} survived process-group reaping "
            "(teardown hang would persist)",
        )


def test_reap_matching_descendants_terminates_leaked_child(tmp_path: Path) -> None:
    """Session cleanup terminates a direct leaked child and its descendant."""
    pid_file = tmp_path / "grandchild.pid"
    child_source = textwrap.dedent(
        f"""
        import subprocess
        import sys
        import time

        grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        with open({str(pid_file)!r}, "w", encoding="utf-8") as handle:
            handle.write(str(grandchild.pid))
        time.sleep(120)
        """
    )
    child = subprocess.Popen([sys.executable, "-c", child_source])
    grandchild_pid = 0
    try:
        # Poll until the pid file exists AND holds a fully-written integer:
        # pid_file.exists() can become true after open() but before the pid is
        # written, so read the content rather than trusting existence alone.
        deadline = time.monotonic() + 10.0
        pid_text = ""
        while time.monotonic() < deadline:
            if pid_file.exists():
                pid_text = pid_file.read_text(encoding="utf-8").strip()
                if pid_text.isdigit():
                    break
            time.sleep(0.1)
        assert pid_text.isdigit(), "child did not record a descendant pid"
        grandchild_pid = int(pid_text)

        reaped = reap_matching_descendants(
            parent_pid=os.getpid(),
            command_substrings=(sys.executable,),
            grace_seconds=0.2,
        )

        assert child.pid in reaped
        assert grandchild_pid in reaped
        deadline = time.monotonic() + 10.0
        while (_pid_alive(child.pid) or _pid_alive(grandchild_pid)) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _pid_alive(child.pid)
        assert not _pid_alive(grandchild_pid)
    finally:
        for pid in (child.pid, grandchild_pid):
            if not pid:
                continue
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass


def test_run_bounded_subprocess_rejects_non_positive_timeout() -> None:
    """A non-positive timeout is rejected before the child is launched."""
    with pytest.raises(ValueError, match="positive"):
        run_bounded_subprocess([sys.executable, "-c", "pass"], timeout_seconds=0)


def test_run_bounded_subprocess_rejects_empty_command() -> None:
    """An empty command is rejected with a clear error."""
    with pytest.raises(ValueError, match="must not be empty"):
        run_bounded_subprocess([], timeout_seconds=5)
