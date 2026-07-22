"""Contract and regression tests for pytest temporary-tree isolation across concurrent sessions."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.conftest import _clean_stale_proc_dirs, _is_pid_running


def test_is_pid_running_returns_expected_status() -> None:
    """_is_pid_running should return True for current PID and False for non-existent PID."""
    assert _is_pid_running(os.getpid()) is True
    # PID 9999999 is out of standard Linux PID range
    assert _is_pid_running(9999999) is False


def test_pytest_temp_isolation_env_set(tmp_path: Path) -> None:
    """PYTEST_DEBUG_TEMPROOT should be configured by conftest.py with worktree and proc isolation."""
    temproot = os.environ.get("PYTEST_DEBUG_TEMPROOT")
    assert temproot is not None, "PYTEST_DEBUG_TEMPROOT must be set"
    assert "/wt-" in temproot, f"PYTEST_DEBUG_TEMPROOT '{temproot}' must contain worktree hash"
    assert "/proc-" in temproot, (
        f"PYTEST_DEBUG_TEMPROOT '{temproot}' must contain process isolation segment"
    )
    assert tmp_path.exists()


def test_stale_process_temproot_cleaned_up(tmp_path: Path) -> None:
    """Stale process temproot directories with dead PIDs should be quietly cleaned up."""
    temproot_env = os.environ.get("PYTEST_DEBUG_TEMPROOT")
    assert temproot_env is not None
    wt_dir = Path(temproot_env).parent

    stale_proc_dir = wt_dir / "proc-9999999"
    stale_proc_dir.mkdir(parents=True, exist_ok=True)
    (stale_proc_dir / "stale_file.txt").write_text("abandoned", encoding="utf-8")

    # Cleaning stale proc dirs under wt_dir should prune stale_proc_dir
    _clean_stale_proc_dirs(wt_dir)

    assert not stale_proc_dir.exists(), "Stale process directory should have been cleaned up"


def test_concurrent_pytest_sessions_no_cleanup_warnings(tmp_path: Path) -> None:
    """Concurrent pytest sessions must not interfere with each other or produce (rm_rf) warnings."""
    helper_code = """
import time
def test_active_writer(tmp_path):
    out_dir = tmp_path / "checkout"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(200):
        (out_dir / f"file_{i}.txt").write_text("data", encoding="utf-8")
        time.sleep(0.005)
"""
    helper_test_file = tmp_path / "test_active_writer.py"
    helper_test_file.write_text(helper_code, encoding="utf-8")

    quiet_code = """
def test_quiet_pass(tmp_path):
    assert tmp_path.exists()
"""
    quiet_test_file = tmp_path / "test_quiet_pass.py"
    quiet_test_file.write_text(quiet_code, encoding="utf-8")

    env1 = dict(os.environ)
    env1.pop("PYTEST_DEBUG_TEMPROOT", None)

    env2 = dict(os.environ)
    env2.pop("PYTEST_DEBUG_TEMPROOT", None)

    proc1 = subprocess.Popen(
        [sys.executable, "-m", "pytest", "-q", str(helper_test_file)],
        env=env1,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Let session 1 start writing
    for _ in range(50):
        if (tmp_path / "checkout").exists():
            break
        import time

        time.sleep(0.02)

    proc2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-W",
            "error::pytest.PytestWarning",
            str(quiet_test_file),
        ],
        env=env2,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    out1, err1 = proc1.communicate(timeout=30)

    assert proc1.returncode == 0, f"Session 1 failed: {err1}\n{out1}"
    assert proc2.returncode == 0, (
        f"Session 2 failed with warning/error: {proc2.stderr}\n{proc2.stdout}"
    )
    assert "(rm_rf)" not in proc2.stderr
    assert "(rm_rf)" not in err1


def test_pytest_temp_isolation_under_xdist(tmp_path: Path) -> None:
    """Pytest temporary directories under xdist should be properly isolated without warnings."""
    xdist_code = """
def test_xdist_1(tmp_path):
    (tmp_path / "file1.txt").write_text("ok", encoding="utf-8")

def test_xdist_2(tmp_path):
    (tmp_path / "file2.txt").write_text("ok", encoding="utf-8")
"""
    xdist_test_file = tmp_path / "test_xdist_isolation.py"
    xdist_test_file.write_text(xdist_code, encoding="utf-8")

    env = dict(os.environ)
    env.pop("PYTEST_DEBUG_TEMPROOT", None)

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "2",
            "-q",
            "-W",
            "error::pytest.PytestWarning",
            str(xdist_test_file),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert res.returncode == 0, f"xdist run failed: {res.stderr}\n{res.stdout}"
    assert "(rm_rf)" not in res.stderr
