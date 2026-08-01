"""Tiny sanity check for the CI step timer helper."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_HAS_TIMEOUT = shutil.which("timeout") is not None


def _wait_for_nonempty_file(path: Path, timeout_seconds: float = 5) -> str:
    """Wait for a subprocess to publish a small readiness or audit file."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            contents = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            contents = ""
        if contents:
            return contents
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for subprocess file: {path}")


def _process_exists(pid: int) -> bool:
    """Return whether a process ID still exists, including as a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _kill_process_group(process_group_id: int) -> None:
    """Best-effort cleanup for a test-owned session on assertion failure."""
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _timeout_fallback_env(tmp_path: Path, *, include_python: bool = True) -> dict[str, str]:
    """Return a PATH with no GNU timeout and an optional Python 3 backend."""
    date_path = shutil.which("date")
    assert date_path, "date is required by ci_step_timer.sh"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    os.symlink(date_path, fake_bin / "date")
    if include_python:
        os.symlink(sys.executable, fake_bin / "python3")

    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    return env


def test_ci_step_timer_shell_syntax():
    """Validate that the CI step timer helper passes bash syntax checks."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    assert script.exists(), "ci_step_timer.sh helper is missing"
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_ci_step_timer_help_flag():
    """--help prints usage and exits 0 without running any command."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "ci_step_timer step_start" not in result.stdout
    assert "ci_step_timer step_end" not in result.stdout
    assert "::group::" not in result.stdout


def test_ci_step_timer_h_flag():
    """-h prints usage and exits 0 without running any command."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script), "-h"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "ci_step_timer step_start" not in result.stdout
    assert "ci_step_timer step_end" not in result.stdout
    assert "::group::" not in result.stdout


def test_ci_step_timer_requires_label_and_command():
    """Ensure the helper exits with usage info when arguments are missing."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_ci_step_timer_propagates_failure():
    """Verify that a failing command is reflected in the reported status."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script), "failing-check", "false"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "failing-check" in result.stdout
    assert "::endgroup::" in result.stdout


def test_ci_step_timer_reports_success_duration():
    """Check that a successful step is reported with zero exit status and duration."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script), "echo-test", "echo", "hello"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "echo-test" in result.stdout
    assert "hello" in result.stdout
    assert 'ci_step_timer step_end label="echo-test" status=0 duration_seconds=' in result.stdout


@pytest.mark.skipif(not _HAS_TIMEOUT, reason="GNU timeout(1) is required for timeout tests")
def test_ci_step_timer_timeout_does_not_affect_fast_command() -> None:
    """A small timeout should not change the outcome of a command that finishes quickly."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    env = os.environ.copy()
    env["CI_STEP_TIMEOUT_SECONDS"] = "5"
    result = subprocess.run(
        ["bash", str(script), "fast-timeout", "echo", "ok"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    assert "ok" in result.stdout
    assert 'ci_step_timer step_end label="fast-timeout" status=0 duration_seconds=' in result.stdout
    assert "::endgroup::" in result.stdout


@pytest.mark.skipif(not _HAS_TIMEOUT, reason="GNU timeout(1) is required for timeout tests")
def test_ci_step_timer_timeout_kills_long_command() -> None:
    """A short timeout must kill a long command and still report the step end."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    env = os.environ.copy()
    env["CI_STEP_TIMEOUT_SECONDS"] = "0.1"
    result = subprocess.run(
        ["bash", str(script), "slow-timeout", "sleep", "10"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 124
    assert "slow-timeout" in result.stdout
    assert "ci_step_timer step_end" in result.stdout
    assert "::notice" in result.stdout
    assert "::endgroup::" in result.stdout


def test_ci_step_timer_python_fallback_preserves_fast_command(tmp_path: Path) -> None:
    """Preserve fast-command output and status when the Python backend is selected."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    bash_path = shutil.which("bash")
    assert bash_path, "bash is required by ci_step_timer.sh"

    env = _timeout_fallback_env(tmp_path)
    env["CI_STEP_TIMEOUT_SECONDS"] = "5"
    result = subprocess.run(
        [
            bash_path,
            str(script),
            "python-fallback-fast",
            sys.executable,
            "-c",
            "print('fallback-ok')",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "fallback-ok" in result.stdout
    assert 'ci_step_timer step_end label="python-fallback-fast" status=0' in result.stdout
    assert "::endgroup::" in result.stdout


def test_ci_step_timer_python_fallback_kills_long_command(tmp_path: Path) -> None:
    """Use Python to bound and reap a long child when GNU timeout is unavailable."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    bash_path = shutil.which("bash")
    assert bash_path, "bash is required by ci_step_timer.sh"

    env = _timeout_fallback_env(tmp_path)
    env["CI_STEP_TIMEOUT_SECONDS"] = "0.2"
    started_at = time.monotonic()
    result = subprocess.run(
        [
            bash_path,
            str(script),
            "python-fallback-timeout",
            sys.executable,
            "-c",
            "import os, time; print(f'child_pid={os.getpid()}', flush=True); time.sleep(10)",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env=env,
    )
    elapsed = time.monotonic() - started_at

    child_pid = int(result.stdout.split("child_pid=", 1)[1].splitlines()[0])
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)

    assert result.returncode == 124
    assert elapsed < 5
    assert 'ci_step_timer step_start label="python-fallback-timeout"' in result.stdout
    assert 'ci_step_timer step_end label="python-fallback-timeout" status=124' in result.stdout
    assert "::endgroup::" in result.stdout


def test_ci_step_timer_python_fallback_preserves_signal_status(tmp_path: Path) -> None:
    """Map a child signal to the conventional shell status through the Python backend."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    bash_path = shutil.which("bash")
    assert bash_path, "bash is required by ci_step_timer.sh"

    env = _timeout_fallback_env(tmp_path)
    env["CI_STEP_TIMEOUT_SECONDS"] = "5"
    result = subprocess.run(
        [
            bash_path,
            str(script),
            "python-fallback-signal",
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 128 + 15
    assert 'ci_step_timer step_end label="python-fallback-signal" status=143' in result.stdout
    assert "::endgroup::" in result.stdout


def test_ci_step_timer_timeout_requires_supported_backend(tmp_path: Path) -> None:
    """Fail clearly instead of running unbounded when no timeout backend exists."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    bash_path = shutil.which("bash")
    true_path = shutil.which("true")
    assert bash_path and true_path, "required system binaries missing"

    env = _timeout_fallback_env(tmp_path, include_python=False)
    env["CI_STEP_TIMEOUT_SECONDS"] = "1"
    result = subprocess.run(
        [bash_path, str(script), "missing-timeout", true_path],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 127
    assert "no supported timeout backend" in result.stderr
    assert "ci_step_timer step_end" in result.stdout
    assert "::endgroup::" in result.stdout


def test_ci_step_timer_python_fallback_kills_stubborn_descendant_before_timeout_return(
    tmp_path: Path,
) -> None:
    """Do not report timeout status until a TERM-ignoring process-group descendant is gone."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    bash_path = shutil.which("bash")
    assert bash_path, "bash is required by ci_step_timer.sh"

    descendant_pid_path = tmp_path / "descendant.pid"
    term_audit_path = tmp_path / "descendant.term"
    stdout_path = tmp_path / "timer.stdout"
    stderr_path = tmp_path / "timer.stderr"
    descendant_code = textwrap.dedent(
        """
        import os
        import signal
        import sys
        import time
        from pathlib import Path

        term_audit_path = Path(sys.argv[2])

        def ignore_term(signum, _frame):
            term_audit_path.write_text(str(signum), encoding="utf-8")

        signal.signal(signal.SIGTERM, ignore_term)
        Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
        while True:
            time.sleep(1)
        """
    )
    leader_code = textwrap.dedent(
        """
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        Path(sys.argv[4]).write_text(str(os.getpid()), encoding="utf-8")
        descendant = subprocess.Popen(
            [sys.executable, "-c", sys.argv[1], sys.argv[2], sys.argv[3]],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 5
        while not Path(sys.argv[2]).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("descendant did not become ready")
            time.sleep(0.01)
        time.sleep(30)
        """
    )
    leader_pid_path = tmp_path / "leader.pid"
    env = _timeout_fallback_env(tmp_path)
    env["CI_STEP_TIMEOUT_SECONDS"] = "1"

    descendant_pid: int | None = None
    leader_pid: int | None = None
    started_at = time.monotonic()
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            result = subprocess.run(
                [
                    bash_path,
                    str(script),
                    "python-fallback-process-group-timeout",
                    sys.executable,
                    "-c",
                    leader_code,
                    descendant_code,
                    str(descendant_pid_path),
                    str(term_audit_path),
                    str(leader_pid_path),
                ],
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
                timeout=15,
                env=env,
            )
        elapsed = time.monotonic() - started_at
        stdout = stdout_path.read_text(encoding="utf-8")
        stderr = stderr_path.read_text(encoding="utf-8")
        descendant_pid = int(_wait_for_nonempty_file(descendant_pid_path))
        leader_pid = int(_wait_for_nonempty_file(leader_pid_path))

        assert result.returncode == 124
        assert elapsed < 12
        assert _wait_for_nonempty_file(term_audit_path) == str(signal.SIGTERM)
        assert not _process_exists(leader_pid)
        assert not _process_exists(descendant_pid)
        assert "process group ignored the termination grace; sending SIGKILL" in stderr
        assert (
            'ci_step_timer step_end label="python-fallback-process-group-timeout" status=124'
            in stdout
        )
        assert "::endgroup::" in stdout
    finally:
        if leader_pid is None and leader_pid_path.exists():
            leader_pid = int(leader_pid_path.read_text(encoding="utf-8"))
        if descendant_pid is None and descendant_pid_path.exists():
            descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
        if leader_pid is not None:
            _kill_process_group(leader_pid)
        elif descendant_pid is not None and _process_exists(descendant_pid):
            try:
                _kill_process_group(os.getpgid(descendant_pid))
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    ("signum", "expected_status"),
    [
        (signal.SIGHUP, 128 + signal.SIGHUP),
        (signal.SIGINT, 128 + signal.SIGINT),
        (signal.SIGTERM, 128 + signal.SIGTERM),
    ],
)
def test_ci_step_timer_python_fallback_forwards_wrapper_signal_and_reaps_tree(
    tmp_path: Path,
    signum: signal.Signals,
    expected_status: int,
) -> None:
    """Forward top-level wrapper signals and retain markers after reaping the command tree."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    bash_path = shutil.which("bash")
    assert bash_path, "bash is required by ci_step_timer.sh"

    command_info_path = tmp_path / "command.info"
    command_signal_path = tmp_path / "command.signal"
    descendant_pid_path = tmp_path / "descendant.pid"
    descendant_signal_path = tmp_path / "descendant.signal"
    stdout_path = tmp_path / "wrapper.stdout"
    stderr_path = tmp_path / "wrapper.stderr"
    descendant_code = textwrap.dedent(
        """
        import os
        import signal
        import sys
        import time
        from pathlib import Path

        signal_path = Path(sys.argv[2])

        def exit_on_signal(signum, _frame):
            signal_path.write_text(str(signum), encoding="utf-8")
            raise SystemExit(128 + signum)

        for handled_signal in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(handled_signal, exit_on_signal)
        Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
        while True:
            time.sleep(1)
        """
    )
    command_code = textwrap.dedent(
        """
        import os
        import signal
        import subprocess
        import sys
        import time
        from pathlib import Path

        signal_path = Path(sys.argv[2])

        def exit_on_signal(signum, _frame):
            signal_path.write_text(str(signum), encoding="utf-8")
            raise SystemExit(128 + signum)

        for handled_signal in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(handled_signal, exit_on_signal)
        subprocess.Popen(
            [sys.executable, "-c", sys.argv[1], sys.argv[3], sys.argv[4]],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 5
        while not Path(sys.argv[3]).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("descendant did not become ready")
            time.sleep(0.01)
        Path(sys.argv[5]).write_text(
            f"{os.getpid()} {os.getppid()}",
            encoding="utf-8",
        )
        while True:
            time.sleep(1)
        """
    )
    env = _timeout_fallback_env(tmp_path)
    env["CI_STEP_TIMEOUT_SECONDS"] = "30"

    wrapper: subprocess.Popen[str] | None = None
    command_pid: int | None = None
    backend_pid: int | None = None
    descendant_pid: int | None = None
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            wrapper = subprocess.Popen(
                [
                    bash_path,
                    str(script),
                    "python-fallback-wrapper-signal",
                    sys.executable,
                    "-c",
                    command_code,
                    descendant_code,
                    str(command_signal_path),
                    str(descendant_pid_path),
                    str(descendant_signal_path),
                    str(command_info_path),
                ],
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=env,
                start_new_session=True,
            )
            command_pid, backend_pid = (
                int(value) for value in _wait_for_nonempty_file(command_info_path).split()
            )
            descendant_pid = int(_wait_for_nonempty_file(descendant_pid_path))
            os.kill(wrapper.pid, signum)
            returncode = wrapper.wait(timeout=12)

        stdout = stdout_path.read_text(encoding="utf-8")
        assert returncode == expected_status
        assert _wait_for_nonempty_file(command_signal_path) == str(signum)
        assert _wait_for_nonempty_file(descendant_signal_path) == str(signum)
        assert not _process_exists(command_pid)
        assert not _process_exists(descendant_pid)
        assert not _process_exists(backend_pid)
        assert (
            'ci_step_timer step_end label="python-fallback-wrapper-signal" '
            f"status={expected_status}" in stdout
        )
        assert "::endgroup::" in stdout
    finally:
        if command_pid is None and command_info_path.exists():
            command_pid, backend_pid = (
                int(value) for value in command_info_path.read_text(encoding="utf-8").split()
            )
        if descendant_pid is None and descendant_pid_path.exists():
            descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
        if wrapper is not None and wrapper.poll() is None:
            _kill_process_group(wrapper.pid)
            wrapper.wait(timeout=5)
        if command_pid is not None:
            _kill_process_group(command_pid)
        elif descendant_pid is not None and _process_exists(descendant_pid):
            try:
                _kill_process_group(os.getpgid(descendant_pid))
            except ProcessLookupError:
                pass
        if backend_pid is not None and _process_exists(backend_pid):
            try:
                os.kill(backend_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_ci_step_timer_python_fallback_replays_early_wrapper_sigint_after_backend_ready(
    tmp_path: Path,
) -> None:
    """Do not lose SIGINT while an asynchronous Bash 3.2 child still ignores it."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    bash_path = shutil.which("bash")
    sleep_path = shutil.which("sleep")
    assert bash_path and sleep_path, "bash and sleep are required by ci_step_timer.sh"

    backend_pid_path = tmp_path / "backend.pid"
    command_pid_path = tmp_path / "command.pid"
    env = _timeout_fallback_env(tmp_path)
    python_shim = Path(env["PATH"]) / "python3"
    python_shim.unlink()
    python_shim.write_text(
        f"""#!{bash_path}
printf '%s' "$$" > "$CI_TIMER_BACKEND_PID_PATH"
"$CI_TIMER_SLEEP_PATH" 0.5
exec "$CI_TIMER_REAL_PYTHON" "$@"
""",
        encoding="utf-8",
    )
    python_shim.chmod(0o755)
    env.update(
        {
            "CI_STEP_TIMEOUT_SECONDS": "3",
            "CI_TIMER_BACKEND_PID_PATH": str(backend_pid_path),
            "CI_TIMER_REAL_PYTHON": sys.executable,
            "CI_TIMER_SLEEP_PATH": sleep_path,
        }
    )
    command_code = (
        "import os, sys, time; "
        "from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8'); "
        "time.sleep(30)"
    )

    wrapper: subprocess.Popen[str] | None = None
    backend_pid: int | None = None
    command_pid: int | None = None
    started_at = time.monotonic()
    try:
        wrapper = subprocess.Popen(
            [
                bash_path,
                str(script),
                "python-fallback-early-sigint",
                sys.executable,
                "-c",
                command_code,
                str(command_pid_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        backend_pid = int(_wait_for_nonempty_file(backend_pid_path))
        os.kill(wrapper.pid, signal.SIGINT)
        stdout, _stderr = wrapper.communicate(timeout=8)
        elapsed = time.monotonic() - started_at

        assert wrapper.returncode == 128 + signal.SIGINT
        assert elapsed < 2.5
        assert not _process_exists(backend_pid)
        assert not command_pid_path.exists()
        assert 'ci_step_timer step_end label="python-fallback-early-sigint" status=130' in stdout
        assert "::endgroup::" in stdout
    finally:
        if command_pid_path.exists():
            command_pid = int(command_pid_path.read_text(encoding="utf-8"))
        if wrapper is not None and wrapper.poll() is None:
            _kill_process_group(wrapper.pid)
            wrapper.wait(timeout=5)
        if command_pid is not None:
            _kill_process_group(command_pid)
        if backend_pid is not None and _process_exists(backend_pid):
            try:
                os.kill(backend_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_ci_step_timer_python_fallback_reaps_backend_after_readiness_interrupts_wait(
    tmp_path: Path,
) -> None:
    """Retry Bash wait when the readiness trap races with immediate backend exit."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    bash_path = shutil.which("bash")
    sleep_path = shutil.which("sleep")
    true_path = shutil.which("true")
    assert bash_path and sleep_path and true_path, "bash, sleep, and true are required"

    env = _timeout_fallback_env(tmp_path)
    python_shim = Path(env["PATH"]) / "python3"
    python_shim.unlink()
    python_shim.write_text(
        f"""#!{bash_path}
"$CI_TIMER_SLEEP_PATH" 0.05
kill -USR1 "$PPID"
exit 0
""",
        encoding="utf-8",
    )
    python_shim.chmod(0o755)
    env["CI_STEP_TIMEOUT_SECONDS"] = "5"
    env["CI_TIMER_SLEEP_PATH"] = sleep_path

    result = subprocess.run(
        [bash_path, str(script), "python-fallback-ready-exit-race", true_path],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
        env=env,
    )

    assert result.returncode == 0
    assert 'ci_step_timer step_end label="python-fallback-ready-exit-race" status=0' in (
        result.stdout
    )
    assert "::endgroup::" in result.stdout


def test_ci_step_timer_python_fallback_updates_signal_status_during_footer(
    tmp_path: Path,
) -> None:
    """Retain conventional signal status when TERM arrives after backend reap."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    bash_path = shutil.which("bash")
    date_path = shutil.which("date")
    sleep_path = shutil.which("sleep")
    assert bash_path and date_path and sleep_path, "bash, date, and sleep are required"

    date_state_path = tmp_path / "date.count"
    footer_ready_path = tmp_path / "footer.ready"
    footer_release_path = tmp_path / "footer.release"
    env = _timeout_fallback_env(tmp_path)
    date_shim = Path(env["PATH"]) / "date"
    date_shim.unlink()
    date_shim.write_text(
        f"""#!{bash_path}
count=0
if [[ -f "$CI_TIMER_DATE_STATE_PATH" ]]; then
  read -r count < "$CI_TIMER_DATE_STATE_PATH"
fi
count=$((count + 1))
printf '%s' "$count" > "$CI_TIMER_DATE_STATE_PATH"
if [[ "$count" -eq 3 ]]; then
  printf 'ready' > "$CI_TIMER_FOOTER_READY_PATH"
  while [[ ! -e "$CI_TIMER_FOOTER_RELEASE_PATH" ]]; do
    "$CI_TIMER_SLEEP_PATH" 0.01
  done
fi
exec "$CI_TIMER_REAL_DATE" "$@"
""",
        encoding="utf-8",
    )
    date_shim.chmod(0o755)
    env.update(
        {
            "CI_STEP_TIMEOUT_SECONDS": "5",
            "CI_TIMER_DATE_STATE_PATH": str(date_state_path),
            "CI_TIMER_FOOTER_READY_PATH": str(footer_ready_path),
            "CI_TIMER_FOOTER_RELEASE_PATH": str(footer_release_path),
            "CI_TIMER_REAL_DATE": date_path,
            "CI_TIMER_SLEEP_PATH": sleep_path,
        }
    )

    wrapper: subprocess.Popen[str] | None = None
    try:
        wrapper = subprocess.Popen(
            [
                bash_path,
                str(script),
                "python-fallback-footer-signal",
                sys.executable,
                "-c",
                "print('footer-ready-command')",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        _wait_for_nonempty_file(footer_ready_path)
        os.kill(wrapper.pid, signal.SIGTERM)
        footer_release_path.touch()
        stdout, _stderr = wrapper.communicate(timeout=5)

        assert wrapper.returncode == 128 + signal.SIGTERM
        assert "footer-ready-command" in stdout
        assert 'ci_step_timer step_end label="python-fallback-footer-signal" status=143' in stdout
        assert "::endgroup::" in stdout
    finally:
        footer_release_path.touch()
        if wrapper is not None and wrapper.poll() is None:
            _kill_process_group(wrapper.pid)
            wrapper.wait(timeout=5)


@pytest.mark.parametrize("timeout_value", ["nan", "inf", "-Infinity"])
def test_ci_step_timer_python_fallback_rejects_nonfinite_timeout_before_command(
    tmp_path: Path,
    timeout_value: str,
) -> None:
    """Reject non-finite fallback durations with status 125 without launching the command."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    bash_path = shutil.which("bash")
    assert bash_path, "bash is required by ci_step_timer.sh"

    sentinel_path = tmp_path / "command-ran"
    env = _timeout_fallback_env(tmp_path)
    env["CI_STEP_TIMEOUT_SECONDS"] = timeout_value
    result = subprocess.run(
        [
            bash_path,
            str(script),
            "python-fallback-nonfinite-timeout",
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
            str(sentinel_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
        env=env,
    )

    assert result.returncode == 125
    assert not sentinel_path.exists()
    assert "timeout must be finite and greater than zero" in result.stderr
    assert 'ci_step_timer step_end label="python-fallback-nonfinite-timeout" status=125' in (
        result.stdout
    )
    assert "::endgroup::" in result.stdout


@pytest.mark.skipif(not _HAS_TIMEOUT, reason="GNU timeout(1) is required for this contract test")
@pytest.mark.parametrize("timeout_value", ["inf", "0", "0e42"])
def test_ci_step_timer_gnu_backend_rejects_invalid_timeout_before_command(
    tmp_path: Path,
    timeout_value: str,
) -> None:
    """Keep invalid GNU timeout values fail-closed before they can run unbounded."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    sentinel_path = tmp_path / "command-ran"
    env = os.environ.copy()
    env["CI_STEP_TIMEOUT_SECONDS"] = timeout_value
    result = subprocess.run(
        [
            "bash",
            str(script),
            "gnu-nonfinite-timeout",
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
            str(sentinel_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
        env=env,
    )

    assert result.returncode == 125
    assert not sentinel_path.exists()
    assert "timeout must be finite and greater than zero" in result.stderr
    assert 'ci_step_timer step_end label="gnu-nonfinite-timeout" status=125' in result.stdout
    assert "::endgroup::" in result.stdout
