"""Contract tests for the issue #5416 native SIPP smoke validator."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.validation import check_issue_5416_sipp_native_smoke as smoke_validator
from scripts.validation.check_issue_5416_sipp_native_smoke import SmokeError, validate_smoke


def test_watchdog_sentinel_identity_uses_its_live_group_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live sentinel, not a portable-but-ambiguous session number, owns cleanup."""
    process = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(os, "getpgid", lambda process_id: process_id)
    monkeypatch.setattr(
        smoke_validator.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("cleanup must not identify groups through ps"),
    )
    monkeypatch.setattr(os, "getsid", lambda _: pytest.fail("cleanup must not use sess=0"))

    assert smoke_validator._watchdog_sentinel_owns_process_group(process, process_group_id=123)


def test_watchdog_sentinel_identity_rejects_a_reused_group_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live process with a different group can never authorize `killpg`."""
    process = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(os, "getpgid", lambda _: 999)

    assert not smoke_validator._watchdog_sentinel_owns_process_group(process, process_group_id=123)


def test_smoke_arguments_are_pinned_before_expensive_execution(tmp_path: Path) -> None:
    """The reusable validator cannot be repurposed into an unreviewed campaign runner."""
    with pytest.raises(SmokeError, match="must stay pinned"):
        validate_smoke(
            packet_path=Path(
                "configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml"
            ),
            native_config_path=Path("configs/algos/sipp_lattice_native_command.yaml"),
            scenario_id="classic_head_on_corridor_low",
            seed=112,
            horizon=500,
            dt=0.1,
            workers=1,
            output_dir=tmp_path,
        )


def test_main_reports_unexpected_standard_exception_as_blocked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Unexpected validator errors still produce the fail-closed JSON contract."""

    def raise_assertion_error(**_: object) -> dict[str, object]:
        raise AssertionError("unexpected validation failure")

    monkeypatch.setattr(smoke_validator, "validate_smoke", raise_assertion_error)

    result = smoke_validator.main(
        [
            "--scenario-id",
            "classic_head_on_corridor_low",
            "--seed",
            "111",
            "--horizon",
            "500",
            "--dt",
            "0.1",
            "--workers",
            "1",
            "--output-dir",
            str(tmp_path),
            "--json",
        ]
    )

    assert result == 1
    assert capsys.readouterr().out == (
        '{"error": "unexpected validation failure", "status": "blocked"}\n'
    )


def test_main_reports_watchdog_timeout_as_structured_blocked_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A watchdog failure preserves its planner and cleanup evidence in JSON output."""

    def raise_watchdog_error(**_: object) -> dict[str, object]:
        raise smoke_validator.SmokeWatchdogError(
            "native smoke exceeded its end-to-end watchdog timeout",
            details={
                "failure_kind": "native_end_to_end_timeout",
                "planner_id": "teb",
                "child_process_terminated": True,
            },
        )

    monkeypatch.setattr(smoke_validator, "validate_smoke", raise_watchdog_error)

    result = smoke_validator.main(
        [
            "--scenario-id",
            "classic_head_on_corridor_low",
            "--seed",
            "111",
            "--horizon",
            "500",
            "--dt",
            "0.1",
            "--workers",
            "1",
            "--output-dir",
            str(tmp_path),
            "--json",
        ]
    )

    assert result == 1
    assert capsys.readouterr().out == (
        '{"child_process_terminated": true, "error": "native smoke exceeded its end-to-end '
        'watchdog timeout", "failure_kind": "native_end_to_end_timeout", "planner_id": "teb", '
        '"status": "blocked"}\n'
    )


def test_five_planner_mode_omits_single_row_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The frozen five-planner mode needs only its own packet and output inputs."""

    monkeypatch.setattr(
        smoke_validator,
        "validate_five_planner_smoke",
        lambda **_: {"status": "ready", "eligible_rows": 5, "excluded_rows": 0},
    )

    result = smoke_validator.main(["--five-planner-smoke", "--output-dir", str(tmp_path), "--json"])

    assert result == 0
    assert capsys.readouterr().out == (
        '{"eligible_rows": 5, "excluded_rows": 0, "status": "ready"}\n'
    )


def test_standard_mode_still_requires_single_row_arguments(tmp_path: Path) -> None:
    """Relaxing the five-planner parser must not broaden the standard smoke contract."""

    with pytest.raises(SystemExit, match="2"):
        smoke_validator.main(["--output-dir", str(tmp_path)])


def test_native_watchdog_reports_end_to_end_timeout(tmp_path: Path) -> None:
    """A row that never writes must become a structured end-to-end timeout."""
    with pytest.raises(smoke_validator.SmokeWatchdogError) as raised:
        smoke_validator._run_native_row_with_watchdog(
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
            episodes_path=tmp_path / "episodes.jsonl",
            timeout_seconds=0.05,
            progress_timeout_seconds=0.05,
            planner_id="teb",
        )

    assert raised.value.details == {
        "failure_kind": "native_end_to_end_timeout",
        "planner_id": "teb",
        "timeout_seconds": 0.05,
        "progress_timeout_seconds": 0.05,
        "episode_path": str(tmp_path / "episodes.jsonl"),
        "child_process_terminated": True,
    }


def test_native_watchdog_resets_progress_deadline_when_episode_output_changes(
    tmp_path: Path,
) -> None:
    """Episode-file changes extend the progress deadline until output becomes stale again."""
    episodes_path = tmp_path / "episodes.jsonl"
    writer = (
        "from pathlib import Path; import sys, time; "
        "path = Path(sys.argv[1]); time.sleep(0.04); path.write_text('first\\n'); "
        "time.sleep(0.04); path.write_text('first\\nsecond\\n'); time.sleep(30)"
    )
    with pytest.raises(smoke_validator.SmokeWatchdogError) as raised:
        smoke_validator._run_native_row_with_watchdog(
            command=[sys.executable, "-c", writer, str(episodes_path)],
            episodes_path=episodes_path,
            timeout_seconds=1.5,
            progress_timeout_seconds=0.5,
            planner_id="teb",
        )

    assert raised.value.details["failure_kind"] == "native_progress_timeout"
    assert episodes_path.read_text(encoding="utf-8") == "first\nsecond\n"


def test_native_watchdog_terminates_when_progress_probe_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unexpected watchdog probe error cannot bypass child cleanup."""
    terminated: list[subprocess.Popen[bytes]] = []

    def record_termination(process: subprocess.Popen[bytes], **_: object) -> None:
        terminated.append(process)
        process.kill()
        process.wait()

    monkeypatch.setattr(
        smoke_validator,
        "_episode_progress_token",
        lambda _: (_ for _ in ()).throw(OSError("episode path became unreadable")),
    )
    monkeypatch.setattr(smoke_validator, "_terminate_process_group", record_termination)

    with pytest.raises(OSError, match="unreadable"):
        smoke_validator._run_native_row_with_watchdog(
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
            episodes_path=tmp_path / "episodes.jsonl",
            timeout_seconds=1.0,
            progress_timeout_seconds=0.5,
            planner_id="teb",
        )

    assert len(terminated) == 1


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-specific")
def test_sentinel_identity_failure_never_kills_a_reused_unrelated_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale numeric group id falls through without signalling the unrelated group."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    signalled_groups: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "getpgid", lambda _: process.pid + 1)
    monkeypatch.setattr(
        os,
        "killpg",
        lambda process_group_id, signal_number: signalled_groups.append(
            (process_group_id, signal_number)
        ),
    )

    smoke_validator._terminate_process_group(
        process,
        process_group_id=process.pid,
    )

    assert process.returncode is not None
    assert signalled_groups == []


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-specific")
def test_native_watchdog_terminates_descendant_processes(tmp_path: Path) -> None:
    """The timeout kills a native child process and its inherited process group."""
    pid_path = tmp_path / "descendant.pid"
    descendant = (
        "from pathlib import Path; import os, sys, time; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    parent = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]]); time.sleep(30)"
    )
    with pytest.raises(smoke_validator.SmokeWatchdogError, match="end-to-end"):
        smoke_validator._run_native_row_with_watchdog(
            command=[sys.executable, "-c", parent, str(pid_path), descendant],
            episodes_path=tmp_path / "episodes.jsonl",
            timeout_seconds=1.5,
            progress_timeout_seconds=1.5,
            planner_id="teb",
        )

    descendant_pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 1.0
    while True:
        try:
            os.kill(descendant_pid, 0)
        except OSError as exc:
            assert exc.errno == errno.ESRCH
            break
        if time.monotonic() >= deadline:
            pytest.fail("watchdog left the native descendant process running")
        time.sleep(0.02)


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-specific")
def test_native_watchdog_sigkills_descendant_that_ignores_sigterm(tmp_path: Path) -> None:
    """The live sentinel permits SIGKILL escalation after a graceful stop is ignored."""
    pid_path = tmp_path / "sigterm_ignoring_descendant.pid"
    descendant = (
        "from pathlib import Path; import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda _signum, _frame: None); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    parent = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]]); time.sleep(30)"
    )
    with pytest.raises(smoke_validator.SmokeWatchdogError, match="end-to-end"):
        smoke_validator._run_native_row_with_watchdog(
            command=[sys.executable, "-c", parent, str(pid_path), descendant],
            episodes_path=tmp_path / "episodes.jsonl",
            timeout_seconds=1.5,
            progress_timeout_seconds=1.5,
            planner_id="teb",
        )

    descendant_pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 1.0
    while True:
        try:
            os.kill(descendant_pid, 0)
        except OSError as exc:
            assert exc.errno == errno.ESRCH
            break
        if time.monotonic() >= deadline:
            pytest.fail("SIGTERM-ignoring native descendant survived watchdog escalation")
        time.sleep(0.02)


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-specific")
def test_native_watchdog_cleans_descendant_after_parent_exits(tmp_path: Path) -> None:
    """A successful parent cannot leave its native descendant behind after returning."""
    pid_path = tmp_path / "descendant.pid"
    descendant = (
        "from pathlib import Path; import os, sys, time; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    parent = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]]); time.sleep(0.04)"
    )
    smoke_validator._run_native_row_with_watchdog(
        command=[sys.executable, "-c", parent, str(pid_path), descendant],
        episodes_path=tmp_path / "episodes.jsonl",
        timeout_seconds=1.0,
        progress_timeout_seconds=0.5,
        planner_id="teb",
    )

    descendant_pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 1.0
    while True:
        try:
            os.kill(descendant_pid, 0)
        except OSError as exc:
            assert exc.errno == errno.ESRCH
            break
        if time.monotonic() >= deadline:
            pytest.fail("success-path cleanup left the native descendant process running")
        time.sleep(0.02)
