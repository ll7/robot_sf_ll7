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


def test_process_group_check_reads_portable_session_ids_from_ps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The group check uses macOS-compatible ``ps`` session fields without `getsid`."""
    observed_command: list[str] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        observed_command.extend(command)
        return SimpleNamespace(returncode=0, stdout="123 456\n")

    monkeypatch.setattr(smoke_validator.subprocess, "run", fake_run)
    monkeypatch.setattr(os, "getsid", lambda _: pytest.fail("unexpected getsid call"))

    assert smoke_validator._process_group_is_current(process_group_id=123, session_id=456)
    assert observed_command == ["ps", "-axo", "pgid=,sess="]


def test_process_session_field_reads_macos_compatible_ps_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the session value in the spelling both macOS and Linux accept."""
    observed_command: list[str] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        observed_command.extend(command)
        return SimpleNamespace(returncode=0, stdout="0\n")

    monkeypatch.setattr(smoke_validator.subprocess, "run", fake_run)

    assert smoke_validator._process_session_field(123) == 0
    assert observed_command == ["ps", "-o", "sess=", "-p", "123"]


def test_process_session_field_rejects_missing_or_malformed_ps_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup fails closed when the platform cannot identify its child session."""
    monkeypatch.setattr(
        smoke_validator.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert smoke_validator._process_session_field(123) is None

    monkeypatch.setattr(
        smoke_validator.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="not-a-session\n"),
    )
    assert smoke_validator._process_session_field(123) is None


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
            timeout_seconds=1.0,
            progress_timeout_seconds=0.12,
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
def test_group_lookup_failure_still_reaps_watchdog_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed group validation falls through to direct-child cleanup."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    monkeypatch.setattr(smoke_validator, "_process_group_is_current", lambda **_: False)

    smoke_validator._terminate_process_group(
        process,
        process_group_id=process.pid,
        session_id=process.pid,
    )

    assert process.returncode is not None


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
            timeout_seconds=0.12,
            progress_timeout_seconds=0.12,
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
