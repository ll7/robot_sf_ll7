"""Tests for compact validation command summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.dev.run_compact_validation import main, run_compact_validation


def test_run_compact_validation_bounds_failure_output(tmp_path: Path, capsys) -> None:
    """Many failure lines should produce a bounded parent-thread summary."""
    artifact_dir = tmp_path / "artifacts"
    command = [
        sys.executable,
        "-c",
        "\n".join(
            [
                "import sys",
                "print('============================= FAILURES =============================')",
                "for i in range(80):",
                "    print(f'FAILED tests/dev/test_many.py::test_{i} - boom')",
                "sys.exit(7)",
            ]
        ),
    ]

    rc = main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--excerpt-lines",
            "5",
            "--",
            *command,
        ]
    )

    stdout = capsys.readouterr().out
    summaries = list(artifact_dir.glob("*.summary.json"))
    logs = list(artifact_dir.glob("*.log"))
    assert rc == 7
    assert "Exit code: 7" in stdout
    assert "Full log:" in stdout
    assert "FAILED tests/dev/test_many.py::test_0 - boom" in stdout
    assert "test_79" not in stdout
    assert "additional matching lines omitted" in stdout
    assert len(summaries) == 1
    assert len(logs) == 1
    assert "test_79" in logs[0].read_text(encoding="utf-8")
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["exit_code"] == 7
    assert summary["excerpt_line_count"] == 5
    assert summary["excerpt_truncated"] is True
    assert summary["failing_node_ids"][0] == "tests/dev/test_many.py::test_0"


def test_run_compact_validation_json_summary_for_success(tmp_path: Path, capsys) -> None:
    """Successful commands should save logs without a failure-labelled excerpt."""
    artifact_dir = tmp_path / "artifacts"
    command = [sys.executable, "-c", "print('ok')"]

    rc = main(["--artifact-dir", str(artifact_dir), "--json", "--", *command])

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert rc == 0
    assert payload["exit_code"] == 0
    assert payload["failure_excerpt"] == []
    assert payload["excerpt_line_count"] == 0
    assert payload["excerpt_truncated"] is False
    assert Path(payload["log_path"]).read_text(encoding="utf-8") == "ok\n"
    assert Path(payload["summary_path"]).exists()


def test_run_compact_validation_suppresses_failure_details_on_success(
    tmp_path: Path, capsys
) -> None:
    """Passing pytest-like logs should not look like failed test summaries."""
    artifact_dir = tmp_path / "artifacts"
    command = [
        sys.executable,
        "-c",
        "\n".join(
            [
                "print('tests/planner/test_policy_stack_v1.py::test_policy_stack_runs_atomic_topology_smoke_through_map_runner')",
                "print('tests/examples/test_examples_run.py::test_example_runs_without_error[quickstart/01_basic_robot.py]')",
                "print('============================= slowest 10 durations =============================')",
                "print('18.15s call     tests/planner/test_policy_stack_v1.py::test_policy_stack_runs_atomic_topology_smoke_through_map_runner')",
                "print('=========== 7588 passed, 12 skipped, 9 warnings in 328.89s ===========')",
            ]
        ),
    ]

    rc = main(["--artifact-dir", str(artifact_dir), "--", *command])
    stdout = capsys.readouterr().out
    summary = json.loads(next(artifact_dir.glob("*.summary.json")).read_text(encoding="utf-8"))

    assert rc == 0
    assert summary["exit_code"] == 0
    assert summary["failing_node_ids"] == []
    assert summary["failure_excerpt"] == []
    assert "Failure excerpt:" not in stdout


def test_run_compact_validation_keeps_node_ids_on_failure(tmp_path: Path) -> None:
    """Nonzero pytest-like output should still report failing node ids."""
    artifact_dir = tmp_path / "artifacts"
    command = [
        sys.executable,
        "-c",
        "\n".join(
            [
                "import sys",
                "print('FAILED tests/dev/test_compact.py::test_real_failure - AssertionError')",
                "sys.exit(1)",
            ]
        ),
    ]

    summary = run_compact_validation(command, artifact_dir=artifact_dir)

    assert summary["exit_code"] == 1
    assert summary["failing_node_ids"] == ["tests/dev/test_compact.py::test_real_failure"]


def test_run_compact_validation_suppresses_plain_output_on_success(tmp_path: Path) -> None:
    """Successful output should not create a failure excerpt."""
    artifact_dir = tmp_path / "artifacts"
    command = [
        sys.executable,
        "-c",
        "for i in range(30): print(f'plain output line {i}')",
    ]

    summary = run_compact_validation(command, artifact_dir=artifact_dir, excerpt_lines=3)

    assert summary["exit_code"] == 0
    assert summary["excerpt_truncated"] is False
    assert summary["failure_excerpt"] == []


def test_run_compact_validation_emits_summary_on_timeout(tmp_path: Path, capsys) -> None:
    """Timed-out commands should still leave compact evidence artifacts."""
    artifact_dir = tmp_path / "artifacts"
    command = [
        sys.executable,
        "-c",
        "import time; print('before timeout', flush=True); time.sleep(5)",
    ]

    rc = main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--timeout-seconds",
            "1.0",
            "--",
            *command,
        ]
    )

    stdout = capsys.readouterr().out
    summaries = list(artifact_dir.glob("*.summary.json"))
    logs = list(artifact_dir.glob("*.log"))
    assert rc == 124
    assert "Exit code: 124" in stdout
    assert "Timed out: 1.0 seconds" in stdout
    assert len(summaries) == 1
    assert len(logs) == 1
    log_text = logs[0].read_text(encoding="utf-8")
    assert "before timeout" in log_text
    assert "Command timed out after 1 seconds." in log_text
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["schema"] == "compact_validation_summary.v2"
    assert summary["exit_code"] == 124
    assert summary["timed_out"] is True
    assert summary["timeout_seconds"] == 1.0
    assert summary["timeout_message"] == "Command timed out after 1 seconds."
    assert summary["cleanup_status"] == "process_group_terminated_and_waited"
    assert "before timeout" in summary["failure_excerpt"]
    assert "Command timed out after 1 seconds." in summary["failure_excerpt"]


def test_run_compact_validation_timeout_handles_descendant_output_handles(tmp_path: Path) -> None:
    """Grandchildren inheriting stdout should not prevent timeout summary emission."""
    artifact_dir = tmp_path / "artifacts"
    command = [
        sys.executable,
        "-c",
        "\n".join(
            [
                "import subprocess, sys, time",
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])",
                "print('spawned descendant', flush=True)",
                "time.sleep(5)",
            ]
        ),
    ]

    summary = run_compact_validation(
        command,
        artifact_dir=artifact_dir,
        timeout_seconds=1.0,
    )

    assert summary["exit_code"] == 124
    assert summary["timed_out"] is True
    assert summary["cleanup_status"] == "process_group_terminated_and_waited"
    assert "spawned descendant" in "\n".join(summary["failure_excerpt"])


def test_run_compact_validation_rejects_non_positive_timeout() -> None:
    """Timeout configuration should fail before running a command when invalid."""
    try:
        main(["--timeout-seconds", "0", "--", sys.executable, "-c", "print('unused')"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected SystemExit")


def test_run_compact_validation_rejects_empty_command() -> None:
    """The library helper should fail loudly for an empty command."""
    try:
        run_compact_validation([])
    except ValueError as exc:
        assert "command must not be empty" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")
