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
    """Successful commands should still save full logs and emit summary JSON."""
    artifact_dir = tmp_path / "artifacts"
    command = [sys.executable, "-c", "print('ok')"]

    rc = main(["--artifact-dir", str(artifact_dir), "--json", "--", *command])

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert rc == 0
    assert payload["exit_code"] == 0
    assert payload["failure_excerpt"] == ["ok"]
    assert Path(payload["log_path"]).read_text(encoding="utf-8") == "ok\n"
    assert Path(payload["summary_path"]).exists()


def test_run_compact_validation_marks_truncated_plain_output(tmp_path: Path) -> None:
    """Large output without failure keywords should still report truncation."""
    artifact_dir = tmp_path / "artifacts"
    command = [
        sys.executable,
        "-c",
        "for i in range(30): print(f'plain output line {i}')",
    ]

    summary = run_compact_validation(command, artifact_dir=artifact_dir, excerpt_lines=3)

    assert summary["exit_code"] == 0
    assert summary["excerpt_truncated"] is True
    assert summary["failure_excerpt"] == [
        "plain output line 27",
        "plain output line 28",
        "plain output line 29",
    ]


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
