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


def test_run_compact_validation_rejects_empty_command() -> None:
    """The library helper should fail loudly for an empty command."""
    try:
        run_compact_validation([])
    except ValueError as exc:
        assert "command must not be empty" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")
