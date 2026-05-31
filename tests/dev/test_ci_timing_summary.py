"""Tests for GitHub Actions CI timing summary helpers."""

from __future__ import annotations

import json

import pytest

from scripts.dev.ci_timing_summary import (
    format_markdown,
    main,
    parse_phase_timings,
    summarize_run,
)


def _sample_run_payload() -> dict[str, object]:
    """Return a compact GitHub Actions run payload with timed steps."""
    return {
        "databaseId": 123,
        "displayTitle": "sample",
        "createdAt": "2026-05-05T11:00:00Z",
        "updatedAt": "2026-05-05T11:03:00Z",
        "jobs": [
            {
                "name": "ci",
                "startedAt": "2026-05-05T11:00:10Z",
                "completedAt": "2026-05-05T11:03:00Z",
                "steps": [
                    {
                        "name": "Set up job",
                        "startedAt": "2026-05-05T11:00:10Z",
                        "completedAt": "2026-05-05T11:00:20Z",
                    },
                    {
                        "name": "Unit tests",
                        "startedAt": "2026-05-05T11:00:30Z",
                        "completedAt": "2026-05-05T11:02:30Z",
                    },
                    {
                        "name": "Validation smoke tests",
                        "startedAt": "2026-05-05T11:02:30Z",
                        "completedAt": "2026-05-05T11:03:00Z",
                    },
                ],
            }
        ],
    }


def test_summarize_run_reports_queue_job_and_slowest_steps() -> None:
    """Timing summary should separate queue, job, and step-level durations."""
    summary = summarize_run(_sample_run_payload(), top=2)

    assert summary.run_id == 123
    assert summary.queue_seconds == 10.0
    assert summary.job_seconds == 170.0
    assert summary.slowest_jobs[0].name == "ci"
    assert summary.slowest_jobs[0].duration_seconds == 170.0
    assert [step.name for step in summary.slowest_steps] == [
        "Unit tests",
        "Validation smoke tests",
    ]
    assert summary.slowest_steps[0].duration_seconds == 120.0


def test_format_markdown_includes_phase_totals() -> None:
    """Markdown output should be compact enough for issues and PR comments."""
    summary = summarize_run(_sample_run_payload(), top=2)

    markdown = format_markdown(summary)

    assert "Run 123" in markdown
    assert "| queue | 10.0s |" in markdown
    assert "| ci | 170.0s |" in markdown
    assert "| Unit tests | 120.0s |" in markdown
    assert "| Validation smoke tests | 30.0s |" in markdown


def test_format_markdown_escapes_table_separators() -> None:
    """Markdown tables should not break when GitHub names contain pipes."""
    payload = _sample_run_payload()
    payload["jobs"][0]["name"] = "ci | linux"
    payload["jobs"][0]["steps"][1]["name"] = "Unit | tests"

    markdown = format_markdown(summarize_run(payload, top=2))

    assert "| ci \\| linux | 170.0s |" in markdown
    assert "| Unit \\| tests | 120.0s |" in markdown


def test_summarize_run_ignores_malformed_job_payloads() -> None:
    """Malformed GitHub job payloads should degrade to an empty timing summary."""
    payload = {
        "databaseId": 123,
        "displayTitle": "sample",
        "createdAt": "2026-05-05T11:00:00Z",
        "updatedAt": "2026-05-05T11:03:00Z",
        "jobs": None,
    }

    summary = summarize_run(payload)

    assert summary.slowest_jobs == []
    assert summary.slowest_steps == []
    assert summary.queue_seconds == 0.0
    assert summary.job_seconds == 0.0


def test_format_markdown_reports_missing_step_timestamps() -> None:
    """Markdown should still expose job timing when steps lack timestamps."""
    payload = _sample_run_payload()
    for job in payload["jobs"]:
        for step in job["steps"]:
            step.pop("startedAt", None)
            step.pop("completedAt", None)

    markdown = format_markdown(summarize_run(payload))

    assert "| ci | 170.0s |" in markdown
    assert "No step timestamps reported" in markdown


def test_parse_phase_timings_extracts_ci_driver_phase_end_lines() -> None:
    """ci_driver phase_end log lines should become sortable repository phase timings."""
    log_text = "\n".join(
        [
            "ci_driver phase_start phase=lint started_at=2026-05-05T11:00:00Z",
            "ci_driver phase_end phase=lint status=0 duration_seconds=12 completed_at=2026-05-05T11:00:12Z",
            "ci_driver phase_end phase=test status=1 duration_seconds=540 completed_at=2026-05-05T11:09:00Z",
            "ci_driver phase_end phase=bad duration_seconds=not-a-number",
        ]
    )

    phases = parse_phase_timings(log_text, source="fast-feedback.log")

    assert [(phase.name, phase.duration_seconds, phase.status) for phase in phases] == [
        ("lint", 12.0, 0),
        ("test", 540.0, 1),
    ]
    assert phases[0].source == "fast-feedback.log"


def test_parse_phase_timings_accepts_shell_quoted_fields() -> None:
    """ci_driver phase_end fields may contain shell-quoted names and values."""
    log_text = (
        "ci_driver phase_end phase='unit tests shard 1' status='0' "
        "duration_seconds='12.5' completed_at='2026-05-05T11:00:12Z'\n"
    )

    phases = parse_phase_timings(log_text, source="quoted.log")

    assert [(phase.name, phase.duration_seconds, phase.status) for phase in phases] == [
        ("unit tests shard 1", 12.5, 0),
    ]
    assert phases[0].completed_at == "2026-05-05T11:00:12Z"


def test_parse_phase_timings_skips_malformed_shell_fields() -> None:
    """Malformed shell syntax in one log line should not abort parsing."""
    log_text = "\n".join(
        [
            "ci_driver phase_end phase='unterminated status=0 duration_seconds=12",
            "ci_driver phase_end phase=lint status=0 duration_seconds=3 completed_at=2026-05-05T11:00:03Z",
        ]
    )

    phases = parse_phase_timings(log_text, source="malformed.log")

    assert [(phase.name, phase.duration_seconds, phase.status) for phase in phases] == [
        ("lint", 3.0, 0),
    ]


def test_format_markdown_includes_repository_phase_timings() -> None:
    """Markdown summaries should surface ci_driver phase timings when logs are provided."""
    summary = summarize_run(
        _sample_run_payload(),
        top=2,
        phase_timings=parse_phase_timings(
            "ci_driver phase_end phase=test status=0 duration_seconds=540 completed_at=2026-05-05T11:09:00Z\n",
            source="fast-feedback.log",
        ),
    )

    markdown = format_markdown(summary)

    assert "## Slowest repository phases" in markdown
    assert "| test | 540.0s | 0 | 2026-05-05T11:09:00Z | fast-feedback.log |" in markdown


def test_main_reads_run_json_and_prints_markdown(tmp_path, capsys) -> None:
    """CLI should summarize saved `gh run view --json ...` output."""
    run_json = tmp_path / "run.json"
    run_json.write_text(json.dumps(_sample_run_payload()), encoding="utf-8")

    exit_code = main(["--run-json", str(run_json), "--top", "1"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "sample" in output
    assert "Unit tests" in output
    assert "Validation smoke tests" not in output


def test_main_accepts_log_only_timing_summary(tmp_path, capsys) -> None:
    """CLI should support a saved log when no gh run JSON is available."""
    log_path = tmp_path / "fast-feedback.log"
    log_path.write_text(
        "ci_driver phase_end phase=test status=0 duration_seconds=540 completed_at=2026-05-05T11:09:00Z\n",
        encoding="utf-8",
    )

    exit_code = main(["--log", str(log_path), "--top", "1"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "CI log timing summary" in output
    assert "| test | 540.0s | 0 | 2026-05-05T11:09:00Z |" in output


def test_main_rejects_missing_log_path(tmp_path) -> None:
    """CLI should fail cleanly before reading a missing log path."""
    missing_log = tmp_path / "missing.log"

    with pytest.raises(SystemExit) as excinfo:
        main(["--log", str(missing_log)])

    assert excinfo.value.code == f"Log file not found: {missing_log}"


def test_main_rejects_directory_log_path(tmp_path) -> None:
    """CLI should fail cleanly when a log path points to a directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        main(["--log", str(log_dir)])

    assert excinfo.value.code == f"Log file not found: {log_dir}"
