"""Tests for GitHub Actions CI timing summary helpers."""

from __future__ import annotations

import json

from scripts.dev.ci_timing_summary import format_markdown, main, summarize_run


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
    assert "| Unit tests | 120.0s |" in markdown
    assert "| Validation smoke tests | 30.0s |" in markdown


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
