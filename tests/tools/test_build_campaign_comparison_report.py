"""Tests for campaign result-store comparison reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.tools.build_campaign_comparison_report import (
    build_markdown,
    build_report,
    main,
)
from scripts.tools.campaign_result_store import write_result_store


def _write_fixture_store(path: Path) -> None:
    """Create a small result store with valid and limited rows."""
    fixture = Path("tests/fixtures/campaign_result_store/issue_3063_episode_rows.json")
    rows = json.loads(fixture.read_text(encoding="utf-8"))
    write_result_store(
        path,
        rows,
        study_id="issue-3063-fixture",
        command="uv run python scripts/tools/build_campaign_comparison_report.py ...",
        source_commit="abc123",
    )


def test_build_report_surfaces_uncertainty_denominators_and_caveats(tmp_path: Path) -> None:
    """Report payload should expose metrics, denominators, and invalid-row caveats."""
    result_store = tmp_path / "result-store"
    _write_fixture_store(result_store)

    payload = build_report(
        result_store,
        input_label="tests/fixtures/campaign_result_store/issue_3063_episode_rows.json",
        min_sample=3,
    )

    assert payload["schema_version"] == "campaign-comparison-report.v1"
    assert payload["report_status"] == "analysis_only"
    assert (
        payload["input"]["durable_input_label"]
        == "tests/fixtures/campaign_result_store/issue_3063_episode_rows.json"
    )
    assert payload["row_status"]["benchmark_valid_episode_count"] == 2
    assert payload["row_status"]["excluded_or_limited_episode_count"] == 2
    caveats = {row["row_status"]: row["interpretation"] for row in payload["row_status"]["caveats"]}
    assert caveats["fallback"] == "excluded_or_limited"
    assert caveats["degraded"] == "excluded_or_limited"
    planner_rows = {row["planner"]: row for row in payload["planner_summaries"]}
    assert planner_rows["goal"]["metrics"]["success"]["denominator"] == 2
    assert planner_rows["goal"]["metrics"]["success"]["mean"] == 0.5
    assert planner_rows["orca"]["metrics"]["snqi"]["denominator"] == 2
    assert any(row["metric"] == "snqi" for row in payload["metric_visual_summaries"])
    assert any(hook["sample_gate"] == "underpowered" for hook in payload["statistical_hooks"])


def test_build_markdown_includes_visual_summaries_and_statistical_hooks(tmp_path: Path) -> None:
    """Markdown should make caveats and descriptive hooks visible."""
    result_store = tmp_path / "result-store"
    _write_fixture_store(result_store)

    markdown = build_markdown(build_report(result_store, min_sample=1))

    assert "## Row Status Caveats" in markdown
    assert "| fallback | 1 | excluded_or_limited |" in markdown
    assert "## Metric Visual Summaries" in markdown
    assert "social_compliance" in markdown
    assert "## Statistical Hooks" in markdown
    assert "descriptive_only_formal_test_not_run" in markdown


def test_main_writes_json_and_markdown_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """CLI should write both report artifacts from a valid result store."""
    result_store = tmp_path / "result-store"
    _write_fixture_store(result_store)
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_campaign_comparison_report.py",
            "--result-store",
            str(result_store),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--input-label",
            "tests/fixtures/campaign_result_store/issue_3063_episode_rows.json",
            "--min-sample",
            "1",
        ],
    )

    assert main() == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["input"]["study_id"] == "issue-3063-fixture"
    assert (
        payload["input"]["durable_input_label"]
        == "tests/fixtures/campaign_result_store/issue_3063_episode_rows.json"
    )
    assert "Campaign Comparison Report" in output_md.read_text(encoding="utf-8")


def test_main_fails_closed_for_incomplete_result_store(tmp_path: Path, monkeypatch) -> None:
    """Invalid result-store inputs should not produce ad hoc reports."""
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_campaign_comparison_report.py",
            "--result-store",
            str(incomplete),
            "--output-json",
            str(tmp_path / "report.json"),
            "--output-md",
            str(tmp_path / "report.md"),
        ],
    )

    assert main() == 1
