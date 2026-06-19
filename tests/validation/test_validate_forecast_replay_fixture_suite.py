"""Tests for the issue #3146 forecast replay fixture suite validator."""

from __future__ import annotations

import json

from scripts.validation import validate_forecast_replay_fixture_suite


def test_manifest_suite_reports_diverse_forecast_replay_rows() -> None:
    """The tracked suite should cover diverse fixtures and classify each row."""

    summary = validate_forecast_replay_fixture_suite.run_suite(
        validate_forecast_replay_fixture_suite.DEFAULT_MANIFEST,
        generated_at_utc="2026-06-19T00:00:00Z",
    )

    assert summary["status"] == "passed"
    assert summary["fixture_count"] >= 3
    assert summary["scenario_family_count"] >= 3
    assert set(summary["variants"]) == {
        "none",
        "cv",
        "semantic",
        "interaction_aware",
        "risk_filtered",
    }
    assert summary["row_status_summary"]["native"] >= 3
    assert summary["row_status_summary"]["degraded"] >= 1
    assert max(row["non_none_closed_loop_signature_count"] for row in summary["rows"]) >= 2

    for row in summary["rows"]:
        assert row["execution_mode"] in {"native", "degraded", "diagnostic_only", "blocked"}
        assert row["row_classification"] == row["execution_mode"]
        assert len(row["variant_results"]) == 5
        for variant in ("cv", "semantic", "interaction_aware", "risk_filtered"):
            assert variant in row["same_seed_deltas"]
            assert "progress_m" in row["same_seed_deltas"][variant]


def test_cli_writes_compact_summary(tmp_path) -> None:
    """The CLI should write reviewable JSON and Markdown evidence."""

    exit_code = validate_forecast_replay_fixture_suite.main(
        [
            "--output-dir",
            str(tmp_path),
            "--generated-at-utc",
            "2026-06-19T00:00:00Z",
        ]
    )

    assert exit_code == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert (tmp_path / "README.md").exists()
