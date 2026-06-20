"""Tests for the Issue #2904 forecast-risk eligibility fixtures.

These durable fixtures exist so the forecast-risk calibration filter produces
risk-scoring-eligible rows instead of returning ``wait``-for-lack-of-rows. The
tests assert eligibility, actor-class coverage, observation-tier coverage, and
that the calibration report recommendation is a real decision (not ``wait``).
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.forecast_calibration_report import (
    build_forecast_calibration_report,
)
from scripts.validation import validate_forecast_risk_calibration_filter as filter_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = REPO_ROOT / "tests/fixtures/benchmark/forecast_risk_eligibility"
_PEDESTRIAN_FIXTURE = _FIXTURE_DIR / "metric_pedestrian_deployable_tracked.json"
_BICYCLE_FIXTURE = _FIXTURE_DIR / "metric_bicycle_deployable_observation.json"
_DURABLE_CALIBRATION_REPORT = (
    REPO_ROOT
    / "docs/context/evidence/issue_2904_forecast_risk_eligibility_2026-06-20"
    / "calibration_report.json"
)


def _load_fixtures() -> list[dict[str, object]]:
    return [
        json.loads(_PEDESTRIAN_FIXTURE.read_text(encoding="utf-8")),
        json.loads(_BICYCLE_FIXTURE.read_text(encoding="utf-8")),
    ]


def _build_report() -> dict[str, object]:
    return build_forecast_calibration_report(
        _load_fixtures(),
        report_id="issue_2904_forecast_risk_eligibility_2026-06-20",
        coverage_target=0.9,
        coverage_tolerance=0.05,
        generated_at_utc="2026-06-20T00:00:00+00:00",
    )


def test_fixtures_produce_risk_scoring_eligible_rows() -> None:
    """At least one reliability row must be eligible for forecast-risk scoring."""
    report = _build_report()
    eligible = [
        row
        for row in report["reliability_rows"]
        if row["risk_scoring_eligibility"] == "eligible_analysis_only"
    ]
    assert len(eligible) >= 1
    assert report["limitation_rows"] == []


def test_fixtures_cover_pedestrian_and_dynamic_actor_classes() -> None:
    """The fixtures must contribute a pedestrian and a bicycle actor class."""
    report = _build_report()
    actor_classes = {row["actor_class"] for row in report["reliability_rows"]}
    assert "pedestrian" in actor_classes
    assert "bicycle" in actor_classes
    assert "unavailable" not in actor_classes


def test_fixtures_span_at_least_two_observation_tiers() -> None:
    """The fixtures must span at least two distinct observation tiers."""
    report = _build_report()
    tiers = {row["observation_tier"] for row in report["reliability_rows"]}
    assert tiers == {"deployable_tracked", "deployable_observation"}
    assert len(tiers) >= 2


def test_fixtures_yield_real_recommendation_not_wait() -> None:
    """The calibration recommendation must be a real decision, not ``wait``."""
    report = _build_report()
    decision = report["recommendation"]["decision"]
    assert decision in {"continue", "revise"}
    assert decision != "wait"


def test_durable_report_unblocks_filter_calibration_filtered_mode() -> None:
    """End-to-end: the durable report unblocks the filter's calibration_filtered mode (#2904)."""
    assert _DURABLE_CALIBRATION_REPORT.exists()
    diagnostic = filter_validator.build_report(calibration_path=_DURABLE_CALIBRATION_REPORT)
    by_mode = {row["mode"]: row["status"] for row in diagnostic["modes"]}
    assert by_mode["calibration_filtered"] == "available"
    assert by_mode["actor_class_aware"] == "available"
    assert by_mode["observation_tier_aware"] == "available"
    # Overall recommendation must be a real bounded decision, not wait-for-lack-of-rows.
    assert diagnostic["recommendation"] != "wait"
