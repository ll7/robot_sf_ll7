"""Tests for the fail-closed stale-base observation report."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.dev.measure_stale_base_policy import (
    REPORT_SCHEMA_VERSION,
    analyze_observation,
    main,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "stale_base_observation_window.v1.json"


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_reports_metrics_but_remains_fixture_only() -> None:
    report = analyze_observation(_payload(), input_path=str(FIXTURE))

    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["status"] == "fixture_only"
    ordinary = report["metrics"]["by_risk_tier"]["ordinary"]
    assert ordinary["records"] == 3
    assert ordinary["holds"] == 2
    assert ordinary["stale_base_holds"] == 2
    assert ordinary["p50_wait_seconds"] == 30.0
    assert ordinary["p95_wait_seconds"] == 90.0
    assert report["red_main"]["by_classification"] == {
        "stale_base_attributable": 1,
        "not_attributable": 1,
        "unknown": 1,
    }


def test_baseline_comparison_preserves_fixture_boundary() -> None:
    report = analyze_observation(_payload())

    assert report["baseline"]["status"] == "fixture_only"
    assert report["comparison"]["status"] == "available"
    assert report["comparison"]["by_risk_tier"]["ordinary"]["p50_wait_seconds"] == -15.0


def test_stale_attribution_with_head_mismatch_becomes_unknown() -> None:
    payload = _payload()
    payload["records"][0]["evidence"]["cas_head_sha"] = "4" * 40

    report = analyze_observation(payload)

    assert report["status"] == "fixture_only"
    ordinary = report["metrics"]["by_risk_tier"]["ordinary"]
    assert ordinary["stale_base_holds"] == 1
    assert ordinary["unknown_attribution"] == 1
    assert ordinary["wait_seconds_denominator"] == 1


def test_missing_duration_is_counted_without_becoming_zero() -> None:
    payload = _payload()
    payload["records"][2]["hold_started_at"] = None
    payload["records"][2]["hold_ended_at"] = None

    report = analyze_observation(payload)

    base_sensitive = report["metrics"]["by_risk_tier"]["base_sensitive"]
    assert base_sensitive["holds"] == 1
    assert base_sensitive["missing_wait_duration"] == 1
    assert base_sensitive["p50_wait_seconds"] is None


def test_invalid_schema_fails_closed() -> None:
    payload = _payload()
    payload["schema_version"] = "stale_base_observation_window.v0"

    report = analyze_observation(payload)

    assert report["status"] == "invalid_contract"
    assert report["metrics"] is None
    assert "schema_version" in " ".join(report["validation_errors"])


def test_missing_input_is_not_available_and_returns_exit_two(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.json"

    exit_code = main([str(missing)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert output["status"] == "not_available"
    assert output["metrics"] is None


def test_incompatible_baseline_does_not_block_current_window() -> None:
    payload = copy.deepcopy(_payload())
    payload["baseline"]["schema_version"] = "stale_base_observation_window.v0"

    report = analyze_observation(payload)

    assert report["status"] == "fixture_only"
    assert report["baseline"]["status"] == "incompatible"
    assert report["comparison"]["status"] == "not_available"
