"""Tests for the fail-closed stale-base observation report."""

from __future__ import annotations

import copy
import hashlib
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


def _verified_observation_payload() -> dict:
    payload = _payload()
    payload["evidence_status"] = "workflow_observation"
    sources = [
        payload["policy"]["source_snapshot"],
        *payload["window"]["source_snapshots"],
        *payload["red_main_coverage"]["source_snapshots"],
    ]
    for source in sources:
        source["source_kind"] = "external_snapshot"
        source["path"] = f"sha256://{source['sha256']}"
    return payload


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
    base_sensitive = report["metrics"]["by_risk_tier"]["base_sensitive"]
    assert base_sensitive["base_sensitive_refresh_waits"] == 1
    assert base_sensitive["ordinary_cas_waits"] == 0
    assert report["red_main"]["by_classification"] == {
        "stale_base_attributable": 1,
        "not_attributable": 1,
        "unknown": 1,
    }
    assert report["red_main"]["rollback_condition"] == "unknown"
    assert report["evidence_class"] == "workflow_only"
    assert report["observations"][0]["evidence"]["head_sha"] == "a" * 40
    unknown_event = report["red_main"]["events"][2]
    assert unknown_event["evidence"] == {}
    assert unknown_event["classification_reason"] == "red-main exact head/base evidence is missing"


def test_baseline_comparison_preserves_fixture_boundary() -> None:
    report = analyze_observation(_payload())

    assert report["baseline"]["status"] == "fixture_only"
    assert report["comparison"]["status"] == "fixture_only"


def test_fixture_cannot_be_promoted_by_evidence_status_edit() -> None:
    payload = _payload()
    payload["evidence_status"] = "workflow_observation"

    report = analyze_observation(payload)

    assert report["status"] == "invalid_contract"
    assert any("fixture source cannot support" in error for error in report["validation_errors"])


def test_fixture_baseline_cannot_become_available_in_observation_report() -> None:
    report = analyze_observation(_verified_observation_payload())

    assert report["status"] == "available"
    assert report["baseline"]["status"] == "fixture_only"
    assert report["comparison"]["status"] == "fixture_only"


def test_non_hold_record_never_enters_wait_percentiles() -> None:
    payload = _payload()
    payload["records"][3]["attribution"] = "stale_base"
    payload["records"][3]["evidence"] = copy.deepcopy(payload["records"][0]["evidence"])

    report = analyze_observation(payload)

    assert report["status"] == "invalid_contract"
    assert any("requires a hold wait type" in error for error in report["validation_errors"])


def test_hold_outside_named_window_is_invalid() -> None:
    payload = _payload()
    payload["records"][0]["hold_started_at"] = "2026-08-17T23:00:00Z"

    report = analyze_observation(payload)

    assert report["status"] == "invalid_contract"
    assert any("outside the named window" in error for error in report["validation_errors"])


def test_missing_red_main_coverage_is_not_an_audited_zero() -> None:
    payload = _payload()
    payload.pop("red_main_coverage")

    report = analyze_observation(payload)

    assert report["status"] == "not_available"
    assert report["red_main"]["coverage_status"] == "not_available"
    assert report["red_main"]["rollback_condition"] == "unknown"
    assert report["red_main"]["by_classification"] == {
        "stale_base_attributable": 0,
        "not_attributable": 0,
        "unknown": 0,
    }


def test_complete_empty_red_main_coverage_is_an_explicit_zero() -> None:
    payload = _payload()
    payload["red_main_events"] = []

    report = analyze_observation(payload)

    assert report["red_main"]["coverage_status"] == "complete"
    assert report["red_main"]["events_total"] == 0
    assert report["red_main"]["rollback_condition"] == "not_met"


def test_report_preserves_record_evidence_missingness_and_input_digest(
    tmp_path: Path, capsys
) -> None:
    input_path = tmp_path / "window.json"
    input_bytes = FIXTURE.read_bytes()
    input_path.write_bytes(input_bytes)

    exit_code = main([str(input_path)])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["input_sha256"] == hashlib.sha256(input_bytes).hexdigest()
    missing = next(item for item in report["observations"] if item["pr_number"] == 1005)
    assert missing["duration_seconds"] is None
    assert missing["duration_missing"] == "hold_timestamps_unavailable"
    first = report["observations"][0]
    assert first["evidence"]["ci_base_sha"] == "b" * 40


def test_post_rollout_baseline_is_incompatible() -> None:
    payload = _verified_observation_payload()
    payload["baseline"]["window"]["end_at"] = "2026-08-18T11:00:00Z"

    report = analyze_observation(payload)

    assert report["status"] == "available"
    assert report["baseline"]["status"] == "incompatible"
    assert "must not follow policy.rollout_at" in " ".join(report["baseline"]["validation_errors"])
    assert report["comparison"]["status"] == "not_available"


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
    assert report["comparison"]["status"] == "fixture_only"
