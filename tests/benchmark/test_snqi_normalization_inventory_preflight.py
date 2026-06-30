"""Tests for the issue #3699 SNQI normalization-inventory preflight."""

from __future__ import annotations

import json

import pytest

from robot_sf.benchmark.snqi.compute import compute_snqi
from scripts.validation.check_snqi_normalization_inventory import (
    build_normalization_preflight_report,
    main,
)

_BASELINE_STATS = {
    "collisions": {"med": 1.0, "p95": 5.0},
    "near_misses": {"med": 2.0, "p95": 8.0},
    "force_exceed_events": {"med": 0.0, "p95": 4.0},
    "jerk_mean": {"med": 0.1, "p95": 1.0},
}

_WEIGHTS = {
    "w_success": 1.0,
    "w_time": 0.8,
    "w_collisions": 2.0,
    "w_near": 1.0,
    "w_comfort": 0.5,
    "w_force_exceed": 1.5,
    "w_jerk": 0.3,
}

_MIXED_BASIS_FIXTURE = {
    "success": 1.0,
    "time_to_goal_norm": 4.0,
    "collisions": 4.0,
    "near_misses": 9.0,
    "comfort_exposure": 12.5,
    "force_exceed_events": 5.0,
    "jerk_mean": 2.0,
}


def test_preflight_reports_mixed_basis_without_changing_snqi_output() -> None:
    """Synthetic mixed-basis fixture fails closed while scoring remains unchanged."""
    before = compute_snqi(_MIXED_BASIS_FIXTURE, _WEIGHTS, _BASELINE_STATS)
    report = build_normalization_preflight_report(
        _BASELINE_STATS,
        metrics=_MIXED_BASIS_FIXTURE,
        weights=_WEIGHTS,
    )
    after = compute_snqi(_MIXED_BASIS_FIXTURE, _WEIGHTS, _BASELINE_STATS)

    assert before == pytest.approx(after)
    assert report["status"] == "failed"
    assert [blocker["kind"] for blocker in report["blockers"]] == ["mixed_normalization_basis"]
    blocker = report["blockers"][0]
    assert blocker["raw_penalty_terms"] == ["time", "comfort"]
    assert blocker["normalized_penalty_terms"] == [
        "collisions",
        "near",
        "force_exceed",
        "jerk",
    ]
    status_by_term = {
        term["term"]: term["normalization_status"] for term in report["normalization"]["terms"]
    }
    assert status_by_term["time"] == "raw_unbounded"
    assert status_by_term["comfort"] == "raw_unbounded"
    assert status_by_term["collisions"] == "baseline_normalized_bounded"
    basis_by_term = {
        term["term"]: term["measurement_basis"] for term in report["normalization"]["terms"]
    }
    assert basis_by_term["time"] == "raw time-to-goal ratio"
    assert basis_by_term["comfort"] == "raw accumulated comfort-exposure value"
    assert basis_by_term["collisions"] == "baseline-relative median/p95 clamped value"
    contributions = report["contributions"]
    signed_total = sum(term["signed_contribution"] for term in contributions["terms"])
    assert contributions["diagnostic_only"] is True
    assert contributions["mixed_basis"] is True
    assert contributions["normalization_contract"]["status"] == "mixed_unbounded_penalty_basis"
    assert contributions["normalization_contract"]["weights_comparable"] is False
    assert signed_total == pytest.approx(before)


def test_preflight_main_fails_closed_but_allows_inspection(tmp_path) -> None:
    """Default command fails closed; explicit inspection mode exits successfully."""
    json_out = tmp_path / "snqi_normalization_inventory.json"

    assert main(["--json-out", str(json_out)]) == 2
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "snqi_normalization_inventory_preflight.v1"
    assert payload["status"] == "failed"
    assert payload["blockers"][0]["kind"] == "mixed_normalization_basis"

    assert main(["--json", "--allow-mixed-basis"]) == 0


def test_preflight_text_lists_each_term_status(capsys: pytest.CaptureFixture[str]) -> None:
    """Human output names each term's normalization status."""
    assert main(["--allow-mixed-basis"]) == 0
    out = capsys.readouterr().out

    assert "Term normalization status:" in out
    assert "time (time_to_goal_norm, w_time): raw_unbounded" in out
    assert "basis=raw time-to-goal ratio" in out
    assert "comfort (comfort_exposure, w_comfort): raw_unbounded" in out
    assert "basis=raw accumulated comfort-exposure value" in out
    assert "collisions (collisions, w_collisions): baseline_normalized_bounded" in out
    assert "basis=baseline-relative median/p95 clamped value" in out
    assert "jerk (jerk_mean, w_jerk): baseline_normalized_bounded" in out


def test_preflight_cli_writes_synthetic_fixture_contribution_report(tmp_path, capsys) -> None:
    """CLI accepts a synthetic mixed-basis fixture and writes contribution diagnostics."""
    baseline_path = tmp_path / "baseline_stats.json"
    metrics_path = tmp_path / "metrics.json"
    weights_path = tmp_path / "weights.json"
    report_path = tmp_path / "normalization_report.json"
    baseline_path.write_text(json.dumps(_BASELINE_STATS), encoding="utf-8")
    metrics_path.write_text(json.dumps(_MIXED_BASIS_FIXTURE), encoding="utf-8")
    weights_path.write_text(json.dumps(_WEIGHTS), encoding="utf-8")

    assert (
        main(
            [
                "--baseline-stats",
                str(baseline_path),
                "--metrics",
                str(metrics_path),
                "--weights",
                str(weights_path),
                "--json-out",
                str(report_path),
                "--allow-mixed-basis",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "Contribution contract: status=mixed_unbounded_penalty_basis" in captured.out
    assert payload["status"] == "failed"
    assert payload["contributions"]["normalization_contract"]["weights_comparable"] is False
    signed_total = sum(term["signed_contribution"] for term in payload["contributions"]["terms"])
    assert signed_total == pytest.approx(
        compute_snqi(_MIXED_BASIS_FIXTURE, _WEIGHTS, _BASELINE_STATS)
    )


def test_preflight_requires_metrics_and_weights_together(tmp_path, capsys) -> None:
    """Fixture diagnostics fail closed when metrics/weights are incomplete."""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_MIXED_BASIS_FIXTURE), encoding="utf-8")

    assert main(["--metrics", str(metrics_path)]) == 1

    captured = capsys.readouterr()
    assert "metrics and weights must be provided together" in captured.err


def test_preflight_reports_optional_missing_baseline_coverage(tmp_path) -> None:
    """Supplying incomplete baseline stats surfaces normalized-term coverage gaps."""
    baseline_stats = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
    }
    path = tmp_path / "baseline_stats.json"
    path.write_text(json.dumps(baseline_stats), encoding="utf-8")

    assert main(["--baseline-stats", str(path)]) == 2
    report = build_normalization_preflight_report(baseline_stats)
    missing = [b for b in report["blockers"] if b["kind"] == "missing_baseline_coverage"]

    assert missing
    assert set(missing[0]["metrics"]) == {"force_exceed_events", "jerk_mean"}
