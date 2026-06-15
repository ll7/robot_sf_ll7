"""Tests for the forecast risk calibration filter diagnostic (issue #2869)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.validation import validate_forecast_risk_calibration_filter as validator

if TYPE_CHECKING:
    from pathlib import Path

# -- Fixtures ----------------------------------------------------------------


def _write_calibration_report(
    tmp_path: Path,
    *,
    eligible: bool = False,
    actor_class_available: bool = False,
    tiers: list[str] | None = None,
) -> Path:
    """Write a synthetic Issue #2865-style calibration report and return its path."""
    if tiers is None:
        tiers = ["deployable_tracked"]
    rows: list[dict[str, object]] = []
    for tier in tiers:
        risk_eligibility = "eligible" if eligible else "blocked_no_denominator"
        actor_class = "pedestrian" if actor_class_available else "unavailable"
        rows.append(
            {
                "scenario_family": "corridor_interaction",
                "predictor_family": "cv",
                "observation_tier": tier,
                "horizon_s": 1.0,
                "actor_class": actor_class,
                "risk_scoring_eligibility": risk_eligibility,
                "calibration_status": "calibrated" if eligible else "unavailable",
            }
        )
    data = {
        "schema_version": "ForecastCalibrationReport.v1",
        "report_id": "synthetic",
        "reliability_rows": rows,
        "limitation_rows": [],
    }
    p = tmp_path / "calibration_report.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# -- Unit tests for availability logic ----------------------------------------


class TestModeAvailability:
    """Tests for _mode_availability classification."""

    def test_no_risk_always_available(self) -> None:
        """no_risk mode is always available."""
        status, reason = validator._mode_availability("no_risk", [])
        assert status == "available"
        assert "baseline" in reason

    def test_raw_risk_always_available(self) -> None:
        """raw_risk mode is always available as a diagnostic."""
        status, reason = validator._mode_availability("raw_risk", [])
        assert status == "available"
        assert "diagnostic" in reason

    def test_calibration_filtered_blocked_without_eligible_rows(self) -> None:
        """calibration_filtered is blocked when no rows are risk-scoring eligible."""
        rows = [{"risk_scoring_eligibility": "blocked_no_denominator"}]
        status, reason = validator._mode_availability("calibration_filtered", rows)
        assert status == "blocked"
        assert "no_rows_risk_scoring_eligible" in reason

    def test_calibration_filtered_available_with_eligible_rows(self) -> None:
        """calibration_filtered is available when at least one row is eligible."""
        rows = [{"risk_scoring_eligibility": "eligible"}]
        status, reason = validator._mode_availability("calibration_filtered", rows)
        assert status == "available"
        assert "eligible" in reason

    def test_actor_class_aware_blocked_when_unavailable(self) -> None:
        """actor_class_aware is blocked when all rows have actor_class unavailable."""
        rows = [{"actor_class": "unavailable"}]
        status, reason = validator._mode_availability("actor_class_aware", rows)
        assert status == "blocked"
        assert "actor_class_unavailable" in reason

    def test_actor_class_aware_available_with_actor_class(self) -> None:
        """actor_class_aware is available when rows carry a real actor class."""
        rows = [{"actor_class": "pedestrian"}]
        status, _reason = validator._mode_availability("actor_class_aware", rows)
        assert status == "available"

    def test_observation_tier_aware_blocked_with_single_tier(self) -> None:
        """observation_tier_aware is blocked with only one observation tier."""
        rows = [{"observation_tier": "deployable_tracked"}]
        status, reason = validator._mode_availability("observation_tier_aware", rows)
        assert status == "blocked"
        assert "single_observation_tier" in reason

    def test_observation_tier_aware_available_with_multiple_tiers(self) -> None:
        """observation_tier_aware is available with multiple observation tiers."""
        rows = [
            {"observation_tier": "deployable_tracked"},
            {"observation_tier": "oracle"},
        ]
        status, _reason = validator._mode_availability("observation_tier_aware", rows)
        assert status == "available"


# -- Report structure tests ---------------------------------------------------


class TestBuildReport:
    """Structural assertions on the diagnostic report."""

    def test_report_schema_version(self, tmp_path: Path) -> None:
        """Report schema version should match the module constant."""
        report_path = _write_calibration_report(tmp_path)
        report = validator.build_report(report_path)
        assert report["schema_version"] == validator.SCHEMA_VERSION

    def test_claim_boundary_is_diagnostic_only(self, tmp_path: Path) -> None:
        """Claim boundary must be diagnostic-only, not benchmark evidence."""
        report_path = _write_calibration_report(tmp_path)
        report = validator.build_report(report_path)
        assert report["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"

    def test_five_modes_present(self, tmp_path: Path) -> None:
        """Report should cover exactly the five required risk modes."""
        report_path = _write_calibration_report(tmp_path)
        report = validator.build_report(report_path)
        modes = [row["mode"] for row in report["modes"]]
        assert modes == list(validator._RISK_MODES)

    def test_current_evidence_blocks_three_modes(self, tmp_path: Path) -> None:
        """Synthetic report mirroring #2865 blocks calibration, actor-class, and tier-aware."""
        report_path = _write_calibration_report(tmp_path)
        report = validator.build_report(report_path)
        blocked = {row["mode"] for row in report["modes"] if row["status"] == "blocked"}
        assert blocked == {
            "calibration_filtered",
            "actor_class_aware",
            "observation_tier_aware",
        }

    def test_recommendation_wait_with_blocked_calibration_filter(self, tmp_path: Path) -> None:
        """When calibration filtering is blocked, recommendation must be wait."""
        report_path = _write_calibration_report(tmp_path)
        report = validator.build_report(report_path)
        assert report["recommendation"] == "wait"


# -- Metric/tradeoff tests ----------------------------------------------------


class TestModeMetrics:
    """Tests for per-mode metric extraction."""

    def test_no_risk_zero_forecast_penalty(self, tmp_path: Path) -> None:
        """no_risk should apply zero forecast penalty on the high-risk case."""
        report_path = _write_calibration_report(tmp_path)
        report = validator.build_report(report_path)
        no_risk_row = next(row for row in report["modes"] if row["mode"] == "no_risk")
        high = no_risk_row["high_risk"]
        assert isinstance(high, dict)
        assert high["forecast_penalty"] == 0.0
        assert high["selected_proposal"] == "goal"

    def test_raw_risk_penalizes_goal(self, tmp_path: Path) -> None:
        """raw_risk should apply a positive forecast penalty on the high-risk case."""
        report_path = _write_calibration_report(tmp_path)
        report = validator.build_report(report_path)
        raw_row = next(row for row in report["modes"] if row["mode"] == "raw_risk")
        high = raw_row["high_risk"]
        assert isinstance(high, dict)
        assert high["forecast_penalty"] > 0.0

    def test_raw_risk_false_positive_suppresses_penalty(self, tmp_path: Path) -> None:
        """raw_risk should suppress penalty when false_positive risk is 1."""
        report_path = _write_calibration_report(tmp_path)
        report = validator.build_report(report_path)
        raw_row = next(row for row in report["modes"] if row["mode"] == "raw_risk")
        fp = raw_row["false_positive"]
        assert isinstance(fp, dict)
        assert fp["forecast_penalty"] == 0.0
        assert fp["selected_proposal"] == "goal"

    def test_blocked_modes_have_null_metrics(self, tmp_path: Path) -> None:
        """Blocked modes should not produce per-case metrics."""
        report_path = _write_calibration_report(tmp_path)
        report = validator.build_report(report_path)
        for row in report["modes"]:
            if row["status"] == "blocked":
                assert row["high_risk"] is None
                assert row["false_positive"] is None


class TestTradeoffs:
    """Tests for cross-mode tradeoff computation."""

    def test_route_progress_tradeoff_positive(self, tmp_path: Path) -> None:
        """raw_risk should slow the high-risk case relative to no_risk."""
        report_path = _write_calibration_report(tmp_path)
        report = validator.build_report(report_path)
        tradeoff = report["tradeoffs"]["route_progress_tradeoff"]
        assert tradeoff is not None
        assert tradeoff > 0.0

    def test_false_positive_stopping_avoided(self, tmp_path: Path) -> None:
        """False-positive suppression should avoid unnecessary stopping."""
        report_path = _write_calibration_report(tmp_path)
        report = validator.build_report(report_path)
        assert report["tradeoffs"]["false_positive_stopping_avoided"] is True

    def test_false_positive_unnecessary_slowdown_zero(self, tmp_path: Path) -> None:
        """False-positive case should not slow relative to no_risk."""
        report_path = _write_calibration_report(tmp_path)
        report = validator.build_report(report_path)
        assert report["tradeoffs"]["false_positive_unnecessary_slowdown_count"] == 0


# -- CLI tests ----------------------------------------------------------------


class TestMainCLI:
    """Tests for the main() CLI entry point."""

    def test_main_returns_zero(self, tmp_path: Path) -> None:
        """CLI should exit with code 0 on success."""
        report_path = _write_calibration_report(tmp_path)
        assert validator.main(["--calibration-report", str(report_path)]) == 0

    def test_main_json_output(self, tmp_path: Path, capsys) -> None:
        """CLI should print JSON with summary and report keys."""
        report_path = _write_calibration_report(tmp_path)
        validator.main(["--calibration-report", str(report_path)])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "summary" in data
        assert "report" in data
        assert data["summary"]["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"

    def test_main_writes_artifacts(self, tmp_path: Path) -> None:
        """--out-dir should produce report.json, report.md, summary.json, and README.md."""
        report_path = _write_calibration_report(tmp_path)
        out = tmp_path / "out"
        exit_code = validator.main(
            ["--calibration-report", str(report_path), "--out-dir", str(out)]
        )
        assert exit_code == 0
        assert (out / "report.json").exists()
        assert (out / "report.md").exists()
        assert (out / "summary.json").exists()
        assert (out / "README.md").exists()

    def test_main_summary_json_content(self, tmp_path: Path) -> None:
        """summary.json should carry the diagnostic claim boundary and recommendation."""
        report_path = _write_calibration_report(tmp_path)
        out = tmp_path / "out"
        validator.main(["--calibration-report", str(report_path), "--out-dir", str(out)])
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert summary["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"
        assert summary["recommendation"] == "wait"
        assert summary["blocked_mode_count"] == 3

    def test_main_missing_report_returns_1(self, tmp_path: Path) -> None:
        """Missing calibration report returns exit code 1."""
        rc = validator.main(["--calibration-report", str(tmp_path / "missing.json")])
        assert rc == 1

    def test_main_malformed_report_returns_1(self, tmp_path: Path) -> None:
        """Malformed calibration report returns exit code 1."""
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"no_reliability_rows": []}), encoding="utf-8")
        rc = validator.main(["--calibration-report", str(p)])
        assert rc == 1
