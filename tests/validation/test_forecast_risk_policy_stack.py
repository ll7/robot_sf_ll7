"""Tests for the forecast risk policy stack diagnostic comparison validator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.validation import validate_forecast_risk_policy_stack as validator

if TYPE_CHECKING:
    from pathlib import Path


class TestBuildReport:
    """Structural assertions on the diagnostic report."""

    def test_report_schema_version(self) -> None:
        """Report schema version should match the module constant."""
        report = validator.build_report()
        assert report["schema_version"] == validator.SCHEMA_VERSION

    def test_claim_boundary_is_diagnostic_only(self) -> None:
        """Claim boundary must be diagnostic-only, not benchmark evidence."""
        report = validator.build_report()
        assert report["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"

    def test_diagnostic_weight_matches_constant(self) -> None:
        """Diagnostic weight should match the module constant."""
        report = validator.build_report()
        assert report["diagnostic_weight"] == validator._DIAGNOSTIC_WEIGHT

    def test_case_count(self) -> None:
        """Report should contain exactly two diagnostic cases."""
        report = validator.build_report()
        assert len(report["cases"]) == 2

    def test_case_names(self) -> None:
        """Both expected case names should be present."""
        report = validator.build_report()
        names = [c["case_name"] for c in report["cases"]]
        assert "high_risk_diagnostic_slows_goal" in names
        assert "false_positive_suppresses_penalty" in names


class TestHighRiskCase:
    """Verify the high-risk case diagnostic behavior."""

    def _get_case(self) -> dict:
        """Return the high_risk_diagnostic_slows_goal case."""
        report = validator.build_report()
        return next(
            c for c in report["cases"] if c["case_name"] == "high_risk_diagnostic_slows_goal"
        )

    def test_baseline_selects_goal(self) -> None:
        """Baseline should select goal without forecast risk penalty."""
        case = self._get_case()
        assert case["baseline"]["selected_proposal"] == "goal"

    def test_diagnostic_penalizes_goal(self) -> None:
        """Diagnostic scoring should apply a positive forecast penalty."""
        case = self._get_case()
        assert case["diagnostic"]["forecast_penalty"] > 0.0

    def test_diagnostic_reduces_progress_proxy(self) -> None:
        """Diagnostic progress proxy should not exceed baseline."""
        case = self._get_case()
        assert case["diagnostic"]["progress_proxy"] < case["baseline"]["progress_proxy"]
        assert case["delta_progress_proxy"] == -0.8

    def test_delta_forecast_penalty_positive(self) -> None:
        """Delta forecast penalty should be positive for the high-risk case."""
        case = self._get_case()
        assert case["delta_forecast_penalty"] > 0.0


class TestFalsePositiveCase:
    """Verify the false-positive suppression case."""

    def _get_case(self) -> dict:
        """Return the false_positive_suppresses_penalty case."""
        report = validator.build_report()
        return next(
            c for c in report["cases"] if c["case_name"] == "false_positive_suppresses_penalty"
        )

    def test_baseline_selects_goal(self) -> None:
        """Baseline should select goal without forecast risk."""
        case = self._get_case()
        assert case["baseline"]["selected_proposal"] == "goal"

    def test_diagnostic_suppresses_penalty(self) -> None:
        """False positive risk=1 should suppress the forecast penalty."""
        case = self._get_case()
        assert case["diagnostic"]["forecast_penalty"] == 0.0

    def test_diagnostic_still_selects_goal(self) -> None:
        """With penalty suppressed, goal should remain the selected proposal."""
        case = self._get_case()
        assert case["diagnostic"]["selected_proposal"] == "goal"

    def test_no_false_positive_unnecessary_slowdown(self) -> None:
        """False-positive suppression should avoid unnecessary slowing."""
        case = self._get_case()
        assert case["false_positive_unnecessary_slowdown_count"] == 0
        assert case["diagnostic"]["shield_stop_count"] == 0

    def test_delta_forecast_penalty_zero(self) -> None:
        """Delta should be zero when false positive suppresses the penalty."""
        case = self._get_case()
        assert case["delta_forecast_penalty"] == 0.0


class TestRecommendation:
    """Verify the recommendation field."""

    def test_recommendation_is_deterministic(self) -> None:
        """Two builds should produce the same recommendation."""
        r1 = validator.build_report()["recommendation"]
        r2 = validator.build_report()["recommendation"]
        assert r1 == r2

    def test_recommendation_valid_label(self) -> None:
        """Recommendation must be one of the known diagnostic labels."""
        valid = {
            "forecast_risk_scoring_diagnostic_consistent",
            "false_positive_suppression_inconsistent",
            "high_risk_penalization_inconsistent",
            "no_diagnostic_signal",
            "incomplete_cases",
        }
        report = validator.build_report()
        assert report["recommendation"] in valid


class TestMainCLI:
    """Verify the main() CLI entry point."""

    def test_main_returns_zero(self) -> None:
        """CLI should exit with code 0 on success."""
        assert validator.main([]) == 0

    def test_main_json_output(self, capsys) -> None:
        """CLI should print JSON with summary and report keys."""
        validator.main([])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "summary" in data
        assert "report" in data
        assert data["summary"]["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"

    def test_main_writes_artifacts(self, tmp_path: Path) -> None:
        """--out-dir should produce report.json, report.md, and summary.json."""
        exit_code = validator.main(["--out-dir", str(tmp_path)])
        assert exit_code == 0
        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "report.md").exists()
        assert (tmp_path / "summary.json").exists()

    def test_main_summary_json_content(self, tmp_path: Path) -> None:
        """summary.json should carry the diagnostic claim boundary."""
        validator.main(["--out-dir", str(tmp_path)])
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert summary["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"
        assert summary["false_positive_unnecessary_slowdown_count"] == 0
        assert summary["recommendation"] in {
            "forecast_risk_scoring_diagnostic_consistent",
            "false_positive_suppression_inconsistent",
            "high_risk_penalization_inconsistent",
            "no_diagnostic_signal",
            "incomplete_cases",
        }

    def test_main_report_md_contains_claim_boundary(self, tmp_path: Path) -> None:
        """report.md should state the diagnostic-only claim boundary."""
        validator.main(["--out-dir", str(tmp_path)])
        md = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "diagnostic_only_not_benchmark_evidence" in md
        assert "No safety claim" in md
