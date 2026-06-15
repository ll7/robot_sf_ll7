"""Tests for the closed-loop forecast coupling gate validation (issue #2843)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validation.validate_closed_loop_forecast_coupling_gate import (
    _decide_recommendation,
    _parse_gate_metrics_from_readme,
    _summarize_interaction_effect,
    build_gate_report,
    main,
)

# -- Fixtures ----------------------------------------------------------------


def _write_forecast(tmp_path: Path, interaction_effect: dict, rows: list | None = None) -> Path:
    """Write a synthetic forecast comparison JSON and return its path."""
    if rows is None:
        rows = [
            {
                "baseline": "cv",
                "family": "corridor_interaction",
                "label": "default_social_force",
                "status": "evaluated",
                "evaluable_samples": 30.0,
                "mean_ade_0.5s": 0.024,
                "mean_ade_1s": 0.077,
                "mean_negative_log_likelihood_1s": 1.952,
                "mean_miss_rate_1s": 0.0,
                "mean_within_95ci_1s": 1.0,
            },
            {
                "baseline": "interaction_aware",
                "family": "corridor_interaction",
                "label": "default_social_force",
                "status": "evaluated",
                "evaluable_samples": 30.0,
                "mean_ade_0.5s": 0.045,
                "mean_ade_1s": 0.114,
                "mean_negative_log_likelihood_1s": 1.350,
                "mean_miss_rate_1s": 0.0,
                "mean_within_95ci_1s": 1.0,
            },
        ]
    data = {"issue": 2781, "comparison_rows": rows, "interaction_effect": interaction_effect}
    p = tmp_path / "comparison_report.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_gate_readme(
    tmp_path: Path,
    status: str = "failed",
    reason: str = "global_success_delta_below_gate",
    global_delta: float = 0.0,
    hard_delta: float = 0.0,
    distance_delta: float = 0.0108,
) -> Path:
    """Write a synthetic gate README and return its path."""
    content = f"""\
# Issue #1897 Predictive Coupling Gate Evidence

Closed-loop gate:

- status: `{status}`
- reason: `{reason}`
- global success delta: `{global_delta:.4f}`
- hard success delta: `{hard_delta:.4f}`
- global mean-min-distance delta: `{distance_delta:.4f}`
"""
    p = tmp_path / "gate_readme.md"
    p.write_text(content, encoding="utf-8")
    return p


# -- Unit tests for gate logic ------------------------------------------------


class TestDecideRecommendation:
    """Tests for the _decide_recommendation decision function."""

    def test_gate_failed_returns_revise(self):
        """Gate failure always returns revise regardless of forecast signal."""
        effect = {"mean_ade_1s_delta_vs_cv": -0.01, "mean_nll_1s_delta_vs_cv": -0.5}
        gate = {"status": "failed", "reason": "x"}
        assert _decide_recommendation(effect, gate) == "revise"

    def test_gate_passed_positive_forecast_returns_continue(self):
        """Both gate and forecast positive yields continue."""
        effect = {"mean_ade_1s_delta_vs_cv": -0.01, "mean_nll_1s_delta_vs_cv": -0.5}
        gate = {"status": "passed"}
        assert _decide_recommendation(effect, gate) == "continue"

    def test_gate_passed_mixed_forecast_returns_stop(self):
        """Gate passed but mixed forecast (ADE worsened) yields stop."""
        effect = {"mean_ade_1s_delta_vs_cv": 0.02, "mean_nll_1s_delta_vs_cv": -0.5}
        gate = {"status": "passed"}
        assert _decide_recommendation(effect, gate) == "stop"

    def test_gate_passed_worsened_both_returns_stop(self):
        """Gate passed but both metrics worsened yields stop."""
        effect = {"mean_ade_1s_delta_vs_cv": 0.01, "mean_nll_1s_delta_vs_cv": 0.1}
        gate = {"status": "passed"}
        assert _decide_recommendation(effect, gate) == "stop"

    def test_missing_ade_returns_stop(self):
        """None ADE delta with gate passed yields stop."""
        effect = {"mean_ade_1s_delta_vs_cv": None, "mean_nll_1s_delta_vs_cv": -0.5}
        gate = {"status": "passed"}
        assert _decide_recommendation(effect, gate) == "stop"


# -- Unit tests for metric parsing --------------------------------------------


class TestParseGateMetrics:
    """Tests for README metric extraction."""

    def test_parses_all_fields(self, tmp_path: Path):
        """All five gate metrics are extracted correctly."""
        p = _write_gate_readme(
            tmp_path,
            status="passed",
            reason="ok",
            global_delta=0.03,
            hard_delta=0.01,
            distance_delta=-0.005,
        )
        m = _parse_gate_metrics_from_readme(p)
        assert m["status"] == "passed"
        assert m["reason"] == "ok"
        assert m["global_success_delta"] == pytest.approx(0.03)
        assert m["hard_success_delta"] == pytest.approx(0.01)
        assert m["global_min_distance_delta"] == pytest.approx(-0.005)

    def test_missing_field_raises(self, tmp_path: Path):
        """Incomplete README raises ValueError for missing fields."""
        p = tmp_path / "incomplete.md"
        p.write_text("- status: `failed`\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing fields"):
            _parse_gate_metrics_from_readme(p)

    def test_mixed_case_status_normalized(self, tmp_path: Path):
        """Status values like 'Passed', 'PASSED', 'Failed' are normalized to lowercase."""
        for raw_status in ("Passed", "PASSED", "failed", "FAILED"):
            p = _write_gate_readme(tmp_path, status=raw_status, reason="ok")
            m = _parse_gate_metrics_from_readme(p)
            assert m["status"] == raw_status.lower()

    def test_backtick_wrapped_values(self, tmp_path: Path):
        """Values wrapped in backticks are parsed correctly."""
        content = (
            "- status: `Passed`\n"
            "- reason: `ok`\n"
            "- global success delta: `0.05`\n"
            "- hard success delta: `0.02`\n"
            "- global mean-min-distance delta: `-0.005`\n"
        )
        p = tmp_path / "readme.md"
        p.write_text(content, encoding="utf-8")
        m = _parse_gate_metrics_from_readme(p)
        assert m["status"] == "passed"
        assert m["reason"] == "ok"
        assert m["global_success_delta"] == pytest.approx(0.05)
        assert m["hard_success_delta"] == pytest.approx(0.02)
        assert m["global_min_distance_delta"] == pytest.approx(-0.005)

    def test_zero_global_delta_not_lost(self, tmp_path: Path):
        """Zero-valued deltas are preserved, not replaced by a default."""
        p = _write_gate_readme(tmp_path, global_delta=0.0, hard_delta=0.0, distance_delta=0.0)
        m = _parse_gate_metrics_from_readme(p)
        assert m["global_success_delta"] == pytest.approx(0.0)
        assert m["hard_success_delta"] == pytest.approx(0.0)
        assert m["global_min_distance_delta"] == pytest.approx(0.0)


class TestSummarizeInteractionEffect:
    """Tests for interaction effect summarization."""

    def test_basic_summary(self):
        """Evaluable rows counted, deltas and conclusion preserved."""
        forecast = {
            "comparison_rows": [
                {"status": "evaluated"},
                {"status": "evaluated"},
                {"status": "limited_no_pedestrian_motion"},
            ],
            "interaction_effect": {
                "matched_rows": 2,
                "mean_ade_1s_delta_vs_cv": 0.025,
                "mean_nll_1s_delta_vs_cv": -0.67,
                "conclusion": "mixed",
            },
        }
        s = _summarize_interaction_effect(forecast)
        assert s["matched_rows"] == 2
        assert s["evaluable_row_count"] == 2
        assert s["mean_ade_1s_delta_vs_cv"] == 0.025
        assert s["conclusion"] == "mixed"

    def test_interaction_effect_none_raises(self):
        """Null interaction_effect fails closed with a clear validation error."""
        forecast = {"comparison_rows": [], "interaction_effect": None}
        with pytest.raises(ValueError, match="interaction_effect must be an object"):
            _summarize_interaction_effect(forecast)


# -- Integration tests --------------------------------------------------------


class TestBuildGateReport:
    """Tests for full report construction."""

    def test_current_evidence_returns_revise(self, tmp_path: Path):
        """Current issue #2781 + #1897 evidence yields revise recommendation."""
        forecast = _write_forecast(
            tmp_path,
            interaction_effect={
                "matched_rows": 3,
                "mean_ade_1s_delta_vs_cv": 0.0246,
                "mean_nll_1s_delta_vs_cv": -0.6717,
                "conclusion": "revise before closed-loop coupling claims.",
            },
        )
        gate = _write_gate_readme(
            tmp_path,
            status="failed",
            reason="global_success_delta_below_gate",
            global_delta=0.0,
            hard_delta=0.0,
            distance_delta=0.0108,
        )
        report = build_gate_report(forecast, gate)
        assert report["recommendation"] == "revise"
        assert report["issue"] == 2843
        assert "Diagnostic-only" in report["claim_boundary"]

    def test_both_positive_returns_continue(self, tmp_path: Path):
        """Positive forecast + passed gate yields continue."""
        forecast = _write_forecast(
            tmp_path,
            interaction_effect={
                "matched_rows": 3,
                "mean_ade_1s_delta_vs_cv": -0.01,
                "mean_nll_1s_delta_vs_cv": -0.5,
                "conclusion": "positive",
            },
        )
        gate = _write_gate_readme(
            tmp_path,
            status="passed",
            reason="ok",
            global_delta=0.05,
            hard_delta=0.02,
            distance_delta=-0.005,
        )
        report = build_gate_report(forecast, gate)
        assert report["recommendation"] == "continue"


class TestMainCLI:
    """Tests for the CLI entry point."""

    def test_missing_forecast_returns_1(self, tmp_path: Path):
        """Missing forecast file returns exit code 1."""
        gate = _write_gate_readme(tmp_path)
        rc = main(
            ["--forecast-comparison", str(tmp_path / "missing.json"), "--gate-readme", str(gate)]
        )
        assert rc == 1

    def test_missing_gate_returns_1(self, tmp_path: Path):
        """Missing gate readme returns exit code 1."""
        forecast = _write_forecast(
            tmp_path,
            interaction_effect={
                "matched_rows": 1,
                "mean_ade_1s_delta_vs_cv": 0.0,
                "mean_nll_1s_delta_vs_cv": 0.0,
                "conclusion": "x",
            },
        )
        rc = main(
            ["--forecast-comparison", str(forecast), "--gate-readme", str(tmp_path / "missing.md")]
        )
        assert rc == 1

    def test_writes_output_files(self, tmp_path: Path):
        """Output dir receives gate_report.json and gate_report.md."""
        forecast = _write_forecast(
            tmp_path,
            interaction_effect={
                "matched_rows": 1,
                "mean_ade_1s_delta_vs_cv": 0.0,
                "mean_nll_1s_delta_vs_cv": 0.0,
                "conclusion": "x",
            },
        )
        gate = _write_gate_readme(tmp_path)
        out = tmp_path / "out"
        rc = main(
            [
                "--forecast-comparison",
                str(forecast),
                "--gate-readme",
                str(gate),
                "--output-dir",
                str(out),
            ]
        )
        assert rc == 0
        assert (out / "gate_report.json").exists()
        assert (out / "gate_report.md").exists()
        report = json.loads((out / "gate_report.json").read_text(encoding="utf-8"))
        assert report["recommendation"] == "revise"

    def test_writes_markdown_with_missing_metric(self, tmp_path: Path):
        """Missing optional metric values render as NA instead of failing."""
        forecast = _write_forecast(
            tmp_path,
            interaction_effect={
                "matched_rows": 1,
                "mean_ade_1s_delta_vs_cv": None,
                "mean_nll_1s_delta_vs_cv": -0.5,
                "conclusion": "missing ADE",
            },
        )
        gate = _write_gate_readme(
            tmp_path,
            status="passed",
            reason="ok",
            global_delta=0.05,
            hard_delta=0.02,
            distance_delta=0.0,
        )
        out = tmp_path / "out"
        rc = main(
            [
                "--forecast-comparison",
                str(forecast),
                "--gate-readme",
                str(gate),
                "--output-dir",
                str(out),
            ]
        )
        assert rc == 2
        assert "| Mean ADE 1s delta vs CV (m) | NA |" in (out / "gate_report.md").read_text(
            encoding="utf-8"
        )

    def test_stop_returns_exit_code_2(self, tmp_path: Path):
        """Stop recommendation returns exit code 2."""
        forecast = _write_forecast(
            tmp_path,
            interaction_effect={
                "matched_rows": 1,
                "mean_ade_1s_delta_vs_cv": 0.01,
                "mean_nll_1s_delta_vs_cv": 0.1,
                "conclusion": "bad",
            },
        )
        gate = _write_gate_readme(
            tmp_path,
            status="passed",
            reason="ok",
            global_delta=0.05,
            hard_delta=0.02,
            distance_delta=0.0,
        )
        rc = main(["--forecast-comparison", str(forecast), "--gate-readme", str(gate)])
        assert rc == 2


class TestIOErrorPath:
    """Regression tests for OSError handling in main."""

    def test_read_oserror_returns_1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """OSError during file read returns exit code 1 with JSON error."""
        forecast = _write_forecast(
            tmp_path,
            interaction_effect={
                "matched_rows": 1,
                "mean_ade_1s_delta_vs_cv": 0.0,
                "mean_nll_1s_delta_vs_cv": 0.0,
                "conclusion": "x",
            },
        )
        gate = _write_gate_readme(tmp_path)

        original_read_text = Path.read_text

        def _raise_oserror(self: Path, **kwargs: object) -> str:
            raise OSError("simulated read failure")

        monkeypatch.setattr(Path, "read_text", _raise_oserror)
        try:
            rc = main(["--forecast-comparison", str(forecast), "--gate-readme", str(gate)])
        finally:
            monkeypatch.setattr(Path, "read_text", original_read_text)
        assert rc == 1

    def test_write_oserror_returns_1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """OSError during output write returns exit code 1 with JSON error."""
        forecast = _write_forecast(
            tmp_path,
            interaction_effect={
                "matched_rows": 1,
                "mean_ade_1s_delta_vs_cv": 0.0,
                "mean_nll_1s_delta_vs_cv": 0.0,
                "conclusion": "x",
            },
        )
        gate = _write_gate_readme(tmp_path)
        out = tmp_path / "readonly_dir"

        original_write_text = Path.write_text

        def _raise_oserror(self: Path, *args: object, **kwargs: object) -> None:
            raise OSError("simulated write failure")

        monkeypatch.setattr(Path, "write_text", _raise_oserror)
        try:
            rc = main(
                [
                    "--forecast-comparison",
                    str(forecast),
                    "--gate-readme",
                    str(gate),
                    "--output-dir",
                    str(out),
                ]
            )
        finally:
            monkeypatch.setattr(Path, "write_text", original_write_text)
        assert rc == 1
