"""Tests for ForecastCalibrationReport.v1 artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.forecast_calibration_report import (
    FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION,
    build_forecast_calibration_report,
    format_forecast_calibration_markdown,
)


def _metric_report(
    *,
    coverage: float | None = 0.9,
    status: str = "ok",
    expected_ade: float | None = 0.25,
    expected_status: str = "ok",
    minade: float | None = None,
    denominator: int = 4,
) -> dict[str, object]:
    def row(metric: str, value: float | None, row_status: str = status) -> dict[str, object]:
        return {
            "metric": metric,
            "horizon_s": 1.0,
            "value": value,
            "status": row_status,
            "denominator": denominator if row_status == "ok" else 0,
            "actor_class": "pedestrian",
            "scenario_id": "classic_crossing_low",
            "observation_tier": "deployable_observation",
            "dt_s": 0.5,
        }

    return {
        "schema_version": "ForecastMetrics.v1",
        "provenance": {
            "predictor_id": "probabilistic-baseline",
            "predictor_family": "probabilistic_cv",
            "scenario_id": "classic_crossing_low",
            "scenario_family": "classic_crossing",
            "observation_tier": "deployable_observation",
            "dt_s": 0.5,
            "horizons_s": [1.0],
        },
        "aggregate_rows": [
            row("mean_coverage", coverage),
            row("mean_likelihood", -1.2),
            row("mean_expected_ade", expected_ade, expected_status),
            row("mean_minade@k", minade, "ok" if minade is not None else "unavailable"),
        ],
    }


def test_calibration_report_classifies_available_calibrated_rows() -> None:
    """Available coverage near the target should allow continue recommendation."""
    report = build_forecast_calibration_report(
        [_metric_report(coverage=0.91)],
        report_id="calibrated",
        coverage_target=0.9,
        coverage_tolerance=0.05,
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    assert report["schema_version"] == FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION
    assert report["limitation_rows"] == []
    assert report["reliability_rows"][0]["calibration_status"] == "calibrated_within_tolerance"
    assert report["recommendation"]["decision"] == "continue"


def test_calibration_report_distinguishes_over_and_under_confidence() -> None:
    """Coverage gaps should classify over-confidence and under-confidence separately."""
    overconfident = build_forecast_calibration_report(
        [_metric_report(coverage=0.7)],
        report_id="overconfident",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )
    underconfident = build_forecast_calibration_report(
        [_metric_report(coverage=0.99)],
        report_id="underconfident",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    assert overconfident["reliability_rows"][0]["calibration_status"] == (
        "over_confident_under_coverage"
    )
    assert underconfident["reliability_rows"][0]["calibration_status"] == (
        "under_confident_over_coverage"
    )
    assert overconfident["recommendation"]["decision"] == "revise"


def test_calibration_report_aggregates_duplicate_family_rows() -> None:
    """Multiple reports in one scenario-family group should aggregate instead of overwrite."""
    report = build_forecast_calibration_report(
        [
            _metric_report(coverage=0.8, denominator=1),
            _metric_report(coverage=1.0, denominator=3),
        ],
        report_id="family-aggregate",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    row = report["reliability_rows"][0]
    assert row["empirical_coverage"] == 0.95
    assert row["denominator"] == 4


def test_calibration_report_falls_back_to_minade_when_expected_ade_unavailable() -> None:
    """Sharpness should use minADE@K when expected ADE exists but is unavailable."""
    report = build_forecast_calibration_report(
        [
            _metric_report(
                expected_ade=None,
                expected_status="unavailable",
                minade=0.35,
            )
        ],
        report_id="sharpness-fallback",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    assert report["reliability_rows"][0]["sharpness_proxy"] == 0.35


def test_calibration_report_marks_deterministic_uncertainty_unavailable() -> None:
    """Unavailable uncertainty metrics should remain explicit limitation rows."""
    report = _metric_report(
        coverage=None,
        status="unavailable",
        expected_ade=None,
        expected_status="unavailable",
    )
    calibration = build_forecast_calibration_report(
        [report],
        report_id="deterministic",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    row = calibration["reliability_rows"][0]
    assert row["calibration_status"] == "unavailable"
    assert {"coverage", "likelihood", "sharpness_proxy"} <= set(row["unavailable_metrics"])
    assert calibration["limitation_rows"]
    assert calibration["recommendation"]["decision"] == "wait"


def test_calibration_markdown_includes_limitations() -> None:
    """Markdown should carry calibration status and limitation caveats."""
    report = build_forecast_calibration_report(
        [_metric_report(coverage=None, status="unavailable")],
        report_id="markdown",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    markdown = format_forecast_calibration_markdown(report)

    assert "# Forecast Calibration Report" in markdown
    assert "Claim status: diagnostic-only" in markdown
    assert "## Limitations" in markdown


def test_calibration_report_rejects_malformed_rows() -> None:
    """Malformed metric rows should fail before grouping."""
    report = _metric_report()
    report["aggregate_rows"] = ["bad-row"]

    with pytest.raises(ValueError, match=r"aggregate_rows\[0\]"):
        build_forecast_calibration_report([report], report_id="bad")


def test_calibration_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    """CLI should write reviewable JSON and Markdown artifacts."""
    metric_path = tmp_path / "metrics.json"
    out_json = tmp_path / "calibration.json"
    out_md = tmp_path / "calibration.md"
    metric_path.write_text(json.dumps(_metric_report(coverage=0.91)), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[2]
                / "scripts/benchmark/build_forecast_calibration_report.py"
            ),
            str(metric_path),
            "--report-id",
            "cli-calibration",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)

    assert summary["decision"] == "continue"
    assert out_json.exists()
    assert out_md.exists()
    assert json.loads(out_json.read_text(encoding="utf-8"))["report_id"] == "cli-calibration"


def test_calibration_cli_reports_malformed_json(tmp_path: Path) -> None:
    """CLI should return a concise parser error for malformed input JSON."""
    metric_path = tmp_path / "metrics.json"
    out_json = tmp_path / "calibration.json"
    metric_path.write_text("{oops", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[2]
                / "scripts/benchmark/build_forecast_calibration_report.py"
            ),
            str(metric_path),
            "--report-id",
            "bad-json",
            "--out-json",
            str(out_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "could not parse metric report" in result.stderr
