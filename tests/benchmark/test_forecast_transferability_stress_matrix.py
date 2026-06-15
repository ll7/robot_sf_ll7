"""Tests for ForecastTransferabilityStressMatrix.v1 reports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.forecast_transferability_stress_matrix import (
    FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION,
    build_forecast_transferability_stress_matrix,
    format_forecast_transferability_stress_markdown,
)


def _metric_report(
    *,
    observation_tier: str = "deployable_observation",
    transfer_dimensions: dict[str, object] | None = None,
    actor_class: str = "pedestrian",
    metric_status: str = "ok",
    denominator: int = 3,
    value: float | None = 0.42,
) -> dict[str, object]:
    return {
        "schema_version": "ForecastMetrics.v1",
        "provenance": {
            "predictor_id": "cv-baseline",
            "predictor_family": "constant_velocity",
            "scenario_id": "classic_crossing_low",
            "scenario_family": "classic_crossing",
            "observation_tier": observation_tier,
            "dt_s": 0.5,
            "horizons_s": [1.0],
        },
        "transfer_dimensions": transfer_dimensions or {},
        "aggregate_rows": [
            {
                "metric": "mean_ade",
                "horizon_s": 1.0,
                "value": value,
                "status": metric_status,
                "denominator": denominator,
                "actor_class": actor_class,
                "scenario_id": "classic_crossing_low",
                "observation_tier": observation_tier,
                "dt_s": 0.5,
            }
        ],
    }


def _complete_transfer_dimensions() -> dict[str, object]:
    return {
        "observation_noise": "none",
        "latency": "0_steps",
        "dropout": "none",
        "occlusion": "clear",
        "map_family": "classic_crossing",
        "density": "low",
        "pedestrian_model_family": "social_force",
    }


def test_transferability_matrix_can_be_benchmark_eligible() -> None:
    """Complete deployable rows should be classified as benchmark-eligible."""
    report = build_forecast_transferability_stress_matrix(
        [_metric_report(transfer_dimensions=_complete_transfer_dimensions())],
        report_id="complete-transfer",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    assert report["schema_version"] == FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION
    assert report["limitation_rows"] == []
    assert report["matrix_rows"][0]["evidence_status"] == "benchmark-eligible"
    assert report["recommendation"]["decision"] == "continue"
    assert report["recommendation"]["claim_status"] == "benchmark-eligible"
    assert report["dimension_coverage"]["latency"]["observed_values"] == ["0_steps"]


def test_transferability_matrix_reports_unavailable_dimensions() -> None:
    """Missing transfer axes should become explicit limitation rows."""
    report = build_forecast_transferability_stress_matrix(
        [_metric_report(observation_tier="oracle_full_state")],
        report_id="missing-transfer",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    missing_dimensions = {row["dimension"] for row in report["limitation_rows"]}
    assert "latency" in missing_dimensions
    assert "dropout" in missing_dimensions
    assert report["matrix_rows"][0]["evidence_status"] == "diagnostic-only"
    assert report["recommendation"]["decision"] == "revise"
    assert report["recommendation"]["claim_status"] == "diagnostic-only"


def test_transferability_matrix_treats_mixed_case_oracle_as_diagnostic() -> None:
    """Oracle-tier checks should be case-insensitive."""
    report = build_forecast_transferability_stress_matrix(
        [
            _metric_report(
                observation_tier="Oracle_Full_State",
                transfer_dimensions=_complete_transfer_dimensions(),
            )
        ],
        report_id="mixed-case-oracle",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    assert report["matrix_rows"][0]["evidence_status"] == "oracle-only"
    assert report["recommendation"]["claim_status"] == "diagnostic-only"


def test_transferability_matrix_rejects_malformed_aggregate_rows() -> None:
    """Malformed metric reports should fail at the schema boundary."""
    report = _metric_report(transfer_dimensions=_complete_transfer_dimensions())
    report["aggregate_rows"] = ["not-a-row"]

    with pytest.raises(ValueError, match=r"aggregate_rows\[0\]"):
        build_forecast_transferability_stress_matrix(
            [report],
            report_id="malformed",
        )


def test_transferability_matrix_rejects_missing_required_aggregate_fields() -> None:
    """Required aggregate fields should not crash later during row construction."""
    report = _metric_report(transfer_dimensions=_complete_transfer_dimensions())
    del report["aggregate_rows"][0]["denominator"]

    with pytest.raises(ValueError, match="denominator"):
        build_forecast_transferability_stress_matrix(
            [report],
            report_id="missing-denominator",
        )


def test_transferability_markdown_includes_limitations() -> None:
    """Markdown should carry the recommendation and unavailable-row caveats."""
    report = build_forecast_transferability_stress_matrix(
        [_metric_report()],
        report_id="markdown-transfer",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    markdown = format_forecast_transferability_stress_markdown(report)

    assert "# Forecast Transferability Stress Matrix" in markdown
    assert "Claim status: diagnostic-only" in markdown
    assert "## Limitations" in markdown
    assert "observation_noise" in markdown


def test_transferability_cli_writes_json_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should write reviewable JSON and Markdown artifacts."""
    metric_path = tmp_path / "metrics.json"
    out_json = tmp_path / "transfer.json"
    out_md = tmp_path / "transfer.md"
    metric_path.write_text(
        json.dumps(_metric_report(transfer_dimensions=_complete_transfer_dimensions())),
        encoding="utf-8",
    )

    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/build_forecast_transferability_matrix.py",
            str(metric_path),
            "--report-id",
            "cli-transfer",
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
    assert json.loads(out_json.read_text(encoding="utf-8"))["report_id"] == "cli-transfer"


def test_transferability_cli_reports_malformed_json(tmp_path: Path) -> None:
    """CLI should return a concise parser error for malformed input JSON."""
    metric_path = tmp_path / "metrics.json"
    out_json = tmp_path / "transfer.json"
    metric_path.write_text("{oops", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[2]
                / "scripts/benchmark/build_forecast_transferability_matrix.py"
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
