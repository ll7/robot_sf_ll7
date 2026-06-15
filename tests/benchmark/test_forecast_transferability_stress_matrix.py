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


def _metric_report(  # noqa: PLR0913
    *,
    observation_tier: str = "deployable_observation",
    transfer_dimensions: dict[str, object] | None = None,
    horizon_s: float | list[float] = 1.0,
    actor_class: str | None = "pedestrian",
    aggregate_row_fields: dict[str, object] | None = None,
    provenance_fields: dict[str, object] | None = None,
    metric_status: str = "ok",
    denominator: int = 3,
    value: float | None = 0.42,
    report_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    horizons = [horizon_s] if isinstance(horizon_s, float) else list(horizon_s)
    aggregate_rows = [
        {
            "metric": "mean_ade",
            "horizon_s": horizon,
            "value": value,
            "status": metric_status,
            "denominator": denominator,
            "actor_class": actor_class,
            "scenario_id": "classic_crossing_low",
            "observation_tier": observation_tier,
            "dt_s": 0.5,
        }
        for horizon in horizons
    ]
    if aggregate_row_fields:
        for row in aggregate_rows:
            row.update(aggregate_row_fields)
    provenance = {
        "predictor_id": "cv-baseline",
        "predictor_family": "constant_velocity",
        "scenario_id": "classic_crossing_low",
        "scenario_family": "classic_crossing",
        "observation_tier": observation_tier,
        "dt_s": 0.5,
        "horizons_s": [0.5, 1.0, 2.0],
    }
    if provenance_fields:
        provenance.update(provenance_fields)
    report: dict[str, object] = {
        "schema_version": "ForecastMetrics.v1",
        "provenance": provenance,
        "transfer_dimensions": transfer_dimensions or {},
        "aggregate_rows": aggregate_rows,
    }
    if report_fields:
        report.update(report_fields)
    return report


def _complete_transfer_dimensions() -> dict[str, object]:
    return {
        "observation_noise": "none",
        "latency": "0_steps",
        "dropout": "none",
        "occlusion": "clear",
        "map_family": "classic_crossing",
        "density": "low",
        "pedestrian_model_family": "social_force",
        "semantic_metadata_present": "present",
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


def test_transferability_matrix_reports_blocked_dimensions() -> None:
    """Missing transfer axes should become explicit limitation rows."""
    report = build_forecast_transferability_stress_matrix(
        [_metric_report(observation_tier="deployable_vision")],
        report_id="missing-transfer",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    missing_dimensions = {row["dimension"] for row in report["limitation_rows"]}
    assert "latency" in missing_dimensions
    assert "dropout" in missing_dimensions
    assert report["matrix_rows"][0]["evidence_status"] == "blocked"
    assert report["recommendation"]["decision"] == "stop"
    assert report["recommendation"]["claim_status"] == "blocked"


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

    assert report["matrix_rows"][0]["evidence_status"] == "diagnostic-only"
    assert report["recommendation"]["claim_status"] == "diagnostic-only"


def test_transferability_matrix_tracks_actor_type_and_horizons() -> None:
    """Different actor classes and horizons should survive as separate matrix rows."""
    report = build_forecast_transferability_stress_matrix(
        [
            _metric_report(
                actor_class="pedestrian",
                transfer_dimensions=_complete_transfer_dimensions(),
                horizon_s=[0.5, 1.0],
            ),
            _metric_report(
                actor_class="bicycle",
                transfer_dimensions=_complete_transfer_dimensions(),
                horizon_s=[0.5, 1.0],
            ),
        ],
        report_id="actor-horizon-transfer",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    rows = report["matrix_rows"]
    assert len(rows) == 4
    assert {row["actor_type"] for row in rows} == {"pedestrian", "bicycle"}
    assert {float(row["horizon_s"]) for row in rows} == {0.5, 1.0}


def test_transferability_matrix_preserves_row_level_scenario_family_and_actor_cells() -> None:
    """Scenario family and actor class should be carried from durable aggregate rows."""
    transfer_dimensions = {
        "observation_noise": "none",
        "latency": "0_steps",
        "dropout": "none",
        "occlusion": "clear",
        "density": "low",
        "pedestrian_model_family": "social_force",
        "semantic_metadata_present": "present",
    }
    report = build_forecast_transferability_stress_matrix(
        [
            _metric_report(
                transfer_dimensions=transfer_dimensions,
                aggregate_row_fields={"scenario_family": "crossing_split_a"},
                actor_class="pedestrian",
            ),
            _metric_report(
                transfer_dimensions=transfer_dimensions,
                aggregate_row_fields={"scenario_family": "crossing_split_b"},
                actor_class="bicycle",
            ),
        ],
        report_id="scenario-family-transfer",
        required_dimensions=(
            "observation_tier",
            "scenario_family",
            "actor_type",
            "semantic_metadata_present",
        ),
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    rows = report["matrix_rows"]
    assert len(rows) == 2
    assert {row["transfer_dimensions"]["scenario_family"] for row in rows} == {
        "crossing_split_a",
        "crossing_split_b",
    }
    assert {row["actor_type"] for row in rows} == {"pedestrian", "bicycle"}
    assert report["recommendation"]["claim_status"] == "benchmark-eligible"


def test_transferability_matrix_marks_unavailable_actor_and_scenario_family_cells() -> None:
    """Missing actor class or scenario family must stay as explicit blocked cells."""
    complete = _complete_transfer_dimensions()
    complete.pop("semantic_metadata_present", None)
    complete.pop("map_family", None)
    report = build_forecast_transferability_stress_matrix(
        [
            _metric_report(
                transfer_dimensions=_complete_transfer_dimensions(),
                actor_class="pedestrian",
            ),
            _metric_report(
                transfer_dimensions=complete,
                actor_class=None,
                aggregate_row_fields={"scenario_family": None},
                provenance_fields={"scenario_family": None},
            ),
        ],
        report_id="unavailable-cells-transfer",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    rows = report["matrix_rows"]
    assert len(rows) == 2
    blocked_rows = [row for row in rows if row["evidence_status"] == "blocked"]
    assert len(blocked_rows) == 1
    assert None in {row["actor_type"] for row in blocked_rows}
    assert "scenario_family" in blocked_rows[0]["unavailable_dimensions"]
    assert "map_family" in blocked_rows[0]["unavailable_dimensions"]
    assert "actor_type" in blocked_rows[0]["unavailable_dimensions"]
    assert report["limitation_rows"]
    assert report["dimension_coverage"]["actor_type"]["unavailable_report_count"] == 1


def test_transferability_matrix_tracks_semantic_metadata_present_and_absent() -> None:
    """Missing semantic metadata should remain explicit when present and absent differ."""
    transfer_dimensions = _complete_transfer_dimensions()
    report = build_forecast_transferability_stress_matrix(
        [
            _metric_report(
                transfer_dimensions={
                    **transfer_dimensions,
                    "semantic_metadata_present": True,
                },
            ),
            _metric_report(
                transfer_dimensions={
                    **transfer_dimensions,
                    "semantic_metadata_present": False,
                },
            ),
        ],
        report_id="semantic-meta-transfer",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    assert {
        row["transfer_dimensions"]["semantic_metadata_present"] for row in report["matrix_rows"]
    } == {"present", "absent"}


def test_transferability_matrix_captures_row_provenance() -> None:
    """Matrix rows should carry artifact provenance for traceability."""
    report = build_forecast_transferability_stress_matrix(
        [
            _metric_report(
                transfer_dimensions=_complete_transfer_dimensions(),
                report_fields={
                    "artifact_input": {
                        "issue": 2866,
                        "source": "fixture://forecast-metrics",
                        "command": "unit-test",
                    }
                },
            )
        ],
        report_id="provenance-transfer",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    artifact_input = report["matrix_rows"][0]["artifact_input"]
    assert artifact_input is not None
    assert artifact_input["issue"] == 2866
    assert artifact_input["source"] == "fixture://forecast-metrics"


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
    assert "Claim status: blocked" in markdown
    assert "## Limitations" in markdown
    assert "observation_noise" in markdown


def test_transferability_matrix_marks_unavailable_metrics() -> None:
    """Rows with empty denominators should be marked unavailable."""
    report = build_forecast_transferability_stress_matrix(
        [_metric_report(denominator=0, transfer_dimensions=_complete_transfer_dimensions())],
        report_id="unavailable-metrics",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    assert report["matrix_rows"][0]["evidence_status"] == "unavailable"
    assert report["recommendation"]["claim_status"] == "diagnostic-only"


def test_transferability_cli_writes_json_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should write reviewable JSON and Markdown artifacts."""
    metric_path = tmp_path / "metrics.json"
    out_json = tmp_path / "transfer.json"
    out_md = tmp_path / "transfer.md"
    metric_path.write_text(
        json.dumps(
            _metric_report(transfer_dimensions=_complete_transfer_dimensions()),
        ),
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
            "--generated-at-utc",
            "2026-06-15T00:00:00+00:00",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)

    assert summary["decision"] == "continue"
    assert out_json.exists()
    assert out_md.exists()
    output = json.loads(out_json.read_text(encoding="utf-8"))
    assert output["report_id"] == "cli-transfer"
    assert output["generated_at_utc"] == "2026-06-15T00:00:00+00:00"


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
