"""Tests for ForecastConformalPilot.v1 artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.forecast_batch import (
    ActorForecast,
    CoordinateFrame,
    ForecastBatch,
    ForecastBatchProvenance,
)
from robot_sf.benchmark.forecast_conformal_pilot import (
    FORECAST_CONFORMAL_PILOT_SCHEMA_VERSION,
    build_forecast_conformal_pilot_report,
    format_forecast_conformal_pilot_markdown,
    write_forecast_conformal_pilot_report,
)


def _batch(
    *,
    scenario_id: str = "classic_crossing_seed_001",
    scenario_family: str = "classic_crossing",
    deterministic: list[list[float]] | None = None,
) -> ForecastBatch:
    return ForecastBatch(
        provenance=ForecastBatchProvenance(
            predictor_id="cv-baseline",
            predictor_family="constant_velocity",
            observation_tier="deployable_observation",
            frame=CoordinateFrame(name="world"),
            dt_s=0.5,
            horizons_s=[0.5, 1.0],
            scenario_id=scenario_id,
            seed=1,
            timestamp="2026-06-15T12:00:00Z",
            fallback_status="native",
            degraded_status="none",
            actor_ids=["actor-1"],
            actor_mask=[True],
            actor_mask_metadata={"source": "fixture"},
            feature_schema={"features": ["x", "y"]},
        ),
        forecasts=[
            ActorForecast(
                actor_id="actor-1",
                deterministic=deterministic or [[0.0, 0.0], [1.0, 0.0]],
                uncertainty_metadata={"collision_relevance": 0.2},
            )
        ],
        metadata={"scenario_family": scenario_family},
    )


def _case(
    *,
    batch: ForecastBatch | None = None,
    truth: list[list[float]] | None = None,
    split_id: str = "fixture",
) -> dict[str, object]:
    return {
        "batch": (batch or _batch()).to_dict(),
        "ground_truth": {"actor-1": truth or [[0.1, 0.0], [1.1, 0.0]]},
        "split_id": split_id,
    }


def test_conformal_pilot_reports_heldout_coverage_and_radius() -> None:
    """Calibration residuals should produce held-out coverage and set-size diagnostics."""
    report = build_forecast_conformal_pilot_report(
        [
            _case(truth=[[0.1, 0.0], [1.1, 0.0]], split_id="train-calibration"),
            _case(truth=[[0.2, 0.0], [1.2, 0.0]], split_id="train-calibration"),
        ],
        [_case(truth=[[0.15, 0.0], [1.15, 0.0]], split_id="heldout")],
        report_id="pilot",
        coverage_target=0.5,
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    assert report["schema_version"] == FORECAST_CONFORMAL_PILOT_SCHEMA_VERSION
    assert report["recommendation"]["decision"] == "continue"
    assert report["split_provenance"]["calibration_split_ids"] == ["train-calibration"]
    assert report["split_provenance"]["evaluation_split_ids"] == ["heldout"]
    assert {row["horizon_s"] for row in report["pilot_rows"]} == {0.5, 1.0}
    assert all(row["empirical_coverage"] == 1.0 for row in report["pilot_rows"])
    assert all(row["mean_set_size_proxy_m2"] > 0.0 for row in report["pilot_rows"])


def test_conformal_pilot_marks_missing_calibration_denominator_as_limitation() -> None:
    """Evaluation-only groups should remain limitations, not conformal evidence."""
    calibration = _case(batch=_batch(scenario_id="other_seed_001", scenario_family="other"))
    evaluation = _case()

    report = build_forecast_conformal_pilot_report(
        [calibration],
        [evaluation],
        report_id="missing-calibration",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    limited_rows = [
        row for row in report["pilot_rows"] if row["scenario_family"] == "classic_crossing"
    ]
    assert limited_rows
    assert all(
        row["coverage_status"] == "unavailable_no_calibration_denominator" for row in limited_rows
    )
    assert report["recommendation"]["decision"] == "wait"
    assert report["limitation_rows"]


def test_conformal_pilot_marks_under_coverage_as_revise() -> None:
    """Held-out residuals above the fitted radius should require revision."""
    report = build_forecast_conformal_pilot_report(
        [_case(truth=[[0.1, 0.0], [1.1, 0.0]])],
        [_case(truth=[[3.0, 0.0], [4.0, 0.0]])],
        report_id="under-covered",
        coverage_target=0.5,
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    assert report["recommendation"]["decision"] == "revise"
    assert all(row["coverage_status"] == "under_covered_heldout" for row in report["pilot_rows"])
    assert all(row["miss_count"] == 1 for row in report["pilot_rows"])


def test_conformal_pilot_marks_missing_deterministic_scores_as_limitation() -> None:
    """Forecasts without deterministic trajectories cannot fit deterministic tubes."""
    no_deterministic = _batch(deterministic=None)
    no_deterministic.forecasts[0].deterministic = None
    no_deterministic.forecasts[0].occupancy_summary = {"representation": "occupancy_only"}

    report = build_forecast_conformal_pilot_report(
        [_case(batch=no_deterministic)],
        [_case(batch=no_deterministic)],
        report_id="no-deterministic",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    assert report["pilot_rows"] == []
    assert report["limitation_rows"][0]["reason"] == (
        "no deterministic forecast and ground-truth score pairs were available"
    )
    assert report["recommendation"]["decision"] == "wait"


def test_conformal_pilot_rejects_bad_ground_truth_shape() -> None:
    """Ground truth must align with forecast horizons."""
    with pytest.raises(ValueError, match="ground_truth trajectories"):
        build_forecast_conformal_pilot_report(
            [_case(truth=[[0.0, 0.0]])],
            [_case()],
            report_id="bad-truth",
        )


def test_conformal_pilot_markdown_includes_boundaries() -> None:
    """Markdown should include held-out evidence and no-overclaiming language."""
    report = build_forecast_conformal_pilot_report(
        [_case()],
        [_case()],
        report_id="markdown",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )

    markdown = format_forecast_conformal_pilot_markdown(report)

    assert "# Forecast Conformal Pilot Report" in markdown
    assert "Held-out evaluation cases: 1" in markdown
    assert "smoke evidence only" in markdown


def test_conformal_pilot_writer_validates_and_writes_artifacts(tmp_path: Path) -> None:
    """In-process writer should validate schema and write JSON plus Markdown."""
    report = build_forecast_conformal_pilot_report(
        [_case(truth=[[0.1, 0.0], [1.1, 0.0]])],
        [_case(truth=[[3.0, 0.0], [4.0, 0.0]])],
        report_id="writer",
        generated_at_utc="2026-06-15T00:00:00+00:00",
    )
    out_json = tmp_path / "pilot.json"
    out_md = tmp_path / "pilot.md"

    paths = write_forecast_conformal_pilot_report(
        report,
        json_path=out_json,
        markdown_path=out_md,
    )

    assert paths == {"json": out_json, "markdown": out_md}
    assert json.loads(out_json.read_text(encoding="utf-8"))["report_id"] == "writer"
    assert "## Limitations" in out_md.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="report must be a mapping"):
        write_forecast_conformal_pilot_report("bad", json_path=tmp_path / "bad.json")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"ForecastConformalPilot\.v1"):
        write_forecast_conformal_pilot_report({"schema_version": "wrong"}, json_path=out_json)


def test_conformal_pilot_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    """CLI should write reviewable JSON and Markdown artifacts."""
    calibration_batch = tmp_path / "calibration_batch.json"
    calibration_truth = tmp_path / "calibration_truth.json"
    evaluation_batch = tmp_path / "evaluation_batch.json"
    evaluation_truth = tmp_path / "evaluation_truth.json"
    out_json = tmp_path / "conformal.json"
    out_md = tmp_path / "conformal.md"
    batch_payload = _batch().to_dict()
    truth_payload = {"actor-1": [[0.1, 0.0], [1.1, 0.0]]}
    for path in (calibration_batch, evaluation_batch):
        path.write_text(json.dumps(batch_payload), encoding="utf-8")
    for path in (calibration_truth, evaluation_truth):
        path.write_text(json.dumps(truth_payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[2]
                / "scripts/benchmark/build_forecast_conformal_pilot.py"
            ),
            "--calibration-batch",
            str(calibration_batch),
            "--calibration-ground-truth",
            str(calibration_truth),
            "--evaluation-batch",
            str(evaluation_batch),
            "--evaluation-ground-truth",
            str(evaluation_truth),
            "--report-id",
            "cli-conformal",
            "--coverage-target",
            "0.5",
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
    assert json.loads(out_json.read_text(encoding="utf-8"))["report_id"] == "cli-conformal"


def test_conformal_pilot_cli_reports_mismatched_inputs(tmp_path: Path) -> None:
    """CLI should return concise parser errors for mismatched path counts."""
    batch_path = tmp_path / "batch.json"
    truth_path = tmp_path / "truth.json"
    batch_path.write_text(json.dumps(_batch().to_dict()), encoding="utf-8")
    truth_path.write_text(json.dumps({"actor-1": [[0.0, 0.0], [1.0, 0.0]]}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[2]
                / "scripts/benchmark/build_forecast_conformal_pilot.py"
            ),
            "--calibration-batch",
            str(batch_path),
            "--calibration-ground-truth",
            str(truth_path),
            str(truth_path),
            "--evaluation-batch",
            str(batch_path),
            "--evaluation-ground-truth",
            str(truth_path),
            "--report-id",
            "bad-cli",
            "--out-json",
            str(tmp_path / "out.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "calibration batch and ground-truth counts must match" in result.stderr
    assert "Traceback" not in result.stderr
