"""Tests for issue #3973 open-loop and closed-loop prediction evaluation."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.forecast_batch import (
    ActorForecast,
    CoordinateFrame,
    ForecastBatch,
    ForecastBatchProvenance,
)
from robot_sf.benchmark.prediction_open_closed_eval import (
    PREDICTION_OPEN_CLOSED_SCHEMA_VERSION,
    build_open_closed_prediction_report,
    build_paired_prediction_eval_report,
    compute_open_loop_prediction_metrics,
    summarize_closed_loop_prediction_metrics,
)


def _provenance(**overrides: object) -> ForecastBatchProvenance:
    data: dict[str, object] = {
        "predictor_id": "cv",
        "predictor_family": "constant_velocity",
        "observation_tier": "deployable_observation",
        "frame": CoordinateFrame(name="world", units="m", axes=("x", "y")),
        "dt_s": 0.5,
        "horizons_s": [0.5, 1.0],
        "scenario_id": "scenario_a",
        "seed": 3973,
        "timestamp": "2026-07-02T10:00:00Z",
        "fallback_status": "native",
        "degraded_status": "none",
        "actor_ids": ["ped_1"],
        "actor_mask": [True],
        "actor_mask_metadata": {"semantics": "true means included"},
        "feature_schema": {"name": "prediction_eval_fixture_v1"},
    }
    data.update(overrides)
    return ForecastBatchProvenance(**data)


def _gaussian_batch() -> ForecastBatch:
    return ForecastBatch(
        provenance=_provenance(),
        forecasts=[
            ActorForecast(
                actor_id="ped_1",
                deterministic=[[0.0, 0.0], [2.0, 0.0]],
                gaussian=[
                    {"mean": [0.0, 0.0], "covariance": [[0.25, 0.0], [0.0, 0.25]]},
                    {"mean": [2.0, 0.0], "covariance": [[0.25, 0.0], [0.0, 0.25]]},
                ],
            )
        ],
    )


def test_prediction_eval_open_loop_metrics_project_forecast_batch() -> None:
    """Open-loop summary should expose ADE, FDE, calibration, and spread."""
    report = compute_open_loop_prediction_metrics(
        forecast_batches=[_gaussian_batch()],
        ground_truth_by_batch={"ped_1": [[0.0, 0.0], [3.0, 0.0]]},
    )

    assert report["ade"] == pytest.approx(0.5)
    assert report["fde"] == pytest.approx(1.0)
    assert report["calibration_error"] == pytest.approx(0.05)
    assert report["prediction_spread"] == pytest.approx(0.5)
    assert report["sample_count"] == 2
    assert report["statuses"]["calibration_error"]["status"] == "ok"


def test_prediction_eval_open_loop_deterministic_status_is_explicit() -> None:
    """Deterministic-only forecasts should not silently invent calibration."""
    batch = ForecastBatch(
        provenance=_provenance(),
        forecasts=[
            ActorForecast(
                actor_id="ped_1",
                deterministic=[[0.0, 0.0], [1.0, 0.0]],
            )
        ],
    )

    report = compute_open_loop_prediction_metrics(
        forecast_batches=[batch],
        ground_truth_by_batch={"ped_1": [[0.0, 0.0], [1.0, 0.0]]},
    )

    assert report["prediction_spread"] == 0.0
    assert report["statuses"]["prediction_spread"]["status"] == "deterministic_no_spread"
    assert report["calibration_error"] is None
    assert report["statuses"]["calibration_error"]["status"] == "unavailable"


def test_prediction_eval_open_loop_missing_truth_is_unavailable() -> None:
    """Missing ground truth should be visible in denominator health."""
    report = compute_open_loop_prediction_metrics(
        forecast_batches=[_gaussian_batch()],
        ground_truth_by_batch={"other_actor": [[0.0, 0.0], [1.0, 0.0]]},
    )

    assert report["ade"] is None
    assert report["statuses"]["ade"]["status"] == "unavailable"
    assert report["denominators"]["missing_ground_truth_batch_count"] == 0


def test_prediction_eval_open_loop_empty_and_multibatch_missing_statuses() -> None:
    """Empty inputs and missing batch truth should fail closed as unavailable."""
    empty_report = compute_open_loop_prediction_metrics(
        forecast_batches=[],
        ground_truth_by_batch={},
    )
    assert empty_report["statuses"]["ade"]["status"] == "unavailable"
    assert empty_report["sample_count"] == 0

    batch = ForecastBatch(
        provenance=_provenance(scenario_id="scenario_missing"),
        forecasts=[
            ActorForecast(
                actor_id="ped_1",
                deterministic=[[0.0, 0.0], [1.0, 0.0]],
            )
        ],
    )
    missing_report = compute_open_loop_prediction_metrics(
        forecast_batches=[batch, _gaussian_batch()],
        ground_truth_by_batch={
            "scenario_a": {"ped_1": [[0.0, 0.0], [3.0, 0.0]]},
        },
    )
    assert missing_report["denominators"]["missing_ground_truth_batch_count"] == 1
    assert missing_report["missing_ground_truth_batches"] == ["scenario_missing"]


def test_prediction_eval_open_loop_sample_spread_is_supported() -> None:
    """Sampled forecasts should contribute prediction-spread values."""
    batch = ForecastBatch(
        provenance=_provenance(),
        forecasts=[
            ActorForecast(
                actor_id="ped_1",
                deterministic=[[0.0, 0.0], [1.0, 0.0]],
                samples=[
                    [[0.0, 0.0], [1.0, 0.0]],
                    [[0.0, 2.0], [1.0, 2.0]],
                ],
            )
        ],
    )
    report = compute_open_loop_prediction_metrics(
        forecast_batches=[batch],
        ground_truth_by_batch={"ped_1": [[0.0, 0.0], [1.0, 0.0]]},
    )

    assert report["prediction_spread"] == pytest.approx(1.0)
    assert report["statuses"]["prediction_spread"]["status"] == "ok"


def test_prediction_eval_closed_loop_summary_projects_existing_episode_metrics() -> None:
    """Closed-loop summary should populate issue #3973 metrics from episode rows."""
    rows = [
        {
            "metrics": {
                "near_misses": 1,
                "min_distance": 0.5,
                "jerk_mean": 0.2,
                "total_collision_count": 0,
                "time_to_goal": 4.0,
            },
            "termination_reason": "success",
        },
        {
            "metrics": {
                "near_misses": 0,
                "min_distance": 1.2,
                "jerk_mean": 0.4,
                "total_collision_count": 1,
            },
            "termination_reason": "timeout",
        },
    ]

    report = summarize_closed_loop_prediction_metrics(rows)

    assert report["collision_rate"] == pytest.approx(0.5)
    assert report["near_miss_rate"] == pytest.approx(0.5)
    assert report["min_distance"] == pytest.approx(0.5)
    assert report["timeout_rate"] == pytest.approx(0.5)
    assert report["time_to_goal"] == pytest.approx(4.0)
    assert report["jerk"] == pytest.approx(0.3)
    assert report["closed_loop_denominators"]["episode_count"] == 2
    assert report["closed_loop_denominators"]["time_to_goal_denominator"] == 1


def test_prediction_eval_closed_loop_missing_event_fields_are_unavailable() -> None:
    """Missing event signals should not look like zero-risk closed-loop evidence."""
    report = summarize_closed_loop_prediction_metrics(
        [
            {
                "metrics": {
                    "min_distance": 1.2,
                    "jerk_mean": 0.1,
                }
            }
        ]
    )

    assert report["collision_rate"] is None
    assert report["near_miss_rate"] is None
    assert report["timeout_rate"] is None
    assert report["closed_loop_denominators"]["collision_rate_denominator"] == 0
    assert report["closed_loop_denominators"]["near_miss_rate_denominator"] == 0
    assert report["closed_loop_denominators"]["timeout_rate_denominator"] == 0
    assert report["statuses"]["collision_rate"]["status"] == "unavailable"
    assert report["statuses"]["near_miss_rate"]["status"] == "unavailable"
    assert report["statuses"]["timeout_rate"]["status"] == "unavailable"


def test_prediction_eval_open_loop_invalid_covariance_is_unavailable() -> None:
    """Invalid Gaussian covariance matrices should not count as uncertainty evidence."""
    batch = ForecastBatch(
        provenance=_provenance(),
        forecasts=[
            ActorForecast(
                actor_id="ped_1",
                deterministic=[[0.0, 0.0], [1.0, 0.0]],
                gaussian=[
                    {"mean": [0.0, 0.0], "covariance": [[1.0, 0.0], [0.0, -1.0]]},
                    {"mean": [1.0, 0.0], "covariance": [[1.0, 2.0], [0.0, 1.0]]},
                ],
            )
        ],
    )

    report = compute_open_loop_prediction_metrics(
        forecast_batches=[batch],
        ground_truth_by_batch={"ped_1": [[0.0, 0.0], [1.0, 0.0]]},
    )

    assert report["calibration_error"] is None
    assert report["prediction_spread"] is None
    assert report["statuses"]["calibration_error"]["status"] == "unavailable"
    assert report["statuses"]["prediction_spread"]["status"] == "unavailable"


def test_prediction_eval_closed_loop_empty_rows_are_unavailable() -> None:
    """Empty closed-loop inputs should retain the expected schema."""
    report = summarize_closed_loop_prediction_metrics([])

    assert report["collision_rate"] is None
    assert report["closed_loop_denominators"]["episode_count"] == 0
    assert report["statuses"]["jerk"]["status"] == "unavailable"


def test_prediction_eval_paired_report_keeps_divergence_and_claim_boundary() -> None:
    """Paired report should make open-vs-closed divergence inspectable."""
    open_loop = {
        "ade": 0.42,
        "fde": 0.73,
        "calibration_error": 0.08,
        "prediction_spread": 0.55,
        "sample_count": 12,
    }
    closed_loop = {
        "collision_rate": 0.0,
        "near_miss_rate": 0.5,
        "min_distance": 0.62,
        "timeout_rate": 0.0,
        "time_to_goal": 4.3,
        "jerk": 0.18,
    }

    predictor_report = build_open_closed_prediction_report(
        predictor_id="cv",
        open_loop_report=open_loop,
        closed_loop_report=closed_loop,
        predictor_metadata={"predictor_family": "constant_velocity"},
    )
    report = build_paired_prediction_eval_report([predictor_report])

    assert report["schema_version"] == PREDICTION_OPEN_CLOSED_SCHEMA_VERSION
    assert report["issue"] == 3973
    assert "no causal safety" in report["claim_boundary"]
    assert report["predictors"][0]["open_loop"] == open_loop
    assert report["predictors"][0]["closed_loop"] == closed_loop
    assert report["predictors"][0]["divergence"]["rank_comparison_available"] is False


def test_prediction_eval_paired_report_populates_multi_predictor_ranks() -> None:
    """Multiple predictors should receive open and closed-loop rank projections."""
    predictor_a = build_open_closed_prediction_report(
        predictor_id="cv_a",
        open_loop_report={"fde": 0.5},
        closed_loop_report={"collision_rate": 1.0, "near_miss_rate": 0.0, "min_distance": 0.4},
    )
    predictor_b = build_open_closed_prediction_report(
        predictor_id="cv_b",
        open_loop_report={"fde": 0.8},
        closed_loop_report={"collision_rate": 0.0, "near_miss_rate": 0.0, "min_distance": 0.6},
    )

    report = build_paired_prediction_eval_report([predictor_a, predictor_b])
    ranks = {
        predictor["predictor_id"]: predictor["divergence"] for predictor in report["predictors"]
    }

    assert ranks["cv_a"]["rank_comparison_available"] is True
    assert ranks["cv_a"]["open_loop_rank"] == 1
    assert ranks["cv_a"]["closed_loop_rank"] == 2
    assert ranks["cv_b"]["open_loop_rank"] == 2
    assert ranks["cv_b"]["closed_loop_rank"] == 1


def test_prediction_eval_paired_report_marks_missing_rank_inputs() -> None:
    """Multi-predictor rank skips should name missing inputs, not single-predictor smoke."""
    predictor_a = build_open_closed_prediction_report(
        predictor_id="cv_a",
        open_loop_report={"fde": 0.5},
        closed_loop_report={"collision_rate": 0.0, "near_miss_rate": 0.0, "min_distance": 0.6},
    )
    predictor_b = build_open_closed_prediction_report(
        predictor_id="cv_b",
        open_loop_report={"ade": 0.1},
        closed_loop_report={"collision_rate": 0.0, "near_miss_rate": 0.0, "min_distance": 0.7},
    )

    report = build_paired_prediction_eval_report([predictor_a, predictor_b])
    divergence = report["predictors"][1]["divergence"]

    assert divergence["rank_comparison_available"] is False
    assert divergence["reason"] == "missing_rank_inputs"
    assert divergence["missing_rank_inputs"] == ["open_loop.fde"]
