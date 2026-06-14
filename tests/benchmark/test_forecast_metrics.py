"""Tests for ForecastBatch.v1 metric evaluation."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.forecast_batch import (
    ActorForecast,
    CoordinateFrame,
    ForecastBatch,
    ForecastBatchProvenance,
)
from robot_sf.benchmark.forecast_metrics import (
    evaluate_forecast_batch,
    format_forecast_metrics_markdown,
)


def _provenance(**overrides: object) -> ForecastBatchProvenance:
    data: dict[str, object] = {
        "predictor_id": "predictor-v1",
        "predictor_family": "synthetic",
        "observation_tier": "deployable_observation",
        "frame": CoordinateFrame(name="world", units="m", axes=("x", "y")),
        "dt_s": 0.5,
        "horizons_s": [0.5, 1.0],
        "scenario_id": "scenario_a",
        "seed": 11,
        "fallback_status": "native",
        "degraded_status": "none",
        "actor_ids": ["ped_1", "ped_2"],
        "actor_mask": [True, True],
        "actor_mask_metadata": {"semantics": "true means included"},
        "feature_schema": {"name": "forecast_fixture_v1"},
    }
    data.update(overrides)
    return ForecastBatchProvenance(**data)


def _row(report: dict, metric: str, actor_id: str, horizon_s: float) -> dict:
    for row in report["metric_rows"]:
        if (
            row["metric"] == metric
            and row.get("actor_id") == actor_id
            and row["horizon_s"] == horizon_s
        ):
            return row
    raise AssertionError(f"missing row {metric=} {actor_id=} {horizon_s=}")


def _aggregate(
    report: dict, metric: str, horizon_s: float, actor_class: str = "pedestrian"
) -> dict:
    for row in report["aggregate_rows"]:
        if (
            row["metric"] == metric
            and row["horizon_s"] == horizon_s
            and row["actor_class"] == actor_class
        ):
            return row
    raise AssertionError(f"missing aggregate {metric=} {horizon_s=} {actor_class=}")


def test_forecast_metrics_evaluate_deterministic_ade_fde() -> None:
    """Deterministic trajectories should produce ADE rows and final-horizon FDE."""
    batch = ForecastBatch(
        provenance=_provenance(),
        forecasts=[
            ActorForecast(actor_id="ped_1", deterministic=[[0.0, 0.0], [1.0, 0.0]]),
            ActorForecast(actor_id="ped_2", deterministic=[[1.0, 1.0], [1.0, 2.0]]),
        ],
        metadata={"scenario_family": "unit"},
    )
    truth = {"ped_1": [[0.0, 0.0], [2.0, 0.0]], "ped_2": [[1.0, 1.0], [1.0, 3.0]]}

    report = evaluate_forecast_batch(batch, truth)

    assert report["schema_version"] == "ForecastMetrics.v1"
    assert report["provenance"]["observation_tier"] == "deployable_observation"
    assert _row(report, "ade", "ped_1", 1.0)["value"] == 1.0
    assert _row(report, "fde", "ped_1", 1.0)["value"] == 1.0
    assert _row(report, "fde", "ped_1", 0.5)["status"] == "unavailable"
    assert _aggregate(report, "mean_ade", 1.0)["denominator"] == 2
    assert _aggregate(report, "mean_ade", 1.0)["value"] == 1.0


def test_forecast_metrics_evaluate_sampled_minade_minfde() -> None:
    """Sampled trajectories should expose minADE@K/minFDE@K separately."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[
            ActorForecast(
                actor_id="ped_1",
                samples=[
                    [[0.0, 0.0], [2.5, 0.0]],
                    [[0.0, 0.5], [2.0, 0.0]],
                ],
            )
        ],
    )
    truth = {"ped_1": [[0.0, 0.0], [2.0, 0.0]]}

    report = evaluate_forecast_batch(batch, truth)

    assert _row(report, "minade@k", "ped_1", 0.5)["value"] == 0.0
    assert _row(report, "minfde@k", "ped_1", 1.0)["value"] == 0.0
    assert _aggregate(report, "mean_minade@k", 1.0)["denominator"] == 1


def test_forecast_metrics_evaluate_multimodal_expected_error() -> None:
    """Mode probabilities should produce expected displacement rows."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[
            ActorForecast(
                actor_id="ped_1",
                samples=[
                    [[0.0, 0.0], [1.0, 0.0]],
                    [[2.0, 0.0], [3.0, 0.0]],
                ],
                mode_probabilities=[0.75, 0.25],
            )
        ],
    )
    truth = {"ped_1": [[0.0, 0.0], [1.0, 0.0]]}

    report = evaluate_forecast_batch(batch, truth)

    assert _row(report, "expected_ade", "ped_1", 0.5)["value"] == 0.5
    assert _row(report, "expected_fde", "ped_1", 1.0)["value"] == 0.5


def test_forecast_metrics_report_missing_mask_denominators() -> None:
    """Masked actors should be excluded from denominators and reported."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[ActorForecast(actor_id="ped_1", deterministic=[[0.0, 0.0], [1.0, 0.0]])],
    )
    truth = {"ped_1": [[0.0, 0.0], [1.0, 0.0]], "ped_2": [[10.0, 10.0], [10.0, 10.0]]}

    report = evaluate_forecast_batch(batch, truth)

    assert report["denominator_health"]["active_actor_count"] == 1
    assert report["denominator_health"]["excluded_actor_count"] == 1
    assert _aggregate(report, "mean_ade", 1.0)["denominator"] == 1


def test_forecast_metrics_report_empty_denominator_as_unavailable() -> None:
    """Missing ground truth should not become zero performance."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[ActorForecast(actor_id="ped_1", deterministic=[[0.0, 0.0], [1.0, 0.0]])],
    )

    report = evaluate_forecast_batch(batch, {})

    assert report["denominator_health"]["has_empty_denominator"] is True
    assert _row(report, "ade", "ped_1", 0.5)["value"] is None
    assert {row["metric"] for row in report["metric_rows"] if row.get("actor_id") == "ped_1"} == {
        "ade",
        "fde",
        "minade@k",
        "minfde@k",
        "expected_ade",
        "expected_fde",
        "miss_rate",
        "likelihood",
        "coverage",
        "collision_relevance",
    }
    assert _row(report, "miss_rate", "ped_1", 1.0)["status"] == "unavailable"
    assert _row(report, "coverage", "ped_1", 0.5)["status"] == "unavailable"
    assert _aggregate(report, "mean_ade", 0.5)["status"] == "unavailable"
    assert _aggregate(report, "mean_ade", 0.5)["denominator"] == 0


def test_forecast_metrics_mark_probabilistic_metrics_unavailable_for_deterministic() -> None:
    """Deterministic-only forecasts should not report probabilistic metrics as zero."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[ActorForecast(actor_id="ped_1", deterministic=[[0.0, 0.0], [1.0, 0.0]])],
    )
    truth = {"ped_1": [[0.0, 0.0], [1.0, 0.0]]}

    report = evaluate_forecast_batch(batch, truth)

    assert _row(report, "minade@k", "ped_1", 0.5)["status"] == "unavailable"
    assert _row(report, "expected_ade", "ped_1", 0.5)["status"] == "unavailable"
    assert _row(report, "likelihood", "ped_1", 0.5)["status"] == "unavailable"
    assert _row(report, "coverage", "ped_1", 0.5)["status"] == "unavailable"
    assert _row(report, "collision_relevance", "ped_1", 0.5)["status"] == "unavailable"


def test_forecast_metrics_compute_final_horizon_miss_rate() -> None:
    """Miss rate should use final-horizon displacement and stay unavailable earlier."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[ActorForecast(actor_id="ped_1", deterministic=[[0.0, 0.0], [5.0, 0.0]])],
    )
    truth = {"ped_1": [[0.0, 0.0], [1.0, 0.0]]}

    report = evaluate_forecast_batch(batch, truth, miss_threshold_m=2.0)

    assert report["metric_parameters"]["miss_threshold_m"] == 2.0
    assert _row(report, "miss_rate", "ped_1", 0.5)["status"] == "unavailable"
    assert _row(report, "miss_rate", "ped_1", 0.5)["note"] == "final horizon only"
    assert _row(report, "miss_rate", "ped_1", 1.0)["value"] == 1.0
    assert "note" not in _row(report, "miss_rate", "ped_1", 1.0)
    assert _aggregate(report, "mean_miss_rate", 1.0)["value"] == 1.0


def test_forecast_metrics_explain_missing_final_horizon_payloads() -> None:
    """Unavailable final-horizon metrics should carry a specific reason."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[ActorForecast(actor_id="ped_1", deterministic=None, samples=None)],
    )
    truth = {"ped_1": [[0.0, 0.0], [1.0, 0.0]]}

    report = evaluate_forecast_batch(batch, truth)

    assert _row(report, "fde", "ped_1", 1.0)["note"] == "deterministic trajectory unavailable"
    assert _row(report, "minfde@k", "ped_1", 1.0)["note"] == "sampled trajectories unavailable"
    assert _row(report, "expected_fde", "ped_1", 1.0)["note"] == "mode probabilities unavailable"


def test_forecast_metrics_markdown_summary_contains_denominators() -> None:
    """Markdown output should be compact enough for context evidence promotion."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[ActorForecast(actor_id="ped_1", deterministic=[[0.0, 0.0], [1.0, 0.0]])],
    )
    truth = {"ped_1": [[0.0, 0.0], [1.0, 0.0]]}

    markdown = format_forecast_metrics_markdown(evaluate_forecast_batch(batch, truth))

    assert markdown.startswith("# Forecast Metrics Summary")
    assert "active=1, excluded=1" in markdown
    assert "| mean_ade | 1 | pedestrian | 1 | ok | 0 |" in markdown
    assert "open-loop evidence only" in markdown


def test_forecast_metrics_keep_actor_classes_separate() -> None:
    """Actor-class metadata should split aggregate denominators."""
    batch = ForecastBatch(
        provenance=_provenance(),
        forecasts=[
            ActorForecast(
                actor_id="ped_1",
                deterministic=[[0.0, 0.0], [1.0, 0.0]],
                uncertainty_metadata={"actor_class": "pedestrian"},
            ),
            ActorForecast(
                actor_id="ped_2",
                deterministic=[[0.0, 0.0], [3.0, 0.0]],
                uncertainty_metadata={"actor_class": "bicycle"},
            ),
        ],
    )
    truth = {"ped_1": [[0.0, 0.0], [1.0, 0.0]], "ped_2": [[0.0, 0.0], [1.0, 0.0]]}

    report = evaluate_forecast_batch(batch, truth)

    assert _aggregate(report, "mean_ade", 1.0, "pedestrian")["value"] == 0.0
    assert _aggregate(report, "mean_ade", 1.0, "bicycle")["value"] == 2.0


def test_forecast_metrics_accept_actor_class_from_batch_metadata() -> None:
    """Batch-level actor class metadata should work when forecast metadata is absent."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[ActorForecast(actor_id="ped_1", deterministic=[[0.0, 0.0], [1.0, 0.0]])],
        metadata={"actor_classes": {"ped_1": "child"}},
    )
    truth = {"ped_1": [[0.0, 0.0], [1.0, 0.0]]}

    report = evaluate_forecast_batch(batch, truth)

    assert _aggregate(report, "mean_ade", 1.0, "child")["value"] == 0.0


def test_forecast_metrics_reject_misaligned_ground_truth() -> None:
    """Ground-truth trajectories must align with ForecastBatch horizons."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[ActorForecast(actor_id="ped_1", deterministic=[[0.0, 0.0], [1.0, 0.0]])],
    )

    with pytest.raises(ValueError, match="ground_truth"):
        evaluate_forecast_batch(batch, {"ped_1": [[0.0, 0.0]]})
