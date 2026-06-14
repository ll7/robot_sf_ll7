"""Tests for ForecastBatch.v1 metric evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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


def test_forecast_metrics_prefer_provenance_actor_classes() -> None:
    """Provenance actor classes should override legacy per-forecast hints."""
    batch = ForecastBatch(
        provenance=_provenance(
            actor_mask=[True, False],
            actor_classes={"ped_1": "bicycle", "ped_2": "pedestrian"},
        ),
        forecasts=[
            ActorForecast(
                actor_id="ped_1",
                deterministic=[[0.0, 0.0], [1.0, 0.0]],
                uncertainty_metadata={"actor_class": "pedestrian"},
            )
        ],
    )
    truth = {"ped_1": [[0.0, 0.0], [1.0, 0.0]]}

    report = evaluate_forecast_batch(batch, truth)

    assert _aggregate(report, "mean_ade", 1.0, "bicycle")["value"] == 0.0
    with pytest.raises(AssertionError, match="missing aggregate"):
        _aggregate(report, "mean_ade", 1.0, "pedestrian")


def test_forecast_metrics_report_actor_class_denominators_and_caveats() -> None:
    """Mixed fast-agent traces should expose separate active and excluded denominators."""
    batch = ForecastBatch(
        provenance=_provenance(
            actor_ids=["ped_1", "bike_1", "bike_2"],
            actor_mask=[True, True, False],
            actor_mask_metadata={
                "semantics": "true means included",
                "missing_actor_reasons": {"bike_2": "outside forecast crop"},
            },
            actor_classes={
                "ped_1": "pedestrian",
                "bike_1": "bicycle",
                "bike_2": "bicycle",
            },
        ),
        forecasts=[
            ActorForecast(actor_id="ped_1", deterministic=[[0.0, 0.0], [1.0, 0.0]]),
            ActorForecast(actor_id="bike_1", deterministic=[[0.0, 0.0], [3.0, 0.0]]),
        ],
    )
    truth = {"ped_1": [[0.0, 0.0], [1.0, 0.0]], "bike_1": [[0.0, 0.0], [1.0, 0.0]]}

    report = evaluate_forecast_batch(batch, truth)

    denominators = {row["actor_class"]: row for row in report["actor_class_denominators"]}
    assert denominators["pedestrian"]["active_actor_count"] == 1
    assert denominators["pedestrian"]["excluded_actor_count"] == 0
    assert denominators["bicycle"]["active_actor_count"] == 1
    assert denominators["bicycle"]["excluded_actor_count"] == 1
    assert "fast dynamic actor" in denominators["bicycle"]["caveat"]
    assert report["denominator_health"]["by_actor_class"] == report["actor_class_denominators"]
    assert _aggregate(report, "mean_ade", 1.0, "pedestrian")["value"] == 0.0
    assert _aggregate(report, "mean_ade", 1.0, "bicycle")["value"] == 2.0


def test_forecast_metrics_fast_bicycle_fixture_smoke_actor_class_report() -> None:
    """The #2727 fast-bicycle fixture should flow into bicycle metric denominators."""
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "configs/scenarios/single/issue_2727_fast_bicycle_dynamic_actor.yaml"
    )
    scenario = yaml.safe_load(fixture_path.read_text())["scenarios"][0]
    actor = scenario["single_pedestrians"][0]
    actor_class = actor["metadata"]["fast_bicycle_actor"]["actor_type"]

    batch = ForecastBatch(
        provenance=_provenance(
            scenario_id=scenario["name"],
            actor_ids=[actor["id"]],
            actor_mask=[True],
            actor_mask_metadata={"semantics": "true means included"},
            actor_classes={actor["id"]: actor_class},
        ),
        forecasts=[ActorForecast(actor_id=actor["id"], deterministic=[[0.0, 0.0], [1.0, 0.0]])],
        metadata={"scenario_family": scenario["metadata"]["behavior"]},
    )
    truth = {actor["id"]: [[0.0, 0.0], [1.0, 0.0]]}

    report = evaluate_forecast_batch(batch, truth)
    markdown = format_forecast_metrics_markdown(report)

    assert report["actor_class_denominators"] == [
        {
            "actor_class": "bicycle",
            "active_actor_count": 1,
            "excluded_actor_count": 0,
            "horizons_s": [0.5, 1.0],
            "dt_s": 0.5,
            "caveat": (
                "fast dynamic actor forecast denominator; compare only with matching "
                "actor class, horizon, and dt_s settings"
            ),
        }
    ]
    assert _aggregate(report, "mean_ade", 1.0, "bicycle")["denominator"] == 1
    assert "bicycle(active=1, excluded=0)" in markdown


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
    assert report["actor_class_denominators"] == [
        {
            "actor_class": "child",
            "active_actor_count": 1,
            "excluded_actor_count": 0,
            "horizons_s": [0.5, 1.0],
            "dt_s": 0.5,
            "caveat": (
                "fast dynamic actor forecast denominator; compare only with matching "
                "actor class, horizon, and dt_s settings"
            ),
        },
        {
            "actor_class": "pedestrian",
            "active_actor_count": 0,
            "excluded_actor_count": 1,
            "horizons_s": [0.5, 1.0],
            "dt_s": 0.5,
            "caveat": "pedestrian forecast denominator",
        },
    ]


def test_forecast_metrics_apply_batch_actor_class_default_to_exclusions() -> None:
    """Batch actor-class defaults should classify active and excluded actors consistently."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[ActorForecast(actor_id="ped_1", deterministic=[[0.0, 0.0], [1.0, 0.0]])],
        metadata={"actor_classes": {"actor_class": "bicycle"}},
    )
    truth = {"ped_1": [[0.0, 0.0], [1.0, 0.0]]}

    report = evaluate_forecast_batch(batch, truth)

    assert _aggregate(report, "mean_ade", 1.0, "bicycle")["value"] == 0.0
    assert report["actor_class_denominators"] == [
        {
            "actor_class": "bicycle",
            "active_actor_count": 1,
            "excluded_actor_count": 1,
            "horizons_s": [0.5, 1.0],
            "dt_s": 0.5,
            "caveat": (
                "fast dynamic actor forecast denominator; compare only with matching "
                "actor class, horizon, and dt_s settings"
            ),
        }
    ]


def test_forecast_metrics_ignore_none_actor_class_metadata() -> None:
    """Explicit None metadata should not become a literal actor class."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[
            ActorForecast(
                actor_id="ped_1",
                deterministic=[[0.0, 0.0], [1.0, 0.0]],
                uncertainty_metadata={"ped_1": None},
            )
        ],
        metadata={"actor_classes": {"ped_1": None}},
    )
    truth = {"ped_1": [[0.0, 0.0], [1.0, 0.0]]}

    report = evaluate_forecast_batch(batch, truth)

    assert _aggregate(report, "mean_ade", 1.0, "pedestrian")["value"] == 0.0


def test_forecast_metrics_reject_misaligned_ground_truth() -> None:
    """Ground-truth trajectories must align with ForecastBatch horizons."""
    batch = ForecastBatch(
        provenance=_provenance(actor_mask=[True, False]),
        forecasts=[ActorForecast(actor_id="ped_1", deterministic=[[0.0, 0.0], [1.0, 0.0]])],
    )

    with pytest.raises(ValueError, match="ground_truth"):
        evaluate_forecast_batch(batch, {"ped_1": [[0.0, 0.0]]})
