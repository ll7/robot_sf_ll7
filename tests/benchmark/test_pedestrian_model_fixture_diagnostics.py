"""Tests for the issue #3481 pedestrian-model fixture diagnostics harness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robot_sf.benchmark.pedestrian_model_fixture_diagnostics import (
    HSFM_ANISOTROPIC_FOV_V1,
    HSFM_TTC_PREDICTIVE_V1,
    PED_MODEL_FIXTURE_REPORT_SCHEMA_VERSION,
    SOCIAL_FORCE_DEFAULT,
    PedestrianModelFixtureRunConfig,
    PedestrianModelFixtureSpec,
    PedestrianModelFixtureTrace,
    build_pedestrian_model_fixture_scenarios,
    compute_fixture_metrics,
    run_pedestrian_model_fixture_diagnostics,
    run_pedestrian_model_fixture_trace,
    write_pedestrian_model_fixture_report,
)


def test_fixture_harness_emits_expected_scenarios_and_metric_keys() -> None:
    """Short local run covers both issue-requested fixtures and metric families."""

    report = run_pedestrian_model_fixture_diagnostics(
        config=PedestrianModelFixtureRunConfig(duration_s=2.0, dt_s=0.1),
        pedestrian_models=(SOCIAL_FORCE_DEFAULT, HSFM_ANISOTROPIC_FOV_V1),
    )

    assert report["schema_version"] == PED_MODEL_FIXTURE_REPORT_SCHEMA_VERSION
    assert set(report["scenario_ids"]) == {
        "shared_throat_sliding",
        "shared_throat_congestion",
        "narrow_passage_lateral_sliding",
        "bottleneck_freeze_deadlock",
    }
    assert report["status"]["robot_inserted"] is False
    assert report["status"]["thresholds_applied"] is True
    seen_models = {run["pedestrian_model"] for run in report["runs"]}
    assert seen_models == {SOCIAL_FORCE_DEFAULT, HSFM_ANISOTROPIC_FOV_V1}
    for run in report["runs"]:
        metrics = run["metrics"]
        for key in (
            "minimum_pairwise_distance_m",
            "mean_max_lateral_displacement_m",
            "mean_speed_mps",
            "entered_interaction_zone",
            "max_consecutive_interaction_zone_slow_steps",
            "interaction_zone_slow_detected",
            "diagnostic_thresholds",
        ):
            assert key in metrics
        assert isinstance(metrics["entered_interaction_zone"], bool)
        assert metrics["finite_positions"] is True
        assert metrics["finite_velocities"] is True


def test_geometric_fixtures_emit_diagnostic_threshold_checks() -> None:
    """Issue #3481 geometric fixtures expose local thresholded diagnostics."""

    report = run_pedestrian_model_fixture_diagnostics(
        config=PedestrianModelFixtureRunConfig(duration_s=2.0, dt_s=0.1),
        scenarios=("narrow_passage_lateral_sliding", "bottleneck_freeze_deadlock"),
        pedestrian_models=(SOCIAL_FORCE_DEFAULT,),
    )

    by_scenario = {run["scenario_id"]: run for run in report["runs"]}
    sliding_thresholds = by_scenario["narrow_passage_lateral_sliding"]["metrics"][
        "diagnostic_thresholds"
    ]
    assert "mean_max_lateral_displacement_m" in sliding_thresholds
    assert sliding_thresholds["mean_max_lateral_displacement_m"]["threshold"] == pytest.approx(0.04)

    bottleneck_thresholds = by_scenario["bottleneck_freeze_deadlock"]["metrics"][
        "diagnostic_thresholds"
    ]
    assert "max_consecutive_interaction_zone_slow_steps" in bottleneck_thresholds
    assert bottleneck_thresholds["max_consecutive_interaction_zone_slow_steps"][
        "threshold"
    ] == pytest.approx(2.0)


def test_fixture_trace_is_deterministic_for_fixed_seed() -> None:
    """The same scenario/model/seed pair reproduces identical local traces."""

    scenario = build_pedestrian_model_fixture_scenarios()["shared_throat_sliding"]
    config = PedestrianModelFixtureRunConfig(duration_s=0.5, dt_s=0.1, seed=3481)

    first = run_pedestrian_model_fixture_trace(
        scenario,
        pedestrian_model=HSFM_ANISOTROPIC_FOV_V1,
        config=config,
    )
    second = run_pedestrian_model_fixture_trace(
        scenario,
        pedestrian_model=HSFM_ANISOTROPIC_FOV_V1,
        config=config,
    )

    np.testing.assert_allclose(first.positions, second.positions)
    np.testing.assert_allclose(first.velocities, second.velocities)


def test_compute_fixture_metrics_detects_consecutive_interaction_zone_slow_proxy() -> None:
    """Metric helper marks a sustained interaction-zone slowdown proxy."""

    spec = PedestrianModelFixtureSpec(
        scenario_id="unit",
        map_def=build_pedestrian_model_fixture_scenarios()["shared_throat_sliding"].map_def,
        single_pedestrians=(),
        interaction_zone_center=(0.5, 0.0),
        interaction_zone_radius_m=1.0,
        interaction_zone_min_pedestrians=2,
    )
    trace = PedestrianModelFixtureTrace(
        scenario_id="unit",
        pedestrian_model="unit",
        seed=1,
        dt_s=0.1,
        duration_s=0.3,
        positions=np.asarray(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.2, 0.3], [1.0, -0.3]],
                [[0.2, 0.3], [1.0, -0.3]],
                [[0.2, 0.3], [1.0, -0.3]],
            ],
            dtype=float,
        ),
        velocities=np.zeros((4, 2, 2), dtype=float),
    )

    metrics = compute_fixture_metrics(
        trace,
        spec,
        freeze_speed_threshold_mps=0.05,
        freeze_window_steps=3,
    )

    assert metrics["entered_interaction_zone"] is True
    assert metrics["interaction_zone_slow_detected"] is True
    assert metrics["max_consecutive_interaction_zone_slow_steps"] == 4
    assert metrics["mean_max_lateral_displacement_m"] == pytest.approx(0.3)


def test_report_writer_emits_json_and_markdown(tmp_path) -> None:
    """Report writer persists compact JSON and Markdown artifacts."""

    report = run_pedestrian_model_fixture_diagnostics(
        config=PedestrianModelFixtureRunConfig(duration_s=0.4, dt_s=0.1),
        scenarios=("shared_throat_sliding",),
        pedestrian_models=(SOCIAL_FORCE_DEFAULT,),
    )
    written = write_pedestrian_model_fixture_report(report, tmp_path)

    loaded = json.loads(written["summary_json"].read_text(encoding="utf-8"))
    assert loaded["schema_version"] == PED_MODEL_FIXTURE_REPORT_SCHEMA_VERSION
    markdown = written["summary_md"].read_text(encoding="utf-8")
    assert "Pedestrian model fixture diagnostics" in markdown
    assert "shared_throat_sliding" in markdown
    assert SOCIAL_FORCE_DEFAULT in markdown


def test_default_bottleneck_run_exercises_slowdown_proxy_for_predictive_model() -> None:
    """The shipped default runtime is long enough to surface the bottleneck slowdown proxy."""

    report = run_pedestrian_model_fixture_diagnostics(
        scenarios=("shared_throat_congestion",),
        pedestrian_models=(HSFM_TTC_PREDICTIVE_V1,),
    )

    metrics = report["runs"][0]["metrics"]
    assert metrics["entered_interaction_zone"] is True
    assert metrics["interaction_zone_slow_steps"] > 0
    assert metrics["interaction_zone_slow_detected"] is True


def test_fixture_config_and_selection_fail_closed() -> None:
    """Bad scalar controls, scenarios, and model selectors raise clear errors."""

    with pytest.raises(ValueError, match="duration_s"):
        PedestrianModelFixtureRunConfig(duration_s=0.0)
    with pytest.raises(ValueError, match="freeze_window_steps"):
        PedestrianModelFixtureRunConfig(freeze_window_steps=0)
    with pytest.raises(ValueError, match="Unknown pedestrian-model fixture scenario"):
        run_pedestrian_model_fixture_diagnostics(scenarios=("unknown_fixture",))
    with pytest.raises(ValueError, match="Unsupported pedestrian_model"):
        run_pedestrian_model_fixture_diagnostics(pedestrian_models=("unknown_model",))
