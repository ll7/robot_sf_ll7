"""Smoke tests for the issue #2756 occluded-emergence trace fixture."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

from robot_sf.benchmark.observation_perturbation import (
    NOISE_PROFILE_DELAYED_OBSERVATION,
    NOISE_PROFILE_OCCLUSION_MASK,
    ObservationPerturbationSpec,
    ObservationPerturbationState,
    perturb_ground_truth,
)
from robot_sf.benchmark.pedestrian_forecast import compute_batch_forecast_metrics

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
    "occluded_emergence_episode_0000.json"
)
_SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/run_cv_forecast_eval.py"


def _load_trace() -> dict:
    """Load the occluded-emergence fixture."""
    with open(FIXTURE_PATH) as fh:
        return json.load(fh)


def _load_eval_script():
    """Load the CV forecast evaluator as a module."""
    spec = importlib.util.spec_from_file_location("run_cv_forecast_eval_issue_2756", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_cv_forecast_eval_issue_2756"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_occluded_emergence_fixture_records_visibility_and_conflict_fields() -> None:
    """Fixture separates ground truth, observed state, and occlusion metadata."""
    trace = _load_trace()

    assert trace["schema_version"] == "simulation_trace_export.v1"
    assert trace["evidence_boundary"] == "smoke_diagnostic_only_not_benchmark_evidence"
    assert trace["occlusion"]["first_visible_step"] == 5
    assert trace["occlusion"]["conflict_time_s"] == 1.0
    assert len(trace["frames"]) == 16

    first_visible_steps = [frame["step"] for frame in trace["frames"] if frame["first_visible"]]
    assert first_visible_steps == [5]

    for frame in trace["frames"][:5]:
        assert frame["pedestrians"], "ground truth must retain the hidden pedestrian"
        assert frame["observed_pedestrians"] == []
        assert frame["occlusion_status"]["emerging_ped"] == "occluded"

    for frame in trace["frames"][5:]:
        assert len(frame["observed_pedestrians"]) == 1
        assert frame["occlusion_status"]["emerging_ped"] == "visible"

    stop_feasible_steps = [
        frame["step"] for frame in trace["frames"] if frame["conflict_timing"]["stop_feasible"]
    ]
    yield_feasible_steps = [
        frame["step"] for frame in trace["frames"] if frame["conflict_timing"]["yield_feasible"]
    ]
    assert stop_feasible_steps == list(range(7))
    assert yield_feasible_steps == list(range(9))


def test_occlusion_mask_reproduces_previsibility_missing_observation() -> None:
    """Observation perturbation can express the fixture's previsibility occlusion."""
    trace = _load_trace()
    hidden_frame = trace["frames"][0]
    ped = hidden_frame["pedestrians"][0]
    positions = np.array([ped["position"]], dtype=float)
    velocities = np.array([ped["velocity"]], dtype=float)

    result = perturb_ground_truth(
        positions,
        velocities,
        ["emerging_ped"],
        spec=ObservationPerturbationSpec(occlusion_mask=np.array([True])),
        step=hidden_frame["step"],
    )

    assert result["metadata"]["noise_profile"] == NOISE_PROFILE_OCCLUSION_MASK
    assert result["metadata"]["occluded_actor_count"] == 1
    np.testing.assert_array_equal(result["ground_truth"]["positions"], positions)
    np.testing.assert_array_equal(result["observed"]["positions"], np.array([[0.0, 0.0]]))


def test_delayed_observation_lags_after_first_visible_frame() -> None:
    """Observation perturbation can express delayed detection after first visibility."""
    trace = _load_trace()
    visible_frames = trace["frames"][5:8]
    spec = ObservationPerturbationSpec(delay_steps=2)
    state = ObservationPerturbationState(delay_steps=2)
    observed_positions = []

    for frame in visible_frames:
        ped = frame["pedestrians"][0]
        result = perturb_ground_truth(
            np.array([ped["position"]], dtype=float),
            np.array([ped["velocity"]], dtype=float),
            ["emerging_ped"],
            spec=spec,
            state=state,
            step=frame["step"],
        )
        observed_positions.append(result["observed"]["positions"][0].tolist())
        assert result["metadata"]["noise_profile"] == NOISE_PROFILE_DELAYED_OBSERVATION

    assert observed_positions[0] == visible_frames[0]["pedestrians"][0]["position"]
    assert observed_positions[1] == visible_frames[1]["pedestrians"][0]["position"]
    assert observed_positions[2] == visible_frames[0]["pedestrians"][0]["position"]


def test_occluded_emergence_fixture_feeds_forecast_metrics() -> None:
    """The fixture has nonzero motion and enough frames for forecast diagnostics."""
    trace = _load_trace()
    steps = [
        {
            "step": frame["step"],
            "time_s": frame["time_s"],
            "robot": frame["robot"],
            "pedestrians": frame["pedestrians"],
        }
        for frame in trace["frames"]
    ]

    metrics = compute_batch_forecast_metrics(steps, horizons_s=[0.5, 1.0], dt_s=0.1)

    assert metrics["forecast_evaluable_samples"] > 0
    assert metrics["count_ade_0.5s"] > 0
    assert metrics["count_ade_1s"] > 0
    assert metrics["mean_miss_rate_0.5s"] >= 0.0


def test_cv_forecast_eval_registers_occluded_emergence_candidate() -> None:
    """The CV forecast evaluator now treats occluded emergence as evaluable."""
    mod = _load_eval_script()

    candidate = next(
        item for item in mod.TRACE_CANDIDATES if item["family"] == "occluded_emergence"
    )
    assert all(item["family"] != "occluded_emergence" for item in mod.MISSING_FAMILIES)

    result = mod.evaluate_single_trace(candidate)

    assert result["status"] == "evaluated"
    assert result["metrics"]["forecast_evaluable_samples"] > 0
    assert result["trace_path"].endswith("occluded_emergence_episode_0000.json")
