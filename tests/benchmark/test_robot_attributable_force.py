"""Robot-attributable force law, reductions and pre-integration custody."""

from dataclasses import asdict

import numpy as np
import pytest
from pysocialforce.config import SocialForceConfig
from pysocialforce.forces import social_force_ped_ped

from robot_sf.benchmark.metrics import (
    EpisodeData,
    recompute_robot_ped_forces,
    robot_force_metrics,
    robot_force_reductions,
    robot_force_reference,
)

CFG = {
    "prf_active": True,
    "prf_multiplier": 10.0,
    "prf_activation_m": 2.0,
    "prf_robot_radius_m": 1.0,
    "prf_ped_radius_m": 0.35,
}


def _data(peds):
    peds = np.asarray(peds, dtype=float)
    zeros = np.zeros((len(peds), 2))
    return EpisodeData(zeros, zeros, zeros, peds, np.zeros_like(peds), np.ones(2), 0.1)


def test_law_direction_cutoff_and_multiplier():
    data = _data([[[d, 0] for d in (1.35, 2, 3.349, 3.351)]])
    forces = recompute_robot_ped_forces(data, CFG)
    np.testing.assert_allclose(forces[0, :3, 0], 10 / np.array([1.35, 2, 3.349]) ** 3, atol=1e-12)
    assert np.all(forces[..., 1] == 0)
    assert forces[0, 3, 0] == 0
    np.testing.assert_allclose(
        recompute_robot_ped_forces(data, {**CFG, "prf_multiplier": 20}), 2 * forces
    )
    assert not recompute_robot_ped_forces(data, {**CFG, "prf_active": False}).any()


def test_reductions_ignore_despawned_rows():
    forces = np.array([[[3, 4], [np.nan, np.nan]], [[0, 0], [0, 2]]])
    result = robot_force_reductions(forces, dt=0.5, reference=3)
    assert result == {
        "robot_force_impulse_total": 3.5,
        "robot_force_impulse_per_exposed_ped": 1.75,
        "robot_force_peak": 5,
        "robot_force_mean_active": 3.5,
        "robot_force_time_above_ref_s": 0.5,
        "robot_force_exposed_ped_count": 2,
    }


def test_empty_exposure_denominators():
    result = robot_force_reductions(np.zeros((3, 0, 2)), dt=0.1, reference=3)
    assert result["robot_force_impulse_total"] == 0
    assert result["robot_force_peak"] == 0
    assert np.isnan(result["robot_force_mean_active"])
    assert np.isnan(result["robot_force_impulse_per_exposed_ped"])


def test_reference_matches_actual_kernel():
    cfg = asdict(SocialForceConfig())
    expected = cfg["factor"] * np.linalg.norm(
        social_force_ped_ped(
            np.array([0.7, 0.0]),
            np.array([1.0, 0.0]),
            cfg["n"],
            cfg["n_prime"],
            cfg["lambda_importance"],
            cfg["gamma"],
        )
    )
    assert robot_force_reference(cfg, 0.35) == pytest.approx(expected, abs=1e-12)
    assert expected == pytest.approx(5.1 * np.sqrt(2) * np.exp(-0.7 / (1.05 + 1e-8)), abs=1e-12)


def test_absent_component_has_no_output():
    assert robot_force_metrics(_data(np.zeros((1, 0, 2)))) == {}


def test_posthoc_configs_opt_in_to_recomputation_with_estimate_provenance():
    data = _data([[[2.0, 0.0]], [[2.0, 0.0]]])
    data.robot_force_config = CFG
    data.social_force_config = asdict(SocialForceConfig())
    result = robot_force_metrics(data)
    reference = robot_force_reference(data.social_force_config, CFG["prf_ped_radius_m"])
    expected = robot_force_reductions(
        recompute_robot_ped_forces(data, CFG), dt=data.dt, reference=reference
    )
    assert {key: result[key] for key in expected} == expected
    assert result["robot_force_impulse_total"] == pytest.approx(0.25)
    assert result["robot_force_metadata"]["source"] == "posthoc_recomputed"
    assert result["robot_force_metadata"]["sample_timing"] == (
        "caller_supplied_positions_may_be_post_integration"
    )
    assert data.robot_ped_forces is None


@pytest.mark.parametrize("missing", ["robot", "social"])
def test_posthoc_requires_both_explicit_configs(missing):
    data = _data([[[2.0, 0.0]]])
    data.robot_force_config = None if missing == "robot" else CFG
    data.social_force_config = None if missing == "social" else asdict(SocialForceConfig())
    with pytest.raises(ValueError, match="require robot and social force configuration"):
        robot_force_metrics(data)


def test_recorded_force_path_keeps_metadata_and_ignores_snapshot_geometry():
    import json
    from pathlib import Path

    from jsonschema import validate

    data = _data([[[0.0, 0.0]]])  # Recomputing this singular snapshot would raise.
    data.robot_force_config = CFG
    data.social_force_config = asdict(SocialForceConfig())
    data.robot_ped_forces = np.array([[[1.25, 0.0]]])
    result = robot_force_metrics(data)
    assert result["robot_force_peak"] == 1.25
    assert result["robot_force_metadata"]["sample_timing"] == "pre_integration"
    assert result["robot_force_metadata"]["source"] == ("recorded_robot_pedestrian_social_force")
    schema = json.loads(
        (
            Path(__file__).parents[2] / "robot_sf/benchmark/schemas/episode.schema.v1.json"
        ).read_text()
    )
    validate(result, schema["properties"]["metrics"])
    json.dumps(result, allow_nan=False)


def test_simulator_capture_recomputes_and_preserves_dynamics():
    from tests.sim.test_goal_force_instrumentation import _build_simulator

    sim = _build_simulator(oracle_enabled=False, robot_force_enabled=True)
    state = sim.pysf_state.pysf_states()
    state[0, :2] = np.asarray(sim.robot_pos[0]) + np.array([2.0, 0.0])
    state[0, 4:6] = state[0, :2] + np.array([6.0, 0.0])
    sim.step_once([(0.0, 0.0)])
    assert np.linalg.norm(sim.last_robot_ped_forces) > 0
    inputs = sim.last_robot_force_inputs
    data = _data([inputs["peds_pos"]])
    component = inputs["components"][0]
    data.robot_pos = np.array([component["robot_pos"]])
    np.testing.assert_allclose(
        recompute_robot_ped_forces(data, component)[0], sim.last_robot_ped_forces, atol=1e-9
    )
    captured = sum(
        np.asarray(f.last_forces)
        for f in sim.pysf_sim.forces
        if getattr(f, "component_type", None) == "pedestrian_robot"
    )
    np.testing.assert_array_equal(captured, sim.last_robot_ped_forces)


def test_pair_counterfactual_matches_kernel_and_is_not_simulated():
    from robot_sf.benchmark.metrics import robot_force_pp_equivalent

    data = _data([[[2.0, 0.0]], [[1.9, 0.0]]])
    data.robot_force_config = CFG
    data.social_force_config = asdict(SocialForceConfig())
    data.robot_force_samples = [
        {"peds_pos": frame.tolist(), "components": [{**CFG, "robot_pos": [0.0, 0.0]}]}
        for frame in data.peds_pos
    ]
    result = robot_force_pp_equivalent(data)
    cfg = data.social_force_config
    for t, distance in enumerate((1.35, 1.25)):
        expected = cfg["factor"] * np.asarray(
            social_force_ped_ped(
                np.array([distance, 0.0]),
                np.array([1.0, 0.0]),
                cfg["n"],
                cfg["n_prime"],
                cfg["lambda_importance"],
                cfg["gamma"],
            )
        )
        np.testing.assert_allclose(result[t, 0], expected, atol=1e-12)
    assert not np.allclose(result, recompute_robot_ped_forces(data, CFG))


def test_recording_preserves_fixed_seed_trajectory_and_legacy_metric_bytes(monkeypatch):
    import json
    import random

    from robot_sf.benchmark.metrics import compute_all_metrics
    from robot_sf.sim.simulator import Simulator
    from tests.sim.test_goal_force_instrumentation import _build_simulator

    random.seed(9666)
    np.random.seed(9666)
    baseline = _build_simulator(oracle_enabled=False, robot_force_enabled=True)
    random.seed(9666)
    np.random.seed(9666)
    recorded = _build_simulator(oracle_enabled=False, robot_force_enabled=True)
    captures = []
    for sim in (baseline, recorded):
        robots, peds, forces = [], [], []
        with monkeypatch.context() as context:
            if sim is baseline:
                context.setattr(Simulator, "_capture_robot_ped_forces", lambda self: None)
            for _ in range(5):
                sim.step_once([(0.1, 0.0)])
                robots.append(np.array(sim.robot_pos[0], copy=True))
                peds.append(np.array(sim.ped_pos, copy=True))
                forces.append(np.array(sim.last_ped_forces, copy=True))
        data = _data(peds)
        data.robot_pos = np.asarray(robots)
        data.ped_forces = np.asarray(forces)
        captures.append((data, json.dumps(compute_all_metrics(data, horizon=5), sort_keys=True)))
    assert captures[0][1] == captures[1][1]
    np.testing.assert_array_equal(captures[0][0].peds_pos, captures[1][0].peds_pos)
    np.testing.assert_array_equal(captures[0][0].robot_pos, captures[1][0].robot_pos)


def test_registry_and_schema_declare_units():
    import json
    from pathlib import Path

    from robot_sf.benchmark.metric_layers import resolve_metric_source_binding

    schema = json.loads(
        (
            Path(__file__).parents[2] / "robot_sf/benchmark/schemas/episode.schema.v1.json"
        ).read_text()
    )
    for name in robot_force_reductions(np.zeros((1, 0, 2)), dt=0.1, reference=1):
        assert name in schema["properties"]["metrics"]["properties"]
        assert resolve_metric_source_binding(name).unit_status == "available"


def test_new_undefined_values_serialize_as_null_without_legacy_drift():
    import json

    from robot_sf.benchmark.metrics import post_process_metrics

    raw = robot_force_reductions(np.zeros((2, 0, 2)), dt=0.1, reference=1)
    raw["force_q50"] = float("nan")
    result = post_process_metrics(raw, snqi_weights=None, snqi_baseline=None)
    assert result["robot_force_mean_active"] is None
    assert result["robot_force_impulse_per_exposed_ped"] is None
    assert "force_quantiles" not in result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("ped_count", [0, 1])
def test_posthoc_zero_exposure_serializes_against_metric_schema(ped_count):
    import json
    from pathlib import Path

    from jsonschema import validate

    from robot_sf.benchmark.metrics import post_process_metrics

    data = _data(np.full((2, ped_count, 2), 10.0))
    data.robot_force_config = CFG
    data.social_force_config = asdict(SocialForceConfig())
    result = post_process_metrics(robot_force_metrics(data), snqi_weights=None, snqi_baseline=None)
    assert result["robot_force_exposed_ped_count"] == 0
    assert result["robot_force_mean_active"] is None
    assert result["robot_force_impulse_per_exposed_ped"] is None
    assert result["robot_force_metadata"]["source"] == "posthoc_recomputed"
    schema = json.loads(
        (
            Path(__file__).parents[2] / "robot_sf/benchmark/schemas/episode.schema.v1.json"
        ).read_text()
    )
    validate(result, schema["properties"]["metrics"])
    json.dumps(result, allow_nan=False)


def test_recorded_robot_component_subtracts_from_registered_total():
    from tests.sim.test_goal_force_instrumentation import _build_simulator

    sim = _build_simulator(oracle_enabled=True, robot_force_enabled=True)
    sim.step_once([(0.0, 0.0)])
    records = sim.last_oracle_transition_traces[0].force_components.component_records
    remaining = np.sum(
        [record.force_xy for record in records if record.component_type != "pedestrian_robot"],
        axis=0,
    )
    np.testing.assert_allclose(
        sim.last_ped_forces[0] - sim.last_robot_ped_forces[0], remaining, atol=1e-12
    )


def test_disabled_robot_force_capture_is_zero():
    from tests.sim.test_goal_force_instrumentation import _build_simulator

    sim = _build_simulator(oracle_enabled=False, robot_force_enabled=False)
    sim.step_once([(0.0, 0.0)])
    assert not sim.last_robot_ped_forces.any()


@pytest.mark.parametrize("persist", [False, True])
def test_runner_persists_force_series_only_for_explicit_trace(persist):
    import json

    from robot_sf.benchmark.map_runner.map_runner_episode import _compute_post_loop_metrics
    from robot_sf.gym_env.unified_config import RobotSimulationConfig

    samples = [
        {
            "peds_pos": [[2.0, 0.0]],
            "forces": [[1.25, 0.0]],
            "components": [{**CFG, "robot_pos": [0.0, 0.0]}],
            "social_force_config": asdict(SocialForceConfig()),
            "ped_radius_m": 0.35,
        }
        for _ in range(2)
    ]
    result = _compute_post_loop_metrics(
        robot_positions=[np.zeros(2), np.zeros(2)],
        robot_headings=[0.0, 0.0],
        ped_positions=[np.array([[2.0, 0.0]])] * 2,
        ped_forces=[np.array([[1.25, 0.0]])] * 2,
        robot_force_samples=samples,
        persist_robot_force_samples=persist,
        visibility_trace=[None, None],
        track_confidence_trace=[None, None],
        visibility_evidence_statuses=[],
        visibility_evidence_reasons=[],
        reached_goal_step=None,
        collision_seen=False,
        ped_collision_seen=False,
        obstacle_collision_seen=False,
        robot_collision_seen=False,
        map_def=None,
        goal_vec=np.array([10.0, 0.0]),
        scenario={},
        config=RobotSimulationConfig(),
        horizon_val=2,
        record_forces=True,
        experimental_ped_impact=False,
        ped_impact_radius_m=2.0,
        ped_impact_window_steps=5,
    )
    assert result.metrics_raw["robot_force_peak"] == 1.25
    assert "robot_force_metadata" in result.metrics_raw
    assert ("robot_force_samples" in result.metrics_raw) is persist
    if persist:
        assert json.loads(json.dumps(result.metrics_raw["robot_force_samples"])) == samples
