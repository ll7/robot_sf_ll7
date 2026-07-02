"""Tests opt-in benchmark runtime CBF safety-filter binding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark import cbf_safety_filter_runtime, map_runner_episode
from robot_sf.benchmark.cbf_safety_filter_runtime import (
    CBF_COLLISION_CONE_ARM,
    CBF_DYNAMIC_PARABOLIC_V1_ARM,
    CBF_OFF_ARM,
    CBF_SAFETY_FILTER_EPISODE_SUMMARY_SCHEMA,
    CBF_SAFETY_FILTER_RUNTIME_STEP_SCHEMA,
    apply_runtime_cbf_safety_filter,
    compute_cbf_observation_from_env,
    ineligible_cbf_safety_filter_step_record,
    runtime_config_from_mapping,
    summarize_cbf_safety_filter_trace,
)
from robot_sf.benchmark.event_ledger import build_event_ledger
from robot_sf.benchmark.map_runner_identity import _scenario_identity_payload


class _Robot:
    theta = np.array([0.0], dtype=float)


class _Simulator:
    def __init__(self) -> None:
        self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
        self.ped_pos = np.array([[0.2, 0.0]], dtype=float)
        self.robots = [_Robot()]


class _Env:
    def __init__(self) -> None:
        self.simulator = _Simulator()


class _EpisodeSim(_Simulator):
    def __init__(self) -> None:
        super().__init__()
        self.goal_pos = [np.array([1.0, 0.0], dtype=float)]
        self.map_def = SimpleNamespace(obstacles=[], bounds=(0.0, 0.0, 1.0, 1.0))
        self.robot_vel = [np.array([0.0, 0.0], dtype=float)]

    def get_pedestrian_forces(self) -> np.ndarray:
        return np.zeros((1, 2), dtype=float)


class _EpisodeEnv:
    def __init__(self) -> None:
        self.simulator = _EpisodeSim()
        self.action_space = None

    def reset(self, seed=None):
        return (
            {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 0.0]},
                "pedestrians": {"positions": self.simulator.ped_pos},
            },
            {},
        )

    def step(self, _action):
        return self.reset()[0], 0.0, True, False, {"meta": {"is_route_complete": True}}

    def close(self) -> None:
        return None


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        sim_config=SimpleNamespace(
            time_per_step_in_secs=0.1,
            robot_radius=0.1,
            ped_radius=0.1,
        )
    )


def _patch_episode_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        map_runner_episode,
        "_build_env_config",
        lambda _scenario, scenario_path: _config(),
    )
    monkeypatch.setattr(
        map_runner_episode,
        "make_robot_env",
        lambda config, seed, debug: _EpisodeEnv(),
    )
    monkeypatch.setattr(map_runner_episode, "sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(map_runner_episode, "compute_shortest_path_length", lambda *args: 1.0)
    monkeypatch.setattr(
        map_runner_episode,
        "compute_all_metrics",
        lambda *args, **kwargs: {"success": 1.0, "collisions": 0.0},
    )
    monkeypatch.setattr(
        map_runner_episode,
        "post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )


def _run_episode_with_policy(monkeypatch, policy_builder, *, cbf_safety_filter):
    _patch_episode_runtime(monkeypatch)
    return map_runner_episode.run_map_episode(
        {"name": "cbf-runtime", "simulation_config": {"max_episode_steps": 1}},
        seed=1,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=Path(__file__),
        cbf_safety_filter=cbf_safety_filter,
        policy_builder=policy_builder,
    )


def test_runtime_config_disabled_by_default_preserves_off_state() -> None:
    """Missing runtime config keeps CBF filter disabled."""

    runtime = runtime_config_from_mapping(None)

    assert runtime.enabled is False
    assert runtime.arm_key == CBF_OFF_ARM


@pytest.mark.parametrize(
    "variant",
    [
        "collision_cone",
        "collision_cone_cbf_v1",
        "dynamic_parabolic",
        "dynamic_parabolic_cbf_v1",
    ],
)
def test_runtime_config_off_arm_accepts_known_variant_aliases(variant: str) -> None:
    """Disabled runtime accepts the same known variant aliases as the planner builder."""

    runtime = runtime_config_from_mapping(
        {"enabled": False, "arm_key": CBF_OFF_ARM, "variant": variant}
    )

    assert runtime.variant == variant


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"enabled": True, "arm_key": CBF_OFF_ARM}, "enabled=True"),
        ({"enabled": False, "arm_key": CBF_COLLISION_CONE_ARM}, "enabled=False"),
        ({"enabled": True, "arm_key": CBF_COLLISION_CONE_ARM, "alpha": 2.0}, "predeclared"),
        ({"enabled": True, "arm_key": CBF_DYNAMIC_PARABOLIC_V1_ARM, "alpha": 2.0}, "predeclared"),
    ],
)
def test_runtime_config_fails_closed_for_arm_or_threshold_drift(
    payload: dict[str, object], match: str
) -> None:
    """Enabled CBF arm must match predeclared first-slice semantics."""

    with pytest.raises((ValueError, NotImplementedError), match=match):
        runtime_config_from_mapping(payload)


def test_runtime_config_rejects_malformed_mapping() -> None:
    """Runtime config parser fails closed on wrong type and unknown keys."""

    with pytest.raises(TypeError, match="mapping"):
        runtime_config_from_mapping(["not", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown"):
        runtime_config_from_mapping({"unexpected": True})
    with pytest.raises(ValueError, match="arm_key"):
        runtime_config_from_mapping({"enabled": True, "arm_key": "cbf_custom_on"})


def test_runtime_config_accepts_dynamic_parabolic_versioned_arm() -> None:
    """DPCBF uses its own versioned arm and variant in runtime identity."""

    runtime = runtime_config_from_mapping(
        {
            "enabled": True,
            "arm_key": CBF_DYNAMIC_PARABOLIC_V1_ARM,
            "variant": "dynamic_parabolic_cbf_v1",
        }
    )

    assert runtime.enabled is True
    assert runtime.arm_key == CBF_DYNAMIC_PARABOLIC_V1_ARM
    assert runtime.variant == "dynamic_parabolic_cbf_v1"


def test_cbf_on_emits_schema_tagged_intervention_record_and_summary() -> None:
    """A close pre-step pedestrian produces well-formed CBF evidence."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": CBF_COLLISION_CONE_ARM})

    corrected, record = apply_runtime_cbf_safety_filter(
        command=(1.0, 0.25),
        env=_Env(),
        config=_config(),
        runtime=runtime,
        previous_ped_positions=np.array([[0.3, 0.0]], dtype=float),
        step_idx=3,
    )

    assert corrected[0] < 1.0
    assert record["schema_version"] == CBF_SAFETY_FILTER_RUNTIME_STEP_SCHEMA
    assert record["arm_key"] == CBF_COLLISION_CONE_ARM
    assert record["enabled"] is True
    assert record["eligible_for_cbf_filter"] is True
    assert record["context_source"] == "simulator_state_pre_step"
    assert record["pedestrian_velocity_identity"] == "row_order_finite_difference_no_stable_ids"
    assert record["qp_status"] in {"filtered", "fallback_infeasible"}
    assert record["intervened"] is True
    assert record["qp_status"] != "pass_through"

    summary = summarize_cbf_safety_filter_trace([record], runtime=runtime)
    assert summary["schema_version"] == CBF_SAFETY_FILTER_EPISODE_SUMMARY_SCHEMA
    assert summary["intervened_step_count"] == 1
    assert summary["intervention_rate"] == 1.0
    assert summary["status_counts"][record["qp_status"]] == 1
    assert summary["fallback_rows_are_success_evidence"] is False


def test_dynamic_parabolic_arm_emits_runtime_record_and_summary() -> None:
    """DPCBF runtime arm emits distinct arm metadata and projection method."""

    runtime = runtime_config_from_mapping(
        {
            "enabled": True,
            "arm_key": CBF_DYNAMIC_PARABOLIC_V1_ARM,
            "variant": "dynamic_parabolic_cbf_v1",
        }
    )
    corrected, record = apply_runtime_cbf_safety_filter(
        command=(1.0, 0.25),
        env=_Env(),
        config=_config(),
        runtime=runtime,
        previous_ped_positions=np.array([[0.3, 0.0]], dtype=float),
        step_idx=3,
    )
    summary = summarize_cbf_safety_filter_trace([record], runtime=runtime)

    assert corrected[0] <= 1.0
    assert record["arm_key"] == CBF_DYNAMIC_PARABOLIC_V1_ARM
    assert record["variant"] == "dynamic_parabolic_cbf_v1"
    assert record["projection_method"] == "bounded_scalar_dpcbf_grid_refine_v1"
    assert summary["arm_key"] == CBF_DYNAMIC_PARABOLIC_V1_ARM
    assert summary["variant"] == "dynamic_parabolic_cbf_v1"
    assert summary["fallback_rows_are_success_evidence"] is False
    assert "Dynamic Parabolic CBF" in summary["claim_boundary"]


def test_apply_runtime_cbf_rejects_short_command() -> None:
    """Runtime CBF requires linear and angular command entries."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": CBF_COLLISION_CONE_ARM})

    with pytest.raises(TypeError, match="linear and angular"):
        apply_runtime_cbf_safety_filter(
            command=(1.0,),
            env=_Env(),
            config=_config(),
            runtime=runtime,
            previous_ped_positions=None,
            step_idx=0,
        )


def test_compute_cbf_observation_uses_pre_step_simulator_state() -> None:
    """Runtime context is sourced from simulator state, not planner observation."""

    observation, provenance = compute_cbf_observation_from_env(
        env=_Env(),
        config=_config(),
        previous_ped_positions=np.array([[0.3, 0.0]], dtype=float),
        dt=0.1,
    )

    assert provenance["context_source"] == "simulator_state_pre_step"
    assert provenance["obstacle_count"] == 1
    assert observation["agents"][0]["velocity"] == pytest.approx([-1.0, 0.0])


def test_compute_cbf_observation_accepts_robot_poses_heading() -> None:
    """Runtime heading parser supports the real map-runner simulator pose shape."""

    env = _Env()
    env.simulator.robot_poses = [((0.0, 0.0), 1.25)]
    env.simulator.robots = []

    observation, _provenance = compute_cbf_observation_from_env(
        env=env,
        config=_config(),
        previous_ped_positions=None,
        dt=0.1,
    )

    assert observation["robot"]["heading"] == pytest.approx(1.25)


@pytest.mark.parametrize(
    "ped_pos,match",
    [
        (np.array([1.0, 0.0, 2.0], dtype=float), "even-length"),
        (np.array([[1.0], [2.0]], dtype=float), "shaped"),
        (np.array([[np.inf, 0.0]], dtype=float), "finite"),
    ],
)
def test_compute_cbf_observation_rejects_malformed_pedestrians(
    ped_pos: np.ndarray,
    match: str,
) -> None:
    """Malformed simulator pedestrian rows fail closed."""

    env = _Env()
    env.simulator.ped_pos = ped_pos

    with pytest.raises(ValueError, match=match):
        compute_cbf_observation_from_env(
            env=env,
            config=_config(),
            previous_ped_positions=None,
            dt=0.1,
        )


@pytest.mark.parametrize(
    "robot_pos,match",
    [
        ([], "robot_pos"),
        ([np.array([np.nan, 0.0], dtype=float)], "finite xy"),
    ],
)
def test_compute_cbf_observation_rejects_malformed_robot_position(
    robot_pos: list[np.ndarray],
    match: str,
) -> None:
    """Malformed simulator robot position fails closed."""

    env = _Env()
    env.simulator.robot_pos = robot_pos

    with pytest.raises(ValueError, match=match):
        compute_cbf_observation_from_env(
            env=env,
            config=_config(),
            previous_ped_positions=None,
            dt=0.1,
        )


def test_compute_cbf_observation_rejects_nonpositive_dt() -> None:
    """Finite-difference pedestrian velocity requires positive dt."""

    with pytest.raises(ValueError, match="dt"):
        compute_cbf_observation_from_env(
            env=_Env(),
            config=_config(),
            previous_ped_positions=np.array([[0.0, 0.0]], dtype=float),
            dt=0.0,
        )


def test_cbf_runtime_private_status_and_trace_summary_branches() -> None:
    """Status helper and optional step trace summary stay internally consistent."""

    assert (
        cbf_safety_filter_runtime._status_from_decision_label("cbf_disabled", intervened=False)
        == "disabled"
    )
    assert (
        cbf_safety_filter_runtime._status_from_decision_label("cbf_feasible", intervened=False)
        == "pass_through"
    )
    assert (
        cbf_safety_filter_runtime._status_from_decision_label("cbf_best_effort", intervened=False)
        == "fallback_infeasible"
    )
    assert (
        cbf_safety_filter_runtime._status_from_decision_label("unknown", intervened=False)
        == "filtered"
    )

    runtime = runtime_config_from_mapping(
        {
            "enabled": True,
            "arm_key": CBF_COLLISION_CONE_ARM,
            "record_step_trace": True,
        }
    )
    record = ineligible_cbf_safety_filter_step_record(
        runtime=runtime,
        step_idx=1,
        reason="unsupported_command_shape",
    )
    summary = summarize_cbf_safety_filter_trace([record], runtime=runtime)

    assert summary["steps"][0]["ineligible_reason"] == "unsupported_command_shape"


def test_run_map_episode_records_cbf_metadata_metrics_and_ledger(monkeypatch) -> None:
    """Enabled CBF filter records episode summary, metrics, and ledger provenance."""

    def policy_builder(*_args, **_kwargs):
        return (lambda _obs: (1.0, 0.0)), {}

    record = _run_episode_with_policy(
        monkeypatch,
        policy_builder,
        cbf_safety_filter={"enabled": True, "arm_key": CBF_COLLISION_CONE_ARM},
    )

    cbf = record["algorithm_metadata"]["cbf_safety_filter"]
    assert cbf["schema_version"] == CBF_SAFETY_FILTER_EPISODE_SUMMARY_SCHEMA
    assert "cbf_filter_intervention_rate" in record["metrics"]
    ledger = build_event_ledger(record)
    assert ledger["provenance"]["cbf_safety_filter"]["arm_key"] == CBF_COLLISION_CONE_ARM


def test_run_map_episode_fails_closed_for_native_action_when_cbf_enabled(monkeypatch) -> None:
    """CBF filtering absolute commands fails closed for native-action policies."""

    def policy_builder(*_args, **_kwargs):
        def policy(_obs):
            policy._last_step_native = True
            return np.array([0.0, 0.0], dtype=float)

        return policy, {"algorithm": "unit"}

    with pytest.raises(ValueError, match="native environment actions"):
        _run_episode_with_policy(
            monkeypatch,
            policy_builder,
            cbf_safety_filter={"enabled": True, "arm_key": CBF_COLLISION_CONE_ARM},
        )


def test_run_map_episode_fails_closed_for_unsupported_command_when_cbf_enabled(
    monkeypatch,
) -> None:
    """Non-pair commands fail closed when CBF filtering is enabled."""

    def policy_builder(*_args, **_kwargs):
        return (lambda _obs: "forward"), {}

    with pytest.raises(TypeError, match="commands shaped"):
        _run_episode_with_policy(
            monkeypatch,
            policy_builder,
            cbf_safety_filter={"enabled": True, "arm_key": CBF_COLLISION_CONE_ARM},
        )


def test_run_map_episode_rejects_combined_safety_wrapper_and_cbf(monkeypatch) -> None:
    """First slice keeps wrapper and CBF mutually exclusive for attribution."""

    def policy_builder(*_args, **_kwargs):
        return (lambda _obs: (1.0, 0.0)), {}

    _patch_episode_runtime(monkeypatch)
    with pytest.raises(ValueError, match="cannot both be enabled"):
        map_runner_episode.run_map_episode(
            {"name": "cbf-runtime", "simulation_config": {"max_episode_steps": 1}},
            seed=1,
            horizon=1,
            dt=0.1,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo="goal",
            scenario_path=Path(__file__),
            safety_wrapper={"enabled": True, "arm_key": "wrapper_on"},
            cbf_safety_filter={"enabled": True, "arm_key": CBF_COLLISION_CONE_ARM},
            policy_builder=policy_builder,
        )


def test_ineligible_step_records_count_in_summary_denominator() -> None:
    """Fail-open ineligible steps stay visible in rates."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": CBF_COLLISION_CONE_ARM})
    wrapped = apply_runtime_cbf_safety_filter(
        command=(1.0, 0.25),
        env=_Env(),
        config=_config(),
        runtime=runtime,
        previous_ped_positions=np.array([[0.3, 0.0]], dtype=float),
        step_idx=0,
    )[1]
    ineligible = ineligible_cbf_safety_filter_step_record(
        runtime=runtime,
        step_idx=1,
        reason="unsupported_command_shape",
    )

    summary = summarize_cbf_safety_filter_trace([wrapped, ineligible], runtime=runtime)

    assert ineligible["eligible_for_cbf_filter"] is False
    assert ineligible["intervened"] is False
    assert summary["step_count"] == 2
    assert summary["eligible_step_count"] == 1


def test_scenario_identity_includes_only_enabled_cbf_filter() -> None:
    """Enabled CBF filter is resume identity; disabled CBF config is ignored."""

    base = {
        "name": "identity",
        "map": "maps/svg_maps/example.svg",
        "start": [0.0, 0.0],
        "goal": [1.0, 0.0],
    }

    off_payload = _scenario_identity_payload(
        base,
        algo="goal",
        algo_config={},
        horizon=1,
        dt=0.1,
        record_forces=False,
        cbf_safety_filter={"enabled": False, "arm_key": CBF_OFF_ARM},
    )
    on_payload = _scenario_identity_payload(
        base,
        algo="goal",
        algo_config={},
        horizon=1,
        dt=0.1,
        record_forces=False,
        cbf_safety_filter={"enabled": True, "arm_key": CBF_COLLISION_CONE_ARM},
    )
    dpcbf_payload = _scenario_identity_payload(
        base,
        algo="goal",
        algo_config={},
        horizon=1,
        dt=0.1,
        record_forces=False,
        cbf_safety_filter={
            "enabled": True,
            "arm_key": CBF_DYNAMIC_PARABOLIC_V1_ARM,
            "variant": "dynamic_parabolic_cbf_v1",
        },
    )

    assert "cbf_safety_filter" not in off_payload
    assert on_payload["cbf_safety_filter"]["arm_key"] == CBF_COLLISION_CONE_ARM
    assert dpcbf_payload["cbf_safety_filter"]["arm_key"] == CBF_DYNAMIC_PARABOLIC_V1_ARM
    assert dpcbf_payload["cbf_safety_filter"]["variant"] == "dynamic_parabolic_cbf_v1"
    assert dpcbf_payload["cbf_safety_filter"] != on_payload["cbf_safety_filter"]
