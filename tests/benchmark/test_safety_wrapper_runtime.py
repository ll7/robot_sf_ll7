"""Tests for opt-in benchmark runtime safety-wrapper binding."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark import map_runner_episode, safety_wrapper_runtime
from robot_sf.benchmark.event_ledger import build_event_ledger
from robot_sf.benchmark.safety_wrapper_runtime import (
    SAFETY_WRAPPER_EPISODE_SUMMARY_SCHEMA,
    SAFETY_WRAPPER_FALSE_STOP_DIAGNOSTIC_SCHEMA,
    SAFETY_WRAPPER_RUNTIME_STEP_SCHEMA,
    analyze_false_stop_diagnostic,
    apply_runtime_safety_wrapper,
    compute_safety_context_from_env,
    ineligible_safety_wrapper_step_record,
    runtime_config_from_mapping,
    summarize_safety_wrapper_trace,
)
from robot_sf.robot.safety_wrapper import (
    INTERVENTION_HARD_STOP,
    INTERVENTION_NONE,
    INTERVENTION_SPEED_CAP,
)


def _hard_stop_row(step: int, clearance: float, ttc: float | None = 0.5) -> dict[str, object]:
    """Synthetic hard-stop trace row with a given predicted clearance context."""

    return {
        "schema_version": SAFETY_WRAPPER_RUNTIME_STEP_SCHEMA,
        "step": int(step),
        "arm_key": "wrapper_on",
        "enabled": True,
        "eligible_for_wrapper": True,
        "intervention": INTERVENTION_HARD_STOP,
        "intervened": True,
        "context": {
            "min_pedestrian_distance_m": 0.5,
            "min_clearance_m": float(clearance),
            "min_ttc_s": ttc,
        },
    }


def _passthrough_row(step: int, clearance: float) -> dict[str, object]:
    """Synthetic non-intervening trace row carrying a clearance context."""

    return {
        "schema_version": SAFETY_WRAPPER_RUNTIME_STEP_SCHEMA,
        "step": int(step),
        "arm_key": "wrapper_on",
        "enabled": True,
        "eligible_for_wrapper": True,
        "intervention": INTERVENTION_NONE,
        "intervened": False,
        "context": {
            "min_pedestrian_distance_m": 3.0,
            "min_clearance_m": float(clearance),
            "min_ttc_s": None,
        },
    }


class _Robot:
    pose = np.array([0.0, 0.0, 0.0], dtype=float)


class _Simulator:
    def __init__(self) -> None:
        self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
        self.ped_pos = np.array([[0.2, 0.0]], dtype=float)
        self.robots = [_Robot()]


class _Env:
    def __init__(self) -> None:
        self.simulator = _Simulator()


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        sim_config=SimpleNamespace(
            time_per_step_in_secs=0.1,
            robot_radius=0.1,
            ped_radius=0.1,
        )
    )


def _config_with_robot_config_radius() -> SimpleNamespace:
    return SimpleNamespace(
        sim_config=SimpleNamespace(
            time_per_step_in_secs=0.1,
            ped_radius=0.1,
        ),
        robot_config=SimpleNamespace(radius=0.5),
    )


def _policy_builder(*_args, **_kwargs):
    def policy(_obs):
        return (1.0, 0.25)

    return policy, {"algorithm": "unit"}


class _EpisodeSim:
    def __init__(self) -> None:
        self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
        self.goal_pos = [np.array([1.0, 0.0], dtype=float)]
        self.ped_pos = np.array([[0.2, 0.0]], dtype=float)
        self.last_ped_forces = np.zeros((1, 2), dtype=float)
        self.map_def = SimpleNamespace(obstacles=[], bounds=(0.0, 0.0, 2.0, 2.0))


class _EpisodeEnv:
    def __init__(self) -> None:
        self.simulator = _EpisodeSim()
        self.action_space = None

    def reset(self, seed=None):
        return {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {"positions": self.simulator.ped_pos},
        }, {}

    def step(self, _action):
        return self.reset()[0], 0.0, True, False, {"meta": {"is_route_complete": True}}

    def close(self) -> None:
        return None


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


def _run_episode_with_policy(monkeypatch, policy_builder, *, safety_wrapper):
    _patch_episode_runtime(monkeypatch)
    return map_runner_episode.run_map_episode(
        {"name": "wrapper-runtime", "simulation_config": {"max_episode_steps": 1}},
        seed=1,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=Path(__file__),
        safety_wrapper=safety_wrapper,
        policy_builder=policy_builder,
    )


def test_runtime_config_disabled_by_default_preserves_off_state() -> None:
    """Missing runtime config keeps wrapper disabled and therefore opt-in only."""

    runtime = runtime_config_from_mapping(None)

    assert runtime.enabled is False
    assert runtime.arm_key == "wrapper_off"


def test_wrapper_on_emits_schema_tagged_intervention_record_and_summary() -> None:
    """A close pre-step pedestrian produces well-formed wrapper evidence."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": "wrapper_on"})
    corrected, record = apply_runtime_safety_wrapper(
        command=(1.0, 0.25),
        env=_Env(),
        config=_config(),
        runtime=runtime,
        previous_ped_positions=None,
        step_idx=3,
    )

    assert corrected == (0.0, 0.25)
    assert record["schema_version"] == SAFETY_WRAPPER_RUNTIME_STEP_SCHEMA
    assert record["arm_key"] == "wrapper_on"
    assert record["enabled"] is True
    assert record["eligible_for_wrapper"] is True
    assert record["context_source"] == "simulator_state_pre_step"
    assert record["clearance_sources"] == ["pedestrians"]
    assert record["pedestrian_velocity_identity"] == "row_order_finite_difference_no_stable_ids"
    assert record["intervention"] == INTERVENTION_HARD_STOP
    assert record["intervened"] is True

    summary = summarize_safety_wrapper_trace([record], runtime=runtime)
    assert summary["schema_version"] == SAFETY_WRAPPER_EPISODE_SUMMARY_SCHEMA
    assert summary["intervened_step_count"] == 1
    assert summary["intervention_rate"] == 1.0
    assert summary["first_hard_stop_step"] == 3
    assert summary["min_context_clearance_m"] <= 0.0
    assert summary["false_stop_analysis_supported"] is False


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"enabled": True, "arm_key": "wrapper_off"}, "enabled=True"),
        ({"enabled": False, "arm_key": "wrapper_on"}, "enabled=False"),
        ({"enabled": True, "arm_key": "wrapper_on_v2"}, "versioned experimental arm"),
        (
            {"enabled": True, "arm_key": "wrapper_on", "capped_speed_m_s": 0.25},
            "predeclared ablation config",
        ),
    ],
)
def test_runtime_config_fails_closed_for_arm_or_threshold_drift(
    payload: dict[str, object], match: str
) -> None:
    """Wrapper-on semantics must match the predeclared ablation arm."""

    with pytest.raises(ValueError, match=match):
        runtime_config_from_mapping(payload)


@pytest.mark.parametrize(
    "ped_pos,expected_count",
    [
        (np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float), 2),
        (np.array([[1.0, 0.0, 7.0], [2.0, 0.0, 8.0]], dtype=float), 2),
        (np.empty((0, 2), dtype=float), 0),
        (np.array([1.0, 0.0, 2.0, 0.0], dtype=float), 2),
    ],
)
def test_context_accepts_supported_pedestrian_position_shapes(
    ped_pos: np.ndarray, expected_count: int
) -> None:
    """Runtime parser accepts xy rows, xyz rows, empty rows, and even flat xy."""

    env = _Env()
    env.simulator.ped_pos = ped_pos

    context, provenance = compute_safety_context_from_env(
        env=env,
        config=_config(),
        command=(0.0, 0.0),
        previous_ped_positions=None,
        dt=0.1,
    )

    assert provenance["pedestrian_count"] == expected_count
    if expected_count == 0:
        assert math.isinf(context.min_pedestrian_distance_m)
        assert math.isinf(context.min_clearance_m)


@pytest.mark.parametrize(
    "ped_pos,match",
    [
        (np.array([1.0, 0.0, 2.0], dtype=float), "even-length flat"),
        (np.array([[1.0, np.nan]], dtype=float), "finite"),
    ],
)
def test_context_rejects_malformed_or_nonfinite_pedestrian_positions(
    ped_pos: np.ndarray, match: str
) -> None:
    """Malformed simulator pedestrian state fails closed."""

    env = _Env()
    env.simulator.ped_pos = ped_pos

    with pytest.raises(ValueError, match=match):
        compute_safety_context_from_env(
            env=env,
            config=_config(),
            command=(0.0, 0.0),
            previous_ped_positions=None,
            dt=0.1,
        )


@pytest.mark.parametrize(
    "robot_pos,match",
    [
        ([], "robot_pos"),
        ([np.array([math.inf, 0.0], dtype=float)], "finite xy"),
    ],
)
def test_context_rejects_missing_or_nonfinite_robot_position(
    robot_pos: list[np.ndarray], match: str
) -> None:
    """Robot xy position is required for pre-step context construction."""

    env = _Env()
    env.simulator.robot_pos = robot_pos

    with pytest.raises(ValueError, match=match):
        compute_safety_context_from_env(
            env=env,
            config=_config(),
            command=(0.0, 0.0),
            previous_ped_positions=None,
            dt=0.1,
        )


def test_no_pedestrian_episode_has_infinite_clearance_and_no_intervention() -> None:
    """No-pedestrian context stays hazard-free and does not invent an intervention."""

    env = _Env()
    env.simulator.ped_pos = np.empty((0, 2), dtype=float)
    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": "wrapper_on"})

    corrected, record = apply_runtime_safety_wrapper(
        command=(1.0, 0.25),
        env=env,
        config=_config(),
        runtime=runtime,
        previous_ped_positions=None,
        step_idx=0,
    )

    assert corrected == (1.0, 0.25)
    assert math.isinf(record["context"]["min_pedestrian_distance_m"])
    assert math.isinf(record["context"]["min_clearance_m"])
    assert record["context"]["min_ttc_s"] is None
    assert record["intervention"] == INTERVENTION_NONE
    assert record["intervened"] is False


def test_non_closing_relative_velocity_has_no_ttc() -> None:
    """TTC is undefined when pedestrian motion is not closing on the robot."""

    env = _Env()
    env.simulator.ped_pos = np.array([[2.0, 0.0]], dtype=float)

    context, _ = compute_safety_context_from_env(
        env=env,
        config=_config(),
        command=(0.0, 0.0),
        previous_ped_positions=np.array([[1.9, 0.0]], dtype=float),
        dt=0.1,
    )

    assert context.min_ttc_s is None


def test_nonpositive_dt_fails_closed() -> None:
    """Runtime context construction requires a positive timestep."""

    with pytest.raises(ValueError, match="dt must be positive"):
        compute_safety_context_from_env(
            env=_Env(),
            config=_config(),
            command=(0.0, 0.0),
            previous_ped_positions=None,
            dt=0.0,
        )


def test_xy_rows_rejects_unsupported_2d_shape() -> None:
    """Direct parser fixture covers non-xy table shapes before context use."""

    with pytest.raises(ValueError, match="shaped"):
        safety_wrapper_runtime._xy_rows(np.array([[1.0], [2.0]], dtype=float))


def test_context_uses_robot_config_radius_when_sim_radius_absent() -> None:
    """Runtime clearance should match the scenario robot radius source."""

    context, provenance = compute_safety_context_from_env(
        command=(0.0, 0.0),
        env=_Env(),
        config=_config_with_robot_config_radius(),
        previous_ped_positions=None,
        dt=0.1,
    )

    assert provenance["robot_radius_m"] == 0.5
    assert context.min_clearance_m == pytest.approx(-0.4)


def test_ineligible_step_records_count_in_summary_denominator() -> None:
    """Fail-open ineligible wrapper steps should not disappear from rates."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": "wrapper_on"})
    wrapped = apply_runtime_safety_wrapper(
        command=(1.0, 0.25),
        env=_Env(),
        config=_config(),
        runtime=runtime,
        previous_ped_positions=None,
        step_idx=0,
    )[1]
    ineligible = ineligible_safety_wrapper_step_record(
        runtime=runtime,
        step_idx=1,
        reason="unsupported_command_shape",
    )

    summary = summarize_safety_wrapper_trace([wrapped, ineligible], runtime=runtime)

    assert ineligible["eligible_for_wrapper"] is False
    assert ineligible["intervened"] is False
    assert summary["step_count"] == 2
    assert summary["eligible_step_count"] == 1
    assert summary["intervened_step_count"] == 1
    assert summary["intervention_rate"] == 0.5


def test_run_map_episode_records_wrapper_metadata_when_enabled(monkeypatch) -> None:
    """Runtime binding emits episode and ledger-ready evidence only when enabled."""

    record = _run_episode_with_policy(
        monkeypatch,
        _policy_builder,
        safety_wrapper={"enabled": True, "arm_key": "wrapper_on", "record_step_trace": True},
    )

    summary = record["algorithm_metadata"]["safety_wrapper"]
    assert summary["schema_version"] == SAFETY_WRAPPER_EPISODE_SUMMARY_SCHEMA
    assert summary["step_count"] == 1
    assert summary["eligible_step_count"] == 1
    assert summary["intervened_step_count"] == 1
    assert len(summary["step_trace"]) == 1
    assert summary["step_trace"][0]["eligible_for_wrapper"] is True
    assert record["metrics"]["wrapper_intervention_rate"] == 1.0

    ledger = build_event_ledger(record)
    assert ledger["provenance"]["safety_wrapper"] == summary


def test_run_map_episode_fails_closed_for_native_action_when_wrapper_enabled(
    monkeypatch,
) -> None:
    """Native env actions must not be silently transformed by wrapper-on runs."""

    def native_policy_builder(*_args, **_kwargs):
        def policy(_obs):
            return np.array([0.1, 0.0], dtype=float)

        policy._last_step_native = True
        return policy, {"algorithm": "unit"}

    with pytest.raises(ValueError, match="native environment actions"):
        _run_episode_with_policy(
            monkeypatch,
            native_policy_builder,
            safety_wrapper={"enabled": True, "arm_key": "wrapper_on"},
        )


def test_run_map_episode_fails_closed_for_unsupported_command_when_wrapper_enabled(
    monkeypatch,
) -> None:
    """Unsupported structured commands fail closed under wrapper-on by default."""

    def unsupported_policy_builder(*_args, **_kwargs):
        def policy(_obs):
            return {"command_kind": "unsupported"}

        return policy, {"algorithm": "unit"}

    with pytest.raises(TypeError, match="expects commands shaped"):
        _run_episode_with_policy(
            monkeypatch,
            unsupported_policy_builder,
            safety_wrapper={"enabled": True, "arm_key": "wrapper_on"},
        )


def test_run_map_episode_records_fail_open_native_action_as_ineligible(monkeypatch) -> None:
    """Fail-open native env actions keep original command and record provenance."""

    def native_policy_builder(*_args, **_kwargs):
        def policy(_obs):
            return np.array([0.1, 0.0], dtype=float)

        policy._last_step_native = True
        return policy, {"algorithm": "unit"}

    record = _run_episode_with_policy(
        monkeypatch,
        native_policy_builder,
        safety_wrapper={
            "enabled": True,
            "arm_key": "wrapper_on",
            "fail_on_native_action": False,
            "record_step_trace": True,
        },
    )

    summary = record["algorithm_metadata"]["safety_wrapper"]
    assert summary["step_count"] == 1
    assert summary["eligible_step_count"] == 0
    assert summary["intervened_step_count"] == 0
    assert len(summary["step_trace"]) == 1
    step_record = summary["step_trace"][0]
    assert step_record["eligible_for_wrapper"] is False
    assert step_record["ineligible_reason"] == "native_environment_action"
    assert step_record["context"] is None
    assert record["metrics"]["wrapper_intervention_rate"] == 0.0


def test_run_map_episode_records_fail_open_unsupported_command_as_ineligible(
    monkeypatch,
) -> None:
    """Fail-open unsupported command shapes record provenance without wrapper rewrite."""

    def unsupported_policy_builder(*_args, **_kwargs):
        def policy(_obs):
            return {"command_kind": "unsupported"}

        return policy, {"algorithm": "unit"}

    record = _run_episode_with_policy(
        monkeypatch,
        unsupported_policy_builder,
        safety_wrapper={
            "enabled": True,
            "arm_key": "wrapper_on",
            "fail_on_unsupported_command": False,
            "record_step_trace": True,
        },
    )

    summary = record["algorithm_metadata"]["safety_wrapper"]
    assert summary["step_count"] == 1
    assert summary["eligible_step_count"] == 0
    assert summary["intervened_step_count"] == 0
    assert len(summary["step_trace"]) == 1
    step_record = summary["step_trace"][0]
    assert step_record["eligible_for_wrapper"] is False
    assert step_record["ineligible_reason"] == "unsupported_command_shape"
    assert step_record["context"] is None
    assert record["metrics"]["wrapper_intervention_rate"] == 0.0


def test_false_stop_diagnostic_unsupported_without_time_per_step() -> None:
    """Without a positive dt the forward window is undefined and the proxy fails honest."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": "wrapper_on"})
    trace = [_hard_stop_row(0, clearance=-0.1)]

    diagnostic = analyze_false_stop_diagnostic(trace, runtime=runtime, time_per_step_s=None)

    assert diagnostic["schema_version"] == SAFETY_WRAPPER_FALSE_STOP_DIAGNOSTIC_SCHEMA
    assert diagnostic["evidence_kind"] == "diagnostic_proxy"
    assert diagnostic["causal_false_stop_rate_supported"] is False
    assert diagnostic["supported"] is False
    assert diagnostic["unsupported_reason"] == "lookahead_step_window_unresolved"
    assert "hazard_confirmed_count" not in diagnostic


def test_false_stop_diagnostic_confirms_valid_stop_on_predicted_contact() -> None:
    """A veto whose trigger step shows non-positive clearance is a confirmed-valid stop."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": "wrapper_on"})
    # Contact predicted at the trigger step; window is complete (episode extends beyond lookahead).
    trace = [
        _hard_stop_row(0, clearance=-0.2),
        _passthrough_row(1, clearance=1.5),
        _passthrough_row(2, clearance=2.0),
        _passthrough_row(3, clearance=2.5),
    ]

    diagnostic = analyze_false_stop_diagnostic(trace, runtime=runtime, time_per_step_s=0.1)

    # false_stop_lookahead_s defaults to 2.0s; at dt=0.1 that is 20 steps.
    assert diagnostic["lookahead_steps"] == 20
    assert diagnostic["supported"] is True
    assert diagnostic["hard_stop_count"] == 1
    assert diagnostic["analyzed_hard_stop_count"] == 1
    assert diagnostic["hazard_confirmed_count"] == 1
    assert diagnostic["analysis_unsupported_count"] == 0
    assert diagnostic["window_truncated_count"] == 0
    assert diagnostic["hazard_confirmed_rate"] == 1.0


def test_false_stop_diagnostic_flags_unsupported_when_hazard_dissipates() -> None:
    """A veto with a complete positive-clearance window cannot be judged without a counterfactual."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": "wrapper_on"})
    # Triggered by TTC (clearance positive), hazard never reaches contact across a full window.
    trace = [_hard_stop_row(0, clearance=0.25, ttc=0.5)]
    trace += [_passthrough_row(step, clearance=1.0) for step in range(1, 4)]

    # dt=1.0s, lookahead 2.0s -> 2 steps; steps 1..3 present so the window (steps 1,2) is complete.
    diagnostic = analyze_false_stop_diagnostic(trace, runtime=runtime, time_per_step_s=1.0)

    assert diagnostic["lookahead_steps"] == 2
    assert diagnostic["hazard_confirmed_count"] == 0
    assert diagnostic["analysis_unsupported_count"] == 1
    assert diagnostic["window_truncated_count"] == 0
    assert diagnostic["analysis_unsupported_rate"] == 1.0


def test_false_stop_diagnostic_marks_truncated_window_at_episode_end() -> None:
    """A veto without a full lookahead window is partial evidence, not unsupported."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": "wrapper_on"})
    # Hard stop at the last step: no forward steps exist to complete the 2-step window.
    trace = [_passthrough_row(0, clearance=1.0), _hard_stop_row(1, clearance=0.2, ttc=0.5)]

    diagnostic = analyze_false_stop_diagnostic(trace, runtime=runtime, time_per_step_s=1.0)

    assert diagnostic["hazard_confirmed_count"] == 0
    assert diagnostic["window_truncated_count"] == 1
    assert diagnostic["analysis_unsupported_count"] == 0


def test_false_stop_diagnostic_counts_speed_caps_separately() -> None:
    """Speed caps are tallied but not classified as stops in the false-stop proxy."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": "wrapper_on"})
    speed_cap = _passthrough_row(0, clearance=1.0)
    speed_cap["intervention"] = INTERVENTION_SPEED_CAP
    speed_cap["intervened"] = True
    trace = [speed_cap, _hard_stop_row(1, clearance=-0.1), _passthrough_row(2, clearance=2.0)]

    diagnostic = analyze_false_stop_diagnostic(trace, runtime=runtime, time_per_step_s=2.0)

    assert diagnostic["speed_cap_count"] == 1
    assert diagnostic["hard_stop_count"] == 1
    assert diagnostic["hazard_confirmed_count"] == 1


def test_summary_embeds_false_stop_proxy_when_dt_available() -> None:
    """The episode summary carries the proxy block and keeps the causal flag unsupported."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": "wrapper_on"})
    trace = [_hard_stop_row(0, clearance=-0.2), _passthrough_row(1, clearance=2.0)]

    summary = summarize_safety_wrapper_trace(trace, runtime=runtime, time_per_step_s=0.1)

    assert summary["false_stop_analysis_supported"] is False
    assert summary["false_stop_proxy_supported"] is True
    diagnostic = summary["false_stop_diagnostic"]
    assert diagnostic["schema_version"] == SAFETY_WRAPPER_FALSE_STOP_DIAGNOSTIC_SCHEMA
    assert diagnostic["causal_false_stop_rate_supported"] is False
    assert diagnostic["hazard_confirmed_count"] == 1


def test_summary_proxy_unsupported_when_dt_missing() -> None:
    """Without dt the summary still emits the block but marks the proxy unsupported."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": "wrapper_on"})
    trace = [_hard_stop_row(0, clearance=-0.2)]

    summary = summarize_safety_wrapper_trace(trace, runtime=runtime)

    assert summary["false_stop_analysis_supported"] is False
    assert summary["false_stop_proxy_supported"] is False
    assert summary["false_stop_diagnostic"]["supported"] is False
