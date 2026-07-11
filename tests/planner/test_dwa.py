"""Contract tests for the classical Dynamic Window Approach baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.planner.dwa import DWAPlannerAdapter, DWAPlannerConfig, build_dwa_config


def _observation(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    speed: float = 0.0,
    angular_velocity: float = 0.0,
    goal: tuple[float, float] = (3.0, 0.0),
    pedestrians: list[tuple[float, float]] | None = None,
    pedestrian_velocities: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Build a minimal structured observation accepted by DWA."""
    positions = [] if pedestrians is None else pedestrians
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "angular_velocity": np.asarray([angular_velocity], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(positions, dtype=float),
            "velocities": np.asarray(
                [] if pedestrian_velocities is None else pedestrian_velocities, dtype=float
            ),
            "count": np.asarray([len(positions)], dtype=float),
        },
    }


def test_dwa_samples_only_dynamically_reachable_commands() -> None:
    """Selected velocity stays inside the configured acceleration window."""
    config = DWAPlannerConfig(
        max_linear_speed=1.0,
        max_angular_speed=1.0,
        max_linear_acceleration=0.5,
        max_angular_acceleration=1.0,
        control_dt=0.2,
        linear_samples=5,
        angular_samples=5,
    )
    command = DWAPlannerAdapter(config).plan(
        _observation(speed=0.5, angular_velocity=0.2, goal=(3.0, 0.0))
    )
    assert 0.4 <= command[0] <= 0.6
    assert 0.0 <= command[1] <= 0.4


def test_dwa_is_deterministic_and_goal_directed() -> None:
    """Identical structured state produces the same bounded, forward command."""
    config = DWAPlannerConfig()
    observation = _observation(goal=(3.0, 0.0))
    first = DWAPlannerAdapter(config).plan(observation)
    second = DWAPlannerAdapter(config).plan(observation)
    assert first == second
    assert 0.0 < first[0] <= config.max_linear_speed
    assert abs(first[1]) <= config.max_angular_speed


def test_dwa_stops_at_goal_and_rejects_unsafe_forward_rollouts() -> None:
    """The baseline stops at the goal and does not select a colliding forward command."""
    config = DWAPlannerConfig(goal_tolerance=0.3, linear_samples=3, angular_samples=3)
    planner = DWAPlannerAdapter(config)
    assert planner.plan(_observation(goal=(0.1, 0.0))) == (0.0, 0.0)

    command = planner.plan(_observation(goal=(3.0, 0.0), pedestrians=[(0.35, 0.0)]))
    assert command[0] == pytest.approx(0.0)


def test_dwa_config_builder_applies_explicit_acceleration_parameters() -> None:
    """The canonical config parser preserves DWA-specific dynamic-window settings."""
    config = build_dwa_config(
        {
            "max_linear_acceleration": 0.4,
            "max_angular_acceleration": 0.9,
            "linear_samples": 4,
            "angular_samples": 6,
        }
    )
    assert config.max_linear_acceleration == pytest.approx(0.4)
    assert config.max_angular_acceleration == pytest.approx(0.9)
    assert config.linear_samples == 4
    assert config.angular_samples == 6


def test_dwa_prediction_scoring_is_opt_in_in_canonical_configs() -> None:
    """The classic config retains static scoring while the variant opts into forecasts."""
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "algos"
    classic = yaml.safe_load((config_dir / "dwa_classic.yaml").read_text(encoding="utf-8"))
    predictive = yaml.safe_load(
        (config_dir / "dwa_prediction_scoring.yaml").read_text(encoding="utf-8")
    )

    assert build_dwa_config(classic).prediction_scoring_enabled is False
    assert build_dwa_config(predictive).prediction_scoring_enabled is True
    assert {
        key: value for key, value in classic.items() if key != "prediction_scoring_enabled"
    } == {key: value for key, value in predictive.items() if key != "prediction_scoring_enabled"}


def test_dwa_prediction_scoring_avoids_a_future_crossing() -> None:
    """Time-aligned constant-velocity scoring rejects a pedestrian's future crossing point."""
    observation = _observation(
        goal=(3.0, 0.0),
        pedestrians=[(0.6, 0.8)],
        pedestrian_velocities=[(0.0, -1.0)],
    )
    base_command = DWAPlannerAdapter(DWAPlannerConfig()).plan(observation)
    predictive_command = DWAPlannerAdapter(DWAPlannerConfig(prediction_scoring_enabled=True)).plan(
        observation
    )
    repeated_command = DWAPlannerAdapter(DWAPlannerConfig(prediction_scoring_enabled=True)).plan(
        observation
    )

    assert base_command[0] > 0.0
    assert repeated_command == predictive_command
    assert predictive_command != base_command
    assert predictive_command[0] <= base_command[0]


def test_dwa_prediction_scoring_rotates_ego_velocity_to_world() -> None:
    """Pedestrian ego-frame velocities are rotated using the current robot heading."""
    planner = DWAPlannerAdapter(DWAPlannerConfig(prediction_scoring_enabled=True))
    *_, velocities_world = planner._extract_state(
        _observation(
            heading=np.pi / 2,
            pedestrians=[(1.0, 0.0)],
            pedestrian_velocities=[(1.0, 0.0)],
        )
    )

    np.testing.assert_allclose(velocities_world, [[0.0, 1.0]], atol=1e-12)


def test_dwa_prediction_scoring_requires_active_pedestrian_velocities() -> None:
    """The opt-in variant fails closed when an active forecast input is absent."""
    planner = DWAPlannerAdapter(DWAPlannerConfig(prediction_scoring_enabled=True))

    with pytest.raises(ValueError, match="one finite .* velocity"):
        planner.plan(_observation(pedestrians=[(1.0, 0.0)]))


def test_dwa_prediction_scoring_requires_finite_active_pedestrian_positions() -> None:
    """The opt-in variant rejects malformed current state before forecasting."""
    planner = DWAPlannerAdapter(DWAPlannerConfig(prediction_scoring_enabled=True))

    with pytest.raises(ValueError, match="one finite .* world position"):
        planner.plan(
            _observation(
                pedestrians=[(float("nan"), 0.0)],
                pedestrian_velocities=[(0.0, 0.0)],
            )
        )


@pytest.mark.parametrize(("raw", "expected"), [("yes", True), ("off", False)])
def test_dwa_config_builder_parses_boolean_strings(raw: str, expected: bool) -> None:
    """Text config values use explicit boolean parsing rather than string truthiness."""
    assert (
        build_dwa_config({"prediction_scoring_enabled": raw}).prediction_scoring_enabled is expected
    )


def test_dwa_config_builder_rejects_non_boolean_prediction_flag() -> None:
    """Ambiguous prediction-scoring flag values fail closed."""
    with pytest.raises(ValueError, match="prediction_scoring_enabled must be a boolean"):
        build_dwa_config({"prediction_scoring_enabled": 1})


def test_dwa_base_scoring_ignores_pedestrian_velocities() -> None:
    """Velocity payloads do not alter the disabled-by-default base DWA contract."""
    planner = DWAPlannerAdapter(DWAPlannerConfig())
    without_velocity = planner.plan(_observation(pedestrians=[(0.6, 0.8)]))
    with_velocity = planner.plan(
        _observation(
            pedestrians=[(0.6, 0.8)],
            pedestrian_velocities=[(0.0, -1.0)],
        )
    )

    assert with_velocity == without_velocity


def test_dwa_dynamic_window_preserves_reachability_outside_speed_limits() -> None:
    """Out-of-range current commands collapse to the nearest dynamically reachable value."""
    planner = DWAPlannerAdapter(
        DWAPlannerConfig(
            max_linear_speed=1.0,
            max_angular_speed=1.0,
            max_linear_acceleration=0.5,
            max_angular_acceleration=1.0,
            control_dt=0.2,
        )
    )

    assert planner._dynamic_window(1.5, -2.0) == pytest.approx((1.4, 1.4, -1.8, -1.8))


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_linear_speed": float("nan")},
        {"max_angular_acceleration": -0.1},
        {"control_dt": 0.0},
        {"prediction_steps": 0},
    ],
)
def test_dwa_rejects_invalid_runtime_configuration(overrides: dict[str, float]) -> None:
    """The experimental planner fails closed before it can emit invalid commands."""
    with pytest.raises(ValueError):
        DWAPlannerConfig(**overrides)


def test_dwa_diagnostics_populated_after_plan() -> None:
    """Diagnostics are populated after plan() and include expected decision keys."""
    config = DWAPlannerConfig(linear_samples=5, angular_samples=5)
    planner = DWAPlannerAdapter(config)
    observation = _observation(goal=(3.0, 0.0))
    command = planner.plan(observation)
    diag = planner.diagnostics()
    last = diag.get("last_decision", {})
    assert last["selected_source"] == "dwa"
    assert last["selected_command"] == list(command)
    assert last["constraint_reason"] == "best_feasible"
    assert last["candidate_total"] == 25
    assert last["candidate_feasible"] > 0
    assert last["candidate_infeasible"] >= 0
    assert last["feasible_score_min"] is not None
    assert last["feasible_score_max"] is not None
    assert last["feasible_score_max"] >= last["feasible_score_min"]
    window = last["dynamic_window"]
    assert window["v_min"] <= window["v_max"]
    assert window["w_min"] <= window["w_max"]
    assert last["distance_to_goal_m"] == pytest.approx(3.0)
    assert last["target_goal"]["kind"] in {"next", "current"}


def test_dwa_diagnostics_goal_reached() -> None:
    """When within goal_tolerance the diagnostics record the goal_reached constraint reason."""
    config = DWAPlannerConfig(goal_tolerance=0.5)
    planner = DWAPlannerAdapter(config)
    observation = _observation(goal=(0.1, 0.0))
    command = planner.plan(observation)
    diag = planner.diagnostics()
    last = diag.get("last_decision", {})
    assert command == (0.0, 0.0)
    assert last["constraint_reason"] == "goal_reached"
    assert last["candidate_total"] == 0
    assert last["distance_to_goal_m"] == pytest.approx(0.1)


def test_dwa_diagnostics_feasible_infeasible_counts() -> None:
    """Close obstacle produces infeasible candidates and records the count."""
    config = DWAPlannerConfig(
        linear_samples=3,
        angular_samples=3,
        safety_margin=0.5,
        robot_radius=0.25,
        pedestrian_radius=0.30,
    )
    planner = DWAPlannerAdapter(config)
    observation = _observation(goal=(3.0, 0.0), pedestrians=[(0.3, 0.0)])
    planner.plan(observation)
    diag = planner.diagnostics()
    last = diag.get("last_decision", {})
    assert last["candidate_infeasible"] > 0
    assert last["candidate_feasible"] < last["candidate_total"]


def test_dwa_diagnostics_resets_between_episodes() -> None:
    """Diagnostics from a prior episode do not leak into the next episode."""
    config = DWAPlannerConfig(linear_samples=3, angular_samples=3)
    planner = DWAPlannerAdapter(config)
    planner.plan(_observation(goal=(3.0, 0.0)))
    first_diag = planner.diagnostics()["last_decision"]
    assert first_diag["constraint_reason"] == "best_feasible"
    planner.plan(_observation(goal=(0.05, 0.0)))
    second_diag = planner.diagnostics()["last_decision"]
    assert second_diag["constraint_reason"] == "goal_reached"


def test_dwa_diagnostics_returns_an_isolated_snapshot() -> None:
    """Caller mutation of nested diagnostics cannot alter planner state."""
    planner = DWAPlannerAdapter(DWAPlannerConfig())
    planner.plan(_observation(goal=(3.0, 0.0)))

    snapshot = planner.diagnostics()["last_decision"]
    snapshot["selected_command"][0] = 99.0
    snapshot["dynamic_window"]["v_min"] = 99.0
    current = planner.diagnostics()["last_decision"]

    assert current["selected_command"][0] != 99.0
    assert current["dynamic_window"]["v_min"] != 99.0


def test_dwa_target_goal_diagnostics_tolerate_missing_goal_payload() -> None:
    """Diagnostic-only target labels preserve the planner's missing-goal defaults."""
    planner = DWAPlannerAdapter(DWAPlannerConfig())
    observation = _observation(goal=(3.0, 0.0))
    observation["goal"] = None

    planner.plan(observation)

    assert planner.diagnostics()["last_decision"]["target_goal"]["kind"] in {"next", "current"}


def _make_planner_stalled(planner: DWAPlannerAdapter, distance: float, steps: int) -> None:
    """Drive the planner's distance history to simulate a progress stall."""
    for _ in range(steps):
        obs = _observation(goal=(distance, 0.0))
        planner.plan(obs)


def test_dwa_route_rescue_is_disabled_by_default() -> None:
    """Route-rescue does not activate unless explicitly enabled."""
    config = DWAPlannerConfig()
    assert config.route_rescue_enabled is False
    planner = DWAPlannerAdapter(config)
    _make_planner_stalled(planner, 3.0, 25)
    diag = planner.diagnostics()["last_decision"]
    assert diag.get("route_rescue_active") is False
    assert diag.get("route_rescue_type") is None


def test_dwa_route_rescue_activates_on_stall() -> None:
    """Route-rescue triggers after the patience window with no progress."""
    config = DWAPlannerConfig(
        route_rescue_enabled=True,
        route_rescue_window=20,
        route_rescue_patience=5,
        route_rescue_progress_threshold=0.05,
    )
    planner = DWAPlannerAdapter(config)
    for _ in range(10):
        obs = _observation(goal=(3.0, 0.0))
        planner.plan(obs)
    diag = planner.diagnostics()["last_decision"]
    assert diag.get("route_rescue_active") is True
    assert diag.get("route_rescue_type") == "route_rescue"


def test_dwa_route_rescue_does_not_activate_with_progress() -> None:
    """Route-rescue stays inactive when the robot is making progress."""
    config = DWAPlannerConfig(
        route_rescue_enabled=True,
        route_rescue_window=20,
        route_rescue_patience=10,
        route_rescue_progress_threshold=0.05,
    )
    planner = DWAPlannerAdapter(config)
    for step in range(15):
        distance = max(5.0 - step * 0.3, 2.0)
        obs = _observation(goal=(distance, 0.0))
        planner.plan(obs)
    diag = planner.diagnostics()["last_decision"]
    assert diag.get("route_rescue_active") is False


def test_dwa_route_rescue_extends_prediction_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    """When rescue is active, the planner uses a longer prediction horizon."""
    config = DWAPlannerConfig(
        route_rescue_enabled=True,
        prediction_steps=15,
        route_rescue_window=10,
        route_rescue_patience=3,
        route_rescue_horizon_scale=2.0,
        linear_samples=3,
        angular_samples=3,
    )
    planner = DWAPlannerAdapter(config)
    observed_overrides: list[dict[str, object]] = []
    original_score = planner._rollout_score

    def record_score(**kwargs: object) -> float:
        overrides = kwargs.get("rescue_overrides")
        observed_overrides.append(dict(overrides) if isinstance(overrides, dict) else {})
        return original_score(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(planner, "_rollout_score", record_score)
    for _ in range(5):
        obs = _observation(goal=(3.0, 0.0))
        planner.plan(obs)
    planner.plan(_observation(goal=(3.0, 0.0)))
    assert planner._rescue_active is True
    assert any(
        overrides.get("prediction_steps") == 30 and overrides.get("progress_weight") == 2.0
        for overrides in observed_overrides
    ), "Expected active route-rescue scoring to use the configured horizon and progress boost"


def test_dwa_feasibility_slowdown_is_disabled_by_default() -> None:
    """Feasibility-slowdown does not activate unless explicitly enabled."""
    config = DWAPlannerConfig()
    assert config.feasibility_slowdown_enabled is False
    planner = DWAPlannerAdapter(config)
    planner.plan(_observation(goal=(3.0, 0.0)))
    diag = planner.diagnostics()["last_decision"]
    assert diag.get("feasibility_slowdown_active") is False


def test_dwa_feasibility_slowdown_reduces_speed_near_obstacle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With high infeasible ratio, feasibility-slowdown reduces linear speed."""
    config = DWAPlannerConfig(
        feasibility_slowdown_enabled=True,
        feasibility_slowdown_infeasible_ratio=0.1,
        feasibility_slowdown_scale=0.3,
        linear_samples=5,
        angular_samples=5,
    )
    planner = DWAPlannerAdapter(config)
    monkeypatch.setattr(
        planner,
        "_rollout_score",
        lambda **kwargs: float("-inf") if kwargs["command"][0] > 0.081 else kwargs["command"][0],
    )
    observation = _observation(goal=(3.0, 0.0))
    slowdown_cmd = planner.plan(observation)
    diag = planner.diagnostics()["last_decision"]
    assert diag.get("feasibility_slowdown_active") is True
    assert slowdown_cmd[0] == pytest.approx(0.048)


def test_dwa_route_rescue_config_builder_parses_new_fields() -> None:
    """build_dwa_config parses the route-rescue and feasibility-slowdown fields."""
    config = build_dwa_config(
        {
            "route_rescue_enabled": "yes",
            "route_rescue_window": 30,
            "route_rescue_patience": 15,
            "route_rescue_progress_threshold": 0.1,
            "route_rescue_horizon_scale": 3.0,
            "route_rescue_progress_weight_boost": 2.5,
            "feasibility_slowdown_enabled": "true",
            "feasibility_slowdown_infeasible_ratio": 0.3,
            "feasibility_slowdown_scale": 0.6,
        }
    )
    assert config.route_rescue_enabled is True
    assert config.route_rescue_window == 30
    assert config.route_rescue_patience == 15
    assert config.route_rescue_progress_threshold == pytest.approx(0.1)
    assert config.route_rescue_horizon_scale == pytest.approx(3.0)
    assert config.route_rescue_progress_weight_boost == pytest.approx(2.5)
    assert config.feasibility_slowdown_enabled is True
    assert config.feasibility_slowdown_infeasible_ratio == pytest.approx(0.3)
    assert config.feasibility_slowdown_scale == pytest.approx(0.6)


def test_dwa_route_rescue_config_rejects_invalid_values() -> None:
    """Invalid route-rescue config values fail closed."""
    for field in (
        "route_rescue_progress_threshold",
        "route_rescue_horizon_scale",
        "route_rescue_progress_weight_boost",
        "feasibility_slowdown_infeasible_ratio",
        "feasibility_slowdown_scale",
    ):
        for value in (float("inf"), float("nan")):
            with pytest.raises(ValueError, match=field):
                DWAPlannerConfig(**{field: value})
    with pytest.raises(ValueError, match="route_rescue_window"):
        DWAPlannerConfig(route_rescue_window=0)
    with pytest.raises(ValueError, match="route_rescue_patience"):
        DWAPlannerConfig(route_rescue_patience=0)
    with pytest.raises(ValueError, match="route_rescue_patience"):
        DWAPlannerConfig(route_rescue_window=3, route_rescue_patience=4)
    with pytest.raises(ValueError, match="route_rescue_progress_threshold"):
        DWAPlannerConfig(route_rescue_progress_threshold=-0.1)
    with pytest.raises(ValueError, match="route_rescue_horizon_scale"):
        DWAPlannerConfig(route_rescue_horizon_scale=0.5)
    with pytest.raises(ValueError, match="route_rescue_progress_weight_boost"):
        DWAPlannerConfig(route_rescue_progress_weight_boost=0.5)
    with pytest.raises(ValueError, match="feasibility_slowdown_infeasible_ratio"):
        DWAPlannerConfig(feasibility_slowdown_infeasible_ratio=1.5)
    with pytest.raises(ValueError, match="feasibility_slowdown_scale"):
        DWAPlannerConfig(feasibility_slowdown_scale=0.0)
    with pytest.raises(ValueError, match="feasibility_slowdown_scale"):
        DWAPlannerConfig(feasibility_slowdown_scale=1.5)


def test_dwa_route_rescue_classic_config_has_rescue_disabled() -> None:
    """The canonical dwa_classic.yaml has route-rescue disabled."""
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "algos"
    classic = yaml.safe_load((config_dir / "dwa_classic.yaml").read_text(encoding="utf-8"))
    assert build_dwa_config(classic).route_rescue_enabled is False
    assert build_dwa_config(classic).feasibility_slowdown_enabled is False


def test_dwa_route_rescue_rescue_config_is_distinct_from_classic() -> None:
    """The dwa_route_rescue.yaml has rescue enabled and differs from classic only in intervention fields."""
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "algos"
    classic = yaml.safe_load((config_dir / "dwa_classic.yaml").read_text(encoding="utf-8"))
    rescue = yaml.safe_load((config_dir / "dwa_route_rescue.yaml").read_text(encoding="utf-8"))
    assert build_dwa_config(rescue).route_rescue_enabled is True
    assert build_dwa_config(rescue).feasibility_slowdown_enabled is True
    intervention_keys = {
        "route_rescue_enabled",
        "route_rescue_window",
        "route_rescue_patience",
        "route_rescue_progress_threshold",
        "route_rescue_horizon_scale",
        "route_rescue_progress_weight_boost",
        "feasibility_slowdown_enabled",
        "feasibility_slowdown_infeasible_ratio",
        "feasibility_slowdown_scale",
    }
    assert {key: value for key, value in classic.items() if key not in intervention_keys} == {
        key: value for key, value in rescue.items() if key not in intervention_keys
    }


def test_dwa_route_rescue_diagnostics_report_rescue_state() -> None:
    """Diagnostics include route-rescue and feasibility-slowdown state."""
    config = DWAPlannerConfig(route_rescue_enabled=True, route_rescue_patience=3)
    planner = DWAPlannerAdapter(config)
    _make_planner_stalled(planner, 3.0, 10)
    diag = planner.diagnostics()["last_decision"]
    assert "route_rescue_active" in diag
    assert "route_rescue_type" in diag
    assert "feasibility_slowdown_active" in diag
