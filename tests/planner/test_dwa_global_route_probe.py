"""Contract tests for the DWA global-route integration probe (issue #5331)."""

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
    route_waypoints: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Build a minimal structured observation accepted by DWA."""
    positions = [] if pedestrians is None else pedestrians
    obs: dict[str, object] = {
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
    if route_waypoints is not None:
        obs["robot"]["route_waypoints"] = np.asarray(route_waypoints, dtype=float)  # type: ignore[index]
    return obs


def test_global_route_probe_default_off() -> None:
    """Global-route probe is disabled by default."""
    config = DWAPlannerConfig()
    assert config.global_route_probe_enabled is False
    assert config.global_route_probe_waypoint_distance == pytest.approx(2.0)
    assert config.global_route_probe_heading_weight == pytest.approx(0.5)


def test_global_route_probe_enabled_changes_command() -> None:
    """Enabling the global-route probe changes command selection when waypoints present."""
    base_config = DWAPlannerConfig(global_route_probe_enabled=False)
    probe_config = DWAPlannerConfig(
        global_route_probe_enabled=True,
        global_route_probe_heading_weight=2.0,
    )

    waypoints = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]
    observation = _observation(goal=(5.0, 0.0), route_waypoints=waypoints)

    base_command = DWAPlannerAdapter(base_config).plan(observation)
    probe_command = DWAPlannerAdapter(probe_config).plan(observation)

    assert base_command[0] > 0.0
    assert probe_command[0] > 0.0
    assert base_command != probe_command


def test_global_route_probe_without_waypoints_falls_back_to_base() -> None:
    """Without route_waypoints in observation, probe-enabled DWA uses base scoring."""
    config = DWAPlannerConfig(global_route_probe_enabled=True)
    observation = _observation(goal=(3.0, 0.0))

    command = DWAPlannerAdapter(config).plan(observation)
    assert command[0] > 0.0


def test_global_route_probe_diagnostics_record_activation() -> None:
    """Diagnostics record whether the global-route probe activated."""
    config = DWAPlannerConfig(global_route_probe_enabled=True)
    planner = DWAPlannerAdapter(config)

    waypoints = [(1.0, 0.0), (2.0, 0.0)]
    observation = _observation(goal=(3.0, 0.0), route_waypoints=waypoints)
    planner.plan(observation)
    diag = planner.diagnostics()
    last = diag.get("last_decision", {})
    assert last.get("global_route_probe_activated") is True


def test_global_route_probe_diagnostics_no_activation_without_waypoints() -> None:
    """Diagnostics record probe not activated when no waypoints present."""
    config = DWAPlannerConfig(global_route_probe_enabled=True)
    planner = DWAPlannerAdapter(config)

    observation = _observation(goal=(3.0, 0.0))
    planner.plan(observation)
    diag = planner.diagnostics()
    last = diag.get("last_decision", {})
    assert last.get("global_route_probe_activated") is False


def test_global_route_probe_diagnostics_no_activation_when_disabled() -> None:
    """Diagnostics record probe not activated when probe disabled."""
    config = DWAPlannerConfig(global_route_probe_enabled=False)
    planner = DWAPlannerAdapter(config)

    waypoints = [(1.0, 0.0), (2.0, 0.0)]
    observation = _observation(goal=(3.0, 0.0), route_waypoints=waypoints)
    planner.plan(observation)
    diag = planner.diagnostics()
    last = diag.get("last_decision", {})
    assert last.get("global_route_probe_activated") is False


def test_global_route_probe_config_validation_rejects_non_finite() -> None:
    """Config validation rejects non-finite probe values."""
    with pytest.raises(ValueError, match="must be finite"):
        DWAPlannerConfig(global_route_probe_waypoint_distance=float("nan"))
    with pytest.raises(ValueError, match="must be finite"):
        DWAPlannerConfig(global_route_probe_heading_weight=float("inf"))


def test_global_route_probe_config_validation_rejects_non_positive_distance() -> None:
    """Config validation rejects non-positive waypoint distance."""
    with pytest.raises(ValueError, match="must be positive"):
        DWAPlannerConfig(global_route_probe_waypoint_distance=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        DWAPlannerConfig(global_route_probe_waypoint_distance=-1.0)


def test_global_route_probe_config_validation_rejects_negative_weight() -> None:
    """Config validation rejects negative heading weight."""
    with pytest.raises(ValueError, match="must not be negative"):
        DWAPlannerConfig(global_route_probe_heading_weight=-0.1)


def test_global_route_probe_config_builder_parses_new_fields() -> None:
    """Config builder correctly parses global-route probe fields."""
    config = build_dwa_config(
        {
            "global_route_probe_enabled": True,
            "global_route_probe_waypoint_distance": 3.0,
            "global_route_probe_heading_weight": 0.8,
        }
    )
    assert config.global_route_probe_enabled is True
    assert config.global_route_probe_waypoint_distance == pytest.approx(3.0)
    assert config.global_route_probe_heading_weight == pytest.approx(0.8)


def test_global_route_probe_config_builder_defaults_when_omitted() -> None:
    """Config builder uses defaults when probe fields are omitted."""
    config = build_dwa_config({})
    assert config.global_route_probe_enabled is False
    assert config.global_route_probe_waypoint_distance == pytest.approx(2.0)
    assert config.global_route_probe_heading_weight == pytest.approx(0.5)


def test_global_route_probe_config_builder_rejects_non_boolean() -> None:
    """Config builder rejects non-boolean for global_route_probe_enabled."""
    with pytest.raises(ValueError, match="global_route_probe_enabled must be a boolean"):
        build_dwa_config({"global_route_probe_enabled": 1})


def test_global_route_probe_config_is_opt_in_in_canonical_configs() -> None:
    """The classic config retains global-route probe disabled."""
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "algos"
    classic = yaml.safe_load((config_dir / "dwa_classic.yaml").read_text(encoding="utf-8"))
    probe = yaml.safe_load((config_dir / "dwa_global_route_probe.yaml").read_text(encoding="utf-8"))

    assert build_dwa_config(classic).global_route_probe_enabled is False
    assert build_dwa_config(probe).global_route_probe_enabled is True


def test_global_route_probe_ignores_malformed_waypoints() -> None:
    """Probe gracefully ignores malformed waypoint data."""
    config = DWAPlannerConfig(global_route_probe_enabled=True)
    planner = DWAPlannerAdapter(config)

    observation = _observation(goal=(3.0, 0.0))
    observation["robot"]["route_waypoints"] = np.array([])  # type: ignore[index]
    command = planner.plan(observation)
    assert command[0] > 0.0


def test_global_route_probe_ignores_null_robot_state() -> None:
    """A null structured robot payload fails closed without raising."""
    planner = DWAPlannerAdapter(DWAPlannerConfig(global_route_probe_enabled=True))
    observation = _observation(goal=(3.0, 0.0))
    observation["robot"] = None
    assert (
        planner._waypoint_following_score(
            robot_pos=np.zeros(2),
            heading=0.0,
            end_position=np.zeros(2),
            end_orientation=0.0,
            observation=observation,
        )
        == 0.0
    )


def test_global_route_probe_targets_next_waypoint_after_nearest() -> None:
    """The probe advances past the nearest waypoint to preserve route direction."""
    planner = DWAPlannerAdapter(DWAPlannerConfig(global_route_probe_enabled=True))
    observation = _observation(route_waypoints=[(-0.1, 0.0), (1.0, 0.0)])
    assert (
        planner._waypoint_following_score(
            robot_pos=np.zeros(2),
            heading=0.0,
            end_position=np.zeros(2),
            end_orientation=0.0,
            observation=observation,
        )
        > 0.9
    )


def test_global_route_probe_ignores_nan_waypoints() -> None:
    """Probe gracefully ignores NaN waypoint data."""
    config = DWAPlannerConfig(global_route_probe_enabled=True)
    planner = DWAPlannerAdapter(config)

    observation = _observation(goal=(3.0, 0.0))
    observation["robot"]["route_waypoints"] = np.array([[float("nan"), 0.0]])  # type: ignore[index]
    command = planner.plan(observation)
    assert command[0] > 0.0


def test_global_route_probe_ignores_out_of_range_waypoints() -> None:
    """Probe ignores waypoints beyond the configured distance threshold."""
    config = DWAPlannerConfig(
        global_route_probe_enabled=True,
        global_route_probe_waypoint_distance=1.0,
    )
    planner = DWAPlannerAdapter(config)

    waypoints = [(10.0, 10.0)]
    observation = _observation(goal=(3.0, 0.0), route_waypoints=waypoints)
    command = planner.plan(observation)
    assert command[0] > 0.0


def test_global_route_probe_deterministic() -> None:
    """Probe produces deterministic results for identical observations."""
    config = DWAPlannerConfig(global_route_probe_enabled=True)
    planner = DWAPlannerAdapter(config)

    waypoints = [(1.0, 0.0), (2.0, 0.5)]
    observation = _observation(goal=(3.0, 0.0), route_waypoints=waypoints)

    first = planner.plan(observation)
    second = planner.plan(observation)
    assert first == second
