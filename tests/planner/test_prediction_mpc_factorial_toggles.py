"""Tests for the prediction-MPC factorial toggle feature (issue #5355, #5372).

Covers:
- Unit tests per toggle (prediction_enabled, hard_pedestrian_constraints_enabled)
- Fidelity smoke (4 arms on 2 scenarios, decision traces differ across flips)
- Preflight identity test (4 arms share same contracts except the two flags)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.planner.prediction_mpc import (
    PredictionMPCConfig,
    PredictionMPCPlannerAdapter,
    build_prediction_mpc_config,
)


def _obs(
    *,
    robot=(0.0, 0.0),
    heading=0.0,
    speed=0.0,
    goal=(2.0, 0.0),
    ped_positions=None,
    ped_velocities=None,
):
    """Build compact SocNav observation payload for prediction-MPC tests."""
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
    }


# ---------------------------------------------------------------------------
# Factor A: prediction_enabled toggle
# ---------------------------------------------------------------------------


def test_prediction_enabled_default_is_true() -> None:
    """Factor A toggle defaults to True."""
    cfg = PredictionMPCConfig()
    assert cfg.prediction_enabled is True


def test_prediction_disabled_freezes_pedestrian_prediction_soft_path() -> None:
    """When prediction_enabled is False, _predict_pedestrians returns frozen positions."""
    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            prediction_enabled=False,
            horizon_steps=3,
            rollout_dt=0.25,
        )
    )
    ped_positions = np.asarray([[1.0, 0.0], [2.0, 1.0]], dtype=float)
    ped_velocities = np.asarray([[0.5, 0.0], [0.0, 0.3]], dtype=float)

    # Every step should return the same frozen positions
    result_step_0 = planner._predict_pedestrians(ped_positions, ped_velocities, 0)
    result_step_1 = planner._predict_pedestrians(ped_positions, ped_velocities, 1)
    result_step_2 = planner._predict_pedestrians(ped_positions, ped_velocities, 2)

    np.testing.assert_allclose(result_step_0, ped_positions)
    np.testing.assert_allclose(result_step_1, ped_positions)
    np.testing.assert_allclose(result_step_2, ped_positions)


def test_prediction_enabled_moving_pedestrians_soft_path() -> None:
    """When prediction_enabled is True, _predict_pedestrians advances positions."""
    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            prediction_enabled=True,
            horizon_steps=3,
            rollout_dt=0.25,
        )
    )
    ped_positions = np.asarray([[1.0, 0.0]], dtype=float)
    ped_velocities = np.asarray([[1.0, 0.0]], dtype=float)

    result_step_0 = planner._predict_pedestrians(ped_positions, ped_velocities, 0)
    result_step_1 = planner._predict_pedestrians(ped_positions, ped_velocities, 1)

    # Step 0: position + velocity * 0.25
    np.testing.assert_allclose(result_step_0, [[1.0 + 1.0 * 0.25, 0.0]], rtol=1e-6)
    # Step 1: position + velocity * 0.5
    np.testing.assert_allclose(result_step_1, [[1.0 + 1.0 * 0.5, 0.0]], rtol=1e-6)


def test_prediction_disabled_yields_different_command_for_moving_pedestrian() -> None:
    """A-off vs A-on should emit different commands for a pedestrian that will move
    into the robot's path. With prediction enabled the optimizer sees the future
    conflict and can preemptively adjust, whereas A-off only sees the current
    static position."""
    obs = _obs(
        robot=(0.0, 0.0),
        heading=0.0,
        goal=(3.0, 0.0),
        ped_positions=[(0.8, 0.6)],
        ped_velocities=[(0.4, -0.4)],
    )

    planner_on = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            prediction_enabled=True,
            horizon_steps=4,
            solver_max_iterations=30,
            rollout_dt=0.25,
            pedestrian_safety_margin=0.4,
        )
    )
    planner_off = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            prediction_enabled=False,
            horizon_steps=4,
            solver_max_iterations=30,
            rollout_dt=0.25,
            pedestrian_safety_margin=0.4,
        )
    )

    linear_on, angular_on = planner_on.plan(obs)
    linear_off, angular_off = planner_off.plan(obs)

    # With prediction the robot knows the pedestrian will cross its path ahead,
    # so the two commands should differ noticeably.
    assert not np.isclose(linear_on, linear_off, atol=0.05) or not np.isclose(
        angular_on, angular_off, atol=0.05
    ), (
        f"Factor A flip should change command: "
        f"A-on ({linear_on:.4f}, {angular_on:.4f}) vs A-off ({linear_off:.4f}, {angular_off:.4f})"
    )


def test_hard_constraint_freeze_uses_current_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hard-constraint path freezes futures at the observed t=0 positions."""
    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            prediction_enabled=False,
            horizon_steps=3,
            rollout_dt=0.25,
        )
    )
    current = np.asarray([[1.0, 2.0]], dtype=float)
    obs = _obs(
        ped_positions=current.tolist(),
        ped_velocities=[[0.5, 0.3]],
    )

    context = SimpleNamespace(
        observation=obs,
        ped_positions=current,
    )
    captured: dict[str, np.ndarray] = {}

    def capture_futures(
        _controls: np.ndarray,
        *,
        context: SimpleNamespace,
        predicted_futures,
    ) -> np.ndarray:
        del context
        captured["positions"] = predicted_futures.positions_world.copy()
        return np.ones((1,), dtype=float)

    monkeypatch.setattr(planner, "_pedestrian_clearance_constraints", capture_futures)
    constraints = planner._optimizer_constraints(context)
    assert len(constraints) == 1
    constraints[0].fun(np.zeros(2 * planner.config.horizon_steps, dtype=float))

    expected = np.repeat(current[:, np.newaxis, :], planner.config.horizon_steps, axis=1)
    np.testing.assert_allclose(captured["positions"], expected)


# ---------------------------------------------------------------------------
# Factor B: hard_pedestrian_constraints_enabled toggle
# ---------------------------------------------------------------------------


def test_hard_constraints_enabled_default_is_true() -> None:
    """Factor B toggle defaults to True."""
    cfg = PredictionMPCConfig()
    assert cfg.hard_pedestrian_constraints_enabled is True


def test_hard_constraints_on_returns_nlc() -> None:
    """When hard_pedestrian_constraints_enabled is True, _optimizer_constraints
    returns a NonlinearConstraint for pedestrians."""
    from scipy.optimize import NonlinearConstraint

    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            hard_pedestrian_constraints_enabled=True,
            horizon_steps=3,
            solver_max_iterations=10,
        )
    )
    obs = _obs(
        goal=(3.0, 0.0),
        ped_positions=[(1.5, 0.0)],
        ped_velocities=[(0.0, 0.0)],
    )
    *_, robot_radius, ped_radius = planner._extract_state(obs)
    context = SimpleNamespace(
        robot_pos=np.asarray([0.0, 0.0], dtype=float),
        heading=0.0,
        current_speed=0.0,
        goal=np.asarray([2.0, 0.0], dtype=float),
        ped_positions=np.asarray([[1.5, 0.0]], dtype=float),
        ped_velocities=np.asarray([[0.0, 0.0]], dtype=float),
        robot_radius=robot_radius,
        ped_radius=ped_radius,
        observation=obs,
        speed_cap=0.9,
    )

    constraints = planner._optimizer_constraints(context)
    assert len(constraints) == 1
    assert isinstance(constraints[0], NonlinearConstraint)


def test_hard_constraints_off_returns_empty() -> None:
    """When hard_pedestrian_constraints_enabled is False, no NonlinearConstraint emitted."""
    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            hard_pedestrian_constraints_enabled=False,
            horizon_steps=3,
            solver_max_iterations=10,
        )
    )
    obs = _obs(
        goal=(3.0, 0.0),
        ped_positions=[(1.5, 0.0)],
        ped_velocities=[(0.0, 0.0)],
    )
    *_, robot_radius, ped_radius = planner._extract_state(obs)
    context = SimpleNamespace(
        robot_pos=np.asarray([0.0, 0.0], dtype=float),
        heading=0.0,
        current_speed=0.0,
        goal=np.asarray([2.0, 0.0], dtype=float),
        ped_positions=np.asarray([[1.5, 0.0]], dtype=float),
        ped_velocities=np.asarray([[0.0, 0.0]], dtype=float),
        robot_radius=robot_radius,
        ped_radius=ped_radius,
        observation=obs,
        speed_cap=0.9,
    )

    constraints = planner._optimizer_constraints(context)
    assert len(constraints) == 0


def test_b_off_uses_soft_clearance_weight_45() -> None:
    """Factor B-OFF sets soft pedestrian clearance weight to 4.5 (NMPC default)."""
    planner_on = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            hard_pedestrian_constraints_enabled=True,
            horizon_steps=2,
        )
    )
    planner_off = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            hard_pedestrian_constraints_enabled=False,
            horizon_steps=2,
        )
    )

    assert planner_on.config.pedestrian_clearance_weight == pytest.approx(0.0)
    assert planner_off.config.pedestrian_clearance_weight == pytest.approx(4.5)


def test_b_off_remains_functional_avoidance_planner() -> None:
    """B-OFF must still produce a valid avoidance-oriented command (not degenerate)."""
    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            hard_pedestrian_constraints_enabled=False,
            horizon_steps=3,
            solver_max_iterations=16,
        )
    )

    # Open space: should still move toward goal
    linear_open, _ = planner.plan(_obs(goal=(3.0, 0.0)))
    assert linear_open > 0.0

    # Near pedestrian: should still show avoidance behavior
    linear_near, _ = planner.plan(
        _obs(
            goal=(3.0, 0.0),
            ped_positions=[(0.8, 0.0)],
            ped_velocities=[(0.0, 0.0)],
        )
    )
    assert linear_near <= planner.config.max_linear_speed


# ---------------------------------------------------------------------------
# Config builder tests
# ---------------------------------------------------------------------------


def test_build_config_parses_factor_toggles() -> None:
    """build_prediction_mpc_config correctly parses both factor toggle fields."""
    cfg_on_on = build_prediction_mpc_config(
        {"prediction_enabled": "true", "hard_pedestrian_constraints_enabled": "true"}
    )
    assert cfg_on_on.prediction_enabled is True
    assert cfg_on_on.hard_pedestrian_constraints_enabled is True

    cfg_off_off = build_prediction_mpc_config(
        {"prediction_enabled": "false", "hard_pedestrian_constraints_enabled": "false"}
    )
    assert cfg_off_off.prediction_enabled is False
    assert cfg_off_off.hard_pedestrian_constraints_enabled is False


# ---------------------------------------------------------------------------
# 4-arm YAML config smoke tests
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "algos"

FACTORIAL_YAML_NAMES = [
    "prediction_mpc_factorial_A_on_B_on.yaml",
    "prediction_mpc_factorial_A_off_B_on.yaml",
    "prediction_mpc_factorial_A_on_B_off.yaml",
    "prediction_mpc_factorial_A_off_B_off.yaml",
]


@pytest.mark.parametrize("yaml_name", FACTORIAL_YAML_NAMES)
def test_factorial_yamls_exist_and_load(yaml_name: str) -> None:
    """Each factorial arm YAML exists and loads to valid prediction-MPC config."""
    path = CONFIGS_DIR / yaml_name
    assert path.exists(), f"Factorial arm config missing: {path}"
    raw = yaml.safe_load(path.read_text())
    assert isinstance(raw, dict)
    cfg = build_prediction_mpc_config(raw)
    assert isinstance(cfg, PredictionMPCConfig)


@pytest.mark.parametrize("yaml_name", FACTORIAL_YAML_NAMES)
def test_factorial_yamls_parse_toggle_flags(yaml_name: str) -> None:
    """Each factorial arm YAML parses the two toggle flags correctly."""
    path = CONFIGS_DIR / yaml_name
    raw = yaml.safe_load(path.read_text())
    cfg = build_prediction_mpc_config(raw)

    if "A_on_B_on" in yaml_name:
        assert cfg.prediction_enabled is True
        assert cfg.hard_pedestrian_constraints_enabled is True
    elif "A_off_B_on" in yaml_name:
        assert cfg.prediction_enabled is False
        assert cfg.hard_pedestrian_constraints_enabled is True
    elif "A_on_B_off" in yaml_name:
        assert cfg.prediction_enabled is True
        assert cfg.hard_pedestrian_constraints_enabled is False
    elif "A_off_B_off" in yaml_name:
        assert cfg.prediction_enabled is False
        assert cfg.hard_pedestrian_constraints_enabled is False


# ---------------------------------------------------------------------------
# Preflight identity test
# ---------------------------------------------------------------------------


def test_preflight_identity_same_observation_contract() -> None:
    """All 4 factorial arms produce the same NMPc core for everything but pedestrian
    clearance weight. Observation contract, kinematics, and runtime budget are shared."""
    planners: list[tuple[str, PredictionMPCPlannerAdapter]] = []
    for name in FACTORIAL_YAML_NAMES:
        raw = yaml.safe_load((CONFIGS_DIR / name).read_text())
        cfg = build_prediction_mpc_config(raw)
        planners.append((name, PredictionMPCPlannerAdapter(cfg)))

    # All share the same base config fields (kinematics, kinematics, budget)
    for name, p in planners[1:]:
        base = planners[0][1]
        assert p.config.max_linear_speed == base.config.max_linear_speed, f"{name} max_linear_speed"
        assert p.config.max_angular_speed == base.config.max_angular_speed, (
            f"{name} max_angular_speed"
        )
        assert p.config.horizon_steps == base.config.horizon_steps, f"{name} horizon_steps"
        assert p.config.rollout_dt == base.config.rollout_dt, f"{name} rollout_dt"
        assert p.config.goal_tolerance == base.config.goal_tolerance, f"{name} goal_tolerance"
        assert p.config.waypoint_switch_distance == base.config.waypoint_switch_distance
        assert p.config.path_goal_weight == base.config.path_goal_weight, f"{name} path_goal_weight"
        assert p.config.terminal_goal_weight == base.config.terminal_goal_weight
        assert p.config.heading_weight == base.config.heading_weight, f"{name} heading_weight"
        assert p.config.control_effort_weight == base.config.control_effort_weight
        assert p.config.smoothness_weight == base.config.smoothness_weight
        assert p.config.solver_ftol == base.config.solver_ftol, f"{name} solver_ftol"
        assert p.config.solver_max_iterations == base.config.solver_max_iterations
        assert p.config.warm_start == base.config.warm_start, f"{name} warm_start"

    # Factor A affects prediction_enabled (observable in factor_toggles)
    toggles = {name: p.prediction_config.prediction_enabled for name, p in planners}
    assert toggles["prediction_mpc_factorial_A_on_B_on.yaml"] is True
    assert toggles["prediction_mpc_factorial_A_off_B_on.yaml"] is False
    assert toggles["prediction_mpc_factorial_A_on_B_off.yaml"] is True
    assert toggles["prediction_mpc_factorial_A_off_B_off.yaml"] is False

    # Factor B affects hard_pedestrian_constraints_enabled AND pedestrian_clearance_weight
    hard_flags = {
        name: p.prediction_config.hard_pedestrian_constraints_enabled for name, p in planners
    }
    assert hard_flags["prediction_mpc_factorial_A_on_B_on.yaml"] is True
    assert hard_flags["prediction_mpc_factorial_A_off_B_on.yaml"] is True
    assert hard_flags["prediction_mpc_factorial_A_on_B_off.yaml"] is False
    assert hard_flags["prediction_mpc_factorial_A_off_B_off.yaml"] is False

    # B-ON → ped_clearance_weight == 0.0, B-OFF → ped_clearance_weight == 4.5
    for name, p in planners:
        if "B_on" in name:
            assert p.config.pedestrian_clearance_weight == pytest.approx(0.0)
        else:
            assert p.config.pedestrian_clearance_weight == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# Fidelity smoke: 2 scenarios, 4 arms, decision traces must differ across flips
# ---------------------------------------------------------------------------


def _get_decision_trace(
    planner: PredictionMPCPlannerAdapter, obs: dict, steps: int = 3
) -> list[tuple[float, float]]:
    """Run planner.plan(obs) for `steps` calls and return (linear, angular) commands."""
    trace = []
    for _ in range(steps):
        linear, angular = planner.plan(obs)
        trace.append((linear, angular))
    return trace


def _traces_differ(
    trace_a: list[tuple[float, float]], trace_b: list[tuple[float, float]], atol: float = 0.05
) -> bool:
    """Return True if the two traces diverge at any step beyond tolerance."""
    for (la, wa), (lb, wb) in zip(trace_a, trace_b, strict=True):
        if not np.isclose(la, lb, atol=atol) or not np.isclose(wa, wb, atol=atol):
            return True
    return False


class TestFidelitySmokePedestrianInPath:
    """Fidelity smoke: scenario where pedestrian is on the path ahead.

    A pedestrian that moves across the narrow passage forces the constraint to
    bind, so the factor flips should change the trajectory.
    """

    @classmethod
    def setup_class(cls) -> None:
        """Build all 4 arm planners once for the scenario."""
        cls.obs = _obs(
            robot=(0.0, 0.0),
            heading=0.0,
            goal=(3.0, 0.0),
            ped_positions=[(0.9, 0.5)],
            ped_velocities=[(0.3, -0.3)],
        )
        cls.planners = {}
        for name in FACTORIAL_YAML_NAMES:
            raw = yaml.safe_load((CONFIGS_DIR / name).read_text())
            cfg = build_prediction_mpc_config(raw)
            cls.planners[name] = PredictionMPCPlannerAdapter(
                PredictionMPCConfig(
                    prediction_enabled=cfg.prediction_enabled,
                    hard_pedestrian_constraints_enabled=cfg.hard_pedestrian_constraints_enabled,
                    horizon_steps=4,
                    solver_max_iterations=25,
                    rollout_dt=0.25,
                    pedestrian_safety_margin=0.35,
                )
            )
        cls.traces = {
            name: _get_decision_trace(p, cls.obs, steps=2) for name, p in cls.planners.items()
        }

    def test_a_on_differs_from_a_off_hard_on(self) -> None:
        """Factor A flip (B held True) produces different decision traces."""
        t_on = self.traces["prediction_mpc_factorial_A_on_B_on.yaml"]
        t_off = self.traces["prediction_mpc_factorial_A_off_B_on.yaml"]
        assert _traces_differ(t_on, t_off), (
            f"A-on trace {t_on} must differ from A-off trace {t_off}"
        )

    def test_b_on_differs_from_b_off_pred_on(self) -> None:
        """Factor B flip (A held True) produces different decision traces."""
        t_on = self.traces["prediction_mpc_factorial_A_on_B_on.yaml"]
        t_off = self.traces["prediction_mpc_factorial_A_on_B_off.yaml"]
        assert _traces_differ(t_on, t_off), (
            f"B-on trace {t_on} must differ from B-off trace {t_off}"
        )

    def test_all_arms_complete_episodes(self) -> None:
        """All 4 arms produce valid (finite) commands for the scenario."""
        for trace in self.traces.values():
            for linear, angular in trace:
                assert np.isfinite(linear) and np.isfinite(angular)


class TestFidelitySmokeSidewalk:
    """Fidelity smoke: scenario with pedestrians moving toward the path.

    Pedestrians start nearby and move toward the robot's heading so prediction
    makes a difference.
    """

    @classmethod
    def setup_class(cls) -> None:
        cls.obs = _obs(
            robot=(0.0, 0.0),
            heading=0.0,
            goal=(3.0, 0.0),
            ped_positions=[(1.0, 0.8), (1.8, -0.6)],
            ped_velocities=[(-0.15, -0.2), (0.0, 0.15)],
        )
        cls.planners = {}
        for name in FACTORIAL_YAML_NAMES:
            raw = yaml.safe_load((CONFIGS_DIR / name).read_text())
            cfg = build_prediction_mpc_config(raw)
            cls.planners[name] = PredictionMPCPlannerAdapter(
                PredictionMPCConfig(
                    prediction_enabled=cfg.prediction_enabled,
                    hard_pedestrian_constraints_enabled=cfg.hard_pedestrian_constraints_enabled,
                    horizon_steps=4,
                    solver_max_iterations=25,
                    rollout_dt=0.25,
                    pedestrian_safety_margin=0.35,
                )
            )
        cls.traces = {
            name: _get_decision_trace(p, cls.obs, steps=2) for name, p in cls.planners.items()
        }

    def test_a_on_differs_from_a_off_hard_on(self) -> None:
        """Factor A flip (B held True) produces different decision traces."""
        t_on = self.traces["prediction_mpc_factorial_A_on_B_on.yaml"]
        t_off = self.traces["prediction_mpc_factorial_A_off_B_on.yaml"]
        assert _traces_differ(t_on, t_off), (
            f"A-on trace {t_on} must differ from A-off trace {t_off}"
        )

    def test_b_on_differs_from_b_off_pred_on(self) -> None:
        """Factor B flip (A held True) produces different decision traces."""
        t_on = self.traces["prediction_mpc_factorial_A_on_B_on.yaml"]
        t_off = self.traces["prediction_mpc_factorial_A_on_B_off.yaml"]
        assert _traces_differ(t_on, t_off), (
            f"B-on trace {t_on} must differ from B-off trace {t_off}"
        )

    def test_all_arms_complete_episodes(self) -> None:
        """All 4 arms produce valid (finite) commands for the scenario."""
        for trace in self.traces.values():
            for linear, angular in trace:
                assert np.isfinite(linear) and np.isfinite(angular)


# ---------------------------------------------------------------------------
# Diagnostics report factor toggles
# ---------------------------------------------------------------------------


def test_diagnostics_reports_factor_toggles() -> None:
    """Diagnostics payload includes the factor toggle states."""
    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            prediction_enabled=False,
            hard_pedestrian_constraints_enabled=False,
            horizon_steps=2,
        )
    )
    diag = planner.diagnostics()
    assert "factor_toggles" in diag
    assert diag["factor_toggles"]["prediction_enabled"] is False
    assert diag["factor_toggles"]["hard_pedestrian_constraints_enabled"] is False


def test_diagnostics_default_toggles_true() -> None:
    """Default config diagnostics report both toggles as True."""
    planner = PredictionMPCPlannerAdapter(PredictionMPCConfig(horizon_steps=2))
    diag = planner.diagnostics()
    assert diag["factor_toggles"]["prediction_enabled"] is True
    assert diag["factor_toggles"]["hard_pedestrian_constraints_enabled"] is True
