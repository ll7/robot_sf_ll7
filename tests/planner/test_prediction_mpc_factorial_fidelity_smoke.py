"""Fidelity smoke for the issue #5355 prediction-MPC factorial (Refs #5372).

Claim boundary: diagnostic-only, CPU, tiny-horizon fidelity smoke. This module
does **not** run the campaign, touch GPU/Slurm, archive rows, or make any
navigation-quality claim. It proves the two preregistered mechanism toggles
(Factor A = pedestrian prediction, Factor B = hard pedestrian constraints) each
demonstrably change planner behavior, that the knobs are orthogonal, and that
the B-OFF arms remain functional soft-cost planners rather than degenerate ones.

The four canonical arm configs
(``configs/algos/prediction_mpc_factorial_A{0,1}_B{0,1}.yaml``) already exist on
main; this smoke consumes them as-is via ``build_prediction_mpc_config``.

Fidelity requirements implemented here mirror
``docs/context/issue_5355_factorial_preregistration.md`` §6:

- §6.1 Toggle-effect smoke on two named scenarios
  (``classic_doorway`` static-structural, ``francis2023_circular_crossing``
  dynamic): each factor flip changes the decision trace; the freeze path is a
  correctness no-op where pedestrians are static; the toggles are orthogonal.
- §6.2 B-OFF functionality check: the B-OFF arms (A0_B0, A1_B0) complete
  episodes with non-degenerate command statistics and a positive shared soft
  pedestrian clearance weight (``W_soft = 4.5``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from robot_sf.planner.prediction_mpc import (
    PredictionMPCConfig,
    PredictionMPCPlannerAdapter,
    build_prediction_mpc_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALGO_DIR = _REPO_ROOT / "configs" / "algos"
_ARM_IDS = ("A0_B0", "A0_B1", "A1_B0", "A1_B1")


def _arm_config(arm_id: str) -> PredictionMPCConfig:
    """Load a canonical arm config through the production YAML builder."""
    path = _ALGO_DIR / f"prediction_mpc_factorial_{arm_id}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path} did not parse to a mapping"
    return build_prediction_mpc_config(payload)


def _arm(arm_id: str) -> PredictionMPCPlannerAdapter:
    """Build a fresh planner adapter for the given canonical arm."""
    adapter = PredictionMPCPlannerAdapter(_arm_config(arm_id))
    adapter.reset()
    return adapter


def _obs(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    speed: float = 0.0,
    goal: tuple[float, float] = (2.5, 0.0),
    ped_positions: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Build a compact SocNav observation (pedestrian velocities are ego-frame)."""
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


# Two named scenarios from the preregistered smoke set (§6.1). ``classic_doorway``
# is static-structural (a stationary pedestrian near the path); the crossing
# scenario is dynamic (a pedestrian with non-zero ego velocity moving toward the
# robot's straight-line path). Both place a pedestrian within the safety margin so
# every factor flip is expressible.
_SMOKE_SCENARIOS: dict[str, dict[str, Any]] = {
    "classic_doorway": {
        "goal": (2.5, 0.0),
        "ped_positions": [(1.1, 0.3)],
        "ped_velocities": [(0.0, 0.0)],
    },
    "francis2023_circular_crossing": {
        "goal": (2.5, 0.0),
        "ped_positions": [(1.2, 0.55)],
        "ped_velocities": [(0.0, -0.5)],
    },
}


def _predicted_futures(adapter: PredictionMPCPlannerAdapter, obs: dict[str, Any]) -> np.ndarray:
    """Return the pedestrian futures fed to the planner for this arm."""
    futures = adapter._future_predictor.predict(
        obs,
        horizon_steps=max(int(adapter.prediction_config.horizon_steps), 1),
        dt=float(adapter.prediction_config.rollout_dt),
    )
    return np.asarray(futures.positions_world, dtype=float)


def _clearance_context(
    adapter: PredictionMPCPlannerAdapter, obs: dict[str, Any]
) -> SimpleNamespace:
    """Build a rollout context for direct hard-constraint evaluation."""
    *_, robot_radius, ped_radius = adapter._extract_state(obs)
    return SimpleNamespace(
        robot_pos=np.asarray(obs["robot"]["position"], dtype=float),
        heading=float(np.asarray(obs["robot"]["heading"], dtype=float).reshape(-1)[0]),
        current_speed=float(np.asarray(obs["robot"]["speed"], dtype=float).reshape(-1)[0]),
        goal=np.asarray(obs["goal"]["current"], dtype=float),
        ped_positions=np.asarray(obs["pedestrians"]["positions"], dtype=float),
        ped_velocities=np.asarray(obs["pedestrians"]["velocities"], dtype=float),
        robot_radius=robot_radius,
        ped_radius=ped_radius,
        observation=obs,
        speed_cap=float(adapter.prediction_config.max_linear_speed),
    )


def _has_active_hard_constraint(adapter: PredictionMPCPlannerAdapter, obs: dict[str, Any]) -> bool:
    """Return True when the arm hands a non-empty pedestrian constraint set to SLSQP."""
    return len(adapter._optimizer_constraints(_clearance_context(adapter, obs))) > 0


# --------------------------------------------------------------------------- #
# §6.1 Toggle-effect smoke
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("b_level", ["B0", "B1"])
def test_factor_a_bites_on_dynamic_scenario(b_level: str) -> None:
    """Flipping Factor A changes predicted futures and the first-step command.

    On a dynamic scenario (pedestrian with non-zero velocity), prediction-on
    feeds constant-velocity forward futures while prediction-off freezes the
    pedestrian at its current position, and this changes at least some first-step
    commands (prereg §6.1, "Factor A bites").
    """
    scenario = _SMOKE_SCENARIOS["francis2023_circular_crossing"]
    a_off, a_on = _arm(f"A0_{b_level}"), _arm(f"A1_{b_level}")
    obs = _obs(**scenario)

    fut_off = _predicted_futures(a_off, obs)
    fut_on = _predicted_futures(a_on, obs)
    assert not np.allclose(fut_off, fut_on), (
        "Factor A must change the pedestrian futures fed to the planner"
    )
    # A-off holds the pedestrian at its current position; A-on displaces it forward.
    assert np.allclose(fut_off[:, -1, :], fut_off[:, 0, :]), "A-off must freeze futures"
    assert not np.allclose(fut_on[:, -1, :], fut_on[:, 0, :]), "A-on must move futures forward"

    cmd_off = a_off.plan(obs)
    cmd_on = a_on.plan(obs)
    assert not np.allclose(cmd_off, cmd_on), "Factor A must change the first-step command"


@pytest.mark.parametrize("b_level", ["B0", "B1"])
def test_factor_a_freeze_path_coincides_on_static_scenario(b_level: str) -> None:
    """Where pedestrians are static, A-on and A-off traces coincide (freeze-path sanity).

    Correctness check from prereg §6.1: a constant-velocity predictor with zero
    pedestrian velocity must reproduce the frozen-position baseline bit-for-bit,
    so prediction-on and prediction-off cannot diverge on a static scene.
    """
    scenario = _SMOKE_SCENARIOS["classic_doorway"]
    a_off, a_on = _arm(f"A0_{b_level}"), _arm(f"A1_{b_level}")
    obs = _obs(**scenario)

    assert np.allclose(_predicted_futures(a_off, obs), _predicted_futures(a_on, obs))
    assert np.allclose(a_off.plan(obs), a_on.plan(obs))


@pytest.mark.parametrize("a_level", ["A0", "A1"])
@pytest.mark.parametrize("scenario_name", list(_SMOKE_SCENARIOS))
def test_factor_b_bites(a_level: str, scenario_name: str) -> None:
    """Flipping Factor B swaps hard constraints for a soft penalty and changes the trace.

    B-on evaluates a non-empty active pedestrian constraint set and carries soft
    weight 0.0; B-off returns no hard constraint and carries ``W_soft = 4.5``.
    With a pedestrian inside the safety margin the first-step command must differ
    (prereg §6.1, "Factor B bites").
    """
    b_off, b_on = _arm(f"{a_level}_B0"), _arm(f"{a_level}_B1")
    obs = _obs(**_SMOKE_SCENARIOS[scenario_name])

    # Mechanism-level: hard-constraint presence and soft weight flip with B.
    assert _has_active_hard_constraint(b_on, obs), "B-on must supply a hard pedestrian constraint"
    assert not _has_active_hard_constraint(b_off, obs), "B-off must drop the hard constraint set"
    assert b_on.prediction_config.pedestrian_clearance_weight == 0.0
    assert b_off.prediction_config.pedestrian_clearance_weight == 4.5

    # Behavior-level: the emitted decision differs where a pedestrian is within margin.
    assert not np.allclose(b_off.plan(obs), b_on.plan(obs)), (
        "Factor B must change the first-step command near a pedestrian"
    )


@pytest.mark.parametrize("scenario_name", list(_SMOKE_SCENARIOS))
def test_toggles_are_orthogonal(scenario_name: str) -> None:
    """Purity: A selects the prediction mode, B selects the constraint machinery, independently.

    Toggling A must not change hard-constraint selection, and toggling B must not
    change the prediction mode (prereg §6.1, "Purity").
    """
    obs = _obs(**_SMOKE_SCENARIOS[scenario_name])

    # Hard-constraint presence depends only on B (identical across A at fixed B).
    for b_level, expected in (("B0", False), ("B1", True)):
        a0 = _has_active_hard_constraint(_arm(f"A0_{b_level}"), obs)
        a1 = _has_active_hard_constraint(_arm(f"A1_{b_level}"), obs)
        assert a0 == a1 == expected, f"hard-constraint selection must depend only on B ({b_level})"

    # Prediction mode depends only on A (identical futures across B at fixed A).
    for a_level in ("A0", "A1"):
        fut_b0 = _predicted_futures(_arm(f"{a_level}_B0"), obs)
        fut_b1 = _predicted_futures(_arm(f"{a_level}_B1"), obs)
        assert np.allclose(fut_b0, fut_b1), f"prediction mode must depend only on A ({a_level})"


# --------------------------------------------------------------------------- #
# §6.2 B-OFF functionality check
# --------------------------------------------------------------------------- #


def _wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _run_episode(
    adapter: PredictionMPCPlannerAdapter,
    *,
    goal: tuple[float, float],
    ped_positions: list[tuple[float, float]],
    ped_velocities_world: list[tuple[float, float]],
    max_steps: int = 60,
) -> dict[str, Any]:
    """Roll out a tiny CPU episode with the unicycle integrator from prereg §2.2.

    Pedestrians advance in the world frame; the SocNav observation carries their
    velocities rotated into the ego frame, matching the observation contract.
    """
    cfg = adapter.prediction_config
    dt = float(cfg.rollout_dt)
    tol = float(cfg.goal_tolerance)
    goal_xy = np.asarray(goal, dtype=float)
    robot = np.asarray((0.0, 0.0), dtype=float)
    heading = 0.0
    speed = 0.0
    ped = np.asarray(ped_positions, dtype=float).reshape(-1, 2)
    ped_vel_world = np.asarray(ped_velocities_world, dtype=float).reshape(-1, 2)
    min_clearance = float("inf")
    completed = False
    steps = 0
    for step in range(max_steps):
        steps = step + 1
        cos_h, sin_h = np.cos(-heading), np.sin(-heading)
        rot = np.asarray([[cos_h, -sin_h], [sin_h, cos_h]], dtype=float)
        ped_vel_ego = ped_vel_world @ rot.T if len(ped) else np.zeros((0, 2), dtype=float)
        obs = _obs(
            robot=tuple(robot),
            heading=heading,
            speed=speed,
            goal=goal,
            ped_positions=[tuple(p) for p in ped],
            ped_velocities=[tuple(v) for v in ped_vel_ego],
        )
        linear, angular = adapter.plan(obs)
        heading = _wrap(heading + angular * dt)
        robot = robot + np.asarray([linear * np.cos(heading), linear * np.sin(heading)]) * dt
        speed = linear
        if len(ped):
            ped = ped + ped_vel_world * dt
            min_clearance = min(min_clearance, float(np.min(np.linalg.norm(ped - robot, axis=1))))
        if float(np.linalg.norm(goal_xy - robot)) <= tol:
            completed = True
            break
    diag = adapter.diagnostics()
    return {
        "completed": completed,
        "steps": steps,
        "min_clearance": min_clearance,
        "diagnostics": diag,
    }


# Episode geometry: a clear straight route with a pedestrian off the direct line
# (static for the doorway, crossing behind for the dynamic scene). A functional
# soft-cost planner reaches the goal; a degenerate one would stall or collide.
_EPISODE_SCENARIOS: dict[str, dict[str, Any]] = {
    "classic_doorway": {
        "goal": (2.5, 0.0),
        "ped_positions": [(1.3, 0.9)],
        "ped_velocities_world": [(0.0, 0.0)],
    },
    "francis2023_circular_crossing": {
        "goal": (2.5, 0.0),
        "ped_positions": [(1.3, -1.6)],
        "ped_velocities_world": [(0.0, 0.30)],
    },
}


@pytest.mark.parametrize("arm_id", ["A0_B0", "A1_B0"])
@pytest.mark.parametrize("scenario_name", list(_EPISODE_SCENARIOS))
def test_b_off_arms_complete_episodes(arm_id: str, scenario_name: str) -> None:
    """B-OFF arms are functional soft-cost planners, not degenerate ones (prereg §6.2).

    Asserts non-trivial completion, non-degenerate command statistics, no fallback
    saturation, a positive shared soft pedestrian weight, and a non-collapsed
    min-clearance distribution on the smoke scenarios.
    """
    adapter = _arm(arm_id)
    # The maintainer's load-bearing condition: soft pedestrian avoidance stays active.
    assert adapter.prediction_config.hard_pedestrian_constraints_enabled is False
    assert adapter.prediction_config.pedestrian_clearance_weight == 4.5

    result = _run_episode(adapter, **_EPISODE_SCENARIOS[scenario_name])
    diag = result["diagnostics"]

    assert result["completed"], f"{arm_id} failed to complete {scenario_name} in the step budget"
    assert diag["nonzero_command_count"] > 0, "planner must issue non-zero commands"
    assert diag["mean_abs_linear"] > 0.0, "planner must make forward progress"
    assert diag["fallback_stop_count"] < diag["calls"], "fallback-to-stop must not be saturated"
    assert result["min_clearance"] > 0.0, "B-OFF soft avoidance must not collapse to collision"
