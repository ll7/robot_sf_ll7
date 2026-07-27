"""Direct unit coverage for :mod:`robot_sf.planner.fast_pysf_planner`.

These tests lock the public and integration contracts of
``FastPysfPlannerPolicy`` without starting a simulator or loading maps:

* zero-action branches for missing, out-of-range, and within-tolerance goals,
* the desired-motion plus weighted interaction-force composition,
* max-force magnitude clipping,
* simulator-timestep use in both desired-speed and force integration,
* adapter bounds forwarded from the robot configuration,
* ``predict`` returning ``(action, None)``,
* ``reset`` remaining a no-op, and
* the warn-once behavior for a missing goal.

The real :class:`~robot_sf.sim.fast_pysf_wrapper.FastPysfWrapper` and the
:func:`~robot_sf.robot.action_adapters.holonomic_to_diff_drive_action` adapter
are mocked, and the simulator/robot surfaces are lightweight fakes, so no
environment, map, or PySocialForce simulator is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robot_sf.planner.fast_pysf_planner import (
    FastPysfPlannerConfig,
    FastPysfPlannerPolicy,
)

PLANNER_MODULE = "robot_sf.planner.fast_pysf_planner"


@dataclass
class _FakeRobotConfig:
    """Minimal robot config exposing the adapter bound attributes."""

    max_linear_speed: float = 2.0
    max_angular_speed: float = 1.0


@dataclass
class _FakeRobot:
    """Minimal robot surface used by the planner's ``action`` method."""

    pose: tuple[tuple[float, float], float]
    current_speed: tuple[float, float] = (0.0, 0.0)
    config: _FakeRobotConfig = field(default_factory=_FakeRobotConfig)


@dataclass
class _FakeSimConfig:
    """Minimal simulator config exposing the timestep attribute."""

    time_per_step_in_secs: float = 0.1


@dataclass
class _FakeSimulator:
    """Minimal simulator surface consumed by ``FastPysfPlannerPolicy``."""

    robots: list[_FakeRobot]
    goal_pos: list[Any]
    config: _FakeSimConfig = field(default_factory=_FakeSimConfig)
    # ``pysf_sim`` is only passed to the mocked FastPysfWrapper constructor.
    pysf_sim: Any = None


def _build_policy(
    simulator: _FakeSimulator,
    *,
    wrapper: MagicMock | None = None,
    config: FastPysfPlannerConfig | None = None,
    adapter_config: Any | None = None,
    robot_index: int = 0,
) -> tuple[FastPysfPlannerPolicy, MagicMock]:
    """Construct a policy with ``FastPysfWrapper`` patched to a mock.

    Patching is only required during construction because the wrapper is built
    from ``simulator.pysf_sim`` in ``__init__``. The retained mock is returned
    alongside the policy so individual tests can stub ``get_forces_at``.
    """
    wrapper = wrapper if wrapper is not None else MagicMock()
    with patch(f"{PLANNER_MODULE}.FastPysfWrapper", return_value=wrapper):
        policy = FastPysfPlannerPolicy(
            simulator,
            robot_index=robot_index,
            config=config,
            adapter_config=adapter_config,
        )
    return policy, wrapper


def _patched_adapter(return_value: np.ndarray | None = None) -> tuple[MagicMock, patch]:
    """Build a patch target for the diff-drive adapter and its return value."""
    patcher = patch(f"{PLANNER_MODULE}.holonomic_to_diff_drive_action")
    mock = patcher.start()
    mock.return_value = np.zeros(2, dtype=float) if return_value is None else return_value
    return mock, patcher


# ---------------------------------------------------------------------------
# Timestep source
# ---------------------------------------------------------------------------


def test_dt_reads_simulator_time_per_step():
    """The ``_dt`` property must mirror ``simulator.config.time_per_step_in_secs``."""
    sim = _FakeSimulator(
        robots=[_FakeRobot(pose=((0.0, 0.0), 0.0))],
        goal_pos=[(5.0, 0.0)],
        config=_FakeSimConfig(time_per_step_in_secs=0.25),
    )
    policy, _ = _build_policy(sim)

    assert policy._dt == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# reset / no-op
# ---------------------------------------------------------------------------


def test_reset_is_noop_returns_none_and_preserves_state():
    """``reset`` must be a no-op: it returns ``None`` and changes no state."""
    sim = _FakeSimulator(
        robots=[_FakeRobot(pose=((0.0, 0.0), 0.0))],
        goal_pos=[],
    )
    policy, wrapper = _build_policy(sim)
    policy._warned_missing_goal = True  # sentinel to prove reset does not touch state

    result = policy.reset()

    assert result is None
    assert policy._warned_missing_goal is True
    assert policy._wrapper is wrapper


# ---------------------------------------------------------------------------
# Zero-action branches
# ---------------------------------------------------------------------------


def test_action_returns_zero_and_skips_forces_when_goal_list_empty():
    """An empty goal list must short-circuit to a zero action."""
    sim = _FakeSimulator(
        robots=[_FakeRobot(pose=((0.0, 0.0), 0.0))],
        goal_pos=[],
    )
    policy, wrapper = _build_policy(sim)
    adapter, adapter_patcher = _patched_adapter()
    try:
        action = policy.action()
    finally:
        adapter_patcher.stop()

    np.testing.assert_array_equal(action, np.zeros(2, dtype=float))
    wrapper.get_forces_at.assert_not_called()
    adapter.assert_not_called()


def test_action_returns_zero_when_robot_index_out_of_range():
    """A goal list shorter than the robot roster must short-circuit to a zero action.

    The planner reads ``robots[robot_index]`` first, so the out-of-range goal
    contract is exercised when ``robot_index`` is valid for the robots but at
    or beyond the length of ``goal_pos``.
    """
    sim = _FakeSimulator(
        robots=[
            _FakeRobot(pose=((0.0, 0.0), 0.0)),
            _FakeRobot(pose=((1.0, 0.0), 0.0)),
        ],
        goal_pos=[(5.0, 0.0)],  # one goal, query index 1
    )
    policy, wrapper = _build_policy(sim, robot_index=1)
    adapter, adapter_patcher = _patched_adapter()
    try:
        action = policy.action()
    finally:
        adapter_patcher.stop()

    np.testing.assert_array_equal(action, np.zeros(2, dtype=float))
    wrapper.get_forces_at.assert_not_called()
    adapter.assert_not_called()


def test_action_returns_zero_within_goal_tolerance_without_adapter():
    """A goal already inside ``goal_tolerance`` must return an exact zero action.

    The within-tolerance branch must neither query interaction forces nor hand
    a velocity off to the diff-drive adapter.
    """
    goal_tolerance = 0.25
    config = FastPysfPlannerConfig(goal_tolerance=goal_tolerance)
    sim = _FakeSimulator(
        robots=[_FakeRobot(pose=((0.0, 0.0), 0.0))],
        # distance 0.1 < tolerance 0.25 -> within tolerance
        goal_pos=[(0.1, 0.0)],
        config=_FakeSimConfig(time_per_step_in_secs=0.1),
    )
    policy, wrapper = _build_policy(sim, config=config)
    adapter, adapter_patcher = _patched_adapter()
    try:
        action = policy.action()
    finally:
        adapter_patcher.stop()

    np.testing.assert_array_equal(action, np.zeros(2, dtype=float))
    wrapper.get_forces_at.assert_not_called()
    adapter.assert_not_called()


# ---------------------------------------------------------------------------
# Warn-once for missing goal
# ---------------------------------------------------------------------------


def test_missing_goal_warning_emitted_exactly_once():
    """The missing-goal warning must fire once and be suppressed thereafter."""
    sim = _FakeSimulator(
        robots=[_FakeRobot(pose=((0.0, 0.0), 0.0))],
        goal_pos=[],
    )
    with patch(f"{PLANNER_MODULE}.logger") as mock_logger:
        policy, _ = _build_policy(sim)
        first = policy.action()
        second = policy.action()

    np.testing.assert_array_equal(first, np.zeros(2, dtype=float))
    np.testing.assert_array_equal(second, np.zeros(2, dtype=float))
    mock_logger.warning.assert_called_once_with(
        "FastPysfPlannerPolicy missing goal; returning zero action.",
    )
    assert policy._warned_missing_goal is True


# ---------------------------------------------------------------------------
# Desired motion + weighted interaction force + adapter bounds
# ---------------------------------------------------------------------------


def test_action_composes_desired_interaction_and_forwards_adapter_bounds():
    """The adapter must receive the composed holonomic velocity and robot bounds.

    With ``desired_speed`` capped, ``current_speed`` zero, heading zero, and a
    non-axis-aligned interaction force, the expected total force is

    ``desired_force + interaction_weight * interaction_force``

    and the holonomic velocity handed to the adapter is

    ``robot_vel + total_force * dt``.
    """
    # Distinctive adapter bounds so the forwarding contract is unambiguous.
    robot_config = _FakeRobotConfig(max_linear_speed=3.0, max_angular_speed=0.9)
    sim = _FakeSimulator(
        robots=[
            _FakeRobot(
                pose=((0.0, 0.0), 0.0),
                current_speed=(0.0, 0.0),
                config=robot_config,
            )
        ],
        goal_pos=[(5.0, 0.0)],
        config=_FakeSimConfig(time_per_step_in_secs=0.1),
    )
    config = FastPysfPlannerConfig(
        desired_speed=1.5,
        tau=0.5,
        interaction_weight=2.0,
        max_force=40.0,
        goal_tolerance=0.25,
    )
    adapter, adapter_patcher = _patched_adapter(
        return_value=np.array([7.0, 8.0], dtype=float),
    )
    try:
        policy, wrapper = _build_policy(sim, config=config)
        wrapper.get_forces_at.return_value = np.array([0.5, -0.5], dtype=float)

        action = policy.action()
    finally:
        adapter_patcher.stop()

    # desired_speed = min(1.5, 5.0 / 0.1) = 1.5 -> desired_vel = [1.5, 0]
    # desired_force = ([1.5, 0] - [0, 0]) / 0.5 = [3.0, 0]
    # total_force = [3.0, 0] + 2.0 * [0.5, -0.5] = [4.0, -1.0]
    # holonomic_vel = [0, 0] + [4.0, -1.0] * 0.1 = [0.4, -0.1]
    expected_adapter_velocity = np.array([0.4, -0.1], dtype=float)

    wrapper.get_forces_at.assert_called_once()
    force_args, force_kwargs = wrapper.get_forces_at.call_args
    np.testing.assert_allclose(force_args[0], np.array([0.0, 0.0], dtype=float), atol=1e-12)
    assert force_kwargs == {"include_desired": False, "include_robot": False}
    adapter.assert_called_once()
    args, kwargs = adapter.call_args
    np.testing.assert_allclose(args[0], expected_adapter_velocity, atol=1e-12)
    assert args[1] == ((0.0, 0.0), 0.0)
    assert kwargs["max_linear_speed"] == pytest.approx(3.0)
    assert kwargs["max_angular_speed"] == pytest.approx(0.9)
    assert kwargs["config"] is policy.adapter_config
    np.testing.assert_array_equal(action, np.array([7.0, 8.0], dtype=float))


# ---------------------------------------------------------------------------
# Max-force clipping
# ---------------------------------------------------------------------------


def test_action_clips_total_force_to_max_force():
    """When the total force exceeds ``max_force`` it must be clipped to it.

    With the interaction force zeroed and the desired force above ``max_force``,
    the unclipped holonomic velocity would be ``[0.3, 0]``; the clipped contract
    yields ``[0.1, 0]``.
    """
    sim = _FakeSimulator(
        robots=[_FakeRobot(pose=((0.0, 0.0), 0.0))],
        goal_pos=[(5.0, 0.0)],
        config=_FakeSimConfig(time_per_step_in_secs=0.1),
    )
    config = FastPysfPlannerConfig(
        desired_speed=1.5,
        tau=0.5,
        interaction_weight=1.0,
        max_force=1.0,  # desired_force norm of 3.0 must be clipped to 1.0
        goal_tolerance=0.25,
    )
    captured: dict[str, np.ndarray] = {}

    def capture_adapter(velocity, _pose, **_kwargs):
        captured["velocity"] = np.asarray(velocity, dtype=float)
        return np.array([0.0, 0.0], dtype=float)

    with patch(
        f"{PLANNER_MODULE}.holonomic_to_diff_drive_action",
        side_effect=capture_adapter,
    ):
        policy, wrapper = _build_policy(sim, config=config)
        wrapper.get_forces_at.return_value = np.zeros(2, dtype=float)

        policy.action()

    # total_force clipped from [3, 0] (norm 3) to [1, 0] (norm 1 == max_force);
    # holonomic_vel = [0, 0] + [1, 0] * 0.1 = [0.1, 0].
    np.testing.assert_allclose(captured["velocity"], np.array([0.1, 0.0]), atol=1e-12)


@pytest.mark.parametrize(
    ("force", "max_force", "expect_clipped"),
    [
        # Sub-cap force is returned unchanged.
        (np.array([1.0, 0.0]), 40.0, False),
        # Over-cap force is renormalized to the max-force magnitude.
        (np.array([30.0, 40.0]), 5.0, True),
        # Near-zero force must never be scaled (guard against div-by-zero).
        (np.array([0.0, 0.0]), 5.0, False),
    ],
)
def test_clip_force_direct_rule(force, max_force, expect_clipped):
    """``_clip_force`` must cap magnitude while preserving direction."""
    sim = _FakeSimulator(
        robots=[_FakeRobot(pose=((0.0, 0.0), 0.0))],
        goal_pos=[(5.0, 0.0)],
    )
    config = FastPysfPlannerConfig(max_force=max_force)
    policy, _ = _build_policy(sim, config=config)

    clipped = policy._clip_force(np.array(force, dtype=float))
    norm = float(np.linalg.norm(clipped))

    if expect_clipped:
        assert norm == pytest.approx(max_force)
        # Direction preserved.
        original_norm = float(np.linalg.norm(force))
        np.testing.assert_allclose(
            clipped,
            np.asarray(force, dtype=float) / original_norm * max_force,
            atol=1e-12,
        )
    else:
        np.testing.assert_allclose(clipped, np.asarray(force, dtype=float), atol=1e-12)


# ---------------------------------------------------------------------------
# Timestep use in desired speed and force integration
# ---------------------------------------------------------------------------


def test_action_uses_simulator_timestep_in_desired_speed_and_integration():
    """The timestep must feed both ``goal_dist / dt`` and ``total_force * dt``.

    With ``desired_speed`` uncapped (``goal_dist / dt`` binds) and a non-zero
    interaction force, the holonomic velocity changes with ``dt`` because the
    desired contribution cancels across ``dt`` while the interaction term does
    not. Two timesteps therefore yield two distinct adapter velocities.
    """
    config = FastPysfPlannerConfig(
        desired_speed=10.0,
        tau=0.5,
        interaction_weight=2.0,
        max_force=40.0,
        goal_tolerance=0.1,
    )

    captured: dict[float, np.ndarray] = {}

    def run(dt: float) -> np.ndarray:
        sim = _FakeSimulator(
            robots=[_FakeRobot(pose=((0.0, 0.0), 0.0))],
            goal_pos=[(2.0, 0.0)],  # goal_dist / dt binds below desired_speed
            config=_FakeSimConfig(time_per_step_in_secs=dt),
        )

        def capture(velocity, _pose, **_kwargs):
            captured[dt] = np.asarray(velocity, dtype=float)
            return np.zeros(2, dtype=float)

        with patch(
            f"{PLANNER_MODULE}.holonomic_to_diff_drive_action",
            side_effect=capture,
        ):
            policy, wrapper = _build_policy(sim, config=config)
            wrapper.get_forces_at.return_value = np.array([1.0, 0.0], dtype=float)
            policy.action()
        return captured[dt]

    vel_fast = run(0.5)  # dt = 0.5 -> desired_speed=4, holonomic_vel=[5.0, 0]
    vel_slow = run(0.25)  # dt = 0.25 -> desired_speed=8, holonomic_vel=[4.5, 0]

    np.testing.assert_allclose(vel_fast, np.array([5.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(vel_slow, np.array([4.5, 0.0]), atol=1e-12)
    # Distinct velocities prove the timestep is wired into both code paths.
    assert not np.allclose(vel_fast, vel_slow)


# ---------------------------------------------------------------------------
# predict signature
# ---------------------------------------------------------------------------


def test_predict_returns_action_and_none():
    """``predict`` must return a ``(action, None)`` tuple for Gym compatibility."""
    sim = _FakeSimulator(
        robots=[_FakeRobot(pose=((0.0, 0.0), 0.0))],
        goal_pos=[(5.0, 0.0)],
        config=_FakeSimConfig(time_per_step_in_secs=0.1),
    )
    sentinel = np.array([1.23, -0.45], dtype=float)
    _adapter, adapter_patcher = _patched_adapter(return_value=sentinel)
    try:
        policy, wrapper = _build_policy(sim)
        wrapper.get_forces_at.return_value = np.zeros(2, dtype=float)

        result = policy.predict(observation="ignored", foo="bar")
    finally:
        adapter_patcher.stop()

    assert isinstance(result, tuple)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], sentinel)
    assert result[1] is None
