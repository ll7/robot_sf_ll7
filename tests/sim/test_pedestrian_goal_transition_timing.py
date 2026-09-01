"""Timing canary for the real pedestrian step path (issue #8063)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.nav.map_config import SinglePedestrianDefinition
from robot_sf.ped_npc.ped_behavior import SinglePedestrianBehavior
from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates
from robot_sf.prediction.oracle_transition_trace import (
    SIMULATOR_TIMING_ORDER,
    ControllerMutationFlags,
    DynamicsParameters,
    ForceComponents,
    ForceTimeRobotState,
    GoalChangeKind,
    OracleTransitionTraceV1,
    SpeedCap,
    SpeedCapStatus,
    TransitionBoundary,
    TransitionBoundaryKind,
)
from robot_sf.sim.simulator import Simulator


class _FakePeds:
    """Small PySocialForce seam that captures the goal used at integration."""

    def __init__(self, state: np.ndarray, events: list[str]) -> None:
        self._state = state
        self._events = events
        self.goal_at_integration: tuple[float, float] | None = None

    def step(self, _forces: np.ndarray, _groups: list[list[int]]) -> None:
        """Record the goal at integration and advance position one synthetic step."""
        self._events.append("integrate")
        self.goal_at_integration = (float(self._state[0, 4]), float(self._state[0, 5]))
        self._state[0, 0] = 0.1


@dataclass
class _CapturedTransition:
    """Snapshots captured from the real simulator seam during one transition."""

    pre_behavior: TransitionBoundary | None = None
    post_behavior_pre_force: TransitionBoundary | None = None
    post_integration: TransitionBoundary | None = None
    registry_force_xy: tuple[float, float] | None = None
    final_force_xy: tuple[float, float] | None = None


def _capture_boundary(
    state: np.ndarray,
    behavior: SinglePedestrianBehavior,
    *,
    boundary: TransitionBoundaryKind,
    timestamp_s: float,
    step_index: int,
    force_time_robot_state: ForceTimeRobotState | None = None,
    mutation_flags: ControllerMutationFlags | None = None,
) -> TransitionBoundary:
    """Build a boundary solely from the current simulator state and behavior runtime."""
    position = (float(state[0, 0]), float(state[0, 1]))
    velocity = (float(state[0, 2]), float(state[0, 3]))
    active_goal = (float(state[0, 4]), float(state[0, 5]))
    threshold = behavior.goal_proximity_threshold
    assert threshold is not None
    threshold_reached = bool(np.linalg.norm(np.subtract(position, active_goal)) <= threshold)
    return TransitionBoundary(
        boundary=boundary,
        timestamp_s=timestamp_s,
        step_index=step_index,
        position_xy=position,
        velocity_xy=velocity,
        active_goal_xy=active_goal,
        route_waypoint_index=behavior._runtimes[0].waypoint_index,
        goal_threshold_reached=threshold_reached,
        force_time_robot_state=force_time_robot_state,
        mutation_flags=mutation_flags,
    )


def test_real_simulator_step_uses_post_behavior_goal_for_force_transition() -> None:  # noqa: PLR0915
    """The real step path captures a waypoint advance and binds force to its new goal."""
    events = ["state_t"]
    state = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    states = PedestrianStates(lambda: state)
    groups = PedestrianGroupings(states)
    behavior = SinglePedestrianBehavior(
        states=states,
        groups=groups,
        single_pedestrians=[
            SinglePedestrianDefinition(
                id="ped-1",
                start=(0.0, 0.0),
                trajectory=[(0.0, 0.0), (2.0, 0.0)],
            )
        ],
        single_offset=0,
        time_step_s=0.1,
        goal_proximity_threshold=0.2,
    )
    capture = _CapturedTransition()
    capture.pre_behavior = _capture_boundary(
        state,
        behavior,
        boundary=TransitionBoundaryKind.PRE_BEHAVIOR,
        timestamp_s=0.0,
        step_index=0,
    )
    original_behavior_step = behavior.step

    def record_behavior_step() -> None:
        """Record the state before and after the real behavior controller runs."""
        events.append("behavior.step")
        original_behavior_step()
        assert capture.pre_behavior is not None
        capture.post_behavior_pre_force = _capture_boundary(
            state,
            behavior,
            boundary=TransitionBoundaryKind.POST_BEHAVIOR_PRE_FORCE,
            timestamp_s=capture.pre_behavior.timestamp_s,
            step_index=capture.pre_behavior.step_index,
            force_time_robot_state=ForceTimeRobotState(),
            mutation_flags=ControllerMutationFlags(
                goal_redirected=(
                    capture.pre_behavior.active_goal_xy != (float(state[0, 4]), float(state[0, 5]))
                )
            ),
        )
        events.append("goal_after_behavior_t")

    behavior.step = record_behavior_step
    fake_peds = _FakePeds(state, events)

    def compute_forces() -> np.ndarray:
        """Record the goal observed by the force stage."""
        events.append("compute_force_t")
        assert capture.post_behavior_pre_force is not None
        assert tuple(float(value) for value in state[0, 4:6]) == (
            capture.post_behavior_pre_force.active_goal_xy
        )
        forces = np.zeros((1, 2), dtype=float)
        capture.registry_force_xy = (float(forces[0, 0]), float(forces[0, 1]))
        return forces

    fake = SimpleNamespace(
        config=SimpleNamespace(residual_adversary=SimpleNamespace(is_active=False)),
        peds_behaviors=[behavior],
        pysf_state=states,
        pysf_sim=SimpleNamespace(peds=fake_peds, compute_forces=compute_forces),
        groups=groups,
        robots=[],
        robot_navs=[],
        pedestrian_model="pysocialforce",
        last_ped_forces=np.zeros((0, 2), dtype=float),
    )
    for method_name in (
        "_validate_robot_action_count",
        "_apply_residual_adversary",
        "_step_pedestrians",
        "_headings_from_current_ped_velocities",
    ):
        setattr(fake, method_name, getattr(Simulator, method_name).__get__(fake))
    original_apply = fake._apply_residual_adversary

    def record_force_variant(forces: np.ndarray) -> np.ndarray:
        """Record the model/residual boundary while preserving simulator behavior."""
        events.append("apply_model_variant_or_residual")
        final_forces = original_apply(forces)
        capture.final_force_xy = (float(final_forces[0, 0]), float(final_forces[0, 1]))
        return final_forces

    fake._apply_residual_adversary = record_force_variant

    original_step_pedestrians = fake._step_pedestrians

    def record_integration(forces: np.ndarray, groups: list[list[int]]) -> None:
        """Capture the post-integration boundary after the real pedestrian step method returns."""
        original_step_pedestrians(forces, groups)
        assert capture.pre_behavior is not None
        capture.post_integration = _capture_boundary(
            state,
            behavior,
            boundary=TransitionBoundaryKind.POST_INTEGRATION,
            timestamp_s=capture.pre_behavior.timestamp_s + behavior.time_step_s,
            step_index=capture.pre_behavior.step_index + 1,
        )

    fake._step_pedestrians = record_integration

    Simulator.step_once(fake, [])
    events.append("state_t+1")

    assert events == list(SIMULATOR_TIMING_ORDER)
    assert capture.post_behavior_pre_force is not None
    assert capture.post_integration is not None
    assert capture.registry_force_xy is not None
    assert capture.final_force_xy is not None
    assert fake_peds.goal_at_integration == capture.post_behavior_pre_force.active_goal_xy
    assert capture.post_integration.active_goal_xy == capture.post_behavior_pre_force.active_goal_xy

    trace = OracleTransitionTraceV1(
        episode_id="episode-1",
        transition_id="episode-1:t0",
        transition_step_index=0,
        simulator_pedestrian_id="pysf-0",
        actor_track_id="track-1",
        actor_tracking_epoch_id="epoch-1",
        backend="pysocialforce",
        pre_behavior=capture.pre_behavior,
        post_behavior_pre_force=capture.post_behavior_pre_force,
        post_integration=capture.post_integration,
        force_components=ForceComponents(
            registry_total_force_xy=capture.registry_force_xy,
            final_pre_cap_force_xy=capture.final_force_xy,
        ),
        dynamics=DynamicsParameters(goal_threshold_m=behavior.goal_proximity_threshold),
        speed_cap=SpeedCap(SpeedCapStatus.UNKNOWN),
        goal_change_kind=GoalChangeKind.WAYPOINT_ADVANCE,
    )
    payload = trace.to_dict()
    assert payload["pre_behavior"] == capture.pre_behavior.to_dict()
    assert payload["post_behavior_pre_force"] == capture.post_behavior_pre_force.to_dict()
    assert payload["post_integration"] == capture.post_integration.to_dict()
    assert payload["force_components"]["registry_total_force_xy"] == list(capture.registry_force_xy)
    assert payload["force_components"]["final_pre_cap_force_xy"] == list(capture.final_force_xy)
    assert trace.goal_change_kind is GoalChangeKind.WAYPOINT_ADVANCE

    shifted_goal = trace.to_dict()
    shifted_goal["post_behavior_pre_force"]["active_goal_xy"] = shifted_goal["pre_behavior"][
        "active_goal_xy"
    ]
    with pytest.raises(ValueError, match="waypoint_advance must change the active goal"):
        OracleTransitionTraceV1.from_dict(shifted_goal)

    shifted_route = trace.to_dict()
    shifted_route["post_behavior_pre_force"]["route_waypoint_index"] = shifted_route[
        "pre_behavior"
    ]["route_waypoint_index"]
    with pytest.raises(
        ValueError, match="waypoint_advance route_waypoint_index must advance by one"
    ):
        OracleTransitionTraceV1.from_dict(shifted_route)

    shifted_force = trace.to_dict()
    shifted_force["force_components"]["final_pre_cap_force_xy"] = [1.0, 0.0]
    with pytest.raises(
        ValueError, match="final_pre_cap_force_xy does not match the recorded force stages"
    ):
        OracleTransitionTraceV1.from_dict(shifted_force)
