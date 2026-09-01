"""Timing canary for the real pedestrian step path (issue #8063)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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
    """Small PySocialForce seam that observes the goal used at integration."""

    def __init__(self, state: np.ndarray, events: list[str]) -> None:
        self._state = state
        self._events = events

    def step(self, _forces: np.ndarray, _groups: list[list[int]]) -> None:
        """Record the goal at integration and advance position one synthetic step."""
        self._events.append("integrate")
        assert tuple(self._state[0, 4:6]) == (2.0, 0.0)
        self._state[0, 0] = 0.1


def test_real_simulator_step_uses_post_behavior_goal_for_force_transition() -> None:
    """The real ``Simulator.step_once`` order labels a waypoint advance at transition t."""
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
    original_behavior_step = behavior.step

    def record_behavior_step() -> None:
        """Record the state before and after the real behavior controller runs."""
        events.append("behavior.step")
        original_behavior_step()
        events.append("goal_after_behavior_t")

    behavior.step = record_behavior_step
    fake_peds = _FakePeds(state, events)

    def compute_forces() -> np.ndarray:
        """Record the goal observed by the force stage."""
        events.append("compute_force_t")
        assert tuple(state[0, 4:6]) == (2.0, 0.0)
        return np.zeros((1, 2), dtype=float)

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
        return original_apply(forces)

    fake._apply_residual_adversary = record_force_variant

    Simulator.step_once(fake, [])
    events.append("state_t+1")

    assert events == list(SIMULATOR_TIMING_ORDER)
    assert tuple(state[0, 4:6]) == (2.0, 0.0)

    trace = OracleTransitionTraceV1(
        episode_id="episode-1",
        transition_id="episode-1:t0",
        transition_step_index=0,
        simulator_pedestrian_id="pysf-0",
        actor_track_id="track-1",
        backend="pysocialforce",
        pre_behavior=TransitionBoundary(
            TransitionBoundaryKind.PRE_BEHAVIOR,
            0.0,
            0,
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 0.0),
            route_waypoint_index=0,
            goal_threshold_reached=True,
        ),
        post_behavior_pre_force=TransitionBoundary(
            TransitionBoundaryKind.POST_BEHAVIOR_PRE_FORCE,
            0.0,
            0,
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            route_waypoint_index=1,
            goal_threshold_reached=False,
            force_time_robot_state=ForceTimeRobotState(),
            mutation_flags=ControllerMutationFlags(goal_redirected=True),
        ),
        post_integration=TransitionBoundary(
            TransitionBoundaryKind.POST_INTEGRATION,
            0.1,
            1,
            (0.1, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            route_waypoint_index=1,
            goal_threshold_reached=False,
        ),
        force_components=ForceComponents(
            registry_total_force_xy=(0.0, 0.0),
            final_pre_cap_force_xy=(0.0, 0.0),
            uncapped_velocity_xy=(1.0, 0.0),
            applied_velocity_xy=(1.0, 0.0),
        ),
        dynamics=DynamicsParameters(goal_threshold_m=0.2),
        speed_cap=SpeedCap(SpeedCapStatus.UNKNOWN),
        goal_change_kind=GoalChangeKind.WAYPOINT_ADVANCE,
    )
    assert trace.post_behavior_pre_force.active_goal_xy == (2.0, 0.0)
    assert trace.goal_change_kind is GoalChangeKind.WAYPOINT_ADVANCE
