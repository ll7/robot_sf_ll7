"""Production controller holds must survive opt-in contact projection (#10178)."""

import numpy as np
import pytest
from pysocialforce import Simulator
from pysocialforce.config import SimulatorConfig

from robot_sf.nav.map_config import PedestrianWaitRule, SinglePedestrianDefinition
from robot_sf.ped_npc.ped_behavior import SinglePedestrianBehavior
from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates
from robot_sf.sim.simulator import Simulator as RobotSimulator


def config(*, contact=True, wall=False):
    """Construct the explicitly selected physics used by this hold witness.

    Returns:
        Native simulator settings with a shared physical radius.
    """
    cfg = SimulatorConfig()
    cfg.scene_config.agent_radius = 0.28
    if contact:
        cfg.pedestrian_contact_rule = "projection_v2"
    return cfg


def held_pair(*, contact=True, wall=False, hold="start_delay", separation=0.57):
    """Bind the real controller to the dev1001 two-body witness backend.

    Returns:
        Backend and controller with one held body and an approaching neighbor.
    """
    np.random.seed(1001)
    state = np.array([[0, 0, 1, 0, 10, 0, 0.5], [separation, 0, -1, 0, -10, 0, 0.5]])
    sim = Simulator(
        state,
        obstacles=[(-2, 2, -0.1, -0.1)] if wall else None,
        config=config(contact=contact, wall=wall),
    )
    kwargs = {
        "start_delay": {"start_delay_s": 10.0},
        "role_wait": {"role": "wait"},
        "waypoint_wait": {"wait_at": [PedestrianWaitRule(waypoint_index=0, wait_s=10.0)]},
        "proximity": {"hold_until_robot_within_m": 1.0, "hold_ref_point": (10.0, 0.0)},
    }[hold]
    ped = SinglePedestrianDefinition(
        id="held", start=(0.0, 0.0), trajectory=[(0.0, 0.0), (10.0, 0.0)], **kwargs
    )
    states = PedestrianStates(lambda: sim.peds.state)
    behavior = SinglePedestrianBehavior(
        states, PedestrianGroupings(states), [ped], single_offset=0, time_step_s=0.1
    )
    behavior.bind_pysf_peds(sim.peds)
    behavior.step()
    return sim, behavior


def step_backend(sim, integration_path):
    """Exercise both production paths that call contact projection."""
    if integration_path == "native":
        sim.step()
    else:
        wrapper = RobotSimulator.__new__(RobotSimulator)
        wrapper.pysf_sim = sim
        wrapper.pysf_state = PedestrianStates(lambda: sim.peds.state)
        wrapper.pedestrian_model = "social_force"
        wrapper._step_pedestrians(sim.compute_forces(), [])


@pytest.mark.parametrize("integration_path", ["native", "robot_benchmark"])
def test_start_delay_hold_survives_contact_projection(integration_path):
    """The 9.9s/zero-cap held body stays put while its neighbor absorbs correction."""
    off, _ = held_pair(contact=False)
    sim, behavior = held_pair()
    initial = sim.peds.pos().copy()
    assert behavior._runtimes[0].start_delay_remaining_s == pytest.approx(9.9)
    assert sim.peds.max_speeds[0] == 0.0
    step_backend(off, integration_path)
    step_backend(sim, integration_path)
    np.testing.assert_array_equal(off.peds.pos()[0], initial[0])
    np.testing.assert_array_equal(sim.peds.vel()[0], [0.0, 0.0])
    np.testing.assert_array_equal(sim.peds.pos()[0], initial[0])
    assert np.linalg.norm(sim.peds.pos()[1] - sim.peds.pos()[0]) >= 0.56
    assert sim.peds.pos()[1, 0] > off.peds.pos()[1, 0]
    assert sim.contact_projection_unresolved_count == 0


@pytest.mark.parametrize("hold", ["role_wait", "waypoint_wait", "proximity"])
def test_other_active_holds_survive_contact_projection(hold):
    """Zero velocity alone cannot let contact push a controller-held body."""
    sim, _ = held_pair(hold=hold)
    initial = sim.peds.pos()[0].copy()
    sim.step()
    np.testing.assert_array_equal(sim.peds.pos()[0], initial)
    np.testing.assert_array_equal(sim.peds.vel()[0], [0.0, 0.0])
    assert np.linalg.norm(sim.peds.pos()[1] - sim.peds.pos()[0]) >= 0.56
