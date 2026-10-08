"""Production controller holds must survive opt-in contact projection (#10178)."""

import numpy as np
import pytest
from pysocialforce import Simulator, contact

from robot_sf.nav.map_config import PedestrianWaitRule, SinglePedestrianDefinition
from robot_sf.ped_npc.ped_behavior import SinglePedestrianBehavior
from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates
from robot_sf.sim.simulator import Simulator as RobotSimulator
from tests.sim.test_pedestrian_contact import config, free_forces


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


def test_two_fixed_overlapping_bodies_record_unresolved_without_escape():
    """An infeasible held/prescribed pair keeps the existing failure counters."""
    sim, _ = held_pair(separation=0.4)
    sim.config.contact_prescribed_indices = (1,)
    sim.peds.state[1, 2:4] = 0.0
    sim.forces = free_forces(sim, sim.config)
    initial = sim.peds.pos().copy()
    sim.step()
    np.testing.assert_array_equal(sim.peds.pos(), initial)
    np.testing.assert_array_equal(sim.peds.vel(), np.zeros((2, 2)))
    assert sim.contact_projection_fallback_count == 1
    assert sim.contact_projection_unresolved_count == 1


def test_held_body_overlapping_wall_records_unresolved_without_escape():
    """Wall start push-out also cannot move a held body to fabricate feasibility."""
    sim, _ = held_pair(wall=True, separation=2.0)
    initial = sim.peds.pos()[0].copy()
    sim.step()
    np.testing.assert_array_equal(sim.peds.pos()[0], initial)
    np.testing.assert_array_equal(sim.peds.vel()[0], [0.0, 0.0])
    assert sim.contact_projection_fallback_count == 1
    assert sim.contact_projection_unresolved_count == 1


@pytest.mark.parametrize("hold", ["start_delay", "waypoint_wait", "proximity"])
def test_released_hold_can_move_and_reset_reengages_hold(hold):
    """Release removes the constraint; reset restores the configured dwell."""
    sim, behavior = held_pair(hold=hold)
    if hold == "start_delay":
        behavior._runtimes[0].start_delay_remaining_s = 0.1
    elif hold == "waypoint_wait":
        behavior._runtimes[0].wait_remaining_s = 0.1
    else:
        behavior.set_robot_pose_provider(lambda: [((10.0, 0.0), 0.0)])
    behavior.step()
    sim.step()
    assert sim.peds.pos()[0, 0] != 0.0
    assert sim.contact_projection_unresolved_count == 0
    behavior.reset()
    sim.peds.state[0, :2] = (0.0, 0.0)
    behavior.set_robot_pose_provider(None)
    behavior.step()
    sim.step()
    np.testing.assert_array_equal(sim.peds.pos()[0], [0.0, 0.0])
    np.testing.assert_array_equal(sim.peds.vel()[0], [0.0, 0.0])


def test_scenario_delayed_overlap_does_not_abort_other_pairs():
    """Full construction/step keeps delayed actors fixed and separates other pairs."""
    from robot_sf.gym_env.unified_config import RobotSimulationConfig
    from robot_sf.nav.map_config import MapDefinitionPool
    from robot_sf.sim.sim_config import SimulationSettings
    from robot_sf.sim.simulator import init_simulators
    from robot_sf.training.scenario_loader import _apply_single_pedestrian_overrides
    from tests.sim.test_pedestrian_desired_speed import _minimal_map

    np.random.seed(1001)
    map_def = _minimal_map()
    starts = [(3.0, 10.0), (3.4, 10.0), (10.0, 10.0), (10.7, 10.0)]
    goals = [(3.0, 18.0), (3.4, 18.0), (18.0, 10.0), (6.0, 10.0)]
    map_def.single_pedestrians = [
        SinglePedestrianDefinition(id=f"ped{i}", start=start, goal=goal, speed_m_s=1.0)
        for i, (start, goal) in enumerate(zip(starts, goals, strict=True))
    ]
    cfg = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=SimulationSettings(
            difficulty=0,
            ped_density_by_difficulty=[0.0],
            population_size=4,
            pedestrian_radius_m=0.28,
            pedestrian_contact_rule="projection_v1",
            route_spawn_seed=1001,
            archetype_seed=1001,
            response_law_seed=1001,
            desired_speed_seed=1001,
        ),
    )
    _apply_single_pedestrian_overrides(
        cfg, [{"id": "ped0", "start_delay_s": 10.0}, {"id": "ped1", "start_delay_s": 10.0}]
    )
    map_def = cfg.map_pool.map_defs["test"]
    sim = init_simulators(cfg, map_def, num_robots=1, random_start_pos=False)[0]
    backend = sim.pysf_sim
    # Isolate physical contact through the existing force factory interface.
    backend.forces = free_forces(backend, backend.config)
    for step in range(10):
        sim.step_once([(0.0, 0.0)])
        np.testing.assert_array_equal(backend.peds.pos()[:2], starts[:2])
        np.testing.assert_array_equal(backend.peds.vel()[:2], np.zeros((2, 2)))
        assert np.linalg.norm(backend.peds.pos()[2] - backend.peds.pos()[3]) >= 0.56
        assert backend.contact_projection_fallback_count == step + 1
        assert backend.contact_projection_unresolved_count == step + 1


@pytest.mark.parametrize("overlapping_holds", [False, True])
def test_unrelated_holds_allow_local_rollback(monkeypatch, overlapping_holds):
    """A distant hold, even an infeasible held pair, cannot disable jam rollback."""
    np.random.seed(1001)
    cfg = config()
    cfg.scene_config.agent_radius = 0.30
    previous = np.array(
        [
            [0, 0, 1.2, 0, 10, 0, 0.5],
            [0.61, 0, 0.4, 0, 10, 0, 0.5],
            [-0.61, 0, 1.1, 0, 10, 0, 0.5],
            [50, 50, 1.3, 0, 100, 50, 0.5],
            [100, 100, 0, 0, 100, 110, 0.5],
        ]
        + ([[100.4, 100, 0, 0, 100.4, 110, 0.5]] if overlapping_holds else [])
    )
    attempted = previous[:, :2].copy()
    attempted[:4] = [[0.12, 0], [0.65, 0], [-0.50, 0], [50.13, 50]]
    sim = Simulator(previous.copy(), config=cfg, make_forces=free_forces)
    definitions = [
        SinglePedestrianDefinition(
            id=f"held{i}", start=tuple(row[:2]), goal=tuple(row[4:6]), start_delay_s=10.0
        )
        for i, row in enumerate(previous[4:])
    ]
    states = PedestrianStates(lambda: sim.peds.state)
    behavior = SinglePedestrianBehavior(
        states, PedestrianGroupings(states), definitions, single_offset=4, time_step_s=0.1
    )
    behavior.bind_pysf_peds(sim.peds)
    behavior.step()
    sim.peds.max_speeds[:4] = 2
    sim.peds.state[:, :2] = attempted
    # Existing round3 cap-exhaustion seam, with real hold binding added.
    monkeypatch.setattr(contact, "project_step", lambda *args: (attempted.copy(), 1, False))
    contact.apply_contact_step(sim, previous)
    distances = np.linalg.norm(sim.peds.pos()[:3, None] - sim.peds.pos()[None, :3], axis=2)
    assert distances[np.triu_indices(3, 1)].min() >= 0.60
    np.testing.assert_array_equal(sim.peds.pos()[3], attempted[3])
    np.testing.assert_array_equal(sim.peds.vel()[3], [1.3, 0])
    np.testing.assert_array_equal(sim.peds.pos()[4:], previous[4:, :2])
    np.testing.assert_array_equal(sim.peds.vel()[4:], np.zeros((len(definitions), 2)))
    assert sim.contact_projection_fallback_count == 1
    assert sim.contact_projection_unresolved_count == int(overlapping_holds)


@pytest.mark.parametrize("integration_path", ["native", "robot_benchmark"])
def test_prescribed_swap_reports_swept_contact_with_separated_endpoints(integration_path):
    """Two immovable .28m bodies cross at zero separation within a .2s step."""
    np.random.seed(1001)
    cfg = config(dt=0.2)
    cfg.contact_prescribed_indices = (0, 1)
    state = np.array([[-0.29, 0, 3, 0, 10, 0, 0.5], [0.29, 0, -3, 0, -10, 0, 0.5]])
    sim = Simulator(state.copy(), config=cfg, make_forces=free_forces)
    step_backend(sim, integration_path)
    np.testing.assert_allclose(sim.peds.pos(), [[0.31, 0], [-0.31, 0]], atol=1e-12)
    assert np.linalg.norm(sim.peds.pos()[0] - sim.peds.pos()[1]) > 0.56
    assert sim.contact_projection_fallback_count == 1
    assert sim.contact_projection_unresolved_count == 1


@pytest.mark.parametrize("integration_path", ["native", "robot_benchmark"])
def test_prescribed_crossing_of_production_hold_reports_swept_contact(integration_path):
    """The real start-delay hold stays stationary but cannot certify a crossing."""
    sim, _ = held_pair(separation=0.6)
    sim.config.contact_prescribed_indices = (1,)
    sim.peds.d_t = 0.2
    sim.peds.state[1, 2:4] = [-6, 0]
    sim.forces = free_forces(sim, sim.config)
    step_backend(sim, integration_path)
    np.testing.assert_array_equal(sim.peds.pos()[0], [0, 0])
    np.testing.assert_array_equal(sim.peds.vel()[0], [0, 0])
    np.testing.assert_allclose(sim.peds.pos()[1], [-0.6, 0], atol=1e-12)
    assert sim.contact_projection_fallback_count == 1
    assert sim.contact_projection_unresolved_count == 1
