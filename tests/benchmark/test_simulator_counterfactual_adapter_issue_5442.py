"""Tests for the production-simulator CounterfactualModel adapter (issue #5442).

This is the remaining slice named in the issue thread: a real
``CounterfactualModel`` adapter over the live Robot SF ``Simulator`` plus the
RNG-capture seam. The tests build a headless ``Simulator`` (no display), prove the
snapshot/restore seam reproduces a baseline pedestrian-robot episode bit-for-bit,
and drive the frozen-state replay engine end to end on a genuine production fixture
where the native forward-acceleration baseline contacts but native braking
avoids contact.

The fail-closed ``unknown`` path for a nondeterministic baseline is already covered
by the controlled-fixture tests (``nondeterministic_baseline_scenario``); here we
additionally show that omitting the global-RNG capture seam makes a real baseline
replay diverge, exercising the same guard on production state.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.last_avoidable_fixtures import (
    KinematicCollisionModel,
    KinematicScenario,
)
from robot_sf.benchmark.last_avoidable_replay import (
    SUBSTITUTION_HOLD,
    VERDICT_AVOIDABLE,
    VERDICT_UNKNOWN,
    ReplayConfig,
    locate_last_avoidable,
)
from robot_sf.benchmark.simulator_counterfactual_adapter import (
    SimulatorCounterfactualModel,
    _capture_route_navigators,
    _restore_route_navigators,
)
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.navigation import RouteNavigator
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import init_simulators

# A genuine production fixture: the robot drives into a doorway crossing; the
# forward-acceleration baseline contacts after 39 applied ticks, while native
# braking clears it. The explicit global seed controls spawn-zone sampling.
_FIXTURE_MAP = Path(__file__).resolve().parents[2] / "maps/svg_maps/classic_doorway.svg"
_FIXTURE_DENSITY = [0.06]
_FIXTURE_SEED = 21
_FIXTURE_GLOBAL_SEED = 25
_DENSE_GLOBAL_SEED = 26
_FIXTURE_SPEED = 1.0
_FIXTURE_CONTACT_STEP = 39
_COLLISION_RADIUS = 0.5


@pytest.fixture(autouse=True)
def _preserve_numpy_rng():
    """Keep global RNG changes in these seam tests from leaking to other tests."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _build_simulator() -> object:
    """Construct a deterministic, headless ``Simulator`` for the fixture map."""
    np.random.seed(_FIXTURE_GLOBAL_SEED)
    map_def = convert_map(str(_FIXTURE_MAP))
    sim_config = SimulationSettings(
        difficulty=0,
        ped_density_by_difficulty=_FIXTURE_DENSITY,
        route_spawn_seed=_FIXTURE_SEED,
    )
    cfg = RobotSimulationConfig(sim_config=sim_config)
    return init_simulators(cfg, map_def, num_robots=1, random_start_pos=False)[0]


def _build_dense_simulator() -> object:
    """Construct a denser doorway simulator where pedestrians respawn mid-episode.

    The respawns draw from the global numpy RNG (via ``sample_zone``), so a replay
    without the RNG-capture seam diverges — exercising the same guard the engine's
    ``unknown`` determination relies on.
    """
    np.random.seed(_DENSE_GLOBAL_SEED)
    map_def = convert_map(str(_FIXTURE_MAP))
    sim_config = SimulationSettings(
        difficulty=0,
        ped_density_by_difficulty=[0.15],
        route_spawn_seed=21,
    )
    cfg = RobotSimulationConfig(sim_config=sim_config)
    return init_simulators(cfg, map_def, num_robots=1, random_start_pos=False)[0]


def _run_engine(model: SimulatorCounterfactualModel, *, determinism_replays: int = 5):
    """Drive ``locate_last_avoidable`` over the fixture episode."""
    contact_step = _FIXTURE_CONTACT_STEP
    horizon = contact_step + 10
    config = ReplayConfig(
        t_danger=0,
        t_contact=contact_step,
        horizon=horizon,
        substitution_mode=SUBSTITUTION_HOLD,
        determinism_replays=determinism_replays,
        action_set_id="simulator_native_action_lattice",
        feasibility_filter="native_maintain_or_brake",
        collision_predicate="robot_pedestrian_center_distance_v1",
        pedestrian_response="closed_loop",
    )
    baseline = [(float(_FIXTURE_SPEED), 0.0)] * (contact_step + horizon + 2)
    return locate_last_avoidable(model, baseline, config)


# -- snapshot/restore seam ------------------------------------------------
def test_snapshot_restore_reproduces_baseline_deterministically() -> None:
    """Restoring a snapshot and replaying yields the same contact step every time."""
    sim = _build_simulator()
    model = SimulatorCounterfactualModel(sim, collision_radius=_COLLISION_RADIUS)
    snap0 = model.snapshot()
    contacts: list[int | None] = []
    for _ in range(4):
        model.restore(snap0)
        c = None
        for i in range(_FIXTURE_CONTACT_STEP + 5):
            if model.collision():
                c = i
                break
            model.step((float(_FIXTURE_SPEED), 0.0))
        else:
            if model.collision():
                c = _FIXTURE_CONTACT_STEP + 5
        contacts.append(c)
    assert contacts == [_FIXTURE_CONTACT_STEP] * 4


def test_snapshot_restores_groups_behavior_rng_and_residual_state() -> None:
    """Every mutable branch controller state returns to the captured snapshot."""
    ped_state = np.zeros((1, 6), dtype=float)
    groups = SimpleNamespace(
        groups={0: {0}},
        group_by_ped_id={0: 0},
        _groups_as_lists_cache=[[0]],
    )
    behavior_rng = np.random.default_rng(7)
    behavior = SimpleNamespace(rng=behavior_rng)
    residual = SimpleNamespace(counter=1)
    robot = SimpleNamespace(
        pose=((0.0, 0.0), 0.0),
        state=SimpleNamespace(),
        config=SimpleNamespace(radius=0.5),
        pos=(0.0, 0.0),
    )
    sim = SimpleNamespace(
        robots=[robot],
        robot_navs=[],
        peds_behaviors=[behavior],
        groups=groups,
        pysf_sim=SimpleNamespace(peds=SimpleNamespace(groups=[[0]])),
        pysf_state=SimpleNamespace(pysf_states=lambda: ped_state),
        ped_headings=np.zeros(1),
        ped_angular_velocities=np.zeros(1),
        peds_have_obstacle_forces=False,
        _residual_adversary=residual,
    )
    model = SimulatorCounterfactualModel(sim, capture_rng=False)
    snapshot = model.snapshot()

    expected_rng_draw = behavior_rng.random()
    groups.groups[0].clear()
    groups.groups[1] = {0}
    groups.group_by_ped_id[0] = 1
    behavior_rng.random()
    residual.counter = 9

    model.restore(snapshot)

    assert groups.groups == {0: {0}}
    assert groups.group_by_ped_id == {0: 0}
    assert groups._groups_as_lists_cache is None
    assert sim.pysf_sim.peds.groups == [[0]]
    assert behavior_rng.random() == expected_rng_draw
    assert sim._residual_adversary.counter == 1


def test_restore_resynchronizes_backend_groups_and_branch_outcomes() -> None:
    """Restoring mutated membership reproduces forces and outcomes on every branch."""
    sim = _build_simulator()
    model = SimulatorCounterfactualModel(sim, collision_radius=_COLLISION_RADIUS)
    snapshot = model.snapshot()
    expected_groups = [list(group) for group in sim.groups.groups_as_lists]

    def run_branch() -> tuple[np.ndarray, np.ndarray, bool]:
        model.restore(snapshot)
        forces = np.asarray(sim.pysf_sim.compute_forces()).copy()
        model.step((float(_FIXTURE_SPEED), 0.0))
        return forces, sim.pysf_state.pysf_states().copy(), model.collision()

    expected_forces, expected_state, expected_collision = run_branch()

    # Mirror the state transition that normally updates the backend after a
    # behavior step, leaving the public and backend groupings on a branch.
    sim.groups.add_to_group(0, 1)
    sim.pysf_sim.peds.groups = sim.groups.groups_as_lists
    assert sim.groups.groups_as_lists[:2] == [[1], [0, 2]]
    assert sim.pysf_sim.peds.groups[:2] == [[1], [0, 2]]

    first_forces, first_state, first_collision = run_branch()
    second_forces, second_state, second_collision = run_branch()

    assert sim.groups.groups_as_lists == expected_groups
    assert sim.pysf_sim.peds.groups == expected_groups
    np.testing.assert_array_equal(first_forces, expected_forces)
    np.testing.assert_array_equal(second_forces, expected_forces)
    np.testing.assert_array_equal(first_state, expected_state)
    np.testing.assert_array_equal(second_state, expected_state)
    assert (first_collision, second_collision) == (expected_collision, expected_collision)


def test_rng_capture_seam_prevents_divergence() -> None:
    """Without the global-RNG capture seam, a mid-episode respawn replay diverges.

    A dense doorway scenario triggers pedestrian group respawns that draw from the
    global numpy RNG (via ``sample_zone``). With the seam ``capture_rng=True`` the
    snapshot restores that RNG so the replay is bit-for-bit identical; with
    ``capture_rng=False`` the same replay diverges by meters — exactly the
    nondeterministic-baseline condition the engine's ``unknown`` guard protects
    against.
    """
    # With the seam: replay is bit-for-bit reproducible.
    sim = _build_dense_simulator()
    model = SimulatorCounterfactualModel(sim, collision_radius=_COLLISION_RADIUS, capture_rng=True)
    snap0 = model.snapshot()
    initial_positions = np.asarray(sim.ped_pos).copy()
    np.random.seed(999)
    model.restore(snap0)
    for _ in range(100):
        model.step((0.0, 0.0))
    captured_positions = np.asarray(sim.ped_pos).copy()

    # Without the seam: a perturbed global RNG makes the replay diverge.
    sim2 = _build_dense_simulator()
    model2 = SimulatorCounterfactualModel(
        sim2, collision_radius=_COLLISION_RADIUS, capture_rng=False
    )
    snap0b = model2.snapshot()
    assert np.array_equal(initial_positions, np.asarray(sim2.ped_pos))
    model2.restore(snap0b)
    np.random.seed(999)
    for _ in range(100):
        model2.step((0.0, 0.0))
    uncaptured_positions = np.asarray(sim2.ped_pos).copy()

    assert not np.allclose(captured_positions, uncaptured_positions), (
        "the RNG-capture seam must matter for this fixture; mid-episode respawn "
        "should diverge without it"
    )


# -- engine integration on a real production fixture ----------------------
def test_production_fixture_is_avoidable() -> None:
    """The live-simulator adapter reports an avoidable production collision deterministically."""
    sim = _build_simulator()
    model = SimulatorCounterfactualModel(sim, collision_radius=_COLLISION_RADIUS)
    report = _run_engine(model)
    assert report.verdict == VERDICT_AVOIDABLE
    assert report.determinism.deterministic is True
    assert report.determinism.observed_contact_steps == (_FIXTURE_CONTACT_STEP - 1,) * 5
    assert report.t_uca is not None and report.t_inevitable is not None
    assert report.t_uca <= report.t_inevitable <= report.config.t_contact
    assert report.feasible_coverage == pytest.approx(1.0)
    assert report.minimal_sufficient_interventions, "must record preventing interventions"
    # Every decision point in the danger window is preserved.
    assert len(report.branches) == report.config.t_contact - report.config.t_danger


def test_adapter_satisfies_counterfactual_model_protocol() -> None:
    """The adapter exposes the snapshot/restore/step/collision/feasible/label seam."""
    sim = _build_simulator()
    model = SimulatorCounterfactualModel(sim, collision_radius=_COLLISION_RADIUS)
    snap = model.snapshot()
    assert snap is not None
    model.restore(snap)
    model.step((float(_FIXTURE_SPEED), 0.0))
    assert isinstance(model.collision(), bool)
    actions = model.feasible_actions()
    assert len(actions) >= 2
    assert model.action_label(actions[0]).startswith("robot_accel=")
    assert any(action[0] < 0 for action in actions)
    assert any(action[1] < 0 for action in actions)
    assert any(action[1] > 0 for action in actions)
    assert "linear_mps2=" in model.action_label(actions[0])


def test_response_and_collision_metadata_are_bound_to_native_contract() -> None:
    """Native metadata exposes the actual closed-loop response and predicate."""
    sim = _build_simulator()
    model = SimulatorCounterfactualModel(sim, collision_radius=_COLLISION_RADIUS)
    assert model.pedestrian_response == "closed_loop"
    assert model.collision_predicate == "robot_pedestrian_center_distance_v1"


def test_robot_relative_behavior_is_closed_loop_provenance() -> None:
    """Robot-relative scripted pedestrians read the substituted robot pose."""
    robot = SimpleNamespace(pos=(0.0, 0.0), config=SimpleNamespace(radius=0.5))
    sim = SimpleNamespace(
        robots=[robot],
        config=SimpleNamespace(
            prf_config=SimpleNamespace(is_active=False),
            apf_config=SimpleNamespace(is_active=False),
            residual_adversary=SimpleNamespace(is_active=False),
        ),
        peds_behaviors=[
            SimpleNamespace(
                single_pedestrians=[
                    SimpleNamespace(
                        role="follow",
                        goal=None,
                        trajectory=None,
                        hold_until_robot_within_m=None,
                    )
                ]
            )
        ],
    )
    model = SimulatorCounterfactualModel(sim)

    assert model.pedestrian_response == "closed_loop"


def test_all_collision_scope_detects_wall_without_pedestrian() -> None:
    """The explicit all-collisions mode cannot call a wall state safe."""
    robot = SimpleNamespace(
        pos=(1.0, 1.0),
        config=SimpleNamespace(radius=0.5),
        state=SimpleNamespace(),
    )
    sim = SimpleNamespace(
        robots=[robot],
        map_def=SimpleNamespace(width=10.0, height=10.0),
        ped_pos=np.empty((0, 2)),
        config=SimpleNamespace(
            ped_radius=0.4,
            prf_config=SimpleNamespace(is_active=False),
            apf_config=SimpleNamespace(is_active=False),
            residual_adversary=SimpleNamespace(is_active=False),
        ),
        get_obstacle_lines=lambda: np.asarray([[1.4, 0.0, 1.4, 2.0]]),
    )
    model = SimulatorCounterfactualModel(sim, collision_scope="all")
    assert model.collision_predicate == (
        "continuous_occupancy_robot_bounds_obstacles_pedestrians_v1"
    )
    assert model.collision() is True


def test_route_group_navigator_progress_restores() -> None:
    """Route-group waypoint progress is restored rather than left at branch state."""
    navigator = RouteNavigator([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    navigator.waypoint_id = 1
    navigator.reached_waypoint = False
    behavior = SimpleNamespace(navigators={7: navigator})
    snapshot = _capture_route_navigators([behavior])

    navigator.waypoint_id = 2
    navigator.reached_waypoint = True
    _restore_route_navigators([behavior], snapshot)

    assert navigator.waypoint_id == 1
    assert navigator.reached_waypoint is False


def test_no_contact_baseline_abstains() -> None:
    """A stable baseline without contact cannot produce an avoidability verdict."""
    scenario = KinematicScenario(
        robot_x0=0.0,
        robot_speed0=0.0,
        ped_pos0=(10.0, 10.0),
        ped_vel0=(0.0, 0.0),
    )
    report = locate_last_avoidable(
        KinematicCollisionModel(scenario),
        [0.0] * 10,
        ReplayConfig(
            t_danger=0,
            t_contact=3,
            horizon=3,
            action_set_id="test",
            feasibility_filter="test",
            collision_predicate="test",
            pedestrian_response="replayed",
        ),
    )

    assert report.verdict == VERDICT_UNKNOWN
    assert report.abstain_reason == "baseline_no_contact"
