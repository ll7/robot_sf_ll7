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
    _capture_single_runtimes,
    _restore_route_navigators,
    _restore_single_runtimes,
)
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.navigation import RouteNavigator
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.ped_npc.ped_behavior import SinglePedestrianRuntime
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
        action_set_id="unspecified",
        feasibility_filter="unspecified",
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
    assert model.replay_state_complete == "false"

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


def test_single_runtime_snapshot_round_trip_restores_dataclass_fields() -> None:
    """Single-pedestrian runtime fields are deep-copied and reconstructed."""
    runtime = SinglePedestrianRuntime(
        ped_id=3,
        definition=SimpleNamespace(id="ped-3"),
        trajectory=[(1.0, 2.0)],
        waypoint_index=1,
        pending_waits={1: 0.5},
    )
    behavior = SimpleNamespace(_runtimes=[runtime])
    saved = _capture_single_runtimes([behavior])

    runtime.waypoint_index = 9
    runtime.pending_waits.clear()
    _restore_single_runtimes([behavior], saved)

    assert behavior._runtimes[0].waypoint_index == 1
    assert behavior._runtimes[0].pending_waits == {1: 0.5}


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
    assert report.determinism.observed_contact_steps == (_FIXTURE_CONTACT_STEP,) * 5
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


@pytest.mark.parametrize(
    ("definition", "expected"),
    (
        (
            SimpleNamespace(
                role="lead", goal=None, trajectory=None, hold_until_robot_within_m=None
            ),
            "closed_loop",
        ),
        (
            SimpleNamespace(
                role="lead", goal=(2.0, 0.0), trajectory=None, hold_until_robot_within_m=None
            ),
            "replayed",
        ),
        (
            SimpleNamespace(role="wait", goal=None, trajectory=None, hold_until_robot_within_m=1.0),
            "closed_loop",
        ),
    ),
)
def test_pedestrian_response_covers_robot_relative_lead_and_hold(definition, expected) -> None:
    """Lead and proximity-hold definitions bind robot pose provenance correctly."""
    robot = SimpleNamespace(pos=(0.0, 0.0), config=SimpleNamespace(radius=0.5))
    sim = SimpleNamespace(
        robots=[robot],
        config=SimpleNamespace(
            prf_config=SimpleNamespace(is_active=False),
            apf_config=SimpleNamespace(is_active=False),
            residual_adversary=SimpleNamespace(is_active=False),
        ),
        peds_behaviors=[SimpleNamespace(single_pedestrians=[definition])],
    )

    assert SimulatorCounterfactualModel(sim).pedestrian_response == expected


@pytest.mark.parametrize(
    ("config", "state", "label_fragment"),
    (
        (
            SimpleNamespace(max_linear_decel=4.0, max_angular_accel=2.0),
            SimpleNamespace(),
            "angular_radps2=2",
        ),
        (
            SimpleNamespace(max_decel=3.0, max_steer=0.4),
            SimpleNamespace(),
            "steering_angle_rad=0.4",
        ),
        (
            SimpleNamespace(command_mode="vx_vy", max_speed=2.0),
            SimpleNamespace(velocity_xy=(0.5, -0.25)),
            "vy_mps=2",
        ),
        (
            SimpleNamespace(command_mode="unicycle_vw", max_angular_speed=1.2),
            SimpleNamespace(velocity_vw=(0.7, -0.1)),
            "angular_radps=1.2",
        ),
    ),
)
def test_native_action_lattices_and_labels_follow_robot_contracts(
    config, state, label_fragment
) -> None:
    """Every supported native command mode exposes labeled feasible actions."""
    robot = SimpleNamespace(config=config, state=state)
    model = SimulatorCounterfactualModel(SimpleNamespace(robots=[robot]))

    actions = model.feasible_actions()

    assert actions
    assert label_fragment in model.action_label(actions[-1])


def test_native_replay_metadata_binds_action_contract_and_source() -> None:
    """Native reports bind the executed lattice instead of trusting caller labels."""
    robot = SimpleNamespace(
        config=SimpleNamespace(max_linear_decel=4.0, max_angular_accel=2.0),
        state=SimpleNamespace(),
    )
    sim = SimpleNamespace(
        robots=[robot],
        config=SimpleNamespace(
            prf_config=SimpleNamespace(is_active=False),
            apf_config=SimpleNamespace(is_active=False),
            residual_adversary=SimpleNamespace(is_active=False),
        ),
        peds_behaviors=[],
    )
    model = SimulatorCounterfactualModel(sim, capture_rng=False)
    report = locate_last_avoidable(
        model,
        [],
        ReplayConfig(
            t_danger=0,
            t_contact=1,
            horizon=1,
            action_set_id="two_action_lattice",
            feasibility_filter="caller_declared_filter",
            collision_predicate="robot_pedestrian_center_distance_v1",
            pedestrian_response="replayed",
        ),
    )

    assert report.verdict == VERDICT_UNKNOWN
    assert report.abstain_reason == "metadata_mismatch"
    assert "action_set_id" in report.notes[0]
    assert "feasibility_filter" in report.notes[0]

    native = locate_last_avoidable(
        model,
        [],
        ReplayConfig(t_danger=0, t_contact=1, horizon=1),
    )
    assert native.config.source_kind == "live_episode"
    assert native.config.action_set_id.startswith("simulator_native_action_lattice_v1:")
    assert native.config.feasibility_filter == "native_declared_action_lattice_v1:n=5"


def test_unknown_action_contract_fails_closed_and_uses_generic_label() -> None:
    """Unknown command semantics expose no feasible substitutions."""
    robot = SimpleNamespace(config=SimpleNamespace(command_mode="unknown"), state=SimpleNamespace())
    model = SimulatorCounterfactualModel(SimpleNamespace(robots=[robot]))

    assert model.feasible_actions() == ()
    assert model.action_label((1.0, 2.0)) == "robot_cmd=(first=1,second=2)"


def test_adapter_rejects_invalid_robot_count_and_collision_scope() -> None:
    """The production adapter fails closed for unsupported construction contracts."""
    with pytest.raises(ValueError, match="exactly one robot"):
        SimulatorCounterfactualModel(SimpleNamespace(robots=[]))

    robot = SimpleNamespace(config=SimpleNamespace(radius=0.5))
    with pytest.raises(ValueError, match="collision_scope"):
        SimulatorCounterfactualModel(SimpleNamespace(robots=[robot]), collision_scope="invalid")


@pytest.mark.parametrize(
    ("robot_pos", "ped_pos", "expected"),
    (
        ((-0.1, 1.0), np.empty((0, 2)), True),
        ((1.0, 1.0), np.empty((0, 2)), False),
        ((1.0, 1.0), np.asarray([[1.8, 1.0]]), True),
    ),
)
def test_all_collision_scope_covers_bounds_empty_and_pedestrian_contacts(
    robot_pos, ped_pos, expected
) -> None:
    """The opt-in all-collision predicate covers each native contact class."""
    robot = SimpleNamespace(
        pos=robot_pos,
        config=SimpleNamespace(radius=0.5),
        state=SimpleNamespace(),
    )
    sim = SimpleNamespace(
        robots=[robot],
        map_def=SimpleNamespace(width=10.0, height=10.0),
        ped_pos=ped_pos,
        config=SimpleNamespace(ped_radius=0.4),
        get_obstacle_lines=lambda: (),
    )

    model = SimulatorCounterfactualModel(sim, collision_scope="all")

    assert model.collision() is expected


def test_custom_collision_predicate_is_used_and_labeled() -> None:
    """A caller-supplied collision predicate takes precedence over native scope."""
    robot = SimpleNamespace(config=SimpleNamespace(radius=0.5))
    model = SimulatorCounterfactualModel(
        SimpleNamespace(robots=[robot]), collision_fn=lambda _: True
    )

    assert model.collision() is True
    assert model.collision_predicate == "custom_collision_fn"


def test_pedestrian_response_covers_lead_hold_and_replayed_paths() -> None:
    """Classify the remaining native behavior-response modes explicitly."""
    robot = SimpleNamespace(pos=(0.0, 0.0), config=SimpleNamespace(radius=0.5))
    config = SimpleNamespace(
        prf_config=SimpleNamespace(is_active=False),
        apf_config=SimpleNamespace(is_active=False),
        residual_adversary=SimpleNamespace(is_active=False),
    )

    lead_sim = SimpleNamespace(
        robots=[robot],
        config=config,
        peds_behaviors=[
            SimpleNamespace(
                single_pedestrians=[
                    SimpleNamespace(
                        role="lead", goal=None, trajectory=None, hold_until_robot_within_m=None
                    )
                ]
            )
        ],
    )
    assert SimulatorCounterfactualModel(lead_sim).pedestrian_response == "closed_loop"

    hold_sim = SimpleNamespace(
        robots=[robot],
        config=config,
        peds_behaviors=[
            SimpleNamespace(
                single_pedestrians=[
                    SimpleNamespace(
                        role="wander",
                        goal=(1.0, 1.0),
                        trajectory=None,
                        hold_until_robot_within_m=0.5,
                    )
                ]
            )
        ],
    )
    assert SimulatorCounterfactualModel(hold_sim).pedestrian_response == "closed_loop"

    replayed_sim = SimpleNamespace(
        robots=[robot],
        config=config,
        peds_behaviors=[SimpleNamespace(single_pedestrians=[])],
    )
    assert SimulatorCounterfactualModel(replayed_sim).pedestrian_response == "replayed"


def test_native_action_lattices_and_labels_cover_robot_modes() -> None:
    """Use each supported native action convention and fail closed for unknown modes."""
    bicycle = SimpleNamespace(
        config=SimpleNamespace(max_decel=2.0, max_steer=0.4),
        state=SimpleNamespace(),
    )
    bicycle_model = SimulatorCounterfactualModel(SimpleNamespace(robots=[bicycle]))
    bicycle_actions = bicycle_model.feasible_actions()
    assert bicycle_actions[-1] == (0.0, 0.4)
    assert "steering_angle_rad=" in bicycle_model.action_label(bicycle_actions[-1])

    holonomic = SimpleNamespace(
        config=SimpleNamespace(command_mode="vx_vy", max_speed=2.0),
        state=SimpleNamespace(velocity_xy=(0.3, -0.2)),
    )
    holonomic_model = SimulatorCounterfactualModel(SimpleNamespace(robots=[holonomic]))
    assert holonomic_model.feasible_actions()[0] == (0.3, -0.2)
    assert holonomic_model.action_label((0.3, -0.2)).startswith("robot_velocity=(vx_mps=")

    unicycle = SimpleNamespace(
        config=SimpleNamespace(command_mode="unicycle_vw", max_angular_speed=0.7),
        state=SimpleNamespace(velocity_vw=(0.5, 0.1)),
    )
    unicycle_model = SimulatorCounterfactualModel(SimpleNamespace(robots=[unicycle]))
    assert unicycle_model.feasible_actions()[2] == (0.5, -0.7)
    assert "angular_radps=" in unicycle_model.action_label((0.5, -0.7))

    unknown = SimpleNamespace(config=SimpleNamespace(), state=SimpleNamespace())
    unknown_model = SimulatorCounterfactualModel(SimpleNamespace(robots=[unknown]))
    assert unknown_model.feasible_actions() == ()
    assert unknown_model.action_label((1.0, 2.0)) == "robot_cmd=(first=1,second=2)"


def test_adapter_validation_and_custom_collision_provenance() -> None:
    """Reject unsupported construction and preserve custom collision provenance."""
    with pytest.raises(ValueError, match="exactly one robot"):
        SimulatorCounterfactualModel(SimpleNamespace(robots=[]))

    robot = SimpleNamespace(pos=(0.0, 0.0), config=SimpleNamespace(radius=0.5))
    with pytest.raises(ValueError, match="collision_scope"):
        SimulatorCounterfactualModel(SimpleNamespace(robots=[robot]), collision_scope="unknown")

    model = SimulatorCounterfactualModel(
        SimpleNamespace(robots=[robot]), collision_fn=lambda _: True
    )
    assert model.collision_predicate == "custom_collision_fn"
    assert model.collision() is True


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


def test_adapter_contract_edges_cover_native_action_and_collision_branches() -> None:
    """Exercise fail-closed constructor, collision, and action-label branches."""
    robot = SimpleNamespace(
        pos=(1.0, 1.0),
        config=SimpleNamespace(radius=0.5),
        state=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="exactly one robot"):
        SimulatorCounterfactualModel(SimpleNamespace(robots=[]))
    with pytest.raises(ValueError, match="collision_scope"):
        SimulatorCounterfactualModel(SimpleNamespace(robots=[robot]), collision_scope="unknown")

    sim = SimpleNamespace(
        robots=[robot],
        robot_pos=np.asarray([[1.0, 1.0]]),
        ped_pos=np.empty((0, 2)),
        map_def=SimpleNamespace(width=10.0, height=10.0),
        config=SimpleNamespace(ped_radius=0.4),
        get_obstacle_lines=lambda: (),
        peds_behaviors=[],
    )
    model = SimulatorCounterfactualModel(sim)
    assert model.collision() is False
    assert model.collision_predicate == "robot_pedestrian_center_distance_v1"

    robot.pos = (-0.1, 1.0)
    all_model = SimulatorCounterfactualModel(sim, collision_scope="all")
    assert all_model.collision() is True
    robot.pos = (1.0, 1.0)
    sim.ped_pos = np.asarray([[1.2, 1.0]])
    assert all_model.collision() is True

    custom = SimulatorCounterfactualModel(sim, collision_fn=lambda _: True)
    assert custom.collision() is True
    assert custom.collision_predicate == "custom_collision_fn"

    bicycle_config = SimpleNamespace(max_decel=2.0, max_steer=0.4)
    robot.config = bicycle_config
    bicycle = SimulatorCounterfactualModel(sim)
    assert len(bicycle.feasible_actions()) == 5
    assert bicycle.action_label((0.0, 0.0)).startswith("robot_accel=")

    robot.config = SimpleNamespace(command_mode="vx_vy", max_speed=1.5)
    robot.state = SimpleNamespace(velocity_xy=(0.2, -0.1))
    velocity_xy = SimulatorCounterfactualModel(sim)
    assert len(velocity_xy.feasible_actions()) == 4
    assert velocity_xy.action_label((0.0, 0.0)).startswith("robot_velocity=(vx_mps=")

    robot.config = SimpleNamespace(command_mode="unicycle_vw", max_angular_speed=2.0)
    robot.state = SimpleNamespace(velocity_vw=(0.3, 0.1))
    unicycle = SimulatorCounterfactualModel(sim)
    assert len(unicycle.feasible_actions()) == 4
    assert unicycle.action_label((0.0, 0.0)).startswith("robot_velocity=(linear_mps=")

    robot.config = SimpleNamespace()
    unknown = SimulatorCounterfactualModel(sim)
    assert unknown.feasible_actions() == ()
    assert unknown.action_label((0.0, 0.0)).startswith("robot_cmd=")


def test_all_collision_scope_covers_bounds_and_pedestrian_footprints() -> None:
    """Cover non-wall all-scope outcomes for bounds, empty space, and pedestrians."""
    base_config = SimpleNamespace(ped_radius=0.4)

    out_of_bounds_robot = SimpleNamespace(
        pos=(-0.1, 1.0), config=SimpleNamespace(radius=0.5), state=SimpleNamespace()
    )
    out_of_bounds_sim = SimpleNamespace(
        robots=[out_of_bounds_robot],
        map_def=SimpleNamespace(width=10.0, height=10.0),
        ped_pos=np.empty((0, 2)),
        config=base_config,
    )
    assert SimulatorCounterfactualModel(out_of_bounds_sim, collision_scope="all").collision()

    empty_robot = SimpleNamespace(
        pos=(1.0, 1.0), config=SimpleNamespace(radius=0.5), state=SimpleNamespace()
    )
    empty_sim = SimpleNamespace(
        robots=[empty_robot],
        map_def=SimpleNamespace(width=10.0, height=10.0),
        ped_pos=np.empty((0, 2)),
        config=base_config,
    )
    assert not SimulatorCounterfactualModel(empty_sim, collision_scope="all").collision()

    pedestrian_sim = SimpleNamespace(
        robots=[empty_robot],
        map_def=SimpleNamespace(width=10.0, height=10.0),
        ped_pos=np.asarray([[1.8, 1.0]]),
        config=base_config,
        get_obstacle_lines=lambda: (),
    )
    assert SimulatorCounterfactualModel(pedestrian_sim, collision_scope="all").collision()


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
    _restore_route_navigators([behavior], {})
    _restore_route_navigators([behavior], {id(behavior): {8: {}}})


def test_pedestrian_only_collision_is_false_for_empty_crowd() -> None:
    """The historical pedestrian-only predicate ignores an empty crowd."""
    robot = SimpleNamespace(pos=(0.0, 0.0), config=SimpleNamespace(radius=0.5))
    sim = SimpleNamespace(
        robots=[robot],
        robot_pos=np.asarray([[0.0, 0.0]]),
        ped_pos=np.empty((0, 2)),
    )

    assert SimulatorCounterfactualModel(sim).collision() is False


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
