"""Simulator integration tests for the bounded residual adversary (#4360).

Covers the runtime wiring contract: opt-in gating (off by default), base-law
preservation (the adversary perturbs rather than replaces the nominal Social
Force pedestrian behavior), and a short end-to-end smoke that enabling the
adversary keeps pedestrian positions finite over a bounded horizon.

Capability-only slice: no benchmark, planner-ranking, safety, or paper-facing
claim is made here.
"""

from __future__ import annotations

import numpy as np

from robot_sf.common.types import Line2D, Rect
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig, RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.ped_npc.residual_adversary import ResidualAdversaryConfig
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import init_ped_simulators, init_simulators


def _minimal_map() -> MapDefinition:
    """Build a compact map with one robot route and a pedestrian spawn zone."""
    width = 20.0
    height = 20.0
    spawn_zone: Rect = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
    goal_zone: Rect = ((16.0, 16.0), (17.0, 16.0), (16.0, 17.0))
    bounds: list[Line2D] = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.2, 1.2), (16.8, 16.8)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    return MapDefinition(
        width=width,
        height=height,
        obstacles=[],
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[spawn_zone],
        robot_goal_zones=[goal_zone],
        bounds=bounds,
        robot_routes=[route],
        ped_goal_zones=[goal_zone],
        ped_crowded_zones=[],
        ped_routes=[route],
        single_pedestrians=[],
    )


def _build_simulator(residual_active: bool):
    """Construct a single-robot simulator with the residual adversary flag set."""
    map_def = _minimal_map()
    sim_config = SimulationSettings(
        sim_time_in_secs=4.0,
        time_per_step_in_secs=0.1,
        difficulty=0,
        ped_density_by_difficulty=[0.02, 0.02, 0.02, 0.02],
        residual_adversary=ResidualAdversaryConfig(
            is_active=residual_active,
            target_ped_idx=-1,
            max_residual_accel_mps2=1.0,
            max_jerk_mps3=1e9,
        ),
    )
    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=sim_config,
    )
    return init_simulators(
        config,
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]


def test_residual_adversary_is_off_by_default_in_sim_config() -> None:
    """The default SimulationSettings must not activate the residual adversary."""
    settings = SimulationSettings()
    assert settings.residual_adversary.is_active is False


def test_residual_adversary_opt_in_is_reflected_in_sim_config() -> None:
    """Opting in via config must normalize to an active config without error."""
    settings = SimulationSettings(residual_adversary=ResidualAdversaryConfig(is_active=True))
    assert settings.residual_adversary.is_active is True


def test_residual_adversary_mapping_is_normalized_by_sim_config() -> None:
    """A mapping config must become the validated residual-adversary dataclass."""
    settings = SimulationSettings(residual_adversary={"is_active": True, "seed": 7})
    assert isinstance(settings.residual_adversary, ResidualAdversaryConfig)
    assert settings.residual_adversary.seed == 7


def test_simulator_inactive_adversary_does_not_allocate_state() -> None:
    """When off, the simulator must not build any residual adversary state."""
    sim = _build_simulator(residual_active=False)
    assert sim._residual_adversary is None
    # Applying the helper on inactive config returns forces unchanged.
    forces = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(sim._apply_residual_adversary(forces), forces)


def test_base_law_preserved_when_adversary_inactive() -> None:
    """With the adversary off, stepping produces finite state and a no-op helper."""
    sim_off = _build_simulator(residual_active=False)
    sim_off.step_once([(0.0, 0.0)])
    forces_off = np.asarray(sim_off.last_ped_forces, dtype=float).copy()
    positions_off = np.asarray(sim_off.ped_pos, dtype=float).copy()
    # The helper is a pure no-op on the inactive path.
    np.testing.assert_allclose(sim_off._apply_residual_adversary(forces_off), forces_off)
    assert np.all(np.isfinite(positions_off))


def test_active_adversary_perturbs_but_keeps_peds_finite() -> None:
    """An active adversary adds a bounded residual and keeps positions finite."""
    sim = _build_simulator(residual_active=True)
    for _ in range(20):
        sim.step_once([(0.0, 0.0)])
        positions = np.asarray(sim.ped_pos, dtype=float)
        velocities = np.asarray(sim.ped_vel, dtype=float)
        if positions.size and not np.all(np.isfinite(positions)):
            raise AssertionError("non-finite pedestrian position after enabling adversary")
        if velocities.size and not np.all(np.isfinite(velocities)):
            raise AssertionError("non-finite pedestrian velocity after enabling adversary")
    # The residual adversary state must have been lazily allocated.
    assert sim._residual_adversary is not None
    # Cadence: 0.5 s macro-action at 0.1 s physics step => 5 steps per macro-action.
    assert sim._residual_adversary.macro_action_steps == 5


def test_active_adversary_residual_is_additive_within_accel_bound() -> None:
    """The applied residual is bounded by the configured acceleration magnitude."""
    sim = _build_simulator(residual_active=True)
    max_accel = sim.config.residual_adversary.max_residual_accel_mps2
    for _ in range(15):
        sim.step_once([(0.0, 0.0)])
        residual = sim._residual_adversary.last_residual
        if residual.size:
            assert np.linalg.norm(residual, axis=1).max() <= max_accel + 1e-9


def test_simulator_reset_clears_residual_adversary_state() -> None:
    """A fresh episode reset also restarts the stateful residual controller."""
    sim = _build_simulator(residual_active=True)
    sim.step_once([(0.0, 0.0)])
    assert sim._residual_adversary is not None
    assert sim._residual_adversary.step_index == 1
    sim.reset_state()
    assert sim._residual_adversary.step_index == 0
    assert sim._residual_adversary.macro_action_index == 0


def test_active_adversary_changes_forces_vs_inactive() -> None:
    """Enabling the adversary must change at least one force vs the inactive baseline.

    This confirms the residual is genuinely additive (perturbs) rather than a no-op,
    while both runs stay finite. It is not a benchmark or safety claim.
    """
    sim_off = _build_simulator(residual_active=False)
    sim_on = _build_simulator(residual_active=True)
    sim_off.step_once([(0.0, 0.0)])
    sim_on.step_once([(0.0, 0.0)])
    forces_off = np.asarray(sim_off.last_ped_forces, dtype=float)
    forces_on = np.asarray(sim_on.last_ped_forces, dtype=float)
    if forces_off.size and forces_on.size:
        assert not np.allclose(forces_off, forces_on)


def test_apply_residual_adversary_short_circuits_empty_crowd() -> None:
    """An empty force matrix must short-circuit without allocating adversary state."""
    sim = _build_simulator(residual_active=True)
    empty_forces = np.zeros((0, 2), dtype=float)
    out = sim._apply_residual_adversary(empty_forces)
    assert out.shape == (0, 2)
    assert sim._residual_adversary is None


def test_ped_simulator_excludes_controlled_ego_from_adversary_targets() -> None:
    """The residual policy may target NPC rows, never the controlled ego row."""
    map_def = _minimal_map()
    sim_config = SimulationSettings(
        sim_time_in_secs=4.0,
        time_per_step_in_secs=0.1,
        difficulty=0,
        ped_density_by_difficulty=[0.02, 0.02, 0.02, 0.02],
        residual_adversary=ResidualAdversaryConfig(
            is_active=True,
            target_ped_idx=-1,
            max_jerk_mps3=1e9,
        ),
    )
    config = PedestrianSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=sim_config,
    )
    sim = init_ped_simulators(
        config,
        map_def,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]

    adversary = sim._build_residual_adversary()

    assert adversary is not None
    assert adversary._target_mask[:-1].all()
    assert not adversary._target_mask[-1]
    sim.step_once([(0.0, 0.0)], ego_ped_actions=[(0.0, 0.0)])
    assert sim._residual_adversary is not None
    np.testing.assert_allclose(sim._residual_adversary.last_residual[-1], [0.0, 0.0])


def test_collect_helpers_forward_geometry_from_map(monkeypatch) -> None:
    """The collect-helpers forward routes/obstacles/bounds and degrade to None.

    Exercises the realistic defensive branches: a map without pedestrian routes
    yields ``None`` polylines, and a map without obstacles yields ``None`` segments,
    while a normal map forwards finite geometry to the bounded adversary.
    """
    sim = _build_simulator(residual_active=True)
    # Normal map: routes and obstacles (bounds) are present.
    route_polylines = sim._collect_residual_route_polylines()
    assert route_polylines is not None
    assert set(route_polylines).issubset(set(range(sim.pysf_state.num_peds)))
    assert all(polyline.shape[1] == 2 for polyline in route_polylines.values())
    obstacle_segments = sim._collect_residual_obstacle_segments()
    assert obstacle_segments is not None
    # MapDefinition keeps fast-pysf's legacy [x1, x2, y1, y2] tuples. The
    # residual projection must receive conventional endpoint coordinates.
    np.testing.assert_allclose(obstacle_segments[0], [0.0, 0.0, 20.0, 0.0])
    assert sim._collect_residual_map_bounds() is not None

    # No runtime route assignments degrades the polyline source to None.
    for behavior in sim.peds_behaviors:
        if hasattr(behavior, "route_assignments"):
            monkeypatch.setattr(behavior, "route_assignments", {})
    assert sim._collect_residual_route_polylines() is None

    # Map without obstacle segments degrades the obstacle source to None.
    monkeypatch.setattr(sim.map_def, "obstacles_pysf", [])
    assert sim._collect_residual_obstacle_segments() is None

    # Non-finite bounds degrade the bounds source to None.
    monkeypatch.setattr(sim.map_def, "get_map_bounds", lambda: (float("inf"), 1.0, 0.0, 1.0))
    assert sim._collect_residual_map_bounds() is None
