"""Tests for the shared PySocialForce initialization factory (issue #6465).

Covers the extracted ``_compute_pedestrian_response_multipliers`` helper and the
behavior-preserving divergence contract between ``Simulator`` (computed
per-pedestrian multipliers) and ``PedSimulator`` (``None`` multipliers, kept per
the issue #4618 R2 rationale).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from pysocialforce.config import SURFACE_DISTANCE_UNIT_NORMAL_V2
from pysocialforce.forces import ObstacleForce

from robot_sf.common.types import Line2D, Rect
from robot_sf.gym_env.unified_config import (
    PedestrianSimulationConfig,
    RobotSimulationConfig,
)
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import (
    MapDefinition,
    MapDefinitionPool,
    SinglePedestrianDefinition,
)
from robot_sf.ped_npc.ped_behavior import SinglePedestrianBehavior
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import (
    Simulator,
    _compute_pedestrian_response_multipliers,
    init_ped_simulators,
    init_simulators,
)


def _config_stub(**overrides) -> SimpleNamespace:
    """Build a lightweight config exposing only the multiplier-relevant fields."""
    base = {
        "pedestrian_control_trace_labels": None,
        "response_law_composition": None,
        "response_law_seed": None,
        "non_reactive_response_multiplier": 0.5,
        "hesitating_response_multiplier": 0.75,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_response_multipliers_default_to_ones() -> None:
    """Without labels or composition every multiplier stays at the 1.0 default."""
    multipliers = _compute_pedestrian_response_multipliers(_config_stub(), 4)

    np.testing.assert_array_equal(multipliers, np.ones(4))


def test_response_multipliers_empty_for_zero_peds() -> None:
    """Zero pedestrians must yield an empty vector without any index access."""
    multipliers = _compute_pedestrian_response_multipliers(_config_stub(), 0)

    assert multipliers.shape == (0,)


def test_response_multipliers_apply_control_trace_labels_by_index() -> None:
    """Control-trace labels override multipliers per ``simulator_index``."""
    config = _config_stub(
        pedestrian_control_trace_labels=[
            {"simulator_index": 0, "response_law": "non_reactive"},
            {"simulator_index": 2, "response_law": "hesitating"},
            {"simulator_index": 5, "response_law": "non_yielding"},  # out of range -> ignored
            {"simulator_index": None, "response_law": "non_reactive"},  # no index -> ignored
        ],
    )

    multipliers = _compute_pedestrian_response_multipliers(config, 4)

    np.testing.assert_allclose(multipliers, [0.5, 1.0, 0.75, 1.0])


def test_response_multipliers_fall_back_to_archetype_composition() -> None:
    """Without trace labels, ``response_law_composition`` assigns multipliers per archetype."""
    config = _config_stub(
        response_law_composition={"non_reactive": 0.5, "reactive": 0.5},
        response_law_seed=7,
    )

    multipliers = _compute_pedestrian_response_multipliers(config, 5)

    assert multipliers.shape == (5,)
    # Each multiplier is either the default 1.0 (reactive) or the non_reactive scale (0.5).
    assert set(np.unique(multipliers)).issubset({0.5, 1.0})


def test_response_multipliers_composition_applies_hesitating_scale() -> None:
    """The composition branch must also apply the hesitating multiplier scale."""
    config = _config_stub(
        response_law_composition={"hesitating": 1.0},
        response_law_seed=11,
    )

    multipliers = _compute_pedestrian_response_multipliers(config, 3)

    np.testing.assert_allclose(multipliers, np.full(3, 0.75))


def test_control_trace_labels_take_precedence_over_composition() -> None:
    """The trace-labels branch wins over the response_law_composition branch."""
    config = _config_stub(
        pedestrian_control_trace_labels=[{"simulator_index": 1, "response_law": "hesitating"}],
        response_law_composition={"non_reactive": 1.0},
    )

    multipliers = _compute_pedestrian_response_multipliers(config, 3)

    np.testing.assert_allclose(multipliers, [1.0, 0.75, 1.0])


def _minimal_map(
    *, single_pedestrians: list[SinglePedestrianDefinition] | None = None
) -> MapDefinition:
    """Build a compact map with one deterministic robot route (mirrors sim tests)."""
    width = 10.0
    height = 10.0
    spawn_zone: Rect = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
    goal_zone: Rect = ((8.0, 8.0), (9.0, 8.0), (8.0, 9.0))
    bounds: list[Line2D] = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.2, 1.2), (8.8, 8.8)],
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
        single_pedestrians=single_pedestrians or [],
    )


def _zero_ped_sim_config(map_def: MapDefinition) -> RobotSimulationConfig:
    return RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.0]),
    )


def test_simulator_wires_computed_response_multipliers() -> None:
    """Simulator.__post_init__ keeps computing the per-pedestrian multiplier vector."""
    map_def = _minimal_map()
    simulator = init_simulators(
        _zero_ped_sim_config(map_def),
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]

    assert isinstance(simulator.pedestrian_response_multipliers, np.ndarray)


def test_simulator_wires_opt_in_obstacle_force_law_to_fast_pysf() -> None:
    """SimulationSettings law selection reaches PySocialForce and its metadata seam."""
    map_def = _minimal_map()
    config = _zero_ped_sim_config(map_def)
    config.sim_config = SimulationSettings(
        difficulty=0,
        ped_density_by_difficulty=[0.0],
        obstacle_force_law=SURFACE_DISTANCE_UNIT_NORMAL_V2,
    )

    simulator = init_simulators(
        config,
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]

    assert simulator.pysf_sim.config.obstacle_force_config.law_version == (
        SURFACE_DISTANCE_UNIT_NORMAL_V2
    )
    assert simulator.obstacle_force_law_metadata()["law_version"] == SURFACE_DISTANCE_UNIT_NORMAL_V2


def test_simulator_metadata_fallback_reports_disabled_obstacle_forces() -> None:
    """The compatibility metadata path reports inactive obstacle forces safely."""
    simulator = object.__new__(Simulator)
    simulator.config = SimpleNamespace(obstacle_force_law=None)
    simulator.pysf_sim = SimpleNamespace(forces=(), raw_obstacles=())

    metadata = simulator.obstacle_force_law_metadata()

    assert metadata["site"] == "fast_pysf"
    assert metadata["enabled"] is False
    assert metadata["applied"] is False


def test_simulator_metadata_fallback_reports_obstacle_application_state() -> None:
    """The compatibility metadata fallback reports enabled and applied state."""
    simulator = object.__new__(Simulator)
    simulator.pysf_sim = SimpleNamespace(forces=(), raw_obstacles=())
    simulator.config = SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.0])

    metadata = simulator.obstacle_force_law_metadata()

    assert metadata["enabled"] is False
    assert metadata["applied"] is False
    assert metadata["resolution_mode"] == "defaulted_missing"


def test_simulator_metadata_fallback_respects_zero_obstacle_force_factor() -> None:
    """The compatibility metadata path treats a zero force factor as disabled."""
    simulator = object.__new__(Simulator)
    zero_factor_force = object.__new__(ObstacleForce)
    zero_factor_force.config = SimpleNamespace(factor=0.0)
    simulator.pysf_sim = SimpleNamespace(
        forces=(zero_factor_force,),
        raw_obstacles=(object(),),
    )
    simulator.config = SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.0])

    metadata = simulator.obstacle_force_law_metadata()

    assert metadata["enabled"] is False
    assert metadata["applied"] is False


def test_ped_simulator_keeps_none_response_multipliers() -> None:
    """PedSimulator.__post_init__ preserves the issue #4618 R2 ``None`` divergence."""
    map_def = _minimal_map()
    config = PedestrianSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.0]),
    )
    simulator = init_ped_simulators(
        config,
        map_def,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]

    assert simulator.pedestrian_response_multipliers is None


def test_single_pedestrian_provider_tracks_replaced_robot_collection() -> None:
    """The extracted factory must preserve the dynamic ``self.robot_poses`` callback."""
    map_def = _minimal_map(
        single_pedestrians=[
            SinglePedestrianDefinition(
                id="scripted",
                start=(3.0, 3.0),
                trajectory=[(4.0, 3.0)],
            )
        ]
    )
    simulator = init_simulators(
        _zero_ped_sim_config(map_def),
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]
    behavior = next(
        behavior
        for behavior in simulator.peds_behaviors
        if isinstance(behavior, SinglePedestrianBehavior)
    )
    replacement_pose = ((7.0, 8.0), 1.25)
    simulator.robots = [SimpleNamespace(pose=replacement_pose)]  # type: ignore[list-item]

    assert behavior.robot_pose_provider is not None
    assert behavior.robot_pose_provider() == [replacement_pose]


def test_simulator_defaults_obstacle_forces_to_false_when_unset() -> None:
    """A ``None`` obstacle-force flag is resolved to False (issue #6465 guard preserved)."""
    map_def = _minimal_map()
    simulator = init_simulators(
        _zero_ped_sim_config(map_def),
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=None,  # type: ignore[arg-type]
    )[0]

    assert simulator.peds_have_obstacle_forces is False
