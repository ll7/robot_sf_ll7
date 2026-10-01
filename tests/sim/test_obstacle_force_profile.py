"""Versioned wall calibration must reach the actual pedestrian force object.

No environment or planner steps; force samples use dev seed 1001.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from pysocialforce.forces import ObstacleForce

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import _stable_config_hash
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import Simulator, _build_pysf_simulation
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.calibrate_obstacle_force_10061 import FOOTPRINT_RADIUS
from scripts.validation.calibrate_obstacle_force_10061 import config as calibration_config


def _doorway_force(profile: str | None, law: str | None = None):
    """Construct the production pedestrian substrate for a 1.2 m opening."""
    np.random.seed(1001)
    settings = SimulationSettings(
        difficulty=0, ped_density_by_difficulty=[0.0], obstacle_force_law=law
    )
    # Assignment works on the pre-fix class too, exposing silently ignored wiring.
    settings.obstacle_force_profile = profile
    map_def = MapDefinition(
        width=16.0,
        height=4.0,
        obstacles=[
            Obstacle([(8.0, 2.6), (8.1, 2.6), (8.1, 4.0), (8.0, 4.0)]),
            Obstacle([(8.0, 0.0), (8.1, 0.0), (8.1, 1.4), (8.0, 1.4)]),
        ],
        robot_spawn_zones=[],
        ped_spawn_zones=[],
        robot_goal_zones=[],
        bounds=[
            ((0.0, 0.0), (16.0, 0.0)),
            ((16.0, 0.0), (16.0, 4.0)),
            ((16.0, 4.0), (0.0, 4.0)),
            ((0.0, 4.0), (0.0, 0.0)),
        ],
        robot_routes=[],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=[
            SinglePedestrianDefinition(id="walker", start=(6.0, 2.0), goal=(15.0, 2.0))
        ],
    )
    substrate, *_ = _build_pysf_simulation(
        config=settings,
        map_def=map_def,
        robots=[],
        robot_pose_provider=lambda: [],
        peds_have_obstacle_forces=True,
    )
    force = next(f for f in substrate.forces if isinstance(f, ObstacleForce))
    return substrate, force


def test_opt_in_profile_removes_lone_doorway_force_barrier():
    """Upstream repulsion must not balance the released 1.30 m/s² drive."""
    sim, force = _doorway_force("calibrated_v2")
    braking = []
    for x in np.linspace(4.0, 8.0, 81):
        sim.peds.state[0, :2] = [x, 2.0]
        braking.append(-float(force()[0, 0]))
    peak = max(braking)
    assert peak < 1.30, f"1.2 m door has a stand-off barrier: peak {peak:.6f} >= drive 1.30"


def test_default_profile_preserves_legacy_settings_hash():
    """The missing selector must preserve the established pre-profile digest."""
    payload = asdict(SimulationSettings())
    payload.pop("robot_goal_sampling_policy")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    assert hashlib.sha256(encoded.encode()).hexdigest() == (
        "3862ea280966a4e790715babbb7567cbf121031eb38374b08264e4f4d3626be0"
    )


def test_legacy_profile_keeps_released_force_parameters():
    """An explicit legacy selector still constructs factor 10 / offset -0.57."""
    _, force = _doorway_force("legacy_v1")
    assert (force.config.factor, force.config.threshold, force.config.sigma) == (10.0, -0.57, 0.0)


@pytest.mark.parametrize("profile", ["calibrated_v2", "gradient_v3"])
def test_profile_constructor_roundtrip_and_hash_identity(profile):
    """Constructor and serialized settings retain explicit profile selection."""
    settings = SimulationSettings(obstacle_force_profile=profile)
    assert settings.obstacle_force_profile == profile
    assert settings._config_hash_overrides()["obstacle_force_profile"] == profile
    assert SimulationSettings(**settings.to_dict()) == settings
    assert replace(settings) == settings
    assert deepcopy(settings) == settings
    legacy = SimulationSettings()
    assert "obstacle_force_profile" not in replace(legacy).to_dict()
    assert "obstacle_force_profile" not in deepcopy(legacy).to_dict()
    assert _stable_config_hash(EnvSettings(sim_config=settings)) != _stable_config_hash(
        EnvSettings(sim_config=legacy)
    )


@pytest.mark.parametrize("value", ["", "typo_v2", 3, False])
def test_invalid_profile_fails_closed(value):
    """Malformed selectors must never silently restore released parameters."""
    with pytest.raises((ValueError, TypeError), match="obstacle_force_profile"):
        SimulationSettings(obstacle_force_profile=value)


def test_candidate_rejects_a_conflicting_force_law():
    """A different kernel must not receive parameters fitted to the legacy law."""
    with pytest.raises(ValueError, match="calibrated_v2 requires"):
        _doorway_force("calibrated_v2", "surface_distance_unit_normal_v2")


def test_metadata_distinguishes_opt_in_from_missing_selector():
    """Actual substrate records expose the profile without changing legacy payloads."""
    for profile in (None, "legacy_v1", "calibrated_v2", "gradient_v3"):
        substrate, _ = _doorway_force(profile)
        simulator = Simulator.__new__(Simulator)
        simulator.config = SimulationSettings(obstacle_force_profile=profile)
        simulator.pysf_sim = substrate
        record = simulator.obstacle_force_law_metadata()
        if profile is None:
            assert "obstacle_force_profile" not in record
        else:
            assert record["obstacle_force_profile"] == profile


def test_calibration_matches_the_release_pair_kernel():
    """Crowd fitting must use the pair kernel selected by the actual preview matrix."""
    matrix = Path(__file__).resolve().parents[2] / (
        "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    )
    released = yaml.safe_load(matrix.read_text())["scenario_overrides"]["simulation_config"]
    actual = calibration_config(0.003, 0.375).social_force_config.kernel_version
    assert actual == released["social_force_kernel_version"]
    scenario = load_scenarios(matrix)[0]
    footprint = build_env_config(scenario, scenario_path=matrix).sim_config.ped_radius
    assert FOOTPRINT_RADIUS == footprint
    assert calibration_config(0.003, 0.375).scene_config.agent_radius < footprint


def test_gradient_profile_is_the_negative_gradient_of_its_potential():
    """Bind the new profile to real geometry and an independent finite-difference oracle."""
    from shapely.geometry import LineString, Point

    sim, force = _doorway_force("gradient_v3")
    position = np.array([6.7, 2.05])
    sim.peds.state[0, :2] = position
    segments = [LineString([(a, b), (c, d)]) for a, b, c, d in force.get_obstacles()[:, :4]]

    def potential(point):
        return sum(
            0.001 / (2 * (segment.distance(Point(point)) - 0.4) ** 2) for segment in segments
        )

    epsilon = 1e-5
    gradient = np.array(
        [
            (potential(position + epsilon * axis) - potential(position - epsilon * axis))
            / (2 * epsilon)
            for axis in np.eye(2)
        ]
    )
    np.testing.assert_allclose(force()[0], -gradient, rtol=1e-7, atol=1e-9)
    assert force.config.law_version == "surface_distance_unit_normal_v2"


def test_gradient_profile_preserves_explicit_law_conflict_detection():
    """A supplied legacy law cannot silently become a corrected law."""
    with pytest.raises(ValueError, match="gradient_v3 requires"):
        _doorway_force("gradient_v3", "legacy_shifted_gradient_v1")
