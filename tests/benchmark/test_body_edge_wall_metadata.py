"""Native wall-force metadata must survive the episode physics schema."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError
from pysocialforce import Simulator
from pysocialforce.config import (
    BODY_EDGE_EXPONENTIAL_V3,
    BODY_EDGE_EXPONENTIAL_V3_MULTI_SEGMENT,
    LEGACY_SHIFTED_GRADIENT_V1,
    ObstacleForceConfig,
)
from pysocialforce.forces import ObstacleForce

from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("law", [BODY_EDGE_EXPONENTIAL_V3, BODY_EDGE_EXPONENTIAL_V3_MULTI_SEGMENT])
def test_wall_corner_does_not_double_the_same_surface_response(law):
    """Two edges sharing the nearest corner must not double its pedestrian force."""
    edges = np.array([[0.0, -1.0, 0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    simulation = SimpleNamespace(
        peds=SimpleNamespace(agent_radius=0.35, pos=lambda: np.array([[0.3, 0.3]])),
        get_raw_obstacles=lambda: edges[:1],
    )
    config = ObstacleForceConfig(law_version=law)
    single_surface = ObstacleForce(config, simulation)()[0]
    assert np.linalg.norm(single_surface) > 0
    simulation.get_raw_obstacles = lambda: edges
    corner = ObstacleForce(config, simulation)()[0]
    np.testing.assert_allclose(corner, single_surface, rtol=1e-12, atol=1e-12)


def test_multi_segment_wrapper_matches_live_force_and_metadata():
    """Point and batch wrapper queries must retain the same two-wall response."""
    simulation = Simulator(
        np.array([[0.0, 0.0, 0.0, 0.0, 2.0, 0.0]]),
        obstacles=[(0.4, 0.4, -1.0, 1.0), (-1.0, 1.0, -0.5, -0.5)],
    )
    simulation.config.obstacle_force_config.law_version = BODY_EDGE_EXPONENTIAL_V3_MULTI_SEGMENT
    wrapper = FastPysfWrapper(simulation)
    point = np.array([0.0, 0.0])
    native_force = ObstacleForce(simulation.config.obstacle_force_config, simulation)
    expected = native_force()[0]
    assert expected[0] < 0 and expected[1] > 0
    np.testing.assert_allclose(wrapper._compute_obstacle_force_at_point(point), expected)
    np.testing.assert_allclose(
        wrapper._compute_obstacle_forces_at_points(point.reshape(1, 2))[0], expected
    )
    metadata = wrapper.obstacle_force_law_metadata()
    _wall_schema().validate(metadata)
    assert metadata["geometry_convention"] == "all_nearby_distinct_surface_points"
    assert metadata == native_force.law_metadata() | {"site": "fast_pysf_wrapper"}


def _wall_schema():
    """Return the production schema for a live wall-law witness."""
    schema = json.loads((ROOT / "robot_sf/benchmark/schemas/effective-physics.v1.json").read_text())
    return Draft202012Validator(
        schema["properties"]["release_design_parameters"]["properties"]["wall_force_law"]
    )


def _metadata(config):
    """Capture metadata from the actual force provider without stepping a scene."""
    simulation = SimpleNamespace(
        peds=SimpleNamespace(agent_radius=0.35, pos=lambda: []), get_raw_obstacles=lambda: []
    )
    return ObstacleForce(config, simulation).law_metadata()


def test_default_wall_law_metadata_is_schema_valid():
    """The default law must not make every native episode unwritable."""
    _wall_schema().validate(_metadata(ObstacleForceConfig()))


@pytest.mark.parametrize("factor", [0.5, 10.0])
def test_live_wrapper_wall_metadata_validates_against_production_schema(factor):
    """The wrapper's live witness must include every parameter of the applied law."""
    simulation = Simulator(np.array([[0.0, 0.4, 0.0, 0.0, 2.0, 0.4]]))
    simulation.config.obstacle_force_config.factor = factor
    wrapper = FastPysfWrapper(simulation)
    metadata = wrapper.obstacle_force_law_metadata()

    _wall_schema().validate(metadata)
    assert (
        metadata["parameters"]
        == ObstacleForce(simulation.config.obstacle_force_config, simulation).law_metadata()[
            "parameters"
        ]
    )
    assert metadata["parameters"]["amplitude_m_s2"] == pytest.approx(0.3 * factor)


@pytest.mark.parametrize("parameter", ["amplitude_m_s2", "decay_m", "range_m"])
def test_body_edge_wall_schema_rejects_incomplete_active_parameters(parameter):
    """Accepting the new law must retain complete parameter provenance."""
    metadata = _metadata(ObstacleForceConfig())
    validator = _wall_schema()
    validator.validate(metadata)
    metadata["parameters"].pop(parameter)
    with pytest.raises(ValidationError):
        validator.validate(metadata)


def test_legacy_wall_schema_still_requires_its_active_offset():
    """The additive schema branch must not loosen the historical-law contract."""
    metadata = _metadata(ObstacleForceConfig(law_version=LEGACY_SHIFTED_GRADIENT_V1))
    _wall_schema().validate(metadata)
    metadata["parameters"].pop("effective_offset")
    with pytest.raises(ValidationError):
        _wall_schema().validate(metadata)
