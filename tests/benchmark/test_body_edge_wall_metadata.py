"""Native wall-force metadata must survive the episode physics schema."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator, ValidationError
from pysocialforce.config import LEGACY_SHIFTED_GRADIENT_V1, ObstacleForceConfig
from pysocialforce.forces import ObstacleForce

ROOT = Path(__file__).resolve().parents[2]


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


@pytest.mark.parametrize("parameter", ["amplitude_m_s2", "decay_m", "range_m"])
def test_body_edge_wall_schema_rejects_incomplete_active_parameters(parameter):
    """Accepting the new law must retain complete parameter provenance."""
    metadata = _metadata(ObstacleForceConfig())
    metadata["parameters"].pop(parameter)
    with pytest.raises(ValidationError):
        _wall_schema().validate(metadata)


def test_legacy_wall_schema_still_requires_its_active_offset():
    """The additive schema branch must not loosen the historical-law contract."""
    metadata = _metadata(ObstacleForceConfig(law_version=LEGACY_SHIFTED_GRADIENT_V1))
    _wall_schema().validate(metadata)
    metadata["parameters"].pop("effective_offset")
    with pytest.raises(ValidationError):
        _wall_schema().validate(metadata)
