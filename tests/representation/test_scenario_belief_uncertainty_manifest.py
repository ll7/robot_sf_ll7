"""Tests for the issue #2478 ScenarioBelief uncertainty contract."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from robot_sf.gym_env.unified_config import ObservationVisibilitySettings, RobotSimulationConfig
from robot_sf.representation import EntityBelief, Estimate2D, ScenarioBelief
from robot_sf.representation.scenario_belief import (
    scenario_belief_from_simulator_oracle,
    scenario_belief_from_visibility_limited_simulator,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs/representation/scenario_belief_uncertainty_issue_2478.yaml"


def _manifest() -> dict[str, object]:
    """Load the ScenarioBelief uncertainty contract manifest."""
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _simulator_fixture() -> SimpleNamespace:
    """Return a deterministic simulator-like step for uncertainty contract tests."""
    return SimpleNamespace(
        ped_pos=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ped_vel=np.array([[0.5, 0.0], [0.0, -0.25]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.1, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=0.4),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=8.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )


def test_uncertainty_manifest_references_existing_surfaces() -> None:
    """The #2478 manifest should point only at tracked repository surfaces."""
    manifest = _manifest()
    surfaces = manifest["contract_surfaces"]
    assert isinstance(surfaces, dict)

    for key, rel_path_or_paths in surfaces.items():
        if key == "tests":
            assert isinstance(rel_path_or_paths, list)
            for rel_path in rel_path_or_paths:
                assert (REPO_ROOT / str(rel_path)).exists(), rel_path
            continue
        assert (REPO_ROOT / str(rel_path_or_paths)).exists(), rel_path_or_paths


def test_uncertainty_manifest_fields_match_dataclass_contracts() -> None:
    """Manifest field names should resolve against the implemented dataclasses."""
    manifest = _manifest()
    uncertainty_fields = manifest["uncertainty_fields"]
    assert isinstance(uncertainty_fields, dict)

    estimate_fields = {field.name for field in fields(Estimate2D)}
    entity_fields = {field.name for field in fields(EntityBelief)}

    object_class = uncertainty_fields["object_class"]
    pose = uncertainty_fields["pose"]
    velocity = uncertainty_fields["velocity"]
    scalar_confidence = uncertainty_fields["scalar_confidence"]
    existence_probability = uncertainty_fields["existence_probability"]
    assert isinstance(object_class, dict)
    assert isinstance(pose, dict)
    assert isinstance(velocity, dict)
    assert isinstance(scalar_confidence, dict)
    assert isinstance(existence_probability, dict)

    assert object_class["primary_field"] in entity_fields
    assert object_class["distribution_field"] in entity_fields
    assert pose["owning_entity_field"] in entity_fields
    assert velocity["owning_entity_field"] in entity_fields
    assert pose["mean_field"] in estimate_fields
    assert pose["covariance_field"] in estimate_fields
    assert pose["frame_field"] in estimate_fields
    assert velocity["mean_field"] in estimate_fields
    assert velocity["covariance_field"] in estimate_fields
    assert velocity["frame_field"] in estimate_fields
    assert scalar_confidence["field"] in estimate_fields
    assert existence_probability["field"] in entity_fields


def test_adapter_debug_output_exposes_uncertainty_units_and_class_probabilities() -> None:
    """Real adapter fixtures should expose class, frame, unit, and covariance semantics."""
    simulator = _simulator_fixture()
    oracle = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=RobotSimulationConfig(),
        max_pedestrians=4,
    )

    debug = oracle.to_debug_dict()
    first_agent = debug["agents"][0]
    assert first_agent["class_probabilities"] == {"pedestrian": 0.98}
    assert first_agent["position"]["frame_id"] == "map"
    assert first_agent["position"]["units"] == "m"
    assert first_agent["position"]["covariance_units"] == "m^2"
    assert first_agent["velocity"]["frame_id"] == "map"
    assert first_agent["velocity"]["units"] == "m/s"
    assert first_agent["velocity"]["covariance_units"] == "(m/s)^2"
    assert debug["ego"]["class_probabilities"] == {"ego_robot": 1.0}


def test_visibility_limited_adapter_increases_covariance_without_projection_key_drift() -> None:
    """Partial observations should alter uncertainty metadata, not policy key layout."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=90.0,
    )
    simulator = _simulator_fixture()

    oracle = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )
    partial = scenario_belief_from_visibility_limited_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    oracle_agent = oracle.agents[1]
    partial_agent = partial.agents[1]
    assert partial_agent.position.confidence < oracle_agent.position.confidence
    assert partial_agent.class_probabilities == (("pedestrian", partial_agent.position.confidence),)
    assert partial_agent.position.covariance_xy[0][0] > oracle_agent.position.covariance_xy[0][0]
    assert partial_agent.velocity.covariance_xy[0][0] > oracle_agent.velocity.covariance_xy[0][0]
    assert partial_agent.position.units == "m"
    assert partial_agent.velocity.units == "m/s"
    assert set(partial.to_socnav_struct()) == set(oracle.to_socnav_struct())


def test_uncertainty_manifest_keeps_non_benchmark_claim_boundary() -> None:
    """The #2478 contract should not be mistaken for benchmark evidence."""
    manifest = _manifest()
    claim_boundary = str(manifest["claim_boundary"]).lower()
    known_gaps = manifest["known_gaps"]
    assert isinstance(known_gaps, list)

    assert manifest["benchmark_evidence"] is False
    assert "does not prove" in claim_boundary
    assert any("heading uncertainty" in str(gap) for gap in known_gaps)
    assert hasattr(ScenarioBelief, "to_socnav_struct")
