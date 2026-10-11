"""Runtime physics witnesses for release/campaign manifests.

These are observations of constructed objects, never a reconstruction from defaults.
Null speed parameters mean the active spawn-coupled model has no scalar normal
parameter/global cap. Per-agent caps remain available in the speed-model record.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pysocialforce.forces import ObstacleForce, SocialForce

from robot_sf.benchmark.metrics import CLEARANCE_DEFINITION, TTC_DEFINITION
from robot_sf.ped_npc.ped_robot_force import PedRobotForce

PHYSICS_SCHEMA_VERSION = "effective-physics.v1"
_SCHEMA_PATH = Path(__file__).with_name("schemas") / "effective-physics.v1.json"


def capture_effective_physics(env: Any) -> dict[str, Any]:
    """Snapshot the live simulator, plant, force objects and collision geometry.

    Returns:
        A schema-validated snapshot detached from mutable simulator buffers.
    """
    sim = env.simulator
    peds = sim.pysf_sim.peds
    forces = sim.pysf_sim.forces
    robot = sim.robots[0]
    force_radius = float(peds.agent_radius)
    physical_radius = float(env.state.occupancy.ped_radius)
    metric_radius = float(env.config.sim_config.ped_radius)
    radius_values = {physical_radius, force_radius, metric_radius, float(sim.config.ped_radius)}
    if env.occupancy_grid is not None:
        radius_values.update(env._last_grid_ped_radii.tolist())
    wall = next((force for force in forces if isinstance(force, ObstacleForce)), None)
    social = next((force for force in forces if isinstance(force, SocialForce)), None)
    robot_forces = [force for force in forces if isinstance(force, PedRobotForce)]
    # This is the actual sampler path in _enforce_ped_desired_speeds, which
    # overrides backend sampling. A null mean selects spawn-coupled caps.
    sampling = peds.effective_desired_speed_parameters
    mean = sampling.get("mean_m_s")
    speed_model = {
        "identity": "clipped_normal_v1" if mean is not None else "spawn_coupled_v1",
        "mean_m_s": float(mean) if mean is not None else None,
        "sd_m_s": sampling["sd_m_s"] if mean is not None else None,
        "cap_m_s": sampling["cap_m_s"] if mean is not None else None,
        "cap_rule": "clip_normal_to_[0,high]; velocity_norm<=per_agent_desired_speed"
        if mean is not None
        else "velocity_norm<=max_speed_multiplier*initial_speed",
        "max_speed_multiplier": float(peds.max_speed_multiplier),
        "per_agent_caps_m_s": peds.max_speeds.tolist(),
    }
    robot_force_parameters = [asdict(force.config) for force in robot_forces]
    activation_ranges = [
        float(
            force.config.activation_threshold + force.peds.agent_radius + force.config.robot_radius
        )
        for force in robot_forces
    ]
    # The disabled force still has a resolved configured range, but is not applied.
    inactive_range = float(
        sim.config.prf_config.activation_threshold + force_radius + robot.config.radius
    )
    group_forces = [
        {"identity": type(force).__name__, "parameters": asdict(force.config)}
        for force in forces
        if type(force).__name__.startswith("Group")
    ]
    parameters = {
        "pedestrian_speed_mean_m_s": speed_model["mean_m_s"],
        "pedestrian_speed_sd_m_s": speed_model["sd_m_s"],
        "pedestrian_speed_cap_m_s": speed_model["cap_m_s"],
        "pedestrian_speed_tier": sim.config.ped_speed_tier or False,
        "pedestrian_physical_radius_m": physical_radius,
        "pedestrian_force_radius_m": force_radius,
        "pedestrian_metric_radius_m": metric_radius,
        "pedestrian_radius_convention": "role_specific_radii_v1"
        if len(radius_values) != 1
        else "unified_radius_v1",
        "wall_force_law": wall.law_metadata()
        if wall is not None
        else {"identity": "disabled", "enabled": False},
        "pedestrian_robot_force_enabled": bool(robot_forces),
        "pedestrian_robot_force_activation_range_m": activation_ranges[0]
        if activation_ranges
        else inactive_range,
        "pedestrian_robot_force_law": {
            "identity": "inverse_cubic_center_distance_v1",
            "enabled": bool(robot_forces),
            "activation_rule": "distance<=activation_threshold+ped_force_radius+robot_radius",
            "parameters": robot_force_parameters,
            "activation_ranges_m": activation_ranges,
            "response_multipliers": sim.pedestrian_response_multipliers.tolist()
            if sim.pedestrian_response_multipliers is not None
            else None,
        },
        # No separate pedestrian-robot steering term exists in the active force
        # graph. Heading variants respond to total force; that is recorded separately.
        "pedestrian_robot_steering_enabled": False,
        "robot_reverse_speed_cap_m_s": max(0.0, -float(robot.config.min_linear_speed))
        if hasattr(robot.config, "min_linear_speed")
        else float(robot.config.max_speed),
        "ttc_definition": {
            **TTC_DEFINITION,
            "dt_s": float(env.config.sim_config.time_per_step_in_secs),
        },
    }
    snapshot = {
        "schema_version": PHYSICS_SCHEMA_VERSION,
        "release_design_parameters": parameters,
        "pedestrian_speed_model": speed_model,
        "pedestrian_model": sim.pedestrian_model,
        "pedestrian_contact_law": {
            "identity": "social_force_without_hard_nonpenetration_v1",
            "social_kernel": sim.social_force_kernel_metadata(),
            "enabled": social is not None,
            "parameters": asdict(social.config) if social is not None else {},
            "hard_nonpenetration": False,
        },
        "group_forces": group_forces,
        "radius_roles": {
            "physical_contact_m": physical_radius,
            "force_kernel_m": force_radius,
            "metric_m": metric_radius,
            "placement_m": float(sim.config.ped_radius),
            "occupancy_grid": {
                "enabled": env.occupancy_grid is not None,
                "geometry": "circles",
                "radii_m": env._last_grid_ped_radii.tolist()
                if env.occupancy_grid is not None
                else [],
            },
        },
        "robot_kinematics": type(robot).__name__,
        "integration": {"dt_s": float(peds.d_t), "integrator": peds.integration_scheme},
        "clearance_definition": {
            **CLEARANCE_DEFINITION,
            "robot_radius_m": float(env.config.robot_config.radius),
            "pedestrian_radius_m": metric_radius,
        },
    }
    # Force a detached, finite JSON representation before buffers can be reset.
    snapshot = json.loads(json.dumps(snapshot, allow_nan=False))
    validate_effective_physics(snapshot)
    return snapshot


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    """Build the runtime schema validator once per process.

    Returns:
        The validator for the versioned physics snapshot.
    """
    return Draft202012Validator(json.loads(_SCHEMA_PATH.read_text(encoding="utf-8")))


def validate_effective_physics(snapshot: dict[str, Any], *, env: Any = None) -> None:
    """Reject incomplete/invalid snapshots and, when supplied, disagreement with live objects."""
    _validator().validate(snapshot)
    json.dumps(snapshot, allow_nan=False)
    if env is not None and snapshot != capture_effective_physics(env):
        raise ValueError("effective physics differs from live simulator")


def campaign_physics(campaign_root: Path) -> dict[str, Any]:
    """Collect episode witnesses without hiding variation or missing coverage.

    Returns:
        v1 without physics when no witness exists, otherwise v2 with retained
        distinct snapshots and explicit missing markers. Partial coverage never
        establishes a campaign-wide design parameter.
    """
    latest_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted((campaign_root / "runs").rglob("episodes.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample = _episode_physics_sample(row)
                _retain_sample(latest_by_identity, sample)
    count = len(latest_by_identity)
    witnessed = 0
    common: dict[str, Any] | None = None
    samples: list[dict[str, Any]] = []
    signatures: set[tuple[str, str, str]] = set()
    for sample in latest_by_identity.values():
        if sample["physics_witness"] == "missing":
            samples.append(sample)
            continue
        witnessed += 1
        snapshot = sample["physics"]
        parameters = snapshot["release_design_parameters"]
        if common is None:
            common = {key: value for key, value in parameters.items() if value is not None}
        else:
            common = {key: value for key, value in common.items() if parameters[key] == value}
        signature = (
            sample["scenario_id"],
            sample["config_hash"],
            json.dumps(snapshot, sort_keys=True, allow_nan=False),
        )
        if signature not in signatures:
            samples.append(sample)
            signatures.add(signature)
    if not witnessed:
        return {"schema_version": "benchmark-camera-ready-campaign.v1"}
    missing = count - witnessed
    return {
        "schema_version": "benchmark-camera-ready-campaign.v2",
        "effective_physics_schema_version": PHYSICS_SCHEMA_VERSION,
        "effective_physics_episode_count": count,
        "effective_physics_witnessed_episode_count": witnessed,
        "effective_physics_missing_episode_count": missing,
        "effective_physics_status": "incomplete" if missing else "complete",
        "effective_physics_samples": samples,
        "release_design_parameters": {} if missing else common or {},
    }


def _episode_physics_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Validate a row and preserve its source identity even without physics.

    Returns:
        A recorded witness or a missing marker carrying no invented parameters.
    """
    sample = {key: row[key] for key in ("scenario_id", "config_hash", "episode_id", "seed")}
    _sample_identity(sample)
    snapshot = row.get("effective_physics")
    if snapshot is None:
        return {**sample, "physics_witness": "missing"}
    validate_effective_physics(snapshot)
    if row.get("release_design_parameters") != snapshot["release_design_parameters"]:
        raise ValueError("episode design parameters contradict runtime witness")
    return {**sample, "physics_witness": "recorded", "physics": snapshot}


def _retain_sample(
    samples: dict[tuple[str, str, str], dict[str, Any]], sample: dict[str, Any]
) -> None:
    """Upgrade missing witnesses on retry, rejecting conflicting recorded physics."""
    identity = _sample_identity(sample)
    existing = samples.get(identity)
    if existing is None or existing == sample:
        samples[identity] = sample
        return
    if existing["seed"] != sample["seed"]:
        raise ValueError("conflicting physics for episode identity")
    existing_witness = existing.get("physics_witness", "recorded")
    new_witness = sample.get("physics_witness", "recorded")
    if existing_witness == "missing" and new_witness == "recorded":
        samples[identity] = sample
        return
    if existing_witness == "recorded" and new_witness == "missing":
        return
    raise ValueError("conflicting physics for episode identity")


def validate_campaign_physics(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate physics structure and report incomplete coverage without blocking resume.

    Returns:
        A complete/incomplete report with the missing source markers, or an
        unavailable report for a legacy v1 manifest with no physics block.
        Malformed witnesses and contradictory claims still raise ValueError.
    """
    if manifest.get("schema_version") == "benchmark-camera-ready-campaign.v1":
        if "release_design_parameters" in manifest or any(
            key.startswith("effective_physics") for key in manifest
        ):
            raise ValueError("legacy campaign cannot declare runtime physics")
        return {"status": "unavailable", "missing": []}
    if manifest.get("schema_version") != "benchmark-camera-ready-campaign.v2":
        raise ValueError("runtime physics requires campaign manifest v2")
    if manifest.get("effective_physics_schema_version") != PHYSICS_SCHEMA_VERSION:
        raise ValueError("unsupported effective physics schema version")
    samples = manifest.get("effective_physics_samples")
    count = manifest.get("effective_physics_episode_count")
    if not isinstance(samples, list) or type(count) is not int or count < len(samples):
        raise ValueError("invalid effective physics coverage")
    if bool(count) != bool(samples):
        raise ValueError("effective physics coverage has no runtime samples")
    common = manifest.get("release_design_parameters")
    if not isinstance(common, dict):
        raise ValueError("campaign lacks design parameter mapping")
    missing = _validate_samples(samples, common)
    status = "incomplete" if missing else "complete"
    _validate_coverage_counts(manifest, count, len(missing), status)
    return {"status": status, "missing": missing}


def _validate_samples(
    samples: list[dict[str, Any]], common: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate source identities, missing markers and agreement with global claims.

    Returns:
        The explicit missing-witness markers.
    """
    missing = []
    identities = set()
    for sample in samples:
        identity = _sample_identity(sample)
        if identity in identities:
            raise ValueError("duplicate episode physics sample")
        identities.add(identity)
        if sample.get("physics_witness", "recorded") == "missing":
            if "physics" in sample:
                raise ValueError("missing physics marker cannot contain a witness")
            if common:
                raise ValueError("global design parameters cannot claim incomplete coverage")
            missing.append(sample)
            continue
        if sample.get("physics_witness", "recorded") != "recorded":
            raise ValueError("unsupported physics_witness marker")
        validate_effective_physics(sample["physics"])
        parameters = sample["physics"]["release_design_parameters"]
        if any(
            key not in parameters or value is None or parameters[key] != value
            for key, value in common.items()
        ):
            raise ValueError("global design parameters contradict runtime sample")
    return missing


def _validate_coverage_counts(
    manifest: dict[str, Any], count: int, missing: int, status: str
) -> None:
    """Check optional coverage counters while preserving previously emitted v2 snapshots."""
    for key, expected in (
        ("effective_physics_missing_episode_count", missing),
        ("effective_physics_witnessed_episode_count", count - missing),
        ("effective_physics_status", status),
    ):
        if key in manifest and manifest[key] != expected:
            raise ValueError(f"inconsistent {key}")


def _sample_identity(sample: dict[str, Any]) -> tuple[str, str, str]:
    """Validate a runtime sample's source identity.

    Returns:
        The scenario/config/episode identity for duplicate detection.
    """
    for key in ("scenario_id", "config_hash", "episode_id"):
        if not isinstance(sample.get(key), str) or not sample[key].strip():
            raise ValueError(f"runtime sample lacks {key}")
    if type(sample.get("seed")) is not int:
        raise ValueError("runtime sample lacks integer seed")
    return sample["scenario_id"], sample["config_hash"], sample["episode_id"]
