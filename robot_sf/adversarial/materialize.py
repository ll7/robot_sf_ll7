"""Materializers for adversarial configs that feed existing scenario surfaces."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from robot_sf.adversarial.config import MultiPedAdversarialConfig, MultiPedCandidateSpec
from robot_sf.adversarial.scenario_manifest import (
    AdversarialScenarioManifest,
    ManifestCategory,
)


def _metadata_payload(
    config: MultiPedAdversarialConfig,
    pedestrian: MultiPedCandidateSpec,
) -> dict[str, Any]:
    """Return YAML-safe adversarial metadata for one materialized pedestrian."""

    return {
        "adversarial_family": config.family,
        "adversarial_schema_version": config.schema_version,
        "adversarial_scenario_seed": int(config.scenario_seed),
        "pedestrian_metadata": dict(pedestrian.metadata),
        "spawn_time_s": float(pedestrian.spawn_time_s),
        "delay_s": float(pedestrian.delay_s),
    }


def _runtime_status_payload(config: MultiPedAdversarialConfig) -> dict[str, Any]:
    """Return certification-boundary metadata for development adversarial smoke cases."""

    return {
        "schema_version": config.schema_version,
        "family": config.family,
        "scenario_seed": int(config.scenario_seed),
        "pedestrian_ids": [pedestrian.id for pedestrian in config.pedestrians],
        "evaluation_scope": "development_stress_test",
        "certification_status": "uncertified_development_smoke",
        "benchmark_frozen": False,
    }


def materialize_multi_ped_single_pedestrian_overrides(
    config: MultiPedAdversarialConfig,
) -> list[dict[str, Any]]:
    """Convert a multi-ped adversarial config into scenario-loader overrides.

    The returned dictionaries are designed for the existing ``single_pedestrians`` scenario
    override surface. This remains a pure-data bridge; it does not load maps or run environments.

    Returns:
        YAML/JSON-safe single-pedestrian override dictionaries.
    """

    overrides: list[dict[str, Any]] = []
    for pedestrian in config.pedestrians:
        overrides.append(
            {
                "id": pedestrian.id,
                "start": pedestrian.start.as_waypoint(),
                "goal": pedestrian.goal.as_waypoint(),
                "speed_m_s": float(pedestrian.speed_mps),
                "start_delay_s": float(pedestrian.spawn_time_s) + float(pedestrian.delay_s),
                "note": (
                    f"{config.schema_version} {config.family} "
                    f"seed={int(config.scenario_seed)} ped={pedestrian.id}"
                ),
                "metadata": _metadata_payload(config, pedestrian),
            }
        )
    return overrides


def materialize_multi_ped_scenario_payload(
    config: MultiPedAdversarialConfig,
    scenario_template: dict[str, Any],
) -> dict[str, Any]:
    """Return a scenario-loader manifest payload for a multi-ped adversarial config.

    The helper merges generated ``single_pedestrians`` entries into the first scenario in a template
    payload. It is intentionally pure-data: no map loading, environment reset, or certification is
    performed here.

    Returns:
        YAML/JSON-safe scenario manifest with one materialized scenario.
    """

    scenarios = scenario_template.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios or not isinstance(scenarios[0], dict):
        raise ValueError("scenario_template must contain a non-empty scenarios list")

    scenario = deepcopy(scenarios[0])
    base_name = str(scenario.get("name") or scenario.get("scenario_id") or "scenario")
    scenario["name"] = f"{base_name}_multi_ped_adversarial_{int(config.scenario_seed):04d}"
    scenario["seeds"] = [int(config.scenario_seed)]

    sim_config = dict(scenario.get("simulation_config") or {})
    sim_config["route_spawn_seed"] = int(config.scenario_seed)
    scenario["simulation_config"] = sim_config

    metadata = dict(scenario.get("metadata") or {})
    metadata["adversarial_multi_ped"] = config.to_json()
    metadata["adversarial_multi_ped_runtime"] = _runtime_status_payload(config)
    scenario["metadata"] = metadata

    scenario["single_pedestrians"] = _merge_single_pedestrian_entries(
        list(scenario.get("single_pedestrians") or []),
        materialize_multi_ped_single_pedestrian_overrides(config),
    )
    return {"scenarios": [scenario]}


def materialize_manifest_single_pedestrian_override(
    manifest: AdversarialScenarioManifest,
    *,
    pedestrian_id: str | None = None,
) -> dict[str, Any]:
    """Convert a valid single-ped manifest into a scenario-loader override.

    Returns:
        YAML/JSON-safe ``single_pedestrians`` override dictionary.
    """

    _require_valid_manifest(manifest)
    controls = manifest.candidate_controls or {}
    candidate_index = _manifest_candidate_index(manifest)
    ped_id = pedestrian_id or f"manifest_candidate_{candidate_index:04d}"
    spawn_time_s = float(controls["spawn_time_s"])
    delay_s = float(controls["pedestrian_delay_s"])
    scenario_seed = int(controls["scenario_seed"])

    return {
        "id": ped_id,
        "start": _pose_waypoint(controls, "start"),
        "goal": _pose_waypoint(controls, "goal"),
        "speed_m_s": float(controls["pedestrian_speed_mps"]),
        "start_delay_s": spawn_time_s + delay_s,
        "note": (f"{manifest.schema_version} candidate={candidate_index:04d} seed={scenario_seed}"),
        "metadata": {
            "adversarial_schema_version": manifest.schema_version,
            "adversarial_candidate_index": candidate_index,
            "adversarial_scenario_seed": scenario_seed,
            "normalized_control_hash": _manifest_control_hash(manifest),
            "spawn_time_s": spawn_time_s,
            "delay_s": delay_s,
            "pedestrian_speed_mps": float(controls["pedestrian_speed_mps"]),
            "validation_status": manifest.validation.status.value
            if manifest.validation is not None
            else "unknown",
        },
    }


def materialize_manifest_scenario_payload(
    manifest: AdversarialScenarioManifest,
    scenario_template: dict[str, Any],
    *,
    route_file_name: str | None = None,
) -> dict[str, Any]:
    """Return a scenario-loader manifest for one valid adversarial manifest.

    The helper materializes one generated route candidate into the first scenario of a template. It
    is pure data transformation: no map loading, planner invocation, or benchmark certification
    occurs. When ``route_file_name`` is provided, the caller is responsible for writing a matching
    route-overrides YAML file next to the scenario matrix.

    Returns:
        YAML/JSON-safe scenario manifest with one materialized scenario.
    """

    _require_valid_manifest(manifest)
    scenarios = scenario_template.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios or not isinstance(scenarios[0], dict):
        raise ValueError("scenario_template must contain a non-empty scenarios list")

    controls = manifest.candidate_controls or {}
    scenario_seed = int(controls["scenario_seed"])
    candidate_index = _manifest_candidate_index(manifest)
    scenario = deepcopy(scenarios[0])
    base_name = str(scenario.get("name") or scenario.get("scenario_id") or "scenario")
    scenario["name"] = f"{base_name}_manifest_{candidate_index:04d}"
    scenario["seeds"] = [scenario_seed]
    if route_file_name is not None:
        scenario["route_overrides_file"] = str(route_file_name)

    sim_config = dict(scenario.get("simulation_config") or {})
    sim_config["route_spawn_seed"] = scenario_seed
    sim_config["peds_speed_mult"] = float(controls["pedestrian_speed_mps"])
    scenario["simulation_config"] = sim_config

    metadata = dict(scenario.get("metadata") or {})
    metadata["adversarial_scenario_manifest"] = manifest.to_dict()
    metadata["adversarial_manifest_runtime"] = {
        "schema_version": manifest.schema_version,
        "candidate_index": candidate_index,
        "normalized_control_hash": _manifest_control_hash(manifest),
        "evaluation_scope": "development_stress_test",
        "certification_status": "uncertified_smoke",
        "benchmark_frozen": False,
    }
    scenario["metadata"] = metadata

    if route_file_name is None:
        scenario["single_pedestrians"] = _merge_single_pedestrian_entries(
            list(scenario.get("single_pedestrians") or []),
            [materialize_manifest_single_pedestrian_override(manifest)],
        )
    return {"scenarios": [scenario]}


def materialize_manifest_route_overrides(
    manifest: AdversarialScenarioManifest,
    *,
    route_id: int | None = None,
) -> dict[str, Any]:
    """Return a route-overrides payload for a valid manifest's route candidate.

    Returns:
        YAML/JSON-safe route-overrides payload.
    """

    _require_valid_manifest(manifest)
    controls = manifest.candidate_controls or {}
    candidate_index = _manifest_candidate_index(manifest)
    resolved_route_id = int(route_id) if route_id is not None else 100_000 + candidate_index
    return {
        "robot_routes": [
            {
                "spawn_id": resolved_route_id,
                "goal_id": resolved_route_id,
                "waypoints": [
                    _pose_waypoint(controls, "start"),
                    _pose_waypoint(controls, "goal"),
                ],
            }
        ],
        "ped_routes": [],
    }


def _merge_single_pedestrian_entries(
    existing: list[Any],
    overrides: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge override entries into existing template entries by pedestrian id."""

    merged: list[dict[str, Any]] = [
        dict(entry) for entry in existing if isinstance(entry, dict) and entry.get("id") is not None
    ]
    index_by_id = {str(entry["id"]): index for index, entry in enumerate(merged)}
    for override in overrides:
        ped_id = str(override["id"])
        if ped_id in index_by_id:
            updated = dict(merged[index_by_id[ped_id]])
            updated.update(override)
            merged[index_by_id[ped_id]] = updated
            continue
        index_by_id[ped_id] = len(merged)
        merged.append(dict(override))
    return merged


def _require_valid_manifest(manifest: AdversarialScenarioManifest) -> None:
    """Raise when a manifest cannot be safely materialized into a runnable scenario."""

    validation = manifest.validation
    if validation is None:
        raise ValueError("manifest validation record is required")
    if validation.status is not ManifestCategory.VALID:
        raise ValueError(f"only valid manifests can be materialized: {validation.status.value}")
    controls = manifest.candidate_controls
    if not isinstance(controls, dict):
        raise ValueError("manifest candidate_controls must be a mapping")
    for key in (
        "start",
        "goal",
        "spawn_time_s",
        "pedestrian_speed_mps",
        "pedestrian_delay_s",
        "scenario_seed",
    ):
        if key not in controls:
            raise ValueError(f"manifest candidate_controls.{key} is required")


def _pose_waypoint(controls: dict[str, Any], name: str) -> list[float]:
    """Return one serialized pose as a two-coordinate waypoint."""

    pose = controls.get(name)
    if not isinstance(pose, dict):
        raise ValueError(f"manifest candidate_controls.{name} must be a mapping")
    return [float(pose["x"]), float(pose["y"])]


def _manifest_candidate_index(manifest: AdversarialScenarioManifest) -> int:
    """Return the generator candidate index, defaulting to zero for legacy payloads."""

    if manifest.generator is None:
        return 0
    return int(manifest.generator.candidate_index)


def _manifest_control_hash(manifest: AdversarialScenarioManifest) -> str | None:
    """Return the validation control hash when available."""

    if manifest.validation is None:
        return None
    return manifest.validation.normalized_control_hash
