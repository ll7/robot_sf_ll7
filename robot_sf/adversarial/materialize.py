"""Materializers for adversarial configs that feed existing scenario surfaces."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from robot_sf.adversarial.config import MultiPedAdversarialConfig, MultiPedCandidateSpec


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
