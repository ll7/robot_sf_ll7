"""Materializers for adversarial configs that feed existing scenario surfaces."""

from __future__ import annotations

from typing import Any

from robot_sf.adversarial.config import MultiPedAdversarialConfig, MultiPedCandidateSpec


def _point_payload(pose: Any) -> list[float]:
    """Return a scenario-loader point payload from a pose-like object."""

    return [float(pose.x), float(pose.y)]


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
                "start": _point_payload(pedestrian.start),
                "goal": _point_payload(pedestrian.goal),
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
