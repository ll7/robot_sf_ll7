"""Shared Gymnasium reset metadata helpers."""

from __future__ import annotations

from typing import Any


def resolve_map_id(config: Any, map_def: Any) -> str | None:
    """Resolve the active map id from a map definition, then fall back to config.

    Returns:
        Optional map identifier resolved from the active map or configuration.
    """
    try:
        for map_id, candidate in config.map_pool.map_defs.items():
            if candidate is map_def:
                return map_id
    except (AttributeError, TypeError):
        pass
    return getattr(config, "map_id", None)


def build_reset_metadata(
    config: Any,
    *,
    map_def: Any,
    seed: int | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build stable map/timing reset metadata for Gymnasium environments.

    Returns:
        Reset metadata dictionary with map, timing, seed, and optional extra fields.
    """
    sim_time = float(config.sim_config.sim_time_in_secs)
    time_per_step = float(config.sim_config.time_per_step_in_secs)
    max_sim_steps = getattr(config.sim_config, "max_sim_steps", None)
    if max_sim_steps is None:
        max_sim_steps = int(sim_time / time_per_step)

    metadata: dict[str, Any] = {
        "map_id": resolve_map_id(config, map_def),
        "sim_time_in_secs": sim_time,
        "time_per_step_in_secs": time_per_step,
        "max_sim_steps": int(max_sim_steps),
        "seed": seed,
    }
    if extra:
        metadata.update(extra)
    return metadata
