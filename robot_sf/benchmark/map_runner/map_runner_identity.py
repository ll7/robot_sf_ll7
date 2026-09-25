"""Seed and resume identity helpers for map-runner benchmark batches."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

import yaml

from robot_sf.benchmark.observation_noise import (
    normalize_observation_noise_spec,
    observation_noise_hash,
)
from robot_sf.benchmark.safety.cbf_safety_filter_runtime import (
    runtime_config_from_mapping as cbf_runtime_config_from_mapping,
)
from robot_sf.benchmark.safety.safety_wrapper_runtime import runtime_config_from_mapping
from robot_sf.benchmark.tracking_precision_contract import (
    normalize_tracking_precision_spec,
    tracking_precision_hash,
)
from robot_sf.benchmark.utils import _config_hash

_MAP_RUNNER_SCENARIO_IDENTITY_FIELDS = frozenset(
    {
        "algo",
        "algo_config_hash",
        "record_forces",
        "observation_mode",
        "observation_level",
        "benchmark_track",
        "track_schema_version",
        "observation_noise_profile",
        "observation_noise_hash",
        "tracking_precision",
        "tracking_precision_hash",
        "synthetic_actuation_profile",
        "latency_stress_profile",
        "safety_wrapper",
        "cbf_safety_filter",
        "record_planner_decision_trace",
        "record_simulation_step_trace",
        "run_horizon",
        "run_dt",
    }
)


def _resolve_seed_list(path: Path) -> dict[str, list[int]]:
    """Load named benchmark seed lists from YAML.

    Returns:
        dict[str, list[int]]: Seed lists keyed by suite name.
    """
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return {str(k): [int(s) for s in v] for k, v in data.items() if isinstance(v, list)}


def _suite_key(scenario_path: Path) -> str:
    """Infer the seed-suite key from a scenario config filename.

    Returns:
        str: Suite key used for seed-list lookup.
    """
    stem = scenario_path.stem.lower()
    if "classic" in stem:
        return "classic_interactions"
    if "francis" in stem:
        return "francis2023"
    return "default"


def _select_seeds(
    scenario: dict[str, Any],
    *,
    suite_seeds: dict[str, list[int]],
    suite_key: str,
) -> list[int]:
    """Resolve per-scenario seeds with suite and default fallbacks.

    Returns:
        list[int]: Seeds to run for the scenario.
    """
    seeds = scenario.get("seeds")
    if isinstance(seeds, list) and seeds:
        return [int(s) for s in seeds]
    if suite_seeds.get(suite_key):
        return list(suite_seeds[suite_key])
    if suite_seeds.get("default"):
        return list(suite_seeds["default"])
    return [0]


def _scenario_identity_payload(  # noqa: C901,PLR0913
    scenario: dict[str, Any],
    *,
    algo: str,
    algo_config: dict[str, Any],
    horizon: int | None,
    dt: float | None,
    record_forces: bool,
    observation_mode: str | None = None,
    observation_level: str | None = None,
    benchmark_track: str | None = None,
    track_schema_version: str | None = None,
    observation_noise: dict[str, Any] | None = None,
    tracking_precision: dict[str, Any] | None = None,
    synthetic_actuation_profile: dict[str, Any] | None = None,
    latency_stress_profile: dict[str, Any] | None = None,
    safety_wrapper: dict[str, Any] | None = None,
    cbf_safety_filter: dict[str, Any] | None = None,
    record_planner_decision_trace: bool = False,
    record_simulation_step_trace: bool = False,
) -> dict[str, Any]:
    """Build the canonical scenario payload used for episode identity.

    Resume safety relies on using the same identity dimensions at write-time and
    skip-time. For map runs this includes algorithm and run-shaping options.

    Returns:
        dict[str, Any]: Identity payload consumed by ``compute_map_episode_id``.
    """
    payload = {key: value for key, value in scenario.items() if key not in {"seed", "seeds"}}
    scenario_id = (
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )
    payload.setdefault("id", scenario_id)
    payload["algo"] = str(algo)
    payload["algo_config_hash"] = _config_hash(algo_config)
    payload["record_forces"] = bool(record_forces)
    if observation_mode is not None:
        payload["observation_mode"] = str(observation_mode)
    if observation_level is not None:
        payload["observation_level"] = str(observation_level)
    if benchmark_track is not None:
        payload["benchmark_track"] = str(benchmark_track)
    if track_schema_version is not None:
        payload["track_schema_version"] = str(track_schema_version)
    noise_spec = normalize_observation_noise_spec(observation_noise)
    if bool(noise_spec["enabled"]):
        payload["observation_noise_profile"] = str(noise_spec["profile"])
        payload["observation_noise_hash"] = observation_noise_hash(noise_spec)
    tracking_precision_spec = normalize_tracking_precision_spec(tracking_precision)
    if bool(tracking_precision_spec["enabled"]):
        payload["tracking_precision"] = tracking_precision_spec
        payload["tracking_precision_hash"] = tracking_precision_hash(tracking_precision_spec)
    if synthetic_actuation_profile is not None:
        payload["synthetic_actuation_profile"] = dict(synthetic_actuation_profile)
    if latency_stress_profile is not None:
        payload["latency_stress_profile"] = dict(latency_stress_profile)
    if safety_wrapper is not None:
        resolved_safety_wrapper = runtime_config_from_mapping(safety_wrapper)
        if resolved_safety_wrapper.enabled:
            payload["safety_wrapper"] = asdict(resolved_safety_wrapper)
    if cbf_safety_filter is not None:
        resolved_cbf_filter = cbf_runtime_config_from_mapping(cbf_safety_filter)
        if resolved_cbf_filter.enabled:
            payload["cbf_safety_filter"] = asdict(resolved_cbf_filter)
    payload["record_planner_decision_trace"] = bool(record_planner_decision_trace)
    payload["record_simulation_step_trace"] = bool(record_simulation_step_trace)
    if horizon is not None and int(horizon) > 0:
        payload["run_horizon"] = int(horizon)
    if dt is not None and float(dt) > 0.0:
        payload["run_dt"] = float(dt)
    return payload


def _compute_map_episode_id(identity_payload: dict[str, Any], seed: int) -> str:
    """Return a map-runner episode id scoped to algorithm + run dimensions.

    The default benchmark ``compute_episode_id`` uses ``<scenario_id>--<seed>``.
    Map-batch resume needs richer scoping for mixed algorithm/config runs.
    """
    scenario_id = (
        identity_payload.get("id")
        or identity_payload.get("name")
        or identity_payload.get("scenario_id")
        or "unknown"
    )
    identity_hash = _config_hash(identity_payload)
    return f"{scenario_id}--{seed}--{identity_hash}"


def _scenario_with_episode_seed_defaults(
    scenario: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Return a scenario copy with seed-derived defaults for stochastic subcomponents.

    Some scenario-level generators use their own NumPy ``default_rng`` instances. When those
    fields are left unset they bypass the episode seed and make benchmark rows depend on process
    history. Fill only missing values here so explicit scenario provenance remains unchanged.
    """
    updated = deepcopy(scenario)
    sim_config = updated.setdefault("simulation_config", {})
    if isinstance(sim_config, dict) and sim_config.get("route_spawn_seed") is None:
        sim_config["route_spawn_seed"] = int(seed)
    return updated


def planner_independent_scenario_case_payload(
    scenario: Mapping[str, Any], *, seed: int
) -> dict[str, Any]:
    """Return the candidate-row identity that remains after map-run identity fields.

    The map runner expands the selected scenario, adds an ID default and seed-derived
    route defaults, then overlays run-level identity fields in
    :func:`_scenario_identity_payload`. This projection applies the same row and seed
    rules while dropping only that documented run-level envelope. It lets consumers
    bind an episode row to the selected candidate instead of trusting the episode's
    self-consistent config hash alone.
    """
    identity_scenario = _scenario_with_episode_seed_defaults(dict(scenario), seed=seed)
    payload = {
        key: value for key, value in identity_scenario.items() if key not in {"seed", "seeds"}
    }
    scenario_id = (
        identity_scenario.get("name")
        or identity_scenario.get("scenario_id")
        or identity_scenario.get("id")
        or "unknown"
    )
    payload.setdefault("id", scenario_id)
    for field in _MAP_RUNNER_SCENARIO_IDENTITY_FIELDS:
        payload.pop(field, None)
    return payload


def selected_map_identity_from_runtime_inputs(
    map_id: Any,
    runtime_input_records: list[dict[str, str]],
    *,
    scenario_id: str,
) -> dict[str, Any]:
    """Bind the realized map to one parser-captured source resource.

    An implicit map pool can contain multiple maps, so its complete resource closure does
    not identify which map the environment actually selected. This record ties the
    environment's realized map id to the exact parser-consumed source path and digest.

    Returns:
        A schema-shaped available identity when exactly one source matches, otherwise an
        unavailable identity with a reason.
    """
    unavailable = {
        "schema_version": "selected_map_identity.v1",
        "status": "unavailable",
        "map_id": map_id if isinstance(map_id, str) and map_id.strip() else None,
        "path": None,
        "sha256": None,
        "source_role": None,
        "reason": None,
    }
    if not isinstance(map_id, str) or not map_id.strip():
        unavailable["reason"] = "realized_map_id_unavailable"
        return unavailable
    map_records = [
        record
        for record in runtime_input_records
        if isinstance(record, dict)
        and record.get("scenario_id") == scenario_id
        and record.get("role") in {"map_file", "default_map_pool"}
    ]
    matches = [record for record in map_records if record.get("map_id") == map_id]
    if not matches and len(map_records) == 1 and "map_id" not in map_records[0]:
        # Explicit map_file rows historically omit a map_id when the scenario does not
        # declare one; a single parser-consumed map resource still binds unambiguously.
        matches = map_records
    if len(matches) != 1:
        unavailable["reason"] = "realized_map_source_not_unique"
        return unavailable
    record = matches[0]
    path = record.get("path")
    sha256 = record.get("sha256")
    source_role = record.get("role")
    if not all(isinstance(value, str) and value.strip() for value in (path, sha256, source_role)):
        unavailable["reason"] = "realized_map_source_identity_incomplete"
        return unavailable
    if len(sha256) != 64 or any(character not in "0123456789abcdefABCDEF" for character in sha256):
        unavailable["reason"] = "realized_map_source_digest_invalid"
        return unavailable
    return {
        "schema_version": "selected_map_identity.v1",
        "status": "available",
        "map_id": map_id,
        "path": path,
        "sha256": sha256.lower(),
        "source_role": source_role,
        "reason": None,
    }


resolve_seed_list = _resolve_seed_list
suite_key = _suite_key
select_seeds = _select_seeds
scenario_identity_payload = _scenario_identity_payload
compute_map_episode_id = _compute_map_episode_id
scenario_with_episode_seed_defaults = _scenario_with_episode_seed_defaults
