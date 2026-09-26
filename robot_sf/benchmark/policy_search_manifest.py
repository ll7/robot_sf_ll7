"""Pure resolution of policy-search candidate manifests.

The map runner and evidence validators share this implementation so a recorded
per-scenario config hash can be checked against the same manifest semantics used
to execute the planner.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def _scenario_id(scenario: dict[str, Any]) -> str:
    """Return the map-runner scenario identifier."""
    return str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )


def scenario_family(scenario: dict[str, Any]) -> str:
    """Return the map-runner report family for a scenario."""
    scenario_id = _scenario_id(scenario)
    if scenario_id.startswith("francis2023_"):
        return "francis2023"
    if scenario_id.startswith("classic_"):
        return "classic"
    return str(scenario.get("family") or scenario.get("metadata", {}).get("family") or "nominal")


def is_candidate_manifest(manifest: dict[str, Any]) -> bool:
    """Return whether a mapping declares policy-search candidate fields."""
    return any(
        key in manifest
        for key in (
            "base_config_path",
            "params",
            "family_overrides",
            "scenario_overrides",
            "scenario_algo_overrides",
        )
    )


def deep_merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively apply overrides without mutating either input.

    Returns:
        A new config mapping with overrides applied.
    """
    merged = deepcopy(base)
    _deep_merge_inline(merged, overrides)
    return merged


def _deep_merge_inline(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply nested mapping overrides to an isolated config mapping."""
    for key, value in overrides.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _deep_merge_inline(current, value)
        else:
            base[key] = deepcopy(value)


def resolve_candidate_manifest_runtime(  # noqa: C901
    *,
    default_algo: str,
    manifest: dict[str, Any],
    scenario: dict[str, Any],
    load_config: Callable[[object], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Resolve a candidate manifest to the map-runner algorithm and config.

    ``load_config`` resolves a manifest's optional base-config path using the
    caller's source policy (mutable runtime paths or immutable Git blobs).

    Returns:
        Effective algorithm and flattened runtime config for this scenario.
    """
    if not is_candidate_manifest(manifest):
        return default_algo, manifest

    scenario_key = _scenario_id(scenario)
    algo_overrides = manifest.get("scenario_algo_overrides")
    if isinstance(algo_overrides, dict):
        override = algo_overrides.get(scenario_key)
        if isinstance(override, dict):
            algo = str(override.get("algo", default_algo)).strip().lower()
            if not algo:
                raise ValueError(f"Scenario algo override is missing algo: {scenario_key}")
            base_cfg = load_config(override.get("base_config_path"))
            params = override.get("params") or {}
            if not isinstance(params, dict):
                raise TypeError("Policy-search scenario_algo_overrides params must be a mapping.")
            return algo, deep_merge_config(base_cfg, params)

    params = manifest.get("params") or {}
    if not isinstance(params, dict):
        raise TypeError("Policy-search candidate params must be a mapping.")
    effective = deep_merge_config(load_config(manifest.get("base_config_path")), params)
    family_overrides = manifest.get("family_overrides")
    if isinstance(family_overrides, dict):
        family_cfg = family_overrides.get(scenario_family(scenario), {})
        if isinstance(family_cfg, dict):
            effective = deep_merge_config(effective, family_cfg)
    scenario_overrides = manifest.get("scenario_overrides")
    if isinstance(scenario_overrides, dict):
        scenario_cfg = scenario_overrides.get(scenario_key, {})
        if isinstance(scenario_cfg, dict):
            effective = deep_merge_config(effective, scenario_cfg)
    return default_algo, effective
