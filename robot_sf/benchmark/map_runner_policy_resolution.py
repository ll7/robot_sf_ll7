"""Policy runtime resolution helpers extracted from :mod:`map_runner`.

These helpers are used by the map benchmark runner when resolving policy-search
candidate manifests, selector-v2 runtime wiring, and prediction metadata overrides.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.local_model_artifacts import validate_no_local_model_artifacts
from robot_sf.benchmark.map_runner_trace import _scenario_id
from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)
from robot_sf.planner.planner_selector_v2_diagnostic import (
    PlannerSelectorV2DiagnosticAdapter,
    build_planner_selector_v2_diagnostic_config,
)
from robot_sf.planner.socnav import ORCAPlannerAdapter, SocNavPlannerConfig


def _parse_algo_config(algo_config_path: str | None) -> dict[str, Any]:
    """Load an optional planner YAML config.

    Returns:
        dict[str, Any]: Parsed config mapping, or an empty mapping when omitted.
    """
    if not algo_config_path:
        return {}
    path = Path(algo_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Algorithm config file not found: {algo_config_path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("Algorithm config must be a mapping (YAML dict).")
    validate_no_local_model_artifacts(data, config_path=path)
    return data


def _deep_merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge nested planner config overrides without mutating either input.

    Returns:
        A new mapping containing ``base`` with ``overrides`` applied recursively.
    """
    merged = deepcopy(base)
    _deep_merge_inline(merged, overrides)
    return merged


def _deep_merge_inline(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Merge override values into an already isolated base mapping."""
    for key, value in overrides.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _deep_merge_inline(current, value)
        else:
            base[key] = deepcopy(value)


_UNCERTAINTY_ENVELOPE_ALGOS = {
    "cv_prediction_mpc",
    "learned_prediction_mpc",
    "nmpc",
    "nmpc_social",
    "prediction_aware_mpc",
    "prediction_mpc",
}
_UNCERTAINTY_ENVELOPE_FIELDS = (
    "pedestrian_uncertainty_envelope_enabled",
    "pedestrian_uncertainty_alpha_mps",
)


def _apply_scenario_uncertainty_envelope_config(
    algo: str,
    algo_config: dict[str, Any],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Thread scenario-level pedestrian uncertainty-envelope fields into planner config.

    Returns:
        Planner config with scenario envelope fields merged for supported planners.
    """
    algo_key = str(algo).strip().lower()
    if algo_key not in _UNCERTAINTY_ENVELOPE_ALGOS:
        return algo_config
    sim_config = scenario.get("simulation_config")
    if not isinstance(sim_config, dict):
        return algo_config
    overrides = {
        field: sim_config[field] for field in _UNCERTAINTY_ENVELOPE_FIELDS if field in sim_config
    }
    if not overrides:
        return algo_config
    merged = deepcopy(algo_config)
    merged.update(overrides)
    return merged


def _resolve_config_path(anchor: Path | None, raw_path: Any) -> Path | None:
    """Resolve candidate-manifest config paths from manifest-local or repo-root form.

    Returns:
        Resolved path, or ``None`` when the raw path is empty or not a string.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    if anchor is not None:
        anchored = (anchor / path).resolve()
        if anchored.exists():
            return anchored
    return path.resolve()


def _scenario_family(scenario: dict[str, Any]) -> str:
    """Classify a scenario into the report family used by benchmark summaries.

    Returns:
        str: Scenario family label.
    """
    scenario_id = _scenario_id(scenario)
    if scenario_id.startswith("francis2023_"):
        return "francis2023"
    if scenario_id.startswith("classic_"):
        return "classic"
    return str(scenario.get("family") or scenario.get("metadata", {}).get("family") or "nominal")


def _is_policy_search_candidate_manifest(config: dict[str, Any]) -> bool:
    """Return whether a config has policy-search candidate manifest fields."""
    return any(
        key in config
        for key in (
            "base_config_path",
            "params",
            "family_overrides",
            "scenario_overrides",
            "scenario_algo_overrides",
        )
    )


def _load_base_candidate_config(
    manifest: dict[str, Any],
    *,
    config_anchor: Path | None,
) -> dict[str, Any]:
    """Load and merge a policy-search candidate's base config and params.

    Returns:
        dict[str, Any]: Effective candidate planner config.
    """
    base_cfg: dict[str, Any] = {}
    base_path = _resolve_config_path(config_anchor, manifest.get("base_config_path"))
    if base_path is not None:
        base_cfg = _parse_algo_config(str(base_path))
    params = manifest.get("params") or {}
    if not isinstance(params, dict):
        raise TypeError("Policy-search candidate params must be a mapping.")
    return _deep_merge_config(base_cfg, params)


def _scenario_algo_override_runtime(
    override: dict[str, Any],
    *,
    default_algo: str,
    scenario_key: str,
    config_anchor: Path | None,
) -> tuple[str, dict[str, Any]]:
    """Resolve one scenario-level algorithm override.

    Returns:
        Effective algorithm key and flattened runtime config for the scenario.
    """
    algo = str(override.get("algo", default_algo)).strip().lower()
    if not algo:
        raise ValueError(f"Scenario algo override is missing algo: {scenario_key}")
    base_cfg: dict[str, Any] = {}
    base_path = _resolve_config_path(config_anchor, override.get("base_config_path"))
    if base_path is not None:
        base_cfg = _parse_algo_config(str(base_path))
    params = override.get("params") or {}
    if not isinstance(params, dict):
        raise TypeError("Policy-search scenario_algo_overrides params must be a mapping.")
    return algo, _deep_merge_config(base_cfg, params)


def _resolve_policy_search_candidate_runtime(
    *,
    default_algo: str,
    algo_config_path: str | None,
    scenario: dict[str, Any],
    algo_config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve a policy-search candidate manifest to the runtime algo/config for a scenario.

    Returns:
        Effective algorithm key and flattened runtime config for the scenario.
    """
    manifest = (
        dict(algo_config) if algo_config is not None else _parse_algo_config(algo_config_path)
    )
    if not _is_policy_search_candidate_manifest(manifest):
        return default_algo, manifest

    config_anchor = Path(algo_config_path).resolve().parent if algo_config_path else None
    scenario_key = _scenario_id(scenario)
    algo_overrides = manifest.get("scenario_algo_overrides")
    if isinstance(algo_overrides, dict):
        override = algo_overrides.get(scenario_key)
        if isinstance(override, dict):
            return _scenario_algo_override_runtime(
                override,
                default_algo=default_algo,
                scenario_key=scenario_key,
                config_anchor=config_anchor,
            )

    effective = _load_base_candidate_config(manifest, config_anchor=config_anchor)
    family_overrides = manifest.get("family_overrides")
    if isinstance(family_overrides, dict):
        family_cfg = family_overrides.get(_scenario_family(scenario), {})
        if isinstance(family_cfg, dict):
            effective = _deep_merge_config(effective, family_cfg)
    scenario_overrides = manifest.get("scenario_overrides")
    if isinstance(scenario_overrides, dict):
        scenario_cfg = scenario_overrides.get(scenario_key, {})
        if isinstance(scenario_cfg, dict):
            effective = _deep_merge_config(effective, scenario_cfg)
    return default_algo, effective


def _apply_planner_selector_v2_context(
    algo: str,
    policy_cfg: dict[str, Any],
    *,
    scenario: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Attach selector-v2 scenario context wherever effective runtime config is materialized.

    Returns:
        Runtime config with selector context for planner-selector v2, otherwise the original config.
    """
    if str(algo).strip().lower() != "planner_selector_v2_diagnostic":
        return policy_cfg
    return _deep_merge_config(
        policy_cfg,
        {
            "selector_context": {
                "scenario_id": _scenario_id(scenario),
                "scenario_family": _scenario_family(scenario),
                "seed": int(seed),
            }
        },
    )


def _build_socnav_config(cfg: dict[str, Any]) -> SocNavPlannerConfig:
    """Build a SocNav planner config from a loose mapping.

    Returns:
        SocNavPlannerConfig: Filtered planner configuration.
    """
    if not isinstance(cfg, dict):
        return SocNavPlannerConfig()
    allowed = {f.name for f in fields(SocNavPlannerConfig)}
    filtered = {key: value for key, value in cfg.items() if key in allowed}
    return SocNavPlannerConfig(**filtered)


def _build_planner_selector_v2_child_adapter(
    *,
    candidate_name: str,
    candidate_config_path: str,
    scenario: dict[str, Any],
) -> Any:
    """Build one existing local candidate adapter for planner-selector v2.

    Returns:
        Adapter instance for a supported local child candidate.
    """
    path = Path(candidate_config_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    manifest = _parse_algo_config(str(path))
    child_default_algo = str(manifest.get("algo", "")).strip().lower()
    if not child_default_algo:
        raise ValueError(f"Selector child candidate is missing algo: {candidate_name}")
    child_algo, child_config = _resolve_policy_search_candidate_runtime(
        default_algo=child_default_algo,
        algo_config_path=str(path),
        algo_config=manifest,
        scenario=scenario,
    )
    if child_algo == "hybrid_rule_local_planner":
        return HybridRuleLocalPlannerAdapter(
            config=build_hybrid_rule_local_planner_config(child_config)
        )
    if child_algo == "orca":
        return ORCAPlannerAdapter(
            config=_build_socnav_config(child_config),
            allow_fallback=bool(child_config.get("allow_fallback", False)),
        )
    raise ValueError(
        "planner_selector_v2_diagnostic only supports existing local hybrid-rule/ORCA "
        f"child candidates; {candidate_name!r} resolved to {child_algo!r}"
    )


def _build_planner_selector_v2_adapter(
    algo_config: dict[str, Any],
) -> PlannerSelectorV2DiagnosticAdapter:
    """Build the diagnostic selector and all configured local child candidates.

    Returns:
        Configured planner-selector v2 adapter.
    """
    build = build_planner_selector_v2_diagnostic_config(algo_config)
    scenario_stub = {
        "name": build.selector.scenario_id,
        "family": build.selector.scenario_family,
    }
    adapters = {
        name: _build_planner_selector_v2_child_adapter(
            candidate_name=name,
            candidate_config_path=path,
            scenario=scenario_stub,
        )
        for name, path in sorted(build.candidate_config_paths.items())
    }
    return PlannerSelectorV2DiagnosticAdapter(
        config=build.selector,
        candidate_adapters=adapters,
    )


def _prediction_planner_metadata_overrides(
    algo_config: dict[str, Any],
) -> dict[str, Any]:
    """Expose predictive-planner search and uncertainty modes as first-class metadata.

    Returns:
        dict[str, Any]: Explicit mode labels for audit-friendly benchmark metadata.
    """
    uncertainty_mode = str(algo_config.get("predictive_uncertainty_mode", "deterministic")).strip()
    search_mode = "mcts_lite"
    if not bool(algo_config.get("predictive_mcts_enabled", False)):
        search_mode = (
            "sequence_beam"
            if bool(algo_config.get("predictive_sequence_search_enabled", False))
            else "lattice"
        )
    sample_count = int(algo_config.get("predictive_risk_sample_count", 1))
    return {
        "prediction_mode": "probabilistic"
        if uncertainty_mode.lower() != "deterministic" or sample_count > 1
        else "deterministic",
        "predictive_uncertainty_mode": uncertainty_mode,
        "predictive_risk_objective": str(
            algo_config.get("predictive_risk_objective", "mean")
        ).strip(),
        "predictive_risk_sample_count": sample_count,
        "predictive_search_mode": search_mode,
    }
