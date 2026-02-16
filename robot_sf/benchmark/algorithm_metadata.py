"""Shared algorithm metadata helpers for benchmark episode outputs.

This module centralizes baseline category labels, policy semantics, and
planner-kinematics metadata so benchmark writers emit a consistent contract.
"""

from __future__ import annotations

from typing import Any

from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness

_BASELINE_CATEGORY_BY_CANONICAL: dict[str, str] = {
    "goal": "classical",
    "social_force": "classical",
    "orca": "classical",
    "ppo": "learning",
    "socnav_sampling": "classical",
    "sacadrl": "learning",
    "socnav_bench": "classical",
    "random": "diagnostic",
    "fast_pysf_planner": "diagnostic",
    "rvo": "classical",
    "dwa": "classical",
    "teb": "classical",
}

_POLICY_SEMANTICS_BY_CANONICAL: dict[str, str] = {
    "goal": "deterministic_goal_seeking",
    "social_force": "social_force_adapter",
    "orca": "orca_adapter",
    "ppo": "policy_network_inference",
    "socnav_sampling": "stochastic_sampling_adapter",
    "sacadrl": "learned_value_adapter",
    "socnav_bench": "socnav_adapter",
    "random": "stochastic_uniform_action_reference",
    "fast_pysf_planner": "social_force_reference",
    "rvo": "placeholder_adapter",
    "dwa": "placeholder_adapter",
    "teb": "placeholder_adapter",
}

_KINEMATICS_PROFILE_BY_CANONICAL: dict[str, dict[str, Any]] = {
    "goal": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": True,
        "supports_adapter_commands": False,
        "default_execution_mode": "native",
    },
    "social_force": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SocialForcePlannerAdapter",
    },
    "orca": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "ORCAPlannerAdapter",
    },
    "ppo": {
        "planner_command_space": "mixed_vw_or_vxy",
        "supports_native_commands": True,
        "supports_adapter_commands": True,
        "default_execution_mode": "mixed",
        "default_adapter_name": "ppo_action_to_unicycle",
    },
    "socnav_sampling": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SocNavBenchSamplingAdapter",
    },
    "sacadrl": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SACADRLPlannerAdapter",
    },
    "socnav_bench": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SocNavBenchSamplingAdapter",
    },
    "random": {
        "planner_command_space": "randomized_vxy_or_vw",
        "supports_native_commands": True,
        "supports_adapter_commands": False,
        "default_execution_mode": "native",
    },
    "fast_pysf_planner": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "PlannerActionAdapter",
    },
    "rvo": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SamplingPlannerAdapter",
    },
    "dwa": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SamplingPlannerAdapter",
    },
    "teb": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SamplingPlannerAdapter",
    },
}


def canonical_algorithm_name(algo: str) -> str:
    """Normalize algorithm aliases to canonical benchmark names when possible.

    Returns:
        Canonical algorithm name when known, otherwise normalized input alias.
    """
    alias = str(algo).strip().lower()
    readiness = get_algorithm_readiness(alias)
    return readiness.canonical_name if readiness is not None else alias


def _base_kinematics_metadata(
    canonical_algo: str,
    *,
    execution_mode: str | None = None,
    adapter_name: str | None = None,
    robot_kinematics: str | None = None,
) -> dict[str, Any]:
    profile = _KINEMATICS_PROFILE_BY_CANONICAL.get(canonical_algo, {})
    metadata = {
        "robot_kinematics": robot_kinematics or "unknown",
        "planner_command_space": profile.get("planner_command_space", "unknown"),
        "supports_native_commands": bool(profile.get("supports_native_commands", False)),
        "supports_adapter_commands": bool(profile.get("supports_adapter_commands", False)),
        "execution_mode": execution_mode or profile.get("default_execution_mode", "unknown"),
        "adapter_name": adapter_name or profile.get("default_adapter_name", "none"),
    }
    metadata["adapter_active"] = metadata["execution_mode"] in {"adapter", "mixed"}
    return metadata


def enrich_algorithm_metadata(
    *,
    algo: str,
    metadata: dict[str, Any] | None = None,
    execution_mode: str | None = None,
    adapter_name: str | None = None,
    robot_kinematics: str | None = None,
    adapter_impact_requested: bool | None = None,
) -> dict[str, Any]:
    """Return metadata enriched with baseline category and compatibility fields.

    Args:
        algo: Algorithm label as selected by the caller.
        metadata: Existing metadata payload to preserve and augment.
        execution_mode: Optional runtime override (`native`/`adapter`/`mixed`).
        adapter_name: Optional runtime adapter override.
        robot_kinematics: Optional robot kinematics tag for this episode/run.
        adapter_impact_requested: Optional marker that adapter-impact probing was requested.

    Returns:
        A metadata dictionary with stable benchmark contract keys.
    """
    enriched: dict[str, Any] = dict(metadata or {})
    requested = str(algo).strip().lower()
    canonical = canonical_algorithm_name(requested)

    enriched.setdefault("algorithm", requested)
    enriched.setdefault("status", "ok")
    enriched["canonical_algorithm"] = canonical
    enriched.setdefault(
        "baseline_category",
        _BASELINE_CATEGORY_BY_CANONICAL.get(canonical, "unknown"),
    )
    enriched.setdefault(
        "policy_semantics",
        _POLICY_SEMANTICS_BY_CANONICAL.get(canonical, "unspecified"),
    )

    current_kinematics = enriched.get("planner_kinematics")
    base_kinematics = _base_kinematics_metadata(
        canonical,
        execution_mode=execution_mode,
        adapter_name=adapter_name,
        robot_kinematics=robot_kinematics,
    )
    if isinstance(current_kinematics, dict):
        merged = dict(current_kinematics)
        for key, value in base_kinematics.items():
            merged.setdefault(key, value)
        if execution_mode is not None:
            merged["execution_mode"] = execution_mode
            merged["adapter_active"] = execution_mode in {"adapter", "mixed"}
        if adapter_name is not None:
            merged["adapter_name"] = adapter_name
        if robot_kinematics is not None:
            merged["robot_kinematics"] = robot_kinematics
        enriched["planner_kinematics"] = merged
    else:
        enriched["planner_kinematics"] = base_kinematics

    if canonical == "random":
        enriched.setdefault("stochastic_reference", True)
        enriched.setdefault("distinct_from_goal_baseline", True)

    if adapter_impact_requested is not None:
        impact = enriched.get("adapter_impact")
        if not isinstance(impact, dict):
            impact = {}
        impact.setdefault("requested", bool(adapter_impact_requested))
        impact.setdefault("native_steps", 0)
        impact.setdefault("adapted_steps", 0)
        impact.setdefault("status", "pending" if adapter_impact_requested else "disabled")
        enriched["adapter_impact"] = impact

    return enriched


def infer_execution_mode_from_counts(native_steps: int, adapted_steps: int) -> str:
    """Infer execution mode from runtime native/adapted step counters.

    Returns:
        One of ``native``, ``adapter``, ``mixed``, or ``unknown``.
    """
    if native_steps > 0 and adapted_steps > 0:
        return "mixed"
    if adapted_steps > 0:
        return "adapter"
    if native_steps > 0:
        return "native"
    return "unknown"


__all__ = [
    "canonical_algorithm_name",
    "enrich_algorithm_metadata",
    "infer_execution_mode_from_counts",
]
