"""Readiness inventory for issue #3557 uncertainty-source episode runs.

This module is intentionally a preflight inventory only. It does not run the
#3471 episode harness, compute new metrics, or decide whether the uncertainty
drop safety effect generalizes. It answers the smaller question needed before
those runs: which uncertainty sources already have the builders, scenario hooks,
and surrogate output fields required for a per-source episode-level contrast?
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

UNCERTAINTY_SOURCE_READINESS_SCHEMA = "uncertainty_source_readiness.v1"

logger = logging.getLogger(__name__)

_GENERALIZATION_SURROGATE_OUTPUTS = frozenset(
    {
        "source",
        "retained_unsafe_commit_rate",
        "dropped_unsafe_commit_rate",
        "min_separation_delta_m",
        "n_episodes",
    }
)


@dataclass(frozen=True, slots=True)
class UncertaintySourceRunSpec:
    """Inventory contract for one candidate uncertainty source."""

    source: str
    condition_builder: str | None
    scenario_hook: str | None
    expected_surrogate_outputs: tuple[str, ...] = tuple(sorted(_GENERALIZATION_SURROGATE_OUTPUTS))


DEFAULT_UNCERTAINTY_SOURCE_SPECS: tuple[UncertaintySourceRunSpec, ...] = (
    UncertaintySourceRunSpec(
        source="existence_degraded",
        condition_builder="_condition_existence_degraded",
        scenario_hook="build_belief_for_mode",
    ),
    UncertaintySourceRunSpec(
        source="visibility_limited",
        condition_builder="_condition_visibility_limited",
        scenario_hook=None,
    ),
    UncertaintySourceRunSpec(
        source="covariance_inflated",
        condition_builder="_condition_covariance_inflated",
        scenario_hook=None,
    ),
    UncertaintySourceRunSpec(
        source="class_probability",
        condition_builder="_condition_class_probability",
        scenario_hook=None,
    ),
    UncertaintySourceRunSpec(
        source="tracking_noise",
        condition_builder=None,
        scenario_hook=None,
    ),
)


def _missing_expected_outputs(spec: UncertaintySourceRunSpec) -> list[str]:
    """Return required surrogate fields absent from ``spec``."""

    declared = set(spec.expected_surrogate_outputs)
    return sorted(_GENERALIZATION_SURROGATE_OUTPUTS - declared)


def _discover_condition_builders() -> frozenset[str]:
    """Return condition-builder symbols present in the #2546 diagnostic owner."""

    try:
        module = importlib.import_module(
            "scripts.analysis.run_scenario_belief_uncertainty_diagnostic_issue_2546"
        )
    except Exception:
        logger.exception("Failed to import uncertainty diagnostic owner for readiness discovery")
        return frozenset()
    return frozenset(name for name in dir(module) if name.startswith("_condition_"))


def _discover_scenario_hooks() -> frozenset[str]:
    """Return episode scenario hook symbols present in the #3471 runner owner."""

    try:
        module = importlib.import_module(
            "scripts.validation.run_scenario_belief_episode_safety_issue_3471"
        )
    except Exception:
        logger.exception("Failed to import episode safety owner for readiness discovery")
        return frozenset()
    return frozenset(
        {"build_belief_for_mode"} if hasattr(module, "build_belief_for_mode") else set()
    )


def classify_uncertainty_source_readiness(
    spec: UncertaintySourceRunSpec,
    *,
    known_condition_builders: set[str] | frozenset[str] | None = None,
    known_scenario_hooks: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Classify one source as ready or blocked for a future episode run.

    The result is a static inventory row. A source is ``ready`` only when it has
    an existing condition builder, an existing scenario hook, and the surrogate
    outputs needed by the already-landed #3557 decision layer.

    Returns:
        dict[str, Any]: Source readiness row with status and missing contract parts.
    """

    known_condition_builders = (
        _discover_condition_builders()
        if known_condition_builders is None
        else known_condition_builders
    )
    known_scenario_hooks = (
        _discover_scenario_hooks() if known_scenario_hooks is None else known_scenario_hooks
    )
    missing: list[str] = []

    if spec.condition_builder is None:
        missing.append("condition_builder")
        condition_builder_present = False
    else:
        condition_builder_present = spec.condition_builder in known_condition_builders
        if not condition_builder_present:
            missing.append("condition_builder")

    if spec.scenario_hook is None:
        missing.append("scenario_hook")
        scenario_hook_present = False
    else:
        scenario_hook_present = spec.scenario_hook in known_scenario_hooks
        if not scenario_hook_present:
            missing.append("scenario_hook")

    missing_outputs = _missing_expected_outputs(spec)
    expected_surrogate_outputs_present = not missing_outputs
    if missing_outputs:
        missing.append("expected_surrogate_outputs")

    return {
        "source": spec.source,
        "status": "ready" if not missing else "blocked",
        "missing": sorted(set(missing)),
        "condition_builder": spec.condition_builder,
        "condition_builder_present": condition_builder_present,
        "scenario_hook": spec.scenario_hook,
        "scenario_hook_present": scenario_hook_present,
        "expected_surrogate_outputs": list(spec.expected_surrogate_outputs),
        "expected_surrogate_outputs_present": expected_surrogate_outputs_present,
        "missing_expected_surrogate_outputs": missing_outputs,
    }


def build_uncertainty_source_readiness_inventory(
    specs: tuple[UncertaintySourceRunSpec, ...] | list[UncertaintySourceRunSpec] = (
        DEFAULT_UNCERTAINTY_SOURCE_SPECS
    ),
) -> dict[str, Any]:
    """Build a versioned readiness inventory for issue #3557.

    Returns:
        dict[str, Any]: Diagnostic readiness report only. ``ready_sources`` names sources
        that can be wired into an episode-level contrast with current hooks;
        ``blocked_sources`` names sources that still need builder/hook/output work.
    """

    if not specs:
        raise ValueError("at least one uncertainty source spec is required")

    known_condition_builders = _discover_condition_builders()
    known_scenario_hooks = _discover_scenario_hooks()
    rows = [
        classify_uncertainty_source_readiness(
            spec,
            known_condition_builders=known_condition_builders,
            known_scenario_hooks=known_scenario_hooks,
        )
        for spec in specs
    ]
    ready_sources = sorted(row["source"] for row in rows if row["status"] == "ready")
    blocked_sources = sorted(row["source"] for row in rows if row["status"] == "blocked")

    return {
        "schema_version": UNCERTAINTY_SOURCE_READINESS_SCHEMA,
        "issue": 3557,
        "claim_boundary": "readiness_inventory_only",
        "not_benchmark_evidence": True,
        "runs_executed": False,
        "surrogate_semantics_changed": False,
        "sources": rows,
        "ready_sources": ready_sources,
        "blocked_sources": blocked_sources,
        "all_sources_ready": not blocked_sources,
    }


__all__ = [
    "DEFAULT_UNCERTAINTY_SOURCE_SPECS",
    "UNCERTAINTY_SOURCE_READINESS_SCHEMA",
    "UncertaintySourceRunSpec",
    "build_uncertainty_source_readiness_inventory",
    "classify_uncertainty_source_readiness",
]
