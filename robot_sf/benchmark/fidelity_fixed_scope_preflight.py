"""Full fixed-scope fidelity-sensitivity preflight (issue #3207).

This module pre-registers the *full fixed-scope* simulator-fidelity sensitivity
campaign before it is launched. It does three things and nothing more:

1. **Materializes** the explicit run plan from the study config — every planner
   group crossed with every axis variant and every fixed seed over the fixed
   scenario set — so the intended full scope is enumerated rather than implied.
2. **Resolves planner availability** against the canonical
   :mod:`robot_sf.benchmark.algorithm_readiness` catalog and **fails closed** on
   any planner group that cannot be resolved to a non-placeholder algorithm.
3. **Validates the primary (ranking) metric** is identifiable *by contract*
   (declared, present in the metric list, with a direction, and not flagged
   non-identifiable/zero-variance per issue #3299) and **fails closed** otherwise.

It runs no benchmark episodes and promotes no claim. ``preflight_ready`` is a
pre-registration gate only: it means the contract-level launch checks pass, not
that runtime dependencies (e.g. rvo2 for ORCA) are present or that the measured
ranking will be identifiable at runtime. Runtime dependencies remain explicit
``launch_prerequisites``; measured rank identifiability is a post-run contract.
The emitted packet is a launch/readiness artifact and
is not benchmark evidence, not simulator-realism evidence, not sim-to-real
evidence, and not paper-facing evidence.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.benchmark.fidelity_rank_stability import PRIMARY_METRIC_ZERO_VARIANCE_REASON
from robot_sf.benchmark.fidelity_sensitivity import validate_fidelity_sensitivity_config

SCHEMA_VERSION = "fidelity-fixed-scope-preflight.v1"

DECISION_READY = "preflight_ready"
DECISION_BLOCKED = "blocked"

EVIDENCE_STATUS = "launch_readiness_preflight_not_benchmark_evidence"

# Phrases the decision-checker family (simulator_dependence_validity_boundary)
# already requires in every no-claim boundary; the preflight packet reuses them
# so downstream consumers see a consistent claim boundary.
REQUIRED_CLAIM_BOUNDARY_PHRASES = (
    "not benchmark evidence",
    "not simulator-realism evidence",
    "not sim-to-real evidence",
    "not paper-facing evidence",
)

PREFLIGHT_CLAIM_BOUNDARY = (
    "Launch/readiness preflight only: pre-registers the full fixed-scope fidelity-sensitivity "
    "campaign scope, planner availability, and primary-metric identifiability. "
    "preflight_ready is a pre-registration gate, not an execution-ready signal; it is "
    "not benchmark evidence, not simulator-realism evidence, not sim-to-real evidence, "
    "and not paper-facing evidence."
)


def build_fixed_scope_preflight(
    config: Mapping[str, Any],
    *,
    config_path: str,
    git_head: str,
    date: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed full fixed-scope preflight/readiness packet.

    Args:
        config: Raw fidelity-sensitivity study config mapping.
        config_path: Repo-relative path of the config, recorded for provenance.
        git_head: Git head recorded for provenance.
        date: Optional ISO date string recorded for provenance.

    Returns:
        JSON-serializable readiness packet. ``decision`` is ``preflight_ready``
        only when no blockers are present; otherwise it is ``blocked``.
    """
    validated = validate_fidelity_sensitivity_config(config)

    blockers: list[str] = []
    launch_prerequisites: list[str] = []

    materialized = _materialize_full_fixed_scope(validated)
    planner_resolution = _resolve_planner_groups(
        validated, blockers=blockers, launch_prerequisites=launch_prerequisites
    )
    primary_metric = _check_primary_metric(validated, blockers=blockers)

    post_run_contracts = [
        "runtime_rank_identifiability_recheck_required: measured primary-metric variance is "
        "re-validated post-run via robot_sf/benchmark/fidelity_rank_stability.py"
    ]

    decision = DECISION_BLOCKED if blockers else DECISION_READY

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": int(validated.get("issue", 3207)),
        "study_id": str(validated["study_id"]),
        "decision": decision,
        "preflight_ready": decision == DECISION_READY,
        "evidence_status": EVIDENCE_STATUS,
        "claim_boundary": PREFLIGHT_CLAIM_BOUNDARY,
        "config_path": config_path,
        "git_head": git_head,
        "date": date,
        "materialized_scope": materialized,
        "planner_resolution": planner_resolution,
        "primary_metric": primary_metric,
        "blockers": blockers,
        "launch_prerequisites": launch_prerequisites,
        "post_run_contracts": post_run_contracts,
        "next_command_template": (
            "uv run python scripts/benchmark/run_fidelity_sensitivity_campaign.py --config "
            f"{config_path} --out output/fidelity_sensitivity/"
        ),
    }


def write_fixed_scope_preflight(packet: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write a deterministic JSON preflight packet.

    Returns:
        Path to the written ``fidelity_fixed_scope_preflight.json`` file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    packet_path = out / "fidelity_fixed_scope_preflight.json"
    packet_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return packet_path


def _materialize_full_fixed_scope(validated: Mapping[str, Any]) -> dict[str, Any]:
    """Enumerate the explicit full fixed-scope run plan.

    Returns:
        Counts and identifiers for the planner × axis-variant × seed cells that
        the full campaign must cover per scenario in the fixed scenario set.
    """
    fixed_scope = validated["fixed_scope"]
    seeds = list(fixed_scope["seeds"])
    planner_groups = list(fixed_scope["planner_groups"])
    axes = validated["axes"]
    variant_count = sum(len(axis["variants"]) for axis in axes)
    run_cells = len(planner_groups) * variant_count * len(seeds)
    return {
        "scenario_set": str(fixed_scope["scenario_set"]),
        "seed_count": len(seeds),
        "seeds": seeds,
        "planner_group_count": len(planner_groups),
        "planner_groups": planner_groups,
        "axis_count": len(axes),
        "variant_count": variant_count,
        "run_cells_per_scenario": run_cells,
        "run_cells_note": (
            "run_cells_per_scenario counts planner_group x axis-variant x seed cells for each "
            "scenario in scenario_set; scenario multiplicity is resolved by the campaign runner "
            "and is not expanded here."
        ),
    }


def _resolve_planner_groups(
    validated: Mapping[str, Any],
    *,
    blockers: list[str],
    launch_prerequisites: list[str],
) -> list[dict[str, Any]]:
    """Resolve each planner group to a catalog algorithm, failing closed.

    A planner group resolves through an explicit ``fixed_scope.planner_algorithms``
    mapping when present, otherwise via the group label directly. Unresolved or
    placeholder-tier planners are recorded as blockers; experimental planners that
    require explicit opt-in are recorded as launch prerequisites.

    Returns:
        Per-planner resolution records in config order.
    """
    fixed_scope = validated["fixed_scope"]
    planner_groups = list(fixed_scope["planner_groups"])
    algorithm_map = fixed_scope.get("planner_algorithms")
    if algorithm_map is not None and not isinstance(algorithm_map, Mapping):
        raise ValueError("fixed_scope.planner_algorithms must be a mapping when set")
    opt_in_map = fixed_scope.get("planner_opt_ins") or {}
    if not isinstance(opt_in_map, Mapping):
        raise ValueError("fixed_scope.planner_opt_ins must be a mapping when set")

    resolution: list[dict[str, Any]] = []
    for group in planner_groups:
        group_name = str(group)
        algorithm = (
            str(algorithm_map[group_name]) if _in_map(algorithm_map, group_name) else group_name
        )
        readiness = get_algorithm_readiness(algorithm)
        record: dict[str, Any] = {
            "planner_group": group_name,
            "algorithm": algorithm,
            "explicit_algorithm_binding": _in_map(algorithm_map, group_name),
        }
        if readiness is None:
            record.update(
                available=False,
                canonical_name=None,
                tier=None,
                catalog_requires_explicit_opt_in=None,
                requires_explicit_opt_in=None,
                explicit_opt_in_satisfied=False,
                note="unresolved: no algorithm-readiness catalog entry",
            )
            blockers.append(f"planner_unavailable:{group_name}")
        elif readiness.tier == "placeholder":
            record.update(
                available=False,
                canonical_name=readiness.canonical_name,
                tier=readiness.tier,
                catalog_requires_explicit_opt_in=readiness.requires_explicit_opt_in,
                requires_explicit_opt_in=readiness.requires_explicit_opt_in,
                explicit_opt_in_satisfied=False,
                note=readiness.note,
            )
            blockers.append(f"planner_placeholder:{group_name}")
        else:
            opt_in_satisfied = not readiness.requires_explicit_opt_in or _explicit_opt_in_satisfied(
                opt_in_map, group_name
            )
            record.update(
                available=True,
                canonical_name=readiness.canonical_name,
                tier=readiness.tier,
                catalog_requires_explicit_opt_in=readiness.requires_explicit_opt_in,
                requires_explicit_opt_in=readiness.requires_explicit_opt_in
                and not opt_in_satisfied,
                explicit_opt_in_satisfied=opt_in_satisfied,
                note=readiness.note,
            )
            if readiness.requires_explicit_opt_in and not opt_in_satisfied:
                launch_prerequisites.append(
                    f"planner_requires_explicit_opt_in:{group_name} "
                    "(set allow_testing_algorithms in the algo config before launch)"
                )
            if readiness.canonical_name == "orca":
                if not _rvo2_importable():
                    launch_prerequisites.append(f"planner_requires_rvo2:{group_name}")
        resolution.append(record)
    return resolution


def _explicit_opt_in_satisfied(opt_in_map: Mapping[str, Any], group_name: str) -> bool:
    """Return whether a planner group carries the fixed-scope explicit opt-in."""
    raw = opt_in_map.get(group_name)
    if raw is True:
        return True
    if isinstance(raw, Mapping):
        return bool(raw.get("allow_testing_algorithms"))
    return False


def _rvo2_importable() -> bool:
    """Return whether the ORCA runtime dependency is importable."""
    return importlib.util.find_spec("rvo2") is not None


def _check_primary_metric(
    validated: Mapping[str, Any],
    *,
    blockers: list[str],
) -> dict[str, Any]:
    """Validate the primary (ranking) metric is identifiable by contract.

    Fails closed when the ranking metric is absent from the declared metric list
    or is explicitly flagged non-identifiable / zero-variance (issue #3299).

    Returns:
        Primary-metric identifiability record.
    """
    ranking = validated["ranking"]
    primary = str(ranking["metric"])
    metric_entries = {
        str(entry.get("name")): entry
        for entry in validated["metrics"]
        if isinstance(entry, Mapping)
    }
    entry = metric_entries.get(primary)
    present = entry is not None

    non_identifiable_reason: str | None = None
    if not present:
        # validate_fidelity_sensitivity_config already requires membership, but
        # keep the fail-closed branch explicit for defense in depth.
        non_identifiable_reason = "primary_metric_not_in_metrics"
        blockers.append(f"primary_metric_missing:{primary}")
    elif _entry_declares_non_identifiable(entry):
        non_identifiable_reason = _declared_non_identifiable_reason(entry)
        blockers.append(f"primary_metric_non_identifiable:{non_identifiable_reason}")

    direction = None
    if present:
        direction = entry.get("direction")

    return {
        "metric": primary,
        "direction": direction,
        "higher_is_better": bool(ranking.get("higher_is_better", False)),
        "present_in_metrics": present,
        "identifiable_by_contract": non_identifiable_reason is None,
        "non_identifiable_reason": non_identifiable_reason,
        "runtime_identifiability_deferred": (
            "measured zero-variance / degenerate rankings are detected post-run; "
            "see robot_sf/benchmark/fidelity_rank_stability.py"
        ),
    }


def _entry_declares_non_identifiable(entry: Mapping[str, Any]) -> bool:
    """Return whether a metric entry pre-declares itself non-identifiable."""
    if entry.get("identifiable") is False:
        return True
    if entry.get("non_identifiable") is True:
        return True
    return entry.get("expected_variance") == "zero"


def _declared_non_identifiable_reason(entry: Mapping[str, Any]) -> str:
    """Return the declared non-identifiability reason for a metric entry."""
    reason = entry.get("non_identifiable_reason")
    if reason:
        return str(reason)
    return PRIMARY_METRIC_ZERO_VARIANCE_REASON


def _in_map(mapping: Mapping[str, Any] | None, key: str) -> bool:
    return isinstance(mapping, Mapping) and key in mapping


__all__ = [
    "DECISION_BLOCKED",
    "DECISION_READY",
    "EVIDENCE_STATUS",
    "PREFLIGHT_CLAIM_BOUNDARY",
    "REQUIRED_CLAIM_BOUNDARY_PHRASES",
    "SCHEMA_VERSION",
    "build_fixed_scope_preflight",
    "write_fixed_scope_preflight",
]
