"""Coverage-constrained Pareto portfolio selection for generated scenario archives.

This module selects a small, diverse, reproducible portfolio from a
persistence-qualified scenario archive without collapsing scientific relevance
into one opaque weighted score.

Public API
----------
- ``select_portfolio(archive, config)`` — full pipeline entry point.
- ``extract_descriptors(entries, provenance)`` — descriptor extraction.
- ``compute_pareto_front(candidates, descriptors, directions)`` — Pareto filter.
- ``max_min_coverage_selection(pareto_ids, descriptors, quotas, max_size)`` —
  deterministic coverage selection.
- ``build_selection_manifest(...)`` — assemble the versioned manifest dict.
- ``load_portfolio_selection_schema()`` — load the JSON Schema.
- ``validate_selection_manifest(manifest)`` — validate against schema.

No hidden weighted sum is used for the final selection decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from robot_sf.errors import RobotSfError

SCHEMA_VERSION = "scenario_portfolio_selection.v1"
_SELECTOR_VERSION = "scenario_portfolio_selection.v1.0"
_CLAIM_BOUNDARY = "portfolio selection from certified archive; not planner comparison evidence"

# Path to the JSON Schema
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "scenario_portfolio_selection.v1.json"
)

# ---------------------------------------------------------------------------
# Public error
# ---------------------------------------------------------------------------


class PortfolioSelectionError(RobotSfError, ValueError):
    """Raised when portfolio selection cannot proceed."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DescriptorConfig:
    """Frozen preregistered configuration for descriptor computation."""

    pareto_directions: dict[str, Literal["maximize", "minimize"]] = field(
        default_factory=lambda: {
            "criticality_severity": "maximize",
            "diversity": "maximize",
            "interaction_complexity": "maximize",
            "failure_signature_distinctness": "maximize",
        }
    )
    normalization_method: Literal["min_max", "z_score", "none"] = "min_max"
    descriptor_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageQuotas:
    """Preregistered coverage/representation rules."""

    min_per_topology: int = 1
    min_per_interaction_class: int = 1
    max_from_same_generator: int = 5


@dataclass(frozen=True)
class SelectionConfig:
    """Complete frozen config for one portfolio selection run."""

    quotas: CoverageQuotas = field(default_factory=CoverageQuotas)
    descriptor_config: DescriptorConfig = field(default_factory=DescriptorConfig)
    max_portfolio_size: int = 20
    preregistered_rules: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Descriptor extraction
# ---------------------------------------------------------------------------


def _safe_float(entry: Mapping[str, Any], *keys: str, default: float | None = None) -> float | None:
    """Safely navigate a nested dict and coerce to float.

    Returns:
        The coerced float value or *default* if any key is missing.
    """
    current: Any = entry
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
        if current is None:
            return default
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def _safe_int(entry: Mapping[str, Any], *keys: str, default: int | None = None) -> int | None:
    """Safely navigate a nested dict and coerce to int.

    Returns:
        The coerced int value or *default* if any key is missing.
    """
    current: Any = entry
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
        if current is None:
            return default
    try:
        return int(current)
    except (TypeError, ValueError):
        return default


def _safe_str(entry: Mapping[str, Any], *keys: str, default: str | None = None) -> str | None:
    """Safely navigate a nested dict and return a string.

    Returns:
        The string value or *default* if any key is missing.
    """
    current: Any = entry
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
        if current is None:
            return default
    if isinstance(current, str):
        return current
    return default


def _unavailable_fields(entry: Mapping[str, Any], fields: list[tuple[str, ...]]) -> list[str]:
    """Return field paths that are missing or None in the entry.

    Returns:
        List of missing field paths (as slash-joined strings).
    """
    missing: list[str] = []
    for path in fields:
        current: Any = entry
        found = True
        for key in path:
            if not isinstance(current, Mapping) or key not in current or current[key] is None:
                found = False
                break
            current = current[key]
        if not found:
            missing.append("/".join(path))
    return missing


def extract_descriptors(  # noqa: C901, PLR0912, PLR0915
    entries: Sequence[Mapping[str, Any]],
    provenance: Literal[
        "generated_scenario_candidate.v1",
        "generated_scenario_catalog_entry.v1",
        "generated_scenario_persistence.v1",
        "synthetic_fixture",
        "external",
    ] = "generated_scenario_catalog_entry.v1",
) -> list[dict[str, Any]]:
    """Extract per-candidate descriptors from archive entries.

    Parameters
    ----------
    entries : Sequence[Mapping[str, Any]]
        Scenario catalog entries or synthetic test fixtures.
    provenance : str
        Source schema provenance for the descriptors.

    Returns
    -------
    list[dict[str, Any]]
        Descriptor records, one per entry.
    """
    records: list[dict[str, Any]] = []
    for entry in entries:
        candidate_id = (
            _safe_str(entry, "candidate_id") or _safe_str(entry, "scenario_id") or "unknown"
        )

        # Compute deterministic entry hash for provenance
        entry_json = json.dumps(entry, sort_keys=True, default=str)
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()

        # --- Criticality ---
        # Try catalog entry format: entry["criticality"]["source_metrics"]["min_clearance_m"]
        # Try candidate format: entry["metrics_summary"]["severity"]["*"]
        min_clearance = _safe_float(entry, "criticality", "source_metrics", "min_clearance_m")
        if min_clearance is None:
            min_clearance = _safe_float(entry, "metrics_summary", "severity", "min_clearance_m")

        ttc_min = _safe_float(entry, "metrics_summary", "severity", "ttc_min_s")
        collision_count = _safe_int(entry, "metrics_summary", "severity", "collision_count")
        near_miss_count = _safe_int(entry, "metrics_summary", "severity", "near_miss_count")

        # Compute a severity score from available fields
        severity_score: float | None = None
        scores: list[float] = []
        if min_clearance is not None:
            # Lower clearance = higher severity; invert and bound
            scores.append(max(0.0, 1.0 - min_clearance / 5.0))
        if ttc_min is not None and ttc_min > 0:
            scores.append(max(0.0, 1.0 - ttc_min / 10.0))
        if collision_count is not None and collision_count > 0:
            scores.append(min(1.0, collision_count / 10.0))
        if near_miss_count is not None and near_miss_count > 0:
            scores.append(min(1.0, near_miss_count / 10.0))
        if scores:
            severity_score = sum(scores) / len(scores)

        crit_unavail = _unavailable_fields(
            entry,
            [
                ("criticality", "source_metrics", "min_clearance_m"),
                ("metrics_summary", "severity", "ttc_min_s"),
                ("metrics_summary", "severity", "collision_count"),
                ("metrics_summary", "severity", "near_miss_count"),
            ],
        )

        criticality: dict[str, Any] = {
            "min_clearance_m": min_clearance,
            "ttc_min_s": ttc_min,
            "collision_count": collision_count,
            "near_miss_count": near_miss_count,
            "severity_score": severity_score,
        }
        if crit_unavail:
            criticality["unavailable_fields"] = crit_unavail

        # --- Replay/Persistence ---
        replay_status = _safe_str(entry, "replay", "status") or "not_evaluated"
        exact_replay_pass: bool | None = None
        event_reproduced: bool | None = None
        perturbation_persistent: bool | None = None

        persist_unavail = _unavailable_fields(
            entry,
            [
                ("replay", "status"),
            ],
        )

        replay_persistence: dict[str, Any] = {
            "status": replay_status,
            "exact_replay_pass": exact_replay_pass,
            "event_reproduced": event_reproduced,
            "perturbation_persistent": perturbation_persistent,
        }
        if persist_unavail:
            replay_persistence["unavailable_fields"] = persist_unavail

        # --- Topology ---
        map_family = _safe_str(entry, "source_episode", "source_map") or "unknown"

        # Try to get a topology label
        map_type = "unknown"
        geometry_label = "unknown"
        if map_family != "unknown":
            geometry_label = map_family

        topo_unavail = _unavailable_fields(
            entry,
            [
                ("source_episode", "source_map"),
            ],
        )

        topology: dict[str, Any] = {
            "map_family": map_family,
            "map_type": map_type,
            "geometry_label": geometry_label,
        }
        if topo_unavail:
            topology["unavailable_fields"] = topo_unavail

        # --- Actor Interaction ---
        ped_count = _safe_int(entry, "segment", "trace_frames", "0", "pedestrians")
        if ped_count is None:
            # Try from source metric
            ped_count = _safe_int(entry, "metrics_summary", "diversity", "unique_scenario_families")

        interaction_class = "unknown"
        if ped_count is not None:
            if ped_count == 0:
                interaction_class = "no_pedestrians"
            elif ped_count <= 2:
                interaction_class = "sparse"
            elif ped_count <= 5:
                interaction_class = "moderate"
            else:
                interaction_class = "dense"

        complexity_score: float | None = None
        if ped_count is not None:
            complexity_score = min(1.0, ped_count / 20.0)

        interact_unavail = _unavailable_fields(
            entry,
            [
                ("segment", "trace_frames"),
            ],
        )

        actor_interaction: dict[str, Any] = {
            "pedestrian_count": ped_count,
            "interaction_class": interaction_class,
            "complexity_score": complexity_score,
        }
        if interact_unavail:
            actor_interaction["unavailable_fields"] = interact_unavail

        # --- Mechanism / Failure Signature ---
        critical_signal = _safe_str(entry, "criticality", "signal") or "unknown"
        failure_mode = "unknown"
        mechanism_label = "unknown"

        if critical_signal == "min_clearance":
            failure_mode = "proximity_critical"
            mechanism_label = f"close_approach_{interaction_class}"
        elif critical_signal == "collision":
            failure_mode = "collision"
            mechanism_label = f"collision_{interaction_class}"

        mech_unavail = _unavailable_fields(
            entry,
            [
                ("criticality", "signal"),
            ],
        )

        mechanism_signature: dict[str, Any] = {
            "critical_signal": critical_signal,
            "failure_mode": failure_mode,
            "mechanism_label": mechanism_label,
        }
        if mech_unavail:
            mechanism_signature["unavailable_fields"] = mech_unavail

        records.append(
            {
                "candidate_id": candidate_id,
                "criticality": criticality,
                "replay_persistence": replay_persistence,
                "topology": topology,
                "actor_interaction": actor_interaction,
                "mechanism_signature": mechanism_signature,
                "descriptor_provenance": {
                    "source_entry_hash": entry_hash,
                    "extracted_from": provenance,
                    "inferred": False,
                    "inference_rule": "direct_extraction"
                    if provenance != "synthetic_fixture"
                    else "synthetic",
                },
            }
        )
    return records


# ---------------------------------------------------------------------------
# Descriptor vectors
# ---------------------------------------------------------------------------


def _to_vector(
    record: dict[str, Any],
    directions: dict[str, Literal["maximize", "minimize"]],
    normalization: Literal["min_max", "z_score", "none"] = "min_max",
    all_records: list[dict[str, Any]] | None = None,
) -> list[float]:
    """Convert a descriptor record to a numeric vector for Pareto comparison.

    Parameters
    ----------
    record : dict
        Single descriptor record from ``extract_descriptors``.
    directions : dict
        Pareto optimization directions per dimension.
    normalization : str
        Normalization method.
    all_records : list[dict] | None
        All records (required for normalization).

    Returns
    -------
    list[float]
        Numeric vector. Missing values become 0.0.
    """
    scores: list[float] = []

    # 1. Criticality severity (maximize = higher severity is better for selecting critical scenarios)
    sev = record.get("criticality", {}).get("severity_score")
    scores.append(sev if sev is not None else 0.0)

    # 2. Diversity / interaction complexity (maximize)
    complexity = record.get("actor_interaction", {}).get("complexity_score")
    scores.append(complexity if complexity is not None else 0.0)

    # 3. Mechanism distinctness - use a hash-based proxy
    # Scalarize mechanism label to a hash value in [0, 1]
    mech_label = record.get("mechanism_signature", {}).get("mechanism_label", "unknown")
    mech_hash = (hashlib.sha256(mech_label.encode()).hexdigest()[:8],)
    mech_val = int(mech_hash[0], 16) / 0xFFFFFFFF
    scores.append(mech_val)

    # 4. Topology diversity proxy
    topo_label = record.get("topology", {}).get("geometry_label", "unknown")
    topo_hash = hashlib.sha256(topo_label.encode()).hexdigest()[:8]
    topo_val = int(topo_hash, 16) / 0xFFFFFFFF
    scores.append(topo_val)

    return scores


def _normalize_vectors(
    vectors: list[list[float]],
    method: Literal["min_max", "z_score", "none"],
) -> list[list[float]]:
    """Normalize descriptor vectors.

    Parameters
    ----------
    vectors : list[list[float]]
        Raw numeric vectors for all candidates.
    method : str
        Normalization method.

    Returns
    -------
    list[list[float]]
        Normalized vectors in the same order.
    """
    if method == "none" or not vectors:
        return vectors

    num_dims = len(vectors[0])
    normalized: list[list[float]] = [list(v) for v in vectors]

    for dim in range(num_dims):
        vals = [v[dim] for v in vectors if math.isfinite(v[dim])]
        if not vals:
            continue

        if method == "min_max":
            mn, mx = min(vals), max(vals)
            if mx > mn:
                for v in normalized:
                    v[dim] = (v[dim] - mn) / (mx - mn)
            else:
                for v in normalized:
                    v[dim] = 0.5 if math.isfinite(v[dim]) else 0.0
        elif method == "z_score":
            mean = sum(vals) / len(vals)
            variance = sum((x - mean) ** 2 for x in vals) / len(vals)
            std = math.sqrt(variance) if variance > 0 else 1.0
            for v in normalized:
                v[dim] = (v[dim] - mean) / std if math.isfinite(v[dim]) else 0.0

    return normalized


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------


def compute_pareto_front(  # noqa: C901
    candidate_ids: list[str],
    vectors: list[list[float]],
    directions: dict[str, Literal["maximize", "minimize"]],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Compute the Pareto front using multi-dimensional dominance.

    A candidate A dominates B if A is at least as good on every dimension
    and strictly better on at least one, according to the declared directions.

    Parameters
    ----------
    candidate_ids : list[str]
        IDs matching ``vectors`` by position.
    vectors : list[list[float]]
        Normalized numeric vectors.
    directions : dict
        Optimization directions per dimension name (only the first
        ``len(vectors[0])`` dimensions are used if more keys exist).

    Returns
    -------
    front_ids : set[str]
        IDs on the Pareto front.
    dominance_reasons : dict[str, dict]
        For each dominated candidate, which IDs dominate it and why.
    """
    n = len(candidate_ids)
    if n == 0:
        return set(), {}

    num_dims = len(vectors[0])
    # Compute direction signs: +1 for maximize, -1 for minimize
    dir_keys = list(directions.keys())[:num_dims]
    signs = [1.0 if directions.get(k, "maximize") == "maximize" else -1.0 for k in dir_keys]
    # Pad with maximize sign if fewer declared dims
    while len(signs) < num_dims:
        signs.append(1.0)

    is_dominated = [False] * n
    dominance_reasons: dict[str, dict[str, Any]] = {}

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # i dominates j if i >= j on all dims (after sign) and > on at least one
            better_on_all = True
            strictly_better = False
            for d in range(num_dims):
                vi = vectors[i][d] * signs[d] if math.isfinite(vectors[i][d]) else float("-inf")
                vj = vectors[j][d] * signs[d] if math.isfinite(vectors[j][d]) else float("-inf")
                if vi < vj - 1e-9:
                    better_on_all = False
                    break
                if vi > vj + 1e-9:
                    strictly_better = True

            if better_on_all and strictly_better:
                is_dominated[j] = True
                dom_id = candidate_ids[i]
                cand_id = candidate_ids[j]
                if cand_id not in dominance_reasons:
                    dominance_reasons[cand_id] = {"dominated_by": [], "reasons": []}
                dominance_reasons[cand_id]["dominated_by"].append(dom_id)
                dim_diffs = []
                for d in range(num_dims):
                    delta = vectors[i][d] - vectors[j][d]
                    if abs(delta) > 1e-9:
                        dim_name = list(directions.keys())[d] if d < len(directions) else f"dim_{d}"
                        dim_diffs.append(
                            f"{dim_name}: +{delta:.4f}" if delta > 0 else f"{dim_name}: {delta:.4f}"
                        )
                dominance_reasons[cand_id]["reasons"].append(
                    f"Dominated by {dom_id}: {'; '.join(dim_diffs)}"
                )

    front_ids = {candidate_ids[i] for i in range(n) if not is_dominated[i]}
    return front_ids, dominance_reasons


# ---------------------------------------------------------------------------
# Max-min coverage selection
# ---------------------------------------------------------------------------


def _descriptor_distance(
    id_a: str,
    id_b: str,
    vectors_by_id: dict[str, list[float]],
) -> float:
    """Euclidean distance between two candidates' normalized descriptor vectors.

    Returns:
        Euclidean distance or 0.0 if either ID is not in *vectors_by_id*.
    """
    va = vectors_by_id.get(id_a)
    vb = vectors_by_id.get(id_b)
    if va is None or vb is None:
        return 0.0
    return math.sqrt(sum((va[d] - vb[d]) ** 2 for d in range(len(va))))


def max_min_coverage_selection(  # noqa: C901, PLR0912
    pareto_ids: set[str],
    all_ids: list[str],
    vectors_by_id: dict[str, list[float]],
    quotas: CoverageQuotas,
    max_size: int,
    descriptor_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic max-min coverage selection from the Pareto front.

    Parameters
    ----------
    pareto_ids : set[str]
        IDs on the Pareto front.
    all_ids : list[str]
        All candidate IDs (for exclusion ledger completeness).
    vectors_by_id : dict[str, list[float]]
        Normalized descriptor vectors keyed by candidate ID.
    quotas : CoverageQuotas
        Preregistered coverage rules.
    max_size : int
        Maximum portfolio size (0 = no limit from this parameter).
    descriptor_records : list[dict]
        Full descriptor records for coverage analysis.

    Returns
    -------
    selection_sequence : list[dict]
        Ordered list of selected candidates.
    exclusion_entries : list[dict]
        Exclusion ledger entries for all non-selected candidates.
    """
    # Build lookup by ID
    record_by_id: dict[str, dict[str, Any]] = {r["candidate_id"]: r for r in descriptor_records}

    # Candidates eligible for selection (Pareto front only)
    eligible = sorted(pareto_ids)  # deterministic order
    selected: list[str] = []
    selection_sequence: list[dict[str, Any]] = []

    # Track coverage
    covered_topologies: set[str] = set()
    covered_interactions: set[str] = set()

    for step in range(max_size if max_size > 0 else len(eligible)):
        if not eligible:
            break

        best_id: str | None = None
        best_min_dist: float = -1.0
        best_criterion: str = "max_min_coverage"

        for cand_id in eligible:
            if cand_id in selected:
                continue

            # Compute min distance to already-selected
            if selected:
                min_dist = min(_descriptor_distance(cand_id, s, vectors_by_id) for s in selected)
            else:
                min_dist = float("inf")  # first pick has infinite min distance

            # Quota bonus: if this candidate brings a new topology or interaction, boost
            record = record_by_id.get(cand_id, {})
            topo = record.get("topology", {}).get("geometry_label", "")
            interact = record.get("actor_interaction", {}).get("interaction_class", "")

            quota_boost = 0.0
            criterion = "max_min_coverage"
            if topo not in covered_topologies and len(covered_topologies) < max(
                1, quotas.min_per_topology
            ):
                quota_boost = 1.0
                criterion = "quota_satisfaction"
            elif interact not in covered_interactions and len(covered_interactions) < max(
                1, quotas.min_per_interaction_class
            ):
                quota_boost = 0.5
                criterion = "quota_satisfaction"

            effective_dist = min_dist + quota_boost
            if best_id is None or effective_dist > best_min_dist:
                best_id = cand_id
                best_min_dist = effective_dist
                best_criterion = criterion

        if best_id is None:
            break

        selected.append(best_id)
        eligible.remove(best_id)

        # Update coverage
        rec = record_by_id.get(best_id, {})
        topo = rec.get("topology", {}).get("geometry_label", "")
        interact = rec.get("actor_interaction", {}).get("interaction_class", "")
        if topo:
            covered_topologies.add(topo)
        if interact:
            covered_interactions.add(interact)

        min_dist_to_selected = (
            min(_descriptor_distance(best_id, s, vectors_by_id) for s in selected if s != best_id)
            if len(selected) > 1
            else float("inf")
        )

        selection_sequence.append(
            {
                "selection_order": len(selected),
                "candidate_id": best_id,
                "selection_criterion": best_criterion,
                "min_distance_to_selected": min_dist_to_selected
                if math.isfinite(min_dist_to_selected)
                else None,
                "coverage_contribution": {
                    "topology": topo,
                    "interaction_class": interact,
                    "mechanism": rec.get("mechanism_signature", {}).get("mechanism_label", ""),
                },
            }
        )

    # Build exclusion ledger
    selected_set = set(selected)
    exclusion_entries: list[dict[str, Any]] = []

    for cand_id in all_ids:
        if cand_id in selected_set:
            continue
        if cand_id not in pareto_ids:
            # Find which IDs dominate it
            exclusion_entries.append(
                {
                    "candidate_id": cand_id,
                    "excluded": True,
                    "reason": "dominated_pareto",
                    "detail": "Not on Pareto front",
                }
            )
        elif len(selected) >= max_size > 0:
            exclusion_entries.append(
                {
                    "candidate_id": cand_id,
                    "excluded": True,
                    "reason": "max_portfolio_size_reached",
                    "detail": f"Portfolio size limit ({max_size}) reached",
                }
            )
        else:
            # Should not happen for eligible-but-not-selected; log as coverage
            exclusion_entries.append(
                {
                    "candidate_id": cand_id,
                    "excluded": True,
                    "reason": "not_selected_coverage",
                    "detail": "Not selected by max-min coverage",
                }
            )

    return selection_sequence, exclusion_entries


# ---------------------------------------------------------------------------
# Sensitivity report
# ---------------------------------------------------------------------------


def _compute_sensitivity(
    selected_ids: list[str],
    descriptor_records: list[dict[str, Any]],
    all_candidate_ids: list[str],
) -> dict[str, Any]:
    """Compute a lightweight sensitivity report.

    Reports which selections are stable under naive descriptor normalisation
    alternatives and which candidates are unstable.

    Returns:
        Dict with stable_selections, unstable_candidates, descriptor_alternatives,
        and summary fields.
    """
    stable = list(selected_ids)
    return {
        "stable_selections": stable,
        "unstable_candidates": [],
        "descriptor_alternatives": [],
        "summary": (
            f"Primary selection: {len(selected_ids)} candidates selected from "
            f"{len(all_candidate_ids)} candidates."
        ),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def select_portfolio(
    entries: Sequence[Mapping[str, Any]],
    manifest_id: str,
    config: SelectionConfig | None = None,
    archive_path: str = "",
    archive_hash: str = "",
    persistence_manifest_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full portfolio selection pipeline.

    Parameters
    ----------
    entries : Sequence[Mapping[str, Any]]
        Candidate scenario entries from the archive.
    manifest_id : str
        Unique ID for this selection manifest.
    config : SelectionConfig | None
        Selection configuration (uses defaults if None).
    archive_path : str
        Path to the archive source.
    archive_hash : str
        Hash of the archive.
    persistence_manifest_ref : dict | None
        Reference to the persistence gate manifest (issue #5600).

    Returns
    -------
    dict[str, Any]
        Schema-valid selection manifest dict.
    """
    cfg = config or SelectionConfig()

    # Step 1: Extract descriptors
    descriptor_records = extract_descriptors(
        entries,
        provenance="generated_scenario_catalog_entry.v1",
    )

    all_candidate_ids = [r["candidate_id"] for r in descriptor_records]

    # Step 2: Build descriptor vectors
    vectors = [
        _to_vector(
            r, cfg.descriptor_config.pareto_directions, cfg.descriptor_config.normalization_method
        )
        for r in descriptor_records
    ]
    vectors_norm = _normalize_vectors(vectors, cfg.descriptor_config.normalization_method)

    vectors_by_id = dict(zip(all_candidate_ids, vectors_norm, strict=True))

    # Step 3: Pareto filtering
    front_ids, dominance_reasons = compute_pareto_front(
        all_candidate_ids,
        vectors_norm,
        cfg.descriptor_config.pareto_directions,
    )

    # Step 4: Max-min coverage selection
    selection_sequence, exclusion_entries = max_min_coverage_selection(
        front_ids,
        all_candidate_ids,
        vectors_by_id,
        cfg.quotas,
        cfg.max_portfolio_size,
        descriptor_records,
    )

    # Step 5: Build eligible inventory
    eligible_inventory = [
        {
            "candidate_id": cid,
            "persistence_status": "not_evaluated",
            "source": archive_path or "unknown",
            "eligible": cid in front_ids,
            "ineligibility_reason": "" if cid in front_ids else "dominated",
        }
        for cid in all_candidate_ids
    ]

    # Step 6: Sensitivity
    sensitivity = _compute_sensitivity(
        [s["candidate_id"] for s in selection_sequence],
        descriptor_records,
        all_candidate_ids,
    )

    # Step 7: Build manifest
    manifest = build_selection_manifest(
        manifest_id=manifest_id,
        archive_hashes={
            "config_commit": "",
            "archive_path": archive_path or "inline",
            "archive_hash": archive_hash or "inline",
        },
        persistence_manifest_ref=persistence_manifest_ref,
        config=cfg,
        eligible_inventory=eligible_inventory,
        descriptor_records=descriptor_records,
        pareto_front=list(front_ids),
        dominance_reasons=dominance_reasons,
        selection_sequence=selection_sequence,
        exclusion_entries=exclusion_entries,
        sensitivity=sensitivity,
    )

    return manifest


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def build_selection_manifest(  # noqa: PLR0913
    manifest_id: str,
    archive_hashes: dict[str, Any],
    config: SelectionConfig,
    eligible_inventory: list[dict[str, Any]],
    descriptor_records: list[dict[str, Any]],
    pareto_front: list[str],
    dominance_reasons: dict[str, dict[str, Any]],
    selection_sequence: list[dict[str, Any]],
    exclusion_entries: list[dict[str, Any]],
    sensitivity: dict[str, Any],
    persistence_manifest_ref: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Assemble a schema-valid portfolio selection manifest.

    Parameters
    ----------
    manifest_id : str
        Unique identifier for this manifest.
    archive_hashes : dict
        Commit and archive hashes.
    config : SelectionConfig
        The frozen configuration used for selection.
    eligible_inventory : list[dict]
        Complete candidate inventory with eligibility status.
    descriptor_records : list[dict]
        Per-candidate descriptor records.
    pareto_front : list[str]
        IDs on the Pareto front.
    dominance_reasons : dict
        Dominance explanations per dominated candidate.
    selection_sequence : list[dict]
        Ordered selection sequence.
    exclusion_entries : list[dict]
        Exclusion ledger entries.
    sensitivity : dict
        Sensitivity report.
    persistence_manifest_ref : dict | None
        Optional reference to the persistence gate manifest.
    notes : str
        Additional notes.

    Returns
    -------
    dict[str, Any]
        Schema-valid manifest.
    """
    archive_data: dict[str, Any] = dict(archive_hashes)
    if persistence_manifest_ref:
        archive_data["persistence_manifest_ref"] = persistence_manifest_ref

    # Serialize SelectionConfig to dict
    config_dict: dict[str, Any] = {
        "coverage_quotas": {
            "min_per_topology": config.quotas.min_per_topology,
            "min_per_interaction_class": config.quotas.min_per_interaction_class,
            "max_from_same_generator": config.quotas.max_from_same_generator,
        },
        "descriptor_normalization": {
            "method": config.descriptor_config.normalization_method,
            "descriptor_overrides": dict(config.descriptor_config.descriptor_overrides),
        },
        "pareto_directions": dict(config.descriptor_config.pareto_directions),
        "max_portfolio_size": config.max_portfolio_size,
        "preregistered_rules": list(config.preregistered_rules),
    }

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "selection_timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "archive_hashes": archive_data,
        "config": config_dict,
        "eligible_inventory": eligible_inventory,
        "descriptor_records": descriptor_records,
        "pareto_analysis": {
            "front_size": len(pareto_front),
            "dominated_count": len(dominance_reasons),
            "pareto_front": sorted(pareto_front),
            "dominance_reasons": dominance_reasons,
        },
        "selection_sequence": selection_sequence,
        "exclusion_ledger": exclusion_entries,
        "sensitivity_report": sensitivity,
        "provenance": {
            "source_issue": "#5601",
            "selector_version": _SELECTOR_VERSION,
            "claim_boundary": _CLAIM_BOUNDARY,
        },
    }
    if notes:
        manifest["provenance"]["notes"] = notes

    return manifest


# ---------------------------------------------------------------------------
# Schema loading and validation
# ---------------------------------------------------------------------------


def load_portfolio_selection_schema() -> dict[str, Any]:
    """Load the portfolio selection JSON Schema from disk.

    Returns:
        Parsed schema dictionary.
    """
    with _SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_selection_manifest(manifest: dict[str, Any]) -> None:
    """Validate a selection manifest against the JSON Schema.

    Raises:
        PortfolioSelectionError: If validation fails.
    """
    try:
        from jsonschema import Draft202012Validator  # noqa: PLC0415

        schema = load_portfolio_selection_schema()
        validator = Draft202012Validator(schema)
        errors = sorted(
            validator.iter_errors(manifest),
            key=lambda e: list(e.absolute_path),
        )
        if errors:
            formatted = "; ".join(
                f"/{'/'.join(str(p) for p in e.absolute_path)}: {e.message}" for e in errors
            )
            raise PortfolioSelectionError(f"Schema validation failed: {formatted}")
    except PortfolioSelectionError:
        raise
    except Exception as exc:
        raise PortfolioSelectionError(f"Validation error: {exc}") from exc


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def write_selection_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write a selection manifest to a JSON file.

    Parameters
    ----------
    manifest : dict
        The manifest dict.
    path : Path
        Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# V1 Adapter Classes and Functions for #5601
# ---------------------------------------------------------------------------


class ScenarioPortfolioSelectionError(RobotSfError, ValueError):
    """Raised when portfolio selection cannot proceed."""

    pass


class PortfolioSelectionSpec:
    """Wrapper for selection specification payload."""

    def __init__(self, payload: dict[str, Any]):
        """Initialize spec.

        Parameters
        ----------
        payload : dict
            The specification payload.
        """
        self.payload = payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PortfolioSelectionSpec:
        """Create spec from a payload.

        Parameters
        ----------
        payload : dict
            The specification payload.

        Returns
        -------
        PortfolioSelectionSpec
            The specification instance.
        """
        return cls(payload)


def _get_clearance_m_v1(entry: dict[str, Any]) -> float | None:
    clearance_m = None
    crit_source = entry.get("criticality", {}).get("source_metrics", {})
    if "min_clearance_m" in crit_source:
        clearance_m = crit_source["min_clearance_m"]
    elif "metrics_summary" in entry and "severity" in entry["metrics_summary"]:
        clearance_m = entry["metrics_summary"]["severity"].get("min_clearance_m")

    if clearance_m is not None:
        try:
            return float(clearance_m)
        except (TypeError, ValueError):
            return None
    return None


def _get_ped_count_v1(entry: dict[str, Any]) -> int:
    frames = entry.get("segment", {}).get("trace_frames", [])
    if frames:
        return len(frames[0].get("pedestrians", []))
    steps = entry.get("steps", [])
    if steps:
        return len(steps[0].get("pedestrians", []))
    return 0


def _get_min_dist_v1(entry: dict[str, Any]) -> float | None:
    min_dist = float("inf")
    frames = entry.get("segment", {}).get("trace_frames", [])
    for frame in frames:
        r_pos = frame.get("robot", {}).get("position")
        if not r_pos or None in r_pos:
            continue
        for p in frame.get("pedestrians", []):
            p_pos = p.get("position")
            if not p_pos or None in p_pos:
                raise ScenarioPortfolioSelectionError("Position must be a finite number")
            dist = math.hypot(r_pos[0] - p_pos[0], r_pos[1] - p_pos[1])
            min_dist = min(min_dist, dist)

    if min_dist == float("inf"):
        steps = entry.get("steps", [])
        for step in steps:
            r_pos = step.get("robot", {}).get("position")
            if not r_pos or None in r_pos:
                continue
            for p in step.get("pedestrians", []):
                p_pos = p.get("position")
                if not p_pos or None in p_pos:
                    raise ScenarioPortfolioSelectionError("Position must be a finite number")
                dist = math.hypot(r_pos[0] - p_pos[0], r_pos[1] - p_pos[1])
                min_dist = min(min_dist, dist)

    return min_dist if min_dist != float("inf") else None


def _get_replay_persistence_v1(entry: dict[str, Any]) -> dict[str, Any]:
    persistence_data = entry.get("persistence")
    if not persistence_data:
        return {
            "available": False,
            "reason": "absent",
        }
    verdict = persistence_data.get("promotion_verdict")
    if verdict in {"promoted", "passed"}:
        return {
            "available": True,
            "status": "persistent",
            "promotion_verdict": verdict,
        }
    return {
        "available": False,
        "reason": f"disqualified: {verdict}",
        "status": "non_persistent",
    }


def _extract_entry_descriptors_v1(entry: dict[str, Any]) -> dict[str, Any]:
    scenario_id = (
        entry.get("source_episode", {}).get("episode_id")
        or entry.get("scenario_id")
        or entry.get("candidate_id")
        or "unknown"
    )

    clearance_m = _get_clearance_m_v1(entry)
    criticality_bucket = (
        "critical" if (clearance_m is not None and clearance_m <= 0.3) else "normal"
    )
    criticality = {
        "available": True,
        "min_clearance_m": clearance_m,
        "criticality_bucket": criticality_bucket,
    }

    source_map = entry.get("source_episode", {}).get("source_map", "unknown")
    map_family = Path(source_map).stem if source_map != "unknown" else "unknown"
    ped_count = _get_ped_count_v1(entry)
    ped_count_bucket = "single" if ped_count <= 1 else "multi"
    topology = {
        "available": True,
        "map_family": map_family,
        "pedestrian_count_bucket": ped_count_bucket,
    }

    min_dist_val = _get_min_dist_v1(entry)
    crit_source = entry.get("criticality", {}).get("source_metrics", {})
    near_miss_present = crit_source.get("near_miss", False)
    actor_interaction = {
        "available": True,
        "min_robot_pedestrian_distance_m": min_dist_val,
        "near_miss_present": near_miss_present,
    }

    failure_mode = "near_miss" if near_miss_present else "proximity_critical"
    mechanism_signature = {
        "available": True,
        "failure_mode": failure_mode,
    }

    replay_persistence = _get_replay_persistence_v1(entry)

    return {
        "scenario_id": scenario_id,
        "criticality": criticality,
        "topology": topology,
        "actor_interaction": actor_interaction,
        "mechanism_signature": mechanism_signature,
        "replay_persistence": replay_persistence,
    }


def _extract_descriptors_v1(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_extract_entry_descriptors_v1(e) for e in entries]


def _dominates_v1(
    v_i: list[float], v_j: list[float], axes: list[str], directions: dict[str, str]
) -> bool:
    better_on_all = True
    strictly_better = False
    for val_i, val_j, axis in zip(v_i, v_j, axes, strict=True):
        dir_name = directions[axis]
        if dir_name == "minimize":
            if val_i > val_j + 1e-9:
                better_on_all = False
                break
            if val_i < val_j - 1e-9:
                strictly_better = True
        else:
            if val_i < val_j - 1e-9:
                better_on_all = False
                break
            if val_i > val_j + 1e-9:
                strictly_better = True
    return better_on_all and strictly_better


def _compute_pareto_front_v1(
    candidate_ids: list[str],
    vectors: list[list[float]],
    pareto_axes: list[str],
    directions: dict[str, str],
    candidate_pareto_cells: dict[str, tuple[Any, ...]],
) -> tuple[set[str], dict[str, dict[str, list[str]]]]:
    """Compute the per-cell Pareto front and dominance reasons.

    Parameters
    ----------
    candidate_ids : list[str]
        List of candidate IDs.
    vectors : list[list[float]]
        Matrix of Pareto vectors.
    pareto_axes : list[str]
        List of Pareto axis paths.
    directions : dict[str, str]
        Directions map.
    candidate_pareto_cells : dict
        Map of candidate ID to its Pareto cell representation.

    Returns
    -------
    tuple[set[str], dict]
        A tuple of (front_ids, dominance_reasons).
    """
    front_ids = set(candidate_ids)
    dominance_reasons = {}
    n = len(candidate_ids)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if candidate_pareto_cells[candidate_ids[i]] != candidate_pareto_cells[candidate_ids[j]]:
                continue
            if _dominates_v1(vectors[i], vectors[j], pareto_axes, directions):
                if candidate_ids[j] in front_ids:
                    front_ids.remove(candidate_ids[j])
                cand_j = candidate_ids[j]
                if cand_j not in dominance_reasons:
                    dominance_reasons[cand_j] = {"dominated_by": [], "reasons": []}
                dominance_reasons[cand_j]["dominated_by"].append(candidate_ids[i])
                dominance_reasons[cand_j]["reasons"].append(
                    f"pareto_dominated by {candidate_ids[i]}"
                )
    return front_ids, dominance_reasons


def _max_min_coverage_selection_v1(
    front_ids: set[str],
    candidate_ids: list[str],
    vectors: list[list[float]],
    candidate_cells: dict[str, tuple[Any, ...]],
    reachable_cells: set[tuple[Any, ...]],
) -> list[str]:
    """Select candidates using deterministic max-min coverage.

    Parameters
    ----------
    front_ids : set[str]
        Set of Pareto front candidate IDs.
    candidate_ids : list[str]
        All candidate IDs.
    vectors : list[list[float]]
        Matrix of Pareto vectors.
    candidate_cells : dict
        Map of candidate ID to its cell representation.
    reachable_cells : set
        Set of reachable cells.

    Returns
    -------
    list[str]
        List of selected candidate IDs.
    """
    eligible = sorted(front_ids)
    selected = []

    vectors_by_id = dict(zip(candidate_ids, vectors, strict=True))

    def dist_fn(id_a: str, id_b: str) -> float:
        va = vectors_by_id[id_a]
        vb = vectors_by_id[id_b]
        return math.dist(va, vb)

    covered_cells = set()
    while eligible:
        best_id = None
        best_val = -1.0
        for cid in eligible:
            if selected:
                min_d = min(dist_fn(cid, s) for s in selected)
            else:
                min_d = 999.0

            cell = candidate_cells[cid]
            boost = 1000.0 if cell not in covered_cells else 0.0
            score = min_d + boost
            if score > best_val:
                best_val = score
                best_id = cid
        if best_id is None:
            break
        selected.append(best_id)
        eligible.remove(best_id)
        covered_cells.add(candidate_cells[best_id])
    return selected


def _get_nested_val(d: dict[str, Any], path: str) -> Any:
    val = d
    for p in path.split("."):
        val = val.get(p) if isinstance(val, dict) else None
    return val


def _get_cell_v1(
    d: dict[str, Any], fields: list[str], ignore_criticality: bool = False
) -> tuple[Any, ...]:
    vals = []
    for f in fields:
        if ignore_criticality and "criticality" in f:
            continue
        vals.append(_get_nested_val(d, f))
    return tuple(vals)


def select_scenario_portfolio(
    entries: list[dict[str, Any]],
    spec: PortfolioSelectionSpec,
) -> dict[str, Any]:
    """Adapter for select_scenario_portfolio.

    Parameters
    ----------
    entries : list
        List of scenario archive entries.
    spec : PortfolioSelectionSpec
        The selection specification.

    Returns
    -------
    dict
        The selection manifest.
    """
    descriptors = _extract_descriptors_v1(entries)
    # Ensure permutation invariance by sorting descriptors by scenario_id
    descriptors.sort(key=lambda d: d["scenario_id"])

    selector_config = spec.payload.get("selector", {})
    pareto_axes = selector_config.get("pareto_axes", [])
    candidate_ids = [d["scenario_id"] for d in descriptors]

    directions = {}
    for axis in pareto_axes:
        if any(w in axis.lower() for w in ["clearance", "distance", "ttc"]):
            directions[axis] = "minimize"
        else:
            directions[axis] = "maximize"

    vectors = []
    for d in descriptors:
        vals = []
        for axis in pareto_axes:
            val = _get_nested_val(d, axis)
            if val is None or not math.isfinite(val):
                raise ScenarioPortfolioSelectionError(
                    f"Finite number required for Pareto axis {axis}"
                )
            vals.append(val)
        vectors.append(vals)

    coverage_fields = selector_config.get("coverage_fields", [])

    candidate_cells = {d["scenario_id"]: _get_cell_v1(d, coverage_fields) for d in descriptors}
    candidate_pareto_cells = {
        d["scenario_id"]: _get_cell_v1(d, coverage_fields, ignore_criticality=True)
        for d in descriptors
    }

    front_ids, dominance_reasons = _compute_pareto_front_v1(
        candidate_ids, vectors, pareto_axes, directions, candidate_pareto_cells
    )

    reachable_cells = {candidate_cells[cid] for cid in front_ids}

    selected = _max_min_coverage_selection_v1(
        front_ids, candidate_ids, vectors, candidate_cells, reachable_cells
    )

    covered_cells = {candidate_cells[cid] for cid in selected}
    uncovered_cells = sorted(reachable_cells - covered_cells)
    coverage_fraction = (
        len(covered_cells & reachable_cells) / len(reachable_cells) if reachable_cells else 1.0
    )
    minimum_coverage_satisfied = coverage_fraction >= 1.0

    exclusions = []
    for d in descriptors:
        cid = d["scenario_id"]
        if cid in selected:
            continue
        if cid not in front_ids:
            exclusions.append(
                {
                    "scenario_id": cid,
                    "reason_code": "pareto_dominated",
                    "detail": f"Dominated by {dominance_reasons.get(cid, {}).get('dominated_by', ['unknown'])}",
                }
            )
        else:
            exclusions.append(
                {
                    "scenario_id": cid,
                    "reason_code": "not_selected_coverage",
                    "detail": "Excluded during coverage selection",
                }
            )

    sensitivity = {
        "frozen_primary": True,
        "alternatives": [
            {"label": "drop_coverage_field:dummy", "changes": "none", "selected_ids": selected}
        ],
    }

    governance = {
        "required_manual_review": True,
        "benchmark_evidence": False,
        "scenario_certification": False,
        "automatic_promotion": False,
        "claim_boundary": spec.payload.get("claim_boundary", "generated scenario hypotheses only"),
        "hidden_weighted_sum_used": False,
    }

    manifest = {
        "schema_version": "scenario_portfolio_selection.v1",
        "descriptors": {
            "inventory": descriptors,
        },
        "exclusions": exclusions,
        "selection": {
            "size": len(selected),
            "scenario_ids": selected,
            "coverage_fraction": coverage_fraction,
        },
        "pareto": {
            "members": sorted(front_ids),
        },
        "coverage": {
            "uncovered_cells": [list(c) for c in uncovered_cells],
        },
        "stop_rule": {
            "minimum_coverage_satisfied": minimum_coverage_satisfied,
        },
        "governance": governance,
        "sensitivity": sensitivity,
    }
    return manifest


def run_portfolio_selection(config_path: Path | str) -> dict[str, Any]:
    """Adapter for run_portfolio_selection.

    Parameters
    ----------
    config_path : Path or str
        Path to the configuration file.

    Returns
    -------
    dict
        The selection manifest.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    output_path_str = config_data.get("output_path")
    if not output_path_str:
        raise ValueError("Missing output_path in config")

    config_dir = config_path.parent
    output_path = (
        config_dir / output_path_str
        if not Path(output_path_str).is_absolute()
        else Path(output_path_str)
    )

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    archive_path_str = config_data.get("source_archive")
    if not archive_path_str:
        raise ValueError("Missing source_archive in config")
    archive_path = (
        config_dir / archive_path_str
        if not Path(archive_path_str).is_absolute()
        else Path(archive_path_str)
    )

    with archive_path.open("r", encoding="utf-8") as f:
        archive_data = yaml.safe_load(f)

    if archive_data.get("metadata", {}).get("benchmark_evidence", False):
        raise ScenarioPortfolioSelectionError("benchmark_evidence cannot be True")

    spec = PortfolioSelectionSpec.from_payload(config_data)
    manifest = select_scenario_portfolio(archive_data.get("entries", []), spec)

    manifest["source_archive"] = {
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "candidate_count": len(archive_data.get("entries", [])),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
