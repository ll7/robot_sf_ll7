"""Metric-semantics reconciliation table over an episode event ledger (issue #3482).

The paired protocol that motivated `EpisodeEventLedger.v1` found that an exact environment
collision flag could fire while sampled collision metrics stayed zero. The ledger and its
reconciliation guard (`robot_sf/benchmark/event_ledger.py`) already surface that exact-vs-derived
disagreement instead of silently resolving it. This module adds the remaining integrity artifact:
a **metric-semantics reconciliation table** that, for each reported field, records its producer,
sampling level, kind (exact/sampled/derived), representative downstream consumers, and the audit
result — so every table in the empirical chapter has a single, testable provenance row.

The table is built purely from a ledger payload; it is reproducible and changes no behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from robot_sf.benchmark.event_ledger import reconcile_event_ledger

RECONCILIATION_TABLE_SCHEMA = "event_ledger_reconciliation.v1"

# Representative (not exhaustive) downstream consumers per ledger metric field, for audit
# orientation. Derived from the current benchmark tree; intentionally conservative.
_REPRESENTATIVE_CONSUMERS: dict[str, tuple[str, ...]] = {
    "collision_count": ("map_runner_metrics", "camera_ready_campaign", "aggregate"),
    "near_miss": ("failure_extractor", "hybrid_evidence_matrix", "seed_variance"),
    "clearance_breach": ("aggregate", "hybrid_evidence_matrix"),
    "ttc_breach": ("aggregate",),
    "oscillation": ("safety_predicates", "aggregate"),
    "late_evasive": ("safety_predicates", "aggregate"),
    "occlusion_near_miss": ("safety_predicates", "aggregate"),
}


def build_metric_semantics_table(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Build the metric-semantics reconciliation table for one episode ledger.

    Args:
        ledger: An ``EpisodeEventLedger.v1`` payload (e.g. from ``build_event_ledger``).

    Returns:
        dict[str, Any]: A versioned table with one row per reported metric field
        (field, level, kind, producer, representative consumers), plus the overall audit
        result and any reconciliation violations.
    """
    metric_definitions = ledger.get("metric_definitions")
    metric_definitions = metric_definitions if isinstance(metric_definitions, Mapping) else {}
    reconciliation = ledger.get("reconciliation")
    reconciliation = reconciliation if isinstance(reconciliation, Mapping) else {}
    violations = list(reconcile_event_ledger(ledger))

    rows = []
    for field in sorted(metric_definitions):
        definition = metric_definitions[field]
        definition = definition if isinstance(definition, Mapping) else {}
        rows.append(
            {
                "field": field,
                "level": "episode",
                "kind": str(definition.get("kind", "unknown")),
                "producer": str(definition.get("source", "missing")),
                "representative_consumers": list(_REPRESENTATIVE_CONSUMERS.get(field, ())),
            }
        )

    return {
        "schema_version": RECONCILIATION_TABLE_SCHEMA,
        "ledger_schema_version": ledger.get("schema_version"),
        "audit_result": str(reconciliation.get("audit_result", "unchecked")),
        "reconciles": not violations,
        "violations": violations,
        "rows": rows,
    }


__all__ = [
    "RECONCILIATION_TABLE_SCHEMA",
    "build_metric_semantics_table",
]
