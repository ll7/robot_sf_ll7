#!/usr/bin/env python3
"""Extract topology diagnostic score instrumentation for issue #2307.
Reads JSONL diagnostic traces and emits per-row fields required by the issue.
"""

import json
import sys
from pathlib import Path
from typing import Any


def _topology_payload(record: dict) -> dict:
    """Return the first dict-like topology payload in canonical priority order."""
    for key in ("topology", "topology_instrumentation"):
        payload = record.get(key)
        if isinstance(payload, dict):
            return payload
    return {}


def _score_component_hypothesis(score_components: Any) -> str | None:
    """Select the best fallback hypothesis from score component rows."""
    if not isinstance(score_components, list):
        return None
    dict_components = [row for row in score_components if isinstance(row, dict)]
    if not dict_components:
        return None
    selected = next((row for row in dict_components if row.get("selected") is True), None)
    if selected is not None:
        hypothesis = selected.get("hypothesis")
        return str(hypothesis) if hypothesis is not None else None
    scored = [
        row
        for row in dict_components
        if isinstance(row.get("score"), int | float) and "hypothesis" in row
    ]
    if scored:
        return str(max(scored, key=lambda row: row["score"])["hypothesis"])
    hypothesis = dict_components[0].get("hypothesis")
    return str(hypothesis) if hypothesis is not None else None


def extract_topology(record: dict) -> dict:
    """Extract topology score-selection fields from one diagnostic row."""
    out = {
        "per_frame_hypothesis_count": None,
        "alternative_hypothesis_count": None,
        "selected_hypothesis": None,
        "rejection_reason": None,
        "score_margin_to_primary_route": None,
        "switch_opportunity_count": None,
        "primary_vs_best_alternative_route_distance": None,
        "near_parity_gate_reason": None,
        "selected_static_clearance_min_m": None,
        "best_alternative_static_clearance_min_m": None,
    }
    # heuristics: look for 'topology_instrumentation' or 'score_components'
    topo = _topology_payload(record)
    out["per_frame_hypothesis_count"] = topo.get("per_frame_hypothesis_count")
    out["alternative_hypothesis_count"] = topo.get("alternative_hypothesis_count")
    out["selected_hypothesis"] = topo.get("selected_hypothesis")
    out["rejection_reason"] = topo.get("rejection_reason")
    out["score_margin_to_primary_route"] = topo.get("score_margin_to_primary_route")
    out["switch_opportunity_count"] = topo.get("switch_opportunity_count")
    out["primary_vs_best_alternative_route_distance"] = topo.get(
        "primary_vs_best_alternative_route_distance"
    )
    out["near_parity_gate_reason"] = topo.get("near_parity_gate_reason")
    out["selected_static_clearance_min_m"] = topo.get("selected_static_clearance_min_m")
    out["best_alternative_static_clearance_min_m"] = topo.get(
        "best_alternative_static_clearance_min_m"
    )
    # fallback: inspect score_components list
    if out["selected_hypothesis"] is None and "score_components" in record:
        out["selected_hypothesis"] = _score_component_hypothesis(record.get("score_components"))
    return out


def main() -> None:
    """Run the topology extractor CLI."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(2)
    res = []
    with p.open() as f:
        for i, line in enumerate(f):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out = extract_topology(rec)
            out["_row_index"] = i
            res.append(out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
