#!/usr/bin/env python3
"""Extract topology diagnostic score instrumentation for issue #2307.
Reads JSONL diagnostic traces and emits per-row fields required by the issue.
"""
import sys, json
from pathlib import Path

def extract_topology(record: dict) -> dict:
    out = {
        "per_frame_hypothesis_count": None,
        "alternative_hypothesis_count": None,
        "selected_hypothesis": None,
        "rejection_reason": None,
        "score_margin_to_primary_route": None,
        "switch_opportunity_count": None,
    }
    # heuristics: look for 'topology_instrumentation' or 'score_components'
    topo = record.get("topology") or record.get("topology_instrumentation") or {}
    if isinstance(topo, dict):
        out["per_frame_hypothesis_count"] = topo.get("per_frame_hypothesis_count")
        out["alternative_hypothesis_count"] = topo.get("alternative_hypothesis_count")
        out["selected_hypothesis"] = topo.get("selected_hypothesis")
        out["rejection_reason"] = topo.get("rejection_reason")
        out["score_margin_to_primary_route"] = topo.get("score_margin_to_primary_route")
        out["switch_opportunity_count"] = topo.get("switch_opportunity_count")
    # fallback: inspect score_components list
    if out["selected_hypothesis"] is None and "score_components" in record:
        sc = record.get("score_components")
        if isinstance(sc, list) and sc:
            out["selected_hypothesis"] = sc[0].get("hypothesis") if isinstance(sc[0], dict) else None
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(2)
    res = []
    with p.open() as f:
        for i,line in enumerate(f):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            out = extract_topology(rec)
            out["_row_index"] = i
            res.append(out)
    print(json.dumps(res, indent=2))

if __name__=="__main__":
    main()
