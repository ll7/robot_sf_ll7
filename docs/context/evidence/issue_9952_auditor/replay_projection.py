"""Diagnostic comparator: python replay_projection.py BUNDLE NEW_OUTPUT_DIR.

Run with the intended checkout on PYTHONPATH; never runs planner episodes.
"""

import collections
import json
import math
import sys
import time
from pathlib import Path

from robot_sf.analysis_workbench.audit_detectors import default_registry
from robot_sf.analysis_workbench.audit_scan import scan_campaign
from robot_sf.analysis_workbench.release_row_bundle import load_release_rows

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=False)
start = time.monotonic()
rows, source = load_release_rows(archive)
groups = collections.defaultdict(list)
# Identical diagnostic projection on both revisions. No invented trace or
# initial-state equivalence; preserve outcome and finite scalar metrics only.
for raw in rows:
    metrics = {
        k: v
        for k, v in raw["metrics"].items()
        if isinstance(v, (bool, int, float)) and (not isinstance(v, float) or math.isfinite(v))
    }
    row = {
        "episode_id": raw["_release_arm"] + "::" + raw["episode_id"],
        "planner_id": raw["_release_arm"],
        "scenario_id": raw["scenario_id"],
        "seed": raw["seed"],
        "config": {"config_id": raw["algorithm_metadata"]["config_hash"]},
        "outcome": raw["outcome"],
        "steps": raw["steps"],
        "metrics": metrics,
        "source_commit": raw["git_hash"],
        "campaign_id": "published-release-0.0.7",
    }
    groups[row["scenario_id"]].append(row)
counts = collections.Counter()
findings = []
for n, (scenario, group) in enumerate(sorted(groups.items())):
    report = scan_campaign(
        {
            "schema_version": "episode-jsonl.v1",
            "episodes": group,
            "config_identity": "publication-manifest-" + source["manifest_sha256"],
        }
    )
    counts.update((s.detector_id, s.status) for s in report.signals)
    for s in report.signals:
        if s.status == "flagged":
            findings.append(
                {
                    "episode_id": s.episode_id,
                    "detector_id": s.detector_id,
                    "measured": dict(s.measured),
                    "reason": s.reason_code,
                }
            )
    print(n + 1, scenario, flush=True)
out = {
    "rows": len(rows),
    "projection": "identical finite scalar/outcome replay; status failure omitted to isolate detectors; no initial-state/trace invented",
    "scan_counts": {
        d: {status: counts[d, status] for status in ("flagged", "clear", "error", "unavailable")}
        for d in default_registry().ids
    },
    "seconds": round(time.monotonic() - start, 2),
}
(destination / "replay.json").write_text(json.dumps(out, indent=2))
(destination / "findings.json").write_text(json.dumps(findings))
