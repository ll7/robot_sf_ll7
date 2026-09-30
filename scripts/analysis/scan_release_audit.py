#!/usr/bin/env python3
"""Audit recorded campaign rows against their own frozen arm and cell inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from robot_sf.analysis_workbench.release_row_anomalies import (
    ReleaseRowError,
    analyze_release_rows,
    render_markdown,
)
from robot_sf.benchmark.identity.hash_utils import read_jsonl
from scripts.tools.analyze_camera_ready_campaign import _resolve_safe_campaign_path


def scan_campaign(campaign_root: Path, *, config: dict | None = None) -> dict:
    """Read campaign bytes without stepping a simulator or inferring missing arms."""
    root = campaign_root.resolve()
    manifest_bytes = (root / "campaign_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    summary = json.loads((root / "reports/campaign_summary.json").read_text())
    preview = json.loads((root / "preflight/preview_scenarios.json").read_text())
    if preview.get("truncated") is not False:
        raise ReleaseRowError("audit requires a complete campaign scenario inventory")
    roster = [planner["key"] for planner in manifest["planners"] if planner.get("enabled", True)]
    if not roster or len(roster) != len(set(roster)):
        raise ReleaseRowError("campaign manifest must declare unique enabled arms")
    resolved_seeds = manifest["seed_policy"]["resolved_seeds"]
    cells = []
    for scenario in preview["scenarios"]:
        scenario_id = scenario.get("id", scenario.get("name"))
        for seed in scenario.get("seeds", resolved_seeds):
            cells.append({"scenario_id": scenario_id, "seed": seed, "invalid_run": False})
    if not cells:
        raise ReleaseRowError("campaign manifest/preview declares no expected cells")
    rows = []
    for run in summary["runs"]:
        arm = run["planner"]["key"]
        raw_path = run.get("episodes_path")
        if not raw_path:
            continue  # Missing arms remain in the manifest denominator.
        path = _resolve_safe_campaign_path(root, raw_path, label="episodes_path")
        rows.extend({**row, "_release_arm": arm} for row in read_jsonl(path))
    return analyze_release_rows(
        rows,
        source={
            "planner_ids": roster,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        preflight=cells,
        config=config,
    )


def main(argv: list[str] | None = None) -> int:
    """Emit diagnostic JSON/Markdown and optionally enforce the anomaly gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--release-gate", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text()) if args.config else None
    report = scan_campaign(args.campaign_root, config=config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "release_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "release_audit.md").write_text(render_markdown(report))
    return int(args.release_gate and report["gate"]["blocked"])


if __name__ == "__main__":
    raise SystemExit(main())
