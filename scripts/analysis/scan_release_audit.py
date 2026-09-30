"""Scan verified release rows in serial scenario shards, without running episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from robot_sf.analysis_workbench.audit_queue import QueueCandidate, QueueDataset
from robot_sf.analysis_workbench.audit_scan import scan_campaign as scan_audit_campaign
from robot_sf.analysis_workbench.release_row_anomalies import (
    ReleaseRowError,
    _release_0_0_8_roster,
    analyze_release_rows,
    render_markdown,
)
from robot_sf.analysis_workbench.release_row_bundle import load_release_rows
from robot_sf.benchmark.identity.hash_utils import read_jsonl
from scripts.tools.analyze_camera_ready_campaign import _resolve_safe_campaign_path


def write_json(path: Path, payload) -> str:
    """Write a new strict JSON artifact and return its SHA-256."""
    raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load_campaign_rows(campaign_root: Path) -> tuple[list[dict], dict, list[dict]]:
    """Read the manifest denominator and row bytes without simulation."""
    root = campaign_root.resolve()
    manifest_bytes = (root / "campaign_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    summary = json.loads((root / "reports/campaign_summary.json").read_text())
    preview = json.loads((root / "preflight/preview_scenarios.json").read_text())
    if preview.get("truncated") is not False:
        raise ReleaseRowError("audit requires a complete campaign scenario inventory")
    roster = sorted(
        planner["key"] for planner in manifest["planners"] if planner.get("enabled", True)
    )
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
        rows.extend(
            {
                **row,
                "_release_arm": arm,
                "_source_member": f"payload/runs/{arm}__differential_drive/episodes.jsonl",
            }
            for row in read_jsonl(path)
        )
    source = {"planner_ids": roster, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    return rows, source, cells


def analyze_rows(rows, source, *, preflight=None, config=None, release_gate=False) -> dict:
    """Bind release admission to the committed roster independently of the input."""
    expected = _release_0_0_8_roster() if release_gate else None
    declared = set(source.get("planner_ids", []))
    reason = None
    if release_gate:
        if expected is None:
            reason = "release_roster_source_unavailable"
        elif declared != expected:
            reason = "release_roster_source_mismatch"
    analysis_source = source
    if reason == "release_roster_source_mismatch":
        # Admit rows from the independent denominator even if the manifest lost
        # an arm but still references its run; report the mismatch as BLOCKED.
        analysis_source = {
            **source,
            "manifest_planner_ids": sorted(declared),
            "planner_ids": sorted(declared | expected),
        }
    report = analyze_release_rows(rows, source=analysis_source, preflight=preflight, config=config)
    if release_gate:
        report["release_roster_binding"] = {
            "source": "committed_0_0_8_template",
            "expected_planner_ids": sorted(expected) if expected is not None else None,
            "matches": reason is None,
        }
        if reason:
            report["gate"]["blocked"] = True
            report["gate"]["reasons"] = sorted(set(report["gate"]["reasons"]) | {reason})
    return report


def scan_campaign(campaign_root: Path, *, config=None, release_gate=False) -> dict:
    """Audit a complete campaign inventory; preserve the original Python entrypoint."""
    rows, source, preflight = load_campaign_rows(campaign_root)
    return analyze_rows(rows, source, preflight=preflight, config=config, release_gate=release_gate)


def main(argv=None) -> int:
    """Verify one bundle and retain scan shards plus an aggregate queue/receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bundle", type=Path)
    inputs.add_argument("--campaign-root", type=Path)
    parser.add_argument("--config", type=Path, help="release anomaly configuration JSON")
    parser.add_argument(
        "--release-gate",
        action="store_true",
        help="bind the committed 0.0.8 roster and exit 1 if blocked",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new private artifact directory"
    )
    args = parser.parse_args(argv)
    try:
        return run_scan(args)
    except Exception as error:  # noqa: BLE001 - CLI boundary maps every crash to exit 2.
        print(f"release audit failed: {error}", file=sys.stderr)
        return 2


def run_scan(args) -> int:
    """Write #9997 scan shards, queue, and receipt for either verified input kind."""
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    repo = Path(__file__).resolve().parents[2]
    producer_paths = [
        "scripts/analysis/scan_release_audit.py",
        "robot_sf/analysis_workbench/audit_detectors.py",
        "robot_sf/analysis_workbench/audit_scan.py",
        "robot_sf/analysis_workbench/audit_queue.py",
        "robot_sf/analysis_workbench/audit_release_adapter.py",
        "robot_sf/analysis_workbench/release_row_anomalies.py",
        "robot_sf/analysis_workbench/release_row_bundle.py",
        "scripts/tools/analyze_camera_ready_campaign.py",
        "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml",
    ]
    producer_digests = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in producer_paths
    }
    config = json.loads(args.config.read_text()) if args.config else None
    if args.bundle:
        rows, source = load_release_rows(args.bundle)
        preflight = None
        if config is None:
            config = {"require_preflight": False}
    else:
        rows, source, preflight = load_campaign_rows(args.campaign_root)
    gate = analyze_rows(
        rows, source, preflight=preflight, config=config, release_gate=args.release_gate
    )
    files = {"release-row-gate.json": write_json(args.output_dir / "release-row-gate.json", gate)}
    (args.output_dir / "release-row-gate.md").write_text(render_markdown(gate))
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario_id"]].append(row)
    counts = Counter()
    inventory = Counter()
    candidates = []
    for index, (scenario, group) in enumerate(sorted(groups.items())):
        report = scan_audit_campaign(
            {
                "schema_version": "episode-jsonl.v1",
                "episodes": group,
                "release_bundle_provenance": source,
                "campaign_id": "release-" + source["manifest_sha256"],
                "config_identity": "publication-manifest-" + source["manifest_sha256"],
            }
        )
        counts.update((signal.detector_id, signal.status) for signal in report.signals)
        inventory.update(item.status for item in report.inventory)
        name = f"scan-{index:02d}.json"
        files[name] = write_json(args.output_dir / name, report.to_dict())
        by_episode = defaultdict(list)
        for signal in report.signals:
            by_episode[signal.episode_id].append(signal)
        for episode in report.episode_refs:
            candidate = QueueCandidate(
                episode,
                signals=tuple(by_episode[episode.execution_id]),
                metadata={"report_episode_id": episode.execution_id},
            )
            candidates.append(candidate.to_dict())
        print(f"{index + 1}/{len(groups)} {scenario}: {len(group)} rows", flush=True)
    queue = {
        "schema_version": "audit-queue-input.v1",
        "candidates": candidates,
        "accounting": {"expected_rows": len(rows), "indexed_rows": sum(inventory.values())},
    }
    # Validate the same portable input that the queue CLI will read.
    QueueDataset.from_mapping(queue)
    files["queue.json"] = write_json(args.output_dir / "queue.json", queue)
    if producer_digests != {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in producer_paths
    }:
        raise RuntimeError("producer source changed during scan; receipt withheld")
    receipt = {
        "producer_files_sha256": producer_digests,
        "schema_version": "release-audit-scan-receipt.v1",
        "status": "complete",
        "claim_boundary": "offline diagnostic only; no preflight or trace adjudication",
        "source": source,
        "rows": len(rows),
        "shards": len(groups),
        "processes": 1,
        "inventory": dict(inventory),
        "gate_counts": gate["counts"]["by_detector"],
        "scan_counts": {
            detector: {
                status: counts[detector, status]
                for status in ("flagged", "clear", "error", "unavailable")
            }
            for detector in sorted({key[0] for key in counts})
        },
        "seconds": round(time.monotonic() - start, 2),
        "files_sha256": files,
    }
    write_json(args.output_dir / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "rows": receipt["rows"],
                "seconds": receipt["seconds"],
                "inventory": receipt["inventory"],
            }
        )
    )
    return int(args.release_gate and gate["gate"]["blocked"])


if __name__ == "__main__":
    raise SystemExit(main())
