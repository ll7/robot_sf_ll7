"""Scan verified release rows in serial scenario shards, without running episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from robot_sf.analysis_workbench.audit_queue import QueueCandidate, QueueDataset
from robot_sf.analysis_workbench.audit_scan import scan_campaign
from robot_sf.analysis_workbench.release_row_anomalies import analyze_release_rows
from robot_sf.analysis_workbench.release_row_bundle import load_release_rows


def write_json(path: Path, payload) -> str:
    """Write a new strict JSON artifact and return its SHA-256."""
    raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def main(argv=None) -> int:
    """Verify one bundle and retain scan shards plus an aggregate queue/receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new private artifact directory"
    )
    args = parser.parse_args(argv)
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
    ]
    producer_digests = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in producer_paths
    }
    rows, source = load_release_rows(args.bundle)
    gate = analyze_release_rows(rows, source=source, config={"require_preflight": False})
    files = {"release-row-gate.json": write_json(args.output_dir / "release-row-gate.json", gate)}
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario_id"]].append(row)
    counts = Counter()
    inventory = Counter()
    candidates = []
    for index, (scenario, group) in enumerate(sorted(groups.items())):
        report = scan_campaign(
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
