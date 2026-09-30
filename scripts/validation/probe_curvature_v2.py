"""Recompute D-055 curvature locally from recorded dev1001 episode traces.

No planner/environment is imported or stepped by this probe. The output is a
measurement receipt, not a calibration asset: frozen SNQI K anchors are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from robot_sf.benchmark.metric_definitions import LEGACY_METRIC_SCHEMA_VERSION
from robot_sf.benchmark.metrics import EpisodeData, curvature_mean
from robot_sf.evidence.writers import sha256_file, write_json


def _load_dev_rows(root: Path, cohort: str) -> tuple[dict, list]:
    """Read seed-1001 rows and bind their source files to digests.

    Returns:
        Identity-indexed rows and file provenance; other seeds are never computed.
    """
    rows, sources = {}, []
    for path in sorted(root.glob("runs/*/episodes.jsonl")):
        raw = path.read_bytes()
        sources.append(
            {
                "cohort": cohort,
                "arm": path.parent.name,
                "artifact_path": path.relative_to(root).as_posix(),
                "location": f"local://{root.name}/{path.relative_to(root).as_posix()}",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        for line in raw.splitlines():
            row = json.loads(line)
            if row["seed"] != 1001:
                continue
            key = (path.parent.name, row["scenario_id"])
            if key in rows:
                raise ValueError(f"duplicate {cohort} identity: {key}")
            rows[key] = row
    return rows, sources


def _extreme_diagnostics(data: EpisodeData) -> dict:
    """Explain large geometric curvature from counted displacement directions.

    Returns:
        Trace-level length and direction-change diagnostics.
    """
    displacement = np.diff(np.vstack([data.initial_robot_pos, data.robot_pos]), axis=0)
    lengths = np.linalg.norm(displacement, axis=1)
    counted = lengths >= 1e-3
    angles = np.arctan2(displacement[counted, 1], displacement[counted, 0])
    turns = np.abs((np.diff(angles) + np.pi) % (2 * np.pi) - np.pi)
    return {
        "counted_steps": int(counted.sum()),
        "counted_length_m": float(lengths[counted].sum()),
        "total_turn_rad": float(turns.sum()),
        "turns_over_pi_2": int((turns > np.pi / 2).sum()),
        "median_counted_step_m": float(np.median(lengths[counted])),
        "explanation": "Frequent direction changes on counted >=1 mm steps; not subthreshold creeping.",
    }


def _measure_row(key: tuple, row: dict, original: dict) -> tuple[dict, bool, float]:
    """Recompute both definitions and check the recorded companion values.

    Returns:
        Episode values, exact historical-scalar equality and relative roundoff.
    """
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    positions = np.asarray([s["robot"]["position"] for s in trace["steps"]], dtype=float)
    if len(positions) != row["steps"] or not np.isfinite(positions).all():
        raise ValueError("incomplete/nonfinite position trace")
    data = EpisodeData(
        robot_pos=positions,
        robot_vel=np.zeros_like(positions),
        robot_acc=np.zeros_like(positions),
        peds_pos=np.empty((len(positions), 0, 2)),
        ped_forces=np.empty((len(positions), 0, 2)),
        goal=np.zeros(2),
        dt=trace["dt"],
    )
    old = curvature_mean(data, metric_schema_version=LEGACY_METRIC_SCHEMA_VERSION)
    data.initial_robot_pos = np.asarray(trace["reset"]["robot"]["position"], dtype=float)
    if not np.isfinite(data.initial_robot_pos).all():
        raise ValueError("nonfinite reset position")
    new = curvature_mean(data)
    if any(row[field] != original[field] for field in ("steps", "termination_reason")):
        raise ValueError(f"rehearsal outcome mismatch: {key}")
    recorded = row["metrics"]["curvature_mean"]
    if recorded != original["metrics"]["curvature_mean"] or not np.isclose(
        old, recorded, rtol=1e-14, atol=0
    ):
        raise ValueError(f"old curvature does not reproduce recorded scalars: {key}")
    if not np.isfinite(new) or new < 0:
        raise ValueError(f"invalid v2 curvature: {key}")
    record = {
        "arm": key[0],
        "scenario_id": key[1],
        "seed": 1001,
        "old": old,
        "new": new,
        "steps": len(positions),
    }
    if new > 10:
        record["extreme_trace"] = _extreme_diagnostics(data)
    return record, old == recorded, abs(old - recorded) / max(abs(recorded), 1e-300)


def measure(trace_root: Path, rehearsal_root: Path) -> dict:
    """Measure only seed 1001 and verify the rehearsal companion row identities.

    Returns:
        Source digests, per-episode values, arm quantiles and extreme diagnostics.
    """
    rehearsal, rehearsal_sources = _load_dev_rows(rehearsal_root, "rehearsal")
    traces, trace_sources = _load_dev_rows(trace_root, "traces")
    if rehearsal.keys() != traces.keys() or len(traces) != 672:
        raise ValueError(f"expected complete 672-episode dev1001 cohort; got {len(traces)}")
    records, exact_matches, relative_errors = [], [], []
    grouped = defaultdict(list)
    for key, row in traces.items():
        record, exact, error = _measure_row(key, row, rehearsal[key])
        records.append(record)
        exact_matches.append(exact)
        relative_errors.append(error)
        grouped[key[0]].append(record)
    arms = []
    for arm, rows in sorted(grouped.items()):
        summary = {"arm": arm, "n": len(rows)}
        for version in ("old", "new"):
            values = [r[version] for r in rows]
            summary[version] = dict(
                zip(
                    ("median", "p95", "max"),
                    map(float, np.quantile(values, [0.5, 0.95, 1])),
                    strict=True,
                )
            )
        arms.append(summary)
    return {
        "decision": "D-055",
        "seed": 1001,
        "episodes": len(records),
        "trace_source_commits": sorted({r["git_hash"] for r in traces.values()}),
        "old_exact_matches": sum(exact_matches),
        "old_max_relative_error": max(relative_errors),
        "new_includes_reset": True,
        "sources": rehearsal_sources + trace_sources,
        "arms": arms,
        "new_pooled_p95": float(np.quantile([r["new"] for r in records], 0.95)),
        "extremes": [r for r in records if r["new"] > 10],
        "records": records,
    }


def main() -> None:
    """Write a reproducible receipt from local recorded rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--rehearsal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Compact marked summary.")
    parser.add_argument(
        "--full-output",
        type=Path,
        required=True,
        help="Local full receipt, outside the checkout's evidence directory.",
    )
    args = parser.parse_args()
    if "docs/context/evidence" in args.full_output.resolve().as_posix():
        raise ValueError("keep the full per-episode receipt outside the tracked evidence tree")
    result = measure(args.trace_root, args.rehearsal_root)
    write_json(args.full_output, result)
    summary = {key: value for key, value in result.items() if key != "records"}
    summary["full_receipt"] = {
        "artifact_path": args.full_output.name,
        "location": f"local://{args.full_output.parent.name}/{args.full_output.name}",
        "sha256": sha256_file(args.full_output),
        "size_bytes": args.full_output.stat().st_size,
        "record_count": len(result["records"]),
        "custody": "Local diagnostic receipt only; no public or durable external custody claimed.",
    }
    write_json(args.output, summary, catalog_area="benchmark_evidence")
    print(
        f"episodes={result['episodes']}; pooled p95={result['new_pooled_p95']:.9g}; extremes={len(result['extremes'])}"
    )


if __name__ == "__main__":
    main()
