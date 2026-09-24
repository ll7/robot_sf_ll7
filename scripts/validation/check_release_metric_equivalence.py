#!/usr/bin/env python3
"""Fail closed when a successor campaign changes a predecessor episode metric.

The predecessor is an immutable, SHA-pinned publication archive. The successor is
an unpacked campaign root containing ``runs/<arm>/episodes.jsonl``. The command
compares all predecessor metric fields and episode outcomes by arm, scenario, and
seed; new metric fields in the successor are allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(row: dict[str, Any]) -> str:
    provenance = row.get("result_provenance")
    provenance_sha = provenance.get("repo_commit") if isinstance(provenance, dict) else None
    git_sha = row.get("git_hash")
    if provenance_sha and git_sha and provenance_sha != git_sha:
        raise ValueError("row has conflicting source commits")
    source = provenance_sha or git_sha
    if not isinstance(source, str) or len(source) != 40:
        raise ValueError("row has no full source commit")
    return source


def _key(arm: str, row: dict[str, Any]) -> tuple[str, str, int]:
    scenario = row.get("scenario_id")
    seed = row.get("seed")
    if not isinstance(scenario, str) or not scenario or type(seed) is not int:
        raise ValueError("row needs scenario_id and integer seed")
    return arm, scenario, seed


def _legacy_view(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("row has no metrics object")
    return {
        "metrics": metrics,
        "metric_values": row.get("metric_values"),
        "outcome": row.get("outcome"),
        "status": row.get("status"),
        "steps": row.get("steps"),
    }


def _read_archive(
    archive: Path, expected_source: str
) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    with tarfile.open(archive, "r:gz") as handle:
        members = sorted(
            (
                member
                for member in handle.getmembers()
                if member.isfile()
                and "/payload/runs/" in member.name
                and member.name.endswith("/episodes.jsonl")
            ),
            key=lambda member: member.name,
        )
        if not members:
            raise ValueError("archive contains no campaign episode rows")
        for member in members:
            arm = Path(member.name).parent.name.removesuffix("__differential_drive")
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read {member.name}")
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if _source(row) != expected_source:
                    raise ValueError(f"archive source mismatch at {member.name}:{line_number}")
                key = _key(arm, row)
                if key in rows:
                    raise ValueError(f"duplicate archive identity: {key}")
                rows[key] = _legacy_view(row)
    return rows


def _read_candidate(root: Path, expected_source: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    paths = sorted((root / "runs").glob("*/episodes.jsonl"))
    if not paths:
        raise ValueError("candidate contains no runs/*/episodes.jsonl")
    for path in paths:
        arm = path.parent.name.removesuffix("__differential_drive")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if _source(row) != expected_source:
                    raise ValueError(f"candidate source mismatch at {path}:{line_number}")
                key = _key(arm, row)
                if key in rows:
                    raise ValueError(f"duplicate candidate identity: {key}")
                rows[key] = _legacy_view(row)
    return rows


def _numbers_equal(old: int | float, new: int | float, *, tolerance: float) -> bool:
    """Compare finite numbers by tolerance and legacy unavailable sentinels by class."""
    if math.isnan(old) or math.isnan(new):
        return math.isnan(old) and math.isnan(new)
    if math.isinf(old) or math.isinf(new):
        return old == new
    return abs(old - new) <= tolerance


def _differences(old: Any, new: Any, path: str, *, tolerance: float) -> list[str]:
    if isinstance(old, dict):
        if not isinstance(new, dict):
            return [path]
        result = [f"{path}.{key}" for key in old.keys() - new.keys()]
        for key in sorted(old.keys() & new.keys()):
            result.extend(_differences(old[key], new[key], f"{path}.{key}", tolerance=tolerance))
        return result
    if isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            return [path]
        result = []
        for index, (left, right) in enumerate(zip(old, new, strict=True)):
            result.extend(_differences(left, right, f"{path}[{index}]", tolerance=tolerance))
        return result
    if isinstance(old, bool) or isinstance(new, bool):
        return [] if type(old) is type(new) and old == new else [path]
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        # Frozen releases use NaN as an explicit unavailable sentinel in a few
        # legacy fields. Preserve its class exactly instead of treating NaN == NaN
        # as a numerical comparison or making a self-comparison fail.
        return [] if _numbers_equal(old, new, tolerance=tolerance) else [path]
    return [] if type(old) is type(new) and old == new else [path]


def compare(
    baseline: dict[tuple[str, str, int], dict[str, Any]],
    candidate: dict[tuple[str, str, int], dict[str, Any]],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Return every missing, extra, or changed predecessor field by episode identity."""
    mismatches: list[dict[str, Any]] = []
    for key in sorted(baseline.keys() | candidate.keys()):
        if key not in baseline:
            paths = ["unexpected_identity"]
        elif key not in candidate:
            paths = ["missing_identity"]
        else:
            paths = _differences(baseline[key], candidate[key], "episode", tolerance=tolerance)
        if paths:
            mismatches.append(
                {"arm": key[0], "scenario_id": key[1], "seed": key[2], "paths": paths}
            )
    return {
        "status": "pass" if not mismatches else "mismatch",
        "baseline_rows": len(baseline),
        "candidate_rows": len(candidate),
        "paired_rows": len(baseline.keys() & candidate.keys()),
        "mismatch_episodes": len(mismatches),
        "mismatches": mismatches,
        "absolute_tolerance": tolerance,
    }


def main() -> int:
    """Verify a pinned archive against one exact-source successor campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--baseline-source-sha", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-source-sha", required=True)
    parser.add_argument("--expected-rows", type=int, default=20160)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.baseline_archive) != args.baseline_sha256:
        raise ValueError("baseline archive SHA-256 mismatch")
    baseline = _read_archive(args.baseline_archive, args.baseline_source_sha)
    candidate = _read_candidate(args.candidate_root, args.candidate_source_sha)
    report = compare(baseline, candidate)
    report.update(
        {
            "baseline_archive_sha256": args.baseline_sha256,
            "baseline_source_sha": args.baseline_source_sha,
            "candidate_source_sha": args.candidate_source_sha,
            "expected_rows": args.expected_rows,
        }
    )
    if len(baseline) != args.expected_rows or len(candidate) != args.expected_rows:
        report["status"] = "identity_count_mismatch"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{report['status']}: {report['paired_rows']} paired; {report['mismatch_episodes']} mismatches"
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
