"""Compare #9671 trace episodes with the frozen 0.0.7 release rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

SOURCE_SHA = "07f7e8d43084de748915e1b1eb8b2a1603357c6e"
ARCHIVE_SHA256 = "684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f"
TRACE_KEYS = ("record_forces", "record_planner_decision_trace", "record_simulation_step_trace")
EXPECTED_TUPLES = {
    (planner, scenario, seed)
    for planner in ("goal", "social_force", "orca")
    for scenario in ("classic_head_on_corridor_medium", "classic_group_crossing_medium")
    for seed in (22, 23, 24)
} | {("ppo", "classic_doorway_medium", seed) for seed in (113, 114)}


def _rows(lines: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in lines if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["scenario_params"]["algo"]), str(row["scenario_id"]), int(row["seed"])


def _release_rows(archive: Path, planners: set[str]) -> dict[tuple[str, str, int], dict[str, Any]]:
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.name.endswith("/episodes.jsonl") or "/payload/runs/" not in member.name:
                continue
            planner = member.name.split("/payload/runs/", 1)[1].split("__", 1)[0]
            if planner not in planners:
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read release member {member.name}")
            for row in _rows(stream):
                key = (planner, str(row["scenario_id"]), int(row["seed"]))
                if key in indexed:
                    raise ValueError(f"duplicate release tuple {key}")
                indexed[key] = row
    return indexed


def _validate_steps(key: tuple[str, str, int], trace: dict[str, Any]) -> None:
    """Require robot, pedestrian, and available force state at each recorded step."""
    if not isinstance(trace.get("steps"), list) or not trace["steps"]:
        raise ValueError(f"trace tuple {key} has no per-step state")
    for index, step in enumerate(trace["steps"]):
        robot = step.get("robot") if isinstance(step, dict) else None
        pedestrians = step.get("pedestrians") if isinstance(step, dict) else None
        if not isinstance(robot, dict) or not isinstance(robot.get("position"), list):
            raise ValueError(f"trace tuple {key} step {index} lacks robot state")
        if not isinstance(pedestrians, list) or any(
            not isinstance(ped, dict) or not isinstance(ped.get("position"), list)
            for ped in pedestrians
        ):
            raise ValueError(f"trace tuple {key} step {index} lacks pedestrian state")
        planner = step.get("planner") or {}
        ammv = planner.get("ammv") or {}
        forces = ammv.get("pedestrian_force_vectors")
        if pedestrians and (not isinstance(forces, list) or len(forces) != len(pedestrians)):
            raise ValueError(f"trace tuple {key} step {index} lacks pedestrian forces")


def _compare_row(
    key: tuple[str, str, int], row: dict[str, Any], frozen: dict[str, Any] | None
) -> dict[str, Any]:
    metadata = row.get("algorithm_metadata") or {}
    trace = metadata.get("simulation_step_trace") or {}
    if row.get("git_hash") != SOURCE_SHA:
        raise ValueError(f"trace tuple {key} has wrong source commit")
    _validate_steps(key, trace)
    params = row.get("scenario_params") or {}
    if any(params.get(flag) is not True for flag in TRACE_KEYS):
        raise ValueError(f"trace tuple {key} lacks requested recording flags")
    if frozen is not None:
        frozen_params = dict(frozen["scenario_params"])
        rerun_params = dict(params)
        for flag in TRACE_KEYS:
            frozen_params.pop(flag, None)
            rerun_params.pop(flag, None)
        if frozen_params != rerun_params:
            raise ValueError(f"trace tuple {key} changes release scenario/planner parameters")
    return {
        "planner": key[0],
        "scenario": key[1],
        "seed": key[2],
        "trace_status": row.get("status"),
        "release_status": frozen.get("status") if frozen else None,
        "comparison": (
            "no_release_row"
            if frozen is None
            else "match"
            if row.get("status") == frozen.get("status")
            else "mismatch"
        ),
    }


def check(
    archive: Path,
    traces: list[Path],
    *,
    expected: set[tuple[str, str, int]] | None = None,
    expected_archive_sha256: str | None = ARCHIVE_SHA256,
) -> dict[str, Any]:
    """Return every outcome comparison, including absent release rows."""
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    if expected_archive_sha256 is not None and archive_sha256 != expected_archive_sha256:
        raise ValueError("release archive SHA-256 does not match frozen 0.0.7 identity")
    trace_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    digests: dict[str, str] = {}
    for path in traces:
        data = path.read_bytes()
        digests[str(path)] = hashlib.sha256(data).hexdigest()
        for row in _rows(data.splitlines()):
            key = _key(row)
            if key in trace_rows:
                raise ValueError(f"duplicate trace tuple {key}")
            trace_rows[key] = row
    required = EXPECTED_TUPLES if expected is None else expected
    if set(trace_rows) != required:
        raise ValueError(
            f"trace tuple inventory mismatch: missing={sorted(required - set(trace_rows))}; "
            f"extra={sorted(set(trace_rows) - required)}"
        )
    release = _release_rows(archive, {key[0] for key in trace_rows})
    comparisons = [
        _compare_row(key, row, release.get(key)) for key, row in sorted(trace_rows.items())
    ]
    return {
        "schema_version": "issue_9671_trace_release_comparison.v1",
        "source_commit": SOURCE_SHA,
        "release_archive_sha256": archive_sha256,
        "trace_inputs_sha256": digests,
        "comparisons": comparisons,
    }


def main() -> int:
    """Validate the diagnostic traces and write a deterministic comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-archive", required=True, type=Path)
    parser.add_argument("--traces", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = check(args.release_archive, args.traces)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
