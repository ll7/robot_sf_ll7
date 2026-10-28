"""Compare a reconstructed run with archived raw and full-precision extraction.

The reference can contain more episodes than the rerun. See README.md for the
exact comparison and provenance contract. No simulator is invoked.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ORIGINAL_COMMIT = "a4a31e3575a130bbc0dfccec7fbc173ff114fa42"
KEY_FIELDS = ("config", "cell", "seed")
EPISODE_PROVENANCE = {"git_hash", "source_file"}
RAW_SCIENTIFIC_FIELDS = (
    "algo",
    "scenario_id",
    "seed",
    "status",
    "termination_reason",
    "outcome",
    "metrics",
    "metric_values",
    "interaction_exposure",
    "safety_predicates",
    "spawn_validity",
    "steps",
    "horizon",
)
RAW_VOLATILE_FIELDS = (
    "git_hash",
    "config_hash",
    "timestamps",
    "timing",
    "wall_time_sec",
    "episode_id",
    "provenance",
    "result_provenance",
    "runtime_input_records",
    "scenario_params",
    "selected_map_identity",
    "event_ledger",
    "failure_mechanism",
)


def _key(config: str, cell: str, seed: object) -> tuple[str, str, str]:
    return config, cell, str(int(seed))


def _raw_index(directory: Path, stage: str, expected_commit: str) -> dict:  # noqa: C901
    index = {}
    files = sorted(directory.glob(f"{stage}__*.jsonl"))
    if not files:
        raise ValueError(f"no {stage} raw JSONL files in {directory}")
    for path in files:
        parts = path.name.split("__")
        if len(parts) < 3:
            raise ValueError(f"invalid raw file name: {path.name}")
        sidecar = json.loads(Path(f"{path}.provenance.json").read_text(encoding="utf-8"))
        if sidecar.get("run", {}).get("repo_commit") != expected_commit:
            raise ValueError(f"sidecar commit mismatch: {path.name}")
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                record = json.loads(line)
                if not set(RAW_SCIENTIFIC_FIELDS).issubset(record):
                    raise ValueError(f"raw scientific fields missing: {path.name}")
                key = _key(parts[1], record["scenario_id"], record["seed"])
                if key in index:
                    raise ValueError(f"duplicate raw episode: {key}")
                if record.get("git_hash") != expected_commit:
                    raise ValueError(f"raw git_hash mismatch: {key}")
                for section, field in (
                    ("provenance", "git_hash"),
                    ("result_provenance", "repo_commit"),
                    ("event_ledger", "software_commit"),
                ):
                    if (record.get(section) or {}).get(field) != expected_commit:
                        raise ValueError(f"raw {section}.{field} mismatch: {key}")
                index[key] = (path, offset)
    if not index:
        raise ValueError(f"no raw episodes in {directory}")
    return index


def _read_record(location: tuple[Path, int]) -> dict:
    path, offset = location
    with path.open("rb") as handle:
        handle.seek(offset)
        return json.loads(handle.readline())


def _episodes(directory: Path, raw: dict, expected_commit: str) -> tuple[dict, list]:
    with (directory / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not set(KEY_FIELDS).union(EPISODE_PROVENANCE).issubset(fields):
            raise ValueError("episode table missing key or provenance fields")
        rows = {}
        for row in reader:
            key = _key(row["config"], row["cell"], row["seed"])
            if key in rows:
                raise ValueError(f"duplicate extracted episode: {key}")
            if key not in raw:
                raise ValueError(f"extracted episode absent from raw: {key}")
            if row["git_hash"] != expected_commit or row["source_file"] != raw[key][0].name:
                raise ValueError(f"extracted source identity mismatch: {key}")
            rows[key] = row
    if rows.keys() != raw.keys():
        raise ValueError("raw and extracted episode keys do not match")
    return rows, fields


def _steps(directory: Path, selected: set, known_episodes: set) -> tuple[dict, list]:
    rows = {}
    with gzip.open(directory / "steps.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not set(KEY_FIELDS).union({"step"}).issubset(fields):
            raise ValueError("step table missing key fields")
        for row in reader:
            episode = _key(row["config"], row["cell"], row["seed"])
            if episode not in known_episodes:
                raise ValueError(f"step belongs to unknown episode: {episode}")
            if episode not in selected:
                continue
            key = (*episode, row["step"])
            if key in rows:
                raise ValueError(f"duplicate extracted step: {key}")
            value = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            rows[key] = hashlib.sha256(value).digest()
    return rows, fields


def _compare(args: argparse.Namespace) -> dict:
    reference = _raw_index(args.reference_raw, args.stage, ORIGINAL_COMMIT)
    rerun = _raw_index(args.rerun_raw, args.stage, args.rerun_commit)
    reference_rows, reference_fields = _episodes(args.reference_extract, reference, ORIGINAL_COMMIT)
    rerun_rows, rerun_fields = _episodes(args.rerun_extract, rerun, args.rerun_commit)
    if not rerun.keys() <= reference.keys():
        raise ValueError("rerun includes episode keys absent from the reference")
    if reference_fields != rerun_fields:
        raise ValueError("episode table schemas differ")
    selected = set(rerun)
    reference_steps, reference_step_fields = _steps(
        args.reference_extract, selected, set(reference)
    )
    rerun_steps, rerun_step_fields = _steps(args.rerun_extract, selected, set(rerun))
    if reference_step_fields != rerun_step_fields:
        raise ValueError("step table schemas differ")
    if reference_steps.keys() != rerun_steps.keys():
        raise ValueError("extracted step keys differ")
    mismatches = []
    volatile = Counter()
    trace_step_keys = set()
    for key in sorted(selected):
        episode_mismatches, episode_volatile, episode_steps = _compare_episode(
            key, reference_rows[key], rerun_rows[key], reference[key], rerun[key], reference_fields
        )
        mismatches.extend(episode_mismatches)
        volatile.update(episode_volatile)
        trace_step_keys.update(episode_steps)
    changed_steps = [key for key in reference_steps if reference_steps[key] != rerun_steps[key]]
    if trace_step_keys != reference_steps.keys():
        mismatches.append({"extracted_step_coverage": "does not match the raw trace"})
    if changed_steps:
        mismatches.append(
            {"extracted_step_mismatches": len(changed_steps), "first": changed_steps[0]}
        )
    return {
        "reference_commit_verified": ORIGINAL_COMMIT,
        "rerun_commit_verified": args.rerun_commit,
        "source_file_bound_to_each_raw_file": True,
        "reference_episode_count": len(reference),
        "compared_episode_count": len(selected),
        "compared_step_count": len(reference_steps),
        "volatile_episode_difference_counts": dict(sorted(volatile.items())),
        "mismatches": mismatches,
    }


def _compare_episode(key, left, right, reference_location, rerun_location, fields):
    mismatches = []
    volatile = Counter()
    trace_steps = set()
    changed = [f for f in fields if f not in EPISODE_PROVENANCE and left[f] != right[f]]
    if changed:
        mismatches.append({"episode": key, "episode_fields": changed})
    volatile["source_file"] += left["source_file"] != right["source_file"]
    old, new = _read_record(reference_location), _read_record(rerun_location)
    changed = [f for f in RAW_SCIENTIFIC_FIELDS if old.get(f) != new.get(f)]
    if changed:
        mismatches.append({"episode": key, "raw_scientific_fields": changed})
    old_trace = old.get("algorithm_metadata", {}).get("simulation_step_trace")
    new_trace = new.get("algorithm_metadata", {}).get("simulation_step_trace")
    if old_trace is None or new_trace is None or old_trace != new_trace:
        mismatches.append({"episode": key, "raw_trace": "missing or differs"})
    if isinstance(old_trace, dict):
        for step in old_trace.get("steps", []):
            instrument = step.get("xorca") or {}
            if instrument.get("pre") and instrument.get("post"):
                trace_key = (*key, str(step["step"]))
                if trace_key in trace_steps:
                    raise ValueError(f"duplicate raw step: {trace_key}")
                trace_steps.add(trace_key)
    for field in RAW_VOLATILE_FIELDS:
        volatile[field] += old.get(field) != new.get(field)
    return mismatches, volatile, trace_steps


def main() -> int:
    """Check exact scientific outputs and record differences in provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--reference-extract", type=Path, required=True)
    parser.add_argument("--rerun-raw", type=Path, required=True)
    parser.add_argument("--rerun-extract", type=Path, required=True)
    parser.add_argument("--rerun-commit", required=True, help="expected 40-character source commit")
    parser.add_argument("--stage", default="main")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.rerun_commit):
        parser.error("--rerun-commit must be a full lowercase Git commit SHA")

    try:
        report = _compare(args)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if report["mismatches"] else 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"reconstruction comparison failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
