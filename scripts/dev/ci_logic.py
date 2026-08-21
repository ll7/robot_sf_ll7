"""Importable CI workflow logic helpers (issue #7666).

Three logic groups extracted from inline ``ci.yml`` scripts so they can be
unit-tested deterministically:

- :func:`derive_model_cache_key`: exact-repeat model-cache key from
  registry-pinned digests;
- :func:`merge_duration_stores` / :func:`merge_duration_artifacts`: pytest
  duration-store validation and merge;
- :func:`evaluate_required_jobs`: final aggregate required-needs evaluation.

Each has a thin CLI wrapper under ``scripts/dev/`` used by the workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

EXPECTED_DURATION_SHARDS = 4


def derive_model_cache_key(config_path: str | Path) -> str:
    """Return the exact-repeat model-cache key for a training config.

    The key is the first 16 hex digits of SHA-256 over the ``|``-joined
    registry-pinned GitHub-release digests of every required model ID, in
    registry order. Fails closed when a digest is missing or empty.
    """
    from robot_sf.models.preflight import required_model_ids_for_config
    from robot_sf.models.registry import get_registry_entry

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    ids = required_model_ids_for_config(cfg)
    if not ids:
        raise SystemExit(f"No required model IDs resolved for config: {config_path}")
    shas: list[str] = []
    for model_id in ids:
        entry: dict[str, Any] = get_registry_entry(model_id)
        digest = str(entry.get("github_release", {}).get("sha256", ""))
        if not digest:
            raise SystemExit(
                f"Missing registry-pinned github_release.sha256 for model {model_id!r}"
            )
        shas.append(digest)
    return hashlib.sha256("|".join(shas).encode()).hexdigest()[:16]


def merge_duration_stores(shard_files: list[Path]) -> dict[str, float]:
    """Validate and merge pytest duration stores; return a deterministic dict.

    Raises ``SystemExit`` with an actionable message when the shard set is not
    exactly the expected four, any store is malformed/non-finite/negative, or
    two stores claim the same node id.
    """
    expected_names = {
        f"pytest-durations-{index}" for index in range(1, EXPECTED_DURATION_SHARDS + 1)
    }
    actual_names = {path.parent.name for path in shard_files}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise SystemExit(
            "Expected exactly one pytest duration store from each of four shards; "
            f"missing={missing or 'none'} unexpected={unexpected or 'none'}."
        )

    merged: dict[str, float] = {}
    for path in sorted(shard_files):
        try:
            durations = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid pytest duration store: {path} ({exc})") from exc
        if not isinstance(durations, dict) or any(
            not isinstance(nodeid, str)
            or not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration < 0
            for nodeid, duration in durations.items()
        ):
            raise SystemExit(f"Invalid pytest duration store: {path}")
        overlap = set(merged).intersection(durations)
        if overlap:
            raise SystemExit(f"Overlapping pytest duration stores: {path}")
        merged.update(durations)
    return dict(sorted(merged.items()))


def merge_duration_artifacts(artifacts_dir: Path, output_path: Path) -> int:
    """Merge downloaded ``pytest-durations-*`` artifact stores into one file.

    Returns the number of merged test durations. The output JSON is written
    with sorted keys and trailing newline for byte-stable caching.
    """
    files = sorted(artifacts_dir.glob("*/.test_durations"))
    merged = merge_duration_stores(files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(merged, indent=4, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(merged)


# (job-id, only-events, excluded-events) where None means "no restriction".
# Event-specific rules preserve current ci.yml behavior exactly:
# - coverage-gate is required on every event except pull_request (including
#   merge_group and push);
# - changed-coverage-gate applies to pull_request and merge_group only.
REQUIRED_JOB_RULES: tuple[tuple[str, frozenset[str] | None, frozenset[str] | None], ...] = (
    ("fast-feedback", None, None),
    ("coverage-gate", None, frozenset({"pull_request"})),
    ("changed-coverage-gate", frozenset({"pull_request", "merge_group"}), None),
    ("compat-matrix", None, None),
    ("fast-pysf-compat", None, None),
    ("smoke-artifacts", None, None),
    ("xdist-scratch-isolation", None, None),
    ("wheel-smoke-install", None, None),
    ("examples-smoke", None, None),
    ("notebooks-smoke", None, None),
    ("determinism-gate", None, None),
    ("exact-repeat-model-preflight", None, None),
)


def evaluate_required_jobs(job_results: dict[str, str], event_name: str) -> list[str]:
    """Return fail-closed failure reasons for the final aggregate ``ci`` job.

    ``job_results`` maps job ID to its needs result (success, failure,
    cancelled, skipped, ...). A job missing from the map is treated as unknown
    and fails closed.
    """
    failures: list[str] = []
    for job_id, only_events, excluded_events in REQUIRED_JOB_RULES:
        if only_events is not None and event_name not in only_events:
            continue
        if excluded_events is not None and event_name in excluded_events:
            continue
        result = job_results.get(job_id)
        if result != "success":
            failures.append(f"{job_id} finished with {result or 'unknown result'}")
    unexpected = sorted(set(job_results) - {job for job, _, _ in REQUIRED_JOB_RULES})
    for job_id in unexpected:
        failures.append(f"{job_id} is not part of the required-check manifest")
    return failures


def main(argv: list[str] | None = None) -> int:
    """Thin CLI dispatch for the three workflow-facing helpers."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    key_parser = sub.add_parser("model-cache-key", help="Print the exact-repeat model-cache key")
    key_parser.add_argument("--config", required=True)

    merge_parser = sub.add_parser("merge-durations", help="Merge pytest duration artifacts")
    merge_parser.add_argument("--artifacts-dir", type=Path, default=Path(".duration-artifacts"))
    merge_parser.add_argument("--output", type=Path, default=Path(".test_durations"))

    agg_parser = sub.add_parser("aggregate-result", help="Evaluate the final required-needs map")
    agg_parser.add_argument("--event", required=True)
    agg_parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="JOB=RESULT",
        help="Job result mapping; repeat per needed job",
    )

    args = parser.parse_args(argv)
    if args.command == "model-cache-key":
        print(derive_model_cache_key(args.config))
        return 0
    if args.command == "merge-durations":
        count = merge_duration_artifacts(args.artifacts_dir, args.output)
        print(f"Merged duration stores with {count} test durations.")
        return 0
    results: dict[str, str] = {}
    for item in args.result:
        job_id, _, value = item.partition("=")
        results[job_id] = value
    failures = evaluate_required_jobs(results, args.event)
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
