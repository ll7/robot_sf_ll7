#!/usr/bin/env python3
"""Run a batch of generated-catalog candidates through the persistence gate.

End-to-end CPU-only runner that takes episode traces or catalog entries and
produces `generated_scenario_persistence.v1` conformance records with promotion
verdicts.

Usage:
    # Run with synthetic candidates for contract validation
    uv run python scripts/tools/run_persistence_candidate_smoke.py --synth --output-dir /tmp/smoke

    # Run with episode JSONL traces
    uv run python scripts/tools/run_persistence_candidate_smoke.py episodes.jsonl \\
        --output-dir /tmp/conformance

    # Run with catalog-entry JSONL
    uv run python scripts/tools/run_persistence_candidate_smoke.py catalog.jsonl \\
        --output-dir /tmp/conformance

    # Combine with validation CLI
    uv run python scripts/tools/run_persistence_candidate_smoke.py episodes.jsonl \\
        --output-dir /tmp/conformance
    uv run python scripts/tools/validate_generated_scenario_persistence.py \\
        --batch /tmp/conformance/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.scenario_generation.candidate_runner import (
    run_candidate_persistence_smoke,
)

_DEFAULT_CONFIG = {
    "config_id": "issue-5600-persistence-gate",
    "frozen": True,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Episode trace JSONL, catalog-entry JSONL, or JSON files to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write individual persistence-record JSON files.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Write all records as a single JSONL file.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write a compact summary JSON (counts, verdicts, exclusion reasons).",
    )
    parser.add_argument(
        "--synth",
        action="store_true",
        help="Generate synthetic candidates to demonstrate both promotion paths.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path. Defaults to the issue-5600 frozen config.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print a compact summary to stderr.",
    )
    return parser


def _load_candidates_from_paths(paths: list[Path]) -> list[dict[str, Any]]:
    """Load candidates from JSONL or JSON file paths."""
    candidates: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                candidates.append(record)
            elif isinstance(record, list):
                candidates.extend(record)
    return candidates


def _generate_synthetic_candidates() -> list[dict[str, Any]]:
    """Generate two synthetic candidates: one promoting, one rejected."""
    # Candidate 1: persistent event that survives all perturbations -> promote
    candidate_promote = {
        "episode_id": "synth-promote-001",
        "seed": 42,
        "source_map": "maps/svg_maps/classic_crossing.svg",
        "steps": _build_trace(
            robot_trajectory=[
                [1.0, 0.0],
                [1.5, 3.0],
                [2.0, 5.0],
                [2.5, 8.0],
                [3.0, 10.0],
            ],
            ped_trajectories={
                0: [[5.0, 10.0], [5.0, 8.0], [5.0, 5.0], [5.0, 3.0], [5.0, 0.0]],
                1: [[8.0, 10.0], [8.0, 8.0], [8.0, 5.0], [8.0, 3.0], [8.0, 0.0]],
            },
        ),
    }
    # Candidate 2: non-persistent event that dies under perturbation -> reject
    candidate_reject = {
        "episode_id": "synth-reject-001",
        "seed": 43,
        "source_map": "maps/svg_maps/classic_crossing.svg",
        "steps": _build_trace(
            robot_trajectory=[
                [3.0, 0.0],
                [3.0, 2.0],
                [3.0, 4.0],
                [3.0, 6.0],
                [3.0, 8.0],
            ],
            ped_trajectories={
                0: [[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0]],
            },
        ),
    }
    return [candidate_promote, candidate_reject]


def _build_trace(
    robot_trajectory: list[list[float]],
    ped_trajectories: dict[int, list[list[float]]],
) -> list[dict[str, Any]]:
    """Build an episode trace from robot and pedestrian trajectories."""
    n_steps = len(robot_trajectory)
    steps: list[dict[str, Any]] = []
    for t in range(n_steps):
        step: dict[str, Any] = {
            "time_s": float(t),
            "robot": {"position": robot_trajectory[t]},
            "pedestrians": [],
        }
        for pid, trajectory in sorted(ped_trajectories.items()):
            step["pedestrians"].append({"position": trajectory[t]})
        steps.append(step)
    return steps


def _print_summary(stdout_text: str) -> None:
    """Print summary to stderr."""
    print(stdout_text, file=sys.stderr)


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.synth:
        candidates = _generate_synthetic_candidates()
    elif args.inputs:
        candidates = _load_candidates_from_paths(args.inputs)
    else:
        print("error: supply input paths or --synth", file=sys.stderr)
        sys.exit(2)

    if not candidates:
        print("error: no candidates loaded", file=sys.stderr)
        sys.exit(2)

    try:
        results = run_candidate_persistence_smoke(
            candidates=candidates,
            config=_DEFAULT_CONFIG,
            output_root=args.output_dir,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: persistence smoke failed: {exc}", file=sys.stderr)
        sys.exit(1)

    promoted = [r for r in results if r["promotion"]["verdict"] == "promote"]
    rejected = [r for r in results if r["promotion"]["verdict"] != "promote"]

    lines: list[str] = []
    for record in results:
        verdict = record["promotion"]["verdict"].upper()
        lines.append(
            f"{verdict} {record['scenario_id']}: {record['promotion']['exclusion_reason']}"
        )
    summary_text = (
        f"candidates: {len(results)}  "
        f"promoted: {len(promoted)}  "
        f"rejected: {len(rejected)}\n" + "\n".join(lines)
    )

    if args.output_jsonl:
        args.output_jsonl.write_text(
            "\n".join(json.dumps(r, sort_keys=True, indent=2) + "\n" for r in results),
            encoding="utf-8",
        )

    if args.summary_json:
        summary = {
            "schema_version": "persistence-smoke-summary.v1",
            "total_candidates": len(results),
            "promoted_count": len(promoted),
            "rejected_count": len(rejected),
            "records": [
                {
                    "scenario_id": r["scenario_id"],
                    "verdict": r["promotion"]["verdict"],
                    "exclusion_reason": r["promotion"]["exclusion_reason"],
                    "exact_replay_status": r["exact_replay"]["status"],
                    "critical_event_status": r["critical_event_reproduced"]["status"],
                    "persistence_rate": r["perturbation_persistence"]["persistence_rate"],
                }
                for r in results
            ],
        }
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(summary_text)

    sys.exit(2 if rejected else 0)
