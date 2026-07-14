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
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.scenario_generation.candidate_runner import (
    run_candidate_persistence_smoke,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "analysis" / "issue_5600_persistence_gate.yaml"


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
        help=f"YAML config path (default: {_DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print a compact summary to stderr.",
    )
    return parser


def _load_gate_config(path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:  # noqa: C901
    """Load and validate the frozen gate config used by the runner."""

    config_path = path or _DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config must contain a mapping: {config_path}")
    if raw.get("frozen") is not True:
        raise ValueError("config.frozen must be true")
    config_id = raw.get("config_id")
    if not isinstance(config_id, str) or not config_id.strip():
        raise ValueError("config.config_id must be a non-empty string")

    perturbation = raw.get("perturbation")
    critical_event = raw.get("critical_event")
    promotion = raw.get("promotion")
    if not isinstance(perturbation, dict) or not isinstance(critical_event, dict):
        raise ValueError("config must define perturbation and critical_event mappings")
    if not isinstance(promotion, dict):
        raise ValueError("config must define a promotion mapping")
    try:
        min_persistence_rate = float(promotion["min_persistence_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("promotion.min_persistence_rate must be numeric") from exc
    if min_persistence_rate != 1.0:
        raise ValueError("only min_persistence_rate=1.0 is supported by the fail-closed gate")

    grid = {
        "timing_offsets_s": perturbation.get("timing_offsets_s"),
        "speed_deltas_m_s": perturbation.get("speed_deltas_m_s"),
    }
    for name, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"config.perturbation.{name} must be a non-empty list")

    try:
        time_tolerance_s = float(critical_event["time_tolerance_s"])
        location_tolerance_m = float(critical_event["location_tolerance_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("critical event tolerances must be numeric") from exc
    if time_tolerance_s < 0.0 or location_tolerance_m < 0.0:
        raise ValueError("critical event tolerances must be non-negative")

    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config_record = {
        "config_id": config_id,
        "frozen": True,
        "config_hash": config_hash,
    }
    gate_parameters = {
        "perturbation_grid": grid,
        "event_time_tolerance_s": time_tolerance_s,
        "event_location_tolerance_m": location_tolerance_m,
    }
    return config_record, gate_parameters


def _git_commit() -> str:
    """Return the checked-out commit for output provenance."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("unable to resolve the repository commit for provenance")
    return result.stdout.strip()


def _load_candidates_from_paths(paths: list[Path]) -> list[dict[str, Any]]:
    """Load candidates from JSONL or JSON file paths."""
    candidates: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        try:
            parsed: Any = json.loads(text)
            records = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            records = []
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                records.append(json.loads(line))
        for record in records:
            if isinstance(record, dict):
                candidates.append(record)
            elif isinstance(record, list):
                candidates.extend(item for item in record if isinstance(item, dict))
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


def _synthetic_replay_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return explicit replay fixtures for the synthetic conformance smoke only."""

    expected_verdict = "pass" if candidate.get("episode_id") == "synth-promote-001" else "fail"

    def cell_verdict_fn(**_: Any) -> dict[str, str]:
        return {
            "verdict": expected_verdict,
            "reason": "synthetic conformance fixture; no simulator replay claim",
        }

    return {
        "replayed_episode": candidate,
        "cell_verdict_fn": cell_verdict_fn,
    }


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
        config, gate_parameters = _load_gate_config(args.config)
        results = run_candidate_persistence_smoke(
            candidates=candidates,
            config=config,
            commit_hashes={"code": _git_commit(), "config": config["config_hash"]},
            event_time_tolerance_s=gate_parameters["event_time_tolerance_s"],
            event_location_tolerance_m=gate_parameters["event_location_tolerance_m"],
            perturbation_grid=gate_parameters["perturbation_grid"],
            replay_evidence_fn=_synthetic_replay_evidence if args.synth else None,
            output_root=args.output_dir,
        )
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
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
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text(
            "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in results),
            encoding="utf-8",
        )

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
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
