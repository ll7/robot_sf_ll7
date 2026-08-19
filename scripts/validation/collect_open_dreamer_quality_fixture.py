#!/usr/bin/env python3
"""Collect a bounded native Robot SF trace fixture for issue #6318 Step 3 diagnostics.

This helper runs the existing map runner and sends its per-step traces through the canonical
``RLTrajectoryDataset.v1`` recorder. It is intentionally a small local diagnostic path: generated
datasets and source records remain worktree-local until separately promoted, and no benchmark or
paper-facing result is implied by a successful collection.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.classic_interactions_loader import load_classic_matrix, select_scenario
from robot_sf.benchmark.map_runner.map_runner import _run_map_episode
from scripts.benchmark.record_rl_trajectory_dataset import (
    convert_source_records,
    write_dataset_and_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _scenario_seed(value: str) -> tuple[str, int]:
    """Parse one ``SCENARIO_ID:SEED`` collection specification."""
    scenario_id, separator, raw_seed = value.rpartition(":")
    if not separator or not scenario_id or not raw_seed:
        raise argparse.ArgumentTypeError("expected SCENARIO_ID:SEED")
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("SEED must be an integer") from exc
    if seed < 0:
        raise argparse.ArgumentTypeError("SEED must be non-negative")
    return scenario_id, seed


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the bounded native-fixture collector parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-matrix",
        type=Path,
        default=Path("configs/scenarios/classic_interactions.yaml"),
    )
    parser.add_argument(
        "--scenario-seed",
        action="append",
        required=True,
        type=_scenario_seed,
        metavar="SCENARIO_ID:SEED",
        help="Repeat for each episode; use distinct scenario IDs for train and holdout splits.",
    )
    parser.add_argument("--planner", default="simple_policy")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    return parser


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=_REPOSITORY_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _collect_records(
    *,
    scenario_matrix: Path,
    scenario_seeds: Sequence[tuple[str, int]],
    planner: str,
    horizon: int,
    dt: float,
) -> list[dict[str, Any]]:
    scenarios = load_classic_matrix(str(scenario_matrix))
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for scenario_id, seed in scenario_seeds:
        key = (scenario_id, seed)
        if key in seen:
            raise ValueError(f"duplicate scenario/seed specification: {scenario_id}:{seed}")
        seen.add(key)
        scenario = select_scenario(scenarios, scenario_id)
        record = _run_map_episode(
            scenario,
            seed,
            horizon=horizon,
            dt=dt,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo=planner,
            scenario_path=scenario_matrix,
            record_simulation_step_trace=True,
        )
        record["scenario_id"] = scenario_id
        record["seed"] = seed
        record["algorithm"] = planner
        records.append(record)
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Collect traces, record the canonical dataset, and print artifact paths."""
    args = build_arg_parser().parse_args(argv)
    if args.horizon <= 1:
        raise SystemExit("--horizon must be greater than one")
    if args.dt <= 0.0:
        raise SystemExit("--dt must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_jsonl = output_dir / f"{args.dataset_id}.source.jsonl"
    records = _collect_records(
        scenario_matrix=args.scenario_matrix,
        scenario_seeds=args.scenario_seed,
        planner=args.planner,
        horizon=args.horizon,
        dt=args.dt,
    )
    source_jsonl.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    episodes = convert_source_records(
        records,
        dataset_id=args.dataset_id,
        source_jsonl=source_jsonl,
    )
    dataset_path, manifest_path, manifest = write_dataset_and_manifest(
        episodes=episodes,
        output_dir=output_dir,
        dataset_id=args.dataset_id,
        source_jsonl=source_jsonl,
    )
    metadata_path = output_dir / f"{args.dataset_id}.collection.json"
    _write_json(
        metadata_path,
        {
            "schema_version": "open_dreamer_quality_fixture_collection.v1",
            "evidence_boundary": "diagnostic_only",
            "source_route": "native_map_runner_trace",
            "artifact_durability": "worktree_local_until_promoted",
            "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "git_commit": _git_commit(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "scenario_matrix": str(args.scenario_matrix),
            "scenario_seeds": [f"{scenario}:{seed}" for scenario, seed in args.scenario_seed],
            "planner": args.planner,
            "horizon": args.horizon,
            "dt": args.dt,
            "dataset_path": str(dataset_path),
            "manifest_path": str(manifest_path),
            "source_jsonl": str(source_jsonl),
        },
    )
    print(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "manifest_path": str(manifest_path),
                "collection_metadata_path": str(metadata_path),
                "source_jsonl": str(source_jsonl),
                "episode_count": manifest["episode_count"],
                "step_count": manifest["step_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
