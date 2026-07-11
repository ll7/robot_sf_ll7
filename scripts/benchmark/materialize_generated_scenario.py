"""Materialize one generated catalog entry as a standalone replay scenario YAML."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.scenario_generation import (
    dump_generated_scenario_yaml,
    generated_replay_status_entry,
    materialize_generated_scenario,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-entry-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument(
        "--replay-smoke-output",
        type=Path,
        help="Optional JSONL destination for one bounded CPU replay smoke.",
    )
    return parser.parse_args()


def main() -> int:
    """Write materialized YAML and an updated, explicit replay-status entry."""

    args = _parse_args()
    entry: Any = json.loads(args.catalog_entry_json.read_text(encoding="utf-8"))
    result = materialize_generated_scenario(entry, max_episode_steps=args.max_episode_steps)
    if result.scenario_document is None:
        _write_status(args.status_output, entry, result)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_map = Path(entry["source_episode"]["source_map"])
    if not source_map.is_absolute():
        source_map = Path(__file__).resolve().parents[2] / source_map
    result.scenario_document["scenarios"][0]["map_file"] = os.path.relpath(
        source_map.resolve(), args.output.parent.resolve()
    )
    args.output.write_text(dump_generated_scenario_yaml(result), encoding="utf-8")
    smoke_failed = False
    if args.replay_smoke_output is not None:
        try:
            from robot_sf.benchmark.map_runner import run_map_batch

            summary = run_map_batch(
                args.output,
                args.replay_smoke_output,
                Path(__file__).resolve().parents[2]
                / "robot_sf/benchmark/schemas/episode.schema.v1.json",
                horizon=args.max_episode_steps,
                dt=0.1,
                record_forces=False,
                algo="goal",
                workers=1,
                resume=False,
            )
            if summary["successful_jobs"] == 1 and summary["failed_jobs"] == 0:
                result = replace(result, status="replay_validated")
                result.scenario_document["scenarios"][0]["metadata"]["generated_replay"][
                    "replay_status"
                ] = result.status
                args.output.write_text(dump_generated_scenario_yaml(result), encoding="utf-8")
            else:
                smoke_failed = True
                result = replace(
                    result,
                    warnings=(
                        "replay_smoke: materialized scenario loaded but did not complete successfully",
                    ),
                )
        except (
            OSError,
            RuntimeError,
            ValueError,
            yaml.YAMLError,
        ) as exc:  # pragma: no cover - runtime dependent
            smoke_failed = True
            result = replace(result, warnings=(f"replay_smoke_error: {exc}",))
    _write_status(args.status_output, entry, result)
    return 3 if smoke_failed else 0


def _write_status(args_output: Path, entry: Any, result: Any) -> None:
    """Persist one explicit adapter/replay status entry."""

    status_entry = generated_replay_status_entry(entry, result)
    args_output.parent.mkdir(parents=True, exist_ok=True)
    args_output.write_text(
        json.dumps(status_entry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
