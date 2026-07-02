"""Record an offline RL trajectory dataset from benchmark episode traces."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.rl_trajectory_dataset import (
    RETURN_CONVENTION,
    REWARD_CONVENTION,
    RLTrajectoryEpisode,
    assign_deterministic_split,
    build_rl_trajectory_manifest,
    compute_return_to_go,
    load_rl_trajectory_dataset,
    sha256_file,
    write_rl_trajectory_dataset,
)
from robot_sf.benchmark.schemas.rl_trajectory_dataset_schema import (
    validate_rl_trajectory_dataset_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the recorder CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--skip-missing-trace",
        action="store_true",
        help="Skip source records without simulation_step_trace instead of failing closed.",
    )
    return parser


def read_source_records(source_jsonl: Path) -> list[dict[str, Any]]:
    """Read benchmark episode JSONL records."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        source_jsonl.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source_jsonl}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{source_jsonl}:{line_number}: source record must be an object")
        records.append(payload)
    if not records:
        raise ValueError(f"{source_jsonl}: no source records found")
    return records


def convert_source_records(
    records: Sequence[dict[str, Any]],
    *,
    dataset_id: str,
    source_jsonl: Path,
    skip_missing_trace: bool = False,
) -> list[RLTrajectoryEpisode]:
    """Convert benchmark episode records into RL trajectory episodes."""
    episodes: list[RLTrajectoryEpisode] = []
    for index, record in enumerate(records):
        metadata = _mapping(record.get("algorithm_metadata"), field="algorithm_metadata")
        trace = metadata.get("simulation_step_trace")
        if not isinstance(trace, dict) or not isinstance(trace.get("steps"), list):
            if skip_missing_trace:
                continue
            raise ValueError(
                f"source record {index} lacks algorithm_metadata.simulation_step_trace"
            )
        episode = _episode_from_trace_record(
            record,
            trace,
            dataset_id=dataset_id,
            source_jsonl=source_jsonl,
            source_index=index,
        )
        episodes.append(episode)
    if not episodes:
        raise ValueError("no source records produced RL trajectory episodes")
    return episodes


def write_dataset_and_manifest(
    *,
    episodes: Sequence[RLTrajectoryEpisode],
    output_dir: Path,
    dataset_id: str,
    source_jsonl: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write dataset JSONL and manifest, then validate both."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"{dataset_id}.jsonl"
    manifest_path = output_dir / f"{dataset_id}.manifest.json"
    write_rl_trajectory_dataset(episodes, dataset_path)
    manifest = build_rl_trajectory_manifest(
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        episodes=episodes,
        created_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        provenance={
            "source_jsonl": str(source_jsonl),
            "source_sha256": sha256_file(source_jsonl),
            "git_commit": _git_commit(),
            "command": " ".join(
                ["record_rl_trajectory_dataset", "--source-jsonl", str(source_jsonl)]
            ),
            "reward_convention": REWARD_CONVENTION,
            "return_convention": RETURN_CONVENTION,
            "artifact_durability": "worktree_local_until_promoted",
        },
    )
    validate_rl_trajectory_dataset_manifest(manifest)
    tmp_manifest_path = manifest_path.with_suffix(".json.tmp")
    tmp_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp_manifest_path.replace(manifest_path)
    load_rl_trajectory_dataset(dataset_path)
    return dataset_path, manifest_path, manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the recorder CLI."""
    args = build_arg_parser().parse_args(argv)
    records = read_source_records(args.source_jsonl)
    episodes = convert_source_records(
        records,
        dataset_id=args.dataset_id,
        source_jsonl=args.source_jsonl,
        skip_missing_trace=args.skip_missing_trace,
    )
    dataset_path, manifest_path, manifest = write_dataset_and_manifest(
        episodes=episodes,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        source_jsonl=args.source_jsonl,
    )
    payload = {
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "episode_count": manifest["episode_count"],
        "step_count": manifest["step_count"],
        "dataset_sha256": manifest["dataset_sha256"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _episode_from_trace_record(
    record: dict[str, Any],
    trace: dict[str, Any],
    *,
    dataset_id: str,
    source_jsonl: Path,
    source_index: int,
) -> RLTrajectoryEpisode:
    steps = trace["steps"]
    scenario_id = str(
        record.get("scenario_id")
        or record.get("scenario")
        or record.get("map_name")
        or f"source_record_{source_index}"
    )
    seed = _parse_seed(record.get("seed", record.get("scenario_seed", source_index)))
    source_policy_id = str(
        record.get("algo") or record.get("algorithm") or record.get("planner") or "unknown"
    )
    episode_id = str(
        record.get("episode_id")
        or f"{scenario_id}:seed{seed}:{source_policy_id}:{source_index:06d}"
    )
    split = assign_deterministic_split(scenario_id, seed)

    observations: list[Any] = []
    actions: list[Any] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    pedestrians: list[Any] = []
    robot_states: list[Any] = []
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(
                f"source record {source_index} step {step_index}: step must be an object"
            )
        rl = _mapping(step.get("rl"), field=f"step {step_index} rl")
        if "reward" not in rl:
            raise ValueError(f"source record {source_index} step {step_index}: missing rl.reward")
        rewards.append(float(rl["reward"]))
        terminated.append(bool(rl.get("terminated", False)))
        truncated.append(bool(rl.get("truncated", False)))
        robot = _mapping(step.get("robot"), field=f"step {step_index} robot")
        ped_state = step.get("pedestrians", [])
        planner = _mapping(step.get("planner"), field=f"step {step_index} planner")
        observations.append(step.get("observation", {"robot": robot, "pedestrians": ped_state}))
        actions.append(planner.get("selected_action", step.get("action")))
        pedestrians.append(ped_state)
        robot_states.append(robot)

    return RLTrajectoryEpisode(
        dataset_id=dataset_id,
        episode_id=episode_id,
        scenario_id=scenario_id,
        seed=seed,
        source_policy_id=source_policy_id,
        split=split,
        observations=tuple(observations),
        actions=tuple(actions),
        rewards=tuple(rewards),
        return_to_go=tuple(compute_return_to_go(rewards)),
        terminated=tuple(terminated),
        truncated=tuple(truncated),
        pedestrians=tuple(pedestrians),
        robot_states=tuple(robot_states),
        provenance={
            "source": "map_runner_episode.simulation_step_trace",
            "source_jsonl": str(source_jsonl),
            "source_record_index": source_index,
            "trace_schema_version": trace.get("schema_version"),
            "reward_convention": REWARD_CONVENTION,
            "return_convention": RETURN_CONVENTION,
        },
    )


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"{field} must be an object")


def _parse_seed(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("seed must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer") from exc


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
