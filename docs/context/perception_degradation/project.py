"""Project raw campaign producer files into compact, endpoint-complete episode rows.

Run in the acquisition environment (needs the repository's fallback parser):

    uv run python docs/context/perception_degradation/project.py \
        --raw-root <stage run root> --stage main --output <episodes.jsonl.gz>

Each row binds to the exact producer line by path, line number and SHA-256.
Trajectories, step traces and perception traces stay in the raw archive.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker

PLANNERS = {
    "orca": "orca",
    "hybrid_v4_continuous_guard_on": "hybrid_rule_local_planner",
    "hybrid_v4_continuous_guard_off": "hybrid_rule_local_planner",
    "ppo": "ppo",
}
METRICS = (
    "success",
    "collisions",
    "ped_collision_count",
    "obstacle_collision_count",
    "min_distance",
    "time_to_goal_norm",
    "path_efficiency",
)


def project_line(raw: dict, planner_key: str) -> dict:
    """Return the compact row for one producer line.

    Returns:
        JSON-safe compact row.
    """
    meta = raw["algorithm_metadata"]
    runtime_meta = {
        k: v
        for k, v in meta.items()
        if k
        not in (
            "simulation_step_trace",
            "paired_effect_metric_producer",
            "pedestrian_perception_trace",
        )
    }
    # The canonical parser reads every *fallback* key as a counter, so PPO's boolean
    # config flag ``fallback_to_goal: false`` would be reported as "invalid". It is
    # recorded explicitly below and must be False; everything else is parsed as-is.
    config_fallback_to_goal = (runtime_meta.get("config") or {}).get("fallback_to_goal")
    if config_fallback_to_goal is False:
        runtime_meta["config"] = {
            k: v for k, v in runtime_meta["config"].items() if k != "fallback_to_goal"
        }
    marker = runtime_fallback_or_degraded_marker(
        runtime_meta, expected_algorithm=PLANNERS[planner_key]
    )
    if config_fallback_to_goal not in (None, False):
        marker = marker or ["config.fallback_to_goal", "enabled"]
    checkpoint = (meta.get("planner_runtime") or {}).get("checkpoint_provenance") or {}
    perception = meta.get("pedestrian_perception") or {}
    noise = raw.get("observation_noise") or {}
    kin = meta.get("planner_kinematics") or {}
    runtime = meta.get("planner_runtime") or {}
    config = meta.get("config") or {}
    view = (raw.get("integrity") or {}).get("effective_view") or {}
    return {
        "condition": (noise.get("pedestrian_perception") or {}).get("condition"),
        "planner_key": planner_key,
        "scenario_id": raw["scenario_id"],
        "seed": raw["seed"],
        "git_hash": raw["git_hash"],
        "mode": kin.get("execution_mode"),
        "status": meta.get("status"),
        "fallback_count": runtime.get("fallback_count"),
        "fallback_or_degraded_marker": marker,
        "config_fallback_to_goal": config_fallback_to_goal,
        "checkpoint_load_status": checkpoint.get("load_status"),
        "checkpoint_fallback_triggered": checkpoint.get("fallback_triggered"),
        "pedestrian_model_status": (raw.get("pedestrian_model") or {}).get(
            "fallback_degraded_status"
        ),
        "effective_view_degraded": view.get("degraded"),
        "ablation_disable_hard_guard": config.get("ablation_disable_hard_guard"),
        "effective_budget_steps": raw.get("effective_budget_steps"),
        "steps": raw.get("steps"),
        "termination_reason": raw.get("termination_reason"),
        "outcome": raw.get("outcome"),
        "metrics": {k: (raw.get("metrics") or {}).get(k) for k in METRICS},
        "perception": {
            "condition": perception.get("condition"),
            "episode_key": perception.get("episode_key"),
            "all_channels_consumed_every_snapshot": perception.get(
                "all_channels_consumed_every_snapshot"
            ),
            "expected_channels": perception.get("expected_channels"),
            "stats": perception.get("stats"),
        },
    }


def main() -> None:
    """Project every producer file under ``--raw-root``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, sources = [], []
    for path in sorted(args.raw_root.glob("*/runs/*__differential_drive/episodes.jsonl")):
        planner_key = path.parent.name.split("__", 1)[0]
        if planner_key not in PLANNERS:
            raise ValueError(f"unknown planner key {planner_key}")
        rel = str(path.relative_to(args.raw_root))
        source = {"path": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": 0}
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = project_line(json.loads(line), planner_key)
                row.update(
                    source_path=rel,
                    source_line=line_number,
                    source_line_sha256=hashlib.sha256(line).hexdigest(),
                )
                rows.append(row)
                source["rows"] += 1
        sources.append(source)
    payload = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(payload.encode(), compresslevel=9, mtime=0))
    sources_path = args.output.with_name(args.output.name.replace(".jsonl.gz", "_sources.json"))
    sources_path.write_text(
        json.dumps(
            {
                "rows": len(rows),
                "files": sources,
                "compact_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(rows)} rows from {len(sources)} files -> {args.output}")


if __name__ == "__main__":
    main()
