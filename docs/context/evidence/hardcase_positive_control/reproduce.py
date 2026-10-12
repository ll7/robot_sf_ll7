"""Reproduce one dev-seed positive control and the failing generated-window parity.

Run from a checkout of SOURCE with the project environment. This is a diagnostic
evidence recipe, not a new search or replay implementation. Raw records stay in
the requested output directory. Expected exit: 2 (window parity fails).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

SOURCE = "53721be3248600e08641a9816d094b605e3c93e4"
MATRIX = "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
CONFIG = "configs/algos/social_force_release_v0_0_8.yaml"
MAP = "maps/successor_svg_maps/issue_9762_classic_crossing_goal_zone_entry_v2.svg"
SCENARIO = "classic_cross_trap_high"
SEED = 1001
TOLERANCE_M = 1e-9


def write_json(path: Path, value: object) -> None:
    """Preserve a readable receipt without rewriting canonical episode records."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def episode_summary(row: dict) -> dict:
    """Use the episode's own radius and near-miss contracts."""
    trace = row["algorithm_metadata"]["simulation_step_trace"]["steps"]
    radii = row["effective_physics"]["clearance_definition"]
    total_radius = radii["robot_radius_m"] + radii["pedestrian_radius_m"]
    threshold = row["metric_parameters"]["threshold_profile"]["near_miss_distance_m"]
    near = []
    contact = []
    for frame in trace:
        clearance = (
            min(
                math.dist(frame["robot"]["position"], ped["position"])
                for ped in frame["pedestrians"]
            )
            - total_radius
        )
        if 0 <= clearance < threshold:
            near.append(frame["step"])
        if any(frame.get("collision", {}).values()):
            contact.append(frame["step"])
    return {
        "outcome": row["outcome"],
        "step_count": len(trace),
        "near_misses": row["metrics"]["near_misses"],
        "min_clearance_m": row["metrics"]["min_clearance"],
        "near_miss_steps_zero_based": near,
        "first_near_miss_step": near[0] if near else None,
        "first_contact_step": contact[0] if contact else None,
    }


def main() -> int:  # noqa: PLR0915 - linear evidence recipe preserves stage order
    """Run canonical source, extraction, materialization and two window replays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE:
        raise SystemExit(f"source checkout must be exactly {SOURCE}")
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            SOURCE,
            "--",
            "robot_sf",
            "fast-pysf",
            "configs",
            "maps",
            "pyproject.toml",
            "uv.lock",
        ],
        check=True,
    )
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit("output must be empty; preserve previous records separately")
    # All real episode jobs below have only this explicitly allowed dev seed.
    assert 1001 <= SEED <= 1200
    from loguru import logger

    from robot_sf.benchmark.map_runner.map_runner import run_map_batch
    from robot_sf.benchmark.scenario_generation.replay_adapter import (
        dump_generated_scenario_yaml,
        materialize_generated_scenario,
    )
    from robot_sf.benchmark.scenario_generation.segment_extraction import extract_critical_segment
    from robot_sf.benchmark.step_trace_comparator import (
        canonical_step_trace_digest,
        find_first_trace_difference,
    )
    from robot_sf.training.scenario_loader import load_scenarios

    logger.remove()
    scenario = dict(
        copy.deepcopy(next(s for s in load_scenarios(root / MATRIX) if s["name"] == SCENARIO))
    )
    scenario["seeds"] = [SEED]
    write_json(out / "effective_scenario.json", scenario)
    options: dict[str, Any] = {
        "horizon": 600,
        "dt": 0.1,
        "record_forces": True,
        "algo": "social_force",
        "algo_config_path": CONFIG,
        "benchmark_profile": "experimental",
        "workers": 1,
        "resume": False,
        "record_simulation_step_trace": True,
    }
    schema = root / "robot_sf/benchmark/schemas/episode.schema.v1.json"
    rows = []
    for name in ("source", "source_repeat"):
        summary = run_map_batch(
            [scenario], out / f"{name}.jsonl", schema, scenario_path=root / MATRIX, **options
        )
        write_json(out / f"{name}_summary.json", summary)
        assert summary["successful_jobs"] == 1 and summary["failed_jobs"] == 0
        assert summary["benchmark_availability"] == {
            "availability_reason": None,
            "availability_status": "available",
            "benchmark_success": True,
            "execution_mode": "adapter",
            "readiness_status": "adapter",
        }
        rows.append(json.loads((out / f"{name}.jsonl").read_text()))
    source, repeat = rows
    trace = source["algorithm_metadata"]["simulation_step_trace"]["steps"]
    repeat_trace = repeat["algorithm_metadata"]["simulation_step_trace"]["steps"]
    difference = find_first_trace_difference(trace, repeat_trace)
    assert difference is None and source["outcome"] == repeat["outcome"]
    assert source["event_ledger"]["surrogate_events"]["near_miss"] is True
    entry = extract_critical_segment(
        {"episode_id": source["episode_id"], "seed": SEED, "source_map": MAP, "steps": trace},
        pre_margin_s=1,
        post_margin_s=1,
    )
    write_json(out / "entry.json", entry)
    frames = entry["segment"]["trace_frames"]
    result = materialize_generated_scenario(entry, max_episode_steps=len(frames) - 1)
    assert result.scenario_document is not None
    result.scenario_document["scenarios"][0]["map_file"] = os.path.relpath(root / MAP, out)
    (out / "window.yaml").write_text(dump_generated_scenario_yaml(result))
    start_index = next(i for i, frame in enumerate(trace) if frame["time_s"] == frames[0]["time_s"])
    expected = trace[start_index + 1 : start_index + len(frames)]
    report = {
        "source_revision": SOURCE,
        "source": episode_summary(source),
        "source_repeat": {
            "outcome_equal": True,
            "first_difference": difference,
            "trace_digest": canonical_step_trace_digest(trace),
            "repeat_trace_digest": canonical_step_trace_digest(repeat_trace),
        },
        "critical_window": {
            "start_s": frames[0]["time_s"],
            "end_s": frames[-1]["time_s"],
            "event": entry["criticality"],
            "frame_count": len(frames),
        },
        "materialization_status": result.status,
        "tolerance_m": TOLERANCE_M,
        "expected_first_near_miss_local_step": episode_summary(source)["first_near_miss_step"]
        - start_index
        - 1,
        "replays": [],
    }
    window_options = options.copy()
    window_options["horizon"] = len(frames) - 1
    for index in (1, 2):
        summary = run_map_batch(
            out / "window.yaml",
            out / f"window_replay_{index}.jsonl",
            schema,
            **window_options,
        )
        write_json(out / f"window_replay_{index}_summary.json", summary)
        replay = json.loads((out / f"window_replay_{index}.jsonl").read_text())
        actual = replay["algorithm_metadata"]["simulation_step_trace"]["steps"]
        pairs = list(zip(expected, actual, strict=False))
        max_robot = max(math.dist(s["robot"]["position"], a["robot"]["position"]) for s, a in pairs)
        max_ped = max(
            math.dist(p["position"], q["position"])
            for s, a in pairs
            for p, q in zip(s["pedestrians"], a["pedestrians"], strict=True)
        )
        report["replays"].append(
            {
                "index": index,
                "availability": summary["benchmark_availability"],
                **episode_summary(replay),
                "expected_step_count": len(expected),
                "paired_steps": len(pairs),
                "max_robot_error_on_paired_steps_m": max_robot,
                "max_pedestrian_slot_error_on_paired_steps_m": max_ped,
                "actor_binding_status": "unverified_reindexed_by_adapter",
                "length_pass": len(expected) == len(actual),
                "trajectory_pass": len(expected) == len(actual)
                and max_robot <= TOLERANCE_M
                and max_ped <= TOLERANCE_M,
                "trace_digest": canonical_step_trace_digest(actual),
                "reset": replay["algorithm_metadata"]["simulation_step_trace"]["reset"],
            }
        )
    report["stage"] = "exact_window_replay_parity"
    report["status"] = (
        "PASS"
        if all(
            r["trajectory_pass"]
            and r["first_near_miss_step"] == report["expected_first_near_miss_local_step"]
            for r in report["replays"]
        )
        else "FAIL"
    )
    report["downstream"] = {"persistence": "NOT_RUN", "corpus_admission": "NOT_RUN"}
    write_json(out / "comparison.json", report)
    print("FAIL: generated-window parity; downstream stages not run; see comparison.json")
    return 2 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
