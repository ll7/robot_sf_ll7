"""Capture and replay the seed-1001 window twice using the committed repair.

This is a one-case development recipe. It produces real simulation records,
then submits them unchanged to the existing persistence CLI. Rejection is an
expected diagnostic result: that CLI compares whole-episode identities/traces
and has no continuation-window admission contract. No corpus is admitted.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

FIX = "1baaa3eb08cfbc1866f18702eb939c1f60b400fd"
MATRIX = "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
CONFIG = "configs/algos/social_force_release_v0_0_8.yaml"
MAP = "maps/successor_svg_maps/issue_9762_classic_crossing_goal_zone_entry_v2.svg"


def write_json(path: Path, value: object) -> None:
    """Write a readable evidence receipt."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:  # noqa: PLR0915 - explicit one-case stage order
    """Produce source, checkpoint, two fresh replays and a fail-closed gate receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    evidence = Path(__file__).resolve().parent.parent
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit("output must be empty; preserve previous runs")
    fix = subprocess.check_output(["git", "rev-parse", FIX], text=True).strip()
    subprocess.run(
        ["git", "diff", "--exit-code", fix, "--", "robot_sf", "fast-pysf", "configs", "maps"],
        check=True,
    )
    historical = json.loads((evidence / "provenance.json").read_text())
    for name, digest in historical["source_input_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    reference = json.loads(gzip.decompress((evidence / "source.jsonl.gz").read_bytes()))

    import yaml
    from loguru import logger

    from robot_sf.benchmark.map_runner.map_runner import build_map_policy, run_map_batch
    from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode
    from robot_sf.benchmark.scenario_generation.continuation import capture_window_checkpoint
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
    scenario = copy.deepcopy(
        next(s for s in load_scenarios(root / MATRIX) if s["name"] == "classic_cross_trap_high")
    )
    scenario["seeds"] = [1001]
    checkpoint = out / "checkpoint.json"

    def capture(**context) -> None:
        if context["step_idx"] == 370:
            capture_window_checkpoint(
                checkpoint,
                env=context["env"],
                state=context["state"],
                loop_config=context["loop_config"],
                source_episode_id=reference["episode_id"],
                seed=1001,
                source_map=root / MAP,
                window_steps=20,
            )

    source = run_map_episode(
        scenario,
        1001,
        horizon=600,
        dt=0.1,
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        algo="social_force",
        scenario_path=root / MATRIX,
        algo_config=yaml.safe_load((root / CONFIG).read_text()),
        algo_config_path=CONFIG,
        record_simulation_step_trace=True,
        before_step_hook=capture,
        policy_builder=build_map_policy,
    )
    (out / "source.jsonl").write_text(json.dumps(source, sort_keys=True) + "\n")
    trace = source["algorithm_metadata"]["simulation_step_trace"]["steps"]
    gold = reference["algorithm_metadata"]["simulation_step_trace"]["steps"]
    assert trace == gold and source["outcome"] == reference["outcome"]
    entry = extract_critical_segment(
        {"episode_id": source["episode_id"], "seed": 1001, "source_map": MAP, "steps": trace},
        pre_margin_s=1,
        post_margin_s=1,
    )
    assert entry == json.loads((evidence / "entry.json").read_text())
    write_json(out / "entry.json", entry)
    result = materialize_generated_scenario(
        entry, max_episode_steps=20, source_scenario=scenario, continuation_checkpoint=checkpoint
    )
    assert result.scenario_document is not None
    materialized = result.scenario_document["scenarios"][0]
    materialized["map_file"] = os.path.relpath(root / MAP, out)
    materialized["generated_replay"]["checkpoint_file"] = "checkpoint.json"
    (out / "window.yaml").write_text(dump_generated_scenario_yaml(result))
    expected = gold[370:390]
    comparisons = []
    replays = []
    for index in (1, 2):
        output = out / f"window_replay_{index}.jsonl"
        summary = run_map_batch(
            out / "window.yaml",
            output,
            root / "robot_sf/benchmark/schemas/episode.schema.v1.json",
            horizon=600,
            dt=0.1,
            record_forces=True,
            algo="social_force",
            algo_config_path=CONFIG,
            benchmark_profile="experimental",
            record_simulation_step_trace=True,
            resume=False,
            workers=1,
        )
        assert summary["successful_jobs"] == 1 and summary["failed_jobs"] == 0
        replay = json.loads(output.read_text())
        replays.append(replay)
        actual = replay["algorithm_metadata"]["simulation_step_trace"]["steps"]
        assert actual == expected
        assert replay["metrics"]["near_misses"] == 11
        assert replay["metrics"]["min_clearance"] == 0.47727323760374385
        assert replay["outcome"] == source["outcome"]
        robot_error = max(
            math.dist(a["robot"]["position"], b["robot"]["position"])
            for a, b in zip(expected, actual, strict=True)
        )
        ped_error = max(
            math.dist(p["position"], q["position"])
            for a, b in zip(expected, actual, strict=True)
            for p, q in zip(a["pedestrians"], b["pedestrians"], strict=True)
        )
        comparisons.append(
            {
                "index": index,
                "status": "PASS",
                "step_count": len(actual),
                "first_difference": find_first_trace_difference(expected, actual),
                "trace_digest": canonical_step_trace_digest(actual),
                "outcome": replay["outcome"],
                "near_misses": 11,
                "min_clearance_m": replay["metrics"]["min_clearance"],
                "first_near_miss_global_step": actual[4]["step"],
                "first_near_miss_time_s": 37.5,
                "first_near_miss_local_step": 4,
                "first_contact_step": None,
                "max_robot_error_m": robot_error,
                "max_pedestrian_error_m": ped_error,
                "continuation": replay["algorithm_metadata"]["generated_window_continuation"],
                "availability": summary["benchmark_availability"],
            }
        )
    write_json(
        out / "comparison.json",
        {
            "status": "PASS",
            "fix_commit": fix,
            "tolerance_m": 1e-9,
            "source_full_trace_digest": canonical_step_trace_digest(trace),
            "source_window_trace_digest": canonical_step_trace_digest(expected),
            "window_boundary_step": 370,
            "original_budget_steps": 600,
            "window_stop": "censored after twenty transitions; not original episode termination",
            "replays": comparisons,
        },
    )
    # Add explicit identity fields from the actual runner provenance only. Do
    # not rename the replay episode or truncate the source to appease this gate.
    for row in [source, *replays]:
        row["source_seed"] = row["seed"]
        row["source_map"] = MAP
    (out / "gate_source.jsonl").write_text(json.dumps(source) + "\n")
    cells = [
        {
            "timing_offset_s": t,
            "speed_delta_m_s": s,
            "verdict": "unknown",
            "reason": "independent simulator perturbation not executed",
        }
        for t in (-0.25, 0.0, 0.25)
        for s in (-0.2, 0.0, 0.2)
    ]
    write_json(
        out / "gate_replay_results.json",
        {entry["scenario_id"]: {"episode": replays[0], "cells": cells}},
    )
    command = [
        ".venv/bin/python",
        "scripts/tools/evaluate_pipeline_persistence_gate.py",
        "--candidate",
        str(out / "entry.json"),
        "--episodes",
        str(out / "gate_source.jsonl"),
        "--config",
        "configs/analysis/issue_5600_persistence_gate.yaml",
        "--output",
        str(out / "persistence"),
        "--replay-results",
        str(out / "gate_replay_results.json"),
        "--commit-hash",
        fix,
    ]
    gate = subprocess.run(command, capture_output=True, text=True, check=False)
    write_json(
        out / "gate_command.json",
        {
            "argv": command,
            "returncode": gate.returncode,
            "stdout": gate.stdout,
            "stderr": gate.stderr,
        },
    )
    print("PASS: two exact window replays; persistence/admission diagnostic in gate_command.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
