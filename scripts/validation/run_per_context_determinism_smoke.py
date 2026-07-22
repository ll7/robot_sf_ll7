#!/usr/bin/env python3
"""CI-grade fixed-episode per-context determinism smoke test (issue #6126).

This script runs a fixed scenario/planner/seed/horizon episode twice in-process
with numerical thread-pin controls enabled, extracts canonical simulation step
traces, and asserts that the step traces are identical. It also provides a
negative test mode to verify actionable first-difference reporting upon trace
divergence.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from robot_sf._numerical_thread_env import pin_thread_env_for_determinism

# Force thread pinning before any heavy benchmark / scientific imports
THREAD_ENV = pin_thread_env_for_determinism()

from robot_sf.benchmark.map_runner import run_map_batch  # noqa: E402
from robot_sf.benchmark.runner import load_scenario_matrix  # noqa: E402
from robot_sf.benchmark.step_trace_comparator import (  # noqa: E402
    canonical_step_trace_digest,
    compare_step_traces,
)

DEFAULT_SCENARIO_PATH = "configs/scenarios/archetypes/classic_crossing.yaml"
DEFAULT_SCENARIO_ID = "classic_crossing_low"
DEFAULT_SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"
DEFAULT_PLANNER = "goal"
DEFAULT_SEED = 42
DEFAULT_HORIZON = 30


def run_single_episode_trace(
    scenario_path: str = DEFAULT_SCENARIO_PATH,
    scenario_id: str = DEFAULT_SCENARIO_ID,
    schema_path: str = DEFAULT_SCHEMA_PATH,
    planner: str = DEFAULT_PLANNER,
    seed: int = DEFAULT_SEED,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one fixed episode in-process and return (episode_row, step_trace)."""
    scenarios = load_scenario_matrix(scenario_path)
    matching_scenarios = [
        scenario
        for scenario in scenarios
        if str(scenario.get("scenario_id", scenario.get("name", ""))) == scenario_id
    ]
    if len(matching_scenarios) != 1:
        raise RuntimeError(
            f"Expected exactly one scenario_id={scenario_id!r} in {scenario_path!r}; "
            f"found {len(matching_scenarios)}."
        )

    selected_scenario = copy.deepcopy(matching_scenarios[0])
    selected_scenario["seeds"] = [seed]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "results.jsonl"
        summary = run_map_batch(
            [selected_scenario],
            out_path=str(out_file),
            schema_path=schema_path,
            scenario_path=scenario_path,
            horizon=horizon,
            algo=planner,
            record_simulation_step_trace=True,
            workers=1,
            resume=False,
        )
        with open(out_file, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

    written = summary.get("written")
    if len(rows) != 1 or written != 1:
        raise RuntimeError(
            "Per-context determinism smoke must execute exactly one episode; "
            f"runner reported written={written!r} and emitted {len(rows)} row(s)."
        )
    row = rows[0]

    algo_metadata = row.get("algorithm_metadata", {})
    actual_identity = {
        "scenario_id": row.get("scenario_id"),
        "planner": algo_metadata.get("algorithm") if isinstance(algo_metadata, dict) else None,
        "seed": row.get("seed"),
        "horizon": row.get("horizon"),
    }
    expected_identity = {
        "scenario_id": scenario_id,
        "planner": planner,
        "seed": seed,
        "horizon": horizon,
    }
    if actual_identity != expected_identity:
        raise RuntimeError(
            "Executed episode identity does not match the requested fixed contract: "
            f"expected={expected_identity!r}, actual={actual_identity!r}."
        )

    trace = algo_metadata.get("simulation_step_trace")
    if (
        not isinstance(trace, dict)
        or not isinstance(trace.get("steps"), list)
        or not trace["steps"]
    ):
        raise RuntimeError(
            "Episode row algorithm_metadata did not contain a valid simulation_step_trace."
        )
    return row, trace


def run_determinism_smoke(
    scenario_path: str = DEFAULT_SCENARIO_PATH,
    scenario_id: str = DEFAULT_SCENARIO_ID,
    schema_path: str = DEFAULT_SCHEMA_PATH,
    planner: str = DEFAULT_PLANNER,
    seed: int = DEFAULT_SEED,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Run two in-process episodes and verify per-context step trace identity."""
    t0 = time.perf_counter()
    row1, trace1 = run_single_episode_trace(
        scenario_path=scenario_path,
        scenario_id=scenario_id,
        schema_path=schema_path,
        planner=planner,
        seed=seed,
        horizon=horizon,
    )
    _row2, trace2 = run_single_episode_trace(
        scenario_path=scenario_path,
        scenario_id=scenario_id,
        schema_path=schema_path,
        planner=planner,
        seed=seed,
        horizon=horizon,
    )
    elapsed = time.perf_counter() - t0

    equal, diff_msg = compare_step_traces(trace1, trace2)
    if not equal:
        raise RuntimeError(
            f"Per-context determinism check failed for scenario='{scenario_path}' "
            f"scenario_id='{scenario_id}' planner='{planner}' seed={seed} horizon={horizon}:\n"
            f"{diff_msg}"
        )

    digest = canonical_step_trace_digest(trace1)
    step_count = len(trace1["steps"])
    return {
        "status": "pass",
        "scenario_path": scenario_path,
        "scenario_id": row1["scenario_id"],
        "planner": row1["algorithm_metadata"]["algorithm"],
        "seed": row1["seed"],
        "horizon": row1["horizon"],
        "step_count": step_count,
        "trace_sha256": digest,
        "thread_env": THREAD_ENV,
        "elapsed_sec": round(elapsed, 4),
    }


def run_negative_test_smoke(
    scenario_path: str = DEFAULT_SCENARIO_PATH,
    scenario_id: str = DEFAULT_SCENARIO_ID,
    schema_path: str = DEFAULT_SCHEMA_PATH,
    planner: str = DEFAULT_PLANNER,
    seed: int = DEFAULT_SEED,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Verify that a modified step trace produces an actionable first-difference report."""
    _row, trace1 = run_single_episode_trace(
        scenario_path=scenario_path,
        scenario_id=scenario_id,
        schema_path=schema_path,
        planner=planner,
        seed=seed,
        horizon=horizon,
    )
    trace2 = copy.deepcopy(trace1)

    # Artificially modify a field at step index 2
    if len(trace2["steps"]) > 2:
        trace2["steps"][2]["robot"]["position"][0] += 0.001
        expected_step = 2
    else:
        trace2["steps"][0]["robot"]["position"][0] += 0.001
        expected_step = 0

    equal, diff_msg = compare_step_traces(trace1, trace2)
    if equal or diff_msg is None:
        raise RuntimeError("Negative test failed: mutated trace was incorrectly reported as equal.")

    if f"steps[{expected_step}]" not in diff_msg:
        raise RuntimeError(
            f"Negative test failed: diff report did not identify step index {expected_step}.\n"
            f"Report: {diff_msg}"
        )

    return {
        "status": "pass",
        "negative_test": True,
        "expected_differing_step": expected_step,
        "diff_report": diff_msg,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run fixed-episode per-context determinism smoke check (issue #6126)."
    )
    parser.add_argument("--scenario-path", default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--scenario-id", default=DEFAULT_SCENARIO_ID)
    parser.add_argument("--schema-path", default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--planner", default=DEFAULT_PLANNER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--negative-test",
        action="store_true",
        help="Run negative test verifying actionable first-difference reporting.",
    )

    args = parser.parse_args()

    if args.negative_test:
        res = run_negative_test_smoke(
            scenario_path=args.scenario_path,
            scenario_id=args.scenario_id,
            schema_path=args.schema_path,
            planner=args.planner,
            seed=args.seed,
            horizon=args.horizon,
        )
        print("Negative test passed successfully:")
        print(json.dumps(res, indent=2))
        return 0

    try:
        res = run_determinism_smoke(
            scenario_path=args.scenario_path,
            scenario_id=args.scenario_id,
            schema_path=args.schema_path,
            planner=args.planner,
            seed=args.seed,
            horizon=args.horizon,
        )
        print("Per-context determinism smoke check passed:")
        print(json.dumps(res, indent=2))
        return 0
    except RuntimeError as exc:
        print(f"Per-context determinism smoke check FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
