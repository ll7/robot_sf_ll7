"""Paired native default comparison on the standard development scenario set.

Run with --workers 2. Empty-world uses the sweep's two development seeds;
crowded comparisons use all thirty development seeds. No release seeds resolve
or execute. Raw per-step traces support every failure classification.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import distributions
from pathlib import Path

from loguru import logger

from robot_sf.common.hybrid_defaults import defaults_for_source
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_empty_world_sweep import assert_dev_seeds
from scripts.validation.run_hybrid_feasibility_diagnostics import (
    MAIN_MATRIX,
    ROOT,
    WIDTH_MATRIX,
    load_cells,
    run_cell,
)


def run_pair_cell(task):
    """Run the real diagnostic episode with current fill-in and explicit old controls.

    Returns:
        Native outcome and complete diagnostics for one paired cell.
    """
    with defaults_for_source(None):
        result = run_cell(task)
    expected = task[2] == "current_defaults"
    if set(result["effective_switches"].values()) != {expected}:
        raise RuntimeError("Executed switch values differ from the admitted arm")
    result["default_policy"] = {"default_set": "current", "explicit_old_overrides": not expected}
    return result


def classify(result, output):
    """Classify an observed failure from executed contacts and feasibility traces.

    Returns:
        Diagnostic failure class and its direct trace evidence, or None for success.
    """
    if result["outcome"] == "success":
        return None
    suffix = "empty" if result["empty"] else "crowd"
    path = output / f"{result['scenario']}__{result['seed']}__{result['arm']}__{suffix}.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    contacts = sorted({k for r in rows for k in r["collision_types"]})
    feasible = [(r.get("debug") or {}).get("feasible_moving_count", 0) for r in rows]
    if contacts:
        category = "executed_contact"
    elif feasible and all(v == 0 for v in feasible[-min(100, len(feasible)) :]):
        category = "forced_stop_timeout"
    elif result["metrics"]["freezing"]:
        category = "low_progress_or_livelock_timeout"
    else:
        category = "horizon_exhausted_with_moving_candidates"
    return {
        "classification": category,
        "contacts": contacts,
        "last_feasible_moving_count": feasible[-1] if feasible else None,
        "final_goal_distance_m": result["final_goal_distance_m"],
        "no_feasible_moving_s": result["metrics"]["no_feasible_moving_s"],
        "stopped_time_fraction": result["metrics"]["stopped_time_fraction"],
    }


def summarize(results, output):
    """Publish complete pair accounting and descriptive development deltas.

    Returns:
        Summary with pooled outcomes, paired success times and all failure classes.
    """
    summary = {"evidence_status": "diagnostic-only", "failures": [], "comparisons": {}}
    for empty in (True, False):
        subset = [r for r in results if r["empty"] is empty]
        arms = {}
        for arm in ("off", "current_defaults"):
            values = [r for r in subset if r["arm"] == arm]
            successes = [r["duration_s"] for r in values if r["outcome"] == "success"]
            arms[arm] = {
                "episodes": len(values),
                "successes": sum(r["outcome"] == "success" for r in values),
                "collisions": sum(r["outcome"] == "collision" for r in values),
                "timeouts": sum(r["outcome"] == "timeout" for r in values),
                "mean_success_time_s": sum(successes) / len(successes) if successes else None,
            }
        paired = {}
        for r in subset:
            paired.setdefault((r["scenario"], r["seed"]), {})[r["arm"]] = r
        times = [
            p["current_defaults"]["duration_s"] - p["off"]["duration_s"]
            for p in paired.values()
            if len(p) == 2 and all(r["outcome"] == "success" for r in p.values())
        ]
        summary["comparisons"]["empty" if empty else "crowd"] = {
            "arms": arms,
            "success_count_delta": arms["current_defaults"]["successes"] - arms["off"]["successes"],
            "collision_count_delta": arms["current_defaults"]["collisions"]
            - arms["off"]["collisions"],
            "paired_successes": len(times),
            "paired_mean_time_delta_s": sum(times) / len(times) if times else None,
            "new_failures": [
                {
                    "scenario": p["current_defaults"]["scenario"],
                    "seed": p["current_defaults"]["seed"],
                }
                for p in paired.values()
                if len(p) == 2
                and p["off"]["outcome"] == "success"
                and p["current_defaults"]["outcome"] != "success"
            ],
        }
    for result in results:
        failure = classify(result, output)
        if failure:
            summary["failures"].append(
                {k: result[k] for k in ("scenario", "seed", "arm", "empty", "outcome")} | failure
            )
    return summary


def main():  # noqa: C901 - one bounded experiment orchestration path
    """Resolve the development-only experiment and run no more than two simulations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--probe", action="store_true", help="First two standard scenarios at seed 1001 only"
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = [s["name"] for s in load_scenarios(MAIN_MATRIX)]
    widths = [s["name"] for s in load_scenarios(WIDTH_MATRIX)]
    cells = load_cells(names + widths)
    tasks = []
    for empty in (True, False):
        selected = names + widths if empty else names
        seeds = assert_dev_seeds([1001, 1002] if empty else list(range(1001, 1031)))
        if args.probe:
            selected, seeds = names[:2], [1001]
        for name in selected:
            horizon = int(cells[name][0]["simulation_config"]["max_episode_steps"])
            for seed in seeds:
                for arm in ("off", "current_defaults"):
                    tasks.append((name, seed, arm, empty, str(args.output), horizon))
    files = [
        Path(__file__),
        ROOT / "scripts/validation/run_hybrid_feasibility_diagnostics.py",
        ROOT / "robot_sf/common/hybrid_defaults.py",
        ROOT / "robot_sf/common/legacy_hybrid_defaults.json",
        ROOT / "robot_sf/planner/hybrid_rule_local_planner.py",
        ROOT / "robot_sf/gym_env/unified_config.py",
        MAIN_MATRIX,
        WIDTH_MATRIX,
    ]
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    files.extend(sorted((ROOT / "robot_sf").rglob("*.py")))
    files.extend([ROOT / "uv.lock", ROOT / "pyproject.toml"])
    manifest = {
        "status": "diagnostic-only",
        "python": sys.version,
        "dependencies": [
            list(pair)
            for pair in sorted({(d.metadata["Name"], d.version) for d in distributions()})
        ],
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seeds": list(range(1001, 1031)),
        "empty_seeds": [1001, 1002],
        "workers": args.workers,
        "expected_episodes": len(tasks),
        "standard_scenarios": names,
        "empty_width_scenarios": widths,
        "hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Existing output has a different source/input identity")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    pending = []
    for task in tasks:
        n, s, a, e, _, _ = task
        path = args.output / f"{n}__{s}__{a}__{'empty' if e else 'crowd'}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
        else:
            pending.append(task)
    if pending and args.summarize_only:
        raise RuntimeError(f"Incomplete comparison: {len(pending)} missing episodes")
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_pair_cell, t): t for t in pending}
        for future in as_completed(futures):
            results.append(future.result())
            print(
                f"completed {len(results)}/{len(tasks)} elapsed_s={time.monotonic() - start:.1f}",
                flush=True,
            )
    summary = summarize(results, args.output)
    summary["manifest"] = manifest
    summary["complete"] = len(results) == len(tasks)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["comparisons"], indent=2))


if __name__ == "__main__":
    main()
