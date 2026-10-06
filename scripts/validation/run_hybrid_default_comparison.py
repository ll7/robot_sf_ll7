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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from importlib.metadata import distributions
from pathlib import Path

from loguru import logger

from robot_sf.common.hybrid_defaults import defaults_for_source
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_empty_world_sweep import assert_dev_seeds
from scripts.validation.run_hybrid_feasibility_diagnostics import (
    ARM_SWITCHES,
    CANDIDATE,
    MAIN_MATRIX,
    ROOT,
    WIDTH_MATRIX,
    load_cells,
    run_cell,
)
from scripts.validation.run_policy_search_candidate import (
    _DEFAULT_REGISTRY,
    load_candidate_definition,
)

PER_SWITCH_ARMS = (
    "off",
    "static_only",
    "sensor_only",
    "goal_validity_with_sensor",
    "current_defaults",
)
SWITCH_NAMES = (
    "physical_static_exclusion_enabled",
    "goal_next_validity_enabled",
    "include_goal_next_valid",
)


def run_pair_cell(task):
    """Run the real diagnostic episode with current fill-in and explicit old controls.

    Returns:
        Native outcome and complete diagnostics for one paired cell.
    """
    with defaults_for_source(None):
        result = run_cell(task)
    validate_result(result, task)
    return result


def validate_result(result, task):
    """Reject a completed record with a different admitted cell or default contract."""
    name, seed, arm, empty, _, _ = task
    if (result["scenario"], result["seed"], result["arm"], result["empty"]) != (
        name,
        seed,
        arm,
        empty,
    ):
        raise RuntimeError("Completed record has a different scenario/seed/arm identity")
    expected = dict(zip(SWITCH_NAMES, ARM_SWITCHES[arm], strict=True))
    if result["effective_switches"] != expected:
        raise RuntimeError("Executed switch values differ from the admitted arm")
    if result["default_policy"]["default_set"] != "current":
        raise RuntimeError("Comparison did not apply the admitted current default set")


def load_completed_result(path, task, producer_head):
    """Read a completed record only when its cell and producer identity match.

    Returns:
        A verified native record eligible for continuation of this same campaign.
    """
    result = json.loads(path.read_text())
    validate_result(result, task)
    if result["execution_head"] != producer_head:
        raise RuntimeError("Completed record has a different producer revision")
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
    evaluated = [r["debug"] for r in rows if (r.get("debug") or {}).get("candidate_count", 0)]
    tail = rows[-min(100, len(rows)) :]
    if contacts:
        category = "executed_contact"
    elif tail and all((r.get("decision") or {}).get("planner_mode") == "GOAL_STOP" for r in tail):
        category = "goal_stop_before_route_completion"
    elif tail and all(
        (r.get("debug") or {}).get("candidate_count", 0) > 0
        and r["debug"]["feasible_moving_count"] == 0
        for r in tail
    ):
        category = "forced_stop_timeout"
    elif result["metrics"]["freezing"]:
        category = "low_progress_or_livelock_timeout"
    else:
        category = "horizon_exhausted_with_moving_candidates"
    evidence = {
        "classification": category,
        "contacts": contacts,
        "last_feasible_moving_count": evaluated[-1]["feasible_moving_count"] if evaluated else None,
        "last_decision_mode": (rows[-1].get("decision") or {}).get("planner_mode")
        if rows
        else None,
        "final_goal_distance_m": result["final_goal_distance_m"],
        "no_feasible_moving_s": result["metrics"]["no_feasible_moving_s"],
        "stopped_time_fraction": result["metrics"]["stopped_time_fraction"],
    }
    if contacts:
        contact_index = next(i for i, row in enumerate(rows) if row["collision_types"])
        contact_row = rows[contact_index]
        stationary = contact_row["displacement_m"] <= 1e-6
        stopped_steps = 0
        for row in reversed(rows[: contact_index + 1]):
            if row["displacement_m"] > 1e-6:
                break
            stopped_steps += 1
        contact_class = {
            "is_pedestrian_collision": (
                "pedestrian_contact_while_robot_stationary"
                if stationary
                else "pedestrian_contact_while_robot_moving"
            ),
            "is_obstacle_collision": "obstacle_contact",
            "is_robot_collision": "robot_contact",
        }
        evidence["contact_class"] = (
            "mixed_contact" if len(contacts) > 1 else contact_class[contacts[0]]
        )
        evidence["contact_time_s"] = (contact_row["step"] + 1) * 0.1
        evidence["robot_displacement_on_contact_m"] = contact_row["displacement_m"]
        evidence["stationary_before_contact_s"] = stopped_steps * 0.1
    return evidence


def summarize_per_switch(results, output):
    """Compare each executable switch configuration per scenario and against all-off.

    Returns:
        Complete arm accounting, paired times, scenario rows and all failed cells.
    """

    def counts(values):
        times = [r["duration_s"] for r in values if r["outcome"] == "success"]
        return {
            "episodes": len(values),
            "successes": sum(r["outcome"] == "success" for r in values),
            "collisions": sum(r["outcome"] == "collision" for r in values),
            "timeouts": sum(r["outcome"] == "timeout" for r in values),
            "mean_success_time_s": sum(times) / len(times) if times else None,
        }

    crowd = [r for r in results if not r["empty"]]
    cells = {(r["scenario"], r["seed"], r["arm"]): r for r in crowd}
    summary = {
        "evidence_status": "diagnostic-only",
        "switches": {
            a: dict(zip(SWITCH_NAMES, ARM_SWITCHES[a], strict=True)) for a in PER_SWITCH_ARMS
        },
        "invalid_combination": {
            "switches": dict(zip(SWITCH_NAMES, (False, True, False), strict=True)),
            "status": "invalid_observation_contract",
            "reason": "Goal validity requires next_valid; literal goal-only fails closed. The executable validity arm includes its required sensor, compared separately with sensor-only.",
        },
        "arms": {},
        "per_scenario": {},
        "failures": [],
    }
    for arm in PER_SWITCH_ARMS:
        values = [r for r in crowd if r["arm"] == arm]
        record = counts(values)
        deltas = [
            r["duration_s"] - cells[(r["scenario"], r["seed"], "off")]["duration_s"]
            for r in values
            if r["outcome"] == "success"
            and cells[(r["scenario"], r["seed"], "off")]["outcome"] == "success"
        ]
        record["paired_successes_with_off"] = len(deltas)
        record["paired_mean_time_delta_s"] = sum(deltas) / len(deltas) if deltas else None
        record["new_failures_against_off"] = [
            {"scenario": r["scenario"], "seed": r["seed"], "outcome": r["outcome"]}
            for r in values
            if r["outcome"] != "success"
            and cells[(r["scenario"], r["seed"], "off")]["outcome"] == "success"
        ]
        summary["arms"][arm] = record
    for scenario in sorted({r["scenario"] for r in crowd}):
        summary["per_scenario"][scenario] = {
            arm: counts([r for r in crowd if r["scenario"] == scenario and r["arm"] == arm])
            for arm in PER_SWITCH_ARMS
        }
    for result in results:
        failure = classify(result, output)
        if failure:
            summary["failures"].append(
                {k: result[k] for k in ("scenario", "seed", "arm", "empty", "outcome")} | failure
            )
    summary["empty_world"] = summarize([r for r in results if r["empty"]], output)["comparisons"][
        "empty"
    ]
    return summary


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


def run_pending(pending, workers, already_complete, total):
    """Keep only the admitted number of simulations in flight.

    Returns:
        Completed native results; any failed future aborts without a queued campaign.
    """
    results = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        remaining = iter(pending)
        futures = {}
        for task in [next(remaining, None) for _ in range(workers)]:
            if task is not None:
                futures[pool.submit(run_pair_cell, task)] = task
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                del futures[future]
                results.append(future.result())
                print(
                    f"completed {already_complete + len(results)}/{total} elapsed_s={time.monotonic() - start:.1f}",
                    flush=True,
                )
                task = next(remaining, None)
                if task is not None:
                    futures[pool.submit(run_pair_cell, task)] = task
    return results


def main():
    """Resolve the development-only experiment and run no more than two simulations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Small development canary (four highlighted scenarios with --per-switch)",
    )
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--per-switch",
        action="store_true",
        help="Five executable arms, with the validity sensor dependency explicit",
    )
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
            selected, seeds = (
                (
                    [
                        "francis2023_narrow_hallway",
                        "francis2023_robot_crowding",
                        "classic_bottleneck_high",
                        "francis2023_exiting_room",
                    ],
                    [1001] if empty else [1001, 1013, 1020],
                )
                if args.per_switch
                else (names[:2], [1001])
            )
        for name in selected:
            horizon = int(cells[name][0]["simulation_config"]["max_episode_steps"])
            for seed in seeds:
                for arm in (
                    PER_SWITCH_ARMS
                    if args.per_switch and not empty
                    else ("off", "current_defaults")
                ):
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
    files.extend(sorted((ROOT / "robot_sf").rglob("*.svg")))
    files.extend(sorted((ROOT / "fast-pysf/pysocialforce").rglob("*.py")))
    files.extend(sorted((ROOT / "maps").rglob("*.svg")))
    files.extend(sorted((ROOT / "configs/scenarios").rglob("*.yaml")))
    _, candidate, _, candidate_path = load_candidate_definition(ROOT / _DEFAULT_REGISTRY, CANDIDATE)
    files.extend([ROOT / _DEFAULT_REGISTRY, candidate_path, ROOT / candidate["base_config_path"]])
    files.extend(
        ROOT / f"scripts/validation/{name}.py"
        for name in (
            "run_empty_world_sweep",
            "run_policy_search_candidate",
            "run_policy_search_step_diagnostics",
        )
    )
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
        "per_switch": args.per_switch,
        "executable_arms": list(PER_SWITCH_ARMS)
        if args.per_switch
        else ["off", "current_defaults"],
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
            results.append(load_completed_result(path, task, manifest["head"]))
        else:
            pending.append(task)
    if pending and args.summarize_only:
        raise RuntimeError(f"Incomplete comparison: {len(pending)} missing episodes")
    results.extend(run_pending(pending, args.workers, len(results), len(tasks)))
    summary = (
        summarize_per_switch(results, args.output)
        if args.per_switch
        else summarize(results, args.output)
    )
    summary["manifest"] = manifest
    summary["complete"] = len(results) == len(tasks)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["arms"] if args.per_switch else summary["comparisons"], indent=2))


if __name__ == "__main__":
    main()
