#!/usr/bin/env python3
"""Prepare or smoke #10190; full execution requires explicit dispatch acknowledgement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import yaml

from robot_sf.benchmark.pedestrian_sensitivity import (
    metric_values,
    profile_scenario,
    require_dev_seeds,
    summarize,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/benchmarks/pedestrian_sensitivity_10190.yaml"
STUDY_FAILURE_EXCEPTIONS = (
    OSError,
    ValueError,
    RuntimeError,
    ImportError,
    LookupError,
    TypeError,
    ArithmeticError,
)


def sha256(path):
    """Return file identity."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Atomically replace a JSON receipt, prohibiting NaN."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(config_path, *, smoke=False, seeds=None, factors=None, check_dependencies=True):
    """Load release matrix/roster, bind authored horizons and refuse seed leakage.

    Returns:
        Study config, source-bound release config, matrix path, scenarios and roster.
    """
    from robot_sf.benchmark.camera_ready._config import _apply_scenario_horizon_schedule
    from robot_sf.training.scenario_loader import load_scenarios

    cfg = yaml.safe_load(config_path.read_text())
    if cfg["schema_version"] != "pedsens.v1" or cfg["ranking_metric"] != "success":
        raise ValueError("unsupported study contract")
    # Validate every declared study seed, even when smoke overrides execution.
    require_dev_seeds(cfg["seeds"])
    require_dev_seeds(cfg["smoke"]["seeds"])
    require_dev_seeds([cfg["bootstrap_seed"]])
    selected_seeds = (
        seeds if seeds is not None else (cfg["smoke"]["seeds"] if smoke else cfg["seeds"])
    )
    require_dev_seeds(selected_seeds)
    selected_factors = factors or list(cfg["profiles"])
    if len(set(selected_factors)) != len(selected_factors) or "legacy" not in selected_factors:
        raise ValueError("unique factors including legacy required")
    cfg["selected_seeds"] = selected_seeds
    cfg["selected_factors"] = selected_factors
    release = yaml.safe_load((ROOT / cfg["release_config"]).read_text())
    matrix_path = ROOT / release["scenario_matrix"]
    scenarios = _apply_scenario_horizon_schedule(
        [dict(s) for s in load_scenarios(matrix_path)],
        schedule_path=ROOT / release["scenario_horizons"],
        expected_sha256=release["scenario_horizons_sha256"],
        protocol_version="0.0.8",
    )
    planners = release["planners"]
    if smoke:
        scenarios = [s for s in scenarios if s["name"] == cfg["smoke"]["scenario"]]
        planners = [p for p in planners if p["key"] in cfg["smoke"]["planners"]]
        if len(scenarios) != 1 or len(planners) != 2 or len(selected_seeds) != 2:
            raise ValueError("smoke requires 1 scenario x 2 release planners x 2 dev seeds")
    for factor in selected_factors:
        if factor not in cfg["profiles"]:
            raise ValueError(f"unknown study factor: {factor}")
        if check_dependencies:
            profile_scenario(scenarios[0], cfg["profiles"][factor], selected_seeds[0])
    return cfg, release, matrix_path, scenarios, planners


def run_slot(job):
    """Execute one native schema-validated episode with fail-fast prerequisites.

    Returns:
        A paired summary row and elapsed core-seconds for this slot.
    """
    from robot_sf.benchmark.map_runner.map_runner import run_map_batch
    from scripts.benchmark.run_issue_8871_pedestrian_speed_canary import _execution_disposition

    factor, planner, scenario, seed, profile, matrix_path, out, release = job
    require_dev_seeds([seed])  # Worker boundary, before native dispatch.
    start = time.monotonic()
    bound = profile_scenario(scenario, profile, seed)
    bound["seeds"] = [seed]
    path = out / factor / planner["key"] / f"{scenario['name']}--{seed}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    result = run_map_batch(
        [bound],
        path,
        ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json",
        scenario_path=matrix_path,
        provenance_scenario_path=matrix_path,
        horizon=None,
        dt=release["dt"],
        record_forces=release["record_forces"],
        algo=planner["algo"],
        algo_config_path=planner.get("algo_config"),
        benchmark_profile=planner["benchmark_profile"],
        socnav_missing_prereq_policy="fail-fast",
        adapter_impact_eval=planner.get("adapter_impact_eval", False),
        workers=1,
        resume=False,
        record_simulation_step_trace=True,
    )
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 1 or result.get("failures"):
        raise ValueError(
            f"native episode failed: {factor}/{planner['key']}/{scenario['name']}/{seed}"
        )
    record = records[0]
    status, reason = _execution_disposition(record)
    if status != "native":
        raise ValueError(f"non-native episode refused: {status}: {reason}")
    if record["seed"] != seed or record["scenario_id"] != scenario["name"]:
        raise ValueError("native episode identity mismatch")
    return {
        "factor": factor,
        "planner": planner["key"],
        "scenario": scenario["name"],
        "seed": seed,
        "values": metric_values(record),
        "episode_id": record["episode_id"],
        "episode_path": path.relative_to(out).as_posix(),
        "execution_mode": record["algorithm_metadata"]["planner_kinematics"]["execution_mode"],
        "elapsed_core_seconds": time.monotonic() - start,
        "steps": record["steps"],
        "failure_evidence": {
            "termination_reason": record["termination_reason"],
            "exact_events": record["event_ledger"]["exact_events"],
            "collision_events": record["event_ledger"]["collision_events"],
            "spawn_validity": record["spawn_validity"],
        },
    }


def main(argv=None):
    """CLI with plan default, bounded smoke, and guarded future study mode.

    Returns:
        Zero after successful preparation or execution.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("plan", "smoke", "study"), default="plan")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("-n", "--workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--factors", nargs="+")
    parser.add_argument("--after-sealed-dispatch", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.mode != "study" and args.workers > 2):
        parser.error("plan/smoke supports workers 1..2 only")
    if args.mode == "study" and not args.after_sealed_dispatch:
        parser.error("full study starts only after 0.0.8 sealed main AND doorway dispatch")
    os.chdir(ROOT)
    cfg, release, matrix, scenarios, planners = prepare(
        args.config,
        smoke=args.mode == "smoke",
        seeds=args.seeds,
        factors=args.factors,
        check_dependencies=args.mode != "plan",
    )
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("output exists; retain partial evidence and use a fresh directory")
    args.out.mkdir(parents=True)
    jobs = [
        (factor, planner, scenario, seed, cfg["profiles"][factor], matrix, args.out, release)
        for factor in cfg["selected_factors"]
        for planner in planners
        for scenario in scenarios
        for seed in cfg["selected_seeds"]
    ]
    inputs = {cfg["release_config"], release["scenario_matrix"], release["scenario_horizons"]}
    inputs.update(p["algo_config"] for p in planners if p.get("algo_config"))
    inputs.add(args.config.resolve().relative_to(ROOT).as_posix())
    manifest = {
        "schema_version": "pedsens-run.v1",
        "status": "planned",
        "mode": args.mode,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        },
        "config": cfg,
        "input_sha256": {name: sha256(ROOT / name) for name in sorted(inputs)},
        "episode_count": len(jobs),
        "scenarios": [s["name"] for s in scenarios],
        "planners": deepcopy(planners),
        "workers": args.workers,
        "evidence_status": "smoke evidence" if args.mode == "smoke" else "diagnostic-only",
    }
    write_json(args.out / "manifest.json", manifest)
    if args.mode == "plan":
        print(json.dumps({"status": "planned", "episodes": len(jobs)}))
        return 0
    manifest["status"] = "running"
    write_json(args.out / "manifest.json", manifest)
    start = time.monotonic()
    try:
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            with (args.out / "paired_rows.jsonl").open("w") as handle:
                for row in pool.map(run_slot, jobs):
                    rows.append(row)
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
        summary = summarize(
            rows,
            samples=cfg["bootstrap_samples"],
            confidence=cfg["confidence"],
            bootstrap_seed=cfg["bootstrap_seed"],
        )
        if args.mode == "smoke":
            summary["evidence_status"] = "smoke evidence"
        from robot_sf.benchmark.pedestrian_manipulation_checks import manipulation_checks

        for factor in cfg["selected_factors"]:
            write_json(
                args.out / f"manipulation_{factor}.json",
                manipulation_checks(cfg["profiles"][factor], cfg["selected_seeds"]),
            )
        write_json(args.out / "summary.json", summary)
        core_seconds = sum(row["elapsed_core_seconds"] for row in rows)
        # 120960 episodes in the six-factor full grid. Provision 2x measured slot cost.
        summary["runtime"] = {
            "wall_seconds": time.monotonic() - start,
            "episode_core_seconds": core_seconds,
            "mean_slot_seconds": core_seconds / len(rows),
            "estimated_full_core_hours_2x": core_seconds / len(rows) * 120960 / 3600 * 2,
            "estimate_limitations": "Only goal/social_force on one scenario measured; "
            "learned/search planners and matrix costs unmeasured. Not a resource guarantee.",
        }
        write_json(args.out / "summary.json", summary)
        manifest["status"] = "complete"
    except STUDY_FAILURE_EXCEPTIONS as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        write_json(args.out / "manifest.json", manifest)
        raise
    manifest["output_sha256"] = {
        path.relative_to(args.out).as_posix(): sha256(path)
        for path in sorted(args.out.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "episodes": len(rows), "runtime": summary["runtime"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
