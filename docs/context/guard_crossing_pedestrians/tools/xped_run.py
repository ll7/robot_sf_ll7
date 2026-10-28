"""Run one (arm, scenario) job of the guard / crossing-pedestrian diagnostic.

Robot arms use the unmodified benchmark episode runner (``run_batch`` -> map runner,
serial, one process per job). This script only wraps three private runner functions
at import time to (1) record true pedestrian and robot positions after reset and after
every step, and (2) continue the same simulator after the robot step loop returns,
with the declared parking rule, until the fixed observation window ends. The runner
assembles the episode record after this continuation. Invariance of recorded robot
fields is an empirical comparison, not a consequence of an earlier record write.

The no-robot arm resets the same scenario/seed through the benchmark env builder,
removes the robot from the scene and steps pedestrians only.

Usage (repo root):
  python docs/context/guard_crossing_pedestrians/tools/xped_run.py \
      --arm G1E1 --scenario classic_doorway_medium__traverse --seeds 1001,1002 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xped_lib as L

ARMS = {
    "G1E1": ("hybrid_rule_local_planner", L.PACKET / "inputs/G1E1.yaml", "experimental"),
    "G0E1": ("hybrid_rule_local_planner", L.PACKET / "inputs/G0E1.yaml", "experimental"),
    "orca": ("orca", Path("configs/algos/orca_release_v0_0_8.yaml"), "baseline-safe"),
}
CTX: dict = {}


def _save(out_dir: Path, payload: dict) -> None:
    name = f"{CTX['arm']}__{CTX['scenario']}__s{CTX['seed']}.npz"
    np.savez_compressed(out_dir / "traj" / name, **payload)


def install_wrappers(out_dir: Path, arm: str) -> None:
    """Capture trajectories and extend observation after the robot step loop."""
    import robot_sf.benchmark.map_runner.map_runner_episode as E

    orig_setup = E._setup_and_run_step_loop
    orig_snap = E._step_snapshot_and_record
    orig_exec = E._execute_step_loop

    def setup_wrapper(args: E._StepLoopSetupArgs) -> E._EpisodeStepLoopResult:
        if args.scenario is None:
            raise ValueError("The crossing-pedestrian wrapper requires a scenario")
        CTX.update(arm=arm, seed=int(args.seed), scenario=str(args.scenario.get("name")))
        CTX["frames"], CTX["robot"] = [], []
        return orig_setup(args)

    def snap_wrapper(state: E._StepLoopState, slc: E._StepLoopConfig, *, env: Any, obs: Any):
        result = orig_snap(state, slc, env=env, obs=obs)
        CTX["frames"].append(L.ped_xy(env))
        CTX["robot"].append(L.robot_xy(env))
        return result

    def exec_wrapper(
        state: E._StepLoopState,
        slc: E._StepLoopConfig,
        *,
        env: Any,
        planner_stats: Any,
        horizon_val: int,
        normalize_budget_timeout: bool = True,
    ) -> None:
        CTX["frames"], CTX["robot"] = [L.ped_xy(env)], [L.robot_xy(env)]
        orig_exec(
            state,
            slc,
            env=env,
            planner_stats=planner_stats,
            horizon_val=horizon_val,
            normalize_budget_timeout=normalize_budget_timeout,
        )
        robot_steps = len(CTX["frames"]) - 1
        reason = str(state.termination_reason)
        # Declared parking rule: a robot that completed its route leaves the scene;
        # any other termination (collision etc.) stops the robot where it is.
        remove = reason == "success" or CTX.get("failure_parking") == "remove"
        if robot_steps < L.WINDOW_STEPS:
            L.park_robot(env, remove=remove)
        for _ in range(L.WINDOW_STEPS - robot_steps):
            L.step_peds_only(env)
            CTX["frames"].append(L.ped_xy(env))
            CTX["robot"].append(L.robot_xy(env))
        _save(
            out_dir,
            {
                "ped_xy": np.stack(CTX["frames"]).astype(np.float32),
                "robot_xy": np.stack(CTX["robot"]).astype(np.float32),
                "robot_steps": np.int32(robot_steps),
                "termination_reason": np.str_(reason),
                "parking": np.str_(
                    "none"
                    if robot_steps >= L.WINDOW_STEPS
                    else ("removed" if remove else "stopped")
                ),
                "sim_config_sha256": np.str_(L.sim_config_digest(env)),
            },
        )

    # These runner hooks are deliberately replaced for the diagnostic process.
    for name, replacement in (
        ("_setup_and_run_step_loop", setup_wrapper),
        ("_step_snapshot_and_record", snap_wrapper),
        ("_execute_step_loop", exec_wrapper),
    ):
        setattr(E, name, replacement)


def run_robot(arm: str, scenario: str, seeds: list[int], out_dir: Path) -> dict:
    """Run one robot arm through the benchmark runner."""
    from robot_sf.benchmark.runner import run_batch

    install_wrappers(out_dir, arm)
    algo, cfg, profile = ARMS[arm]
    scenarios = L.load_scenarios([scenario], seeds)
    return run_batch(
        scenarios,
        out_path=out_dir / "episodes" / f"{arm}__{scenario}.jsonl",
        schema_path=Path.cwd() / "robot_sf/benchmark/schemas/episode.schema.v1.json",
        scenario_path=L.repo_scenario_path(),
        provenance_scenario_path=L.SCENARIO_FILE,
        horizon=L.WINDOW_STEPS,
        dt=L.DT,
        record_forces=True,
        algo=algo,
        algo_config_path=str(cfg),
        benchmark_profile=profile,
        socnav_missing_prereq_policy="fail-fast",
        workers=1,
        resume=False,
    )


def run_none(scenario: str, seeds: list[int], out_dir: Path) -> dict:
    """Run paired no-robot controls and save trajectories."""
    sc = L.load_scenarios([scenario], seeds)[0]
    for seed in seeds:
        CTX.update(arm="none", seed=seed, scenario=scenario)
        r = L.run_no_robot(sc, seed)
        _save(
            out_dir,
            {
                "ped_xy": r["ped_xy"],
                "robot_xy": r["robot_xy"],
                "robot_steps": np.int32(0),
                "termination_reason": np.str_("no_robot"),
                "parking": np.str_("removed_at_reset"),
                "sim_config_sha256": np.str_(r["sim_config_sha256"]),
            },
        )
    return {"written": len(seeds)}


def main() -> None:
    """Run one arm and route-condition job."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=[*ARMS, "none"])
    p.add_argument("--scenario", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--failure-parking",
        choices=["stop", "remove"],
        default="stop",
        help="post-hoc sensitivity only; the contract's parking rule is 'stop'",
    )
    a = p.parse_args()
    CTX["failure_parking"] = a.failure_parking
    seeds = [int(s) for s in a.seeds.split(",")]
    for s in seeds:
        if s < 1001 or 50000 <= s < 60000:
            raise SystemExit("seed outside the development allowlist")
    for sub in ("traj", "episodes", "logs"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = (
        run_none(a.scenario, seeds, a.out)
        if a.arm == "none"
        else run_robot(a.arm, a.scenario, seeds, a.out)
    )
    wall = time.time() - t0
    rec = {
        "arm": a.arm,
        "scenario": a.scenario,
        "seeds": seeds,
        "wall_s": round(wall, 2),
        "s_per_episode": round(wall / len(seeds), 3),
        "written": summary.get("written"),
        "failed_jobs": summary.get("failed_jobs"),
        "failures": summary.get("failures"),
    }
    (a.out / "logs" / f"{a.arm}__{a.scenario}.summary.json").write_text(
        json.dumps(rec, indent=2, default=str) + "\n"
    )
    print(json.dumps(rec, default=str))


if __name__ == "__main__":
    main()
