#!/usr/bin/env python3
"""Diagnostic-only HZEV acquisition. Never use its outcomes as benchmark scores.

Uses the canonical campaign loader, map episode, trace writer and mover policies.
Process-local instrumentation retains first terminal events but prevents breaks.
No reset follows a contact; movers park on goal completion. Stationary emits zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmarks/hzev_dynamic_window_dev5_h800.yaml"
DEV_SEEDS = (1001, 1002, 1003, 1004, 1005)
ARMS = ("stationary", "orca", "goal")


def digest(path):
    """Hash exact input bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    """Replace a complete JSON artifact atomically; .partial never means done."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, allow_nan=False) + "\n")
    partial.replace(path)


def load_packet(config_path):  # noqa: C901 - explicit diagnostic input gates
    """Resolve and validate all 48 scenarios without running an environment."""
    from robot_sf.benchmark.camera_ready._config import (
        _load_campaign_scenarios,
        load_campaign_config,
    )

    raw = yaml.safe_load(config_path.read_text())
    cfg = load_campaign_config(config_path, repository_root=ROOT)
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    template = ROOT / raw["hzev"]["source_template"]
    if digest(template) != raw["hzev"]["source_template_sha256"]:
        raise ValueError("source template digest moved")
    original = _load_campaign_scenarios(load_campaign_config(template), repository_root=ROOT)
    by_name = {s["name"]: s for s in original}
    if len(scenarios) != 48 or {s["name"] for s in scenarios} != set(by_name):
        raise ValueError("diagnostic must contain the exact 48-scenario template roster")
    for s in scenarios:
        if s["simulation_config"]["max_episode_steps"] != 800 or s["seeds"] != list(DEV_SEEDS):
            raise ValueError("diagnostic budget/seed binding differs from H800 dev5")
        source = by_name[s["name"]]
        if (
            s["metadata"]["hzev_authored_budget_steps"]
            != source["simulation_config"]["max_episode_steps"]
        ):
            raise ValueError("authored budget provenance mismatch")

        # Frozen expansion may normalize paths, seeds and add horizon metadata only.
        def science_view(row):
            row = json.loads(json.dumps(row))
            row.pop("seeds", None)
            row["simulation_config"].pop("max_episode_steps", None)
            row["metadata"].pop("hzev_authored_budget_steps", None)
            row["metadata"].pop("scenario_horizon", None)
            return row

        if science_view(s) != science_view(source):
            raise ValueError(f"frozen scenario changed beyond horizon/seeds: {s['name']}")
    if cfg.horizon is not None or not cfg.record_simulation_step_trace:
        raise ValueError("use explicit scenario_horizons with traces enabled")
    if tuple(p.key for p in cfg.planners) != ("stationary", "goal", "orca"):
        raise ValueError("unexpected arm roster")
    if raw["hzev"]["stationary_command"] != [0.0, 0.0]:
        raise ValueError("stationary command must be exactly zero")
    return cfg, raw, scenarios


def acquire(job):  # noqa: C901, PLR0915 - one process-local instrumentation boundary
    """Run one dev episode with process-local continuation and respawn recording."""
    from robot_sf.benchmark.map_runner import map_runner as runner
    from robot_sf.benchmark.map_runner import map_runner_episode as episode
    from robot_sf.ped_npc.ped_behavior import CrowdedZoneBehavior, FollowRouteBehavior

    scenario, seed, arm, steps, cfg_path, identity, output = job
    if seed not in DEV_SEEDS or arm not in ARMS or not 1 <= steps <= 800:
        raise ValueError("unauthorized seed, arm or horizon")
    logger.remove()
    logger.add(os.sys.stderr, level="ERROR")
    cfg, raw, _ = load_packet(Path(cfg_path))
    receipt = {
        "identity": identity,
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "scenario_id": scenario["name"],
        "respawns": [],
        "first_terminal": None,
        "park_after_goal": True,
        "continue_after_terminal": True,
    }
    parked = False
    original_init = episode._init_step_loop_state
    original_terminal = episode._step_collision_and_termination

    def initial_capture(**kwargs):
        state = original_init(**kwargs)
        sim = kwargs["env"].simulator
        nav = sim.robot_navs[0]
        receipt["start"] = state.initial_robot_pos.tolist()
        receipt["goal"] = list(nav.waypoints[-1])
        receipt["route_waypoints"] = [list(p) for p in nav.waypoints]
        receipt["initial_frame"] = {
            "time_s": 0.0,
            "robot": {"position": receipt["start"]},
            "pedestrians": [
                {"id": i, "position": p.tolist()} for i, p in enumerate(state.initial_ped_positions)
            ],
        }
        receipt["robot_radius_m"] = float(sim.robots[0].config.radius)
        receipt["behavior_inventory"] = []
        receipt["recurring_flow"] = False
        for behavior in sim.peds_behaviors:
            count = len(behavior.navigators) if isinstance(behavior, FollowRouteBehavior) else 0
            recurring = count > 0 or (
                isinstance(behavior, CrowdedZoneBehavior) and bool(behavior.groups.group_ids)
            )
            receipt["recurring_flow"] |= recurring
            receipt["behavior_inventory"].append(
                {
                    "type": type(behavior).__name__,
                    "route_group_count": count,
                    "recurring": recurring,
                }
            )
            if isinstance(behavior, FollowRouteBehavior):
                original_respawn = behavior.respawn_group_at_start

                def capture_respawn(gid, *, guard_robot=True, b=behavior, fn=original_respawn):
                    fn(gid, guard_robot=guard_robot)
                    receipt["respawns"].append(
                        {
                            "group_id": int(gid),
                            "behavior_step": b.step_count,
                            "time_s": b.step_count * float(cfg.dt),
                        }
                    )

                behavior.respawn_group_at_start = capture_respawn
        return state

    def continue_terminal(state, slc, *, step_idx, sim):
        nonlocal parked
        should_stop = original_terminal(state, slc, step_idx=step_idx, sim=sim)
        if should_stop and receipt["first_terminal"] is None:
            receipt["first_terminal"] = {
                "step": step_idx,
                "time_s": (step_idx + 1) * float(cfg.dt),
                "reason": state.termination_reason,
            }
        if state.reached_goal_step is not None:
            parked = True
        return False

    def policy_builder(algo, algo_config, **kwargs):
        if arm == "stationary":

            def policy(_obs):
                return (0.0, 0.0)

            return policy, {"algorithm": "hzev_stationary", "diagnostic_only": True}
        policy, meta = runner._build_policy(algo, algo_config, **kwargs)

        def moving_policy(obs):
            return (0.0, 0.0) if parked else policy(obs)

        moving_policy.__dict__.update(policy.__dict__)
        return moving_policy, meta

    planner = next(p for p in cfg.planners if p.key == arm)
    with ExitStack() as stack:
        stack.enter_context(patch.object(episode, "_init_step_loop_state", initial_capture))
        stack.enter_context(
            patch.object(episode, "_step_collision_and_termination", continue_terminal)
        )
        record = episode.run_map_episode(
            scenario=scenario,
            seed=seed,
            horizon=steps,
            dt=cfg.dt,
            record_forces=cfg.record_forces,
            snqi_weights=None,
            snqi_baseline=None,
            algo=planner.algo,
            scenario_path=cfg.scenario_matrix_path,
            algo_config={},
            record_simulation_step_trace=True,
            policy_builder=policy_builder,
        )
    trace = record["algorithm_metadata"]["simulation_step_trace"]["steps"]
    if len(trace) != steps:
        raise ValueError(f"early stop: {len(trace)} != {steps}")
    receipt.update(
        evidence_status="diagnostic-only",
        dt_s=cfg.dt,
        authored_budget_steps=scenario["metadata"]["hzev_authored_budget_steps"],
        observed_duration_s=trace[-1]["time_s"],
        definition=raw["hzev"],
    )
    if arm == "stationary":
        import math

        if any(math.dist(f["robot"]["position"], receipt["start"]) > 1e-8 for f in trace):
            raise ValueError("stationary robot moved")
    record["hzev"] = receipt
    write_json(output, record)
    return {
        "path": Path(output).name,
        "sha256": digest(output),
        "scenario_id": scenario["name"],
        "seed": seed,
        "arm": arm,
        "trace_steps": len(trace),
        "first_terminal": receipt["first_terminal"],
        "respawn_count": len(receipt["respawns"]),
    }


def main():  # noqa: C901, PLR0912, PLR0915 - bounded diagnostic CLI and custody gates
    """Validate, acquire the bounded roster, and emit an immutable identity manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--scenario")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    logger.remove()
    logger.add(os.sys.stderr, level="ERROR")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.head_sha:
        parser.error("checkout HEAD differs from pinned SHA")
    if not set(args.seeds) <= set(DEV_SEEDS) or len(set(args.seeds)) != len(args.seeds):
        parser.error("only distinct dev seeds 1001–1005 are allowed")
    if len(set(args.arms)) != len(args.arms):
        parser.error("duplicate arms")
    if args.smoke_steps is not None and not (
        1 <= args.smoke_steps <= 100
        and args.scenario
        and len(args.seeds) == 1
        and len(args.arms) == 1
        and args.workers == 1
    ):
        parser.error("local smoke requires <=100 steps, one scenario/seed/arm, workers=1")
    cfg, raw, scenarios = load_packet(args.config.resolve())
    if args.scenario:
        scenarios = [s for s in scenarios if s["name"] == args.scenario]
        if not scenarios:
            parser.error("unknown scenario")
    steps = args.smoke_steps or 800
    files = [
        args.config.resolve(),
        cfg.scenario_matrix_path,
        cfg.scenario_horizons_path,
        ROOT / raw["hzev"]["source_template"],
        ROOT / "uv.lock",
    ]
    identity = {
        "head_sha": head,
        "input_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
        "scenario_ids": [s["name"] for s in scenarios],
        "seeds": args.seeds,
        "arms": args.arms,
        "steps": steps,
        "dt_s": cfg.dt,
        "smoke": args.smoke_steps is not None,
    }
    # Validate builders and the native dependency without silently accepting an ORCA fallback.
    from robot_sf.benchmark.map_runner.map_runner import _build_policy

    for arm in args.arms:
        if arm == "stationary":
            continue
        if arm == "orca":
            import rvo2  # noqa: F401 - fail fast if native ORCA unavailable
        policy, _ = _build_policy(arm, {}, robot_kinematics="differential_drive")
        closer = getattr(policy, "_planner_close", None)
        if closer:
            closer()
    count = len(scenarios) * len(args.seeds) * len(args.arms)
    if args.check_only:
        print(json.dumps({"status": "packet-valid", "episodes": count, "identity": identity}))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["identity"] != identity:
        raise ValueError(
            "output directory belongs to another source/config/roster; use a fresh directory"
        )
    jobs, rows = [], []
    for s in scenarios:
        for seed in args.seeds:
            for arm in args.arms:
                output = args.output_dir / f"{s['name']}__{seed}__{arm}.json"
                if output.exists():
                    row = json.loads(output.read_text())
                    if (
                        row["hzev"]["identity"] != identity
                        or len(row["algorithm_metadata"]["simulation_step_trace"]["steps"]) != steps
                    ):
                        raise ValueError(f"invalid resume row: {output}")
                    rows.append(
                        {
                            "path": output.name,
                            "sha256": digest(output),
                            "scenario_id": s["name"],
                            "seed": seed,
                            "arm": arm,
                            "trace_steps": steps,
                        }
                    )
                else:
                    jobs.append(
                        (s, seed, arm, steps, str(args.config.resolve()), identity, str(output))
                    )
    manifest = {
        "schema_version": "hzev-acquisition.v1",
        "evidence_status": "diagnostic-only",
        "identity": identity,
        "expected_episodes": count,
        "complete": False,
        "rows": rows,
    }
    write_json(manifest_path, manifest)
    if args.workers == 1:
        results = map(acquire, jobs)
        for result in results:
            rows.append(result)
            write_json(manifest_path, manifest)
            print(json.dumps({"completed": len(rows), "expected": count}), flush=True)
    else:
        # Parent + at most 3 workers = at most 4 acquisition processes.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(acquire, jobs, chunksize=1):
                rows.append(result)
                write_json(manifest_path, manifest)
                print(json.dumps({"completed": len(rows), "expected": count}), flush=True)
    manifest["complete"] = len(rows) == count
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
