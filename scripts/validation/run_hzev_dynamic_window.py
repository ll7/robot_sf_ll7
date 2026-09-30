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
import math
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from copy import copy, deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmarks/hzev_dynamic_window_dev5_h800.yaml"
DEV_SEEDS = (1001, 1002, 1003, 1004, 1005)
ARMS = ("stationary", "orca", "goal")


def worker_limit():
    """Respect the Slurm task allocation when present, otherwise host CPU count."""
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    limit = int(allocation) if allocation is not None else (os.cpu_count() or 1)
    if limit < 1:
        raise ValueError("CPU allocation must be positive")
    return limit


class DiagnosticWindow:
    """Keep continued simulation state out of ordinary outcome/metric calculation."""

    def __init__(self, episode, receipt, dt):
        """Bind process-local ordinary helpers before acquisition patches them."""
        self.episode = episode
        self.receipt = receipt
        self.dt = float(dt)
        self.terminal = episode._step_collision_and_termination
        self.build_result = episode._build_step_loop_result
        self.ordinary_result = None
        self.parked = False
        self.env = None
        self.policy = None

    def snapshot_terminal(self, state):
        """Capture the runtime context normally refreshed after loop termination."""
        import numpy as np

        terminal_state = copy(state)
        simulator = self.env.simulator
        terminal_state.map_def = simulator.map_def
        terminal_state.goal_vec = np.asarray(simulator.goal_pos[0], dtype=float)
        terminal_state.respawn_overlap_events = self.episode._read_respawn_overlap_events(simulator)
        terminal_state.simulator_obstacle_force_law_metadata = (
            self.episode._read_obstacle_force_law_metadata(self.env)
        )
        terminal_state.planner_obstacle_force_law_metadata = (
            self.episode._read_policy_obstacle_force_law_metadata(self.policy)
        )
        stats = getattr(self.policy, "_planner_stats", None)
        if callable(stats):
            try:
                payload = stats()
            except (RuntimeError, ValueError, TypeError):
                payload = None
            if isinstance(payload, dict):
                terminal_state.planner_runtime_snapshot = dict(payload)
        return deepcopy(self.build_result(terminal_state))

    def continue_terminal(self, state, slc, *, step_idx, sim):
        """Snapshot the complete terminal prefix before any continuation mutates it."""
        collision = self.episode.collision_event(sim.info)
        goal = self.episode.route_complete_success(sim.info)
        event = {"step": step_idx, "time_s": (step_idx + 1) * self.dt}
        if collision and self.receipt.get("first_collision") is None:
            self.receipt["first_collision"] = dict(event)
        if goal and not collision and self.receipt.get("first_goal") is None:
            self.receipt["first_goal"] = dict(event)
            self.parked = True
        should_stop = self.terminal(state, slc, step_idx=step_idx, sim=sim)
        if should_stop and self.ordinary_result is None:
            self.receipt["first_terminal"] = {**event, "reason": state.termination_reason}
            # Copy every trajectory, event ledger and flag, not just outcome bits:
            # production metrics must see exactly the ordinary terminal prefix.
            self.ordinary_result = self.snapshot_terminal(state)
        return False

    def finish(self, state):
        """Export the full diagnostic trace and return only the ordinary result."""
        self.receipt["diagnostic_trace"] = {
            "schema_version": "hzev-diagnostic-trace.v1",
            "evidence_status": "diagnostic-only",
            "steps": state.simulation_step_trace,
        }
        return (
            self.ordinary_result if self.ordinary_result is not None else self.build_result(state)
        )


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


def episode_scenario(scenario):
    """Bind campaign-loader repo-relative maps for the direct episode builder."""
    scenario = json.loads(json.dumps(scenario))
    if scenario.get("map_file"):
        scenario["map_file"] = str((ROOT / scenario["map_file"]).resolve())
    return scenario


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
    started = time.perf_counter()
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
        "first_collision": None,
        "first_goal": None,
        "park_after_goal": True,
        "continue_after_terminal": True,
    }
    original_init = episode._init_step_loop_state
    window = DiagnosticWindow(episode, receipt, cfg.dt)

    def initial_capture(**kwargs):
        state = original_init(**kwargs)
        window.env = kwargs["env"]
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
                isinstance(behavior, CrowdedZoneBehavior) and bool(behavior.zone_assignments)
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

    def policy_builder(algo, algo_config, **kwargs):
        if arm == "stationary":

            def policy(_obs):
                return (0.0, 0.0)

            window.policy = policy
            return policy, {"algorithm": "hzev_stationary", "diagnostic_only": True}
        policy, meta = runner._build_policy(algo, algo_config, **kwargs)

        def moving_policy(obs):
            return (0.0, 0.0) if window.parked else policy(obs)

        moving_policy.__dict__.update(policy.__dict__)
        window.policy = moving_policy
        return moving_policy, meta

    scenario = episode_scenario(scenario)
    planner = next(p for p in cfg.planners if p.key == arm)
    with ExitStack() as stack:
        stack.enter_context(patch.object(episode, "_init_step_loop_state", initial_capture))
        stack.enter_context(
            patch.object(episode, "_step_collision_and_termination", window.continue_terminal)
        )
        stack.enter_context(patch.object(episode, "_build_step_loop_result", window.finish))
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
    trace = receipt["diagnostic_trace"]["steps"]
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
        if any(math.dist(f["robot"]["position"], receipt["start"]) > 1e-8 for f in trace):
            raise ValueError("stationary robot moved")
    record["hzev"] = receipt
    from scripts.validation.analyze_hzev_dynamic_window import (
        validate_ordinary_episode,
        validated_frames,
    )

    # Geometry/time must be finite. Some auxiliary benchmark fields use infinity
    # as a sentinel (e.g. no predicted TTC); retain their paths and original values
    # explicitly while writing standards-compliant diagnostic JSON.
    validated_frames(record)
    validate_ordinary_episode(record)
    nonfinite_fields = []

    def json_safe(value, path="$"):
        if isinstance(value, dict):
            return {k: json_safe(v, f"{path}.{k}") for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [json_safe(v, f"{path}[{i}]") for i, v in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite_fields.append({"path": path, "original_value": repr(value)})
            return None
        return value

    record = json_safe(record)
    record["hzev"]["nonfinite_auxiliary_fields"] = nonfinite_fields
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
        "wall_time_s": time.perf_counter() - started,
    }


def main():  # noqa: C901, PLR0912, PLR0915 - bounded diagnostic CLI and custody gates
    """Validate, acquire the bounded roster, and emit an immutable identity manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--scenario", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()
    try:
        limit = worker_limit()
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.workers <= limit:
        parser.error(f"--workers must be in 1..{limit}")
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
        and len(args.scenario) == 1
    ):
        parser.error("short smoke requires <=100 steps, one scenario and one dev seed")
    cfg, raw, scenarios = load_packet(args.config.resolve())
    if args.scenario:
        if not set(args.scenario) <= {s["name"] for s in scenarios}:
            parser.error("unknown scenario")
        scenarios = [s for s in scenarios if s["name"] in args.scenario]
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
        from robot_sf.benchmark.map_runner.map_runner_env import build_env_config

        # Parse all selected maps and construct configs, without environment steps.
        for scenario in scenarios:
            build_env_config(episode_scenario(scenario), scenario_path=cfg.scenario_matrix_path)
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
                    from scripts.validation.analyze_hzev_dynamic_window import (
                        validate_ordinary_episode,
                        validated_frames,
                    )

                    validated_frames(row)
                    validate_ordinary_episode(row)
                    if (
                        row["hzev"]["identity"] != identity
                        or len(row["hzev"]["diagnostic_trace"]["steps"]) != steps
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
        # Independent, seeded jobs; patches never cross process boundaries.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(acquire, jobs, chunksize=1):
                rows.append(result)
                write_json(manifest_path, manifest)
                print(json.dumps({"completed": len(rows), "expected": count}), flush=True)
    manifest["complete"] = len(rows) == count
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
