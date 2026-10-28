"""Acquisition driver: mean-matched heterogeneous pedestrian speeds (dev-seed diagnostic).

Runs the repository's normal campaign entry point (``run_map_batch``) on the
0.0.8 release source with the frozen release planner configs. Each authored
cell is run twice per seed: ``uniform`` (all 12 pedestrians at 0.65 m/s) and
``mixed`` (3 x 0.325, 6 x 0.65, 3 x 0.975 m/s). Both conditions use the same
archetype code path (``uniform`` is the one-archetype composition with factor
1.0), so spawn positions, routes, groups and radii are drawn identically.

A read-only, in-process observer wraps ``_prepare_episode_env`` to write the
realized population after reset (receipt JSON) and per-step pedestrian/robot
state (NPZ). It does not change any simulator or planner state.

Usage (from the repository root):
    python docs/context/heterogeneous_speed_mean_matched/driver.py \
        --planner orca --seeds 1001 1002 --out <dir> [--cells all]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import robot_sf.benchmark.map_runner.map_runner_episode as _episode
from robot_sf.benchmark.map_runner.map_runner import run_map_batch
from robot_sf.ped_npc.ped_archetypes import assign_archetype_labels
from robot_sf.sim.simulator import _spawn_reaction_buffer
from robot_sf.training.scenario_loader import load_scenarios

SCENARIO_MATRIX = "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
SCHEMA = "robot_sf/benchmark/schemas/episode.schema.v1.json"
CELLS = {
    "crossing": "classic_cross_trap_medium",
    "group_crossing": "classic_group_crossing_medium",
    "head_on_corridor": "classic_head_on_corridor_medium",
    "overtaking": "classic_overtaking_medium",
}
POPULATION_SIZE = 12
CONDITIONS = {
    "uniform": {
        "archetype_composition": {"standard": 1.0},
        "archetype_speed_factors": {"standard": 1.0},
    },
    "mixed": {
        "archetype_composition": {"slow": 0.25, "standard": 0.5, "fast": 0.25},
        "archetype_speed_factors": {"slow": 0.5, "standard": 1.0, "fast": 1.5},
    },
}
PLANNERS = {
    "hybrid_v4_continuous": {
        "algo": "hybrid_rule_local_planner",
        "algo_config_path": "configs/policy_search/candidates/"
        "hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release_0_0_8_frozen.yaml",
        "benchmark_profile": "experimental",
        "adapter_impact_eval": False,
    },
    "orca": {
        "algo": "orca",
        "algo_config_path": "configs/algos/orca_release_v0_0_8.yaml",
        "benchmark_profile": "baseline-safe",
        "adapter_impact_eval": False,
    },
    "social_force": {
        "algo": "social_force",
        "algo_config_path": "configs/algos/social_force_release_v0_0_8.yaml",
        "benchmark_profile": "baseline-safe",
        "adapter_impact_eval": False,
    },
    "ppo_release": {
        "algo": "ppo",
        "algo_config_path": "configs/baselines/ppo_release_robot_0_0_8_cpu.yaml",
        "benchmark_profile": "experimental",
        "adapter_impact_eval": True,
    },
}


def build_scenarios(cell: str, seeds: list[int]) -> list[dict]:
    """Return the two condition variants of one authored release cell."""
    base = [s for s in load_scenarios(SCENARIO_MATRIX) if s["name"] == CELLS[cell]]
    if len(base) != 1:
        raise RuntimeError(f"expected one scenario {CELLS[cell]}, found {len(base)}")
    out = []
    for cond, pop in CONDITIONS.items():
        sc = copy.deepcopy(dict(base[0]))
        sc["name"] = f"{CELLS[cell]}__xhet_{cond}"
        sc["seeds"] = [int(s) for s in seeds]
        sim = sc.setdefault("simulation_config", {})
        if sim.get("desired_speed_mean") is not None or sim.get("ped_speed_tier") is not None:
            raise RuntimeError("explicit desired-speed override would erase the mixture")
        sim["population_size"] = POPULATION_SIZE
        sim["archetype_composition"] = dict(pop["archetype_composition"])
        sim["archetype_speed_factors"] = dict(pop["archetype_speed_factors"])
        # archetype_seed is set per episode by the observer-free path below: the
        # runner reads it from the scenario, so one scenario object per seed.
        meta = sc.setdefault("metadata", {})
        meta["xhet_cell"] = cell
        meta["xhet_condition"] = cond
        out.append(sc)
    return out


def expand_per_seed(scenarios: list[dict]) -> list[dict]:
    """Bind ``archetype_seed`` to the episode seed (stratum-to-slot assignment varies by seed)."""
    expanded = []
    for sc in scenarios:
        for seed in sc["seeds"]:
            one = copy.deepcopy(sc)
            one["seeds"] = [seed]
            # run_map_batch rejects duplicate ids; the seed suffix is bookkeeping only
            # (episode RNG comes from the seed, verified identical across conditions).
            one["name"] = f"{sc['name']}__a{int(seed)}"
            one["simulation_config"]["archetype_seed"] = int(seed)
            expanded.append(one)
    return expanded


def _relocation(sim) -> dict | None:
    """Summarize the release reset spawn-clearance relocation (speed-coupled buffer)."""
    rep = getattr(sim, "last_spawn_relocation", None)
    if rep is None:
        return None
    out = {}
    for k in dir(rep):
        if k.startswith("_"):
            continue
        v = getattr(rep, k)
        if callable(v):
            continue
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = repr(v)
    return out


class Observer:
    """Read-only after-reset and per-step capture of realized pedestrian state."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.pending: dict | None = None

    def flush(self) -> None:
        p = self.pending
        if p is None:
            return
        self.pending = None
        stem = self.out_dir / f"{p['scenario']}__s{p['seed']}"
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in p["steps"].items()}
        np.savez_compressed(f"{stem}.npz", **arrays)
        p["receipt"]["captured_steps"] = int(arrays["ped_pos"].shape[0])
        Path(f"{stem}.receipt.json").write_text(json.dumps(p["receipt"], sort_keys=True))

    def _snapshot(self, env, action=None) -> None:
        sim = env.simulator
        peds = sim.pysf_sim.peds
        st = np.asarray(peds.state, dtype=float)
        s = self.pending["steps"]
        s["ped_pos"].append(st[:, 0:2].copy())
        s["ped_vel"].append(st[:, 2:4].copy())
        s["ped_goal"].append(st[:, 4:6].copy())
        s["ped_max_speed"].append(np.asarray(peds.max_speeds, dtype=float).copy())
        rp = np.asarray(sim.robot_poses[0][0] if hasattr(sim, "robot_poses") else sim.robot_pos[0])
        s["robot_pos"].append(np.asarray(rp, dtype=float).reshape(-1)[:2].copy())
        a = np.full(2, np.nan) if action is None else np.asarray(action, dtype=float).reshape(-1)[:2]
        s["action"].append(a.copy())

    def wrap(self, original):
        observer = self

        def _prepare(env, **kwargs):
            observer.flush()
            obs = original(env, **kwargs)
            scenario = kwargs.get("scenario") or {}
            seed = int(kwargs["seed"])
            sim = env.simulator
            peds = sim.pysf_sim.peds
            sim_cfg = scenario.get("simulation_config", {})
            comp = sim_cfg.get("archetype_composition")
            n = int(np.asarray(peds.state).shape[0])
            labels = assign_archetype_labels(n, comp, seed=sim_cfg.get("archetype_seed"))
            scene_cfg = sim.pysf_sim.config.scene_config
            groups = getattr(sim, "groups", None)
            group_map = {
                str(k): sorted(int(x) for x in v) for k, v in getattr(groups, "groups", {}).items()
            }
            st = np.asarray(peds.state, dtype=float)
            receipt = {
                "scenario": scenario.get("name"),
                "cell": scenario.get("metadata", {}).get("xhet_cell"),
                "condition": scenario.get("metadata", {}).get("xhet_condition"),
                "seed": seed,
                "n_peds": n,
                "archetype_seed": sim_cfg.get("archetype_seed"),
                "stratum_labels": [str(x) for x in labels],
                "max_speeds": [float(x) for x in peds.max_speeds],
                "initial_speeds": [float(x) for x in peds.initial_speeds],
                "max_speed_multiplier": float(peds.max_speed_multiplier),
                "desired_speed_mean": scene_cfg.desired_speed_mean,
                "explicit_desired_speeds": None
                if peds._explicit_desired_speeds is None
                else [float(x) for x in peds._explicit_desired_speeds],
                "agent_radius": float(scene_cfg.agent_radius),
                "start_pos": st[:, 0:2].round(6).tolist(),
                "start_vel": st[:, 2:4].round(6).tolist(),
                "goal": st[:, 4:6].round(6).tolist(),
                "tau": st[:, 6].round(6).tolist(),
                "groups": group_map,
                "behaviors": [type(b).__name__ for b in sim.peds_behaviors],
                "robot_pose": np.asarray(sim.robot_poses[0][0]).round(6).tolist()
                if hasattr(sim, "robot_poses")
                else None,
                "robot_goal": np.asarray(sim.goal_pos[0]).round(6).tolist(),
                "spawn_relocation": _relocation(sim),
                "spawn_reaction_buffer_m": float(
                    _spawn_reaction_buffer(
                        max(
                            float(np.max(peds.max_speeds)),
                            float(np.linalg.norm(np.asarray(sim.ped_vel), axis=1).max()),
                        )
                    )
                ),
            }
            observer.pending = {
                "scenario": scenario.get("name"),
                "seed": seed,
                "receipt": receipt,
                "steps": {
                    k: []
                    for k in (
                        "ped_pos", "ped_vel", "ped_goal", "ped_max_speed", "robot_pos", "action",
                    )
                },
            }
            observer._snapshot(env)
            original_step = env.step

            def _step(action, *a, **kw):
                result = original_step(action, *a, **kw)
                try:
                    observer._snapshot(env, action)
                except Exception as exc:  # observer must never change the episode
                    observer.pending["receipt"]["observer_error"] = repr(exc)
                return result

            env.step = _step
            return obs

        return _prepare


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", required=True, choices=sorted(PLANNERS))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    banned = [s for s in args.seeds if 111 <= s <= 140 or 10000 <= s <= 99999]
    if banned:
        raise SystemExit("refusing reserved seed block")
    out = Path(args.out)
    obs_dir = out / "observer"
    obs_dir.mkdir(parents=True, exist_ok=True)
    observer = Observer(obs_dir)
    _episode._prepare_episode_env = observer.wrap(_episode._prepare_episode_env)
    spec = PLANNERS[args.planner]
    log = out / "batches.jsonl"
    for cell in args.cells:
        scenarios = expand_per_seed(build_scenarios(cell, args.seeds))
        t0 = time.time()
        summary = run_map_batch(
            scenarios,
            out / f"{args.planner}__{cell}.jsonl",
            SCHEMA,
            scenario_path=SCENARIO_MATRIX,
            horizon=None,
            dt=0.1,
            record_forces=True,
            algo=spec["algo"],
            algo_config_path=spec["algo_config_path"],
            benchmark_profile=spec["benchmark_profile"],
            adapter_impact_eval=spec["adapter_impact_eval"],
            socnav_missing_prereq_policy="fail-fast",
            workers=1,
            resume=True,
        )
        observer.flush()
        with log.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "planner": args.planner,
                        "cell": cell,
                        "seeds": args.seeds,
                        "elapsed_s": time.time() - t0,
                        "pid": os.getpid(),
                        "written": summary.get("written"),
                        "successful_jobs": summary.get("successful_jobs"),
                        "failed_jobs": summary.get("failed_jobs"),
                        "skipped_jobs": summary.get("skipped_jobs"),
                        "failures": summary.get("failures"),
                        "algo_config_sha256": sha256_file(spec["algo_config_path"]),
                    },
                    default=str,
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
