"""CPU-only acquisition wrapper around the unchanged pinned episode executor.

Run with producer and producer/fast-pysf first on PYTHONPATH and cwd=producer.
Inputs, plan receipt and raw outputs must be outside worktrees.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PRODUCER = "373dbfde4f39667cf9e8732dabe7df5118bdeab1"


def allowed_seed(seed):
    """Fail closed to the campaign's development schedule."""
    return type(seed) is int and 1091 <= seed <= 1120


def sha(path):
    """Hash preserved evidence bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    """Return an explicit UTC receipt time."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def safe(value):
    """Use strict-JSON null for undefined aggregate metrics in the diagnostic wrapper."""
    if hasattr(value, "tolist"):
        return safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def run(task):
    """Observe respawn calls; forward every original call unchanged exactly once."""
    cell, seed, policy, root_string, output_string = task
    assert allowed_seed(seed)
    root, output = Path(root_string), Path(output_string)
    from loguru import logger

    from robot_sf.benchmark.map_runner import map_runner_episode as episode
    from robot_sf.benchmark.map_runner.map_runner import _build_policy
    from robot_sf.ped_npc.ped_behavior import FollowRouteBehavior

    logger.remove()
    pending_rows = set()
    original_respawn = FollowRouteBehavior.respawn_group_at_start
    original_trace = episode._step_build_simulation_trace
    geometry = {}

    def respawn(self, gid, *, guard_robot=True):
        rows = [int(pid) + int(self.global_ped_offset) for pid in self.groups.groups[gid]]
        result = original_respawn(self, gid, guard_robot=guard_robot)
        if guard_robot:
            pending_rows.update(rows)
        return result

    def observe(state, slc, **kwargs):
        result = original_trace(state, slc, **kwargs)
        ids = state.trace_actor_ids
        state.simulation_step_trace[-1]["respawn_actor_ids"] = [
            ids[row] if ids is not None else row for row in sorted(pending_rows)
        ]
        pending_rows.clear()
        return result

    def reset_hook(env, obs):
        geometry.update(
            obstacles=[safe(o.vertices) for o in env.simulator.map_def.obstacles],
            width=env.simulator.map_def.width,
            height=env.simulator.map_def.height,
        )
        pending_rows.clear()
        return {"observer": "geometry and respawn boundaries only"}

    FollowRouteBehavior.respawn_group_at_start = respawn
    episode._step_build_simulation_trace = observe
    scenario = copy.deepcopy(cell["scenario"])
    scenario["simulation_config"]["route_spawn_seed"] = seed
    dest = output / "raw" / f"{cell['name']}__{seed}__{policy['key']}.json.gz"
    assert not dest.exists(), "Never overwrite existing episode evidence"
    try:
        record = episode.run_map_episode(
            scenario,
            seed,
            horizon=cell["horizon"],
            dt=0.1,
            record_forces=True,
            snqi_weights=None,
            snqi_baseline=None,
            algo=policy["algo"],
            scenario_path=root
            / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml",
            algo_config=policy["config"],
            algo_config_path=str(root / policy["path"]),
            policy_builder=_build_policy,
            pair_reset_hook=reset_hook,
            observation_mode="socnav_state",
            observation_level="tracked_agents_no_noise",
            record_simulation_step_trace=True,
        )
        meta = record["algorithm_metadata"]
        assert record["git_hash"] == PRODUCER
        assert meta.get("status") == "ok" and not meta.get("fallback_triggered")
        assert meta["planner_kinematics"]["execution_mode"] in ("native", "adapter")
        assert not record["spawn_validity"]["invalid_run"]
        payload = {
            "cell": cell["name"],
            "seed": seed,
            "policy": policy["key"],
            "producer": PRODUCER,
            "geometry": geometry,
            "record": record,
        }
        partial = Path(str(dest) + ".partial")
        with gzip.open(partial, "wt") as handle:
            json.dump(safe(payload), handle, allow_nan=False)
        partial.replace(dest)
        return {
            "cell": cell["name"],
            "seed": seed,
            "policy": policy["key"],
            "sha256": sha(dest),
            "bytes": dest.stat().st_size,
            "steps": record["steps"],
            "path": str(dest.relative_to(output)),
            "finished_utc": now(),
        }
    finally:
        FollowRouteBehavior.respawn_group_at_start = original_respawn
        episode._step_build_simulation_trace = original_trace


def remaining_identities(expected, completed):
    """Refuse foreign identities before resuming only missing episodes."""
    if not completed <= expected:
        raise ValueError("unexpected completed identities")
    return expected - completed


def _resume_records(output, expected):
    journal = [json.loads(line) for line in (output / "completed.jsonl").read_text().splitlines()]
    known = {r["path"]: r for r in journal}
    completed = set()
    for path in sorted((output / "raw").glob("*.json.gz")):
        wrapper = json.load(gzip.open(path, "rt"))
        key = wrapper["cell"], wrapper["seed"], wrapper["policy"]
        if key in completed or key not in expected:
            raise ValueError("unexpected or duplicated persisted episode")
        record = wrapper["record"]
        assert record["git_hash"] == PRODUCER and record["seed"] == key[1]
        assert record["algorithm_metadata"]["status"] == "ok"
        assert record["algorithm_metadata"]["simulation_step_trace"]["steps"]
        assert not record["spawn_validity"]["invalid_run"]
        relative = str(path.relative_to(output))
        hashed = sha(path)
        if relative in known:
            assert known[relative]["sha256"] == hashed
        else:
            adopted = {
                "cell": key[0],
                "seed": key[1],
                "policy": key[2],
                "sha256": hashed,
                "bytes": path.stat().st_size,
                "steps": record["steps"],
                "path": relative,
                "finished_utc": now(),
                "adopted_after_controller_drain": True,
            }
            with (output / "completed.jsonl").open("a") as handle:
                handle.write(json.dumps(adopted) + "\n")
        completed.add(key)
    assert len(completed) >= len(journal)
    return completed


def main():
    """Launch once, preserve receipts, and stop dispatch on infrastructure errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 7))
    parser.add_argument("--resume-infrastructure", action="store_true")
    args = parser.parse_args()
    root, output = args.producer.resolve(), args.campaign.resolve()
    assert Path.cwd().resolve() == root
    assert (
        subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        == PRODUCER
    )
    assert not subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    assert shutil.disk_usage(output).free > 50 * 1024**3
    assert (output / "launch.json").exists() == args.resume_infrastructure
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        assert os.environ[key] == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    import pysocialforce

    import robot_sf

    assert str(root) in robot_sf.__file__ and str(root / "fast-pysf") in pysocialforce.__file__
    cells = json.loads((output / "inputs/cells.json").read_text())
    policies = json.loads((output / "inputs/policies.json").read_text())
    for c in cells:
        assert sha(root / c["map_file"]) == c["map_sha256"]
    for p in policies:
        assert sha(root / p["path"]) == p["sha256"]
    manifest = {
        "started_utc": now(),
        "producer_commit": PRODUCER,
        "plan_receipt": (output / "plan_receipt.txt").read_text().strip(),
        "tool_sha256": sha(__file__),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "workers": args.workers,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "robot_sf_path": robot_sf.__file__,
        "pysocialforce_path": pysocialforce.__file__,
        "command": sys.argv,
        "input_sha256": {p.name: sha(p) for p in (output / "inputs").iterdir() if p.is_file()},
        "environment": {
            k: os.environ.get(k)
            for k in (
                "PYTHONPATH",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            )
        },
    }
    if args.resume_infrastructure:
        original = json.loads((output / "launch.json").read_text())
        for name, hashed in original["input_sha256"].items():
            assert sha(output / "inputs" / name) == hashed
        assert original["packages"] == manifest["packages"]
        manifest["original_launch_sha256"] = sha(output / "launch.json")
        resume_path = output / "resume_six_workers.json"
        assert not resume_path.exists()
        resume_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        (output / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "raw").mkdir(exist_ok=True)
    tasks = [
        (c, s, p, str(root), str(output))
        for s in range(1091, 1121)
        for c in cells
        for p in policies
    ]
    assert len(tasks) == 2700
    expected = {(c["name"], seed, p["key"]) for c, seed, p, _, _ in tasks}
    completed = _resume_records(output, expected) if args.resume_infrastructure else set()
    missing = remaining_identities(expected, completed)
    tasks = [task for task in tasks if (task[0]["name"], task[1], task[2]["key"]) in missing]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # Bounded queue: do not pre-submit the full campaign.
        pending = {pool.submit(run, t) for t in tasks[: args.workers]}
        next_task, count = min(args.workers, len(tasks)), len(completed)
        while pending:
            future = next(as_completed(pending))
            pending.remove(future)
            receipt = future.result()
            with (output / "completed.jsonl").open("a") as handle:
                handle.write(json.dumps(receipt) + "\n")
            count += 1
            if count % 30 == 0:
                print(json.dumps({"completed": count, "utc": now()}), flush=True)
            if next_task < len(tasks):
                assert shutil.disk_usage(output).free > 50 * 1024**3
                pending.add(pool.submit(run, tasks[next_task]))
                next_task += 1
    (output / "acquisition_complete.json").write_text(
        json.dumps({"completed": count, "finished_utc": now()}) + "\n"
    )


if __name__ == "__main__":
    main()
