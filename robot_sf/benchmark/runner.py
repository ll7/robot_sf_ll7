"""Episode runner scaffold for the Social Navigation Benchmark.

Responsibilities:
 - Load scenario parameter matrix (YAML list) and expand repeats
 - Deterministically generate scenarios (using `generate_scenario`)
 - Step a simple baseline policy toward each agent's goal (very naive)
 - Collect per-step robot + pedestrian states and (optionally) forces
 - Build EpisodeData and compute metrics + SNQI (weights optional)
 - Validate record against JSON schema and write JSONL

This is an initial lightweight runner; performance optimizations and
multi-processing are future work.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import yaml

from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, snqi
from robot_sf.benchmark.scenario_generator import generate_scenario
from robot_sf.benchmark.schema_validator import load_schema, validate_episode
from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper


def _git_hash_fallback() -> str:
    # Best effort; avoid importing subprocess if not needed later
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _config_hash(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def load_scenario_matrix(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    # Allow either YAML stream of docs or a single list
    if len(docs) == 1 and isinstance(docs[0], list):
        scenarios = docs[0]
    else:
        scenarios = docs
    return [dict(s) for s in scenarios]


def _simple_robot_policy(robot_pos: np.ndarray, goal: np.ndarray, speed: float = 1.0) -> np.ndarray:
    """Return a velocity vector pointing toward goal with capped speed."""
    vec = goal - robot_pos
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return np.zeros(2)
    dir_unit = vec / dist
    v = dir_unit * min(speed, dist)
    return v


def _prepare_robot_points(
    robot_start: Sequence[float] | None, robot_goal: Sequence[float] | None
) -> tuple[np.ndarray, np.ndarray]:
    if robot_start is None:
        rs = np.array([0.3, 3.0], dtype=float)
    else:
        rs = np.asarray(robot_start, dtype=float)
    if robot_goal is None:
        rg = np.array([9.7, 3.0], dtype=float)
    else:
        rg = np.asarray(robot_goal, dtype=float)
    return rs, rg


def _build_episode_data(
    robot_pos_traj: list[np.ndarray],
    robot_vel_traj: list[np.ndarray],
    robot_acc_traj: list[np.ndarray],
    peds_pos_traj: list[np.ndarray],
    ped_forces_traj: list[np.ndarray],
    goal: np.ndarray,
    dt: float,
    reached_goal_step: int | None,
) -> EpisodeData:
    return EpisodeData(
        robot_pos=np.vstack(robot_pos_traj),
        robot_vel=np.vstack(robot_vel_traj),
        robot_acc=np.vstack(robot_acc_traj),
        peds_pos=(np.stack(peds_pos_traj) if peds_pos_traj else np.zeros((0, 0, 2))),
        ped_forces=(np.stack(ped_forces_traj) if ped_forces_traj else np.zeros((0, 0, 2))),
        goal=goal,
        dt=dt,
        reached_goal_step=reached_goal_step,
    )


def run_episode(
    scenario_params: Dict[str, Any],
    seed: int,
    *,
    horizon: int = 100,
    dt: float = 0.1,
    robot_start: Sequence[float] | None = None,
    robot_goal: Sequence[float] | None = None,
    record_forces: bool = True,
    snqi_weights: Dict[str, float] | None = None,
    snqi_baseline: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    """Run a single episode and return a metrics record dict.

    The robot is modeled separately from pedestrians (independent); robot motion does not influence pedestrians yet.
    Future work: integrate robot into social-force environment for two-way coupling.
    """
    gen = generate_scenario(scenario_params, seed=seed)
    sim = gen.simulator
    if sim is None:
        raise RuntimeError("pysocialforce not available; cannot run episode")
    wrapper = FastPysfWrapper(sim)

    # Determine robot start/goal: default start left boundary mid-height, goal right boundary mid-height
    robot_start_arr, robot_goal_arr = _prepare_robot_points(robot_start, robot_goal)

    robot_pos = robot_start_arr.copy()
    robot_vel = np.zeros(2)

    # Buffers
    robot_pos_traj: List[np.ndarray] = []
    robot_vel_traj: List[np.ndarray] = []
    robot_acc_traj: List[np.ndarray] = []
    peds_pos_traj: List[np.ndarray] = []
    ped_forces_traj: List[np.ndarray] = []

    reached_goal_step: int | None = None
    goal_radius = 0.3

    last_vel = robot_vel.copy()

    def _step_robot(
        curr_pos: np.ndarray, curr_vel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        new_vel = _simple_robot_policy(curr_pos, robot_goal_arr, speed=1.0)
        new_pos = curr_pos + new_vel * dt
        acc_vec = (new_vel - curr_vel) / dt
        return new_pos, new_vel, acc_vec

    for t in range(horizon):
        # Policy
        robot_pos, robot_vel, acc = _step_robot(robot_pos, last_vel)
        last_vel = robot_vel.copy()

        # Capture ped state (positions from sim state columns 0:2)
        ped_state = sim.state
        ped_pos = ped_state[:, 0:2].copy()
        peds_pos_traj.append(ped_pos)

        # Force sampling per pedestrian if requested
        if record_forces:
            forces = np.zeros_like(ped_pos)
            for i, p in enumerate(ped_pos):
                forces[i] = wrapper.get_forces_at(p)
            ped_forces_traj.append(forces)
        else:
            ped_forces_traj.append(np.zeros_like(ped_pos))

        robot_pos_traj.append(robot_pos.copy())
        robot_vel_traj.append(robot_vel.copy())
        robot_acc_traj.append(acc.copy())

        # Goal check
        if reached_goal_step is None and np.linalg.norm(robot_goal - robot_pos) < goal_radius:
            reached_goal_step = t
            break

        # Advance pedestrian simulation one step
        sim.step()

    ep: EpisodeData = _build_episode_data(
        robot_pos_traj,
        robot_vel_traj,
        robot_acc_traj,
        peds_pos_traj,
        ped_forces_traj,
        np.asarray(robot_goal_arr, dtype=float),
        dt,
        reached_goal_step,
    )

    metrics_raw = compute_all_metrics(ep, horizon=horizon)
    metrics: Dict[str, Any] = {k: v for k, v in metrics_raw.items()}
    # Adapt booleans for schema: success currently 0/1 float
    metrics["success"] = bool(metrics["success"] == 1.0)
    # Extract force quantiles into nested object matching schema (if present)
    fq = {k: v for k, v in metrics.items() if k.startswith("force_q")}
    if fq:
        metrics["force_quantiles"] = {
            "q50": float(fq.get("force_q50", float("nan"))),
            "q90": float(fq.get("force_q90", float("nan"))),
            "q95": float(fq.get("force_q95", float("nan"))),
        }
        for k in list(fq.keys()):
            metrics.pop(k, None)

    if snqi_weights is not None:
        metrics["snqi"] = snqi(metrics, snqi_weights, baseline_stats=snqi_baseline)

    record = {
        "episode_id": f"{scenario_params.get('id', 'unknown')}--{seed}",
        "scenario_id": scenario_params.get("id", "unknown"),
        "seed": seed,
        "scenario_params": scenario_params,
        "metrics": metrics,
        "config_hash": _config_hash(scenario_params),
        "git_hash": _git_hash_fallback(),
        "timestamps": {
            "start": datetime.now(timezone.utc).isoformat(),
            "end": datetime.now(timezone.utc).isoformat(),
        },
    }
    return record


def validate_and_write(
    record: Dict[str, Any], schema_path: str | Path, out_path: str | Path
) -> None:
    schema = load_schema(schema_path)
    validate_episode(record, schema)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


__all__ = [
    "load_scenario_matrix",
    "run_episode",
    "validate_and_write",
]
