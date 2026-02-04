#!/usr/bin/env python3
"""Run a full policy analysis sweep with metrics and optional videos.

This script evaluates a policy across all scenarios in a training config (or a
scenario YAML), computes benchmark metrics, and emits a digestible report that
highlights strengths and weaknesses. Video rendering is optional.

Examples:
  uv run python scripts/tools/policy_analysis_run.py \
    --training-config configs/training/ppo_imitation/expert_ppo_issue_403_grid_diffdrive_reverse_no_holdout_15m.yaml \
    --policy fast_pysf

  uv run python scripts/tools/policy_analysis_run.py \
    --training-config configs/training/ppo_imitation/expert_ppo_issue_403_grid_diffdrive_reverse_no_holdout_15m.yaml \
    --policy ppo --model-path output/wandb/.../model.zip --videos

  uv run python scripts/tools/policy_analysis_run.py \
    --scenario configs/scenarios/classic_interactions_francis2023.yaml \
    --policy-sweep --seed-set eval
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from loguru import logger

from robot_sf.benchmark.aggregate import compute_aggregates, read_jsonl
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, post_process_metrics
from robot_sf.benchmark.obstacle_sampling import sample_obstacle_points
from robot_sf.benchmark.path_utils import compute_shortest_path_length
from robot_sf.benchmark.schema_validator import load_schema, validate_episode
from robot_sf.benchmark.utils import _config_hash, _git_hash_fallback, compute_episode_id
from robot_sf.common.artifact_paths import (
    ensure_canonical_tree,
    get_artifact_category_path,
    resolve_artifact_path,
)
from robot_sf.common.logging import configure_logging
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.nav.occupancy_grid import GridConfig
from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter
from robot_sf.planner.fast_pysf_planner import FastPysfPlannerPolicy
from robot_sf.planner.socnav import (
    SocNavBenchComplexPolicy,
    SocNavPlannerConfig,
    SocNavPlannerPolicy,
    make_orca_policy,
    make_sacadrl_policy,
    make_social_force_policy,
)
from robot_sf.training.observation_wrappers import resolve_policy_obs_adapter, sync_policy_spaces
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    select_scenario,
)
from scripts.tools.render_scenario_videos import _defensive_obs_adapter
from scripts.training.train_expert_ppo import _apply_env_overrides, load_expert_training_config

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


_POLICY_CHOICES = (
    "goal",
    "ppo",
    "socnav_sampling",
    "socnav_social_force",
    "socnav_orca",
    "socnav_sacadrl",
    "socnav_bench",
    "fast_pysf",
    "fast_pysf_planner",
)
_SUPPORTED_POLICIES = frozenset(_POLICY_CHOICES)

_TIMESTAMP_TZ = ZoneInfo("Europe/Berlin")
_DEFAULT_SEED_SET_PATH = Path("configs/benchmarks/seed_sets_v1.yaml")

_SUMMARY_METRICS = (
    "path_efficiency",
    "shortest_path_len",
    "comfort_exposure",
    "jerk_mean",
    "jerk_mean_eps0p1",
    "curvature_mean",
    "curvature_mean_eps0p1",
    "low_speed_frac",
)

_POLICY_CATEGORIES = {
    "fast_pysf": "oracle",
    "fast_pysf_planner": "oracle",
    "goal": "heuristic",
    "socnav_sampling": "heuristic",
    "socnav_social_force": "heuristic",
    "socnav_orca": "heuristic",
    "socnav_sacadrl": "learned",
    "socnav_bench": "heuristic",
    "ppo": "learned",
}


@dataclass
class EpisodeArtifacts:
    """Per-episode artifact outputs."""

    video_path: Path | None = None


@dataclass
class PolicyAdapter:
    """Adapter that converts policy outputs into environment actions."""

    policy_name: str
    policy_model: Any | None
    socnav_policy: SocNavPlannerPolicy | None
    fast_pysf_policy: FastPysfPlannerPolicy | None
    planner_action_adapter: PlannerActionAdapter | None

    def action(self, obs: Any, env, *, robot_speed: float) -> np.ndarray:
        """Return an environment action for the given observation."""
        if self.policy_name == "goal":
            return _goal_action(env, speed=robot_speed)
        if self.policy_model is not None:
            action, _ = self.policy_model.predict(obs, deterministic=True)
            return np.asarray(action, dtype=float)
        if self.socnav_policy is not None:
            command = self.socnav_policy.act(obs)
            return self._velocity_command_to_action(command)
        if self.fast_pysf_policy is not None:
            command = self.fast_pysf_policy.action()
            return self._velocity_command_to_action(command)
        action_space = getattr(env, "action_space", None)
        if action_space is not None:
            return action_space.sample()
        return np.zeros(2, dtype=float)

    def _velocity_command_to_action(self, command: Sequence[float]) -> np.ndarray:
        """Convert a (v, w) command into the simulator action space."""
        if self.planner_action_adapter is None:
            return np.asarray(command, dtype=float)
        return self.planner_action_adapter.from_velocity_command(command)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-config",
        type=Path,
        help="Training config providing scenario path and env overrides.",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        help="Scenario YAML when no training config is supplied.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        help="Scenario id/name to run (defaults to all scenarios).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all scenarios (default when no scenario-id).",
    )
    parser.add_argument(
        "--policy",
        choices=_POLICY_CHOICES,
        default="goal",
        help="Policy name to evaluate.",
    )
    parser.add_argument(
        "--policy-sweep",
        action="store_true",
        help="Run a multi-policy sweep (fast_pysf_planner, ppo, socnav_orca) in one invocation.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        help="Comma-separated policy list for --policy-sweep overrides.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model/run_023.zip"),
        help="Model path for PPO policies.",
    )
    parser.add_argument(
        "--robot-speed",
        type=float,
        default=1.0,
        help="Nominal speed for goal policy.",
    )
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Enable video rendering for each episode.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="FPS for recorded videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for benchmark artifacts (defaults to timestamped folder).",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        help="Output directory for videos (defaults to output/recordings).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override seed (single seed for all scenarios).",
    )
    parser.add_argument(
        "--seed-set",
        choices=["dev", "eval"],
        help="Use a named deterministic seed set for evaluation (overrides scenario seeds).",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        help="Limit number of seeds per scenario.",
    )
    parser.add_argument(
        "--no-record-forces",
        dest="record_forces",
        action="store_false",
        help="Disable pedestrian force logging.",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames for top problem episodes (requires ffmpeg/ffprobe).",
    )
    parser.add_argument(
        "--extract-frames-output",
        type=Path,
        help="Override output directory for extracted frames (defaults under the run output).",
    )
    parser.set_defaults(record_forces=True)
    parser.add_argument(
        "--socnav-root",
        type=Path,
        help=(
            "Optional SocNavBench root for upstream sampling planner. "
            "Set ROBOT_SF_SOCNAV_ALLOW_UNTRUSTED_ROOT=1 to allow roots outside the repo."
        ),
    )
    parser.add_argument(
        "--socnav-allow-fallback",
        action="store_true",
        help="Allow heuristic fallback when SocNavBench dependencies are unavailable.",
    )
    parser.add_argument(
        "--socnav-use-grid",
        action="store_true",
        help="Enable occupancy grid fields for SocNav planners.",
    )
    parser.add_argument(
        "--socnav-grid-resolution",
        type=float,
        help="Override SocNav occupancy grid resolution.",
    )
    parser.add_argument(
        "--socnav-grid-width",
        type=float,
        help="Override SocNav occupancy grid width.",
    )
    parser.add_argument(
        "--socnav-grid-height",
        type=float,
        help="Override SocNav occupancy grid height.",
    )
    parser.add_argument(
        "--socnav-grid-center-on-robot",
        action="store_true",
        help="Center the SocNav occupancy grid on the robot.",
    )
    parser.add_argument(
        "--socnav-orca-time-horizon",
        type=float,
        help="Override ORCA time horizon for SocNav ORCA policy.",
    )
    parser.add_argument(
        "--socnav-orca-neighbor-dist",
        type=float,
        help="Override ORCA neighbor distance for SocNav ORCA policy.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--write-plausibility-metrics",
        action="store_true",
        help=(
            "Update scenario metadata.plausibility.metrics with aggregated metrics from this run. "
            "Opt-in only; status remains unchanged."
        ),
    )
    return parser


def _resolve_output_root(base: Path | None, *, policy: str, stamp: str | None = None) -> Path:
    """Resolve benchmark output directory."""
    if base is not None:
        return resolve_artifact_path(base)
    ts = stamp or datetime.now(_TIMESTAMP_TZ).strftime("%Y%m%d_%H%M%S")
    return get_artifact_category_path("benchmarks") / f"{ts}_policy_analysis_{policy}"


def _resolve_video_root(base: Path | None, *, policy: str, stamp: str | None = None) -> Path:
    """Resolve video output directory."""
    if base is not None:
        return resolve_artifact_path(base)
    ts = stamp or datetime.now(_TIMESTAMP_TZ).strftime("%Y%m%d_%H%M%S")
    return get_artifact_category_path("recordings") / f"{ts}_policy_analysis_{policy}"


def _resolve_sweep_root(base: Path | None, *, stamp: str | None = None) -> Path:
    """Resolve output root for multi-policy sweeps."""
    if base is not None:
        return resolve_artifact_path(base)
    ts = stamp or datetime.now(_TIMESTAMP_TZ).strftime("%Y%m%d_%H%M%S")
    return get_artifact_category_path("benchmarks") / f"{ts}_policy_sweep"


def _resolve_sweep_video_root(base: Path | None, *, stamp: str | None = None) -> Path:
    """Resolve video output root for multi-policy sweeps."""
    if base is not None:
        return resolve_artifact_path(base)
    ts = stamp or datetime.now(_TIMESTAMP_TZ).strftime("%Y%m%d_%H%M%S")
    return get_artifact_category_path("recordings") / f"{ts}_policy_sweep"


def _load_seed_sets(path: Path) -> dict[str, list[int]]:
    """Load named seed sets from YAML."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Seed set file '{path}' must contain a mapping.")
    seed_sets: dict[str, list[int]] = {}
    for key, value in data.items():
        if not isinstance(value, list):
            continue
        seed_sets[str(key)] = [int(s) for s in value]
    return seed_sets


def _resolve_seeds(
    scenario: Mapping[str, Any],
    *,
    seed_override: int | None,
    max_seeds: int | None,
    training_seeds: Sequence[int] | None,
    seed_set: Sequence[int] | None,
) -> list[int]:
    """Resolve per-scenario seeds, honoring overrides and training config defaults."""
    if seed_override is not None:
        return [int(seed_override)]
    seeds: list[int] = []
    if seed_set:
        seeds = [int(s) for s in seed_set]
    elif training_seeds:
        seeds = [int(s) for s in training_seeds]
    else:
        raw = scenario.get("seeds")
        if isinstance(raw, list) and raw:
            seeds = [int(s) for s in raw]
    if not seeds:
        seeds = [0]
    if max_seeds is not None and max_seeds > 0:
        return seeds[: int(max_seeds)]
    return seeds


def _goal_action(env, *, speed: float) -> np.ndarray:
    """Return a simple goal-seeking action in (v, w) space."""
    if getattr(env, "simulator", None) is None:
        return np.zeros(2, dtype=float)
    (x, y), heading = env.simulator.robot_poses[0]
    goal = np.asarray(env.simulator.goal_pos[0], dtype=float)
    vec = goal - np.array([x, y], dtype=float)
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        return np.zeros(2, dtype=float)
    desired_heading = float(math.atan2(vec[1], vec[0]))
    heading_error = ((desired_heading - heading + math.pi) % (2 * math.pi)) - math.pi
    angular = float(np.clip(heading_error, -1.0, 1.0))
    linear = float(np.clip(dist, 0.0, speed * max(0.0, 1.0 - abs(heading_error) / math.pi)))
    return np.array([linear, angular], dtype=float)


def _build_socnav_policy(
    policy_name: str,
    *,
    socnav_root: Path | None,
    orca_time_horizon: float | None,
    orca_neighbor_dist: float | None,
    socnav_allow_fallback: bool,
) -> SocNavPlannerPolicy | None:
    """Construct the SocNav planner policy for the selected CLI mode."""
    if policy_name == "socnav_sampling":
        return SocNavBenchComplexPolicy(
            socnav_root=socnav_root,
            adapter_config=SocNavPlannerConfig(),
            allow_fallback=socnav_allow_fallback,
        )
    if policy_name == "socnav_social_force":
        return make_social_force_policy()
    if policy_name == "socnav_orca":
        policy = make_orca_policy()
        if orca_time_horizon is not None:
            policy.adapter.config.orca_time_horizon = float(orca_time_horizon)
        if orca_neighbor_dist is not None:
            policy.adapter.config.orca_neighbor_dist = float(orca_neighbor_dist)
        return policy
    if policy_name == "socnav_sacadrl":
        return make_sacadrl_policy()
    if policy_name == "socnav_bench":
        return SocNavBenchComplexPolicy(
            socnav_root=socnav_root,
            adapter_config=SocNavPlannerConfig(),
            allow_fallback=socnav_allow_fallback,
        )
    return None


def _apply_socnav_grid_overrides(
    config: Any,
    *,
    resolution: float | None,
    width: float | None,
    height: float | None,
    center_on_robot: bool,
) -> None:
    """Apply occupancy grid overrides for SocNav planners."""
    config.use_occupancy_grid = True
    config.include_grid_in_observation = True
    if config.grid_config is None:
        config.grid_config = GridConfig()
    if resolution is not None:
        config.grid_config.resolution = float(resolution)
    if width is not None:
        config.grid_config.width = float(width)
    if height is not None:
        config.grid_config.height = float(height)
    if center_on_robot:
        config.grid_config.center_on_robot = True


def _build_planner_action_adapter(env, config: Any) -> PlannerActionAdapter | None:
    """Create a planner action adapter when the simulator is available."""
    if getattr(env, "simulator", None) is None:
        return None
    return PlannerActionAdapter(
        robot=env.simulator.robots[0],
        action_space=env.action_space,
        time_step=config.sim_config.time_per_step_in_secs,
    )


def _vel_and_acc(positions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocities and accelerations from positions."""
    if positions.shape[0] < 2:
        return np.zeros_like(positions), np.zeros_like(positions)
    vel = np.gradient(positions, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def _filtered_jerk_mean(
    vel: np.ndarray,
    acc: np.ndarray,
    *,
    speed_eps: float,
) -> float:
    """Return mean jerk magnitude for timesteps with speed >= speed_eps."""
    if acc.shape[0] < 3:
        return 0.0
    diffs = acc[1:] - acc[:-1]
    jerk_vecs = diffs[:-1]
    norms = np.linalg.norm(jerk_vecs, axis=1)
    if vel.shape[0] < 3:
        return 0.0
    speeds = np.linalg.norm(vel, axis=1)
    mask = speeds[1:-1] >= float(speed_eps)
    if not np.any(mask):
        return 0.0
    return float(np.mean(norms[mask]))


def _filtered_curvature_mean(
    pos: np.ndarray,
    dt: float,
    *,
    speed_eps: float,
) -> float:
    """Return mean curvature for timesteps with speed >= speed_eps."""
    if pos.shape[0] < 4:
        return 0.0
    if not np.isfinite(dt) or dt <= 0.0:
        return 0.0
    vel = np.diff(pos, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    v = vel[1:]
    a = acc
    cross = np.abs(v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0])
    v_mag = np.linalg.norm(v, axis=1)
    mask = v_mag >= float(speed_eps)
    if not np.any(mask):
        return 0.0
    denom = v_mag**3
    kappa = np.zeros_like(v_mag)
    kappa[mask] = cross[mask] / denom[mask]
    finite = np.isfinite(kappa) & mask
    if not np.any(finite):
        return 0.0
    return float(np.mean(kappa[finite]))


def _stack_ped_positions(traj: list[np.ndarray], *, fill_value: float = np.nan) -> np.ndarray:
    """Stack pedestrian trajectories into a padded array."""
    if not traj:
        return np.zeros((0, 0, 2), dtype=float)
    max_k = max(p.shape[0] for p in traj)
    stacked = np.full((len(traj), max_k, 2), fill_value, dtype=float)
    for i, arr in enumerate(traj):
        if arr.size == 0:
            continue
        stacked[i, : arr.shape[0]] = arr
    return stacked


def _episode_video_metadata(video_path: Path | None, *, frames: int) -> dict[str, Any] | None:
    """Build video metadata for the episode record."""
    if video_path is None or not video_path.exists():
        return None
    size = int(video_path.stat().st_size)
    if size <= 0:
        return None
    try:
        path_str = str(video_path.relative_to(Path.cwd()))
    except ValueError:
        path_str = str(video_path)
    return {
        "path": path_str,
        "format": "mp4",
        "filesize_bytes": size,
        "frames": int(frames),
        "renderer": "sim-view",
    }


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a compact summary payload from episode records."""
    if not records:
        return {"episodes": 0, "success_rate": 0.0, "collision_rate": 0.0}
    successes = sum(1 for r in records if r.get("metrics", {}).get("success"))
    collisions = sum(1 for r in records if int(r.get("metrics", {}).get("collisions", 0)) > 0)
    timeouts = sum(1 for r in records if r.get("status") == "failure")
    metric_means: dict[str, float] = {}
    for metric in _SUMMARY_METRICS:
        values: list[float] = []
        for rec in records:
            raw = rec.get("metrics", {}).get(metric)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                values.append(val)
        if values:
            metric_means[metric] = float(np.mean(values))
    return {
        "episodes": len(records),
        "success_rate": successes / len(records),
        "collision_rate": collisions / len(records),
        "failures": timeouts,
        "metric_means": metric_means,
    }


def _rank_scenarios(
    aggregates: dict[str, dict[str, dict[str, float]]],
    *,
    metric: str,
    reverse: bool,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return top-N scenarios by a metric mean."""
    rows: list[tuple[str, float]] = []
    for scenario_id, stats in aggregates.items():
        if scenario_id == "_meta":
            continue
        value = stats.get(metric, {}).get("mean")
        if value is None or not math.isfinite(value):
            continue
        rows.append((scenario_id, float(value)))
    rows.sort(key=lambda x: x[1], reverse=reverse)
    return [
        {"scenario_id": scenario_id, "mean": value} for scenario_id, value in rows[: max(top_n, 0)]
    ]


def _find_problem_episodes(
    records: list[dict[str, Any]],
    *,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Identify problematic episodes based on collision or comfort exposure."""
    scored: list[tuple[float, dict[str, Any]]] = []
    for rec in records:
        metrics = rec.get("metrics", {})
        collisions = float(metrics.get("collisions", 0.0) or 0.0)
        comfort = float(metrics.get("comfort_exposure", 0.0) or 0.0)
        failure = 1.0 if rec.get("status") == "failure" else 0.0
        score = collisions * 10.0 + comfort * 2.0 + failure
        if score <= 0:
            continue
        scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for score, rec in scored[: max(top_n, 0)]:
        out.append(
            {
                "scenario_id": rec.get("scenario_id"),
                "seed": rec.get("seed"),
                "score": score,
                "metrics": rec.get("metrics", {}),
                "video": rec.get("video"),
            }
        )
    return out


def _write_report(
    out_dir: Path,
    *,
    summary: dict[str, Any],
    aggregates: dict[str, dict[str, dict[str, float]]],
    problem_episodes: list[dict[str, Any]],
    policy: str,
    scenario_path: Path,
) -> tuple[Path, Path]:
    """Write markdown + JSON reports."""
    report_json = {
        "policy": policy,
        "scenario_path": str(scenario_path),
        "summary": summary,
        "aggregates": aggregates,
        "problem_episodes": problem_episodes,
    }
    report_json_path = out_dir / "report.json"
    report_json_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    top_collision = _rank_scenarios(aggregates, metric="collisions", reverse=True)
    top_success = _rank_scenarios(aggregates, metric="success", reverse=True)
    top_timeout = _rank_scenarios(aggregates, metric="failure_to_progress", reverse=True)

    lines = [
        "# Policy Analysis Report",
        "",
        f"- Policy: `{policy}`",
        f"- Scenario file: `{scenario_path}`",
        f"- Episodes: {summary.get('episodes', 0)}",
        f"- Success rate: {summary.get('success_rate', 0.0):.3f}",
        f"- Collision rate: {summary.get('collision_rate', 0.0):.3f}",
        "",
        "## Strengths (top success rate)",
    ]
    for row in top_success:
        lines.append(f"- {row['scenario_id']}: success_mean={row['mean']:.3f}")

    lines.extend(["", "## Weaknesses (top collision counts)"])
    for row in top_collision:
        lines.append(f"- {row['scenario_id']}: collisions_mean={row['mean']:.3f}")

    if top_timeout:
        lines.extend(["", "## Failure to progress (timeouts)"])
        for row in top_timeout:
            lines.append(f"- {row['scenario_id']}: failure_to_progress_mean={row['mean']:.3f}")

    if problem_episodes:
        lines.extend(["", "## Top problem episodes"])
        for item in problem_episodes:
            scen = item.get("scenario_id")
            seed = item.get("seed")
            score = item.get("score")
            video_meta = item.get("video") or {}
            video = video_meta.get("path")
            lines.append(f"- {scen} seed={seed} score={score:.2f} video={video}")

    report_md_path = out_dir / "report.md"
    metric_means = summary.get("metric_means", {})
    if metric_means:
        lines.extend(["", "## Key metrics (mean over episodes)"])
        for key, value in sorted(metric_means.items()):
            lines.append(f"- {key}: {value:.4f}")

    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_md_path, report_json_path


def _write_combined_report(
    out_dir: Path,
    *,
    policy_entries: list[dict[str, Any]],
    scenario_path: Path,
    seed_set_name: str | None,
) -> tuple[Path, Path]:
    """Write a combined report for a multi-policy sweep."""
    report = {
        "scenario_path": str(scenario_path),
        "seed_set": seed_set_name,
        "policies": policy_entries,
    }
    report_json_path = out_dir / "combined_report.json"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Policy Sweep Report",
        "",
        f"- Scenario file: `{scenario_path}`",
        f"- Seed set: `{seed_set_name or 'scenario/default'}`",
        "",
        "## Summary",
        "",
        "| Policy | Category | Success rate | Collision rate | Failures |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in policy_entries:
        summary = entry.get("summary", {})
        lines.append(
            "| {policy} | {category} | {success:.3f} | {collision:.3f} | {failures} |".format(
                policy=entry.get("policy", "-"),
                category=entry.get("category", "-"),
                success=summary.get("success_rate", 0.0),
                collision=summary.get("collision_rate", 0.0),
                failures=summary.get("failures", 0),
            )
        )

    lines.extend(["", "## Key metrics (mean over episodes)", ""])
    metric_keys = sorted({key for entry in policy_entries for key in entry.get("metric_means", {})})
    if not metric_keys:
        lines.append("_No metric means available._")
    else:
        header = "| Policy | " + " | ".join(metric_keys) + " |"
        divider = "| --- | " + " | ".join("---" for _ in metric_keys) + " |"
        lines.extend([header, divider])
        for entry in policy_entries:
            means = entry.get("metric_means", {})
            row = [entry.get("policy", "-")]
            for key in metric_keys:
                value = means.get(key)
                row.append(f"{value:.4f}" if isinstance(value, (int, float)) else "-")
            lines.append("| " + " | ".join(row) + " |")

    report_md_path = out_dir / "combined_report.md"
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_md_path, report_json_path


def _finalize_run(
    output_root: Path,
    *,
    scenario_path: Path,
    records: list[dict[str, Any]],
    policy_name: str,
    video_root: Path | None,
    write_plausibility_metrics: bool,
    seed_set_name: str | None,
) -> dict[str, Any]:
    summary = _summarize_records(records)
    aggregates = compute_aggregates(
        records, group_by="scenario_id", fallback_group_by="scenario_id"
    )
    problem_episodes = _find_problem_episodes(records)

    summary_payload = {
        "summary": summary,
        "aggregates": aggregates,
        "problem_episodes": problem_episodes,
        "scenario_path": str(scenario_path),
        "seed_set": seed_set_name,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    report_md, report_json = _write_report(
        output_root,
        summary=summary,
        aggregates=aggregates,
        problem_episodes=problem_episodes,
        policy=policy_name,
        scenario_path=scenario_path,
    )

    logger.info("Wrote episodes to {}", output_root / "episodes.jsonl")
    logger.info("Wrote summary to {}", summary_path)
    logger.info("Wrote report to {} and {}", report_md, report_json)
    if video_root is not None:
        logger.info("Videos saved under {}", video_root)
    if write_plausibility_metrics:
        _update_plausibility_metadata(
            scenario_path,
            records=records,
            policy_name=policy_name,
        )
    return {
        "summary": summary,
        "aggregates": aggregates,
        "problem_episodes": problem_episodes,
        "report_md": str(report_md),
        "report_json": str(report_json),
        "summary_json": str(summary_path),
    }


def _write_jsonl(out_path: Path, schema: dict[str, Any], record: dict[str, Any]) -> None:
    validate_episode(record, schema)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _collect_scenario_source_files(path: Path) -> list[Path]:
    """Collect scenario YAML files referenced by a manifest entry point."""
    resolved = path.resolve()
    visited: set[Path] = set()

    def _walk(file_path: Path, sources: set[Path]) -> None:
        candidate = file_path.resolve()
        if candidate in visited:
            return
        visited.add(candidate)
        data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        if isinstance(data, list):
            sources.add(candidate)
            return
        if not isinstance(data, dict):
            return
        includes = data.get("includes") or data.get("include") or data.get("scenario_files")
        if includes:
            entries = [includes] if isinstance(includes, (str, Path)) else includes
            for entry in entries:
                entry_path = Path(entry)
                if not entry_path.is_absolute():
                    entry_path = (candidate.parent / entry_path).resolve()
                _walk(entry_path, sources)
        if "scenarios" in data:
            sources.add(candidate)

    sources: set[Path] = set()
    _walk(resolved, sources)
    return sorted(sources)


def _aggregate_plausibility_metrics(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate plausibility metrics per scenario from episode records."""
    metrics_keys = (
        "min_distance",
        "mean_distance",
        "robot_ped_within_5m_frac",
        "ped_force_mean",
        "force_q95",
    )
    collected: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        scenario_id = str(rec.get("scenario_id") or "")
        if not scenario_id:
            continue
        metrics = rec.get("metrics", {})
        bucket = collected.setdefault(scenario_id, {key: [] for key in metrics_keys})
        for key in metrics_keys:
            if key == "force_q95":
                fq = metrics.get("force_quantiles") or {}
                val = fq.get("q95")
            else:
                val = metrics.get(key)
            if isinstance(val, (int, float)) and math.isfinite(val):
                bucket[key].append(float(val))

    aggregated: dict[str, dict[str, float]] = {}
    for scenario_id, values in collected.items():
        aggregated[scenario_id] = {}
        for key, samples in values.items():
            if samples:
                aggregated[scenario_id][key] = float(sum(samples) / len(samples))
    return aggregated


def _extract_scenarios_from_yaml(data: Any) -> list[dict[str, Any]] | None:
    """Return scenario list from YAML data or None if not applicable."""
    if isinstance(data, list):
        return [sc for sc in data if isinstance(sc, dict)]
    if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
        return [sc for sc in data["scenarios"] if isinstance(sc, dict)]
    return None


def _apply_plausibility_update(
    scenarios: list[dict[str, Any]],
    *,
    metrics_by_scenario: dict[str, dict[str, float]],
    timestamp: str,
) -> int:
    """Update scenarios in-place with plausibility metrics and return count updated."""
    updated = 0
    for scenario in scenarios:
        scenario_id = str(scenario.get("name") or scenario.get("scenario_id") or "")
        if not scenario_id or scenario_id not in metrics_by_scenario:
            continue
        metadata = scenario.setdefault("metadata", {})
        plausibility = metadata.setdefault("plausibility", {})
        plausibility.setdefault("status", "unverified")
        plausibility.setdefault("verified_on", None)
        plausibility.setdefault("verified_by", None)
        plausibility.setdefault("method", None)
        plausibility.setdefault("notes", None)
        plausibility["metrics_updated_on"] = timestamp
        plausibility_metrics = plausibility.setdefault("metrics", {})
        for key, value in metrics_by_scenario[scenario_id].items():
            plausibility_metrics[key] = value
        updated += 1
    return updated


def _update_plausibility_metadata(
    scenario_path: Path,
    *,
    records: list[dict[str, Any]],
    policy_name: str,
) -> None:
    """Update scenario files with plausibility metrics from the current run."""
    metrics_by_scenario = _aggregate_plausibility_metrics(records)
    if not metrics_by_scenario:
        logger.warning("No metrics available to update plausibility metadata.")
        return
    timestamp = datetime.now(_TIMESTAMP_TZ).isoformat()
    sources = _collect_scenario_source_files(scenario_path)
    if not sources:
        logger.warning("No scenario source files found for plausibility update.")
        return

    updated_scenarios = 0
    updated_files = 0
    for path in sources:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        scenarios = _extract_scenarios_from_yaml(data)
        if scenarios is None:
            continue
        updated = _apply_plausibility_update(
            scenarios,
            metrics_by_scenario=metrics_by_scenario,
            timestamp=timestamp,
        )
        if updated:
            updated_scenarios += updated
            if isinstance(data, list):
                out = data
            else:
                out = dict(data)
                out["scenarios"] = scenarios
            path.write_text(
                yaml.safe_dump(out, sort_keys=False, width=100),
                encoding="utf-8",
            )
            updated_files += 1

    logger.info(
        "Updated plausibility metrics for {} scenarios across {} files (policy={}, timestamp={}).",
        updated_scenarios,
        updated_files,
        policy_name,
        timestamp,
    )


@dataclass
class EpisodeTrajectory:
    """Container for trajectory samples collected during an episode."""

    robot_positions: list[np.ndarray]
    ped_positions: list[np.ndarray]
    ped_forces: list[np.ndarray]


def _prepare_episode_config(
    scenario: Mapping[str, Any],
    *,
    scenario_path: Path,
    policy_name: str,
    socnav_policy: SocNavPlannerPolicy | None,
    socnav_use_grid: bool,
    socnav_grid_resolution: float | None,
    socnav_grid_width: float | None,
    socnav_grid_height: float | None,
    socnav_grid_center_on_robot: bool,
    env_overrides: Mapping[str, object],
    max_steps_override: int | None,
    videos: bool,
    video_dir: Path | None,
    seed: int,
) -> tuple[Any, int, Path | None]:
    """Build the env config, max steps, and optional video path for an episode."""
    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    _apply_env_overrides(config, env_overrides)
    config.use_planner = False
    if socnav_policy is not None:
        config.observation_mode = ObservationMode.SOCNAV_STRUCT
        if socnav_use_grid:
            _apply_socnav_grid_overrides(
                config,
                resolution=socnav_grid_resolution,
                width=socnav_grid_width,
                height=socnav_grid_height,
                center_on_robot=socnav_grid_center_on_robot,
            )

    max_steps = int(scenario.get("simulation_config", {}).get("max_episode_steps", 0) or 0)
    if max_steps_override is not None:
        max_steps = max_steps_override
    if max_steps <= 0:
        max_steps = 200

    episode_video = None
    if videos and video_dir is not None:
        scenario_name = str(scenario.get("name") or scenario.get("scenario_id") or "scenario")
        slug = re.sub(r"[^\w.-]+", "_", scenario_name).strip("._")
        if not slug or slug in {".", ".."}:
            slug = "scenario"
        episode_video = video_dir / f"{slug}_seed{seed}_{policy_name}.mp4"
    return config, max_steps, episode_video


def _make_env(
    *,
    config: Any,
    seed: int,
    videos: bool,
    video_path: Path | None,
    video_fps: int,
    env_factory_kwargs: Mapping[str, object],
) -> Any:
    """Create the environment for a single episode."""
    return make_robot_env(
        config=config,
        seed=int(seed),
        debug=bool(videos),
        record_video=bool(videos),
        video_path=str(video_path) if video_path is not None else None,
        video_fps=float(video_fps),
        **env_factory_kwargs,
    )


def _reset_env(
    env,
    *,
    seed: int,
    policy_model: Any | None,
    policy_obs_adapter: Callable[[Mapping[str, Any]], np.ndarray] | None,
) -> Any:
    """Reset the environment and apply optional observation adapters."""
    obs, _ = env.reset(seed=int(seed))
    sync_policy_spaces(env, policy_model)
    if policy_model is not None and policy_obs_adapter is not None:
        return policy_obs_adapter(obs)
    return obs


def _build_policy_adapter(
    env,
    config: Any,
    *,
    policy_name: str,
    policy_model: Any | None,
    socnav_policy: SocNavPlannerPolicy | None,
) -> PolicyAdapter:
    """Instantiate the policy adapter for the current environment."""
    fast_pysf_policy: FastPysfPlannerPolicy | None = None
    if policy_name in {"fast_pysf", "fast_pysf_planner"}:
        if getattr(env, "simulator", None) is None:
            raise RuntimeError("fast_pysf policy requires env.simulator")
        fast_pysf_policy = FastPysfPlannerPolicy(env.simulator)

    planner_action_adapter = None
    if socnav_policy is not None or fast_pysf_policy is not None:
        planner_action_adapter = _build_planner_action_adapter(env, config)
    return PolicyAdapter(
        policy_name=policy_name,
        policy_model=policy_model,
        socnav_policy=socnav_policy,
        fast_pysf_policy=fast_pysf_policy,
        planner_action_adapter=planner_action_adapter,
    )


def _collect_episode_trajectories(
    env,
    obs: Any,
    *,
    policy_adapter: PolicyAdapter,
    policy_model: Any | None,
    policy_obs_adapter: Callable[[Mapping[str, Any]], np.ndarray] | None,
    robot_speed: float,
    record_forces: bool,
    max_steps: int,
    videos: bool,
) -> tuple[EpisodeTrajectory, int | None, float]:
    """Run the episode loop and return trajectory data."""
    robot_positions: list[np.ndarray] = []
    ped_positions: list[np.ndarray] = []
    ped_forces: list[np.ndarray] = []
    reached_goal_step: int | None = None

    start_time = time.time()
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    for step_idx in range(max_steps):
        action = policy_adapter.action(obs, env, robot_speed=robot_speed)
        obs, _reward, terminated, truncated, info = env.step(action)
        if policy_model is not None and policy_obs_adapter is not None:
            obs = policy_obs_adapter(obs)
        if videos:
            env.render()

        robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
        peds = np.asarray(env.simulator.ped_pos, dtype=float)
        if record_forces:
            forces = getattr(env.simulator, "last_ped_forces", None)
            if forces is None:
                forces_arr = np.zeros_like(peds, dtype=float)
            else:
                forces_arr = np.asarray(forces, dtype=float)
                if forces_arr.shape != peds.shape:
                    forces_arr = np.zeros_like(peds, dtype=float)
        else:
            forces_arr = np.zeros_like(peds, dtype=float)

        robot_positions.append(robot_pos)
        ped_positions.append(peds)
        ped_forces.append(forces_arr)

        if reached_goal_step is None and bool(info.get("success") or info.get("is_success")):
            reached_goal_step = step_idx
        if terminated or truncated:
            break

    wall_time = float(max(1e-9, time.time() - start_time))
    return (
        EpisodeTrajectory(robot_positions, ped_positions, ped_forces),
        reached_goal_step,
        wall_time,
    )


def _build_episode_record(
    scenario: Mapping[str, Any],
    *,
    seed: int,
    policy_name: str,
    map_def: Any,
    goal_vec: np.ndarray,
    trajectory: EpisodeTrajectory,
    reached_goal_step: int | None,
    wall_time: float,
    max_steps: int,
    dt: float,
    ts_start: str,
    video_path: Path | None,
) -> dict[str, Any]:
    """Compute metrics and assemble an episode record."""
    robot_pos_arr = np.asarray(trajectory.robot_positions, dtype=float)
    robot_vel_arr, robot_acc_arr = _vel_and_acc(robot_pos_arr, dt)
    ped_pos_arr = _stack_ped_positions(trajectory.ped_positions)
    ped_forces_arr = _stack_ped_positions(trajectory.ped_forces, fill_value=np.nan)

    obstacles = sample_obstacle_points(map_def.obstacles, map_def.bounds) if map_def else None
    shortest_path = (
        compute_shortest_path_length(map_def, robot_pos_arr[0], goal_vec)
        if robot_pos_arr.size and map_def is not None
        else float("nan")
    )

    if robot_pos_arr.size == 0:
        metrics_raw = {"success": 0.0, "time_to_goal_norm": float("nan"), "collisions": 0.0}
    else:
        ep = EpisodeData(
            robot_pos=robot_pos_arr,
            robot_vel=robot_vel_arr,
            robot_acc=robot_acc_arr,
            peds_pos=ped_pos_arr,
            ped_forces=ped_forces_arr,
            obstacles=obstacles,
            goal=goal_vec,
            dt=float(dt),
            reached_goal_step=reached_goal_step,
        )
        metrics_raw = compute_all_metrics(ep, horizon=max_steps, shortest_path_len=shortest_path)
        metrics_raw["shortest_path_len"] = float(shortest_path)
        # Filter jerk/curvature at low speeds to avoid numerical blow-ups when |v| ~ 0.
        speed_eps = 0.1
        speeds = np.linalg.norm(robot_vel_arr, axis=1)
        if speeds.size:
            metrics_raw["low_speed_frac"] = float(np.mean(speeds < speed_eps))
        metrics_raw["jerk_mean_eps0p1"] = _filtered_jerk_mean(
            robot_vel_arr,
            robot_acc_arr,
            speed_eps=speed_eps,
        )
        metrics_raw["curvature_mean_eps0p1"] = _filtered_curvature_mean(
            robot_pos_arr,
            float(dt),
            speed_eps=speed_eps,
        )

    metrics = post_process_metrics(metrics_raw, snqi_weights=None, snqi_baseline=None)
    status = "success" if metrics.get("success") else "failure"
    if metrics.get("collisions"):
        status = "collision"

    scenario_id = str(scenario.get("name") or scenario.get("scenario_id") or scenario.get("id"))
    scenario_params = dict(scenario)
    scenario_params.setdefault("id", scenario_id)
    scenario_params.setdefault("algo", policy_name)

    record = {
        "version": "v1",
        "episode_id": compute_episode_id(scenario_params, seed),
        "scenario_id": scenario_id,
        "seed": int(seed),
        "scenario_params": scenario_params,
        "metrics": metrics,
        "algorithm_metadata": {"algorithm": policy_name},
        "algo": policy_name,
        "config_hash": _config_hash(scenario_params),
        "git_hash": _git_hash_fallback(),
        "timestamps": {"start": ts_start, "end": datetime.now(_TIMESTAMP_TZ).isoformat()},
        "status": status,
        "steps": int(robot_pos_arr.shape[0]),
        "horizon": int(max_steps),
        "wall_time_sec": wall_time,
        "timing": {"steps_per_second": float(robot_pos_arr.shape[0]) / wall_time},
    }

    video_meta = _episode_video_metadata(video_path, frames=int(robot_pos_arr.shape[0]))
    if video_meta:
        record["video"] = video_meta
    return record


def _run_episode(
    scenario: Mapping[str, Any],
    *,
    scenario_path: Path,
    seed: int,
    policy_name: str,
    policy_model: Any | None,
    socnav_policy: SocNavPlannerPolicy | None,
    policy_obs_adapter: Callable[[Mapping[str, Any]], np.ndarray] | None,
    robot_speed: float,
    env_overrides: Mapping[str, object],
    env_factory_kwargs: Mapping[str, object],
    record_forces: bool,
    max_steps_override: int | None,
    videos: bool,
    video_dir: Path | None,
    video_fps: int,
    socnav_use_grid: bool,
    socnav_grid_resolution: float | None,
    socnav_grid_width: float | None,
    socnav_grid_height: float | None,
    socnav_grid_center_on_robot: bool,
) -> tuple[dict[str, Any], EpisodeArtifacts]:
    """Run a single episode, compute metrics, and return a record."""
    config, max_steps, episode_video = _prepare_episode_config(
        scenario,
        scenario_path=scenario_path,
        policy_name=policy_name,
        socnav_policy=socnav_policy,
        socnav_use_grid=socnav_use_grid,
        socnav_grid_resolution=socnav_grid_resolution,
        socnav_grid_width=socnav_grid_width,
        socnav_grid_height=socnav_grid_height,
        socnav_grid_center_on_robot=socnav_grid_center_on_robot,
        env_overrides=env_overrides,
        max_steps_override=max_steps_override,
        videos=videos,
        video_dir=video_dir,
        seed=seed,
    )
    env = _make_env(
        config=config,
        seed=seed,
        videos=videos,
        video_path=episode_video,
        video_fps=video_fps,
        env_factory_kwargs=env_factory_kwargs,
    )
    ts_start = datetime.now(_TIMESTAMP_TZ).isoformat()
    record: dict[str, Any] | None = None
    try:
        obs = _reset_env(
            env, seed=seed, policy_model=policy_model, policy_obs_adapter=policy_obs_adapter
        )
        policy_adapter = _build_policy_adapter(
            env,
            config,
            policy_name=policy_name,
            policy_model=policy_model,
            socnav_policy=socnav_policy,
        )
        trajectory, reached_goal_step, wall_time = _collect_episode_trajectories(
            env,
            obs,
            policy_adapter=policy_adapter,
            policy_model=policy_model,
            policy_obs_adapter=policy_obs_adapter,
            robot_speed=robot_speed,
            record_forces=record_forces,
            max_steps=max_steps,
            videos=videos,
        )
        record = _build_episode_record(
            scenario,
            seed=seed,
            policy_name=policy_name,
            map_def=env.simulator.map_def,
            goal_vec=np.asarray(env.simulator.goal_pos[0], dtype=float),
            trajectory=trajectory,
            reached_goal_step=reached_goal_step,
            wall_time=wall_time,
            max_steps=max_steps,
            dt=config.sim_config.time_per_step_in_secs,
            ts_start=ts_start,
            video_path=episode_video,
        )
        return record, EpisodeArtifacts(video_path=episode_video)
    finally:
        _close_env(env)
        if record is not None and episode_video is not None and "video" not in record:
            # Video metadata is only available after the env exits and writes the file.
            video_meta = _episode_video_metadata(episode_video, frames=record.get("steps", 0))
            if video_meta:
                record["video"] = video_meta


def _close_env(env) -> None:
    """Best-effort shutdown for the environment to flush recordings."""
    for method in ("exit", "close"):
        try:
            getattr(env, method)()
        except Exception as exc:
            logger.warning("Failed to call env.{}(): {}", method, exc)


@dataclass
class TrainingContext:
    """Resolved training context for policy analysis."""

    training_config: Any | None
    env_overrides: Mapping[str, object]
    env_factory_kwargs: Mapping[str, object]
    training_seeds: Sequence[int] | None
    orca_time_horizon: float | None
    orca_neighbor_dist: float | None


def _load_training_context(args) -> TrainingContext:
    """Load training config and derived context when provided."""
    training_config = None
    env_overrides: Mapping[str, object] = {}
    env_factory_kwargs: Mapping[str, object] = {}
    training_seeds: Sequence[int] | None = None
    orca_time_horizon = args.socnav_orca_time_horizon
    orca_neighbor_dist = args.socnav_orca_neighbor_dist

    if args.training_config is not None:
        training_config = load_expert_training_config(args.training_config)
        env_overrides = training_config.env_overrides
        env_factory_kwargs = training_config.env_factory_kwargs
        training_seeds = training_config.seeds
        if orca_time_horizon is None:
            orca_time_horizon = training_config.socnav_orca_time_horizon
        if orca_neighbor_dist is None:
            orca_neighbor_dist = training_config.socnav_orca_neighbor_dist

    return TrainingContext(
        training_config=training_config,
        env_overrides=env_overrides,
        env_factory_kwargs=env_factory_kwargs,
        training_seeds=training_seeds,
        orca_time_horizon=orca_time_horizon,
        orca_neighbor_dist=orca_neighbor_dist,
    )


def _resolve_scenarios(
    args,
    *,
    training_config: Any | None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Resolve scenario path and scenario list."""
    scenario_path = (
        args.scenario
        or (training_config.scenario_config if training_config is not None else None)
        or Path("configs/scenarios/francis2023.yaml")
    ).resolve()
    scenarios = load_scenarios(scenario_path, base_dir=scenario_path)
    if args.scenario_id and args.all:
        raise ValueError("Use --all or --scenario-id, not both.")
    if args.scenario_id:
        scenarios = [select_scenario(scenarios, args.scenario_id)]
    elif not args.all:
        logger.info("No scenario-id provided; defaulting to --all.")
    return scenario_path, scenarios


def _load_policy_model(
    policy_name: str,
    *,
    model_path: Path,
) -> tuple[Any | None, Callable[[Mapping[str, Any]], np.ndarray] | None]:
    """Load PPO policy and observation adapter when requested."""
    if policy_name != "ppo":
        return None, None
    from robot_sf.benchmark.helper_catalog import load_trained_policy

    policy_model = load_trained_policy(str(model_path))
    policy_obs_adapter = resolve_policy_obs_adapter(
        policy_model,
        fallback_adapter=_defensive_obs_adapter,
    )
    return policy_model, policy_obs_adapter


def _resolve_socnav_policy(
    args,
    *,
    ctx: TrainingContext,
    policy_name: str,
) -> SocNavPlannerPolicy | None:
    """Resolve SocNav policy if selected."""
    return _build_socnav_policy(
        policy_name,
        socnav_root=args.socnav_root,
        orca_time_horizon=ctx.orca_time_horizon,
        orca_neighbor_dist=ctx.orca_neighbor_dist,
        socnav_allow_fallback=args.socnav_allow_fallback,
    )


def _prepare_outputs(
    args,
    *,
    policy_name: str,
    stamp: str | None = None,
    output_base: Path | None = None,
    video_base: Path | None = None,
) -> tuple[Path, Path | None, Path, dict[str, Any]]:
    """Prepare output directories and schema."""
    output_root = _resolve_output_root(
        output_base if output_base is not None else args.output,
        policy=policy_name,
        stamp=stamp,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    video_root = (
        _resolve_video_root(
            video_base if video_base is not None else args.video_output,
            policy=policy_name,
            stamp=stamp,
        )
        if args.videos
        else None
    )
    if video_root is not None:
        video_root.mkdir(parents=True, exist_ok=True)
    out_jsonl = output_root / "episodes.jsonl"
    repo_root = Path(__file__).resolve().parents[2]
    schema = load_schema(repo_root / "robot_sf/benchmark/schemas/episode.schema.v1.json")
    return output_root, video_root, out_jsonl, schema


def _run_frame_extraction(report_json: Path, *, output_root: Path | None) -> None:
    """Invoke extract_failure_frames.py for a policy analysis report."""
    if not report_json.exists():
        logger.warning("Skipping frame extraction; report not found: {}", report_json)
        return
    script = Path(__file__).resolve().parent / "extract_failure_frames.py"
    cmd = [sys.executable, str(script), "--report", str(report_json)]
    if output_root is not None:
        cmd.extend(["--output", str(output_root)])
    timeout_sec = 60
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        extra = "\n".join(val for val in (stderr, stdout) if val)
        message = (
            f"Frame extraction timed out after {timeout_sec}s for {report_json}."
            if not extra
            else f"Frame extraction timed out after {timeout_sec}s for {report_json}:\n{extra}"
        )
        logger.warning("{}", message)
        return
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        logger.warning("Frame extraction failed for {}: {}", report_json, msg)
        return
    if result.stdout.strip():
        logger.info("Frame extraction output:\n{}", result.stdout.strip())


def _resolve_seed_set(
    args: argparse.Namespace,
    *,
    repo_root: Path,
) -> tuple[str | None, list[int] | None]:
    """Resolve the selected seed set, if any.

    Returns:
        tuple[str | None, list[int] | None]: Seed set name and seed list, if configured.
    """
    seed_sets = _load_seed_sets(repo_root / _DEFAULT_SEED_SET_PATH)
    seed_set_name = args.seed_set
    if seed_set_name is None:
        return None, None
    seed_set = seed_sets.get(seed_set_name)
    if not seed_set:
        raise ValueError(f"Seed set '{seed_set_name}' not found in {_DEFAULT_SEED_SET_PATH}.")
    logger.info("Using seed set '{}' with {} seeds.", seed_set_name, len(seed_set))
    return seed_set_name, seed_set


def _resolve_policies(args: argparse.Namespace) -> list[str]:
    """Resolve which policies to evaluate.

    Returns:
        list[str]: Policy identifiers to evaluate.
    """
    if not args.policy_sweep:
        return [args.policy]
    if args.policies:
        items = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
        invalid = [policy for policy in items if policy not in _SUPPORTED_POLICIES]
        if invalid:
            allowed = ", ".join(sorted(_SUPPORTED_POLICIES))
            raise ValueError(
                f"Invalid policies in --policies: {', '.join(invalid)}. Allowed: {allowed}."
            )
        return items
    return ["fast_pysf_planner", "ppo", "socnav_orca"]


def _prepare_sweep_roots(
    args: argparse.Namespace,
    *,
    sweep_stamp: str | None,
) -> tuple[Path | None, Path | None]:
    """Create sweep output roots when running multiple policies.

    Returns:
        tuple[Path | None, Path | None]: Sweep output and video roots.
    """
    if not args.policy_sweep:
        return None, None
    sweep_root = _resolve_sweep_root(args.output, stamp=sweep_stamp)
    sweep_root.mkdir(parents=True, exist_ok=True)
    sweep_video_root: Path | None = None
    if args.videos:
        sweep_video_root = _resolve_sweep_video_root(args.video_output, stamp=sweep_stamp)
        sweep_video_root.mkdir(parents=True, exist_ok=True)
    return sweep_root, sweep_video_root


def _resolve_frame_root(
    args: argparse.Namespace,
    *,
    policy_name: str,
    output_root: Path,
) -> Path:
    """Resolve the root directory for extracted failure frames.

    Returns:
        Path: Base directory for extracted frames.
    """
    frame_root = args.extract_frames_output
    if frame_root is not None:
        frame_root = resolve_artifact_path(frame_root)
        if args.policy_sweep:
            frame_root = frame_root / policy_name
    else:
        frame_root = output_root / "failure_frames"
    return frame_root


def _maybe_extract_frames(
    args: argparse.Namespace,
    *,
    result: dict[str, Any],
    policy_name: str,
    output_root: Path,
) -> None:
    """Optionally extract failure frames after a policy run."""
    if not args.extract_frames:
        return
    if not args.videos:
        logger.warning("Skipping frame extraction; --videos is not set.")
        return
    frame_root = _resolve_frame_root(args, policy_name=policy_name, output_root=output_root)
    _run_frame_extraction(Path(result["report_json"]), output_root=frame_root)


def _run_policy_episodes(
    scenarios: list[Mapping[str, Any]],
    *,
    scenario_path: Path,
    args: argparse.Namespace,
    ctx: TrainingContext,
    policy_name: str,
    policy_model: Any,
    socnav_policy: SocNavPlannerPolicy | None,
    policy_obs_adapter: Callable[[Mapping[str, Any]], np.ndarray] | None,
    out_jsonl: Path,
    schema: dict[str, Any],
    video_root: Path | None,
    seed_set: list[int] | None,
) -> list[dict[str, Any]]:
    """Run all scenarios for a single policy and collect records.

    Returns:
        list[dict[str, Any]]: Episode record dictionaries.
    """
    for scenario in scenarios:
        scenario_name = str(scenario.get("name") or scenario.get("scenario_id") or "scenario")
        seeds = _resolve_seeds(
            scenario,
            seed_override=args.seed,
            max_seeds=args.max_seeds,
            training_seeds=ctx.training_seeds,
            seed_set=seed_set,
        )
        logger.info("Scenario {} seeds={}", scenario_name, seeds)
        for seed in seeds:
            record, _artifacts = _run_episode(
                scenario,
                scenario_path=scenario_path,
                seed=seed,
                policy_name=policy_name,
                policy_model=policy_model,
                socnav_policy=socnav_policy,
                policy_obs_adapter=policy_obs_adapter,
                robot_speed=args.robot_speed,
                env_overrides=ctx.env_overrides,
                env_factory_kwargs=ctx.env_factory_kwargs,
                record_forces=args.record_forces,
                max_steps_override=args.max_steps,
                videos=args.videos,
                video_dir=video_root,
                video_fps=args.video_fps,
                socnav_use_grid=args.socnav_use_grid,
                socnav_grid_resolution=args.socnav_grid_resolution,
                socnav_grid_width=args.socnav_grid_width,
                socnav_grid_height=args.socnav_grid_height,
                socnav_grid_center_on_robot=args.socnav_grid_center_on_robot,
            )
            _write_jsonl(out_jsonl, schema, record)
    return read_jsonl(out_jsonl)


def _run_policy_analysis_for_policy(
    policy_name: str,
    *,
    args: argparse.Namespace,
    ctx: TrainingContext,
    scenario_path: Path,
    scenarios: list[Mapping[str, Any]],
    seed_set: list[int] | None,
    seed_set_name: str | None,
    sweep_stamp: str | None,
    sweep_root: Path | None,
    sweep_video_root: Path | None,
) -> dict[str, Any]:
    """Run analysis for a single policy and return its summary entry.

    Returns:
        dict[str, Any]: Summary metadata for the policy run.
    """
    output_base = sweep_root / policy_name if sweep_root is not None else args.output
    video_base = (
        sweep_video_root / policy_name if sweep_video_root is not None else args.video_output
    )
    policy_model, policy_obs_adapter = _load_policy_model(
        policy_name,
        model_path=args.model_path,
    )
    socnav_policy = _resolve_socnav_policy(args, ctx=ctx, policy_name=policy_name)
    output_root, video_root, out_jsonl, schema = _prepare_outputs(
        args,
        policy_name=policy_name,
        stamp=sweep_stamp,
        output_base=output_base,
        video_base=video_base,
    )
    records = _run_policy_episodes(
        scenarios,
        scenario_path=scenario_path,
        args=args,
        ctx=ctx,
        policy_name=policy_name,
        policy_model=policy_model,
        socnav_policy=socnav_policy,
        policy_obs_adapter=policy_obs_adapter,
        out_jsonl=out_jsonl,
        schema=schema,
        video_root=video_root,
        seed_set=seed_set,
    )
    result = _finalize_run(
        output_root,
        scenario_path=scenario_path,
        records=records,
        policy_name=policy_name,
        video_root=video_root,
        write_plausibility_metrics=args.write_plausibility_metrics,
        seed_set_name=seed_set_name,
    )
    _maybe_extract_frames(
        args,
        result=result,
        policy_name=policy_name,
        output_root=output_root,
    )
    return {
        "policy": policy_name,
        "category": _POLICY_CATEGORIES.get(policy_name, "unknown"),
        "summary": result["summary"],
        "metric_means": result["summary"].get("metric_means", {}),
        "report_md": result["report_md"],
        "report_json": result["report_json"],
        "summary_json": result["summary_json"],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for policy analysis runs."""
    args = _build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    ensure_canonical_tree()

    repo_root = Path(__file__).resolve().parents[2]
    seed_set_name, seed_set = _resolve_seed_set(args, repo_root=repo_root)

    ctx = _load_training_context(args)
    scenario_path, scenarios = _resolve_scenarios(args, training_config=ctx.training_config)
    policies = _resolve_policies(args)

    sweep_stamp = (
        datetime.now(_TIMESTAMP_TZ).strftime("%Y%m%d_%H%M%S") if args.policy_sweep else None
    )
    sweep_root, sweep_video_root = _prepare_sweep_roots(args, sweep_stamp=sweep_stamp)

    policy_entries: list[dict[str, Any]] = []
    for policy_name in policies:
        policy_entries.append(
            _run_policy_analysis_for_policy(
                policy_name,
                args=args,
                ctx=ctx,
                scenario_path=scenario_path,
                scenarios=scenarios,
                seed_set=seed_set,
                seed_set_name=seed_set_name,
                sweep_stamp=sweep_stamp,
                sweep_root=sweep_root,
                sweep_video_root=sweep_video_root,
            )
        )
    if args.policy_sweep and sweep_root is not None:
        _write_combined_report(
            sweep_root,
            policy_entries=policy_entries,
            scenario_path=scenario_path,
            seed_set_name=seed_set_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
