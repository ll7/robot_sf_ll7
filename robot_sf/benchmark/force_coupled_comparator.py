"""Deterministic diagnostic comparison harness for force-coupled potential-field local planning.

Evaluates the opt-in ``force_coupled_potential_field`` local planner (issue #7889)
against reference local planners on canonical analytic scenarios. Outputs a typed,
versioned diagnostic receipt (``force_coupled_comparator_receipt.v1``).

Evidence boundary: diagnostic comparator only; does not establish benchmark ranking,
general human predictability, social compliance, or safety certification.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.common.math_utils import wrap_angle_pi
from robot_sf.planner.force_coupled_potential_field import (
    ForceCoupledPotentialFieldConfig,
    ForceCoupledPotentialFieldPlanner,
    build_force_coupled_potential_field_config,
)
from robot_sf.planner.protocol import LocalPlannerProtocol

SCHEMA_VERSION = "force_coupled_comparator_receipt.v1"
CANONICAL_CONFIG_PATH = "configs/algos/issue_7889_force_coupled_potential_field.yaml"
CLAIM_BOUNDARY = (
    "diagnostic_comparator_only; implementation-integrity evidence comparing "
    "force_coupled_potential_field to reference baselines on analytic fixtures; "
    "no ranking, social compliance, or paper-grade claim is established."
)


@dataclass(frozen=True)
class ComparatorScenarioSpec:
    """Specification of a deterministic analytic comparison scenario."""

    scenario_id: str
    seed: int
    robot_start: tuple[float, float, float]
    goal: tuple[float, float]
    obstacles: tuple[tuple[float, float], ...] = ()
    pedestrians: tuple[tuple[float, float], ...] = ()
    max_steps: int = 60
    control_dt: float = 0.2
    goal_tolerance: float = 0.25
    robot_radius: float = 0.25
    near_miss_radius: float = 0.40

    def to_dict(self) -> dict[str, Any]:
        """Serialize scenario spec for receipt output.

        Returns:
            Dictionary representation of the scenario specification.
        """
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "robot_start": list(self.robot_start),
            "goal": list(self.goal),
            "obstacles": [list(pt) for pt in self.obstacles],
            "pedestrians": [list(pt) for pt in self.pedestrians],
        }


@dataclass(frozen=True)
class ComparatorRunResult:
    """Outcome of one planner rollout in a comparison scenario."""

    planner_id: str
    scenario_id: str
    seed: int
    steps: int
    completed: bool
    collision: bool
    near_miss: bool
    min_clearance_obstacle_m: float | None
    min_clearance_pedestrian_m: float | None
    path_length_m: float
    mean_linear_speed_mps: float
    max_linear_speed_mps: float
    mean_angular_rate_radps: float
    max_angular_rate_radps: float
    jerk_metric: float
    mean_latency_ms: float
    status: str
    degraded: bool
    degradation_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result for receipt output.

        Returns:
            Dictionary representation of the rollout result.
        """
        return {
            "planner_id": self.planner_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "steps": self.steps,
            "completed": self.completed,
            "collision": self.collision,
            "near_miss": self.near_miss,
            "min_clearance_obstacle_m": (
                round(self.min_clearance_obstacle_m, 4)
                if self.min_clearance_obstacle_m is not None
                else None
            ),
            "min_clearance_pedestrian_m": (
                round(self.min_clearance_pedestrian_m, 4)
                if self.min_clearance_pedestrian_m is not None
                else None
            ),
            "path_length_m": round(self.path_length_m, 4),
            "mean_linear_speed_mps": round(self.mean_linear_speed_mps, 4),
            "max_linear_speed_mps": round(self.max_linear_speed_mps, 4),
            "mean_angular_rate_radps": round(self.mean_angular_rate_radps, 4),
            "max_angular_rate_radps": round(self.max_angular_rate_radps, 4),
            "jerk_metric": round(self.jerk_metric, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 4),
            "status": self.status,
            "degraded": self.degraded,
            "degradation_reasons": list(self.degradation_reasons),
        }


class PurePursuitGoalPlanner(LocalPlannerProtocol):
    """Deterministic reference baseline: pure pursuit straight to goal."""

    def __init__(
        self,
        *,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 1.2,
        gain: float = 0.8,
    ) -> None:
        """Initialize the pure-pursuit reference planner.

        Args:
            max_linear_speed: Maximum forward linear speed clip.
            max_angular_speed: Maximum angular velocity clip.
            gain: Distance-to-speed gain factor.
        """
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.gain = gain
        self._closed = False
        self._last_diagnostics: dict[str, Any] = {}

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Compute direct pure-pursuit velocity towards goal.

        Args:
            observation: Standard observation mapping containing robot pose and goal.

        Returns:
            Tuple of (linear_speed, angular_rate).
        """
        if self._closed:
            raise ValueError("planner is closed")
        robot = np.asarray(observation["robot"], dtype=float).reshape(-1)
        goal = np.asarray(observation["goal"], dtype=float).reshape(-1)
        dx = goal[0] - robot[0]
        dy = goal[1] - robot[1]
        dist = float(math.hypot(dx, dy))
        if dist <= 1e-6:
            cmd = (0.0, 0.0)
        else:
            desired_heading = math.atan2(dy, dx)
            heading_err = wrap_angle_pi(desired_heading - robot[2])
            linear = float(np.clip(self.gain * dist, 0.0, self.max_linear_speed))
            angular = float(np.clip(heading_err, -self.max_angular_speed, self.max_angular_speed))
            cmd = (linear, angular)
        self._last_diagnostics = {
            "planner_type": "pure_pursuit_goal",
            "status": "ok",
            "degraded": False,
        }
        return cmd

    def reset(self, *, seed: int | None = None) -> None:
        """Reset internal planner state.

        Args:
            seed: Deterministic integer seed for replay.
        """
        self._last_diagnostics = {}

    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostic dictionary.

        Returns:
            Dictionary containing planner status and telemetry.
        """
        return dict(self._last_diagnostics) or {"planner_type": "pure_pursuit_goal", "status": "ok"}

    def close(self) -> None:
        """Close planner resources."""
        self._closed = True


def get_canonical_comparison_scenarios() -> list[ComparatorScenarioSpec]:
    """Return the canonical deterministic analytic scenarios for local planner comparison.

    Returns:
        List of configured comparison scenario specifications.
    """
    return [
        ComparatorScenarioSpec(
            scenario_id="analytic_static_obstacle",
            seed=1,
            robot_start=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            obstacles=((1.0, 0.5),),
            pedestrians=(),
            max_steps=60,
        ),
        ComparatorScenarioSpec(
            scenario_id="analytic_pedestrian_interaction",
            seed=7,
            robot_start=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            obstacles=(),
            pedestrians=((1.0, 0.0),),
            max_steps=60,
        ),
        ComparatorScenarioSpec(
            scenario_id="analytic_symmetric_obstacle",
            seed=42,
            robot_start=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            obstacles=((2.0, 0.0),),
            pedestrians=(),
            max_steps=60,
        ),
        ComparatorScenarioSpec(
            scenario_id="analytic_unobstructed",
            seed=1,
            robot_start=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            obstacles=(),
            pedestrians=(),
            max_steps=60,
        ),
    ]


def build_planner_registry(
    config: ForceCoupledPotentialFieldConfig | None = None,
) -> dict[str, LocalPlannerProtocol]:
    """Instantiate the standard comparator planner set.

    Args:
        config: Optional base configuration for force-coupled potential field.

    Returns:
        Dictionary mapping planner IDs to protocol-compliant planner instances.
    """
    cfg = config or ForceCoupledPotentialFieldConfig()
    attractive_only_cfg = ForceCoupledPotentialFieldConfig(
        attractive_weight=cfg.attractive_weight,
        repulsive_weight=0.001,
        influence_radius_m=cfg.influence_radius_m,
    )
    repulsive_only_cfg = ForceCoupledPotentialFieldConfig(
        attractive_weight=0.001,
        repulsive_weight=cfg.repulsive_weight,
        influence_radius_m=cfg.influence_radius_m,
    )
    return {
        "force_coupled_potential_field": ForceCoupledPotentialFieldPlanner(
            cfg, planner_type="force_coupled_potential_field"
        ),
        "pure_pursuit_goal": PurePursuitGoalPlanner(),
        "ablation_attractive_dominant": ForceCoupledPotentialFieldPlanner(
            attractive_only_cfg, planner_type="ablation_attractive_dominant"
        ),
        "ablation_repulsive_dominant": ForceCoupledPotentialFieldPlanner(
            repulsive_only_cfg, planner_type="ablation_repulsive_dominant"
        ),
    }


def _update_clearance(
    points: tuple[tuple[float, float], ...],
    rx: float,
    ry: float,
    current_min: float | None,
    robot_radius: float,
    near_miss_radius: float,
) -> tuple[float | None, bool, bool]:
    """Check clearance to a set of entities and update collision/near-miss flags.

    Args:
        points: Entities to check distance against.
        rx: Current robot x-coordinate.
        ry: Current robot y-coordinate.
        current_min: Existing minimum clearance value.
        robot_radius: Collision threshold radius.
        near_miss_radius: Near-miss threshold radius.

    Returns:
        Tuple of (updated_min_clearance, collision_flag, near_miss_flag).
    """
    new_min = current_min
    collision = False
    near_miss = False
    for px, py in points:
        dist = math.hypot(px - rx, py - ry)
        if new_min is None or dist < new_min:
            new_min = dist
        if dist <= robot_radius:
            collision = True
        elif dist <= near_miss_radius:
            near_miss = True
    return new_min, collision, near_miss


def execute_rollout(  # noqa: C901, PLR0915
    planner: LocalPlannerProtocol,
    scenario: ComparatorScenarioSpec,
) -> ComparatorRunResult:
    """Execute one deterministic kinematic rollout in an analytic scenario.

    Args:
        planner: Local planner instance to evaluate.
        scenario: Scenario specification with start, goal, obstacles, and limits.

    Returns:
        Structured rollout outcome containing trajectory metrics and collision status.
    """
    planner.reset(seed=scenario.seed)
    rx, ry, rtheta = scenario.robot_start
    gx, gy = scenario.goal
    dt = scenario.control_dt

    path_length = 0.0
    linear_speeds: list[float] = []
    angular_rates: list[float] = []
    latencies_ms: list[float] = []
    min_obs_dist: float | None = None
    min_ped_dist: float | None = None
    collision = False
    near_miss = False
    completed = False
    degraded = False
    degradation_reasons: list[str] = []
    status = "ok"

    last_v = 0.0
    last_a = 0.0
    jerk_accum = 0.0

    step = 0
    while step < scenario.max_steps:
        dist_to_goal = math.hypot(gx - rx, gy - ry)
        if dist_to_goal <= scenario.goal_tolerance:
            completed = True
            break

        if scenario.obstacles:
            min_obs_dist, col_obs, nm_obs = _update_clearance(
                scenario.obstacles,
                rx,
                ry,
                min_obs_dist,
                scenario.robot_radius,
                scenario.near_miss_radius,
            )
            collision = collision or col_obs
            near_miss = near_miss or nm_obs

        if scenario.pedestrians:
            min_ped_dist, col_ped, nm_ped = _update_clearance(
                scenario.pedestrians,
                rx,
                ry,
                min_ped_dist,
                scenario.robot_radius,
                scenario.near_miss_radius,
            )
            collision = collision or col_ped
            near_miss = near_miss or nm_ped

        obs = {
            "robot": [rx, ry, rtheta],
            "goal": [gx, gy],
            "obstacles": {"positions": [list(p) for p in scenario.obstacles]},
            "pedestrians": {
                "positions": [list(p) for p in scenario.pedestrians],
                "count": [len(scenario.pedestrians)],
            },
            "sim": {"timestep": dt},
        }

        t0 = time.perf_counter()
        try:
            linear_cmd, angular_cmd = planner.plan(obs)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            degraded = True
            degradation_reasons.append(f"plan_exception: {exc}")
            break

        diag = planner.diagnostics()
        if diag.get("status") == "degraded":
            degraded = True
            for reason in diag.get("degradation_reasons", []):
                if reason not in degradation_reasons:
                    degradation_reasons.append(str(reason))

        linear_speeds.append(float(linear_cmd))
        angular_rates.append(float(angular_cmd))

        accel = (linear_cmd - last_v) / dt
        if step > 0:
            jerk = (accel - last_a) / dt
            jerk_accum += jerk * jerk
        last_v = linear_cmd
        last_a = accel

        dx = linear_cmd * math.cos(rtheta) * dt
        dy = linear_cmd * math.sin(rtheta) * dt
        rx += dx
        ry += dy
        rtheta = wrap_angle_pi(rtheta + angular_cmd * dt)
        path_length += math.hypot(dx, dy)
        step += 1

    mean_linear = float(np.mean(linear_speeds)) if linear_speeds else 0.0
    max_linear = float(np.max(np.abs(linear_speeds))) if linear_speeds else 0.0
    mean_angular = float(np.mean(np.abs(angular_rates))) if angular_rates else 0.0
    max_angular = float(np.max(np.abs(angular_rates))) if angular_rates else 0.0
    mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    if status == "ok" and degraded:
        status = "degraded"

    return ComparatorRunResult(
        planner_id=diag.get("planner_type", type(planner).__name__),
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        steps=step,
        completed=completed,
        collision=collision,
        near_miss=near_miss,
        min_clearance_obstacle_m=min_obs_dist,
        min_clearance_pedestrian_m=min_ped_dist,
        path_length_m=path_length,
        mean_linear_speed_mps=mean_linear,
        max_linear_speed_mps=max_linear,
        mean_angular_rate_radps=mean_angular,
        max_angular_rate_radps=max_angular,
        jerk_metric=float(math.sqrt(jerk_accum / max(1, step))),
        mean_latency_ms=mean_latency,
        status=status,
        degraded=degraded,
        degradation_reasons=tuple(degradation_reasons),
    )


def compute_summary_table(results: list[ComparatorRunResult]) -> list[dict[str, Any]]:
    """Compute per-planner aggregated summary statistics across runs.

    Args:
        results: List of individual scenario rollout results.

    Returns:
        List of summary dictionaries aggregated by planner ID.
    """
    by_planner: dict[str, list[ComparatorRunResult]] = {}
    for res in results:
        by_planner.setdefault(res.planner_id, []).append(res)

    summary: list[dict[str, Any]] = []
    for pid in sorted(by_planner.keys()):
        runs = by_planner[pid]
        n = len(runs)
        successes = sum(1 for r in runs if r.completed and not r.collision)
        collisions = sum(1 for r in runs if r.collision)
        near_misses = sum(1 for r in runs if r.near_miss and not r.collision)
        mean_path = float(np.mean([r.path_length_m for r in runs]))
        mean_jerk = float(np.mean([r.jerk_metric for r in runs]))
        mean_lat = float(np.mean([r.mean_latency_ms for r in runs]))

        status_counts: dict[str, int] = {}
        for r in runs:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1

        summary.append(
            {
                "planner_id": pid,
                "runs": n,
                "success_rate": round(successes / n, 4),
                "collision_rate": round(collisions / n, 4),
                "near_miss_rate": round(near_misses / n, 4),
                "mean_path_length_m": round(mean_path, 4),
                "mean_jerk_metric": round(mean_jerk, 4),
                "mean_latency_ms": round(mean_lat, 4),
                "status_counts": status_counts,
            }
        )
    return summary


def run_force_coupled_comparator(
    *,
    config_path: Path | None = None,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Execute the canonical comparator suite and return the full receipt dictionary.

    Args:
        config_path: Optional explicit path to configuration YAML.
        repo_root: Root directory of repository.

    Returns:
        Full dictionary representation conforming to force_coupled_comparator_receipt.v1.
    """
    repo_root = repo_root.resolve()
    target_config = config_path or repo_root / CANONICAL_CONFIG_PATH
    if target_config.exists():
        raw_bytes = target_config.read_bytes()
        config_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        parsed_yaml = yaml.safe_load(raw_bytes.decode("utf-8"))
        planner_cfg = build_force_coupled_potential_field_config(parsed_yaml)
    else:
        planner_cfg = ForceCoupledPotentialFieldConfig()
        config_sha256 = hashlib.sha256(b"").hexdigest()

    config_digest = planner_cfg.digest()

    scenarios = get_canonical_comparison_scenarios()
    planners = build_planner_registry(planner_cfg)

    all_results: list[ComparatorRunResult] = []
    for scenario in scenarios:
        for planner in planners.values():
            result = execute_rollout(planner, scenario)
            all_results.append(result)

    summary_table = compute_summary_table(all_results)

    env_info = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }

    # Deterministic receipt body for digest computation (excluding wall-clock latency)
    digest_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "claim_boundary": CLAIM_BOUNDARY,
        "config_digest": config_digest,
        "config_sha256": config_sha256,
        "scenarios": [s.to_dict() for s in scenarios],
        "results": [
            {k: v for k, v in r.to_dict().items() if k != "mean_latency_ms"} for r in all_results
        ],
        "summary_table": [
            {k: v for k, v in row.items() if k != "mean_latency_ms"} for row in summary_table
        ],
        "environment": env_info,
    }
    canonical_bytes = json.dumps(digest_payload, sort_keys=True).encode("utf-8")
    receipt_digest = hashlib.sha256(canonical_bytes).hexdigest()

    # Full receipt
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_digest": receipt_digest,
        "status": "ok",
        "claim_boundary": CLAIM_BOUNDARY,
        "config_digest": config_digest,
        "config_sha256": config_sha256,
        "scenarios": [s.to_dict() for s in scenarios],
        "results": [r.to_dict() for r in all_results],
        "summary_table": summary_table,
        "environment": env_info,
    }
    return receipt
