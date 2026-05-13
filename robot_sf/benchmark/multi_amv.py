"""Minimal multi-AMV benchmark helpers.

This module provides the first narrow multi-robot benchmark slice: scenario
settings parsing and inter-robot metric computation. It intentionally avoids a
fleet-optimization abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MultiAmvSettings:
    """Scenario-level settings for the minimal multi-AMV benchmark slice."""

    num_robots: int = 1
    near_miss_distance_m: float = 1.0
    collision_distance_m: float = 0.4
    deadlock_speed_mps: float = 0.05
    deadlock_window_steps: int = 10


class MultiAmvPlannerSupportStatus(StrEnum):
    """Support status for running a planner family in multi-AMV scenarios."""

    NATIVE = "native"
    ADAPTER = "adapter"
    NOT_AVAILABLE = "not_available"
    RESEARCH_ONLY = "research_only"


@dataclass(frozen=True)
class MultiAmvPlannerSupport:
    """Planner-family support classification for multi-AMV execution."""

    planner_family: str
    support_status: MultiAmvPlannerSupportStatus
    contract_kind: str
    action_shape: str
    robot_identity: str
    collision_responsibility: str
    metadata_reporting: str
    rationale: str

    def to_json_dict(self) -> dict[str, str]:
        """Return JSON-compatible support metadata.

        Returns
        -------
        dict[str, str]
            Planner support classification and minimum contract fields.
        """
        return {
            "planner_family": self.planner_family,
            "support_status": self.support_status.value,
            "contract_kind": self.contract_kind,
            "action_shape": self.action_shape,
            "robot_identity": self.robot_identity,
            "collision_responsibility": self.collision_responsibility,
            "metadata_reporting": self.metadata_reporting,
            "rationale": self.rationale,
        }


_MULTI_AMV_PLANNER_SUPPORT: dict[str, MultiAmvPlannerSupport] = {
    "goal_controller_smoke": MultiAmvPlannerSupport(
        planner_family="goal_controller_smoke",
        support_status=MultiAmvPlannerSupportStatus.NATIVE,
        contract_kind="goal_controller_smoke",
        action_shape="array[num_robots, 2] unicycle velocity commands",
        robot_identity="implicit row order from MultiRobotEnv simulators",
        collision_responsibility="smoke runner only; not a coordinated fleet planner",
        metadata_reporting="planner_support block plus inter-robot metrics",
        rationale=(
            "Supported only as the minimal smoke controller used to exercise multi-AMV "
            "episode records; it is not benchmark-comparable planner-family support."
        ),
    ),
    "goal": MultiAmvPlannerSupport(
        planner_family="goal",
        support_status=MultiAmvPlannerSupportStatus.NOT_AVAILABLE,
        contract_kind="single_robot_planner",
        action_shape="single robot action; no fleet action tensor",
        robot_identity="missing",
        collision_responsibility="missing inter-robot coordination contract",
        metadata_reporting="not available",
        rationale="The single-robot goal planner does not define multi-robot identity or actions.",
    ),
    "social_force": MultiAmvPlannerSupport(
        planner_family="social_force",
        support_status=MultiAmvPlannerSupportStatus.RESEARCH_ONLY,
        contract_kind="pedestrian_dynamics_or_single_robot_baseline",
        action_shape="not defined for coordinated robot fleet control",
        robot_identity="missing",
        collision_responsibility="research question; no benchmark contract yet",
        metadata_reporting="not available",
        rationale="Social-force behavior can inform research, but no multi-AMV planner adapter exists.",
    ),
    "orca": MultiAmvPlannerSupport(
        planner_family="orca",
        support_status=MultiAmvPlannerSupportStatus.NOT_AVAILABLE,
        contract_kind="single_robot_or_pairwise_adapter_missing",
        action_shape="not defined for all robots with stable robot ids",
        robot_identity="missing",
        collision_responsibility="missing fleet-level responsibility and metadata contract",
        metadata_reporting="not available",
        rationale=(
            "ORCA is the first plausible non-trivial candidate, but it needs an explicit "
            "multi-robot adapter before benchmark use."
        ),
    ),
    "ppo": MultiAmvPlannerSupport(
        planner_family="ppo",
        support_status=MultiAmvPlannerSupportStatus.NOT_AVAILABLE,
        contract_kind="single_robot_policy",
        action_shape="single policy action; no multi-robot observation/action schema",
        robot_identity="missing",
        collision_responsibility="missing learned fleet-control contract",
        metadata_reporting="not available",
        rationale="Existing PPO checkpoints are trained for single-robot environment contracts.",
    ),
    "guarded_ppo": MultiAmvPlannerSupport(
        planner_family="guarded_ppo",
        support_status=MultiAmvPlannerSupportStatus.NOT_AVAILABLE,
        contract_kind="single_robot_policy_with_guard",
        action_shape="single policy action; no multi-robot observation/action schema",
        robot_identity="missing",
        collision_responsibility="missing learned fleet-control contract",
        metadata_reporting="not available",
        rationale="Guarded PPO inherits the single-robot PPO contract and is not fleet-aware.",
    ),
    "sacadrl": MultiAmvPlannerSupport(
        planner_family="sacadrl",
        support_status=MultiAmvPlannerSupportStatus.NOT_AVAILABLE,
        contract_kind="single_robot_policy",
        action_shape="single robot action",
        robot_identity="missing",
        collision_responsibility="missing fleet-control contract",
        metadata_reporting="not available",
        rationale="SA-CADRL support is single-robot and has no multi-AMV adapter.",
    ),
    "teb": MultiAmvPlannerSupport(
        planner_family="teb",
        support_status=MultiAmvPlannerSupportStatus.RESEARCH_ONLY,
        contract_kind="testing_only_single_robot_planner",
        action_shape="single robot trajectory command",
        robot_identity="missing",
        collision_responsibility="research-only until a fleet adapter exists",
        metadata_reporting="not available",
        rationale="TEB is testing-only in this repo and not a coordinated multi-AMV planner.",
    ),
}


def multi_amv_planner_support_inventory() -> dict[str, dict[str, str]]:
    """Return planner-family multi-AMV support inventory.

    Returns
    -------
    dict[str, dict[str, str]]
        JSON-compatible support records keyed by planner family.
    """
    return {
        planner_family: support.to_json_dict()
        for planner_family, support in sorted(_MULTI_AMV_PLANNER_SUPPORT.items())
    }


def multi_amv_planner_support(planner_family: str) -> MultiAmvPlannerSupport:
    """Return the multi-AMV support classification for a planner family.

    Returns
    -------
    MultiAmvPlannerSupport
        Support classification for the requested planner family.
    """
    normalized = str(planner_family).strip().lower().replace("-", "_")
    try:
        return _MULTI_AMV_PLANNER_SUPPORT[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(_MULTI_AMV_PLANNER_SUPPORT))
        raise ValueError(f"unknown multi-AMV planner family {planner_family!r}; known: {known}") from exc


def ensure_multi_amv_planner_supported(
    planner_family: str,
    *,
    require_non_smoke: bool = False,
) -> MultiAmvPlannerSupport:
    """Fail closed when a planner family lacks a multi-AMV execution contract.

    Returns
    -------
    MultiAmvPlannerSupport
        Support classification for allowed planner families.
    """
    support = multi_amv_planner_support(planner_family)
    if support.support_status not in {
        MultiAmvPlannerSupportStatus.NATIVE,
        MultiAmvPlannerSupportStatus.ADAPTER,
    }:
        raise ValueError(
            f"planner family {support.planner_family!r} is {support.support_status.value} "
            f"for multi-AMV scenarios: {support.rationale}"
        )
    if require_non_smoke and support.contract_kind == "goal_controller_smoke":
        raise ValueError(
            "multi-AMV goal_controller_smoke is a smoke controller, not non-trivial "
            "planner-family support"
        )
    return support


def multi_amv_settings_from_scenario(scenario: dict[str, Any]) -> MultiAmvSettings:
    """Parse the optional ``multi_amv`` scenario block.

    Returns:
        MultiAmvSettings: Validated settings for the minimal multi-AMV slice.
    """
    raw = scenario.get("multi_amv")
    if raw is None:
        return MultiAmvSettings()
    if not isinstance(raw, dict):
        raise ValueError("multi_amv must be a mapping.")
    allowed = {
        "num_robots",
        "near_miss_distance_m",
        "collision_distance_m",
        "deadlock_speed_mps",
        "deadlock_window_steps",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"multi_amv contains unknown keys: {', '.join(unknown)}.")
    settings = MultiAmvSettings(
        num_robots=int(raw.get("num_robots", 1)),
        near_miss_distance_m=float(raw.get("near_miss_distance_m", 1.0)),
        collision_distance_m=float(raw.get("collision_distance_m", 0.4)),
        deadlock_speed_mps=float(raw.get("deadlock_speed_mps", 0.05)),
        deadlock_window_steps=int(raw.get("deadlock_window_steps", 10)),
    )
    if settings.num_robots < 1:
        raise ValueError("multi_amv.num_robots must be >= 1.")
    if settings.collision_distance_m <= 0.0:
        raise ValueError("multi_amv.collision_distance_m must be > 0.")
    if settings.near_miss_distance_m <= settings.collision_distance_m:
        raise ValueError("multi_amv.near_miss_distance_m must be > collision_distance_m.")
    if settings.deadlock_speed_mps < 0.0:
        raise ValueError("multi_amv.deadlock_speed_mps must be >= 0.")
    if settings.deadlock_window_steps < 1:
        raise ValueError("multi_amv.deadlock_window_steps must be >= 1.")
    return settings


def inter_robot_metrics(
    robot_positions: np.ndarray,
    *,
    dt: float,
    settings: MultiAmvSettings,
) -> dict[str, float | bool]:
    """Compute minimal inter-robot safety/deadlock metrics from trajectories.

    Args:
        robot_positions: Array shaped ``(steps, robots, 2)``.
        dt: Simulation step duration in seconds.
        settings: Multi-AMV metric thresholds.

    Returns:
        dict[str, float | bool]: JSON-safe inter-robot metrics where collision/near-miss
        events count contiguous encounter runs, and deadlock detection is fleet-wide for this
        first slice.
    """
    positions = np.asarray(robot_positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("robot_positions must have shape (steps, robots, 2).")
    steps, robots, _ = positions.shape
    pair_count = (robots * (robots - 1)) // 2
    if robots < 2:
        return {
            "robot_count": float(robots),
            "pair_count": 0.0,
            "min_inter_robot_distance_m": float("nan"),
            "inter_robot_collision_events": 0.0,
            "inter_robot_near_miss_events": 0.0,
            "deadlock_steps": 0.0,
            "deadlock_detected": False,
        }
    if steps == 0:
        return {
            "robot_count": float(robots),
            "pair_count": float(pair_count),
            "min_inter_robot_distance_m": float("nan"),
            "inter_robot_collision_events": 0.0,
            "inter_robot_near_miss_events": 0.0,
            "deadlock_steps": 0.0,
            "deadlock_detected": False,
        }

    pair_distances = []
    for i in range(robots):
        for j in range(i + 1, robots):
            pair_distances.append(np.linalg.norm(positions[:, i, :] - positions[:, j, :], axis=1))
    distances = np.stack(pair_distances, axis=1)
    min_per_step = np.min(distances, axis=1)
    collision_events = _count_true_runs(min_per_step < settings.collision_distance_m)
    near_miss_mask = (min_per_step >= settings.collision_distance_m) & (
        min_per_step < settings.near_miss_distance_m
    )
    near_miss_events = _count_true_runs(near_miss_mask)

    deadlock_steps = 0
    deadlock_detected = False
    if steps >= 2:
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=2) / max(float(dt), 1e-9)
        # This minimal first slice only marks deadlock when the whole fleet stays slow.
        all_slow = np.all(speeds <= settings.deadlock_speed_mps, axis=1)
        deadlock_steps = int(np.count_nonzero(all_slow))
        deadlock_detected = _has_consecutive_true(all_slow, settings.deadlock_window_steps)

    return {
        "robot_count": float(robots),
        "pair_count": float(pair_count),
        "min_inter_robot_distance_m": float(np.min(min_per_step)),
        "inter_robot_collision_events": float(collision_events),
        "inter_robot_near_miss_events": float(near_miss_events),
        "deadlock_steps": float(deadlock_steps),
        "deadlock_detected": bool(deadlock_detected),
    }


def multi_amv_episode_extension(
    *,
    settings: MultiAmvSettings,
    inter_robot: dict[str, float | bool],
    planner_family: str = "goal_controller_smoke",
    planner_status: str = "goal_controller_smoke",
    planner_note: str | None = None,
) -> dict[str, Any]:
    """Build an additive episode-record block for multi-AMV benchmark outputs.

    The block is intentionally namespaced under ``multi_amv`` so single-robot
    episode consumers can ignore it without schema migration.

    Returns
    -------
    dict[str, Any]
        Namespaced episode extension containing settings, planner status, and inter-robot metrics.
    """
    if settings.num_robots < 2:
        raise ValueError("multi-AMV episode extension requires at least two robots")
    if not inter_robot:
        raise ValueError("inter_robot metrics must be non-empty")
    planner_support = multi_amv_planner_support(planner_family)
    return {
        "multi_amv": {
            "enabled": True,
            "num_robots": int(settings.num_robots),
            "near_miss_distance_m": float(settings.near_miss_distance_m),
            "collision_distance_m": float(settings.collision_distance_m),
            "deadlock_speed_mps": float(settings.deadlock_speed_mps),
            "deadlock_window_steps": int(settings.deadlock_window_steps),
            "planner_family": planner_support.planner_family,
            "planner_status": str(planner_status),
            "planner_support": planner_support.to_json_dict(),
            "planner_note": planner_note,
            "metrics": {"inter_robot": dict(inter_robot)},
        }
    }


def _count_true_runs(values: np.ndarray) -> int:
    """Count contiguous true runs in a boolean sequence.

    Returns:
        Number of distinct true runs in the sequence.
    """

    mask = np.asarray(values, dtype=bool)
    if mask.size == 0:
        return 0
    padded = np.concatenate((np.array([False]), mask, np.array([False])))
    starts = np.logical_not(padded[:-1]) & padded[1:]
    return int(np.count_nonzero(starts))


def _has_consecutive_true(values: np.ndarray, window: int) -> bool:
    """Return whether a boolean sequence contains ``window`` consecutive true values."""
    run = 0
    for value in values:
        run = run + 1 if bool(value) else 0
        if run >= window:
            return True
    return False
