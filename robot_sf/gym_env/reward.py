"""
This module defines the reward function for the robot environment.
"""

from __future__ import annotations

import importlib
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.gym_env.reward_alyassi import alyassi_reward

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_DEFAULT_SNQI_REWARD_WEIGHTS: dict[str, float] = {
    "w_success": 1.0,
    "w_time": 0.8,
    "w_collisions": 2.0,
    "w_near": 1.0,
    "w_comfort": 0.5,
    "w_force_exceed": 1.5,
    "w_jerk": 0.3,
}
_DEFAULT_SNQI_REWARD_BASELINE: dict[str, dict[str, float]] = {
    "collisions": {"med": 0.0, "p95": 1.0},
    "near_misses": {"med": 0.0, "p95": 1.0},
    "force_exceed_events": {"med": 0.0, "p95": 1.0},
    "jerk_mean": {"med": 0.0, "p95": 1.0},
}
_SNQI_COMPUTE_FN = None


def _compute_snqi_reward_score(
    metric_values: dict[str, float],
    *,
    weights_map: Mapping[str, float],
    baseline_map: Mapping[str, Mapping[str, float]],
) -> float:
    """Compute SNQI score via lazy-loaded benchmark module to avoid import cycles.

    Returns:
        Canonical SNQI score.
    """
    global _SNQI_COMPUTE_FN
    if _SNQI_COMPUTE_FN is None:
        module = importlib.import_module("robot_sf.benchmark.snqi.compute")
        _SNQI_COMPUTE_FN = module.compute_snqi
    return float(_SNQI_COMPUTE_FN(metric_values, weights_map, baseline_map))


def simple_reward(
    meta: dict,
    max_episode_step_discount: float = -0.1,
    ped_coll_penalty: float = -5,
    obst_coll_penalty: float = -2,
    reach_route_reward: float = 1,
) -> float:
    """Calculate the reward for the robot's current state.

    Args:
        meta: Metadata dictionary (see ``RobotState.meta_dict``).
        max_episode_step_discount: Per-step discount divided by ``max_sim_steps``.
        ped_coll_penalty: Penalty applied when colliding with pedestrians/robots.
        obst_coll_penalty: Penalty applied when colliding with obstacles.
        reach_route_reward: Reward granted when the robot completes its route.

    Returns:
        float: Scalar reward for the timestep.
    """

    # Initialize reward with a discount based on the maximum simulation steps
    reward = max_episode_step_discount / meta["max_sim_steps"]

    # If there's a collision with a pedestrian or another robot, apply penalty
    if meta["is_pedestrian_collision"] or meta["is_robot_collision"]:
        reward += ped_coll_penalty

    # If there's a collision with an obstacle, apply penalty
    if meta["is_obstacle_collision"]:
        reward += obst_coll_penalty

    # Success is route completion only.
    if meta.get("is_route_complete"):
        reward += reach_route_reward

    return float(reward)


def simple_ped_reward(
    meta: dict,
    max_episode_step_discount: float = -0.1,
    ped_coll_penalty: float = -5,
    obst_coll_penalty: float = -5,
    robot_coll_reward: float = 5,
    robot_route_complete_penalty: float = -1,
) -> float:
    """Calculate the reward for the pedestrian's current state.

    Args:
        meta: Metadata dictionary describing collisions, goal status, and distances.
        max_episode_step_discount: Per-step discount divided by ``max_sim_steps``.
        ped_coll_penalty: Penalty applied when the ego pedestrian collides with others.
        obst_coll_penalty: Penalty applied when colliding with obstacles.
        robot_coll_reward: Bonus granted when colliding with the robot.
        robot_route_complete_penalty: Penalty applied if the robot completes its route.

    Returns:
        float: Scalar reward for the timestep.
    """

    # Initialize reward with a discount based on the maximum simulation steps

    reward = max_episode_step_discount / meta["max_sim_steps"]

    distance = meta["distance_to_robot"]
    reward += distance * -0.001

    # If there's a collision with a pedestrian or another robot, apply penalty
    if meta["is_pedestrian_collision"]:
        reward += ped_coll_penalty

    # If there's a collision with an obstacle, apply penalty
    if meta["is_obstacle_collision"]:
        reward += obst_coll_penalty

    # there's a collision with a robot, apply reward
    if meta["is_robot_collision"]:
        reward += robot_coll_reward

    # If the robot completed the route, apply penalty.
    if meta.get("is_route_complete"):
        reward += robot_route_complete_penalty

    return float(reward)


def punish_action_reward(
    meta: dict,
    max_episode_step_discount: float = -0.1,
    ped_coll_penalty: float = -5,
    obst_coll_penalty: float = -2,
    reach_route_reward: float = 1,
    punish_action: bool = True,
    punish_action_penalty: float = -0.1,
) -> float:
    """Robot reward variant that penalizes action changes.

    Args:
        meta: Metadata dictionary describing the current state/action.
        max_episode_step_discount: Per-step discount divided by ``max_sim_steps``.
        ped_coll_penalty: Penalty applied when colliding with pedestrians/robots.
        obst_coll_penalty: Penalty applied when colliding with obstacles.
        reach_route_reward: Reward granted when the robot completes its route.
        punish_action: Whether to penalize deviations from the previous action.
        punish_action_penalty: Scaling factor for the action difference penalty.

    Returns:
        float: Scalar reward for the timestep.
    """

    # Initialize reward with a discount based on the maximum simulation steps
    reward = simple_reward(
        meta,
        max_episode_step_discount,
        ped_coll_penalty,
        obst_coll_penalty,
        reach_route_reward,
    )

    # punish the robot taking a different action from the last action
    if punish_action and meta["last_action"] is not None:
        action_diff = np.linalg.norm(np.array(meta["action"]) - np.array(meta["last_action"]))
        if action_diff > 0:
            reward += punish_action_penalty * action_diff

    return float(reward)


def snqi_step_reward(
    meta: dict,
    *,
    weights: Mapping[str, float] | None = None,
    baseline_stats: Mapping[str, Mapping[str, float]] | None = None,
    terminal_bonus: float = 0.0,
    living_penalty: float = 0.0,
) -> float:
    """Compute a per-step reward using canonical SNQI scoring.

    This projects available step metadata into the canonical SNQI formula.
    It is useful when you want reward semantics aligned with benchmark SNQI,
    but it is still an approximation because some SNQI terms may be missing
    from step metadata (e.g., near misses, force exceed events).

    Returns:
        Scalar reward value for the current step.
    """
    weights_map = dict(weights or _DEFAULT_SNQI_REWARD_WEIGHTS)
    baseline_map = {
        key: dict(value) for key, value in (baseline_stats or _DEFAULT_SNQI_REWARD_BASELINE).items()
    }
    step = float(meta.get("step_of_episode", 0.0) or 0.0)
    max_steps = float(meta.get("max_sim_steps", 1.0) or 1.0)
    if max_steps <= 0:
        max_steps = 1.0
    collision = (
        1.0
        if bool(meta.get("is_pedestrian_collision"))
        or bool(meta.get("is_robot_collision"))
        or bool(meta.get("is_obstacle_collision"))
        else 0.0
    )
    goal_reached = bool(meta.get("is_route_complete"))
    # Mirror benchmark semantics: collisions invalidate success.
    success = 1.0 if goal_reached and collision == 0.0 else 0.0
    time_to_goal_norm = min(1.0, max(0.0, step / max_steps))
    metric_values = {
        "success": success,
        "time_to_goal_norm": time_to_goal_norm,
        "collisions": collision,
        "near_misses": float(meta.get("near_misses", 0.0) or 0.0),
        "comfort_exposure": float(meta.get("comfort_exposure", 0.0) or 0.0),
        "force_exceed_events": float(meta.get("force_exceed_events", 0.0) or 0.0),
        "jerk_mean": float(meta.get("jerk_mean", 0.0) or 0.0),
    }
    reward = _compute_snqi_reward_score(
        metric_values,
        weights_map=weights_map,
        baseline_map=baseline_map,
    )
    if success > 0.0:
        reward += float(terminal_bonus)
    reward += float(living_penalty)
    return reward


_ROUTE_COMPLETION_V2_WEIGHTS: dict[str, float] = {
    "progress": 2.5,
    "living": -0.01,
    "collision": -5.0,
    "near_miss": -0.8,
    "ttc_risk": -0.6,
    "comfort": -0.4,
    "smoothness": -0.15,
    "terminal_bonus": 2.0,
}

_ROUTE_COMPLETION_V3_WEIGHTS: dict[str, float] = {
    "progress": 2.2,
    "living": -0.01,
    "collision": -10.0,
    "near_miss": -1.0,
    "ttc_risk": -0.8,
    "comfort": -0.5,
    "smoothness": -0.2,
    "timeout": -3.0,
    "stagnation": -1.2,
    "terminal_bonus": 3.0,
}

_SOCIAL_QUALITY_V1_WEIGHTS: dict[str, float] = {
    "progress": 1.0,
    "living": -0.02,
    "collision": -6.0,
    "near_miss": -1.2,
    "ttc_risk": -0.9,
    "comfort": -0.9,
    "smoothness": -0.2,
    "terminal_bonus": 1.5,
}


def _float(meta: Mapping[str, object], key: str, default: float = 0.0) -> float:
    """Return ``meta[key]`` as a finite float, otherwise ``default``.

    Args:
        meta: Reward metadata mapping.
        key: Metadata key to read.
        default: Fallback value when missing, invalid, or non-finite.

    Returns:
        Finite float value for the selected key.
    """
    value = meta.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(result):
        return float(default)
    return result


def _bounded(value: float, lo: float, hi: float) -> float:
    """Clip a scalar to ``[lo, hi]`` and return it as ``float``.

    Args:
        value: Candidate scalar value.
        lo: Lower clip bound.
        hi: Upper clip bound.

    Returns:
        Clipped float in the configured range.
    """
    return float(np.clip(float(value), float(lo), float(hi)))


def _ttc_risk_from_meta(meta: Mapping[str, object]) -> float:
    """Compute bounded TTC risk from metadata using inverse TTC or near-miss fallback.

    Args:
        meta: Reward metadata mapping; reads ``time_to_collision`` and ``near_misses``.

    Returns:
        Risk proxy in ``[0, 1]`` where larger values indicate higher collision risk.
    """
    # Prefer explicit TTC if available; otherwise fall back to near-miss proxy.
    ttc = _float(meta, "time_to_collision", float("inf"))
    if np.isfinite(ttc) and ttc > 0.0:
        return _bounded(1.0 / max(ttc, 1e-3), 0.0, 1.0)
    near_misses = _float(meta, "near_misses", 0.0)
    return _bounded(near_misses, 0.0, 1.0)


def _progress_term(meta: Mapping[str, object]) -> float:
    """Return bounded goal-progress delta from consecutive distance estimates.

    Args:
        meta: Reward metadata mapping with distance-to-goal fields.

    Returns:
        Progress term in ``[-1, 1]`` as ``prev_distance_to_goal - distance_to_goal``.
    """
    prev_dist = _float(meta, "prev_distance_to_goal", 0.0)
    dist = _float(meta, "distance_to_goal", prev_dist)
    return _bounded(prev_dist - dist, -1.0, 1.0)


def _timeout_term(meta: Mapping[str, object]) -> float:
    """Return 1.0 only for timeout failures without success/collision events.

    Args:
        meta: Reward metadata mapping.

    Returns:
        Timeout indicator in ``{0.0, 1.0}``.
    """
    timeout = bool(meta.get("is_timesteps_exceeded"))
    collision = bool(
        meta.get("is_pedestrian_collision")
        or meta.get("is_robot_collision")
        or meta.get("is_obstacle_collision")
    )
    route_complete = bool(meta.get("is_route_complete"))
    return 1.0 if timeout and not collision and not route_complete else 0.0


def _stagnation_term(meta: Mapping[str, object]) -> float:
    """Return a bounded stagnation proxy from non-positive route progress.

    Args:
        meta: Reward metadata mapping with goal-distance fields.

    Returns:
        Scalar in ``[0, 1]`` where larger values indicate stronger stagnation.
    """
    progress = _progress_term(meta)
    if progress > 0.0:
        return 0.0
    return _bounded(-progress, 0.0, 1.0)


def _reward_with_terms(
    meta: dict[str, object],
    *,
    weights: Mapping[str, float],
) -> float:
    """Compute weighted reward terms and write decomposition back into metadata.

    Args:
        meta: Mutable reward metadata dictionary.
        weights: Per-term scalar weights keyed by reward term name.

    Returns:
        Weighted scalar reward total.

    Side effects:
        Mutates ``meta`` by setting ``reward_terms`` and ``reward_total``.
    """
    collision_flag = bool(
        meta.get("is_pedestrian_collision")
        or meta.get("is_robot_collision")
        or meta.get("is_obstacle_collision")
    )
    route_complete = bool(meta.get("is_route_complete"))
    terms = {
        "progress": _progress_term(meta),
        "living": 1.0,
        "collision": 1.0 if collision_flag else 0.0,
        "near_miss": _bounded(_float(meta, "near_misses", 0.0), 0.0, 1.0),
        "ttc_risk": _ttc_risk_from_meta(meta),
        "comfort": _bounded(_float(meta, "comfort_exposure", 0.0), 0.0, 1.0),
        "smoothness": _bounded(_float(meta, "jerk_mean", 0.0), 0.0, 5.0) / 5.0,
        "timeout": _timeout_term(meta),
        "stagnation": _stagnation_term(meta),
        "terminal_bonus": 1.0 if route_complete and not collision_flag else 0.0,
    }
    weighted_terms = {name: float(weights.get(name, 0.0)) * value for name, value in terms.items()}
    total = float(sum(weighted_terms.values()))
    meta["reward_terms"] = weighted_terms
    meta["reward_total"] = total
    return total


def route_completion_v2_reward(
    meta: dict,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Route-completion-first reward profile with bounded social/smoothness shaping.

    Note:
        Mutates ``meta`` in place by writing ``reward_terms`` and ``reward_total``
        for per-step decomposition logging.

    Returns:
        float: Scalar reward with per-term decomposition written into ``meta``.
    """
    weight_map = dict(_ROUTE_COMPLETION_V2_WEIGHTS)
    if weights:
        weight_map.update({k: float(v) for k, v in weights.items()})
    return _reward_with_terms(meta, weights=weight_map)


def route_completion_v3_reward(
    meta: dict,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Route-completion profile with explicit timeout and stagnation penalties.

    Note:
        Mutates ``meta`` in place by writing ``reward_terms`` and ``reward_total``
        for per-step decomposition logging.

    Returns:
        float: Scalar reward with per-term decomposition written into ``meta``.
    """
    weight_map = dict(_ROUTE_COMPLETION_V3_WEIGHTS)
    if weights:
        weight_map.update({k: float(v) for k, v in weights.items()})
    return _reward_with_terms(meta, weights=weight_map)


def social_quality_v1_reward(
    meta: dict,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Social-quality-focused reward profile that remains route-completion compatible.

    Note:
        Mutates ``meta`` in place by writing ``reward_terms`` and ``reward_total``
        for per-step decomposition logging.

    Returns:
        float: Scalar reward with per-term decomposition written into ``meta``.
    """
    weight_map = dict(_SOCIAL_QUALITY_V1_WEIGHTS)
    if weights:
        weight_map.update({k: float(v) for k, v in weights.items()})
    return _reward_with_terms(meta, weights=weight_map)


def build_reward_function(
    reward_name: str,
    reward_kwargs: Mapping[str, object] | None = None,
) -> Callable[[dict], float]:
    """Build a reward callable by name.

    Returns:
        Callable that accepts a reward metadata dictionary and returns a scalar reward.
    """
    normalized = reward_name.strip().lower()
    kwargs = dict(reward_kwargs or {})
    if normalized in {"simple", "simple_reward"}:
        return partial(simple_reward, **kwargs)
    if normalized in {"punish_action", "punish_action_reward"}:
        return partial(punish_action_reward, **kwargs)
    if normalized in {"snqi", "snqi_step", "snqi_step_reward"}:
        return partial(snqi_step_reward, **kwargs)
    if normalized in {"alyassi", "alyassi_reward", "alyassi_composite"}:
        return partial(alyassi_reward, **kwargs)
    if normalized in {"route_completion_v2", "route_completion"}:
        return partial(route_completion_v2_reward, **kwargs)
    if normalized in {"route_completion_v3"}:
        return partial(route_completion_v3_reward, **kwargs)
    if normalized in {"social_quality_v1", "social_quality"}:
        return partial(social_quality_v1_reward, **kwargs)
    supported = (
        "simple",
        "punish_action",
        "snqi_step",
        "alyassi",
        "alyassi_composite",
        "route_completion_v2",
        "route_completion",
        "route_completion_v3",
        "social_quality_v1",
        "social_quality",
    )
    raise ValueError(f"Unknown reward_name '{reward_name}'. Supported: {supported}")
