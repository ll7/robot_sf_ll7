"""Alyassi-inspired multi-objective reward library.

This module implements a practical reward composition aligned with the taxonomy in
Alyassi et al. (2025), Section 3.1 (Objective function, 3.1.1--3.1.12).

The implementation is intentionally robust to missing metadata keys so it can run with
current `robot_sf_ll7` environment metadata and optional extended fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class AlyassiRewardWeights:
    """Weight set for the Alyassi-inspired reward components."""

    w_goal: float = 1.0
    w_collision: float = 2.0
    w_efficiency: float = 0.2
    w_smoothness: float = 0.2
    w_social: float = 0.8
    w_geometric_collision: float = 0.8
    w_human_preference: float = 0.4
    w_human_prediction: float = 0.5
    w_exploration: float = 0.05
    w_task_specific: float = 0.2
    w_demo_learning: float = 0.2
    w_weight_learning: float = 0.0


ALYASSI_COMPONENT_CITATIONS: dict[str, tuple[str, ...]] = {
    "goal": (
        "Long et al. (2018)",
        "Tan et al. (2020)",
        "Liu Z. et al. (2023)",
    ),
    "collision": (
        "Chen et al. (2017b)",
        "Cui et al. (2021)",
    ),
    "efficiency": (
        "Lee and Jeong (2023)",
        "Wang Y. et al. (2018)",
        "Choi et al. (2019)",
    ),
    "smoothness": (
        "Tan et al. (2020)",
        "Xie and Dames (2023)",
        "Hoeller et al. (2021)",
    ),
    "social": ("Chen et al. (2017c)",),
    "geometric_collision": (
        "Xie and Dames (2023)",
        "Han R. et al. (2022)",
        "Zhu et al. (2022)",
        "Samsani and Muhammad (2021)",
    ),
    "human_preference": (
        "Christiano et al. (2017)",
        "Ouyang et al. (2022)",
        "Wang R. et al. (2022)",
    ),
    "human_prediction": (
        "Liu S. et al. (2023)",
        "Eiffert et al. (2020b)",
    ),
    "exploration": (
        "Schulman et al. (2017)",
        "Pathak et al. (2017)",
        "Shi H. et al. (2019)",
        "Martinez-Baselga et al. (2023)",
    ),
    "task_specific": ("Li et al. (2018)",),
    "demo_learning": (
        "Okal and Arras (2016)",
        "Kim and Pineau (2016)",
        "Fahad et al. (2018)",
        "Vasquez et al. (2014)",
        "Brown et al. (2020)",
    ),
    "weight_learning": (
        "Ziebart et al. (2008)",
        "Chiang et al. (2019)",
        "Parker-Holder et al. (2022)",
    ),
}


def _f(meta: Mapping[str, object], key: str, default: float = 0.0) -> float:
    """Safely read numeric metadata values.

    Returns:
        Numeric value when conversion succeeds, otherwise ``default``.
    """
    value = meta.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _b(meta: Mapping[str, object], key: str, default: bool = False) -> bool:
    """Safely read boolean metadata values.

    Returns:
        Boolean cast of metadata field value, or ``default`` when absent.
    """
    value = meta.get(key, default)
    return bool(value)


def _goal_component(meta: Mapping[str, object], terminal_success_bonus: float) -> float:
    """Goal term: sparse success plus optional dense progress shaping.

    Returns:
        Goal component score.
    """
    success = _b(meta, "is_route_complete") or _b(meta, "is_robot_at_goal")
    sparse = terminal_success_bonus if success else 0.0

    curr = meta.get("distance_to_goal")
    prev = meta.get("prev_distance_to_goal")
    if curr is None or prev is None:
        return sparse

    try:
        dense = float(prev) - float(curr)
    except (TypeError, ValueError):
        dense = 0.0
    return sparse + dense


def _collision_component(meta: Mapping[str, object], near_miss_dist: float) -> float:
    """Collision-avoidance term with optional proximity shaping.

    Returns:
        Collision component score.
    """
    collision = (
        _b(meta, "is_pedestrian_collision")
        or _b(meta, "is_robot_collision")
        or _b(meta, "is_obstacle_collision")
    )
    collision_term = -1.0 if collision else 0.0

    min_ped = meta.get("min_ped_distance")
    if min_ped is None:
        return collision_term

    try:
        min_dist = float(min_ped)
    except (TypeError, ValueError):
        return collision_term

    if min_dist >= near_miss_dist:
        prox_term = 0.0
    else:
        prox_term = -max(0.0, (near_miss_dist - min_dist) / max(near_miss_dist, 1e-6))
    return collision_term + prox_term


def _efficiency_component(
    meta: Mapping[str, object], step_cost: float, speed_target: float
) -> float:
    """Efficiency term: per-step cost plus optional speed encouragement.

    Returns:
        Efficiency component score.
    """
    value = -abs(step_cost)

    speed = meta.get("speed")
    if speed is None:
        action = meta.get("action")
        if action is not None:
            try:
                speed = float(np.linalg.norm(np.asarray(action, dtype=float)))
            except (TypeError, ValueError):
                speed = None

    if speed is None:
        return value

    try:
        s = float(speed)
    except (TypeError, ValueError):
        return value
    speed_term = 1.0 - abs(s - speed_target) / max(speed_target, 1e-6)
    # Keep speed shaping bounded so large speed outliers do not dominate all terms.
    speed_term = float(np.clip(speed_term, -1.0, 1.0))
    value += speed_term
    return value


def _smoothness_component(meta: Mapping[str, object]) -> float:
    """Smoothness term from action-change and optional angular velocity penalty.

    Returns:
        Smoothness component score.
    """
    action = meta.get("action")
    last_action = meta.get("last_action")
    term = 0.0

    if action is not None and last_action is not None:
        try:
            diff = float(
                np.linalg.norm(
                    np.asarray(action, dtype=float) - np.asarray(last_action, dtype=float)
                )
            )
            term -= diff
        except (TypeError, ValueError):
            pass

    term -= abs(_f(meta, "angular_velocity"))
    return term


def _social_component(meta: Mapping[str, object]) -> float:
    """Social term from near-miss and comfort proxies.

    Note:
        ``near_misses`` is intentionally used here and also in
        :func:`_geometric_collision_component`, so the final :func:`alyassi_reward`
        can apply distinct tunable weights (`w_social`, `w_geometric_collision`)
        to the same event stream from different conceptual channels.

    Returns:
        Social component score.
    """
    near_misses = _f(meta, "near_misses", 0.0)
    comfort_exposure = _f(meta, "comfort_exposure", 0.0)
    return -(near_misses + comfort_exposure)


def _geometric_collision_component(meta: Mapping[str, object]) -> float:
    """Geometric safety proxy from threshold and force-exceed events.

    Note:
        ``near_misses`` is intentionally shared with :func:`_social_component`.
        This overlap is expected and controlled via :func:`alyassi_reward`
        weights (`w_social`, `w_geometric_collision`).

    Returns:
        Geometric-collision component score.
    """
    return -(_f(meta, "near_misses", 0.0) + _f(meta, "force_exceed_events", 0.0))


def _human_preference_component(meta: Mapping[str, object]) -> float:
    """Optional external preference-model score.

    Returns:
        Human-preference component score.
    """
    return _f(meta, "human_preference_score", 0.0)


def _human_prediction_component(meta: Mapping[str, object]) -> float:
    """Prediction term using optional zone and uncertainty proxies.

    Returns:
        Human-prediction component score.
    """
    zone_intrusion = _f(meta, "prediction_zone_intrusion", 0.0)
    cov_det_sqrt = _f(meta, "prediction_cov_sqrt_det", 0.0)
    return -(zone_intrusion + cov_det_sqrt)


def _exploration_component(meta: Mapping[str, object]) -> float:
    """Optional entropy-style exploration bonus.

    Returns:
        Exploration component score.
    """
    return _f(meta, "policy_entropy_bonus", 0.0)


def _task_specific_component(meta: Mapping[str, object]) -> float:
    """Task-specific scalar term and optional companion-distance shaping.

    Returns:
        Task-specific component score.
    """
    custom = _f(meta, "task_specific_score", 0.0)

    companion_err = meta.get("companion_distance_error")
    if companion_err is None:
        return custom
    return custom - abs(_f(meta, "companion_distance_error", 0.0))


def _demo_learning_component(meta: Mapping[str, object]) -> float:
    """Optional demonstration similarity score for IRL/D-REX style shaping.

    Returns:
        Demonstration-learning component score.
    """
    return _f(meta, "demonstration_similarity", 0.0)


def _weight_learning_component(meta: Mapping[str, object]) -> float:
    """Optional reward-weight adaptation quality signal.

    Returns:
        Weight-learning component score.
    """
    return _f(meta, "weight_learning_score", 0.0)


def alyassi_component_scores(
    meta: Mapping[str, object],
    *,
    terminal_success_bonus: float = 1.0,
    step_cost: float = 0.01,
    near_miss_dist: float = 0.5,
    speed_target: float = 0.7,
) -> dict[str, float]:
    """Compute unweighted Alyassi-inspired component scores for one step.

    Returns:
        Mapping from component names to component scores.
    """
    return {
        "goal": _goal_component(meta, terminal_success_bonus),
        "collision": _collision_component(meta, near_miss_dist),
        "efficiency": _efficiency_component(meta, step_cost, speed_target),
        "smoothness": _smoothness_component(meta),
        "social": _social_component(meta),
        "geometric_collision": _geometric_collision_component(meta),
        "human_preference": _human_preference_component(meta),
        "human_prediction": _human_prediction_component(meta),
        "exploration": _exploration_component(meta),
        "task_specific": _task_specific_component(meta),
        "demo_learning": _demo_learning_component(meta),
        "weight_learning": _weight_learning_component(meta),
    }


def alyassi_reward(
    meta: Mapping[str, object],
    *,
    weights: AlyassiRewardWeights = AlyassiRewardWeights(),
    terminal_success_bonus: float = 1.0,
    step_cost: float = 0.01,
    near_miss_dist: float = 0.5,
    speed_target: float = 0.7,
) -> float:
    """Compute weighted Alyassi-style multi-objective reward.

    Returns:
        Scalar weighted reward value.
    """
    components = alyassi_component_scores(
        meta,
        terminal_success_bonus=terminal_success_bonus,
        step_cost=step_cost,
        near_miss_dist=near_miss_dist,
        speed_target=speed_target,
    )
    total = (
        weights.w_goal * components["goal"]
        + weights.w_collision * components["collision"]
        + weights.w_efficiency * components["efficiency"]
        + weights.w_smoothness * components["smoothness"]
        + weights.w_social * components["social"]
        + weights.w_geometric_collision * components["geometric_collision"]
        + weights.w_human_preference * components["human_preference"]
        + weights.w_human_prediction * components["human_prediction"]
        + weights.w_exploration * components["exploration"]
        + weights.w_task_specific * components["task_specific"]
        + weights.w_demo_learning * components["demo_learning"]
        + weights.w_weight_learning * components["weight_learning"]
    )
    return float(total)


def alyassi_component_citations() -> dict[str, tuple[str, ...]]:
    """Return citation mapping for reward components.

    Returns:
        Mapping from component names to tuples of citation labels.
    """
    return dict(ALYASSI_COMPONENT_CITATIONS)
