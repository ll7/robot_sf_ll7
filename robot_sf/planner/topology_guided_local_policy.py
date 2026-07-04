"""Diagnostic topology-hypothesis wrapper for the hybrid-rule local planner."""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.planner.grid_route import GridRoutePlannerAdapter, build_grid_route_config
from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleCandidate,
    HybridRuleLocalPlannerAdapter,
    HybridRuleLocalPlannerConfig,
    build_hybrid_rule_local_planner_config,
)

_EPS = 1e-9
_DEFAULT_TOPOLOGY_ARBITRATION_WEIGHT = 0.35
_TOPOLOGY_KEYS = {
    "diagnostic_only",
    "claim_boundary",
    "min_hypotheses",
    "max_hypotheses",
    "block_radius_cells",
    "block_stride_cells",
    "max_path_overlap",
    "length_weight",
    "static_clearance_weight",
    "fail_closed_on_missing_inputs",
    "fail_closed_on_insufficient_hypotheses",
    "topology_command_enabled",
    "topology_command_speed",
    "topology_command_heading_gain",
    "topology_command_turn_in_place_error",
    "near_parity_diversity_gate_enabled",
    "near_parity_route_distance_slack_m",
    "near_parity_route_distance_slack_ratio",
    "near_parity_static_clearance_floor_m",
    "near_parity_diversity_bonus",
    "primary_route_reuse_penalty_enabled",
    "primary_route_reuse_penalty_weight",
    "primary_route_reuse_penalty_cooldown_steps",
    "primary_route_reuse_penalty_min_prior_primary_selections",
    "primary_route_progress_gate_enabled",
    "primary_route_progress_gate_threshold_m",
    "primary_route_progress_gate_min_samples",
    "primary_route_progress_gate_use_monotone_accounting",
    "topology_guided",
    "schema_version",
    "enabled",
    "candidate_required",
    "fallback_on_no_candidate",
    "arbitration_weight",
    "near_parity_margin",
    "min_route_progress_delta_m",
    "stall_window_steps",
    "route_hypothesis",
}

# Minimum number of recent primary-route route-remaining samples required before
# route-progress can be evaluated. Two samples are the smallest pair that admits
# a finite-difference progress estimate; this is the historical hardcoded value.
_DEFAULT_PROGRESS_GATE_MIN_SAMPLES = 2


@dataclass(frozen=True)
class TopologyGuidedLocalPolicyConfig:
    """Configuration for the diagnostic topology-guided local policy."""

    hybrid_rule: HybridRuleLocalPlannerConfig
    route_hypothesis: Any
    diagnostic_only: bool = True
    claim_boundary: str = "diagnostic_only"
    schema_version: str = "topology_guided_hybrid_rule.v1"
    enabled: bool = True
    candidate_required: bool = False
    fallback_on_no_candidate: bool = True
    arbitration_weight: float = _DEFAULT_TOPOLOGY_ARBITRATION_WEIGHT
    min_route_progress_delta_m: float = 0.05
    stall_window_steps: int = 20
    min_hypotheses: int = 2
    max_hypotheses: int = 2
    block_radius_cells: int = 3
    block_stride_cells: int = 8
    max_path_overlap: float = 0.88
    length_weight: float = 1.0
    static_clearance_weight: float = 0.6
    fail_closed_on_missing_inputs: bool = True
    fail_closed_on_insufficient_hypotheses: bool = False
    topology_command_enabled: bool = True
    topology_command_speed: float = 0.35
    topology_command_heading_gain: float = 1.0
    topology_command_turn_in_place_error: float = 0.25
    near_parity_diversity_gate_enabled: bool = False
    near_parity_route_distance_slack_m: float = 0.75
    near_parity_route_distance_slack_ratio: float = 0.05
    near_parity_static_clearance_floor_m: float = 0.05
    near_parity_diversity_bonus: float = 0.0
    primary_route_reuse_penalty_enabled: bool = False
    primary_route_reuse_penalty_weight: float = 1.0
    primary_route_reuse_penalty_cooldown_steps: int = 3
    primary_route_reuse_penalty_min_prior_primary_selections: int = 2
    primary_route_progress_gate_enabled: bool = False
    primary_route_progress_gate_threshold_m: float = 0.0
    primary_route_progress_gate_min_samples: int = _DEFAULT_PROGRESS_GATE_MIN_SAMPLES
    primary_route_progress_gate_use_monotone_accounting: bool = False


@dataclass(frozen=True)
class _RouteHypothesis:
    """One masked-route topology hypothesis."""

    hypothesis_id: str
    path: list[tuple[int, int]]
    clearance_map: np.ndarray | None
    blocked_cell: tuple[int, int] | None = None


@dataclass
class RouteProgressState:
    """State for topology route-progress and near-parity churn accounting."""

    last_route_progress_m: float | None = None
    stagnant_steps: int = 0
    last_selected_candidate: str | None = None
    candidate_switch_count: int = 0


def _path_overlap(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> float:
    """Return Jaccard overlap between two grid-cell paths."""
    left_cells = set(left)
    right_cells = set(right)
    union = left_cells | right_cells
    if not union:
        return 1.0
    return float(len(left_cells & right_cells) / len(union))


def _path_length(path: list[tuple[int, int]], *, resolution: float) -> float:
    """Return route length in metres."""
    if len(path) < 2:
        return 0.0
    points = np.asarray(path, dtype=float)
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])) * resolution)


def _first_float(value: Any, default: float) -> float:
    """Return the first finite float from a scalar-like payload."""
    try:
        raw = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return default
    if not raw.size or not np.isfinite(raw[0]):
        return default
    return float(raw[0])


def _block_path_cell(
    blocked: np.ndarray,
    cell: tuple[int, int],
    *,
    radius: int,
    protected: set[tuple[int, int]],
) -> np.ndarray:
    """Return a blocked-grid copy with one local route patch masked."""
    updated = blocked.copy()
    row, col = cell
    radius = max(int(radius), 0)
    for rr in range(max(0, row - radius), min(updated.shape[0], row + radius + 1)):
        for cc in range(max(0, col - radius), min(updated.shape[1], col + radius + 1)):
            if (rr, cc) not in protected:
                updated[rr, cc] = True
    return updated


def _static_clearance_summary(
    path: list[tuple[int, int]],
    clearance_map: np.ndarray | None,
    *,
    resolution: float,
) -> dict[str, float | None]:
    """Summarize static clearance along one route hypothesis.

    Returns:
        dict[str, float | None]: Minimum and mean static clearance in metres.
    """
    if clearance_map is None or not path:
        return {"static_clearance_min_m": None, "static_clearance_mean_m": None}
    values = [
        float(clearance_map[cell]) * resolution
        for cell in path
        if np.isfinite(float(clearance_map[cell]))
    ]
    if not values:
        return {"static_clearance_min_m": None, "static_clearance_mean_m": None}
    return {
        "static_clearance_min_m": float(min(values)),
        "static_clearance_mean_m": float(np.mean(values)),
    }


def _topology_score_payload(
    config: TopologyGuidedLocalPolicyConfig,
    *,
    route_remaining: float,
    static_clearance_min: float | None,
) -> dict[str, Any]:
    """Return score and score components for one topology hypothesis."""
    length_penalty = -float(config.length_weight) * float(route_remaining)
    static_clearance_bonus = 0.0
    if static_clearance_min is not None:
        static_clearance_bonus = float(config.static_clearance_weight) * float(static_clearance_min)
    score = length_penalty + static_clearance_bonus
    return {
        "score": float(score),
        "score_components": {
            "length_penalty": float(length_penalty),
            "static_clearance_bonus": float(static_clearance_bonus),
        },
    }


def _annotate_topology_selection(
    hypotheses: list[dict[str, Any]],
    *,
    selected: dict[str, Any],
) -> None:
    """Annotate topology hypotheses with rank, score margin, and rejection reason."""
    selected_score = float(selected.get("selection_score", selected["score"]))
    for score_rank, item in enumerate(
        sorted(
            hypotheses,
            key=lambda item: float(item.get("selection_score", item["score"])),
            reverse=True,
        )
    ):
        item["score_rank"] = int(score_rank)
        item["score_margin_to_selected"] = float(
            selected_score - float(item.get("selection_score", item["score"]))
        )
        if item["hypothesis_id"] == selected["hypothesis_id"]:
            item["selection_outcome"] = "selected"
            item["rejection_reason"] = None
        else:
            item["selection_outcome"] = "rejected"
            item["rejection_reason"] = item.get(
                "selection_rejection_reason",
                "lower_topology_selection_score",
            )


def _finite_optional_float(value: Any) -> float | None:
    """Return a finite float or ``None`` for optional diagnostic fields."""
    if isinstance(value, int | float | np.integer | np.floating):
        candidate = float(value)
        if np.isfinite(candidate):
            return candidate
    return None


def _near_parity_blocker_reason(
    *,
    route_delta: float | None,
    primary_route: float | None,
    primary_clearance: float | None,
    alt_clearance: float | None,
    config: TopologyGuidedLocalPolicyConfig,
) -> str | None:
    """Return the first reason the near-parity alternative is ineligible."""
    if route_delta is None:
        return "missing_route_distance"
    if primary_clearance is None or alt_clearance is None:
        return "missing_static_clearance"
    absolute_slack = float(config.near_parity_route_distance_slack_m)
    ratio_slack = max(primary_route or 0.0, _EPS) * float(
        config.near_parity_route_distance_slack_ratio
    )
    route_near_parity = route_delta <= absolute_slack or route_delta <= ratio_slack
    clearance_ok = alt_clearance + float(config.near_parity_static_clearance_floor_m) >= (
        primary_clearance
    )
    if not route_near_parity:
        return "route_distance_exceeds_slack"
    if not clearance_ok:
        return "static_clearance_floor_failed"
    return None


def _annotate_near_parity_gate(
    config: TopologyGuidedLocalPolicyConfig,
    hypotheses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Annotate and optionally adjust hypotheses for the near-parity diversity gate.

    Returns:
        Top-level diagnostic fields for the selected topology decision.
    """
    primary = next(
        (item for item in hypotheses if str(item.get("hypothesis_id")) == "primary_route"),
        None,
    )
    alternatives = [
        item for item in hypotheses if str(item.get("hypothesis_id")) != "primary_route"
    ]
    diagnostic = {
        "near_parity_gate_enabled": bool(config.near_parity_diversity_gate_enabled),
        "near_parity_gate_reason": "disabled",
        "primary_vs_best_alternative_route_distance": None,
        "selected_static_clearance_min_m": None,
        "best_alternative_static_clearance_min_m": None,
    }
    for item in hypotheses:
        item["selection_score"] = float(item["score"])
        item["near_parity_gate_reason"] = "disabled"
        item["primary_vs_best_alternative_route_distance"] = None
        item["selected_static_clearance_min_m"] = None
        item["best_alternative_static_clearance_min_m"] = None
    best_alternative = (
        max(
            alternatives,
            key=lambda item: float(item.get("selection_score", item["score"])),
        )
        if alternatives
        else None
    )

    missing_reason = None
    if primary is None:
        missing_reason = "missing_primary_route"
    elif best_alternative is None:
        missing_reason = "no_alternative_hypothesis"
    if missing_reason is not None:
        diagnostic["near_parity_gate_reason"] = missing_reason
        return diagnostic

    primary_route = _finite_optional_float(primary.get("route_remaining_distance_m"))
    alt_route = _finite_optional_float(best_alternative.get("route_remaining_distance_m"))
    primary_clearance = _finite_optional_float(primary.get("static_clearance_min_m"))
    alt_clearance = _finite_optional_float(best_alternative.get("static_clearance_min_m"))
    if primary_route is not None and alt_route is not None:
        route_delta = float(alt_route - primary_route)
        diagnostic["primary_vs_best_alternative_route_distance"] = route_delta
    else:
        route_delta = None
    diagnostic["best_alternative_static_clearance_min_m"] = alt_clearance

    for item in hypotheses:
        item["primary_vs_best_alternative_route_distance"] = route_delta
        item["best_alternative_static_clearance_min_m"] = alt_clearance

    if not bool(config.near_parity_diversity_gate_enabled):
        return diagnostic
    blocker_reason = _near_parity_blocker_reason(
        route_delta=route_delta,
        primary_route=primary_route,
        primary_clearance=primary_clearance,
        alt_clearance=alt_clearance,
        config=config,
    )
    if blocker_reason is not None:
        diagnostic["near_parity_gate_reason"] = blocker_reason
        best_alternative["near_parity_gate_reason"] = blocker_reason
        return diagnostic

    best_alternative["selection_score"] = float(best_alternative["score"]) + float(
        config.near_parity_diversity_bonus
    )
    best_alternative["near_parity_gate_reason"] = "eligible_near_parity_alternative"
    if float(best_alternative["selection_score"]) > float(primary["selection_score"]):
        primary["selection_rejection_reason"] = "near_parity_diversity_gate"
    diagnostic["near_parity_gate_reason"] = "eligible_near_parity_alternative"
    return diagnostic


def _finalize_near_parity_gate_diagnostic(
    config: TopologyGuidedLocalPolicyConfig,
    hypotheses: list[dict[str, Any]],
    selected: dict[str, Any],
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Add selected-hypothesis near-parity fields after scoring.

    Returns:
        Updated top-level diagnostic mapping.
    """
    selected_clearance = selected.get("static_clearance_min_m")
    selected_non_primary = str(selected.get("hypothesis_id")) != "primary_route"
    eligible_gate = str(diagnostic["near_parity_gate_reason"]) == "eligible_near_parity_alternative"
    selected_score = _finite_optional_float(selected.get("selection_score", selected.get("score")))
    raw_score = _finite_optional_float(selected.get("score"))
    gate_boost_applied = (
        selected_score is not None and raw_score is not None and selected_score > raw_score
    )
    reason = (
        "selected_non_primary_near_parity"
        if bool(config.near_parity_diversity_gate_enabled)
        and selected_non_primary
        and eligible_gate
        and gate_boost_applied
        else str(diagnostic["near_parity_gate_reason"])
    )
    diagnostic["selected_static_clearance_min_m"] = selected_clearance
    diagnostic["near_parity_gate_reason"] = reason
    for item in hypotheses:
        item["selected_static_clearance_min_m"] = selected_clearance
        if item["hypothesis_id"] == selected["hypothesis_id"]:
            item["near_parity_gate_reason"] = reason
    return diagnostic


def _recent_primary_route_progress(
    config: TopologyGuidedLocalPolicyConfig,
    recent_primary_distances: list[float],
) -> float:
    """Return recent primary-route progress in metres from route-remaining samples.

    Two accounting modes are supported, both clamped to be non-negative:

    * Legacy (default): ``oldest_sample - newest_sample`` over the recent window.
      A single A* re-plan that transiently raises ``route_remaining`` can make
      this underestimate (clamp to ``0``) even while the route is advancing,
      which can let the reuse penalty fire while the primary route is in fact
      still progressing -- the premature-stall failure mode.
    * Monotone (opt-in via ``primary_route_progress_gate_use_monotone_accounting``):
      ``max_sample - newest_sample``. Using the largest observed remaining
      distance as the baseline makes the estimate robust to a single re-plan
      bump, so steady progress is not masked by transient noise.

    Returns:
        float: Non-negative recent route-progress estimate in metres.
    """
    if len(recent_primary_distances) < 2:
        return 0.0
    newest = float(recent_primary_distances[-1])
    if bool(config.primary_route_progress_gate_use_monotone_accounting):
        baseline = max(float(value) for value in recent_primary_distances)
    else:
        baseline = float(recent_primary_distances[0])
    return max(0.0, baseline - newest)


def _apply_primary_route_reuse_penalty(
    config: TopologyGuidedLocalPolicyConfig,
    hypotheses: list[dict[str, Any]],
    recent_primary_selections: deque[tuple[str, float | None]],
) -> dict[str, Any]:
    """Apply a reuse penalty to primary_route when eligible near-parity alternatives exist.

    When ``primary_route_progress_gate_enabled`` is true and recent primary-route
    progress satisfies the threshold, the penalty is suppressed to preserve the
    current primary route while it is still making meaningful progress.

    Returns:
        dict[str, Any]: Diagnostic fields for the reuse-penalty mechanism.
    """
    primary = next(
        (item for item in hypotheses if str(item.get("hypothesis_id")) == "primary_route"),
        None,
    )
    alternatives = [
        item for item in hypotheses if str(item.get("hypothesis_id")) != "primary_route"
    ]
    eligible_alternative = any(
        str(item.get("near_parity_gate_reason"))
        in {"eligible_near_parity_alternative", "selected_non_primary_near_parity"}
        for item in alternatives
    )
    recent_primary_distances = [
        dist
        for hid, dist in recent_primary_selections
        if hid == "primary_route" and dist is not None
    ]
    recent_progress_m = _recent_primary_route_progress(config, recent_primary_distances)
    min_samples = max(int(config.primary_route_progress_gate_min_samples), 2)
    progress_gate_satisfied = False
    if (
        bool(config.primary_route_progress_gate_enabled)
        and len(recent_primary_distances) >= min_samples
    ):
        progress_gate_satisfied = recent_progress_m >= float(
            config.primary_route_progress_gate_threshold_m
        )
    diagnostic: dict[str, Any] = {
        "reuse_penalty_applied": False,
        "reuse_penalty_reason": None,
        "recent_primary_selection_count": 0,
        "eligible_near_parity_alternative_exists": bool(eligible_alternative),
        "primary_route_recent_progress_m": recent_progress_m,
        "primary_route_recent_progress_sample_count": len(recent_primary_distances),
        "primary_route_recent_progress_min_samples": int(min_samples),
        "primary_route_progress_accounting_mode": (
            "monotone"
            if bool(config.primary_route_progress_gate_use_monotone_accounting)
            else "legacy"
        ),
        "primary_route_progress_gate_satisfied": bool(progress_gate_satisfied),
        "reuse_penalty_suppressed_by_progress": False,
    }
    if not bool(config.primary_route_reuse_penalty_enabled):
        return diagnostic
    recent_count = sum(1 for hid, _ in recent_primary_selections if hid == "primary_route")
    diagnostic["recent_primary_selection_count"] = recent_count
    min_prior = int(config.primary_route_reuse_penalty_min_prior_primary_selections)
    if (
        primary is not None
        and recent_count >= min_prior
        and eligible_alternative
        and len(recent_primary_selections) > 0
    ):
        if progress_gate_satisfied:
            diagnostic["reuse_penalty_suppressed_by_progress"] = True
            diagnostic["reuse_penalty_reason"] = (
                f"progress_gate_suppressed_recent_progress_{recent_progress_m:.3f}m_"
                f">=threshold_{float(config.primary_route_progress_gate_threshold_m):.3f}m"
            )
        else:
            penalty = float(config.primary_route_reuse_penalty_weight) * float(recent_count)
            primary["selection_score"] = (
                float(primary.get("selection_score", primary["score"])) - penalty
            )
            diagnostic["reuse_penalty_applied"] = True
            diagnostic["reuse_penalty_reason"] = (
                f"primary_route_selected_{recent_count}_times_in_last_"
                f"{len(recent_primary_selections)}_steps_with_eligible_near_parity_alternative"
            )
    return diagnostic


def _finite_float_field(raw: dict[str, Any], key: str, default: float) -> float:
    """Return a finite float config field, failing closed on non-finite input.

    Returns:
        float: The parsed finite value.

    Raises:
        ValueError: If the provided value is not a finite number.
    """
    try:
        value = float(raw.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Topology config '{key}' must be a finite number, got {raw.get(key)!r}."
        ) from exc
    if not np.isfinite(value):
        raise ValueError(f"Topology config '{key}' must be finite, got {value!r}.")
    return value


def _progress_gate_min_samples(raw: dict[str, Any]) -> int:
    """Return the validated minimum-samples threshold for the progress gate.

    Returns:
        int: The minimum number of recent primary-route samples (>= 2).

    Raises:
        ValueError: If the value is not an integer, is non-finite, or is below 2.
    """
    value = raw.get("primary_route_progress_gate_min_samples", _DEFAULT_PROGRESS_GATE_MIN_SAMPLES)
    try:
        samples = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Topology config 'primary_route_progress_gate_min_samples' must be an integer, "
            f"got {value!r}."
        ) from exc
    if samples < 2:
        raise ValueError(
            "Topology config 'primary_route_progress_gate_min_samples' must be >= 2 "
            f"(a finite difference needs two samples), got {samples}."
        )
    return samples


def _non_negative_int_field(raw: dict[str, Any], key: str, default: int) -> int:
    """Return a validated non-negative integer topology config field."""
    value = raw.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Topology config '{key}' must be an integer, got {value!r}.") from exc
    if parsed < 0:
        raise ValueError(f"Topology config '{key}' must be non-negative, got {parsed}.")
    return parsed


def _bounded_unit_float_field(raw: dict[str, Any], key: str, default: float) -> float:
    """Return a finite float constrained to the inclusive unit interval."""
    value = _finite_float_field(raw, key, default)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Topology config '{key}' must be in [0, 1], got {value}.")
    return value


def _non_negative_float_field(raw: dict[str, Any], key: str, default: float) -> float:
    """Return a finite non-negative float topology config field."""
    value = _finite_float_field(raw, key, default)
    if value < 0.0:
        raise ValueError(f"Topology config '{key}' must be non-negative, got {value}.")
    return value


def _near_parity_threshold_metadata(config: TopologyGuidedLocalPolicyConfig) -> dict[str, Any]:
    """Return explicit near-parity threshold metadata for diagnostics."""
    return {
        "schema_version": "topology_near_parity_thresholds.v1",
        "enabled": bool(config.near_parity_diversity_gate_enabled),
        "route_distance_slack_m": float(config.near_parity_route_distance_slack_m),
        "route_distance_slack_ratio": float(config.near_parity_route_distance_slack_ratio),
        "static_clearance_floor_m": float(config.near_parity_static_clearance_floor_m),
        "diversity_bonus": float(config.near_parity_diversity_bonus),
        "deterministic_tie_policy": "stable_first_max_score",
    }


def _topology_guided_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Return flat topology config with optional nested ``topology_guided`` overrides applied."""
    payload = dict(raw)
    nested = deepcopy(raw.get("topology_guided"))
    if nested is None:
        return payload
    if not isinstance(nested, dict):
        raise ValueError("Topology config 'topology_guided' must be a mapping when provided.")
    payload.update(nested)
    payload["topology_guided"] = nested
    if "near_parity_margin" in nested and "near_parity_route_distance_slack_ratio" not in raw:
        payload["near_parity_route_distance_slack_ratio"] = nested["near_parity_margin"]
    return payload


def blend_topology_command(
    baseline_cmd: tuple[float, float],
    topology_cmd: tuple[float, float],
    *,
    weight: float,
    command_limits: dict[str, float],
) -> tuple[float, float]:
    """Blend baseline and topology commands with finite bounded arbitration weight.

    Returns:
        Bounded ``(linear, angular)`` command after explicit topology arbitration.
    """
    try:
        blend_weight = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Topology arbitration weight must be finite, got {weight!r}.") from exc
    if not np.isfinite(blend_weight) or blend_weight < 0.0 or blend_weight > 1.0:
        raise ValueError(f"Topology arbitration weight must be in [0, 1], got {weight!r}.")
    baseline = np.asarray(baseline_cmd, dtype=float).reshape(2)
    topology = np.asarray(topology_cmd, dtype=float).reshape(2)
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(topology)):
        raise ValueError("Topology command arbitration requires finite commands.")
    blended = (1.0 - blend_weight) * baseline + blend_weight * topology
    max_linear = max(float(command_limits.get("max_linear", 0.0)), 0.0)
    max_angular = max(float(command_limits.get("max_angular", 0.0)), 0.0)
    return (
        float(np.clip(blended[0], 0.0, max_linear)),
        float(np.clip(blended[1], -max_angular, max_angular)),
    )


def build_topology_guided_local_policy_config(
    cfg: dict[str, Any] | None,
) -> TopologyGuidedLocalPolicyConfig:
    """Build the topology-guided local policy config from a YAML-style mapping.

    Returns:
        Parsed topology-guided local policy configuration.

    Raises:
        ValueError: If a numeric progress-gate field is non-finite, or if
            ``primary_route_progress_gate_min_samples`` is below the minimum of 2.
    """
    raw = _topology_guided_payload(dict(cfg or {}) if isinstance(cfg, dict) else {})
    hybrid_payload = {key: value for key, value in raw.items() if key not in _TOPOLOGY_KEYS}
    route_payload = deepcopy(raw.get("route_hypothesis") or {})
    if not isinstance(route_payload, dict):
        route_payload = {}
    route_payload.setdefault(
        "waypoint_lookahead_cells",
        raw.get("route_guide_waypoint_lookahead_cells", 8),
    )
    route_payload.setdefault(
        "obstacle_inflation_cells",
        raw.get("route_guide_obstacle_inflation_cells", 3),
    )
    route_payload.setdefault(
        "clearance_penalty_weight",
        raw.get("route_guide_clearance_penalty_weight", 0.5),
    )
    return TopologyGuidedLocalPolicyConfig(
        hybrid_rule=build_hybrid_rule_local_planner_config(hybrid_payload),
        route_hypothesis=build_grid_route_config(route_payload),
        diagnostic_only=bool(raw.get("diagnostic_only", True)),
        claim_boundary=str(raw.get("claim_boundary", "diagnostic_only")),
        schema_version=str(raw.get("schema_version", "topology_guided_hybrid_rule.v1")),
        enabled=bool(raw.get("enabled", True)),
        candidate_required=bool(raw.get("candidate_required", False)),
        fallback_on_no_candidate=bool(raw.get("fallback_on_no_candidate", True)),
        arbitration_weight=_bounded_unit_float_field(
            raw, "arbitration_weight", _DEFAULT_TOPOLOGY_ARBITRATION_WEIGHT
        ),
        min_route_progress_delta_m=_non_negative_float_field(
            raw, "min_route_progress_delta_m", 0.05
        ),
        stall_window_steps=_non_negative_int_field(raw, "stall_window_steps", 20),
        min_hypotheses=int(raw.get("min_hypotheses", 2)),
        max_hypotheses=int(raw.get("max_hypotheses", 2)),
        block_radius_cells=int(raw.get("block_radius_cells", 3)),
        block_stride_cells=int(raw.get("block_stride_cells", 8)),
        max_path_overlap=float(raw.get("max_path_overlap", 0.88)),
        length_weight=float(raw.get("length_weight", 1.0)),
        static_clearance_weight=float(raw.get("static_clearance_weight", 0.6)),
        fail_closed_on_missing_inputs=bool(raw.get("fail_closed_on_missing_inputs", True)),
        fail_closed_on_insufficient_hypotheses=bool(
            raw.get("fail_closed_on_insufficient_hypotheses", False)
        ),
        topology_command_enabled=bool(
            raw.get("topology_command_enabled", raw.get("enabled", True))
        ),
        topology_command_speed=float(raw.get("topology_command_speed", 0.35)),
        topology_command_heading_gain=float(raw.get("topology_command_heading_gain", 1.0)),
        topology_command_turn_in_place_error=float(
            raw.get("topology_command_turn_in_place_error", 0.25)
        ),
        near_parity_diversity_gate_enabled=bool(
            raw.get("near_parity_diversity_gate_enabled", False)
        ),
        near_parity_route_distance_slack_m=float(
            raw.get("near_parity_route_distance_slack_m", 0.75)
        ),
        near_parity_route_distance_slack_ratio=float(
            raw.get("near_parity_route_distance_slack_ratio", 0.05)
        ),
        near_parity_static_clearance_floor_m=float(
            raw.get("near_parity_static_clearance_floor_m", 0.05)
        ),
        near_parity_diversity_bonus=float(raw.get("near_parity_diversity_bonus", 0.0)),
        primary_route_reuse_penalty_enabled=bool(
            raw.get("primary_route_reuse_penalty_enabled", False)
        ),
        primary_route_reuse_penalty_weight=float(
            raw.get("primary_route_reuse_penalty_weight", 1.0)
        ),
        primary_route_reuse_penalty_cooldown_steps=int(
            raw.get("primary_route_reuse_penalty_cooldown_steps", 3)
        ),
        primary_route_reuse_penalty_min_prior_primary_selections=int(
            raw.get("primary_route_reuse_penalty_min_prior_primary_selections", 2)
        ),
        primary_route_progress_gate_enabled=bool(
            raw.get("primary_route_progress_gate_enabled", False)
        ),
        primary_route_progress_gate_threshold_m=_finite_float_field(
            raw, "primary_route_progress_gate_threshold_m", 0.0
        ),
        primary_route_progress_gate_min_samples=_progress_gate_min_samples(raw),
        primary_route_progress_gate_use_monotone_accounting=bool(
            raw.get("primary_route_progress_gate_use_monotone_accounting", False)
        ),
    )


class TopologyGuidedHybridRulePlannerAdapter(HybridRuleLocalPlannerAdapter):
    """Hybrid-rule planner that selects among diagnostic topology hypotheses."""

    def __init__(self, config: TopologyGuidedLocalPolicyConfig | None = None) -> None:
        """Initialize the wrapped hybrid-rule scorer and route-hypothesis generator."""
        self.topology_config = config or build_topology_guided_local_policy_config({})
        super().__init__(self.topology_config.hybrid_rule)
        self._route_hypothesis = GridRoutePlannerAdapter(self.topology_config.route_hypothesis)
        self._topology_status_counts: Counter[str] = Counter()
        self._selected_hypothesis_counts: Counter[str] = Counter()
        self._last_topology_decision: dict[str, Any] | None = None
        self._last_topology_command_influence: dict[str, Any] | None = None
        self._recent_primary_selections: deque[tuple[str, float | None]] = deque(
            maxlen=max(int(self.topology_config.primary_route_reuse_penalty_cooldown_steps), 1)
        )
        self._total_primary_selections: int = 0
        self._topology_route_progress = RouteProgressState()

    def reset(self, *, seed: int | None = None) -> None:
        """Reset base planner state and topology-hypothesis diagnostics."""
        super().reset(seed=seed)
        self._topology_status_counts = Counter()
        self._selected_hypothesis_counts = Counter()
        self._last_topology_decision = None
        self._last_topology_command_influence = None
        self._recent_primary_selections = deque(
            maxlen=max(int(self.topology_config.primary_route_reuse_penalty_cooldown_steps), 1)
        )
        self._total_primary_selections = 0
        self._topology_route_progress = RouteProgressState()

    def _update_topology_route_progress(
        self,
        *,
        selected_id: str,
        route_remaining_m: float | None,
    ) -> dict[str, Any]:
        """Update route-progress state and classify stalls versus near-parity churn.

        Returns:
            dict[str, Any]: Serializable progress-state metadata for this planner step.
        """
        state = self._topology_route_progress
        previous_remaining = state.last_route_progress_m
        previous_selected = state.last_selected_candidate
        switched = previous_selected is not None and selected_id != previous_selected
        if switched:
            state.candidate_switch_count += 1

        progress_delta: float | None = None
        if route_remaining_m is None:
            terminal_reason = "missing_route_progress"
        elif previous_remaining is None:
            state.last_route_progress_m = float(route_remaining_m)
            terminal_reason = "insufficient_samples"
        else:
            progress_delta = float(previous_remaining - route_remaining_m)
            state.last_route_progress_m = float(route_remaining_m)
            if progress_delta >= float(self.topology_config.min_route_progress_delta_m):
                state.stagnant_steps = 0
                terminal_reason = "goal_progress"
            elif switched:
                state.stagnant_steps = 0
                terminal_reason = "near_parity_churn"
            else:
                state.stagnant_steps += 1
                terminal_reason = (
                    "true_stall"
                    if state.stagnant_steps >= int(self.topology_config.stall_window_steps)
                    else "route_stagnant"
                )

        state.last_selected_candidate = selected_id
        return {
            "schema_version": "topology_route_progress_state.v1",
            "selected_hypothesis_id": selected_id,
            "previous_selected_hypothesis_id": previous_selected,
            "candidate_switched": bool(switched),
            "candidate_switch_count": int(state.candidate_switch_count),
            "route_remaining_distance_m": route_remaining_m,
            "previous_route_remaining_distance_m": previous_remaining,
            "route_progress_delta_m": progress_delta,
            "min_route_progress_delta_m": float(self.topology_config.min_route_progress_delta_m),
            "stagnant_steps": int(state.stagnant_steps),
            "stall_window_steps": int(self.topology_config.stall_window_steps),
            "terminal_reason": terminal_reason,
        }

    def _alternative_paths(
        self,
        blocked: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[_RouteHypothesis]:
        """Return the primary route plus distinct masked-route alternatives."""
        clearance_map = self._route_hypothesis._compute_clearance_map(blocked)
        base_path = self._route_hypothesis._astar(blocked, start, goal, clearance_map=clearance_map)
        if len(base_path) < 2:
            return []
        hypotheses = [
            _RouteHypothesis(
                hypothesis_id="primary_route",
                path=base_path,
                clearance_map=clearance_map,
            )
        ]
        protected = set(base_path[: max(2, len(base_path) // 12)])
        protected.update(base_path[-max(2, len(base_path) // 12) :])
        stride = max(int(self.topology_config.block_stride_cells), 1)
        for idx in range(max(2, stride), max(len(base_path) - 2, 0), stride):
            if len(hypotheses) >= max(int(self.topology_config.max_hypotheses), 1):
                break
            blocked_cell = base_path[idx]
            if blocked_cell in protected:
                continue
            perturbed = _block_path_cell(
                blocked,
                blocked_cell,
                radius=int(self.topology_config.block_radius_cells),
                protected=protected,
            )
            alt_clearance = self._route_hypothesis._compute_clearance_map(perturbed)
            path = self._route_hypothesis._astar(
                perturbed,
                start,
                goal,
                clearance_map=alt_clearance,
            )
            if len(path) < 2:
                continue
            if any(
                _path_overlap(path, item.path) > float(self.topology_config.max_path_overlap)
                for item in hypotheses
            ):
                continue
            hypotheses.append(
                _RouteHypothesis(
                    hypothesis_id=f"masked_cell_{blocked_cell[0]}_{blocked_cell[1]}",
                    path=path,
                    clearance_map=alt_clearance,
                    blocked_cell=blocked_cell,
                )
            )
        return hypotheses

    def _hypotheses_for_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return selectable route-corridor hypotheses for the current observation."""
        try:
            robot_pos, heading, goal, radius = self._route_hypothesis._extract_state(observation)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return {"status": "not_available", "reason": "missing_robot_or_goal_state"}
        payload = self._route_hypothesis._extract_grid_payload(observation)
        if payload is None:
            return {"status": "not_available", "reason": "missing_occupancy_grid"}
        grid, meta = payload
        blocked = self._route_hypothesis._blocked_grid(grid, meta, radius)
        if blocked is None:
            return {"status": "not_available", "reason": "missing_static_obstacle_channel"}
        start_rc = self._route_hypothesis._world_to_grid(robot_pos, meta, blocked.shape)
        goal_rc = self._route_hypothesis._world_to_grid(goal, meta, blocked.shape)
        if start_rc is None or goal_rc is None:
            return {"status": "not_available", "reason": "route_endpoint_outside_grid"}
        start = self._route_hypothesis._nearest_free(
            blocked,
            start_rc,
            int(self.topology_config.route_hypothesis.clearance_search_cells),
        )
        stop = self._route_hypothesis._nearest_free(
            blocked,
            goal_rc,
            int(self.topology_config.route_hypothesis.clearance_search_cells),
        )
        if start is None or stop is None:
            return {"status": "not_available", "reason": "no_free_route_endpoint"}

        route_paths = self._alternative_paths(blocked, start, stop)
        if len(route_paths) < int(self.topology_config.min_hypotheses):
            return {
                "status": "insufficient_hypotheses",
                "reason": "fewer_than_min_distinct_routes",
                "hypothesis_count": len(route_paths),
                "min_hypotheses": int(self.topology_config.min_hypotheses),
            }

        resolution = _first_float(meta.get("resolution"), 0.2)
        hypotheses: list[dict[str, Any]] = []
        for rank, route_path in enumerate(route_paths[: int(self.topology_config.max_hypotheses)]):
            geometry = self._route_hypothesis._route_geometry_from_path(
                path=route_path.path,
                clearance_map=route_path.clearance_map,
                meta=meta,
                robot_pos=robot_pos,
                heading=heading,
            )
            clearance = _static_clearance_summary(
                route_path.path,
                route_path.clearance_map,
                resolution=resolution,
            )
            route_remaining = _path_length(route_path.path, resolution=resolution)
            static_min = clearance["static_clearance_min_m"]
            score_payload = _topology_score_payload(
                self.topology_config,
                route_remaining=route_remaining,
                static_clearance_min=static_min,
            )
            hypotheses.append(
                {
                    "hypothesis_id": route_path.hypothesis_id,
                    "rank": int(rank),
                    "blocked_cell": list(route_path.blocked_cell)
                    if route_path.blocked_cell is not None
                    else None,
                    "path_cell_count": len(route_path.path),
                    **score_payload,
                    "route_remaining_distance_m": float(route_remaining),
                    **clearance,
                    "route_corridor": geometry,
                }
            )
        if not hypotheses:
            return {
                "status": "insufficient_hypotheses",
                "reason": "no_hypotheses_available",
                "hypothesis_count": 0,
                "min_hypotheses": int(self.topology_config.min_hypotheses),
            }
        near_parity = _annotate_near_parity_gate(self.topology_config, hypotheses)
        selected = max(
            hypotheses,
            key=lambda item: float(item.get("selection_score", item["score"])),
        )
        near_parity = _finalize_near_parity_gate_diagnostic(
            self.topology_config,
            hypotheses,
            selected,
            near_parity,
        )
        reuse_penalty_diagnostic = _apply_primary_route_reuse_penalty(
            self.topology_config,
            hypotheses,
            self._recent_primary_selections,
        )
        selected = max(
            hypotheses,
            key=lambda item: float(item.get("selection_score", item["score"])),
        )
        _annotate_topology_selection(hypotheses, selected=selected)
        selected_id = str(selected["hypothesis_id"])
        if selected_id == "primary_route":
            self._total_primary_selections += 1
        route_remaining = _finite_optional_float(selected.get("route_remaining_distance_m"))
        self._recent_primary_selections.append((selected_id, route_remaining))
        return {
            "status": "ok",
            "reason": "selected_scored_route_hypothesis",
            "hypothesis_count": len(hypotheses),
            "selected_hypothesis_id": selected_id,
            "selected_rank": int(selected["rank"]),
            "selected_score": float(selected["score"]),
            "selection_score": float(selected.get("selection_score", selected["score"])),
            **near_parity,
            **reuse_penalty_diagnostic,
            "hypotheses": hypotheses,
        }

    def _route_corridor_diagnostics(
        self,
        observation: dict[str, Any],
        *,
        current_time: float,
    ) -> dict[str, Any] | None:
        """Select a topology hypothesis and expose it as route-corridor geometry.

        Returns:
            dict[str, Any] | None: Selected route-corridor payload, or ``None``
            when the diagnostic fail-closed gate is unavailable.
        """
        topology = self._hypotheses_for_observation(observation)
        status = str(topology.get("status", "unknown"))
        self._topology_status_counts[status] += 1
        self._last_topology_decision = topology
        if status != "ok":
            return None
        selected_id = str(topology["selected_hypothesis_id"])
        self._selected_hypothesis_counts[selected_id] += 1
        selected = next(
            item for item in topology["hypotheses"] if str(item["hypothesis_id"]) == selected_id
        )
        route_corridor = dict(selected["route_corridor"])
        route_remaining = route_corridor.get("route_remaining_distance")
        if isinstance(route_remaining, int | float | np.integer | np.floating) and np.isfinite(
            route_remaining
        ):
            remaining = float(route_remaining)
            self._route_distance_history.append((current_time, remaining))
            route_corridor["route_arc_progress_windows"] = self._distance_progress_windows(
                self._route_distance_history,
                current_time=current_time,
                current_distance=remaining,
            )
        else:
            remaining = None
        route_corridor["topology_route_progress"] = self._update_topology_route_progress(
            selected_id=selected_id,
            route_remaining_m=remaining,
        )
        route_corridor["topology_hypothesis"] = {
            key: value for key, value in selected.items() if key != "route_corridor"
        }
        route_corridor["topology_hypotheses"] = [
            {key: value for key, value in item.items() if key != "route_corridor"}
            for item in topology["hypotheses"]
        ]
        route_corridor["topology_reuse_penalty"] = {
            "reuse_penalty_applied": topology.get("reuse_penalty_applied", False),
            "reuse_penalty_reason": topology.get("reuse_penalty_reason"),
            "recent_primary_selection_count": topology.get("recent_primary_selection_count", 0),
            "eligible_near_parity_alternative_exists": topology.get(
                "eligible_near_parity_alternative_exists", False
            ),
            "primary_route_recent_progress_m": topology.get("primary_route_recent_progress_m", 0.0),
            "primary_route_recent_progress_sample_count": topology.get(
                "primary_route_recent_progress_sample_count", 0
            ),
            "primary_route_recent_progress_min_samples": topology.get(
                "primary_route_recent_progress_min_samples", _DEFAULT_PROGRESS_GATE_MIN_SAMPLES
            ),
            "primary_route_progress_accounting_mode": topology.get(
                "primary_route_progress_accounting_mode", "legacy"
            ),
            "primary_route_progress_gate_satisfied": topology.get(
                "primary_route_progress_gate_satisfied", False
            ),
            "reuse_penalty_suppressed_by_progress": topology.get(
                "reuse_penalty_suppressed_by_progress", False
            ),
        }
        route_corridor["topology_status"] = "ok"
        route_corridor["topology_guided_config"] = {
            "schema_version": self.topology_config.schema_version,
            "enabled": bool(self.topology_config.enabled),
            "diagnostic_only": bool(self.topology_config.diagnostic_only),
            "claim_boundary": self.topology_config.claim_boundary,
            "candidate_required": bool(self.topology_config.candidate_required),
            "fallback_on_no_candidate": bool(self.topology_config.fallback_on_no_candidate),
            "arbitration_weight": float(self.topology_config.arbitration_weight),
            "min_route_progress_delta_m": float(self.topology_config.min_route_progress_delta_m),
            "stall_window_steps": int(self.topology_config.stall_window_steps),
            "near_parity_thresholds": _near_parity_threshold_metadata(self.topology_config),
        }
        return route_corridor

    def _candidate_source_priority(self, source: str) -> int:
        """Prefer explicit topology-hypothesis candidates over duplicate generic commands.

        Returns:
            int: Candidate-source priority used during duplicate-command collapse.
        """
        if source == "topology_hypothesis":
            return 35
        return super()._candidate_source_priority(source)

    def _topology_hypothesis_candidate(
        self,
        *,
        state: dict[str, Any],
        speed_cap: float,
        route_corridor: dict[str, Any] | None,
        bounds: tuple[float, float, float, float],
    ) -> HybridRuleCandidate | None:
        """Return a bounded command that tracks the selected topology hypothesis."""
        if not bool(self.topology_config.enabled) or not bool(
            self.topology_config.topology_command_enabled
        ):
            return None
        if not isinstance(route_corridor, dict) or route_corridor.get("topology_status") != "ok":
            return None
        waypoint = self._route_point(route_corridor, "route_waypoint_world")
        tangent_heading = self._route_tangent_heading(route_corridor)
        if waypoint is None or tangent_heading is None:
            return None

        v_min, v_max, w_min, w_max = bounds
        robot_pos = state["robot_pos"]
        heading = float(state["heading"])
        waypoint_vec = waypoint - robot_pos
        waypoint_heading = (
            tangent_heading
            if float(np.linalg.norm(waypoint_vec)) <= _EPS
            else float(np.arctan2(waypoint_vec[1], waypoint_vec[0]))
        )
        tangent_error = float((tangent_heading - heading + np.pi) % (2.0 * np.pi) - np.pi)
        turn_in_place_error = max(
            float(self.topology_config.topology_command_turn_in_place_error), 0.0
        )
        if abs(tangent_error) >= turn_in_place_error:
            desired_heading_error = tangent_error
            desired_linear = 0.0
        else:
            blended_heading = float(
                np.arctan2(
                    0.5 * np.sin(tangent_heading) + 0.5 * np.sin(waypoint_heading),
                    0.5 * np.cos(tangent_heading) + 0.5 * np.cos(waypoint_heading),
                )
            )
            desired_heading_error = float(
                (blended_heading - heading + np.pi) % (2.0 * np.pi) - np.pi
            )
            alignment = max(0.0, float(np.cos(desired_heading_error)))
            desired_linear = min(
                float(speed_cap),
                float(self.topology_config.topology_command_speed),
                float(self.config.max_linear_speed),
            )
            desired_linear *= alignment
        desired_angular = float(
            np.clip(
                float(self.topology_config.topology_command_heading_gain)
                * desired_heading_error
                / max(float(self.config.control_period), _EPS),
                w_min,
                w_max,
            )
        )
        linear = float(np.clip(desired_linear, v_min, v_max))
        rollout_sequence = ((float(self.config.rollout_horizon), linear, desired_angular),)
        return HybridRuleCandidate(
            linear,
            desired_angular,
            "topology_hypothesis",
            rollout_sequence,
        )

    def _generate_candidates(
        self,
        state: dict[str, Any],
        speed_cap: float,
        *,
        route_corridor: dict[str, Any] | None = None,
        corridor_subgoal: dict[str, Any] | None = None,
        goal_posterior: dict[str, Any] | None = None,
    ) -> list[HybridRuleCandidate]:
        """Add a selected-hypothesis command to the base hybrid-rule candidate set.

        Returns:
            list[HybridRuleCandidate]: De-duplicated local command candidates.
        """
        self._last_topology_command_influence = None
        candidates = super()._generate_candidates(
            state,
            speed_cap,
            route_corridor=route_corridor,
            corridor_subgoal=corridor_subgoal,
            goal_posterior=goal_posterior,
        )
        topology_candidate = self._topology_hypothesis_candidate(
            state=state,
            speed_cap=speed_cap,
            route_corridor=route_corridor,
            bounds=self._dynamic_window(state["current_speed"], speed_cap),
        )
        if topology_candidate is not None:
            if candidates:
                baseline = candidates[0]
                command_limits = {
                    "max_linear": float(speed_cap),
                    "max_angular": float(self.config.max_angular_speed),
                }
                arbitration_weight = float(self.topology_config.arbitration_weight)
                raw_blended_linear = (1.0 - arbitration_weight) * float(
                    baseline.linear
                ) + arbitration_weight * float(topology_candidate.linear)
                raw_blended_angular = (1.0 - arbitration_weight) * float(
                    baseline.angular
                ) + arbitration_weight * float(topology_candidate.angular)
                blended_linear, blended_angular = blend_topology_command(
                    (baseline.linear, baseline.angular),
                    (topology_candidate.linear, topology_candidate.angular),
                    weight=arbitration_weight,
                    command_limits=command_limits,
                )
                selected_hypothesis_id = None
                if isinstance(route_corridor, dict):
                    hypothesis = route_corridor.get("topology_hypothesis")
                    if isinstance(hypothesis, dict):
                        selected_hypothesis_id = hypothesis.get("hypothesis_id")
                self._last_topology_command_influence = {
                    "schema_version": "topology-command-influence.v1",
                    "source": "topology_hypothesis",
                    "arbitration_weight": arbitration_weight,
                    "selected_hypothesis_id": selected_hypothesis_id,
                    "baseline_command": [float(baseline.linear), float(baseline.angular)],
                    "topology_command": [
                        float(topology_candidate.linear),
                        float(topology_candidate.angular),
                    ],
                    "raw_blended_command": [
                        float(raw_blended_linear),
                        float(raw_blended_angular),
                    ],
                    "projected_command": [float(blended_linear), float(blended_angular)],
                    "command_limits": command_limits,
                    "projection_applied": not (
                        np.isclose(raw_blended_linear, blended_linear)
                        and np.isclose(raw_blended_angular, blended_angular)
                    ),
                    "linear_delta_from_baseline": float(blended_linear - baseline.linear),
                    "angular_delta_from_baseline": float(blended_angular - baseline.angular),
                }
                topology_candidate = HybridRuleCandidate(
                    blended_linear,
                    blended_angular,
                    "topology_hypothesis",
                    ((float(self.config.rollout_horizon), blended_linear, blended_angular),),
                )
            candidates.append(topology_candidate)

        unique: dict[tuple[Any, ...], HybridRuleCandidate] = {}
        for candidate in candidates:
            clipped = self._clip_candidate(candidate, speed_cap=speed_cap)
            key = self._candidate_key(clipped)
            existing = unique.get(key)
            if existing is None or self._candidate_source_priority(
                clipped.source
            ) > self._candidate_source_priority(existing.source):
                unique[key] = clipped
        return list(unique.values())

    def _corridor_subgoal_score_terms(
        self,
        *,
        candidate: HybridRuleCandidate,
        route_corridor: dict[str, Any] | None,
        state: dict[str, Any],
        end_pos: np.ndarray,
        end_heading: float,
        min_static_clearance: float,
        hard_static_clearance: float,
    ) -> dict[str, float]:
        """Score topology-hypothesis commands with the existing route-corridor terms.

        Returns:
            dict[str, float]: Route-corridor score terms for candidate selection.
        """
        if candidate.source != "topology_hypothesis":
            return super()._corridor_subgoal_score_terms(
                candidate=candidate,
                route_corridor=route_corridor,
                state=state,
                end_pos=end_pos,
                end_heading=end_heading,
                min_static_clearance=min_static_clearance,
                hard_static_clearance=hard_static_clearance,
            )
        proxy = HybridRuleCandidate(
            candidate.linear,
            candidate.angular,
            "corridor_subgoal",
            candidate.rollout_sequence,
        )
        return super()._corridor_subgoal_score_terms(
            candidate=proxy,
            route_corridor=route_corridor,
            state=state,
            end_pos=end_pos,
            end_heading=end_heading,
            min_static_clearance=min_static_clearance,
            hard_static_clearance=hard_static_clearance,
        )

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return a command, stopping fail-closed when topology hypotheses are unavailable.

        Returns:
            tuple[float, float]: Selected linear and angular velocity.
        """
        self._last_topology_decision = None
        command = super().plan(observation)
        topology = self._last_topology_decision or {}
        status = str(topology.get("status", "unknown"))
        fail_closed = (
            (status == "not_available" and bool(self.topology_config.fail_closed_on_missing_inputs))
            or (
                status == "insufficient_hypotheses"
                and bool(self.topology_config.fail_closed_on_insufficient_hypotheses)
            )
            or (
                status != "ok"
                and bool(self.topology_config.candidate_required)
                and not bool(self.topology_config.fallback_on_no_candidate)
            )
        )
        if fail_closed:
            self._last_command = (0.0, 0.0)
            if self._last_decision:
                self._last_decision["planner_mode"] = "TOPOLOGY_FAIL_CLOSED"
                self._last_decision["selected_command"] = [0.0, 0.0]
                self._last_decision["selected_source"] = "topology_fail_closed"
                self._last_decision["topology_guided"] = topology
                self._last_decision["topology_lane_status"] = "failed"
            return (0.0, 0.0)
        if self._last_decision:
            self._last_decision["topology_guided"] = topology
            if status != "ok":
                self._last_decision["topology_lane_status"] = "fallback_only"
                self._last_decision["topology_fallback_status"] = status
                self._last_decision["topology_fallback_reason"] = topology.get("reason")
            self._last_decision["topology_guided_config"] = {
                "schema_version": self.topology_config.schema_version,
                "diagnostic_only": bool(self.topology_config.diagnostic_only),
                "claim_boundary": self.topology_config.claim_boundary,
                "candidate_required": bool(self.topology_config.candidate_required),
                "fallback_on_no_candidate": bool(self.topology_config.fallback_on_no_candidate),
                "arbitration_weight": float(self.topology_config.arbitration_weight),
                "min_route_progress_delta_m": float(
                    self.topology_config.min_route_progress_delta_m
                ),
                "stall_window_steps": int(self.topology_config.stall_window_steps),
            }
            if self._last_decision.get("selected_source") == "topology_hypothesis":
                influence = deepcopy(self._last_topology_command_influence or {})
                influence.update(
                    {
                        "source": "topology_hypothesis",
                        "reason": "selected_hypothesis_route_command_won_safety_scoring",
                        "selected_hypothesis_id": topology.get("selected_hypothesis_id"),
                        "selected_score": self._last_decision.get("selected_score"),
                        "selected_terms": self._last_decision.get("selected_terms", {}),
                    }
                )
                self._last_decision["topology_command_influence"] = influence
        return command

    def diagnostics(self) -> dict[str, Any]:
        """Return aggregate base-planner and topology-selection diagnostics."""
        diagnostics = super().diagnostics()
        diagnostics["topology_guided"] = {
            "diagnostic_only": bool(self.topology_config.diagnostic_only),
            "claim_boundary": self.topology_config.claim_boundary,
            "min_hypotheses": int(self.topology_config.min_hypotheses),
            "max_hypotheses": int(self.topology_config.max_hypotheses),
            "topology_command_enabled": bool(self.topology_config.topology_command_enabled),
            "arbitration_weight": float(self.topology_config.arbitration_weight),
            "route_progress_state": {
                "schema_version": "topology_route_progress_state.v1",
                "last_route_progress_m": self._topology_route_progress.last_route_progress_m,
                "stagnant_steps": int(self._topology_route_progress.stagnant_steps),
                "last_selected_candidate": self._topology_route_progress.last_selected_candidate,
                "candidate_switch_count": int(self._topology_route_progress.candidate_switch_count),
            },
            "near_parity_thresholds": _near_parity_threshold_metadata(self.topology_config),
            "status_counts": dict(sorted(self._topology_status_counts.items())),
            "selected_hypothesis_counts": dict(sorted(self._selected_hypothesis_counts.items())),
            "last_topology_decision": self._last_topology_decision,
        }
        return diagnostics


__all__ = [
    "RouteProgressState",
    "TopologyGuidedHybridRulePlannerAdapter",
    "TopologyGuidedLocalPolicyConfig",
    "blend_topology_command",
    "build_topology_guided_local_policy_config",
]
