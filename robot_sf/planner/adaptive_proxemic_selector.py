"""Diagnostic selector over fixed proxemic hybrid-rule profiles."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    HybridRuleLocalPlannerConfig,
    build_hybrid_rule_local_planner_config,
)

_PROFILE_ORDER = ("conservative", "neutral", "open")
_DEFAULT_PROFILE_BASE: dict[str, Any] = {
    "allow_testing_algorithms": True,
    "planner_variant": "hybrid_rule_v3_teb_like_rollout",
    "route_guide_enabled": True,
}
_DEFAULT_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "conservative": {
        "desired_dynamic_clearance": 1.25,
        "dynamic_clearance_weight": 2.4,
        "ttc_weight": 1.2,
        "stop_distance_human": 0.65,
        "slow_distance_human": 1.25,
        "moderate_distance_human": 2.6,
        "moderate_speed": 0.45,
        "near_human_angular_limit_distance": 1.0,
        "near_human_max_angular_speed": 0.55,
    },
    "neutral": {
        "desired_dynamic_clearance": 0.9,
        "dynamic_clearance_weight": 1.8,
        "ttc_weight": 0.8,
        "stop_distance_human": 0.5,
        "slow_distance_human": 1.0,
        "moderate_distance_human": 2.0,
        "moderate_speed": 0.6,
        "near_human_angular_limit_distance": 0.8,
        "near_human_max_angular_speed": 0.7,
    },
    "open": {
        "desired_dynamic_clearance": 0.7,
        "dynamic_clearance_weight": 1.1,
        "ttc_weight": 0.5,
        "stop_distance_human": 0.4,
        "slow_distance_human": 0.75,
        "moderate_distance_human": 1.4,
        "moderate_speed": 0.85,
        "near_human_angular_limit_distance": 0.6,
        "near_human_max_angular_speed": 0.9,
    },
}
_SOURCE_CANDIDATES = {
    "conservative": "proxemic_profile_conservative_issue_1676",
    "neutral": "proxemic_profile_neutral_issue_1676",
    "open": "proxemic_profile_open_issue_1676",
}
_HYBRID_RULE_CONFIG_KEYS = {field.name for field in fields(HybridRuleLocalPlannerConfig)}
_SELECTOR_KEYS = {
    "diagnostic_only",
    "claim_boundary",
    "selector",
    "profile_base",
    "profiles",
}


@dataclass(frozen=True)
class ProxemicProfileSpec:
    """One selectable fixed proxemic profile."""

    name: str
    source_candidate: str
    params: dict[str, Any]


@dataclass(frozen=True)
class AdaptiveProxemicSelectorConfig:
    """Deterministic local-context selector thresholds and fixed profiles."""

    diagnostic_only: bool
    claim_boundary: str
    density_radius_m: float
    close_human_distance_m: float
    neutral_human_distance_m: float
    high_density_count: int
    moderate_density_count: int
    constrained_width_m: float
    low_progress_3s_m: float
    profiles: dict[str, ProxemicProfileSpec]


def _as_1d_float(value: Any, *, default: float = 0.0) -> np.ndarray:
    """Return a one-dimensional float array for compact observation fields."""
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([default], dtype=float)
    return array if array.size else np.asarray([default], dtype=float)


def _xy_rows(value: Any) -> np.ndarray:
    """Return an ``(N, 2)`` float array or an empty array when unavailable."""
    try:
        rows = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.zeros((0, 2), dtype=float)
    if rows.ndim == 1 and rows.size % 2 == 0:
        rows = rows.reshape(-1, 2)
    if rows.ndim != 2 or rows.shape[-1] != 2:
        return np.zeros((0, 2), dtype=float)
    return rows


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return recursive merge without mutating either input."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _profile_defaults() -> dict[str, ProxemicProfileSpec]:
    """Return bundled equivalents of the issue #1676 fixed proxemic candidates."""
    specs: dict[str, ProxemicProfileSpec] = {}
    for name in _PROFILE_ORDER:
        params = _deep_merge(_DEFAULT_PROFILE_BASE, _DEFAULT_PROFILE_OVERRIDES[name])
        specs[name] = ProxemicProfileSpec(
            name=name,
            source_candidate=_SOURCE_CANDIDATES[name],
            params=params,
        )
    return specs


def _selector_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize optional selector threshold overrides.

    Returns:
        dict[str, Any]: Selector thresholds with defaults filled in.
    """
    settings = raw.get("selector") if isinstance(raw.get("selector"), dict) else {}
    return {
        "density_radius_m": float(settings.get("density_radius_m", 2.0)),
        "close_human_distance_m": float(settings.get("close_human_distance_m", 0.8)),
        "neutral_human_distance_m": float(settings.get("neutral_human_distance_m", 1.6)),
        "high_density_count": int(settings.get("high_density_count", 3)),
        "moderate_density_count": int(settings.get("moderate_density_count", 2)),
        "constrained_width_m": float(settings.get("constrained_width_m", 1.0)),
        "low_progress_3s_m": float(settings.get("low_progress_3s_m", 0.05)),
    }


def _common_profile_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Collect root-level hybrid-rule overrides applied to every profile.

    Returns:
        dict[str, Any]: Hybrid-rule config overrides shared by all profiles.
    """
    overrides = (
        dict(raw.get("profile_base") or {}) if isinstance(raw.get("profile_base"), dict) else {}
    )
    for key, value in raw.items():
        if key in _SELECTOR_KEYS:
            continue
        if key in _HYBRID_RULE_CONFIG_KEYS:
            overrides[key] = value
    return overrides


def build_adaptive_proxemic_selector_config(
    cfg: dict[str, Any] | None,
) -> AdaptiveProxemicSelectorConfig:
    """Build a selector config from YAML-style mappings.

    Returns:
        Adaptive proxemic selector configuration.
    """
    raw = dict(cfg or {}) if isinstance(cfg, dict) else {}
    selector = _selector_settings(raw)
    diagnostic_only = bool(raw.get("diagnostic_only", True))
    claim_boundary = str(raw.get("claim_boundary") or "diagnostic_only")
    if not diagnostic_only or claim_boundary != "diagnostic_only":
        raise ValueError(
            "adaptive_proxemic_selector_v0 is diagnostic-only; "
            "set diagnostic_only: true and claim_boundary: diagnostic_only"
        )
    profiles = _profile_defaults()
    common_overrides = _common_profile_overrides(raw)
    profile_payload = raw.get("profiles")
    if isinstance(profile_payload, dict):
        unknown = sorted(set(profile_payload) - set(_PROFILE_ORDER))
        if unknown:
            raise ValueError(f"Unknown adaptive proxemic profile(s): {', '.join(unknown)}")
        for name in _PROFILE_ORDER:
            item = profile_payload.get(name)
            if not isinstance(item, dict):
                continue
            params = profiles[name].params
            profile_params = item.get("params")
            if isinstance(profile_params, dict):
                params = _deep_merge(params, profile_params)
            profiles[name] = ProxemicProfileSpec(
                name=name,
                source_candidate=str(
                    item.get("source_candidate") or profiles[name].source_candidate
                ),
                params=params,
            )
    if common_overrides:
        for name, profile in list(profiles.items()):
            profiles[name] = ProxemicProfileSpec(
                name=name,
                source_candidate=profile.source_candidate,
                params=_deep_merge(profile.params, common_overrides),
            )
    return AdaptiveProxemicSelectorConfig(
        diagnostic_only=diagnostic_only,
        claim_boundary=claim_boundary,
        profiles=profiles,
        **selector,
    )


class AdaptiveProxemicSelectorAdapter:
    """Deterministically switch among fixed proxemic profiles for diagnostics."""

    def __init__(self, config: AdaptiveProxemicSelectorConfig | None = None) -> None:
        """Initialize one hybrid-rule planner per selectable profile."""
        self.config = config or build_adaptive_proxemic_selector_config({})
        self._planners = {
            name: HybridRuleLocalPlannerAdapter(build_hybrid_rule_local_planner_config(spec.params))
            for name, spec in self.config.profiles.items()
        }
        self.reset()

    def bind_env(self, env: Any) -> None:
        """Bind environment geometry to all wrapped profile planners."""
        for planner in self._planners.values():
            bind_env = getattr(planner, "bind_env", None)
            if callable(bind_env):
                bind_env(env)

    def reset(self, *, seed: int | None = None) -> None:
        """Reset selector counters and all wrapped planners."""
        self._step_index = 0
        self._selected_profile_counts: Counter[str] = Counter()
        self._trigger_reason_counts: Counter[str] = Counter()
        self._last_selection: dict[str, Any] | None = None
        for planner in getattr(self, "_planners", {}).values():
            planner.reset(seed=seed)

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Select a fixed profile from local context and return its command.

        Returns:
            tuple[float, float]: Linear and angular velocity from the selected profile planner.
        """
        profile, reason, context = self._select_profile(observation)
        planner = self._planners[profile]
        linear, angular = planner.plan(self._observation_for_selected_planner(observation))
        self._step_index += 1
        self._selected_profile_counts[profile] += 1
        self._trigger_reason_counts[reason] += 1
        spec = self.config.profiles[profile]
        self._last_selection = {
            "step": int(self._step_index),
            "selected_profile": profile,
            "trigger_reason": reason,
            "source_candidate": spec.source_candidate,
            **context,
        }
        return float(linear), float(angular)

    def diagnostics(self) -> dict[str, Any]:
        """Return episode-level selector diagnostics for benchmark metadata."""
        active_profile = (
            self._last_selection.get("selected_profile")
            if isinstance(self._last_selection, dict)
            else None
        )
        active_diagnostics = None
        if isinstance(active_profile, str) and active_profile in self._planners:
            active_diagnostics = self._planners[active_profile].diagnostics()
        return {
            "selector": "adaptive_proxemic_selector_v0",
            "diagnostic_only": bool(self.config.diagnostic_only),
            "claim_boundary": self.config.claim_boundary,
            "steps": int(self._step_index),
            "selected_profile_counts": dict(sorted(self._selected_profile_counts.items())),
            "trigger_reason_counts": dict(sorted(self._trigger_reason_counts.items())),
            "last_selection": dict(self._last_selection) if self._last_selection else None,
            "profiles": {
                name: {"source_candidate": spec.source_candidate}
                for name, spec in sorted(self.config.profiles.items())
            },
            "active_profile_diagnostics": active_diagnostics,
        }

    def _select_profile(self, observation: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        """Return selected profile, reason, and compact local-context diagnostics."""
        robot_pos, ped_pos = self._extract_robot_and_pedestrians(observation)
        distances = (
            np.linalg.norm(ped_pos - robot_pos[None, :], axis=1)
            if ped_pos.size
            else np.asarray([], dtype=float)
        )
        nearest = float(np.min(distances)) if distances.size else float("inf")
        local_density_count = int(np.count_nonzero(distances <= self.config.density_radius_m))
        constrained = self._is_constrained_passage(observation)
        low_progress = self._is_low_progress(observation)
        context = {
            "nearest_human_distance_m": _finite_or_none(nearest),
            "local_density_count": local_density_count,
            "density_radius_m": float(self.config.density_radius_m),
            "constrained_passage": bool(constrained),
            "low_progress_risk": bool(low_progress),
        }
        if nearest <= self.config.close_human_distance_m:
            return "conservative", "near_human", context
        if local_density_count >= self.config.high_density_count:
            return "conservative", "high_local_density", context
        if constrained:
            return "neutral", "constrained_passage", context
        if (
            low_progress
            and local_density_count == 0
            and nearest > self.config.neutral_human_distance_m
        ):
            return "open", "low_progress_clear_space", context
        if low_progress:
            return "neutral", "low_progress_with_humans", context
        if (
            nearest <= self.config.neutral_human_distance_m
            or local_density_count >= self.config.moderate_density_count
        ):
            return "neutral", "moderate_density", context
        return "open", "clear_low_density", context

    def _observation_for_selected_planner(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return observation with invalid pedestrian counts normalized for profile planners."""
        pedestrians = observation.get("pedestrians")
        if not isinstance(pedestrians, dict):
            return observation
        ped_pos = _xy_rows(pedestrians.get("positions", observation.get("pedestrians_positions")))
        count_raw = _as_1d_float(pedestrians.get("count", [ped_pos.shape[0]]))
        count_val = count_raw[0] if count_raw.size else float("nan")
        if np.isfinite(count_val):
            return observation
        sanitized_pedestrians = dict(pedestrians)
        sanitized_pedestrians["count"] = np.asarray([ped_pos.shape[0]], dtype=float)
        sanitized = dict(observation)
        sanitized["pedestrians"] = sanitized_pedestrians
        return sanitized

    def _extract_robot_and_pedestrians(
        self,
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract robot and pedestrian positions from structured observations.

        Returns:
            tuple[np.ndarray, np.ndarray]: Robot position and pedestrian position rows.
        """
        robot = observation.get("robot") if isinstance(observation.get("robot"), dict) else {}
        pedestrians = (
            observation.get("pedestrians")
            if isinstance(observation.get("pedestrians"), dict)
            else {}
        )
        robot_pos = _as_1d_float(
            robot.get("position", observation.get("robot_position", [0.0, 0.0]))
        )
        if robot_pos.size < 2:
            robot_pos = np.pad(robot_pos, (0, 2 - robot_pos.size), constant_values=0.0)
        ped_pos = _xy_rows(pedestrians.get("positions", observation.get("pedestrians_positions")))
        count_raw = _as_1d_float(pedestrians.get("count", [ped_pos.shape[0]]))
        count_val = count_raw[0] if count_raw.size else float("nan")
        count = (
            max(0, min(int(count_val), ped_pos.shape[0]))
            if np.isfinite(count_val)
            else ped_pos.shape[0]
        )
        return robot_pos[:2], ped_pos[:count]

    def _is_constrained_passage(self, observation: dict[str, Any]) -> bool:
        """Return whether observation diagnostics indicate a narrow local passage."""
        route_corridor = observation.get("route_corridor")
        if not isinstance(route_corridor, dict):
            route_corridor = observation.get("route")
        if not isinstance(route_corridor, dict):
            return False
        for key in ("corridor_width_estimate", "corridor_width_m", "passage_width_m"):
            if key not in route_corridor:
                continue
            try:
                width = float(route_corridor[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(width) and 0.0 < width <= self.config.constrained_width_m:
                return True
        return False

    def _is_low_progress(self, observation: dict[str, Any]) -> bool:
        """Return whether route-progress diagnostics indicate local stall risk."""
        windows = observation.get("route_arc_progress_windows")
        if not isinstance(windows, dict):
            route_corridor = observation.get("route_corridor")
            if isinstance(route_corridor, dict):
                windows = route_corridor.get("route_arc_progress_windows")
        if not isinstance(windows, dict):
            return False
        raw_progress = windows.get("3s", windows.get("3.0s", windows.get("three_s")))
        try:
            progress = float(raw_progress)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(progress) and progress <= self.config.low_progress_3s_m)


def _finite_or_none(value: Any) -> float | None:
    """Return finite floats for JSON-safe diagnostics."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


__all__ = [
    "AdaptiveProxemicSelectorAdapter",
    "AdaptiveProxemicSelectorConfig",
    "ProxemicProfileSpec",
    "build_adaptive_proxemic_selector_config",
]
