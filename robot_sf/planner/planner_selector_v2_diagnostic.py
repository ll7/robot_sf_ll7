"""Deterministic diagnostic planner selector for policy-search smoke runs."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_REQUIRED_CANDIDATES = (
    "baseline",
    "topology_route",
    "proxemic_conservative",
    "fast_progress_static_escape",
)


@dataclass(frozen=True)
class PlannerSelectorV2DiagnosticConfig:
    """No-leakage rule inputs for the diagnostic planner selector."""

    scenario_id: str = "unknown"
    scenario_family: str = "unknown"
    seed: int | None = None
    topology_scenarios: tuple[str, ...] = ()
    topology_keywords: tuple[str, ...] = ("bottleneck", "doorway", "narrow", "corridor")
    seed_sensitive_scenarios: tuple[str, ...] = ()
    hard_seed_values: tuple[int, ...] = ()
    evidence_sources: tuple[str, ...] = (
        "docs/context/issue_1608_seed_sensitivity_analysis.md",
        "docs/context/issue_1692_topology_hypothesis_probe.md",
    )
    dense_ped_count: int = 4
    near_field_distance_m: float = 2.3
    comfort_distance_m: float = 1.05
    baseline_candidate: str = "baseline"
    topology_candidate: str = "topology_route"
    proxemic_candidate: str = "proxemic_conservative"
    progress_candidate: str = "fast_progress_static_escape"
    leakage_excluded_fields: tuple[str, ...] = (
        "termination_reason",
        "outcome",
        "metrics",
        "future_observations",
        "benchmark_rank",
        "success",
        "collision",
    )


@dataclass(frozen=True)
class PlannerSelectorV2BuildConfig:
    """Build-time payload for selector routing and child candidate configs."""

    selector: PlannerSelectorV2DiagnosticConfig
    candidate_config_paths: dict[str, str] = field(default_factory=dict)


def build_planner_selector_v2_diagnostic_config(
    cfg: dict[str, Any] | None,
) -> PlannerSelectorV2BuildConfig:
    """Build selector config from a benchmark YAML mapping.

    Returns:
        Build config containing deterministic selector rules and candidate config paths.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    selector_raw = cfg.get("selector")
    selector_raw = selector_raw if isinstance(selector_raw, dict) else {}
    context_raw = cfg.get("selector_context")
    context_raw = context_raw if isinstance(context_raw, dict) else {}

    def _tuple_str(key: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
        raw = selector_raw.get(key, default)
        if isinstance(raw, str):
            return (raw,)
        if isinstance(raw, list | tuple):
            return tuple(str(item) for item in raw)
        return default

    def _tuple_int(key: str) -> tuple[int, ...]:
        raw = selector_raw.get(key, ())
        if isinstance(raw, int):
            return (int(raw),)
        if isinstance(raw, list | tuple):
            return tuple(int(item) for item in raw)
        return ()

    seed_raw = context_raw.get("seed")
    seed = int(seed_raw) if seed_raw is not None else None
    selector = PlannerSelectorV2DiagnosticConfig(
        scenario_id=str(
            context_raw.get("scenario_id") or selector_raw.get("scenario_id", "unknown")
        ),
        scenario_family=str(
            context_raw.get("scenario_family") or selector_raw.get("scenario_family", "unknown")
        ),
        seed=seed,
        topology_scenarios=_tuple_str("topology_scenarios"),
        topology_keywords=_tuple_str(
            "topology_keywords",
            ("bottleneck", "doorway", "narrow", "corridor"),
        ),
        seed_sensitive_scenarios=_tuple_str("seed_sensitive_scenarios"),
        hard_seed_values=_tuple_int("hard_seed_values"),
        evidence_sources=_tuple_str(
            "evidence_sources",
            (
                "docs/context/issue_1608_seed_sensitivity_analysis.md",
                "docs/context/issue_1692_topology_hypothesis_probe.md",
            ),
        ),
        dense_ped_count=int(selector_raw.get("dense_ped_count", 4)),
        near_field_distance_m=float(selector_raw.get("near_field_distance_m", 2.3)),
        comfort_distance_m=float(selector_raw.get("comfort_distance_m", 1.05)),
        baseline_candidate=str(selector_raw.get("baseline_candidate", "baseline")),
        topology_candidate=str(selector_raw.get("topology_candidate", "topology_route")),
        proxemic_candidate=str(selector_raw.get("proxemic_candidate", "proxemic_conservative")),
        progress_candidate=str(
            selector_raw.get("progress_candidate", "fast_progress_static_escape")
        ),
    )
    candidate_paths_raw = cfg.get("candidate_config_paths")
    candidate_paths_raw = candidate_paths_raw if isinstance(candidate_paths_raw, dict) else {}
    return PlannerSelectorV2BuildConfig(
        selector=selector,
        candidate_config_paths={str(key): str(value) for key, value in candidate_paths_raw.items()},
    )


class PlannerSelectorV2DiagnosticAdapter:
    """Select one existing local planner candidate by auditable diagnostic rules."""

    def __init__(
        self,
        *,
        config: PlannerSelectorV2DiagnosticConfig,
        candidate_adapters: dict[str, Any],
    ) -> None:
        """Construct the selector with already-built child planner adapters."""
        missing = [name for name in _REQUIRED_CANDIDATES if name not in candidate_adapters]
        if missing:
            raise ValueError(f"planner_selector_v2_diagnostic missing candidates: {missing}")
        self.config = config
        self.candidate_adapters = dict(candidate_adapters)
        self._selected_candidate_counts: Counter[str] = Counter()
        self._trigger_reason_counts: Counter[str] = Counter()
        self._steps = 0
        self._last_decision: dict[str, Any] | None = None

    def bind_env(self, env: object) -> None:
        """Propagate environment binding to child planners that need map geometry."""
        for adapter in self.candidate_adapters.values():
            bind = getattr(adapter, "bind_env", None)
            if callable(bind):
                bind(env)

    def reset(self) -> None:
        """Clear selector diagnostics and reset all child planners."""
        self._selected_candidate_counts.clear()
        self._trigger_reason_counts.clear()
        self._steps = 0
        self._last_decision = None
        for adapter in self.candidate_adapters.values():
            reset = getattr(adapter, "reset", None)
            if callable(reset):
                reset()

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return the command from the selected diagnostic head."""
        selected, reason, rule_inputs = self._select_candidate(observation)
        adapter = self.candidate_adapters[selected]
        command = adapter.plan(observation)
        self._record_decision(
            selected_candidate=selected,
            trigger_reason=reason,
            rule_inputs=rule_inputs,
        )
        return float(command[0]), float(command[1])

    def _select_candidate(self, observation: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        """Select a child candidate from scenario metadata and current observation only.

        Returns:
            Selected candidate key, trigger reason, and JSON-safe rule inputs.
        """
        near_count, min_ped_distance = self._pedestrian_risk(observation)
        scenario_id = self.config.scenario_id.strip()
        scenario_key = scenario_id.lower()
        scenario_family = self.config.scenario_family.strip().lower()
        topology_match = scenario_id in set(self.config.topology_scenarios) or any(
            keyword.lower() in scenario_key for keyword in self.config.topology_keywords
        )
        seed_sensitive = scenario_id in set(self.config.seed_sensitive_scenarios)
        hard_seed = self.config.seed is not None and int(self.config.seed) in set(
            self.config.hard_seed_values
        )
        dense_or_comfort = near_count >= int(self.config.dense_ped_count) or (
            min_ped_distance is not None
            and min_ped_distance <= float(self.config.comfort_distance_m)
        )
        open_space = near_count == 0 and (
            min_ped_distance is None or min_ped_distance >= float(self.config.near_field_distance_m)
        )
        rule_inputs = {
            "scenario_id": scenario_id,
            "scenario_family": scenario_family,
            "seed": self.config.seed,
            "evidence_sources": list(self.config.evidence_sources),
            "near_pedestrian_count": near_count,
            "min_pedestrian_distance_m": min_ped_distance,
            "topology_match": bool(topology_match),
            "seed_sensitive_scenario": bool(seed_sensitive),
            "hard_seed": bool(hard_seed),
            "open_space": bool(open_space),
        }
        if topology_match:
            return self.config.topology_candidate, "predeclared_topology_signature", rule_inputs
        if dense_or_comfort:
            return self.config.proxemic_candidate, "dense_social_or_comfort_risk", rule_inputs
        if open_space and (seed_sensitive or hard_seed):
            return (
                self.config.progress_candidate,
                "predeclared_seed_sensitive_low_progress_risk",
                rule_inputs,
            )
        return self.config.baseline_candidate, "default_safe_baseline", rule_inputs

    def _pedestrian_risk(self, observation: dict[str, Any]) -> tuple[int, float | None]:
        """Compute near-field pedestrian count and minimum distance from current observation.

        Returns:
            Near-field pedestrian count and minimum pedestrian distance when available.
        """
        robot = observation.get("robot") if isinstance(observation.get("robot"), dict) else {}
        pedestrians = (
            observation.get("pedestrians")
            if isinstance(observation.get("pedestrians"), dict)
            else {}
        )
        raw_robot_pos = robot.get("position")
        robot_pos = np.asarray(
            [0.0, 0.0] if raw_robot_pos is None else raw_robot_pos,
            dtype=float,
        )
        if robot_pos.shape != (2,):
            robot_pos = np.asarray([0.0, 0.0], dtype=float)
        raw_positions = pedestrians.get("positions")
        ped_pos = np.asarray([] if raw_positions is None else raw_positions, dtype=float)
        if ped_pos.ndim != 2 or ped_pos.shape[-1] != 2 or ped_pos.shape[0] == 0:
            return 0, None
        count_raw = np.asarray(pedestrians.get("count", [ped_pos.shape[0]]), dtype=float).reshape(
            -1
        )
        count_val = count_raw[0] if count_raw.size else float("nan")
        ped_count = (
            max(0, min(int(count_val), int(ped_pos.shape[0])))
            if np.isfinite(count_val)
            else int(ped_pos.shape[0])
        )
        ped_pos = ped_pos[:ped_count]
        if ped_pos.shape[0] == 0:
            return 0, None
        distances = np.linalg.norm(ped_pos - robot_pos[None, :], axis=1)
        finite = distances[np.isfinite(distances)]
        if finite.size == 0:
            return 0, None
        near_count = int(np.count_nonzero(finite <= float(self.config.near_field_distance_m)))
        return near_count, float(np.min(finite))

    def _record_decision(
        self,
        *,
        selected_candidate: str,
        trigger_reason: str,
        rule_inputs: dict[str, Any],
    ) -> None:
        """Store JSON-safe decision diagnostics for one step."""
        self._steps += 1
        self._selected_candidate_counts[selected_candidate] += 1
        self._trigger_reason_counts[trigger_reason] += 1
        self._last_decision = {
            "decision": "planner_selector_v2_diagnostic",
            "selected_candidate": selected_candidate,
            "selected_head": selected_candidate,
            "trigger_reason": trigger_reason,
            "rule_inputs": deepcopy(rule_inputs),
            "no_leakage": self._no_leakage_payload(),
        }

    def _no_leakage_payload(self) -> dict[str, Any]:
        """Describe the selector's intentionally bounded input surface.

        Returns:
            JSON-safe no-leakage contract payload for diagnostics.
        """
        return {
            "allowed_inputs": [
                "predeclared_scenario_id",
                "predeclared_scenario_family",
                "predeclared_seed_flags",
                "current_robot_pose",
                "current_goal",
                "current_pedestrian_positions",
            ],
            "excluded_fields": list(self.config.leakage_excluded_fields),
            "current_episode_outcome_fields_used": [],
            "future_observation_fields_used": [],
            "learned_policy_used": False,
            "evidence_sources": list(self.config.evidence_sources),
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return selector diagnostics for benchmark episode metadata."""
        selected = None
        if self._last_decision is not None:
            selected = self._last_decision["selected_candidate"]
        return {
            "schema_version": "planner_selector_v2_diagnostic.v1",
            "diagnostic_only": True,
            "steps": int(self._steps),
            "selected_candidate": selected,
            "selected_candidate_counts": dict(self._selected_candidate_counts),
            "trigger_reason_counts": dict(self._trigger_reason_counts),
            "last_decision": deepcopy(self._last_decision) if self._last_decision else None,
            "no_leakage": self._no_leakage_payload(),
            "candidate_heads": {
                "baseline": self.config.baseline_candidate,
                "topology": self.config.topology_candidate,
                "proxemic": self.config.proxemic_candidate,
                "progress": self.config.progress_candidate,
            },
        }

    def last_decision(self) -> dict[str, Any] | None:
        """Return a copy of the latest selector decision."""
        return deepcopy(self._last_decision) if self._last_decision else None


__all__ = [
    "PlannerSelectorV2BuildConfig",
    "PlannerSelectorV2DiagnosticAdapter",
    "PlannerSelectorV2DiagnosticConfig",
    "build_planner_selector_v2_diagnostic_config",
]
