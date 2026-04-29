"""Hybrid ORCA planner with sampler-based progress repair."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from typing import Any, cast

import numpy as np

from robot_sf.planner.guarded_ppo import GuardedPPOAdapter, GuardedPPOConfig
from robot_sf.planner.mppi_social import (
    MPPISocialConfig,
    MPPISocialPlannerAdapter,
    build_mppi_social_config,
)
from robot_sf.planner.socnav import ORCAPlannerAdapter, SocNavPlannerConfig


@dataclass
class HybridORCASamplerConfig(GuardedPPOConfig):
    """Configuration for ORCA with sampled fallback/repair."""

    sampler_progress_margin: float = 0.05


@dataclass
class HybridORCASamplerBuildConfig:
    """Composite build config for the hybrid ORCA sampler planner."""

    guard: HybridORCASamplerConfig
    socnav: SocNavPlannerConfig
    mppi: MPPISocialConfig


class HybridORCASamplerAdapter(GuardedPPOAdapter):
    """Use ORCA by default and escalate to MPPI when progress or safety degrades."""

    _TRACE_LIMIT = 128
    _PLANNER_EXCEPTIONS = (RuntimeError, TypeError, ValueError)

    def __init__(
        self,
        config: HybridORCASamplerConfig | None = None,
        *,
        orca_adapter: ORCAPlannerAdapter | None = None,
        sampler_adapter: MPPISocialPlannerAdapter | None = None,
    ) -> None:
        """Initialize the hybrid ORCA/sampler planner pair.

        Args:
            config: Hybrid guard configuration.
            orca_adapter: Primary ORCA head to evaluate first.
            sampler_adapter: Sampled fallback/repair head.
        """
        guard_cfg = config or HybridORCASamplerConfig()
        sampler = sampler_adapter or MPPISocialPlannerAdapter()
        super().__init__(config=guard_cfg, fallback_adapter=cast("Any", sampler))
        self.primary_adapter = orca_adapter or ORCAPlannerAdapter()
        self.sampler_adapter = sampler
        self._reset_diagnostics()

    @property
    def hybrid_config(self) -> HybridORCASamplerConfig:
        """Return typed hybrid config view."""
        return cast("HybridORCASamplerConfig", self.config)

    def reset(self) -> None:
        """Reset child planners that keep per-episode state."""
        self._reset_diagnostics()
        for planner in (self.primary_adapter, self.sampler_adapter):
            reset = getattr(planner, "reset", None)
            if callable(reset):
                reset()

    def _reset_diagnostics(self) -> None:
        """Clear decision-trace state for a new episode."""
        self._decision_counts: dict[str, int] = {}
        self._selected_head_counts: dict[str, int] = {
            "orca": 0,
            "sampler": 0,
            "stop": 0,
        }
        self._recent_decisions: deque[dict[str, Any]] = deque(maxlen=self._TRACE_LIMIT)
        self._last_decision: dict[str, Any] | None = None

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        """Convert numeric values to finite floats for JSON-safe diagnostics.

        Returns:
            float | None: Finite float value, or ``None`` for non-finite/invalid inputs.
        """
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    def _sanitize_eval(self, values: dict[str, Any] | None) -> dict[str, Any] | None:
        """Return a JSON-safe view of rollout evaluation metrics."""
        if not isinstance(values, dict):
            return None
        cleaned: dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, bool):
                cleaned[key] = value
            elif isinstance(value, int | float | np.integer | np.floating):
                cleaned[key] = self._finite_float(value)
        return cleaned

    @staticmethod
    def _selected_head_from_guard_label(label: str) -> str:
        """Map inherited guard labels to the selected planner head.

        Returns:
            str: Canonical selected-head label for diagnostics.
        """
        if label.startswith("fallback"):
            return "sampler"
        if label.startswith("stop") or label == "goal_reached":
            return "stop"
        return "orca"

    def _record_decision(
        self,
        *,
        decision: str,
        selected_head: str,
        chosen_command: tuple[float, float],
        clear_scene: bool | None,
        current_min_dist: float,
        primary_eval: dict[str, Any] | None = None,
        sampler_eval: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> tuple[float, float]:
        """Persist one planning decision for later diagnostics retrieval.

        Returns:
            tuple[float, float]: Chosen command normalized to float values.
        """
        record = {
            "decision": decision,
            "selected_head": selected_head,
            "chosen_command": [float(chosen_command[0]), float(chosen_command[1])],
            "clear_scene": clear_scene,
            "current_min_dist": self._finite_float(current_min_dist),
            "primary_eval": self._sanitize_eval(primary_eval),
            "sampler_eval": self._sanitize_eval(sampler_eval),
        }
        if note:
            record["note"] = note
        self._last_decision = record
        self._recent_decisions.append(record)
        self._decision_counts[decision] = self._decision_counts.get(decision, 0) + 1
        self._selected_head_counts[selected_head] = (
            self._selected_head_counts.get(selected_head, 0) + 1
        )
        return float(chosen_command[0]), float(chosen_command[1])

    def last_decision(self) -> dict[str, Any] | None:
        """Return the most recent plan decision record."""
        return dict(self._last_decision) if isinstance(self._last_decision, dict) else None

    def diagnostics(self) -> dict[str, Any]:
        """Return compact aggregate diagnostics for benchmark episode records."""
        return {
            "decision_counts": dict(sorted(self._decision_counts.items())),
            "selected_head_counts": dict(sorted(self._selected_head_counts.items())),
            "last_decision": self.last_decision(),
            "trace_depth": len(self._recent_decisions),
        }

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return ORCA action unless MPPI is needed for safety or progress repair."""
        robot_pos, _heading, goal, ped_pos, _ped_vel = self._extract_state(observation)
        if float(np.linalg.norm(goal - robot_pos)) <= float(self.hybrid_config.goal_tolerance):
            return self._record_decision(
                decision="goal_reached",
                selected_head="stop",
                chosen_command=(0.0, 0.0),
                clear_scene=None,
                current_min_dist=float("inf"),
            )

        try:
            primary_command = self.primary_adapter.plan(observation)
        except self._PLANNER_EXCEPTIONS as exc:
            sampler_command = tuple(
                float(value) for value in self.sampler_adapter.plan(observation)
            )
            return self._record_decision(
                decision="sampler_on_orca_exception",
                selected_head="sampler",
                chosen_command=sampler_command,
                clear_scene=None,
                current_min_dist=float("inf"),
                note=type(exc).__name__,
            )

        primary_eval = self._evaluate_command(observation, primary_command)
        current_min_dist = (
            float(np.min(np.linalg.norm(ped_pos - robot_pos[None, :], axis=1)))
            if ped_pos.size > 0
            else float("inf")
        )
        clear_scene = current_min_dist > float(self.hybrid_config.near_field_distance)
        if (
            clear_scene
            and bool(primary_eval["safe"])
            and float(primary_eval["progress"]) > float(self.hybrid_config.sampler_progress_margin)
        ):
            return self._record_decision(
                decision="orca_clear_scene",
                selected_head="orca",
                chosen_command=primary_command,
                clear_scene=clear_scene,
                current_min_dist=current_min_dist,
                primary_eval=primary_eval,
            )

        try:
            sampler_command = self.sampler_adapter.plan(observation)
        except self._PLANNER_EXCEPTIONS:
            chosen, guard_label = self.choose_command(observation, primary_command)
            return self._record_decision(
                decision=f"guard_{guard_label}",
                selected_head=self._selected_head_from_guard_label(guard_label),
                chosen_command=chosen,
                clear_scene=clear_scene,
                current_min_dist=current_min_dist,
                primary_eval=primary_eval,
            )

        sampler_eval = self._evaluate_command(observation, sampler_command)
        if bool(sampler_eval["safe"]) and (
            not bool(primary_eval["safe"])
            or float(sampler_eval["progress"])
            > float(primary_eval["progress"]) + float(self.hybrid_config.sampler_progress_margin)
        ):
            return self._record_decision(
                decision="sampler_progress_repair",
                selected_head="sampler",
                chosen_command=sampler_command,
                clear_scene=clear_scene,
                current_min_dist=current_min_dist,
                primary_eval=primary_eval,
                sampler_eval=sampler_eval,
            )

        chosen, guard_label = self.choose_command(observation, primary_command)
        return self._record_decision(
            decision=f"guard_{guard_label}",
            selected_head=self._selected_head_from_guard_label(guard_label),
            chosen_command=chosen,
            clear_scene=clear_scene,
            current_min_dist=current_min_dist,
            primary_eval=primary_eval,
            sampler_eval=sampler_eval,
        )


def build_hybrid_orca_sampler_build_config(
    cfg: dict[str, Any] | None,
) -> HybridORCASamplerBuildConfig:
    """Build ORCA, guard, and sampler config from one mapping payload.

    Returns:
        HybridORCASamplerBuildConfig: Parsed composite config for the hybrid planner.
    """
    cfg = cfg if isinstance(cfg, dict) else {}

    guard_raw = cfg.get("hybrid_guard", {}) if isinstance(cfg.get("hybrid_guard"), dict) else {}
    mppi_raw = cfg.get("mppi_social", {}) if isinstance(cfg.get("mppi_social"), dict) else {}

    allowed_socnav = {field.name for field in fields(SocNavPlannerConfig)}
    socnav_kwargs = {key: value for key, value in cfg.items() if key in allowed_socnav}
    socnav = SocNavPlannerConfig(**socnav_kwargs)

    guard = HybridORCASamplerConfig(
        rollout_dt=float(guard_raw.get("rollout_dt", 0.2)),
        rollout_steps=int(guard_raw.get("rollout_steps", 6)),
        goal_tolerance=float(guard_raw.get("goal_tolerance", 0.25)),
        near_field_distance=float(guard_raw.get("near_field_distance", 2.0)),
        hard_ped_clearance=float(guard_raw.get("hard_ped_clearance", 0.58)),
        first_step_ped_clearance=float(guard_raw.get("first_step_ped_clearance", 0.72)),
        hard_obstacle_clearance=float(guard_raw.get("hard_obstacle_clearance", 0.30)),
        min_ttc=float(guard_raw.get("min_ttc", 0.70)),
        obstacle_threshold=float(guard_raw.get("obstacle_threshold", 0.5)),
        obstacle_search_cells=int(guard_raw.get("obstacle_search_cells", 12)),
        sampler_progress_margin=float(
            guard_raw.get("sampler_progress_margin", guard_raw.get("progress_margin", 0.05))
        ),
    )

    mppi_seed = {
        key: value for key, value in cfg.items() if key in {"max_linear_speed", "max_angular_speed"}
    }
    mppi_cfg = build_mppi_social_config({**mppi_seed, **mppi_raw})

    return HybridORCASamplerBuildConfig(guard=guard, socnav=socnav, mppi=mppi_cfg)


__all__ = [
    "HybridORCASamplerAdapter",
    "HybridORCASamplerBuildConfig",
    "HybridORCASamplerConfig",
    "build_hybrid_orca_sampler_build_config",
]
