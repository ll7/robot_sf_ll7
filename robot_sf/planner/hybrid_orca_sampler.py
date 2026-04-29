"""Hybrid ORCA planner with sampler-based progress repair."""

from __future__ import annotations

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

    def __init__(
        self,
        config: HybridORCASamplerConfig | None = None,
        *,
        orca_adapter: ORCAPlannerAdapter | None = None,
        sampler_adapter: MPPISocialPlannerAdapter | None = None,
    ) -> None:
        guard_cfg = config or HybridORCASamplerConfig()
        sampler = sampler_adapter or MPPISocialPlannerAdapter()
        super().__init__(config=guard_cfg, fallback_adapter=cast("Any", sampler))
        self.primary_adapter = orca_adapter or ORCAPlannerAdapter()
        self.sampler_adapter = sampler

    @property
    def hybrid_config(self) -> HybridORCASamplerConfig:
        """Return typed hybrid config view."""
        return cast("HybridORCASamplerConfig", self.config)

    def reset(self) -> None:
        """Reset child planners that keep per-episode state."""
        for planner in (self.primary_adapter, self.sampler_adapter):
            reset = getattr(planner, "reset", None)
            if callable(reset):
                reset()

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return ORCA action unless MPPI is needed for safety or progress repair."""
        robot_pos, _heading, goal, ped_pos, _ped_vel = self._extract_state(observation)
        if float(np.linalg.norm(goal - robot_pos)) <= float(self.hybrid_config.goal_tolerance):
            return 0.0, 0.0

        try:
            primary_command = self.primary_adapter.plan(observation)
        except Exception:
            return tuple(float(value) for value in self.sampler_adapter.plan(observation))

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
            return tuple(float(value) for value in primary_command)

        try:
            sampler_command = self.sampler_adapter.plan(observation)
        except Exception:
            chosen, _ = self.choose_command(observation, primary_command)
            return chosen

        sampler_eval = self._evaluate_command(observation, sampler_command)
        if bool(sampler_eval["safe"]) and (
            not bool(primary_eval["safe"])
            or float(sampler_eval["progress"])
            > float(primary_eval["progress"]) + float(self.hybrid_config.sampler_progress_margin)
        ):
            return tuple(float(value) for value in sampler_command)

        chosen, _ = self.choose_command(observation, primary_command)
        return chosen


def build_hybrid_orca_sampler_build_config(
    cfg: dict[str, Any] | None,
) -> HybridORCASamplerBuildConfig:
    """Build ORCA, guard, and sampler config from one mapping payload."""
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
