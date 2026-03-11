"""Gap-aware predictive planner.

This adapter uses the predictive planner as the main controller and applies the
stream-gap planner as a veto/approach layer when the goal corridor is blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot_sf.planner.socnav import PredictionPlannerAdapter, SocNavPlannerConfig
from robot_sf.planner.stream_gap import StreamGapPlannerAdapter, StreamGapPlannerConfig


@dataclass
class GapPredictionConfig:
    """Composite configuration for gap-aware predictive control."""

    stream_gap: StreamGapPlannerConfig
    predictive: SocNavPlannerConfig
    stop_override_margin: float = 0.02
    approach_speed_cap: float = 0.45


class GapAwarePredictionAdapter:
    """Use predictive control unless gap logic says stop or slow down."""

    def __init__(self, config: GapPredictionConfig, *, allow_fallback: bool = False) -> None:
        """Build predictive and gap-control heads."""
        self.config = config
        self._gap = StreamGapPlannerAdapter(config.stream_gap)
        self._prediction = PredictionPlannerAdapter(
            config=config.predictive,
            allow_fallback=allow_fallback,
        )

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return a gap-aware predictive command."""
        gap_v, gap_w = self._gap.plan(observation)
        pred_v, pred_w = self._prediction.plan(observation)

        if gap_v <= float(self.config.stream_gap.wait_speed) + float(
            self.config.stop_override_margin
        ):
            return float(gap_v), float(gap_w)

        if gap_v <= float(self.config.stream_gap.approach_speed) + float(
            self.config.stop_override_margin
        ):
            return min(float(pred_v), float(self.config.approach_speed_cap)), float(pred_w)

        return float(pred_v), float(pred_w)


def build_gap_prediction_config(cfg: dict[str, Any] | None) -> GapPredictionConfig:
    """Build :class:`GapPredictionConfig` from a mapping payload.

    Returns:
        GapPredictionConfig: Parsed hybrid configuration.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    gap_raw = cfg.get("stream_gap", {}) if isinstance(cfg.get("stream_gap"), dict) else {}
    pred_allowed = {field.name for field in SocNavPlannerConfig.__dataclass_fields__.values()}
    pred_raw = {key: value for key, value in cfg.items() if key in pred_allowed}
    pred_cfg = SocNavPlannerConfig(**pred_raw)
    gap_cfg = StreamGapPlannerConfig(
        max_linear_speed=float(gap_raw.get("max_linear_speed", 1.2)),
        max_angular_speed=float(gap_raw.get("max_angular_speed", 1.2)),
        goal_tolerance=float(gap_raw.get("goal_tolerance", 0.25)),
        heading_gain=float(gap_raw.get("heading_gain", 1.6)),
        turn_in_place_angle=float(gap_raw.get("turn_in_place_angle", 0.7)),
        forward_lookahead=float(gap_raw.get("forward_lookahead", 4.0)),
        rear_margin=float(gap_raw.get("rear_margin", 0.5)),
        corridor_half_width=float(gap_raw.get("corridor_half_width", 0.85)),
        emergency_clearance=float(gap_raw.get("emergency_clearance", 0.55)),
        sample_dt=float(gap_raw.get("sample_dt", 0.2)),
        sample_horizon=float(gap_raw.get("sample_horizon", 4.0)),
        safe_gap_time=float(gap_raw.get("safe_gap_time", 1.0)),
        approach_gap_time=float(gap_raw.get("approach_gap_time", 0.8)),
        wait_speed=float(gap_raw.get("wait_speed", 0.0)),
        creep_speed=float(gap_raw.get("creep_speed", 0.12)),
        approach_speed=float(gap_raw.get("approach_speed", 0.35)),
        commit_speed=float(gap_raw.get("commit_speed", 0.95)),
        commit_hold_steps=int(gap_raw.get("commit_hold_steps", 6)),
    )
    return GapPredictionConfig(
        stream_gap=gap_cfg,
        predictive=pred_cfg,
        stop_override_margin=float(cfg.get("stop_override_margin", 0.02)),
        approach_speed_cap=float(cfg.get("approach_speed_cap", 0.45)),
    )


__all__ = ["GapAwarePredictionAdapter", "GapPredictionConfig", "build_gap_prediction_config"]
