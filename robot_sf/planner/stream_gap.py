"""Gap-acceptance local planner for pedestrian stream crossing.

This planner detects pedestrians occupying the goal corridor, estimates future
free windows with a simple constant-velocity projection, and switches between
wait/approach/commit modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.common.math_utils import wrap_angle_pi as _wrap_angle


@dataclass
class StreamGapPlannerConfig:
    """Configuration for :class:`StreamGapPlannerAdapter`."""

    max_linear_speed: float = 1.2
    max_angular_speed: float = 1.2
    goal_tolerance: float = 0.25
    heading_gain: float = 1.6
    turn_in_place_angle: float = 0.7

    forward_lookahead: float = 4.0
    rear_margin: float = 0.5
    corridor_half_width: float = 0.85
    emergency_clearance: float = 0.55

    sample_dt: float = 0.2
    sample_horizon: float = 4.0
    safe_gap_time: float = 1.0
    approach_gap_time: float = 0.8

    wait_speed: float = 0.0
    creep_speed: float = 0.12
    approach_speed: float = 0.35
    commit_speed: float = 0.95
    commit_hold_steps: int = 6

    uncertainty_gating_enabled: bool = False
    uncertainty_min_existence_probability: float = 0.5
    uncertainty_min_position_confidence: float = 0.5
    uncertainty_min_class_probability: float = 0.5
    uncertainty_max_position_variance: float = 1.0

    def __post_init__(self) -> None:
        """Validate uncertainty-gating thresholds used by ``_uncertainty_keep_mask``."""
        probability_fields = {
            "uncertainty_min_existence_probability": self.uncertainty_min_existence_probability,
            "uncertainty_min_position_confidence": self.uncertainty_min_position_confidence,
            "uncertainty_min_class_probability": self.uncertainty_min_class_probability,
        }
        for name, value in probability_fields.items():
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0.0, 1.0]")
        if (
            not np.isfinite(self.uncertainty_max_position_variance)
            or self.uncertainty_max_position_variance < 0.0
        ):
            raise ValueError(
                "uncertainty_max_position_variance must be a finite non-negative value"
            )


class StreamGapPlannerAdapter:
    """Wait-for-gap and commit planner for lateral pedestrian streams."""

    def __init__(self, config: StreamGapPlannerConfig | None = None) -> None:
        """Initialize adapter and commit-mode state."""
        self.config = config or StreamGapPlannerConfig()
        self._commit_steps_remaining = 0
        self.last_uncertainty_gate: dict[str, Any] = {
            "schema_version": "stream-gap-uncertainty-gate.v1",
            "enabled": bool(self.config.uncertainty_gating_enabled),
            "status": "not_evaluated",
        }

    @staticmethod
    def _as_xy(values: Any) -> np.ndarray:
        """Coerce optional pedestrian arrays to an ``(N, 2)`` float matrix.

        Returns:
            np.ndarray: Valid XY matrix, or an empty matrix for malformed input.
        """
        arr = np.asarray([] if values is None else values, dtype=float)
        if arr.ndim != 2 or arr.shape[-1] != 2:
            return np.zeros((0, 2), dtype=float)
        return arr

    @staticmethod
    def _class_probability(row: dict[str, Any], class_name: str) -> float | None:
        """Return one class probability from a ScenarioBelief uncertainty row."""
        probabilities = row.get("class_probabilities")
        if not isinstance(probabilities, dict):
            return None
        value = probabilities.get(class_name)
        if value is None:
            return None
        probability = float(value)
        if not np.isfinite(probability):
            return None
        return probability

    @staticmethod
    def _position_variance(row: dict[str, Any]) -> float | None:
        """Return average positional variance from a 2x2 covariance matrix."""
        covariance = np.asarray(row.get("position_covariance_xy"), dtype=float)
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            return None
        if not np.allclose(covariance, covariance.T, atol=1e-6):
            return None
        return float(np.trace(covariance) / 2.0)

    def _fail_closed_uncertainty_gate(self, *, count: int, reason: str) -> np.ndarray:
        """Record a fail-closed uncertainty gate and keep every deterministic row.

        Returns:
            np.ndarray: All-true mask preserving deterministic pedestrian input rows.
        """
        self.last_uncertainty_gate.update(
            {
                "status": "fail_closed",
                "reason": reason,
                "kept_count": int(count),
                "dropped_count": 0,
                "dropped_reasons": [],
            }
        )
        return np.ones(count, dtype=bool)

    def _uncertainty_rows(self, pedestrians: dict[str, Any], count: int) -> list[Any] | None:
        """Return uncertainty sidecar rows, or ``None`` when malformed or incomplete."""
        raw = pedestrians.get("uncertainty")
        if isinstance(raw, dict):
            raw = raw.get("agents")
        if not isinstance(raw, list) or len(raw) < count:
            return None
        return raw[:count]

    def _uncertainty_row_metrics(
        self,
        row: dict[str, Any],
    ) -> tuple[float, float, float, float] | None:
        """Parse one uncertainty row into gating metrics.

        Returns:
            tuple[float, float, float, float] | None: Existence probability, position
            confidence, class probability, and position variance, or ``None`` when malformed.
        """
        try:
            existence = float(row.get("existence_probability"))
            confidence = float(row.get("position_confidence"))
            class_probability = self._class_probability(row, "pedestrian")
            variance = self._position_variance(row)
        except (TypeError, ValueError):
            return None
        if (
            not np.isfinite(existence)
            or not np.isfinite(confidence)
            or class_probability is None
            or variance is None
        ):
            return None
        return existence, confidence, class_probability, variance

    @staticmethod
    def _uncertainty_drop_reasons(
        *,
        existence: float,
        confidence: float,
        class_probability: float,
        variance: float,
        thresholds: dict[str, float],
    ) -> list[str]:
        """Return threshold failures for one parsed uncertainty row."""
        reasons: list[str] = []
        if existence < thresholds["min_existence_probability"]:
            reasons.append("existence_probability_below_threshold")
        if confidence < thresholds["min_position_confidence"]:
            reasons.append("position_confidence_below_threshold")
        if class_probability < thresholds["min_class_probability"]:
            reasons.append("class_probability_below_threshold")
        if variance > thresholds["max_position_variance"]:
            reasons.append("position_variance_above_threshold")
        return reasons

    def _uncertainty_keep_mask(
        self,
        *,
        pedestrians: dict[str, Any],
        count: int,
    ) -> np.ndarray:
        """Return which pedestrian rows survive opt-in ScenarioBelief uncertainty gating."""
        keep = np.ones(count, dtype=bool)
        self.last_uncertainty_gate = {
            "schema_version": "stream-gap-uncertainty-gate.v1",
            "enabled": bool(self.config.uncertainty_gating_enabled),
            "status": "disabled",
            "input_count": int(count),
            "kept_count": int(count),
            "dropped_count": 0,
            "dropped_reasons": [],
        }
        if count <= 0:
            self.last_uncertainty_gate["status"] = "empty"
            return keep
        if not bool(self.config.uncertainty_gating_enabled):
            return keep

        if "uncertainty" not in pedestrians:
            return self._fail_closed_uncertainty_gate(
                count=count,
                reason="missing_uncertainty_metadata",
            )
        rows = self._uncertainty_rows(pedestrians, count)
        if rows is None:
            return self._fail_closed_uncertainty_gate(
                count=count,
                reason="malformed_uncertainty_metadata",
            )

        thresholds = {
            "min_existence_probability": float(self.config.uncertainty_min_existence_probability),
            "min_position_confidence": float(self.config.uncertainty_min_position_confidence),
            "min_class_probability": float(self.config.uncertainty_min_class_probability),
            "max_position_variance": float(self.config.uncertainty_max_position_variance),
        }
        dropped_reasons: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                return self._fail_closed_uncertainty_gate(
                    count=count,
                    reason="malformed_uncertainty_metadata",
                )
            metrics = self._uncertainty_row_metrics(row)
            if metrics is None:
                return self._fail_closed_uncertainty_gate(
                    count=count,
                    reason="malformed_uncertainty_metadata",
                )
            existence, confidence, class_probability, variance = metrics
            reasons = self._uncertainty_drop_reasons(
                existence=existence,
                confidence=confidence,
                class_probability=class_probability,
                variance=variance,
                thresholds=thresholds,
            )
            if reasons:
                keep[index] = False
                dropped_reasons.append(
                    {
                        "row_index": int(index),
                        "entity_id": row.get("entity_id"),
                        "reasons": reasons,
                        "existence_probability": existence,
                        "position_confidence": confidence,
                        "class_probability": class_probability,
                        "position_variance": variance,
                    }
                )

        self.last_uncertainty_gate.update(
            {
                "status": "applied",
                "thresholds": thresholds,
                "kept_count": int(np.count_nonzero(keep)),
                "dropped_count": int(count - np.count_nonzero(keep)),
                "dropped_reasons": dropped_reasons,
            }
        )
        return keep

    def _extract_state(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """Extract robot, goal, and bounded pedestrian state from an observation.

        Returns:
            tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]: Robot
            position, heading, goal position, pedestrian positions, and velocities.
        """
        # Accept both the nested SOCNAV observation (robot/goal/pedestrians dicts, used by the
        # ScenarioBelief harness and unit tests) and the flat benchmark-runner observation
        # (robot_position / goal_current / pedestrians_positions ... keys). Without the flat
        # fallback the planner sees an empty observation in map_runner and drives blind.
        robot = (
            observation.get("robot")
            if isinstance(observation.get("robot"), dict)
            else {
                "position": observation.get("robot_position", [0.0, 0.0]),
                "heading": observation.get("robot_heading", [0.0]),
            }
        )
        goal = (
            observation.get("goal")
            if isinstance(observation.get("goal"), dict)
            else {
                "current": observation.get("goal_current", [0.0, 0.0]),
                "next": observation.get("goal_next", [0.0, 0.0]),
            }
        )
        pedestrians = (
            observation.get("pedestrians")
            if isinstance(observation.get("pedestrians"), dict)
            else {
                "positions": observation.get("pedestrians_positions"),
                "velocities": observation.get("pedestrians_velocities"),
                "count": observation.get("pedestrians_count"),
                "uncertainty": observation.get("pedestrians_uncertainty"),
            }
        )

        robot_pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=float).reshape(-1)[:2]
        if robot_pos.size < 2:
            robot_pos = np.zeros(2, dtype=float)
        heading = float(np.asarray(robot.get("heading", [0.0]), dtype=float).reshape(-1)[0])

        goal_next = np.asarray(goal.get("next", [0.0, 0.0]), dtype=float).reshape(-1)
        goal_current = np.asarray(goal.get("current", [0.0, 0.0]), dtype=float).reshape(-1)
        if (
            goal_next.size == 2
            and np.all(np.isfinite(goal_next))
            and np.linalg.norm(goal_next - robot_pos) > 1e-6
        ):
            goal_pos = goal_next
        else:
            goal_pos = goal_current if goal_current.size == 2 else np.zeros(2, dtype=float)
        if not np.all(np.isfinite(goal_pos)):
            goal_pos = np.zeros(2, dtype=float)

        ped_pos = self._as_xy(pedestrians.get("positions"))
        ped_vel = self._as_xy(pedestrians.get("velocities"))
        if ped_vel.shape != ped_pos.shape:
            ped_vel = np.zeros_like(ped_pos)

        count_arr = np.asarray(pedestrians.get("count", [ped_pos.shape[0]]), dtype=float).reshape(
            -1
        )
        count = int(count_arr[0]) if count_arr.size else ped_pos.shape[0]
        count = max(0, min(count, ped_pos.shape[0]))
        ped_pos = ped_pos[:count]
        ped_vel = ped_vel[:count]
        keep = self._uncertainty_keep_mask(pedestrians=pedestrians, count=count)
        return robot_pos, heading, goal_pos, ped_pos[keep], ped_vel[keep]

    def _goal_frame(
        self,
        *,
        robot_pos: np.ndarray,
        goal_pos: np.ndarray,
        ped_pos: np.ndarray,
        ped_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project pedestrians into the goal-aligned longitudinal/lateral frame.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Along-track offsets,
            lateral offsets, and velocities in the goal frame.
        """
        goal_vec = goal_pos - robot_pos
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist <= 1e-6:
            direction = np.array([1.0, 0.0], dtype=float)
        else:
            direction = goal_vec / goal_dist
        lateral = np.array([-direction[1], direction[0]], dtype=float)
        rel = ped_pos - robot_pos[None, :]
        along = rel @ direction
        cross = rel @ lateral
        vel_along = ped_vel @ direction
        vel_cross = ped_vel @ lateral
        return along, cross, np.stack([vel_along, vel_cross], axis=1)

    def _blocked_mask(
        self,
        *,
        along: np.ndarray,
        cross: np.ndarray,
    ) -> np.ndarray:
        """Return which pedestrians currently occupy the crossing corridor.

        Returns:
            np.ndarray: Boolean mask over pedestrian rows.
        """
        return (
            (along >= -float(self.config.rear_margin))
            & (along <= float(self.config.forward_lookahead))
            & (np.abs(cross) <= float(self.config.corridor_half_width))
        )

    def _gap_start_time(
        self,
        *,
        along: np.ndarray,
        cross: np.ndarray,
        vel_goal_frame: np.ndarray,
    ) -> tuple[float | None, bool, float]:
        """Find the first future time with a continuously safe crossing gap.

        Returns:
            tuple[float | None, bool, float]: Gap start time, whether the
            corridor is blocked now, and nearest blocked pedestrian distance.
        """
        dt = max(float(self.config.sample_dt), 1e-3)
        horizon = max(float(self.config.sample_horizon), dt)
        times = np.arange(0.0, horizon + dt * 0.5, dt, dtype=float)
        blocked = np.zeros(times.shape[0], dtype=bool)

        if along.size == 0:
            return 0.0, False, float("inf")

        min_distance = float("inf")
        for idx, t in enumerate(times):
            along_t = along + vel_goal_frame[:, 0] * t
            cross_t = cross + vel_goal_frame[:, 1] * t
            inside = self._blocked_mask(along=along_t, cross=cross_t)
            blocked[idx] = bool(np.any(inside))
            if inside.any():
                min_distance = min(
                    min_distance,
                    float(np.min(np.sqrt(along_t[inside] ** 2 + cross_t[inside] ** 2))),
                )

        free_steps_needed = max(1, int(np.ceil(float(self.config.safe_gap_time) / dt)))
        for start in range(blocked.shape[0]):
            end = start + free_steps_needed
            if end > blocked.shape[0]:
                break
            if not bool(np.any(blocked[start:end])):
                return float(times[start]), bool(blocked[0]), min_distance
        return None, bool(blocked[0]), min_distance

    def _heading_command(
        self, robot_pos: np.ndarray, heading: float, goal_pos: np.ndarray
    ) -> tuple[float, float]:
        """Compute heading error and bounded angular velocity toward the goal.

        Returns:
            tuple[float, float]: Heading error and angular command.
        """
        goal_heading = float(np.arctan2(goal_pos[1] - robot_pos[1], goal_pos[0] - robot_pos[0]))
        heading_err = _wrap_angle(goal_heading - heading)
        angular = float(
            np.clip(
                float(self.config.heading_gain) * heading_err,
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
        )
        return heading_err, angular

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return a unicycle command using wait/approach/commit gap logic."""
        robot_pos, heading, goal_pos, ped_pos, ped_vel = self._extract_state(observation)
        goal_dist = float(np.linalg.norm(goal_pos - robot_pos))
        if goal_dist <= float(self.config.goal_tolerance):
            self._commit_steps_remaining = 0
            return 0.0, 0.0

        heading_err, angular = self._heading_command(robot_pos, heading, goal_pos)
        if abs(heading_err) >= float(self.config.turn_in_place_angle):
            self._commit_steps_remaining = 0
            return 0.0, angular

        along, cross, vel_goal_frame = self._goal_frame(
            robot_pos=robot_pos,
            goal_pos=goal_pos,
            ped_pos=ped_pos,
            ped_vel=ped_vel,
        )
        gap_start, blocked_now, min_distance = self._gap_start_time(
            along=along,
            cross=cross,
            vel_goal_frame=vel_goal_frame,
        )

        if np.isfinite(min_distance) and min_distance <= float(self.config.emergency_clearance):
            self._commit_steps_remaining = 0
            return float(self.config.wait_speed), angular

        if self._commit_steps_remaining > 0 and not blocked_now:
            self._commit_steps_remaining -= 1
            return float(self.config.commit_speed), angular

        if gap_start == 0.0:
            self._commit_steps_remaining = max(int(self.config.commit_hold_steps) - 1, 0)
            return float(self.config.commit_speed), angular

        if gap_start is not None and gap_start <= float(self.config.approach_gap_time):
            self._commit_steps_remaining = 0
            return float(self.config.approach_speed), angular

        if blocked_now:
            self._commit_steps_remaining = 0
            return float(self.config.wait_speed), angular

        self._commit_steps_remaining = 0
        return float(self.config.creep_speed), angular


def build_stream_gap_config(cfg: dict[str, Any] | None) -> StreamGapPlannerConfig:
    """Build :class:`StreamGapPlannerConfig` from a mapping.

    Returns:
        StreamGapPlannerConfig: Parsed planner configuration.
    """
    if not isinstance(cfg, dict):
        return StreamGapPlannerConfig()
    return StreamGapPlannerConfig(
        max_linear_speed=float(cfg.get("max_linear_speed", 1.2)),
        max_angular_speed=float(cfg.get("max_angular_speed", 1.2)),
        goal_tolerance=float(cfg.get("goal_tolerance", 0.25)),
        heading_gain=float(cfg.get("heading_gain", 1.6)),
        turn_in_place_angle=float(cfg.get("turn_in_place_angle", 0.7)),
        forward_lookahead=float(cfg.get("forward_lookahead", 4.0)),
        rear_margin=float(cfg.get("rear_margin", 0.5)),
        corridor_half_width=float(cfg.get("corridor_half_width", 0.85)),
        emergency_clearance=float(cfg.get("emergency_clearance", 0.55)),
        sample_dt=float(cfg.get("sample_dt", 0.2)),
        sample_horizon=float(cfg.get("sample_horizon", 4.0)),
        safe_gap_time=float(cfg.get("safe_gap_time", 1.0)),
        approach_gap_time=float(cfg.get("approach_gap_time", 0.8)),
        wait_speed=float(cfg.get("wait_speed", 0.0)),
        creep_speed=float(cfg.get("creep_speed", 0.12)),
        approach_speed=float(cfg.get("approach_speed", 0.35)),
        commit_speed=float(cfg.get("commit_speed", 0.95)),
        commit_hold_steps=int(cfg.get("commit_hold_steps", 6)),
        uncertainty_gating_enabled=bool(cfg.get("uncertainty_gating_enabled", False)),
        uncertainty_min_existence_probability=float(
            cfg.get("uncertainty_min_existence_probability", 0.5)
        ),
        uncertainty_min_position_confidence=float(
            cfg.get("uncertainty_min_position_confidence", 0.5)
        ),
        uncertainty_min_class_probability=float(cfg.get("uncertainty_min_class_probability", 0.5)),
        uncertainty_max_position_variance=float(cfg.get("uncertainty_max_position_variance", 1.0)),
    )


__all__ = ["StreamGapPlannerAdapter", "StreamGapPlannerConfig", "build_stream_gap_config"]
