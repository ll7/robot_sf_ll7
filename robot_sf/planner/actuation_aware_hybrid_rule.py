"""Diagnostic actuation-aware wrapper for the deterministic hybrid-rule planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    HybridRuleLocalPlannerConfig,
    build_hybrid_rule_local_planner_config,
)

_EPS = 1e-9


def _coerce_bool(key: str, value: Any) -> bool:
    """Parse booleans from YAML/CLI-style values without Python truthiness traps.

    Returns:
        Parsed boolean value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"Expected boolean-compatible value for actuation-aware config '{key}'.")


@dataclass(frozen=True)
class ActuationAwareHybridRuleConfig:
    """Synthetic actuation projection settings for a diagnostic local planner."""

    base_planner: HybridRuleLocalPlannerConfig
    projection_profile_name: str = "amv-actuation-stress-v0"
    projection_profile_version: str = "v0"
    claim_scope: str = "synthetic-only"
    diagnostic_only: bool = True
    calibrated_hardware_evidence: bool = False
    max_linear_accel_m_s2: float = 2.0
    max_linear_decel_m_s2: float = 2.5
    max_yaw_rate_rad_s: float = 1.2
    max_angular_accel_rad_s2: float = 4.0
    projection_dt: float = 0.1


class ActuationAwareHybridRuleAdapter:
    """Project hybrid-rule commands through a synthetic actuation envelope.

    The wrapper is intentionally diagnostic-only. It does not change benchmark
    success or collision interpretation; it exposes command projection
    diagnostics so AMV actuation stress can be inspected separately.
    """

    def __init__(
        self,
        config: ActuationAwareHybridRuleConfig,
        *,
        base_planner: Any | None = None,
    ) -> None:
        """Initialize the wrapped deterministic planner and projection state."""
        self.config = config
        self._base_planner = base_planner or HybridRuleLocalPlannerAdapter(
            config=config.base_planner
        )
        self.reset()

    def bind_env(self, env: Any) -> None:
        """Bind environment geometry to the wrapped planner when available."""
        bind = getattr(self._base_planner, "bind_env", None)
        if callable(bind):
            bind(env)

    def reset(self, *, seed: int | None = None) -> None:
        """Reset wrapped planner state plus per-episode projection counters."""
        reset = getattr(self._base_planner, "reset", None)
        if callable(reset):
            try:
                reset(seed=seed)
            except TypeError:
                reset()
        self._last_projected_command = (0.0, 0.0)
        self._step_count = 0
        self._projected_count = 0
        self._linear_accel_limited_count = 0
        self._angular_accel_limited_count = 0
        self._yaw_rate_limited_count = 0
        self._sum_abs_delta_linear = 0.0
        self._sum_abs_delta_angular = 0.0
        self._max_abs_delta_linear = 0.0
        self._max_abs_delta_angular = 0.0
        self._last_projection: dict[str, Any] | None = None

    def close(self) -> None:
        """Close wrapped planner resources when the planner owns any."""
        close = getattr(self._base_planner, "close", None)
        if callable(close):
            close()

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return a synthetic-actuation-projected hybrid-rule command."""
        requested = self._base_planner.plan(observation)
        projected, projection = self._project_command(
            (float(requested[0]), float(requested[1])),
            dt=self._projection_dt(observation),
        )
        self._record_projection(projection)
        return projected

    def diagnostics(self) -> dict[str, Any]:
        """Return projection diagnostics and wrapped-planner diagnostics."""
        wrapped_diagnostics = None
        diagnostics = getattr(self._base_planner, "diagnostics", None)
        if callable(diagnostics):
            wrapped_diagnostics = diagnostics()
        return {
            "planner_variant": "actuation_aware_hybrid_rule_v0",
            "wrapped_planner": "hybrid_rule_local_planner",
            "diagnostic_only": bool(self.config.diagnostic_only),
            "calibrated_hardware_evidence": bool(self.config.calibrated_hardware_evidence),
            "actuation_projection": self._projection_summary(),
            "wrapped_planner_diagnostics": wrapped_diagnostics,
            "last_projection": dict(self._last_projection) if self._last_projection else None,
        }

    def last_decision(self) -> dict[str, Any] | None:
        """Return the latest projection and wrapped-planner decision diagnostics."""
        base_last = getattr(self._base_planner, "last_decision", None)
        wrapped = base_last() if callable(base_last) else None
        if self._last_projection is None and wrapped is None:
            return None
        return {
            "planner_variant": "actuation_aware_hybrid_rule_v0",
            "actuation_projection": dict(self._last_projection) if self._last_projection else None,
            "wrapped_decision": wrapped,
        }

    def _projection_dt(self, observation: dict[str, Any]) -> float:
        """Return the projection timestep from observation metadata or config."""
        sim = observation.get("sim") if isinstance(observation.get("sim"), dict) else {}
        raw = sim.get("timestep", self.config.projection_dt)
        try:
            dt = float(np.asarray(raw, dtype=float).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            dt = float(self.config.projection_dt)
        if not np.isfinite(dt) or dt <= 0.0:
            dt = float(self.config.projection_dt)
        return max(float(dt), 1e-6)

    def _project_command(
        self,
        requested_command: tuple[float, float],
        *,
        dt: float,
    ) -> tuple[tuple[float, float], dict[str, Any]]:
        """Project one absolute unicycle command through synthetic limits.

        Returns:
            Projected command and JSON-safe per-step projection diagnostics.
        """
        current_linear, current_angular = self._last_projected_command
        requested_linear = float(requested_command[0])
        requested_angular = float(requested_command[1])

        allowed_linear_up = float(self.config.max_linear_accel_m_s2) * dt
        allowed_linear_down = float(self.config.max_linear_decel_m_s2) * dt
        requested_linear_delta = requested_linear - current_linear
        if requested_linear_delta >= 0.0:
            projected_linear = current_linear + min(requested_linear_delta, allowed_linear_up)
        else:
            projected_linear = current_linear + max(requested_linear_delta, -allowed_linear_down)

        bounded_requested_angular = float(
            np.clip(
                requested_angular,
                -float(self.config.max_yaw_rate_rad_s),
                float(self.config.max_yaw_rate_rad_s),
            )
        )
        allowed_angular_delta = float(self.config.max_angular_accel_rad_s2) * dt
        requested_angular_delta = bounded_requested_angular - current_angular
        projected_angular = current_angular + float(
            np.clip(requested_angular_delta, -allowed_angular_delta, allowed_angular_delta)
        )

        projected = (float(projected_linear), float(projected_angular))
        delta_linear = abs(projected[0] - requested_linear)
        delta_angular = abs(projected[1] - requested_angular)
        linear_limited = delta_linear > _EPS
        angular_limited = abs(projected[1] - bounded_requested_angular) > _EPS
        yaw_rate_limited = abs(bounded_requested_angular - requested_angular) > _EPS
        return projected, {
            "schema_version": "actuation-aware-command-projection.v0",
            "profile": self._profile_metadata(),
            "dt": float(dt),
            "requested_command": [requested_linear, requested_angular],
            "previous_projected_command": [float(current_linear), float(current_angular)],
            "projected_command": [float(projected[0]), float(projected[1])],
            "linear_accel_limited": bool(linear_limited),
            "angular_accel_limited": bool(angular_limited),
            "yaw_rate_limited": bool(yaw_rate_limited),
            "projected": bool(linear_limited or angular_limited or yaw_rate_limited),
            "abs_delta_linear": float(delta_linear),
            "abs_delta_angular": float(delta_angular),
        }

    def _record_projection(self, projection: dict[str, Any]) -> None:
        """Accumulate projection counters for episode diagnostics."""
        self._step_count += 1
        if bool(projection["projected"]):
            self._projected_count += 1
        if bool(projection["linear_accel_limited"]):
            self._linear_accel_limited_count += 1
        if bool(projection["angular_accel_limited"]):
            self._angular_accel_limited_count += 1
        if bool(projection["yaw_rate_limited"]):
            self._yaw_rate_limited_count += 1
        delta_linear = float(projection["abs_delta_linear"])
        delta_angular = float(projection["abs_delta_angular"])
        self._sum_abs_delta_linear += delta_linear
        self._sum_abs_delta_angular += delta_angular
        self._max_abs_delta_linear = max(self._max_abs_delta_linear, delta_linear)
        self._max_abs_delta_angular = max(self._max_abs_delta_angular, delta_angular)
        projected = projection["projected_command"]
        self._last_projected_command = (float(projected[0]), float(projected[1]))
        self._last_projection = dict(projection)

    def _projection_summary(self) -> dict[str, Any]:
        """Return JSON-safe aggregate projection diagnostics."""
        if self._step_count <= 0:
            return {
                "schema_version": "actuation-aware-command-projection-summary.v0",
                "status": "not_available",
                "profile": self._profile_metadata(),
                "step_count": 0,
            }
        return {
            "schema_version": "actuation-aware-command-projection-summary.v0",
            "status": "ok",
            "profile": self._profile_metadata(),
            "step_count": int(self._step_count),
            "projection_count": int(self._projected_count),
            "projection_fraction": float(self._projected_count / self._step_count),
            "linear_accel_limited_fraction": float(
                self._linear_accel_limited_count / self._step_count
            ),
            "angular_accel_limited_fraction": float(
                self._angular_accel_limited_count / self._step_count
            ),
            "yaw_rate_limited_fraction": float(self._yaw_rate_limited_count / self._step_count),
            "mean_abs_delta_linear": float(self._sum_abs_delta_linear / self._step_count),
            "mean_abs_delta_angular": float(self._sum_abs_delta_angular / self._step_count),
            "max_abs_delta_linear": float(self._max_abs_delta_linear),
            "max_abs_delta_angular": float(self._max_abs_delta_angular),
        }

    def _profile_metadata(self) -> dict[str, Any]:
        """Return the synthetic projection profile metadata."""
        return {
            "name": str(self.config.projection_profile_name),
            "profile_version": str(self.config.projection_profile_version),
            "claim_scope": str(self.config.claim_scope),
            "max_linear_accel_m_s2": float(self.config.max_linear_accel_m_s2),
            "max_linear_decel_m_s2": float(self.config.max_linear_decel_m_s2),
            "max_yaw_rate_rad_s": float(self.config.max_yaw_rate_rad_s),
            "max_angular_accel_rad_s2": float(self.config.max_angular_accel_rad_s2),
        }


def build_actuation_aware_hybrid_rule_config(
    cfg: dict[str, Any] | None,
) -> ActuationAwareHybridRuleConfig:
    """Build the diagnostic actuation-aware hybrid-rule config from YAML.

    Returns:
        Parsed actuation-aware planner wrapper config.
    """
    raw = dict(cfg or {})
    return ActuationAwareHybridRuleConfig(
        base_planner=build_hybrid_rule_local_planner_config(raw),
        projection_profile_name=str(raw.get("projection_profile_name", "amv-actuation-stress-v0")),
        projection_profile_version=str(raw.get("projection_profile_version", "v0")),
        claim_scope=str(raw.get("claim_scope", "synthetic-only")),
        diagnostic_only=_coerce_bool("diagnostic_only", raw.get("diagnostic_only", True)),
        calibrated_hardware_evidence=_coerce_bool(
            "calibrated_hardware_evidence",
            raw.get("calibrated_hardware_evidence", False),
        ),
        max_linear_accel_m_s2=float(raw.get("max_linear_accel_m_s2", 2.0)),
        max_linear_decel_m_s2=float(raw.get("max_linear_decel_m_s2", 2.5)),
        max_yaw_rate_rad_s=float(raw.get("max_yaw_rate_rad_s", 1.2)),
        max_angular_accel_rad_s2=float(raw.get("max_angular_accel_rad_s2", 4.0)),
        projection_dt=float(raw.get("projection_dt", 0.1)),
    )


__all__ = [
    "ActuationAwareHybridRuleAdapter",
    "ActuationAwareHybridRuleConfig",
    "build_actuation_aware_hybrid_rule_config",
]
