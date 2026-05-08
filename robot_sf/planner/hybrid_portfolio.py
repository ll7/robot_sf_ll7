"""Hybrid portfolio planner combining risk-dwa, ORCA, and prediction heads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from loguru import logger

from robot_sf.planner.mppi_social import (
    MPPISocialConfig,
    MPPISocialPlannerAdapter,
    build_mppi_social_config,
)
from robot_sf.planner.risk_dwa import (
    RiskDWAPlannerAdapter,
    RiskDWAPlannerConfig,
    build_risk_dwa_config,
)
from robot_sf.planner.socnav import (
    ORCAPlannerAdapter,
    PredictionPlannerAdapter,
    SocNavPlannerConfig,
)


@dataclass
class HybridPortfolioConfig:
    """Configuration for risk-regime switching across planner heads."""

    emergency_clearance: float = 0.55
    caution_clearance: float = 1.0
    dense_ped_count: int = 6
    near_field_distance: float = 2.5
    hysteresis_steps: int = 6
    fallback_on_exception: bool = True


class HybridPortfolioAdapter:
    """Switch planner head by local risk level with hysteresis."""

    def __init__(
        self,
        *,
        hybrid_config: HybridPortfolioConfig,
        risk_dwa: RiskDWAPlannerAdapter,
        orca: ORCAPlannerAdapter,
        prediction: PredictionPlannerAdapter,
        mppi: MPPISocialPlannerAdapter | None = None,
    ) -> None:
        """Construct the hybrid adapter with all planner heads."""
        self.config = hybrid_config
        self.risk_dwa = risk_dwa
        self.orca = orca
        self.prediction = prediction
        self.mppi = mppi

        self._active_head = "risk_dwa"
        self._hold_remaining = 0
        self._selected_head_counts: dict[str, int] = {}
        self._fallback_count = 0
        self._steps = 0
        self._last_decision: dict[str, Any] | None = None

    def _extract_ped_clearance(self, observation: dict[str, Any]) -> tuple[int, float]:
        """Return `(near_count, min_clearance)` around the robot."""
        robot = observation.get("robot") if isinstance(observation.get("robot"), dict) else {}
        ped = (
            observation.get("pedestrians")
            if isinstance(observation.get("pedestrians"), dict)
            else {}
        )
        robot_pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=float)
        if robot_pos.shape != (2,):
            robot_pos = np.asarray([0.0, 0.0], dtype=float)
        ped_positions_raw = ped.get("positions")
        ped_pos = np.asarray([] if ped_positions_raw is None else ped_positions_raw, dtype=float)
        if ped_pos.ndim != 2 or ped_pos.shape[-1] != 2:
            return 0, float("inf")
        dists = np.linalg.norm(ped_pos - robot_pos[None, :], axis=1)
        if dists.size == 0:
            return 0, float("inf")
        near_count = int(np.count_nonzero(dists <= float(self.config.near_field_distance)))
        return near_count, float(np.min(dists))

    def _desired_head(self, observation: dict[str, Any]) -> str:
        """Choose the planner head that best matches current crowd clearance.

        Returns:
            str: Name of the desired portfolio head.
        """
        near_count, min_clearance = self._extract_ped_clearance(observation)
        if min_clearance <= float(self.config.emergency_clearance):
            return "orca"
        if near_count >= int(self.config.dense_ped_count) or min_clearance <= float(
            self.config.caution_clearance
        ):
            # In dense regimes, prefer predictive head when available.
            return "prediction"
        # Prefer deterministic Risk-DWA for open-space cruise; MPPI only in very open scenes.
        if (
            self.mppi is not None
            and near_count == 0
            and min_clearance >= float(self.config.caution_clearance) * 2.0
        ):
            return "mppi"
        return "risk_dwa"

    def _switch_head(self, desired: str) -> None:
        """Apply hysteresis when switching active planner heads."""
        emergency = desired == "orca"
        if self._active_head == desired:
            return
        if self._hold_remaining > 0 and not emergency:
            self._hold_remaining -= 1
            return
        self._active_head = desired
        self._hold_remaining = max(int(self.config.hysteresis_steps), 0)

    def _call_head(self, head: str, observation: dict[str, Any]) -> tuple[float, float]:
        """Dispatch planning to one portfolio head.

        Returns:
            tuple[float, float]: Unicycle command returned by the selected head.
        """
        if head == "risk_dwa":
            return self.risk_dwa.plan(observation)
        if head == "mppi" and self.mppi is not None:
            return self.mppi.plan(observation)
        if head == "prediction":
            return self.prediction.plan(observation)
        return self.orca.plan(observation)

    def _record_decision(
        self,
        *,
        desired_head: str,
        selected_head: str,
        fallback: bool = False,
        fallback_from: str | None = None,
        error: str | None = None,
    ) -> None:
        """Store JSON-safe diagnostics for one planner-head decision."""

        self._steps += 1
        self._selected_head_counts[selected_head] = (
            self._selected_head_counts.get(selected_head, 0) + 1
        )
        if fallback:
            self._fallback_count += 1
        self._last_decision = {
            "desired_head": desired_head,
            "selected_head": selected_head,
            "fallback": bool(fallback),
            "fallback_from": fallback_from,
            "error": error,
            "active_head": self._active_head,
            "hold_remaining": int(self._hold_remaining),
        }

    def reset(self) -> None:
        """Clear portfolio hysteresis and reset any stateful child heads."""
        self._active_head = "risk_dwa"
        self._hold_remaining = 0
        self._selected_head_counts.clear()
        self._fallback_count = 0
        self._steps = 0
        self._last_decision = None
        for head in (self.risk_dwa, self.orca, self.prediction, self.mppi):
            reset = getattr(head, "reset", None)
            if callable(reset):
                reset()

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return command from selected planner head."""
        desired = self._desired_head(observation)
        self._switch_head(desired)
        selected = self._active_head
        try:
            command = self._call_head(selected, observation)
            self._record_decision(desired_head=desired, selected_head=selected)
            return command
        except Exception as exc:
            if not bool(self.config.fallback_on_exception):
                raise
            logger.warning(
                "Hybrid portfolio head '{}' failed; falling back to ORCA.",
                selected,
            )
            self._active_head = "orca"
            self._record_decision(
                desired_head=desired,
                selected_head="orca",
                fallback=True,
                fallback_from=selected,
                error=str(exc),
            )
            return self.orca.plan(observation)

    def diagnostics(self) -> dict[str, Any]:
        """Return episode-local planner-head diagnostics.

        Returns:
            JSON-safe diagnostics for benchmark episode metadata.
        """

        return {
            "steps": int(self._steps),
            "active_head": self._active_head,
            "hold_remaining": int(self._hold_remaining),
            "selected_head_counts": dict(self._selected_head_counts),
            "fallback_count": int(self._fallback_count),
            "last_decision": deepcopy(self._last_decision) if self._last_decision else None,
        }

    def last_decision(self) -> dict[str, Any] | None:
        """Return the latest planner-head decision for step-level tooling.

        Returns a deep copy so callers can safely store, compare, or mutate the
        payload without affecting the planner's internal state, even if future
        decisions include nested structures (e.g. candidate score tables).

        Returns:
            Copy of the latest decision payload, or ``None`` before the first step/reset.
        """

        return deepcopy(self._last_decision) if self._last_decision else None


@dataclass
class HybridPortfolioBuildConfig:
    """Composite build config for hybrid head construction."""

    hybrid: HybridPortfolioConfig
    risk_dwa: RiskDWAPlannerConfig
    mppi: MPPISocialConfig
    socnav: SocNavPlannerConfig


def build_hybrid_portfolio_build_config(cfg: dict[str, Any] | None) -> HybridPortfolioBuildConfig:
    """Build merged hybrid + sub-head configurations from a mapping payload.

    Returns:
        HybridPortfolioBuildConfig: Fully parsed build configuration for hybrid planner heads.
    """
    cfg = cfg if isinstance(cfg, dict) else {}

    hybrid_raw = cfg.get("hybrid", {}) if isinstance(cfg.get("hybrid"), dict) else {}
    risk_raw = cfg.get("risk_dwa", {}) if isinstance(cfg.get("risk_dwa"), dict) else {}
    mppi_raw = cfg.get("mppi_social", {}) if isinstance(cfg.get("mppi_social"), dict) else {}

    # Keep SocNav-compatible keys in root to preserve existing ORCA/prediction config format.
    allowed = {f.name for f in fields(SocNavPlannerConfig)}
    socnav_kwargs = {k: v for k, v in cfg.items() if k in allowed}
    socnav = SocNavPlannerConfig(**socnav_kwargs)

    hybrid = HybridPortfolioConfig(
        emergency_clearance=float(hybrid_raw.get("emergency_clearance", 0.55)),
        caution_clearance=float(hybrid_raw.get("caution_clearance", 1.0)),
        dense_ped_count=int(hybrid_raw.get("dense_ped_count", 6)),
        near_field_distance=float(hybrid_raw.get("near_field_distance", 2.5)),
        hysteresis_steps=int(hybrid_raw.get("hysteresis_steps", 6)),
        fallback_on_exception=bool(hybrid_raw.get("fallback_on_exception", True)),
    )

    # Reuse full builders so all sub-head tuning keys remain available.
    risk = build_risk_dwa_config(risk_raw)
    mppi = build_mppi_social_config(mppi_raw)

    return HybridPortfolioBuildConfig(hybrid=hybrid, risk_dwa=risk, mppi=mppi, socnav=socnav)


__all__ = [
    "HybridPortfolioAdapter",
    "HybridPortfolioBuildConfig",
    "HybridPortfolioConfig",
    "build_hybrid_portfolio_build_config",
]
