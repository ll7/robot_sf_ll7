"""Hybrid portfolio planner combining risk-dwa, ORCA, and prediction heads."""

from __future__ import annotations

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
        ped_count, min_clearance = self._extract_ped_clearance(observation)
        if min_clearance <= float(self.config.emergency_clearance):
            return "orca"
        if ped_count >= int(self.config.dense_ped_count) or min_clearance <= float(
            self.config.caution_clearance
        ):
            # In dense regimes, prefer predictive head when available.
            return "prediction"
        # Prefer deterministic Risk-DWA for open-space cruise; MPPI only in very open scenes.
        if (
            self.mppi is not None
            and ped_count == 0
            and min_clearance >= float(self.config.caution_clearance) * 2.0
        ):
            return "mppi"
        return "risk_dwa"

    def _switch_head(self, desired: str) -> None:
        emergency = desired == "orca"
        if self._active_head == desired:
            return
        if self._hold_remaining > 0 and not emergency:
            self._hold_remaining -= 1
            return
        self._active_head = desired
        self._hold_remaining = max(int(self.config.hysteresis_steps), 0)

    def _call_head(self, head: str, observation: dict[str, Any]) -> tuple[float, float]:
        if head == "risk_dwa":
            return self.risk_dwa.plan(observation)
        if head == "mppi" and self.mppi is not None:
            return self.mppi.plan(observation)
        if head == "prediction":
            return self.prediction.plan(observation)
        return self.orca.plan(observation)

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return command from selected planner head."""
        desired = self._desired_head(observation)
        self._switch_head(desired)
        try:
            return self._call_head(self._active_head, observation)
        except Exception:
            if not bool(self.config.fallback_on_exception):
                raise
            logger.warning(
                "Hybrid portfolio head '{}' failed; falling back to ORCA.",
                self._active_head,
            )
            self._active_head = "orca"
            return self.orca.plan(observation)


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
