"""Shared schema constants and dataclasses for behavior-token diagnostics.

This module is intentionally dependency-light (standard library only) so it can be
imported by the offline extraction, quantization, and inspection scripts as well as
by tests. It defines the stable feature vocabulary and schema-version strings that
downstream artifacts embed for provenance.

Claim boundary: everything here supports *diagnostic* tooling only. Token ids and
features are exploratory descriptors, not validated safety metrics or planner-ranking
evidence. See ``README.md`` in this directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Schema versions embedded in generated artifacts so consumers can detect drift.
FEATURE_SCHEMA_VERSION = "behavior-token-features.v1"
WINDOW_SCHEMA_VERSION = "behavior-token-window.v1"
QUANTIZER_SCHEMA_VERSION = "behavior-token-quantizer.v1"
INSPECTION_SCHEMA_VERSION = "behavior-token-inspection.v1"

# One-line claim boundary reused across artifacts and CLI ``--help`` epilogues.
CLAIM_BOUNDARY = (
    "Experimental, low-priority diagnostic tooling. Behavior tokens are exploratory "
    "descriptors of saved interaction traces; they are NOT validated safety metrics, "
    "benchmark evidence, or paper/dissertation claim support. No safety decision may "
    "depend on these tokens."
)

# Ordered, stable feature vocabulary. The order is part of the schema: quantizer
# normalization statistics and token centers are recorded against this ordering.
# ``None`` (JSON null) marks a feature that could not be derived for a window; such
# features are also listed in the window's ``missing_features``. Genuinely
# zero-valued measurements (for example, an all-straight command sequence giving an
# oscillation rate of 0.0) are recorded as ``0.0`` and are NOT treated as missing.
FEATURE_NAMES: tuple[str, ...] = (
    "clearance_min_m",
    "clearance_mean_m",
    "clearance_slope_m_per_s",
    "rel_speed_to_nearest_mean_m_s",
    "ttc_proxy_min_s",
    "ttc_proxy_slope_s_per_s",
    "robot_speed_mean_m_s",
    "robot_speed_min_m_s",
    "robot_speed_max_m_s",
    "robot_accel_rms_m_s2",
    "command_change_rms",
    "ped_speed_change_near_robot_m_s",
    "route_progress_delta_m",
    "stop_yield_fraction",
    "oscillation_rate",
    "near_conflict_recovery_m",
)

# Human-facing descriptions used in motif reports and the README feature table.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "clearance_min_m": "Minimum robot-to-nearest-pedestrian distance over the window (metres).",
    "clearance_mean_m": "Mean robot-to-nearest-pedestrian distance over the window (metres).",
    "clearance_slope_m_per_s": "Linear-fit slope of nearest-pedestrian distance vs time (m/s).",
    "rel_speed_to_nearest_mean_m_s": "Mean relative speed to the nearest pedestrian (m/s); "
    "requires pedestrian velocities.",
    "ttc_proxy_min_s": "Minimum time-to-contact proxy from the closing distance series (s).",
    "ttc_proxy_slope_s_per_s": "Linear-fit slope of the time-to-contact proxy vs time.",
    "robot_speed_mean_m_s": "Mean robot speed over the window (m/s).",
    "robot_speed_min_m_s": "Minimum robot speed over the window (m/s).",
    "robot_speed_max_m_s": "Maximum robot speed over the window (m/s).",
    "robot_accel_rms_m_s2": "RMS of robot speed change per second (acceleration proxy, m/s^2).",
    "command_change_rms": "RMS step-to-step change in the commanded linear velocity.",
    "ped_speed_change_near_robot_m_s": "Mean absolute nearest-pedestrian speed change while near "
    "the robot (m/s); requires pedestrian velocities.",
    "route_progress_delta_m": "Route progress change over the window; null when not present in "
    "the trace.",
    "stop_yield_fraction": "Fraction of steps with low robot speed while a pedestrian is near "
    "(stop/yield proxy).",
    "oscillation_rate": "Rate of angular-command (or heading) sign changes per step "
    "(negotiation/oscillation proxy).",
    "near_conflict_recovery_m": "Clearance recovered after the minimum-clearance step (m); null "
    "when the minimum is at the window end.",
}

# Diagnostic thresholds (documented so token labels remain auditable).
STOP_SPEED_THRESHOLD_M_S = 0.15
NEAR_PEDESTRIAN_THRESHOLD_M = 2.0


@dataclass
class WindowRecord:
    """One trajectory window with stable metadata plus its feature vector.

    Attributes mirror the JSONL row layout written by ``extract_windows.py``.
    ``features`` maps every name in :data:`FEATURE_NAMES` to a float or ``None``.
    ``missing_features`` lists the names whose value is ``None`` (not derivable),
    which is distinct from a genuine zero measurement.
    """

    window_id: str
    source_episode_path: str
    episode_id: str | None
    scenario_id: str
    planner_key: str
    seed: Any
    t_start_s: float | None
    t_end_s: float | None
    step_start: int
    step_end: int
    n_steps: int
    row_status: str | None
    outcome: str | None
    feature_schema_version: str
    features: dict[str, float | None]
    missing_features: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict for one ``windows.jsonl`` row."""
        return {
            "window_id": self.window_id,
            "source_episode_path": self.source_episode_path,
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "planner_key": self.planner_key,
            "seed": self.seed,
            "t_start_s": self.t_start_s,
            "t_end_s": self.t_end_s,
            "step_start": self.step_start,
            "step_end": self.step_end,
            "n_steps": self.n_steps,
            "row_status": self.row_status,
            "outcome": self.outcome,
            "feature_schema_version": self.feature_schema_version,
            "features": {name: self.features.get(name) for name in FEATURE_NAMES},
            "missing_features": list(self.missing_features),
        }


def feature_names() -> list[str]:
    """Return the ordered feature vocabulary as a fresh list."""
    return list(FEATURE_NAMES)
