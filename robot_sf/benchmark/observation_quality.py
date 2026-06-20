"""Observation-quality metadata shared by ODD and forecast artifacts.

The fields describe simulator observation assumptions for benchmark evidence
boundaries.  They are not hardware calibration data and must not be used as
sensor-certification claims.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


def _require_string_list(name: str, value: object) -> list[str]:
    """Return a non-empty list of non-empty strings."""

    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty list")
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized):
        raise ValueError(f"{name} entries must be non-empty strings")
    return normalized


def _require_probability(name: str, value: object) -> float:
    """Return a probability in the closed interval [0, 1]."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return normalized


def _require_non_negative_float(name: str, value: object) -> float:
    """Return a non-negative float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite number >= 0")
    return normalized


def _normalize_range_limit(value: object) -> float | None:
    """Normalize an optional range limit in meters.

    Returns:
        Positive range limit in meters, or None when unbounded.
    """

    if value is None:
        return None
    normalized = _require_non_negative_float("range_limit_m", value)
    if normalized == 0.0:
        raise ValueError("range_limit_m must be > 0 when present")
    return normalized


@dataclass(frozen=True, slots=True)
class ObservationQuality:
    """Simulator observation-quality assumptions for evidence boundaries."""

    visibility: list[str]
    occlusion: list[str]
    latency_s: float
    dropout_probability: float
    range_limit_m: float | None
    angular_noise_std_rad: float
    false_negative_rate: float
    false_positive_rate: float
    notes: str

    def __post_init__(self) -> None:
        """Validate observation-quality fields fail-closed."""

        object.__setattr__(self, "visibility", _require_string_list("visibility", self.visibility))
        object.__setattr__(self, "occlusion", _require_string_list("occlusion", self.occlusion))
        object.__setattr__(
            self, "latency_s", _require_non_negative_float("latency_s", self.latency_s)
        )
        object.__setattr__(
            self,
            "dropout_probability",
            _require_probability("dropout_probability", self.dropout_probability),
        )
        object.__setattr__(self, "range_limit_m", _normalize_range_limit(self.range_limit_m))
        object.__setattr__(
            self,
            "angular_noise_std_rad",
            _require_non_negative_float("angular_noise_std_rad", self.angular_noise_std_rad),
        )
        object.__setattr__(
            self,
            "false_negative_rate",
            _require_probability("false_negative_rate", self.false_negative_rate),
        )
        object.__setattr__(
            self,
            "false_positive_rate",
            _require_probability("false_positive_rate", self.false_positive_rate),
        )
        notes = str(self.notes).strip()
        if not notes:
            raise ValueError("notes must be a non-empty string")
        object.__setattr__(self, "notes", notes)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | ObservationQuality) -> ObservationQuality:
        """Build observation-quality metadata from JSON-compatible data.

        Returns:
            Validated observation-quality metadata.
        """

        if isinstance(data, ObservationQuality):
            return data
        if not isinstance(data, dict):
            raise ValueError("observation_quality must be an object")
        return cls(
            visibility=data.get("visibility", []),
            occlusion=data.get("occlusion", []),
            latency_s=data.get("latency_s"),
            dropout_probability=data.get("dropout_probability"),
            range_limit_m=data.get("range_limit_m"),
            angular_noise_std_rad=data.get("angular_noise_std_rad"),
            false_negative_rate=data.get("false_negative_rate"),
            false_positive_rate=data.get("false_positive_rate"),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe primitives."""

        return asdict(self)


__all__ = ["ObservationQuality"]
