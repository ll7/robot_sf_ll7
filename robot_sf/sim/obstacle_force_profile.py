"""Opt-in, immutable pedestrian wall-force parameters for release successors.

The default deliberately leaves the released fast-pysf configuration untouched.
Numerical calibration evidence is separate from the release's adoption decision.
"""

from pysocialforce.config import (
    LEGACY_SHIFTED_GRADIENT_V1,
    SURFACE_DISTANCE_UNIT_NORMAL_V2,
    ObstacleForceConfig,
)

LEGACY_PROFILE = "legacy_v1"
CALIBRATED_PROFILE = "calibrated_v2"
GRADIENT_PROFILE = "gradient_v3"


class _ResolvedProfile(str):
    """Preserve explicit selection through dataclass replacement and copying."""

    def __new__(cls, value: str, explicit: bool):
        instance = super().__new__(cls, value)
        instance.explicit = explicit
        return instance

    def __reduce_ex__(self, protocol: int):
        """Return the constructor arguments used by deepcopy and pickle."""
        del protocol
        return type(self), (str(self), self.explicit)


def resolve_obstacle_force_profile(value: str | None = None) -> _ResolvedProfile:
    """Resolve a profile identifier, rejecting malformed selectors.

    Returns:
        Canonical profile with explicit-selection provenance.
    """
    if value is None:
        return _ResolvedProfile(LEGACY_PROFILE, False)
    if isinstance(value, _ResolvedProfile):
        return value
    if not isinstance(value, str):
        raise TypeError("obstacle_force_profile must be a string or None")
    if value not in {LEGACY_PROFILE, CALIBRATED_PROFILE, GRADIENT_PROFILE}:
        raise ValueError(f"unsupported obstacle_force_profile: {value!r}")
    return _ResolvedProfile(value, True)


def apply_obstacle_force_profile(config: ObstacleForceConfig, value: str | None) -> None:
    """Apply a fitted parameter set without changing the legacy/default path."""
    profile = resolve_obstacle_force_profile(value)
    if profile == LEGACY_PROFILE:
        return
    if profile == GRADIENT_PROFILE:
        if (
            config.law_version != SURFACE_DISTANCE_UNIT_NORMAL_V2
            and config.obstacle_force_law_resolution_mode != "defaulted_missing"
        ):
            raise ValueError(
                "gradient_v3 requires obstacle_force_law=surface_distance_unit_normal_v2"
            )
        # Development Pareto candidate; matched-flow and release adoption remain gated.
        config.law_version = SURFACE_DISTANCE_UNIT_NORMAL_V2
        config.factor = 0.001
        config.threshold = 0.4
        config.sigma = 0.0
        return
    if config.law_version != LEGACY_SHIFTED_GRADIENT_V1:
        raise ValueError("calibrated_v2 requires obstacle_force_law=legacy_shifted_gradient_v1")
    # Rejected prototype: footprint and narrow-door gates fail. Opt-in diagnostics only.
    config.factor = 0.003
    config.threshold = 0.375
    config.sigma = 0.0
