"""Pedestrian desired-speed tier axis (issue #4972).

This module exposes a speed-tier *axis* that scenarios and campaigns can use to
quantify how planner rankings shift with pedestrian pace. Each tier maps to a
literature-calibrated desired (preferred) walking-speed distribution, decoupled
from the spawn speed.

Tiers:
    - ``slow``: ~0.65 m/s. Matches the legacy spawn-coupled default
      (``max_speed_multiplier * initial_speed`` = 1.3 * 0.5). Keep this tier so
      existing benchmark numbers remain reproducible as an explicit axis value.
    - ``typical``: ~1.3 m/s. Preferred walking speed reported for unimpeded
      adults (Moussaïd et al. 2010, "The walking behaviour of pedestrian
      social groups", doi:10.1371/journal.pone.0010047).
    - ``brisk``: ~1.6 m/s. Fast walkers; stress-tests reactive planners.

All tiers use a 0.2 m/s truncated-normal spread. The legacy default
(``ped_speed_tier is None``) preserves the spawn-coupled ~0.65 m/s behavior
unchanged until a major campaign re-base.
"""

from __future__ import annotations

import numpy as np

PED_SPEED_TIER_SLOW = "slow"
PED_SPEED_TIER_TYPICAL = "typical"
PED_SPEED_TIER_BRISK = "brisk"
SUPPORTED_PED_SPEED_TIERS = frozenset(
    {PED_SPEED_TIER_SLOW, PED_SPEED_TIER_TYPICAL, PED_SPEED_TIER_BRISK}
)

#: Shared truncated-normal spread (m/s) for all tiers.
PED_SPEED_TIER_STD = 0.2

#: Inclusive upper bound (m/s) for the truncated desired-speed distribution.
PED_SPEED_TIER_HIGH = 3.0

#: Tier -> (desired_speed_mean, desired_speed_std). ``slow`` mirrors the legacy
#: ~0.65 m/s default so existing benchmark numbers stay reproducible as a tier.
_PED_SPEED_TIER_PARAMS: dict[str, tuple[float, float]] = {
    PED_SPEED_TIER_SLOW: (0.65, PED_SPEED_TIER_STD),
    PED_SPEED_TIER_TYPICAL: (1.3, PED_SPEED_TIER_STD),
    PED_SPEED_TIER_BRISK: (1.6, PED_SPEED_TIER_STD),
}


def normalize_ped_speed_tier(value: str | None) -> str | None:
    """Return a supported pedestrian speed-tier key, ``None``, or raise.

    Args:
        value: Tier key (``slow``/``typical``/``brisk``) or ``None`` for the
            legacy spawn-coupled default.

    Returns:
        The normalized tier key, or ``None`` when ``value`` is ``None``.
    """
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in SUPPORTED_PED_SPEED_TIERS:
        return normalized
    supported = ", ".join(sorted(SUPPORTED_PED_SPEED_TIERS))
    raise ValueError(f"Unsupported ped_speed_tier {value!r}. Supported values: {supported}.")


def desired_speed_params_for_tier(tier: str | None) -> tuple[float | None, float | None]:
    """Map a speed tier to a ``(desired_speed_mean, desired_speed_std)`` pair.

    Args:
        tier: Tier key (``slow``/``typical``/``brisk``) or ``None`` for the
            legacy spawn-coupled default.

    Returns:
        ``(mean, std)`` for the tier, or ``(None, None)`` when ``tier`` is
        ``None`` (signals: keep the legacy ``max_speed_multiplier * initial_speed``
        derivation).
    """
    normalized = normalize_ped_speed_tier(tier)
    if normalized is None:
        return None, None
    mean, std = _PED_SPEED_TIER_PARAMS[normalized]
    return mean, std


def sample_desired_pedestrian_speeds(
    num_peds: int,
    mean: float,
    std: float | None = None,
    high: float = PED_SPEED_TIER_HIGH,
    seed: int | None = None,
) -> np.ndarray:
    """Sample per-pedestrian desired walking speeds from a truncated normal distribution.

    Standalone sampling that does not depend on the pysocialforce package version.
    Produces values identical to ``pysocialforce.scene.sample_truncated_normal_speeds``
    when given the same parameters (issue #5217 compat layer).

    Args:
        num_peds: Number of pedestrians to sample speeds for.
        mean: Desired-speed distribution mean (m/s).
        std: Standard deviation (m/s). ``None`` defaults to ``PED_SPEED_TIER_STD``
            (0.2 m/s), matching the pysf ``DEFAULT_DESIRED_SPEED_STD`` constant.
        high: Inclusive upper-bound clip for the distribution (m/s).
        seed: Optional RNG seed for deterministic sampling.

    Returns:
        np.ndarray: Non-negative desired speeds, shape ``(num_peds,)``.
    """
    if num_peds <= 0:
        return np.zeros(0, dtype=float)
    std_eff = PED_SPEED_TIER_STD if std is None else float(std)
    rng = np.random.default_rng(seed)
    if std_eff > 0.0:
        speeds = rng.normal(loc=float(mean), scale=std_eff, size=num_peds)
    else:
        speeds = np.full(num_peds, float(mean), dtype=float)
    return np.clip(speeds, 0.0, float(high))
