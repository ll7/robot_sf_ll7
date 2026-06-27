"""Diagnostic inventory for the two collision / near-miss definitions.

The benchmark codebase classifies collision and near-miss events two
mathematically different ways depending on the code path (see GitHub issue
#3724):

* **Clearance regime** (the benchmark metric,
  ``robot_sf/benchmark/metrics.py``): events use radius-aware *clearance*
  ``clearance = center_distance - (robot_radius + ped_radius)``.

  - collision  : ``clearance < 0.0``
  - near-miss   : ``0.0 <= clearance < near_miss_dist``

  This is the regime encoded by
  :func:`robot_sf.benchmark.thresholds.default_threshold_profile`.

* **Center-distance regime** (the SNQI proxy and policy-search validation,
  ``robot_sf/gym_env/snqi_proxy.py`` and
  ``scripts/validation/policy_search_common.py``): events use the *raw center
  distance* against the named constants directly.

  - collision  : ``center_distance < collision_dist``
  - near-miss   : ``collision_dist <= center_distance < near_miss_dist``

  This is the regime encoded by
  :func:`robot_sf.benchmark.thresholds.legacy_missing_threshold_profile`.

With the default geometry (``robot_radius = 1.0``, ``ped_radius = 0.4``) the
clearance regime's collision boundary sits at a center distance of ``1.4 m``
while the center-distance regime's sits at ``0.25 m`` -- a ~5x gap. The two
regimes therefore label the *same* geometry differently across a wide band.

This module is **diagnostic only**. It does not change any threshold, metric,
proxy, or validation behavior, and it deliberately does **not** choose a
canonical definition (that decision is tracked as ``decision-required`` on
issue #3724). It exists to make the divergence explicit and machine-checkable
so the inconsistency cannot silently drift further or be assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.benchmark.constants import COLLISION_DIST, NEAR_MISS_DIST

if TYPE_CHECKING:
    from collections.abc import Iterable

# Default agent geometry, mirrored from ``EpisodeData`` defaults in
# ``robot_sf/benchmark/metrics.py`` (robot_radius=1.0, ped_radius=0.4) so the
# clearance regime here matches what the benchmark metric actually computes.
DEFAULT_ROBOT_RADIUS: float = 1.0
DEFAULT_PED_RADIUS: float = 0.4

# Event labels. Kept as plain strings so inventory payloads are trivially
# JSON-serializable for preflight reports.
LABEL_COLLISION = "collision"
LABEL_NEAR_MISS = "near_miss"
LABEL_CLEAR = "clear"

#: Regime identifiers; the values intentionally match the ``profile_id`` family
#: of the canonical threshold profiles in ``robot_sf/benchmark/thresholds.py``.
REGIME_CLEARANCE = "clearance_benchmark_metric"
REGIME_CENTER_DISTANCE = "center_distance_proxy_validation"


def classify_clearance_regime(
    center_distance: float,
    *,
    radius_sum: float,
    near_miss_dist: float = NEAR_MISS_DIST,
) -> str:
    """Classify one center distance under the radius-aware clearance regime.

    Args:
        center_distance: Robot-pedestrian center-to-center distance in meters.
        radius_sum: ``robot_radius + ped_radius`` in meters.
        near_miss_dist: Upper clearance bound for a near-miss (exclusive).

    Returns:
        One of :data:`LABEL_COLLISION`, :data:`LABEL_NEAR_MISS`,
        :data:`LABEL_CLEAR`.
    """
    clearance = float(center_distance) - float(radius_sum)
    if clearance < 0.0:
        return LABEL_COLLISION
    if clearance < float(near_miss_dist):
        return LABEL_NEAR_MISS
    return LABEL_CLEAR


def classify_center_distance_regime(
    center_distance: float,
    *,
    collision_dist: float = COLLISION_DIST,
    near_miss_dist: float = NEAR_MISS_DIST,
) -> str:
    """Classify one center distance under the raw center-distance regime.

    Args:
        center_distance: Robot-pedestrian center-to-center distance in meters.
        collision_dist: Center-distance collision threshold (exclusive upper
            bound for collision; inclusive lower bound for near-miss).
        near_miss_dist: Center-distance near-miss upper bound (exclusive).

    Returns:
        One of :data:`LABEL_COLLISION`, :data:`LABEL_NEAR_MISS`,
        :data:`LABEL_CLEAR`.
    """
    distance = float(center_distance)
    if distance < float(collision_dist):
        return LABEL_COLLISION
    if distance < float(near_miss_dist):
        return LABEL_NEAR_MISS
    return LABEL_CLEAR


@dataclass(frozen=True)
class DefinitionInventory:
    """Structured comparison of the two collision/near-miss definitions.

    All counts are over the supplied center-distance samples. ``divergent`` is
    the number of samples whose clearance-regime label differs from its
    center-distance-regime label.
    """

    sample_count: int
    radius_sum: float
    collision_dist: float
    near_miss_dist: float
    clearance_counts: dict[str, int]
    center_distance_counts: dict[str, int]
    divergent: int
    divergent_examples: list[dict[str, float | str]] = field(default_factory=list)

    @property
    def divergent_fraction(self) -> float:
        """Fraction of samples whose two regime labels disagree."""
        return self.divergent / self.sample_count if self.sample_count else 0.0

    @property
    def regimes_agree(self) -> bool:
        """Whether the two regimes labeled every sample identically."""
        return self.divergent == 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view for preflight reporting."""
        return {
            "sample_count": self.sample_count,
            "radius_sum_m": self.radius_sum,
            "collision_dist_m": self.collision_dist,
            "near_miss_dist_m": self.near_miss_dist,
            "regimes": {
                REGIME_CLEARANCE: {
                    "definition": (
                        "collision: clearance < 0; "
                        "near_miss: 0 <= clearance < near_miss_dist; "
                        "clearance = center_distance - radius_sum"
                    ),
                    "collision_center_distance_boundary_m": self.radius_sum,
                    "near_miss_center_distance_upper_m": (self.radius_sum + self.near_miss_dist),
                    "counts": dict(self.clearance_counts),
                    "source": "robot_sf/benchmark/metrics.py",
                },
                REGIME_CENTER_DISTANCE: {
                    "definition": (
                        "collision: center_distance < collision_dist; "
                        "near_miss: collision_dist <= center_distance < near_miss_dist"
                    ),
                    "collision_center_distance_boundary_m": self.collision_dist,
                    "near_miss_center_distance_upper_m": self.near_miss_dist,
                    "counts": dict(self.center_distance_counts),
                    "source": (
                        "robot_sf/gym_env/snqi_proxy.py, scripts/validation/policy_search_common.py"
                    ),
                },
            },
            "divergent": self.divergent,
            "divergent_fraction": self.divergent_fraction,
            "regimes_agree": self.regimes_agree,
            "divergent_examples": list(self.divergent_examples),
        }


def collision_definition_inventory(
    center_distances: Iterable[float],
    *,
    robot_radius: float = DEFAULT_ROBOT_RADIUS,
    ped_radius: float = DEFAULT_PED_RADIUS,
    collision_dist: float = COLLISION_DIST,
    near_miss_dist: float = NEAR_MISS_DIST,
    max_examples: int = 10,
) -> DefinitionInventory:
    """Inventory how the two regimes label a set of center distances.

    This is the core diagnostic: it labels every supplied center distance under
    both the clearance regime and the center-distance regime, counts each
    regime's labels, and records where the two disagree.

    Args:
        center_distances: Robot-pedestrian center distances in meters. May be
            any iterable (a flat list, or a flattened distance matrix).
        robot_radius: Robot radius in meters.
        ped_radius: Pedestrian radius in meters.
        collision_dist: Center-distance regime collision threshold.
        near_miss_dist: Shared near-miss upper bound. In the clearance regime
            it bounds *clearance*; in the center-distance regime it bounds
            *center distance* -- this asymmetry is itself part of the
            divergence and is preserved here intentionally.
        max_examples: Maximum number of divergent samples to record verbatim.

    Returns:
        A :class:`DefinitionInventory` describing both regimes and their
        disagreement.
    """
    distances = np.asarray(list(center_distances), dtype=float).ravel()
    radius_sum = float(robot_radius) + float(ped_radius)

    clearance_counts = {LABEL_COLLISION: 0, LABEL_NEAR_MISS: 0, LABEL_CLEAR: 0}
    center_counts = {LABEL_COLLISION: 0, LABEL_NEAR_MISS: 0, LABEL_CLEAR: 0}
    divergent = 0
    examples: list[dict[str, float | str]] = []

    for raw in distances:
        distance = float(raw)
        clearance_label = classify_clearance_regime(
            distance, radius_sum=radius_sum, near_miss_dist=near_miss_dist
        )
        center_label = classify_center_distance_regime(
            distance, collision_dist=collision_dist, near_miss_dist=near_miss_dist
        )
        clearance_counts[clearance_label] += 1
        center_counts[center_label] += 1
        if clearance_label != center_label:
            divergent += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "center_distance_m": distance,
                        REGIME_CLEARANCE: clearance_label,
                        REGIME_CENTER_DISTANCE: center_label,
                    }
                )

    return DefinitionInventory(
        sample_count=distances.size,
        radius_sum=radius_sum,
        collision_dist=float(collision_dist),
        near_miss_dist=float(near_miss_dist),
        clearance_counts=clearance_counts,
        center_distance_counts=center_counts,
        divergent=divergent,
        divergent_examples=examples,
    )


def synthetic_center_distance_sweep(
    *,
    start: float = 0.0,
    stop: float = 3.0,
    step: float = 0.05,
) -> list[float]:
    """Build a deterministic center-distance sweep for preflight reporting.

    The sweep spans from contact out past both regimes' near-miss bands so the
    divergence band is fully covered.

    Returns:
        A list of evenly spaced center distances in meters.
    """
    if step <= 0.0:
        raise ValueError("step must be positive")
    count = round((stop - start) / step) + 1
    return [round(start + i * step, 6) for i in range(max(count, 0))]


def format_divergence_report(inventory: DefinitionInventory) -> str:
    """Render a compact human-readable divergence summary.

    Returns:
        A multi-line string suitable for preflight / CLI output.
    """
    lines = [
        "collision/near-miss definition inventory (issue #3724)",
        f"  samples                : {inventory.sample_count}",
        f"  radius_sum (m)         : {inventory.radius_sum:.3f}",
        f"  collision_dist (m)     : {inventory.collision_dist:.3f}",
        f"  near_miss_dist (m)     : {inventory.near_miss_dist:.3f}",
        f"  clearance regime       : {inventory.clearance_counts}",
        f"  center-distance regime : {inventory.center_distance_counts}",
        f"  divergent samples      : {inventory.divergent} ({inventory.divergent_fraction:.1%})",
        f"  regimes agree          : {inventory.regimes_agree}",
    ]
    return "\n".join(lines)


__all__ = [
    "DEFAULT_PED_RADIUS",
    "DEFAULT_ROBOT_RADIUS",
    "LABEL_CLEAR",
    "LABEL_COLLISION",
    "LABEL_NEAR_MISS",
    "REGIME_CENTER_DISTANCE",
    "REGIME_CLEARANCE",
    "DefinitionInventory",
    "classify_center_distance_regime",
    "classify_clearance_regime",
    "collision_definition_inventory",
    "format_divergence_report",
    "synthetic_center_distance_sweep",
]
