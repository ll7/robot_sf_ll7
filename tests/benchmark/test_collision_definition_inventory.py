"""Tests for the collision/near-miss definition inventory (issue #3724).

These tests are deliberately diagnostic: they assert that the two regimes are
*known to diverge* and that the divergence is surfaced explicitly (fail-closed),
without asserting which definition is "correct". They also anchor the
inventory's clearance regime against the real benchmark metric in
``robot_sf/benchmark/metrics.py`` so the diagnostic cannot silently drift away
from what the metric actually computes.
"""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.collision_definition_inventory import (
    LABEL_CLEAR,
    LABEL_COLLISION,
    LABEL_NEAR_MISS,
    REGIME_CENTER_DISTANCE,
    REGIME_CLEARANCE,
    classify_center_distance_regime,
    classify_clearance_regime,
    collision_definition_inventory,
    format_divergence_report,
    synthetic_center_distance_sweep,
)
from robot_sf.benchmark.constants import COLLISION_DIST, NEAR_MISS_DIST
from robot_sf.benchmark.metrics import EpisodeData, _compute_robot_ped_distance_summary


def _single_ped_episode(center_distances: list[float]) -> EpisodeData:
    """Build an episode where the robot sits at the origin and one pedestrian.

    The pedestrian is placed on the +x axis at each requested center distance,
    one distance per timestep, so the per-timestep robot-ped distance equals the
    requested center distance exactly.
    """
    steps = len(center_distances)
    robot_pos = np.zeros((steps, 2), dtype=float)
    peds_pos = np.zeros((steps, 1, 2), dtype=float)
    peds_pos[:, 0, 0] = np.asarray(center_distances, dtype=float)
    zeros = np.zeros((steps, 2), dtype=float)
    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=zeros,
        robot_acc=zeros,
        peds_pos=peds_pos,
        ped_forces=np.zeros((steps, 1, 2), dtype=float),
        goal=np.array([10.0, 0.0], dtype=float),
        dt=0.1,
    )


def test_classifiers_use_their_own_boundaries():
    """Each regime applies its own boundary, so the same distance can differ."""
    radius_sum = 1.4  # default robot_radius (1.0) + ped_radius (0.4)
    # Center distance 0.30 m: footprints overlap (clearance < 0) -> collision in
    # the clearance regime, but it is above COLLISION_DIST=0.25 -> near-miss in
    # the center-distance regime.
    assert classify_clearance_regime(0.30, radius_sum=radius_sum) == LABEL_COLLISION
    assert classify_center_distance_regime(0.30) == LABEL_NEAR_MISS

    # Center distance 1.6 m: clearance 0.2 m -> near-miss in clearance regime,
    # but well beyond NEAR_MISS_DIST=0.50 in center distance -> clear.
    assert classify_clearance_regime(1.6, radius_sum=radius_sum) == LABEL_NEAR_MISS
    assert classify_center_distance_regime(1.6) == LABEL_CLEAR


def test_inventory_surfaces_divergence_fail_closed():
    """The default sweep must report the two regimes as divergent.

    This is the fail-closed guard: if a future change silently unifies the
    definitions (or makes the diagnostic stop measuring the gap), this test
    fails rather than passing quietly.
    """
    sweep = synthetic_center_distance_sweep(start=0.0, stop=3.0, step=0.05)
    inventory = collision_definition_inventory(sweep)

    assert inventory.sample_count == len(sweep)
    assert inventory.divergent > 0
    assert not inventory.regimes_agree
    assert 0.0 < inventory.divergent_fraction <= 1.0
    # Every recorded example must genuinely disagree.
    for example in inventory.divergent_examples:
        assert example[REGIME_CLEARANCE] != example[REGIME_CENTER_DISTANCE]


def test_inventory_boundaries_match_geometry():
    """Reported center-distance boundaries reflect each regime's definition."""
    inventory = collision_definition_inventory([0.0, 1.0, 2.0])
    payload = inventory.to_dict()
    clearance = payload["regimes"][REGIME_CLEARANCE]
    center = payload["regimes"][REGIME_CENTER_DISTANCE]

    # Clearance-regime collision boundary in center-distance terms == radius_sum.
    assert clearance["collision_center_distance_boundary_m"] == inventory.radius_sum
    assert center["collision_center_distance_boundary_m"] == COLLISION_DIST
    # The ~5x gap the issue describes (1.4 m vs 0.25 m with default radii).
    assert clearance["collision_center_distance_boundary_m"] > (
        5.0 * center["collision_center_distance_boundary_m"]
    )


def test_clearance_regime_matches_benchmark_metric():
    """Inventory clearance counts must equal the real metric's event counts.

    Anchoring against ``_compute_robot_ped_distance_summary`` proves the
    diagnostic mirrors what ``robot_sf/benchmark/metrics.py`` actually computes
    rather than a re-derivation that could drift.
    """
    # A spread of center distances covering collision, near-miss, and clear in
    # the clearance regime (radius_sum = 1.4, near_miss band [1.4, 1.9)).
    center_distances = [0.5, 1.0, 1.39, 1.45, 1.7, 1.89, 2.5, 5.0]
    episode = _single_ped_episode(center_distances)
    summary = _compute_robot_ped_distance_summary(episode)

    inventory = collision_definition_inventory(
        center_distances,
        robot_radius=episode.robot_radius,
        ped_radius=episode.ped_radius,
    )

    assert inventory.clearance_counts[LABEL_COLLISION] == int(summary["human_collisions"])
    assert inventory.clearance_counts[LABEL_NEAR_MISS] == int(summary["near_misses"])


def test_inventory_to_dict_is_json_serializable():
    """The preflight payload must serialize cleanly for reporting."""
    import json

    inventory = collision_definition_inventory(synthetic_center_distance_sweep(stop=2.0))
    encoded = json.dumps(inventory.to_dict())
    decoded = json.loads(encoded)
    assert decoded["divergent"] == inventory.divergent
    assert decoded["regimes_agree"] is inventory.regimes_agree


def test_thresholds_passed_through():
    """Custom thresholds are honored and reported back."""
    inventory = collision_definition_inventory(
        [0.1, 0.4, 0.9],
        collision_dist=COLLISION_DIST,
        near_miss_dist=NEAR_MISS_DIST,
    )
    assert inventory.collision_dist == COLLISION_DIST
    assert inventory.near_miss_dist == NEAR_MISS_DIST
    # format helper returns a non-empty, single-block string.
    report = format_divergence_report(inventory)
    assert "issue #3724" in report
    assert "divergent samples" in report
