"""Edge-case tests for collision / near-miss classification and force thresholds.

These tests validate the exact inequality semantics around COLLISION_DIST and
NEAR_MISS_DIST as well as force exceedance counting.

Classification rules (from `robot_sf.benchmark.metrics` implementation):
  - collision timestep:   min_distance < D_COLL
  - near-miss timestep:   D_COLL <= min_distance < D_NEAR
  - outside event window: min_distance >= D_NEAR

Force threshold rule:
  - force_exceed_events counts (t,k) where |F| > COMFORT_FORCE_THRESHOLD

We construct synthetic `EpisodeData` containers to spike distances / forces at
exact boundary values and just-inside / just-outside values.
"""

from __future__ import annotations

import math

import numpy as np

from robot_sf.benchmark.constants import COLLISION_DIST as D_COLL
from robot_sf.benchmark.constants import COMFORT_FORCE_THRESHOLD as F_THRESH
from robot_sf.benchmark.constants import NEAR_MISS_DIST as D_NEAR
from robot_sf.benchmark.metrics import (
    EpisodeData,
    collisions,
    comfort_exposure,
    force_exceed_events,
    mean_distance,
    min_distance,
    near_misses,
)


def _episode(
    robot_pos: np.ndarray,
    peds_pos: np.ndarray,
    ped_forces: np.ndarray | None = None,
) -> EpisodeData:
    """Episode.

    Args:
        robot_pos: Auto-generated placeholder description.
        peds_pos: Auto-generated placeholder description.
        ped_forces: Auto-generated placeholder description.

    Returns:
        EpisodeData: Auto-generated placeholder description.
    """
    T = robot_pos.shape[0]
    K = peds_pos.shape[1]
    if ped_forces is None:
        ped_forces = np.zeros((T, K, 2), dtype=float)
    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=np.zeros_like(robot_pos),
        robot_acc=np.zeros_like(robot_pos),
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        goal=np.array([0.0, 0.0]),
        dt=0.1,
    )


def test_collision_and_near_miss_boundary_classification():
    """Distances at boundaries classify correctly:

    step0: distance == D_COLL        -> near-miss (NOT collision)
    step1: distance == D_COLL - eps  -> collision
    step2: distance == D_NEAR        -> neither collision nor near-miss
    """

    eps = 1e-6
    # Robot fixed at origin
    robot_pos = np.zeros((3, 2), dtype=float)
    # Single pedestrian positioned to yield the desired distances each step
    ped_positions = np.array(
        [
            [D_COLL, 0.0],  # exactly D_COLL
            [D_COLL - eps, 0.0],  # just inside collision
            [D_NEAR, 0.0],  # exactly D_NEAR
        ],
        dtype=float,
    )[:, None, :]  # (T,1,2)

    data = _episode(robot_pos, ped_positions)
    assert collisions(data) == 1.0, "Exactly one timestep should be a collision (< D_COLL)."
    assert near_misses(data) == 1.0, "Exactly one timestep (== D_COLL) should be a near-miss."


def test_zero_pedestrians_returns_nan_for_distance_and_zero_counts():
    """When K=0: distances are NaN (undefined) and event counters are zero."""
    T = 5
    robot_pos = np.zeros((T, 2), dtype=float)
    peds_pos = np.zeros((T, 0, 2), dtype=float)
    data = _episode(robot_pos, peds_pos)
    assert collisions(data) == 0.0
    assert near_misses(data) == 0.0
    assert math.isnan(min_distance(data))
    assert math.isnan(mean_distance(data))


def test_force_exceed_events_and_comfort_exposure():
    """Force exceed counting uses strict > threshold semantics.

    Matrix (T=2, K=2):
      t0: [F_THRESH - eps, F_THRESH + eps] -> 1 exceed
      t1: [F_THRESH,       F_THRESH + 1e-3] -> second ped exceed (strict >)
    Total exceed events = 2
    comfort_exposure = 2 / (T*K) = 2 / 4 = 0.5
    """
    eps = 1e-6
    T, K = 2, 2
    robot_pos = np.zeros((T, 2), dtype=float)
    peds_pos = np.zeros((T, K, 2), dtype=float)  # positions don't matter for force counting

    mags = np.array(
        [
            [F_THRESH - eps, F_THRESH + eps],
            [F_THRESH, F_THRESH + 1e-3],
        ],
    )
    # Convert magnitudes to 2D vectors (place along x-axis)
    ped_forces = np.zeros((T, K, 2), dtype=float)
    ped_forces[..., 0] = mags

    data = _episode(robot_pos, peds_pos, ped_forces)
    assert force_exceed_events(data) == 2.0
    assert comfort_exposure(data) == 0.5


def test_force_metrics_zero_pedestrians():
    """Zero pedestrians → zero force events and zero comfort exposure."""
    T = 3
    robot_pos = np.zeros((T, 2), dtype=float)
    peds_pos = np.zeros((T, 0, 2), dtype=float)
    data = _episode(robot_pos, peds_pos)
    assert force_exceed_events(data) == 0.0
    assert comfort_exposure(data) == 0.0
