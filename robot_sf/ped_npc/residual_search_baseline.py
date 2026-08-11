"""Deterministic finite-budget grid-search baseline over the residual adversary seam.

This module delivers issue #6911's smallest useful, deterministic, config-first
residual search-baseline slice: a :class:`GridSearchResidualPolicy` that
implements :class:`ResidualAdversaryPolicy` and evaluates a small explicit
action grid of candidate residual accelerations against a simple objective
proxy.

Capability-only slice
---------------------
This is a capability-only slice. It makes no benchmark, planner-ranking,
safety, or paper-facing claim. It implements no learned policy (PPO, etc.).
The objective proxy is a simple heuristic (negative weighted approach
closeness) and is NOT a calibrated pedestrian-behavior probability model.
All candidate proposals are routed through
:class:`BoundedResidualAdversary`, which enforces every hard bound.

Deterministic contract
----------------------
Given a fixed seed, action grid, budget, and macro cadence, repeated runs
produce identical proposal sequences and diagnostic metadata. The algorithm
name, objective proxy, config identity, action order, and bound settings are
recorded in every diagnostic record. The finite macro budget is enforced;
after exhaustion the policy emits zero proposals without further evaluation.

Algorithm
---------
For each macro-action boundary, for each targeted pedestrian independently:

1. Enumerate ``num_directions`` evenly-spaced angular directions.
2. For each direction, evaluate ``num_magnitudes`` magnitude levels in
   ``[0, max_residual_accel_mps2]``.
3. Score each candidate with the objective proxy (lower is better):
   ``-weight_approach * approach_speed + weight_distance * distance_to_robot``.
4. Select the candidate with the lowest score (closest approach).
5. Emit a zero-residual candidate as a baseline comparison.

The budget is the total number of macro-action boundaries processed. Invalid
or rejected candidates are counted and never silently treated as valid.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from math import cos, isfinite, sin
from random import Random
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.ped_npc.residual_adversary import (
    EPSILON,
    ResidualAdversaryObservation,
    _require_finite,
    _validate_finite_array,
)

if TYPE_CHECKING:
    from pathlib import Path

ALGORITHM_NAME = "grid_search_baseline_v1"
"""Canonical algorithm identifier for diagnostic records."""

DEFAULT_NUM_DIRECTIONS = 8
"""Default number of evenly-spaced angular candidates (0, pi/4, ..., 7pi/4)."""

DEFAULT_NUM_MAGNITUDES = 3
"""Default number of magnitude levels per direction (0, mid, max)."""

DEFAULT_WEIGHT_APPROACH = 1.0
"""Default weight on approach-speed component of the objective proxy."""

DEFAULT_WEIGHT_DISTANCE = 0.5
"""Default weight on distance-to-robot component of the objective proxy."""

DEFAULT_CONFIG_ID = "issue_4360_residual_search_baseline"
"""Stable identity for the checked-in issue #6911 search config."""

CONFIG_SCHEMA_VERSION = "residual_search_baseline.v1"
"""Schema identifier recorded in diagnostic provenance."""

OBJECTIVE_NAME = "negative_approach_closeness"
"""Stable identifier for the heuristic objective proxy."""


def _require_positive_int(value: object, name: str) -> int:
    """Return a positive integer or fail closed on malformed config input."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive int")
    if value < 1:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_int(value: object, name: str) -> int:
    """Return an integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return value


def _validate_config_metadata(config: GridSearchResidualConfig) -> None:
    """Validate provenance fields on a search-baseline config."""
    if not isinstance(config.config_id, str) or not config.config_id.strip():
        raise ValueError("config_id must be a non-empty string")
    if config.source_revision is not None and not isinstance(config.source_revision, str):
        raise TypeError("source_revision must be a string or None")
    if config.schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")


@dataclass(frozen=True)
class GridSearchResidualConfig:
    """Validated configuration for the deterministic grid-search baseline.

    Attributes
    ----------
    num_directions:
        Number of evenly-spaced angular directions in ``[0, 2*pi)``.
    num_magnitudes:
        Number of magnitude levels from ``0`` to ``max_residual_accel_mps2``.
    weight_approach:
        Weight on the approach-speed component (negative: approaching is good
        for the adversary).
    weight_distance:
        Weight on the distance-to-robot component (positive: closer is better
        for the adversary).
    seed:
        Fixed seed for deterministic candidate evaluation ordering.
    max_macro_budget:
        Maximum number of macro-action boundaries to process. The runner
        emits a diagnostic record at each boundary and stops counting after
        ``max_macro_budget``.
    config_id:
        Stable config identity recorded in diagnostic provenance.
    source_revision:
        Optional source revision supplied by the caller for provenance. It is
        not inferred from the local checkout.
    schema_version:
        Version of this search-baseline configuration contract.
    """

    num_directions: int = DEFAULT_NUM_DIRECTIONS
    num_magnitudes: int = DEFAULT_NUM_MAGNITUDES
    weight_approach: float = DEFAULT_WEIGHT_APPROACH
    weight_distance: float = DEFAULT_WEIGHT_DISTANCE
    seed: int = 42
    max_macro_budget: int = 1000
    config_id: str = DEFAULT_CONFIG_ID
    source_revision: str | None = None
    schema_version: str = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate finite, positive config values."""
        _require_positive_int(self.num_directions, "num_directions")
        _require_positive_int(self.num_magnitudes, "num_magnitudes")
        _require_finite(self.weight_approach, "weight_approach")
        _require_finite(self.weight_distance, "weight_distance")
        _require_int(self.seed, "seed")
        _require_positive_int(self.max_macro_budget, "max_macro_budget")
        _validate_config_metadata(self)


def _build_action_grid(
    num_directions: int,
    num_magnitudes: int,
    max_accel: float,
) -> np.ndarray:
    """Return an ``(G, 2)`` grid of candidate residual accelerations.

    The grid contains ``num_directions * num_magnitudes`` candidates plus
    one zero-residual baseline (row 0). Rows are sorted by angle then
    magnitude so the order is deterministic and reproducible.

    Returns
    -------
    np.ndarray
        Candidate grid with shape ``(1 + num_directions * num_magnitudes, 2)``.
    """
    angles = np.linspace(0.0, 2.0 * math.pi, num_directions, endpoint=False)
    magnitudes = np.linspace(0.0, max_accel, num_magnitudes)
    grid_rows = [np.array([0.0, 0.0])]
    for angle in angles:
        for mag in magnitudes:
            grid_rows.append(np.array([mag * cos(angle), mag * sin(angle)]))
    return np.array(grid_rows, dtype=float)


def _objective_proxy(
    candidate_accel: np.ndarray,
    ped_position: np.ndarray,
    ped_velocity: np.ndarray,
    robot_position: np.ndarray,
    weight_approach: float,
    weight_distance: float,
    dt_s: float,
) -> float:
    """Evaluate one candidate acceleration for one pedestrian.

    The proxy is a scalar score where **lower is better** for the adversary.
    It combines two components:

    - Approach speed: negative of the velocity component toward the robot
      after applying the candidate acceleration for one macro step.
    - Distance: Euclidean distance from the candidate position to the robot.

    Parameters
    ----------
    candidate_accel:
        ``(2,)`` candidate residual acceleration.
    ped_position:
        ``(2,)`` current pedestrian position.
    ped_velocity:
        ``(2,)`` current pedestrian velocity.
    robot_position:
        ``(2,)`` current robot position.
    weight_approach:
        Weight on approach-speed component.
    weight_distance:
        Weight on distance-to-robot component.
    dt_s:
        Physics timestep in seconds.

    Returns
    -------
    float
        Objective score (lower is better).
    """
    arrays: dict[str, np.ndarray] = {}
    for value, name in (
        (candidate_accel, "candidate_accel"),
        (ped_position, "ped_position"),
        (ped_velocity, "ped_velocity"),
        (robot_position, "robot_position"),
    ):
        array = np.asarray(value, dtype=float)
        if array.shape != (2,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must have shape (2,) with finite values")
        arrays[name] = array
    _require_finite(weight_approach, "weight_approach")
    _require_finite(weight_distance, "weight_distance")
    _require_finite(dt_s, "dt_s", strict_positive=True)
    candidate_accel = arrays["candidate_accel"]
    ped_position = arrays["ped_position"]
    ped_velocity = arrays["ped_velocity"]
    robot_position = arrays["robot_position"]
    delta_v = candidate_accel * dt_s
    future_velocity = ped_velocity + delta_v
    to_robot = robot_position - ped_position
    dist = float(np.linalg.norm(to_robot))
    if dist < EPSILON:
        return 0.0
    unit_to_robot = to_robot / dist
    approach_speed = float(np.dot(future_velocity, unit_to_robot))
    future_pos = ped_position + future_velocity * dt_s
    future_dist = float(np.linalg.norm(robot_position - future_pos))
    return float(-weight_approach * approach_speed + weight_distance * future_dist)


def _validate_observation(
    observation: ResidualAdversaryObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return the arrays consumed by the search policy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Positions, velocities, maximum speeds, target mask, and robot position.
    """
    positions = _validate_finite_array(observation.positions, "observation.positions")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("observation.positions must have shape (N, 2)")
    num_peds = positions.shape[0]
    velocities = _validate_finite_array(observation.velocities, "observation.velocities")
    if velocities.shape != positions.shape:
        raise ValueError("observation.velocities must match positions shape")
    max_speeds = _validate_finite_array(observation.max_speeds, "observation.max_speeds")
    if max_speeds.shape != (num_peds,):
        raise ValueError("observation.max_speeds must have shape (N,)")
    if np.any(max_speeds < 0.0):
        raise ValueError("observation.max_speeds must be >= 0")
    target_mask = np.asarray(observation.target_ped_mask)
    if target_mask.shape != (num_peds,) or target_mask.dtype != bool:
        raise ValueError("observation.target_ped_mask must have shape (N,) and bool dtype")
    robot_pos = np.asarray(observation.robot_pose[0], dtype=float)
    if robot_pos.shape != (2,) or not np.all(np.isfinite(robot_pos)):
        raise ValueError("observation.robot_pose must be finite")
    if not isfinite(float(observation.robot_pose[1])):
        raise ValueError("observation.robot_pose must be finite")
    return positions, velocities, max_speeds, target_mask, robot_pos


@dataclass
class GridSearchResidualPolicy:
    """Deterministic finite-budget grid-search residual adversary.

    This policy evaluates a small explicit action grid of candidate residual
    accelerations and selects the best candidate per targeted pedestrian
    using a simple objective proxy. It implements
    :class:`ResidualAdversaryPolicy` and is intended as the first
    search-baseline slice (issue #6911).

    It is NOT a learned policy. The objective proxy is a heuristic, not a
    calibrated probability model. All proposals are routed through
    :class:`BoundedResidualAdversary`, which enforces every hard bound.
    """

    config: GridSearchResidualConfig
    max_residual_accel_mps2: float
    dt_s: float
    bound_settings: dict[str, Any] = field(default_factory=dict)
    _action_grid: np.ndarray = field(init=False, repr=False)
    _macro_count: int = field(init=False, default=0, repr=False)
    _diagnostic_records: list[dict[str, Any]] = field(init=False, default_factory=list)
    _candidate_order: tuple[int, ...] = field(init=False, repr=False)
    _budget_exhausted: bool = field(init=False, default=False, repr=False)
    _rejected_count: int = field(init=False, default=0)
    _invalid_count: int = field(init=False, default=0)
    _accepted_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Build the action grid and validate parameters."""
        _require_finite(
            self.max_residual_accel_mps2, "max_residual_accel_mps2", strict_positive=True
        )
        _require_finite(self.dt_s, "dt_s", strict_positive=True)
        self._action_grid = _build_action_grid(
            self.config.num_directions,
            self.config.num_magnitudes,
            self.max_residual_accel_mps2,
        )
        candidate_indices = list(range(1, self._action_grid.shape[0]))
        Random(self.config.seed).shuffle(candidate_indices)
        self._candidate_order = (0, *candidate_indices)
        bound_settings = dict(self.bound_settings)
        bound_settings.setdefault("max_residual_accel_mps2", float(self.max_residual_accel_mps2))
        bound_settings.setdefault("dt_s", float(self.dt_s))
        json.dumps(bound_settings, sort_keys=True, allow_nan=False)
        self.bound_settings = bound_settings

    @property
    def macro_count(self) -> int:
        """Number of macro-action boundaries processed so far."""
        return self._macro_count

    @property
    def diagnostic_records(self) -> list[dict[str, Any]]:
        """Return a copy of the accumulated diagnostic records."""
        return list(self._diagnostic_records)

    @property
    def rejected_count(self) -> int:
        """Number of candidates rejected (scored worse than zero-residual baseline)."""
        return self._rejected_count

    @property
    def invalid_count(self) -> int:
        """Number of candidates with non-finite scores."""
        return self._invalid_count

    @property
    def accepted_count(self) -> int:
        """Number of candidates accepted (best non-zero per pedestrian)."""
        return self._accepted_count

    @property
    def budget_exhausted(self) -> bool:
        """Whether the finite search budget has been consumed."""
        return self._budget_exhausted

    def _search_one_ped(
        self,
        ped_pos: np.ndarray,
        ped_vel: np.ndarray,
        robot_position: np.ndarray,
    ) -> tuple[np.ndarray, bool, int]:
        """Search the grid for one pedestrian.

        Returns
        -------
        tuple[np.ndarray, bool, int]
            ``(best_candidate, accepted, invalid_count)``.
        """
        best_score = float("inf")
        best_cand = np.zeros(2, dtype=float)
        invalid = 0
        for candidate_index in self._candidate_order:
            cand = self._action_grid[candidate_index]
            try:
                score = _objective_proxy(
                    cand,
                    ped_pos,
                    ped_vel,
                    robot_position,
                    self.config.weight_approach,
                    self.config.weight_distance,
                    self.dt_s,
                )
            except (ValueError, TypeError):
                invalid += 1
                continue
            if not isfinite(score):
                invalid += 1
                continue
            if score < best_score:
                best_score = score
                best_cand = cand.copy()
        zero_score = _objective_proxy(
            np.zeros(2),
            ped_pos,
            ped_vel,
            robot_position,
            self.config.weight_approach,
            self.config.weight_distance,
            self.dt_s,
        )
        if not isfinite(zero_score):
            invalid += 1
        accepted = best_score < zero_score - EPSILON
        return best_cand, accepted, invalid

    def _search_targeted_pedestrians(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        target_mask: np.ndarray,
        robot_position: np.ndarray,
    ) -> tuple[np.ndarray, int, int, int]:
        """Search all targeted pedestrians and return proposal/accounting.

        Returns
        -------
        tuple[np.ndarray, int, int, int]
            Proposal array, accepted count, rejected count, and invalid count.
        """
        proposal = np.zeros_like(positions)
        rejected = 0
        invalid = 0
        accepted = 0
        for ped_idx in np.flatnonzero(target_mask):
            candidate, was_accepted, ped_invalid = self._search_one_ped(
                positions[ped_idx], velocities[ped_idx], robot_position
            )
            invalid += ped_invalid
            if was_accepted:
                proposal[ped_idx] = candidate
                accepted += 1
            else:
                rejected += 1
        return proposal, accepted, rejected, invalid

    def propose_residual(self, observation: ResidualAdversaryObservation) -> np.ndarray:
        """Return a finite ``(N, 2)`` grid-search proposal for targeted peds.

        For each targeted pedestrian independently, the action grid is
        evaluated and the best candidate is selected. Non-targeted pedestrians
        receive a zero residual.

        Returns
        -------
        np.ndarray
            Proposed residual acceleration with shape ``(N, 2)``.
        """
        positions, velocities, _max_speeds, target_mask, robot_position = _validate_observation(
            observation
        )
        num_peds = positions.shape[0]
        proposal = np.zeros((num_peds, 2), dtype=float)

        if num_peds == 0:
            return proposal
        if not np.any(target_mask):
            return proposal
        if self._macro_count >= self.config.max_macro_budget:
            self._budget_exhausted = True
            return proposal

        proposal, accepted, rejected, invalid = self._search_targeted_pedestrians(
            positions, velocities, target_mask, robot_position
        )

        grid_size = self._action_grid.shape[0]
        record: dict[str, Any] = {
            "algorithm": ALGORITHM_NAME,
            "objective": OBJECTIVE_NAME,
            "config_id": self.config.config_id,
            "schema_version": self.config.schema_version,
            "source_revision": self.config.source_revision,
            "macro_action_index": int(observation.macro_action_index),
            "search_call_index": self._macro_count,
            "grid_size": grid_size,
            "candidate_order": list(self._candidate_order),
            "action_order": "zero_baseline_then_seeded_angle_magnitude",
            "num_directions": self.config.num_directions,
            "num_magnitudes": self.config.num_magnitudes,
            "max_residual_accel_mps2": self.max_residual_accel_mps2,
            "seed": self.config.seed,
            "budget": self.config.max_macro_budget,
            "bound_settings": dict(self.bound_settings),
            "candidates_evaluated": int(np.sum(target_mask)) * grid_size,
            "accepted": accepted,
            "rejected": rejected,
            "invalid": invalid,
            "target_ped_count": int(np.sum(target_mask)),
        }
        self._diagnostic_records.append(record)
        self._macro_count += 1
        self._budget_exhausted = self._macro_count >= self.config.max_macro_budget
        self._rejected_count += rejected
        self._invalid_count += invalid
        self._accepted_count += accepted

        return proposal

    def write_diagnostics(self, output_path: Path) -> Path:
        """Write accumulated diagnostic records as compact JSON.

        Parameters
        ----------
        output_path:
            Destination file path. Parent directory must exist.

        Returns
        -------
        Path
            The written file path.
        """
        payload = {
            "algorithm": ALGORITHM_NAME,
            "objective": OBJECTIVE_NAME,
            "config_id": self.config.config_id,
            "schema_version": self.config.schema_version,
            "source_revision": self.config.source_revision,
            "seed": self.config.seed,
            "budget": self.config.max_macro_budget,
            "candidate_order": list(self._candidate_order),
            "action_order": "zero_baseline_then_seeded_angle_magnitude",
            "bound_settings": dict(self.bound_settings),
            "macro_boundaries_processed": self._macro_count,
            "budget_exhausted": self._budget_exhausted,
            "total_accepted": self._accepted_count,
            "total_rejected": self._rejected_count,
            "total_invalid": self._invalid_count,
            "records": self._diagnostic_records,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return output_path

    def reset(self) -> None:
        """Clear accumulated state for a fresh episode."""
        self._macro_count = 0
        self._budget_exhausted = False
        self._diagnostic_records.clear()
        self._rejected_count = 0
        self._invalid_count = 0
        self._accepted_count = 0


__all__ = [
    "ALGORITHM_NAME",
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_CONFIG_ID",
    "DEFAULT_NUM_DIRECTIONS",
    "DEFAULT_NUM_MAGNITUDES",
    "DEFAULT_WEIGHT_APPROACH",
    "DEFAULT_WEIGHT_DISTANCE",
    "OBJECTIVE_NAME",
    "GridSearchResidualConfig",
    "GridSearchResidualPolicy",
    "_build_action_grid",
    "_objective_proxy",
]
