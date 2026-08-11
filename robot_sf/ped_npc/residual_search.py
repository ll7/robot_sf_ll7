"""Deterministic finite-budget grid search over residual adversary proposals.

Issue #6911 adds a diagnostic-only search baseline that evaluates a finite
grid of candidate residual accelerations through the existing bound pipeline.
No optimizer dependency is introduced; the search is a brute-force enumeration
within an explicit action grid. The algorithm and diagnostic objective proxy
are named in the checked-in config and emitted in every diagnostic record.

Capability-only slice: no benchmark, metric, planner-ranking, safety, or
paper-facing claim.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from robot_sf.ped_npc.residual_adversary import (
    EPSILON,
    BoundedResidualAdversary,
    ResidualAdversaryConfig,
    ResidualAdversaryObservation,
    ResidualBoundConflictError,
    _require_finite,
    _validate_finite_array,
)

_SEARCH_SCHEMA = "residual_search_diagnostic.v1"
_ALGORITHM_NAME = "finite_grid_search_v1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualSearchConfig:
    """Validated config for finite-grid search over residual proposals.

    Attributes
    ----------
    algorithm_name:
        Machine-readable name of the search algorithm for metadata.
    objective_proxy:
        Name of the diagnostic objective function used to rank candidates.
    grid_points_per_dim:
        Number of evenly-spaced grid points per action dimension.  The total
        grid size is ``grid_points_per_dim ** 2``.
    max_candidates:
        Maximum total candidates to evaluate per macro-action boundary
        (budget cap).  When fewer candidates exist than the budget, all are
        evaluated.
    seed:
        Deterministic seed for reproducibility.  The bundled search policy
        does not use a RNG; the seed is recorded for metadata and future
        extensions.
    action_min_mps2:
        Minimum acceleration per dimension in the grid (m/s^2).
    action_max_mps2:
        Maximum acceleration per dimension in the grid (m/s^2).
    """

    algorithm_name: str = _ALGORITHM_NAME
    objective_proxy: str = "maximize_residual_magnitude"
    grid_points_per_dim: int = 3
    max_candidates: int = 9
    seed: int = 42
    action_min_mps2: float = -1.5
    action_max_mps2: float = 1.5

    def __post_init__(self) -> None:
        """Validate types, finiteness, and range constraints."""
        if not isinstance(self.algorithm_name, str) or not self.algorithm_name:
            raise ValueError("algorithm_name must be a non-empty string")
        if not isinstance(self.objective_proxy, str) or not self.objective_proxy:
            raise ValueError("objective_proxy must be a non-empty string")
        if not isinstance(self.grid_points_per_dim, int) or isinstance(
            self.grid_points_per_dim, bool
        ):
            raise TypeError("grid_points_per_dim must be an int")
        _require_finite(self.grid_points_per_dim, "grid_points_per_dim", strict_positive=True)
        if not isinstance(self.max_candidates, int) or isinstance(self.max_candidates, bool):
            raise TypeError("max_candidates must be an int")
        _require_finite(self.max_candidates, "max_candidates", strict_positive=True)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        _require_finite(self.action_min_mps2, "action_min_mps2")
        _require_finite(self.action_max_mps2, "action_max_mps2")
        if self.action_min_mps2 >= self.action_max_mps2:
            raise ValueError(
                f"action_min_mps2 ({self.action_min_mps2}) must be < "
                f"action_max_mps2 ({self.action_max_mps2})"
            )


# ---------------------------------------------------------------------------
# Config digest and source revision
# ---------------------------------------------------------------------------


def compute_config_digest(
    config: ResidualSearchConfig,
    residual_config: ResidualAdversaryConfig | None = None,
) -> str:
    """Return a SHA-256 digest of the canonical config serialization.

    The digest is deterministic: identical config fields always produce the
    same hex string. It uses sorted JSON with no whitespace so two configs
    with the same values compare equal regardless of construction order. When
    supplied, the residual-adversary settings are included in the digest so the
    record identifies the complete search/controller configuration.
    """
    payload: dict[str, Any] = {
        "action_max_mps2": config.action_max_mps2,
        "action_min_mps2": config.action_min_mps2,
        "algorithm_name": config.algorithm_name,
        "grid_points_per_dim": config.grid_points_per_dim,
        "max_candidates": config.max_candidates,
        "objective_proxy": config.objective_proxy,
        "seed": config.seed,
    }
    if residual_config is not None:
        target_ped_idx = residual_config.target_ped_idx
        if isinstance(target_ped_idx, list):
            target_ped_idx = list(target_ped_idx)
        payload["residual_adversary"] = {
            "is_active": residual_config.is_active,
            "macro_action_dt_s": residual_config.macro_action_dt_s,
            "max_residual_accel_mps2": residual_config.max_residual_accel_mps2,
            "max_jerk_mps3": residual_config.max_jerk_mps3,
            "max_speed_delta_mps": residual_config.max_speed_delta_mps,
            "max_heading_change_per_macro_rad": residual_config.max_heading_change_per_macro_rad,
            "max_route_deviation_m": residual_config.max_route_deviation_m,
            "min_separation_m": residual_config.min_separation_m,
            "target_ped_idx": target_ped_idx,
            "obstacle_projection_margin_m": residual_config.obstacle_projection_margin_m,
            "seed": residual_config.seed,
        }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _git_head_sha(repo_root: Path | None = None) -> str | None:
    """Return the current git HEAD full sha, or ``None`` when unavailable."""
    cwd = str(repo_root) if repo_root is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    sha = result.stdout.strip()
    return sha or None


# ---------------------------------------------------------------------------
# Action grid
# ---------------------------------------------------------------------------


def _build_action_grid(
    min_mps2: float,
    max_mps2: float,
    grid_points: int,
) -> np.ndarray:
    """Return a ``(G**2, 2)`` array of all ``(ax, ay)`` grid candidates.

    The grid is the Cartesian product of ``grid_points`` evenly-spaced values
    in ``[min_mps2, max_mps2]`` along each axis.  The order is deterministic
    (row-major meshgrid) so repeated calls produce identical arrays.
    """
    points = np.linspace(min_mps2, max_mps2, grid_points)
    ax_grid, ay_grid = np.meshgrid(points, points)
    return np.stack([ax_grid.ravel(), ay_grid.ravel()], axis=1)


# ---------------------------------------------------------------------------
# Candidate evaluation through the complete bounded controller
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FixedCandidatePolicy:
    """Return one candidate proposal for an isolated controller evaluation."""

    proposal: np.ndarray

    def propose_residual(self, observation: ResidualAdversaryObservation) -> np.ndarray:
        """Return a copy of the candidate after validating the pedestrian count."""
        if observation.positions.shape != (self.proposal.shape[0], 2):
            raise ValueError("candidate proposal and observation pedestrian counts differ")
        return self.proposal.copy()


@dataclass(frozen=True)
class _CandidateEvaluationContext:
    """Immutable runtime inputs shared by each candidate evaluation."""

    positions: np.ndarray
    velocities: np.ndarray
    max_speeds: np.ndarray
    residual_config: ResidualAdversaryConfig
    dt_s: float
    robot_pose: tuple[tuple[float, float], float]
    route_polylines: list[np.ndarray] | dict[int, np.ndarray] | None = None
    obstacle_segments: np.ndarray | list[Any] | None = None
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    ped_radius: float = 0.4


def _evaluate_candidate(
    candidate_2d: np.ndarray,
    target_idx: int,
    context: _CandidateEvaluationContext,
) -> tuple[float, bool]:
    """Evaluate one candidate through an isolated bounded controller.

    Every candidate is placed into a full ``(N, 2)`` proposal and passed through
    a fresh :class:`BoundedResidualAdversary`. This keeps candidate accounting
    faithful to the runtime contract, including stateful jerk handling,
    geometry projection, and inter-agent separation. The selected proposal is
    evaluated once more by the caller's live controller.

    Returns
    -------
    tuple[float, bool]
        ``(score, is_valid)`` where *score* is the Euclidean norm of the
        bounded residual at the targeted pedestrian and *is_valid* is
        ``False`` when the candidate triggers a bound conflict.
    """
    try:
        candidate_array = _validate_finite_array(candidate_2d, "candidate")
        if candidate_array.shape != (2,):
            raise ValueError("candidate must have shape (2,)")
        num_peds = context.positions.shape[0]
        proposal = np.zeros((num_peds, 2), dtype=float)
        proposal[target_idx] = candidate_array
        evaluator = BoundedResidualAdversary(
            config=context.residual_config,
            policy=_FixedCandidatePolicy(proposal),
            dt_s=context.dt_s,
            num_peds=num_peds,
            route_polylines=context.route_polylines,
            obstacle_segments=context.obstacle_segments,
            bounds=context.bounds,
            ped_radius=context.ped_radius,
        )
        bounded = evaluator.step_residual(
            context.positions,
            context.velocities,
            context.max_speeds,
            context.robot_pose,
        )
        score = float(np.linalg.norm(bounded[target_idx]))
        return score, True
    except (IndexError, ResidualBoundConflictError, ValueError, TypeError):
        return 0.0, False


def _bound_settings(
    residual_config: ResidualAdversaryConfig,
    *,
    dt_s: float,
    route_polylines: list[np.ndarray] | dict[int, np.ndarray] | None,
    obstacle_segments: np.ndarray | list[Any] | None,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None,
    ped_radius: float,
) -> dict[str, Any]:
    """Return portable runtime-bound metadata for a diagnostic record."""
    target_ped_idx = residual_config.target_ped_idx
    if isinstance(target_ped_idx, list):
        target_ped_idx = list(target_ped_idx)
    return {
        "dt_s": float(dt_s),
        "macro_action_dt_s": float(residual_config.macro_action_dt_s),
        "macro_action_steps": max(1, round(residual_config.macro_action_dt_s / dt_s)),
        "max_residual_accel_mps2": float(residual_config.max_residual_accel_mps2),
        "max_jerk_mps3": float(residual_config.max_jerk_mps3),
        "max_speed_delta_mps": float(residual_config.max_speed_delta_mps),
        "max_heading_change_per_macro_rad": float(residual_config.max_heading_change_per_macro_rad),
        "max_route_deviation_m": float(residual_config.max_route_deviation_m),
        "min_separation_m": float(residual_config.min_separation_m),
        "obstacle_projection_margin_m": float(residual_config.obstacle_projection_margin_m),
        "ped_radius": float(ped_radius),
        "target_ped_idx": target_ped_idx,
        "route_polylines_supplied": route_polylines is not None,
        "obstacle_segments_supplied": obstacle_segments is not None,
        "map_bounds_supplied": bounds is not None,
    }


# ---------------------------------------------------------------------------
# Diagnostic record
# ---------------------------------------------------------------------------


@dataclass
class SearchDiagnosticRecord:
    """Compact deterministic diagnostic record of a search run.

    All fields are serialised with sorted JSON keys and no trailing
    whitespace so repeated runs from the same config and seed produce
    byte-equivalent canonical records.
    """

    schema_version: str = _SEARCH_SCHEMA
    algorithm_name: str = ""
    objective_proxy: str = ""
    config_digest: str = ""
    seed: int = 0
    source_revision: str = ""
    grid_points_per_dim: int = 0
    action_bounds: dict[str, float] = field(default_factory=dict)
    bound_settings: dict[str, Any] = field(default_factory=dict)
    candidate_order: list[str] = field(default_factory=list)
    candidate_actions_mps2: list[list[float]] = field(default_factory=list)
    budget: int = 0
    num_targeted_peds: int = 0
    total_evaluated: int = 0
    accepted: int = 0
    rejected: int = 0
    invalid: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict with alphabetically sorted keys."""
        return {
            "accepted": self.accepted,
            "action_bounds": {
                "max_mps2": float(self.action_bounds.get("max_mps2", 0.0)),
                "min_mps2": float(self.action_bounds.get("min_mps2", 0.0)),
            },
            "algorithm_name": self.algorithm_name,
            "budget": self.budget,
            "bound_settings": dict(self.bound_settings),
            "candidate_actions_mps2": [list(action) for action in self.candidate_actions_mps2],
            "candidate_order": list(self.candidate_order),
            "config_digest": self.config_digest,
            "grid_points_per_dim": self.grid_points_per_dim,
            "invalid": self.invalid,
            "num_targeted_peds": self.num_targeted_peds,
            "objective_proxy": self.objective_proxy,
            "rejected": self.rejected,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "source_revision": self.source_revision,
            "total_evaluated": self.total_evaluated,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Return deterministic JSON (sorted keys, compact by default)."""
        separators = (",", ":") if indent is None else (",", ": ")
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            indent=indent,
            separators=separators,
        )


# ---------------------------------------------------------------------------
# Search policy
# ---------------------------------------------------------------------------


class FiniteGridSearchPolicy:
    """Deterministic finite-budget grid search over residual proposals.

    This policy implements :class:`ResidualAdversaryPolicy` and evaluates every
    finite-grid candidate through an isolated
    :class:`~robot_sf.ped_npc.residual_adversary.BoundedResidualAdversary`.
    The best candidate per targeted pedestrian is selected using the configured
    diagnostic objective proxy, then passed through the live controller by the
    caller. This preserves the same acceleration, jerk, speed, heading, route,
    walkability, and separation contract for candidate evaluation and runtime.

    A :attr:`last_record` diagnostic summary is updated after each
    :meth:`propose_residual` call.
    """

    def __init__(
        self,
        search_config: ResidualSearchConfig,
        residual_config: ResidualAdversaryConfig,
        dt_s: float,
        num_peds: int,
        *,
        route_polylines: list[np.ndarray] | dict[int, np.ndarray] | None = None,
        obstacle_segments: np.ndarray | list[Any] | None = None,
        bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
        ped_radius: float = 0.4,
    ) -> None:
        """Construct the search policy, pre-computing the action grid and digest."""
        _require_finite(dt_s, "dt_s", strict_positive=True)
        if not isinstance(num_peds, int) or num_peds < 0:
            raise ValueError("num_peds must be a non-negative int")
        if not residual_config.is_active:
            raise ValueError("residual_config.is_active must be True for a search policy")
        _require_finite(ped_radius, "ped_radius")
        if ped_radius < 0:
            raise ValueError("ped_radius must be >= 0")
        self._search_config = search_config
        self._residual_config = residual_config
        self._dt_s = dt_s
        self._num_peds = num_peds
        self._route_polylines = route_polylines
        self._obstacle_segments = obstacle_segments
        self._bounds = bounds
        self._ped_radius = ped_radius
        self._grid = _build_action_grid(
            search_config.action_min_mps2,
            search_config.action_max_mps2,
            search_config.grid_points_per_dim,
        )
        self._config_digest = compute_config_digest(search_config, residual_config)
        self._bound_settings = _bound_settings(
            residual_config,
            dt_s=dt_s,
            route_polylines=route_polylines,
            obstacle_segments=obstacle_segments,
            bounds=bounds,
            ped_radius=ped_radius,
        )
        self._source_revision = _git_head_sha() or "unknown"
        self._record = SearchDiagnosticRecord(
            algorithm_name=search_config.algorithm_name,
            objective_proxy=search_config.objective_proxy,
            config_digest=self._config_digest,
            seed=search_config.seed,
            source_revision=self._source_revision,
            grid_points_per_dim=search_config.grid_points_per_dim,
            action_bounds={
                "min_mps2": search_config.action_min_mps2,
                "max_mps2": search_config.action_max_mps2,
            },
            bound_settings=dict(self._bound_settings),
            budget=search_config.max_candidates,
        )

    @property
    def last_record(self) -> SearchDiagnosticRecord:
        """Return the diagnostic record from the most recent proposal."""
        return self._record

    @property
    def config_digest(self) -> str:
        """Return the canonical config digest computed at construction."""
        return self._config_digest

    @property
    def source_revision(self) -> str:
        """Return the git HEAD SHA captured at construction."""
        return self._source_revision

    @property
    def grid(self) -> np.ndarray:
        """Return the action grid (read-only copy)."""
        return self._grid.copy()

    def propose_residual(self, observation: ResidualAdversaryObservation) -> np.ndarray:
        """Search the grid and return the best bounded proposal.

        For each targeted pedestrian the policy enumerates grid candidates,
        evaluates each through an isolated bounded controller, and selects
        the candidate with the highest objective-proxy score.  Non-targeted
        pedestrians always receive a zero residual.

        The diagnostic record is updated in-place after each call.

        Returns
        -------
        np.ndarray
            ``(N, 2)`` residual acceleration proposal.  Targeted pedestrians
            receive the best grid candidate; non-targeted pedestrians receive
            zeros.
        """
        positions = _validate_finite_array(observation.positions, "observation.positions")
        velocities = _validate_finite_array(observation.velocities, "observation.velocities")
        max_speeds = _validate_finite_array(observation.max_speeds, "observation.max_speeds")

        num_peds = positions.shape[0]
        if num_peds != self._num_peds:
            raise ValueError(
                f"observation pedestrian count must be {self._num_peds}, got {num_peds}"
            )
        target_indices = np.flatnonzero(observation.target_ped_mask)
        num_targeted = int(target_indices.size)

        total_evaluated = 0
        accepted = 0
        rejected = 0
        invalid = 0
        candidate_order: list[str] = []
        candidate_actions_mps2: list[list[float]] = []

        result = np.zeros((num_peds, 2), dtype=float)

        if num_targeted == 0:
            self._record = SearchDiagnosticRecord(
                algorithm_name=self._search_config.algorithm_name,
                objective_proxy=self._search_config.objective_proxy,
                config_digest=self._config_digest,
                seed=self._search_config.seed,
                source_revision=self._source_revision,
                grid_points_per_dim=self._search_config.grid_points_per_dim,
                action_bounds=dict(self._record.action_bounds),
                bound_settings=dict(self._bound_settings),
                candidate_order=[],
                candidate_actions_mps2=[],
                budget=self._search_config.max_candidates,
                num_targeted_peds=0,
                total_evaluated=0,
                accepted=0,
                rejected=0,
                invalid=0,
            )
            return result

        if self._search_config.max_candidates < num_targeted:
            raise ValueError(
                "max_candidates must cover at least one action per targeted pedestrian"
            )
        grid_size = self._grid.shape[0]
        remaining_budget = self._search_config.max_candidates
        remaining_targets = num_targeted
        evaluation_context = _CandidateEvaluationContext(
            positions=positions,
            velocities=velocities,
            max_speeds=max_speeds,
            residual_config=self._residual_config,
            dt_s=self._dt_s,
            robot_pose=observation.robot_pose,
            route_polylines=self._route_polylines,
            obstacle_segments=self._obstacle_segments,
            bounds=self._bounds,
            ped_radius=self._ped_radius,
        )

        for target_idx in target_indices:
            best_score = -float("inf")
            best_candidate = np.zeros(2, dtype=float)
            has_valid = False
            candidates_for_target = min(
                grid_size,
                max(1, remaining_budget // remaining_targets),
            )

            for g_idx in range(candidates_for_target):
                candidate = self._grid[g_idx]
                candidate_order.append(f"ped_{int(target_idx)}:grid_{g_idx:03d}")
                candidate_actions_mps2.append([float(candidate[0]), float(candidate[1])])
                score, is_valid = _evaluate_candidate(
                    candidate,
                    int(target_idx),
                    evaluation_context,
                )
                total_evaluated += 1

                if not is_valid:
                    invalid += 1
                    continue

                if score > best_score + EPSILON:
                    best_score = score
                    best_candidate = candidate.copy()
                    has_valid = True
                    accepted += 1
                else:
                    rejected += 1

            if has_valid:
                result[int(target_idx)] = best_candidate
            remaining_budget -= candidates_for_target
            remaining_targets -= 1

        self._record = SearchDiagnosticRecord(
            algorithm_name=self._search_config.algorithm_name,
            objective_proxy=self._search_config.objective_proxy,
            config_digest=self._config_digest,
            seed=self._search_config.seed,
            source_revision=self._source_revision,
            grid_points_per_dim=self._search_config.grid_points_per_dim,
            action_bounds=dict(self._record.action_bounds),
            bound_settings=dict(self._bound_settings),
            candidate_order=candidate_order,
            candidate_actions_mps2=candidate_actions_mps2,
            budget=self._search_config.max_candidates,
            num_targeted_peds=num_targeted,
            total_evaluated=total_evaluated,
            accepted=accepted,
            rejected=rejected,
            invalid=invalid,
        )

        return result
