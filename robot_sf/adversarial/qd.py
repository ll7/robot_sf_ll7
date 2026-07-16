"""Quality-diversity MAP-Elites search over adversarial scenario candidates.

Implements a dependency-light MAP-Elites (grid-archive QD) wrapper around the
existing adversarial samplers (``robot_sf/adversarial/samplers.py``). The archive
populates a declared 2D behavior grid; each elite must pass the same certification
gate that the single-objective search uses (``certification.candidate_allowed``).

Bounded first slice for issue #5308: Random + CoordinateRefinement emitters feed
the grid. The behavior descriptors are scalars already measured during evaluation:

* ``distance_to_human_min`` - closest robot/pedestrian distance;
* ``time_to_collision_min`` - minimum time-to-collision value.

Quality = the configured objective value (use ``temporal_robustness`` from #5304
when available; ``worst_case_snqi`` fallback otherwise).

This module is capability plumbing only. It does not run benchmarks or claim
evidence; the produced archive artifact and coverage report are archive paths,
not camera-ready findings.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from robot_sf.adversarial.certification import candidate_allowed
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    SearchSpaceConfig,
)
from robot_sf.adversarial.io import read_first_jsonl_record
from robot_sf.adversarial.objectives import get_objective
from robot_sf.adversarial.samplers import (
    CoordinateRefinementSampler,
    RandomCandidateSampler,
)

if TYPE_CHECKING:
    from robot_sf.adversarial.certification import CertificationStatus

QD_ARCHIVE_SCHEMA_VERSION = "adversarial_qd_archive.v1"

_BEHAVIOR_AXES = ("distance_to_human_min", "time_to_collision_min")


class BehaviorDescriptorFn(Protocol):
    """Protocol for a function that maps an evaluation to a 2D behavior descriptor."""

    def __call__(self, evaluation: CandidateEvaluation) -> tuple[float, float] | None:
        """Return (descriptor_x, descriptor_y) or None if not measurable."""


class QDEmitter(Protocol):
    """Protocol for an emitter feeding candidates into the MAP-Elites grid."""

    def sample(self) -> CandidateSpec:
        """Return the next candidate to evaluate."""

    def observe(self, evaluation: CandidateEvaluation) -> None:
        """Observe one evaluated candidate (optional, default no-op)."""


class QDEvaluator(Protocol):
    """Injected evaluator contract for the dependency-light QD capability slice."""

    def __call__(self, config: QDSearchConfig, candidate: CandidateSpec) -> CandidateEvaluation:
        """Evaluate one candidate and return its repository evaluation payload."""


def default_behavior_descriptor(evaluation: CandidateEvaluation) -> tuple[float, float] | None:
    """Return the (min distance, critical time) behavior descriptor from the record.

    Returns ``None`` when the episode record or its metrics are unavailable, so
    non-evaluable candidates do not enter the behavior grid.
    """
    if evaluation.episode_record_path is None:
        return None
    record = read_first_jsonl_record(evaluation.episode_record_path)
    if record is None:
        return None
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    distance = _finite_metric(metrics, "distance_to_human_min")
    minimum_ttc = _finite_metric(metrics, "time_to_collision_min")
    if distance is None or minimum_ttc is None:
        return None
    return (distance, minimum_ttc)


def _finite_metric(metrics: dict[str, Any], key: str) -> float | None:
    """Return a finite metric scalar or None."""
    value = metrics.get(key)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class GridSpec:
    """2D MAP-Elites grid definition over the behavior space.

    Bounds are inclusive. ``bins`` is the number of cells per axis, so the grid
    has ``bins * bins`` cells.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    bins: int = 8

    def __post_init__(self) -> None:
        """Validate grid bounds and bin count."""
        if not math.isfinite(self.x_min) or not math.isfinite(self.x_max):
            raise ValueError("grid x bounds must be finite")
        if not math.isfinite(self.y_min) or not math.isfinite(self.y_max):
            raise ValueError("grid y bounds must be finite")
        if self.x_max <= self.x_min:
            raise ValueError("grid x_max must be strictly greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("grid y_max must be strictly greater than y_min")
        if self.bins < 1:
            raise ValueError("grid bins must be >= 1")

    @property
    def cell_count(self) -> int:
        """Return the total number of cells in the grid."""
        return self.bins * self.bins

    def cell_index(self, descriptor: tuple[float, float]) -> tuple[int, int] | None:
        """Return the (ix, iy) grid cell for a descriptor, or None if out of range."""
        x, y = descriptor
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return None
        ix = min(self.bins - 1, int((x - self.x_min) / (self.x_max - self.x_min) * self.bins))
        iy = min(self.bins - 1, int((y - self.y_min) / (self.y_max - self.y_min) * self.bins))
        return (ix, iy)


@dataclass(frozen=True)
class QDArchiveCell:
    """One MAP-Elites cell holding the best elite observed so far."""

    cell: tuple[int, int]
    descriptor: tuple[float, float]
    objective_value: float
    candidate: CandidateSpec
    primary_failure: str | None
    certification_status: str
    scenario_yaml_path: str | None = None
    bundle_path: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable cell payload."""
        return {
            "cell": list(self.cell),
            "descriptor": list(self.descriptor),
            "objective_value": self.objective_value,
            "candidate": self.candidate.to_json(),
            "primary_failure": self.primary_failure,
            "certification_status": self.certification_status,
            "scenario_yaml_path": self.scenario_yaml_path,
            "bundle_path": self.bundle_path,
        }


@dataclass
class QDArchive:
    """In-memory MAP-Elites grid archive.

    Cells are keyed by ``(ix, iy)``. Only certified, measured, finite-objective
    elites are admitted; a new elite replaces the incumbent only when it scores
    strictly higher quality.
    """

    grid: GridSpec
    require_certification: bool = False
    cells: dict[tuple[int, int], QDArchiveCell] = field(default_factory=dict)

    def try_insert(
        self,
        *,
        descriptor: tuple[float, float],
        evaluation: CandidateEvaluation,
        certification_status: CertificationStatus | None,
    ) -> bool:
        """Attempt to insert an evaluated candidate into the grid.

        Returns ``True`` when the candidate was admitted (new or improving cell).
        """
        if evaluation.objective_value is None or not math.isfinite(
            float(evaluation.objective_value)
        ):
            return False
        cell = self.grid.cell_index(descriptor)
        if cell is None:
            return False
        status = (
            certification_status
            if certification_status is not None
            else evaluation.certification_status
        )
        if not candidate_allowed(status, require_certification=self.require_certification):
            return False
        score = float(evaluation.objective_value)
        incumbent = self.cells.get(cell)
        if incumbent is not None and incumbent.objective_value >= score:
            return False
        primary_failure = (
            evaluation.failure_attribution.primary_failure
            if evaluation.failure_attribution is not None
            else None
        )
        self.cells[cell] = QDArchiveCell(
            cell=cell,
            descriptor=descriptor,
            objective_value=score,
            candidate=evaluation.candidate,
            primary_failure=primary_failure,
            certification_status=status.status,
            scenario_yaml_path=(
                evaluation.scenario_yaml_path.as_posix()
                if evaluation.scenario_yaml_path is not None
                else None
            ),
            bundle_path=(
                evaluation.bundle_path.as_posix() if evaluation.bundle_path is not None else None
            ),
        )
        return True

    def filled_cell_count(self) -> int:
        """Return the number of occupied cells."""
        return len(self.cells)

    def coverage_fraction(self) -> float:
        """Return filled cells / total cells in [0, 1]."""
        total = self.grid.cell_count
        return self.filled_cell_count() / total if total else 0.0

    def qd_score(self) -> float:
        """Return the QD score: sum of objective values over occupied cells."""
        return sum(cell.objective_value for cell in self.cells.values())

    def distinct_failure_modes(self) -> set[str]:
        """Return the set of distinct certified failure mechanisms in the archive."""
        modes: set[str] = set()
        for cell in self.cells.values():
            if cell.primary_failure:
                modes.add(cell.primary_failure)
        return modes

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable archive payload."""
        entries = sorted(
            (cell.to_json() for cell in self.cells.values()),
            key=lambda item: (item["cell"][0], item["cell"][1]),
        )
        return {
            "schema_version": QD_ARCHIVE_SCHEMA_VERSION,
            "behavior_axes": list(_BEHAVIOR_AXES),
            "grid": {
                "x_min": self.grid.x_min,
                "x_max": self.grid.x_max,
                "y_min": self.grid.y_min,
                "y_max": self.grid.y_max,
                "bins": self.grid.bins,
                "cell_count": self.grid.cell_count,
            },
            "summary": {
                "filled_cell_count": self.filled_cell_count(),
                "coverage_fraction": round(self.coverage_fraction(), 6),
                "qd_score": round(self.qd_score(), 6),
                "distinct_failure_modes": sorted(self.distinct_failure_modes()),
            },
            "cells": entries,
        }


@dataclass(frozen=True)
class QDSearchConfig:
    """Configuration for one MAP-Elites adversarial search run."""

    search_space: SearchSpaceConfig
    objective: str
    grid: GridSpec
    budget: int = 64
    seed: int = 0
    require_certification: bool = False
    behavior_descriptor: BehaviorDescriptorFn = default_behavior_descriptor

    def __post_init__(self) -> None:
        """Fail closed on search settings that cannot perform an evaluation."""
        if self.budget < 1:
            raise ValueError("budget must be >= 1")


@dataclass(frozen=True)
class QDSearchResult:
    """Result of a MAP-Elites adversarial search run."""

    archive: QDArchive
    num_evaluated: int
    num_admitted: int

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable result payload."""
        payload = self.archive.to_json()
        payload["search_summary"] = {
            "num_evaluated": self.num_evaluated,
            "num_admitted": self.num_admitted,
        }
        return payload


def run_map_elites(
    config: QDSearchConfig,
    *,
    evaluator: QDEvaluator,
    certifier: Any | None = None,
    emitters: list[QDEmitter] | None = None,
) -> QDSearchResult:
    """Run a bounded MAP-Elites search over the configured emitters.

    Args:
        config: MAP-Elites run configuration.
        evaluator: Injected callable ``(config, candidate) -> CandidateEvaluation``.
            This capability slice does not materialize scenarios or adapt the four-argument
            production evaluator from ``search.run_adversarial_search``; Issue #5308 owns that
            campaign integration.
        certifier: Optional callable (candidate) -> CertificationStatus; when omitted
            the evaluation's own certification status is used as the gate.
        emitters: Optional list of emitters; defaults to Random + CoordinateRefinement.

    Returns:
        QDSearchResult with the populated archive and run counters.
    """
    objective_fn = get_objective(config.objective)
    active_emitters = (
        _default_emitters(config.search_space, seed=config.seed) if emitters is None else emitters
    )
    if not active_emitters:
        raise ValueError("emitters must contain at least one emitter")
    archive = QDArchive(grid=config.grid, require_certification=config.require_certification)

    num_evaluated = 0
    num_admitted = 0

    for index in range(config.budget):
        emitter = active_emitters[index % len(active_emitters)]
        candidate = emitter.sample()
        evaluation = evaluator(config, candidate)
        if evaluation.objective_value is None:
            score = objective_fn(evaluation)
            evaluation = evaluation.with_objective(score)
        num_evaluated += 1

        cert_status = (
            certifier(candidate) if certifier is not None else evaluation.certification_status
        )
        descriptor = config.behavior_descriptor(evaluation)
        if descriptor is None:
            for emitter in active_emitters:
                _observe(emitter, evaluation)
            continue

        admitted = archive.try_insert(
            descriptor=descriptor,
            evaluation=evaluation,
            certification_status=cert_status,
        )
        if admitted:
            num_admitted += 1
        for emitter in active_emitters:
            _observe(emitter, evaluation)

    return QDSearchResult(
        archive=archive,
        num_evaluated=num_evaluated,
        num_admitted=num_admitted,
    )


def _default_emitters(search_space: SearchSpaceConfig, *, seed: int) -> list[QDEmitter]:
    """Return the default Random + CoordinateRefinement emitter pair."""
    return [
        RandomCandidateSampler(search_space, seed=seed),
        CoordinateRefinementSampler(search_space, seed=seed + 1),
    ]


def _observe(emitter: QDEmitter, evaluation: CandidateEvaluation) -> None:
    """Notify feedback-capable emitters about one completed candidate."""
    observe = getattr(emitter, "observe", None)
    if callable(observe):
        observe(evaluation)


def write_qd_archive(result: QDSearchResult, output_path: str | Path) -> Path:
    """Write the QD archive artifact (grid coverage) to ``output_path``."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class QDComparisonRow:
    """One row of an equal-budget QD vs single-objective comparison."""

    method: str
    budget: int
    num_evaluated: int
    num_admitted_or_best: int
    filled_cells: int
    coverage_fraction: float
    qd_score: float
    distinct_failure_modes: int

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable comparison row."""
        return {
            "method": self.method,
            "budget": self.budget,
            "num_evaluated": self.num_evaluated,
            "num_admitted_or_best": self.num_admitted_or_best,
            "filled_cells": self.filled_cells,
            "coverage_fraction": round(self.coverage_fraction, 6),
            "qd_score": round(self.qd_score, 6),
            "distinct_failure_modes": self.distinct_failure_modes,
        }


@dataclass(frozen=True)
class QDComparisonReport:
    """Equal-budget comparison of MAP-Elites against a single-objective baseline."""

    qd: QDComparisonRow
    single_objective: QDComparisonRow
    grid: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable comparison report."""
        return {
            "schema_version": QD_ARCHIVE_SCHEMA_VERSION,
            "comparison_type": "equal_budget_qd_vs_single_objective",
            "grid": self.grid,
            "rows": {
                "map_elites": self.qd.to_json(),
                "single_objective": self.single_objective.to_json(),
            },
            "summary": {
                "qd_filled_cells": self.qd.filled_cells,
                "single_objective_unique_failure_modes": self.single_objective.distinct_failure_modes,
                "qd_distinct_failure_modes": self.qd.distinct_failure_modes,
                "distinct_mode_delta": self.qd.distinct_failure_modes
                - self.single_objective.distinct_failure_modes,
            },
        }


def compare_qd_vs_single_objective(
    *,
    qd_result: QDSearchResult,
    single_objective_evaluations: list[CandidateEvaluation],
    budget: int,
    grid: GridSpec,
    require_certification: bool = False,
    behavior_descriptor: BehaviorDescriptorFn = default_behavior_descriptor,
) -> QDComparisonReport:
    """Build an equal-budget comparison of QD diversity vs the single-objective baseline.

    The single-objective baseline is summarised by how many *distinct certified
    failure mechanisms* its evaluated candidates would have populated into the same
    grid (a fair, budget-matched diversity yardstick), not by its best objective value.
    """
    qd_row = QDComparisonRow(
        method="map_elites",
        budget=budget,
        num_evaluated=qd_result.num_evaluated,
        num_admitted_or_best=qd_result.num_admitted,
        filled_cells=qd_result.archive.filled_cell_count(),
        coverage_fraction=qd_result.archive.coverage_fraction(),
        qd_score=qd_result.archive.qd_score(),
        distinct_failure_modes=len(qd_result.archive.distinct_failure_modes()),
    )

    so_archive = QDArchive(grid=grid, require_certification=require_certification)
    for evaluation in single_objective_evaluations:
        descriptor = behavior_descriptor(evaluation)
        if descriptor is None:
            continue
        so_archive.try_insert(
            descriptor=descriptor,
            evaluation=evaluation,
            certification_status=evaluation.certification_status,
        )

    so_row = QDComparisonRow(
        method="single_objective",
        budget=budget,
        num_evaluated=len(single_objective_evaluations),
        num_admitted_or_best=1 if so_archive.cells else 0,
        filled_cells=so_archive.filled_cell_count(),
        coverage_fraction=so_archive.coverage_fraction(),
        qd_score=so_archive.qd_score(),
        distinct_failure_modes=len(so_archive.distinct_failure_modes()),
    )

    return QDComparisonReport(
        qd=qd_row,
        single_objective=so_row,
        grid={
            "bins": grid.bins,
            "cell_count": grid.cell_count,
            "x_min": grid.x_min,
            "x_max": grid.x_max,
            "y_min": grid.y_min,
            "y_max": grid.y_max,
        },
    )


__all__ = [
    "QD_ARCHIVE_SCHEMA_VERSION",
    "BehaviorDescriptorFn",
    "GridSpec",
    "QDArchive",
    "QDArchiveCell",
    "QDComparisonReport",
    "QDComparisonRow",
    "QDEmitter",
    "QDEvaluator",
    "QDSearchConfig",
    "QDSearchResult",
    "compare_qd_vs_single_objective",
    "default_behavior_descriptor",
    "run_map_elites",
    "write_qd_archive",
]
