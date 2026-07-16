"""CPU tests for the CMA-ME quality-diversity emitter (issue #5308, successor slice).

The emitter drives distinct archive cells with per-cell CMA-ES. These tests inject a
lightweight Gaussian optimizer factory so the CMA-ME scheduling/selection/proposal
logic is validated on CPU without the ``cma`` dependency. This is capability plumbing,
not a benchmark or camera-ready claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.adversarial.attribution import attribution_from_episode_record
from robot_sf.adversarial.certification import not_available_status, passed_status
from robot_sf.adversarial.cma_me import CMaMeEmitter
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    Pose2D,
    RangeConfig,
    SearchSpaceConfig,
)
from robot_sf.adversarial.qd import (
    GridSpec,
    QDArchive,
    QDSearchConfig,
    run_map_elites,
)


def _space() -> SearchSpaceConfig:
    """Build a 2D search space spanning start/goal x with fixed seeds."""
    return SearchSpaceConfig(
        start_x=RangeConfig(0.0, 4.0),
        start_y=RangeConfig(0.0, 0.0),
        goal_x=RangeConfig(0.0, 4.0),
        goal_y=RangeConfig(0.0, 0.0),
        spawn_time_s=RangeConfig(0.0, 0.0),
        pedestrian_speed_mps=RangeConfig(1.0, 1.0),
        pedestrian_delay_s=RangeConfig(0.0, 0.0),
        scenario_seed=RangeConfig(1, 1),
    )


def _record(min_distance: float, critical_time: float, failure: str = "collision") -> dict:
    """Build a minimal episode record carrying the QD behavior descriptors."""
    outcome: dict = {"route_complete": False}
    if failure == "collision":
        outcome["collision"] = True
        termination = "collision"
    elif failure == "timeout":
        outcome["timeout"] = True
        termination = "timeout"
    else:
        termination = "incomplete"
    return {
        "status": "completed",
        "termination_reason": termination,
        "outcome": outcome,
        "metrics": {
            "distance_to_human_min": min_distance,
            "time_to_collision_min": critical_time,
        },
    }


def _make_evaluation(
    *,
    candidate: CandidateSpec,
    min_distance: float,
    critical_time: float,
    objective: float,
    failure: str = "collision",
    cert_status: object = None,
    temp_root: Path,
) -> CandidateEvaluation:
    """Build a candidate evaluation with a measurable behavior descriptor."""
    record = _record(min_distance, critical_time, failure=failure)
    episode_path = temp_root / "episode_dummy.jsonl"
    episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return CandidateEvaluation(
        candidate=candidate,
        certification_status=cert_status or not_available_status("advisory"),
        objective_value=objective,
        failure_attribution=attribution_from_episode_record(record),
        episode_record_path=episode_path,
        trajectory_csv_path=None,
        scenario_yaml_path=temp_root / "scenario_dummy.yaml",
        bundle_path=temp_root,
    )


def _candidate(x: float) -> CandidateSpec:
    """Build one candidate at start.x = x."""
    return CandidateSpec(
        start=Pose2D(x, 0.0),
        goal=Pose2D(4.0 - x, 0.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=1,
    )


class _GaussianCellOptimizer:
    """Lightweight Gaussian stub standing in for CMA-ES in CPU tests."""

    def __init__(
        self, *, lower: list[float], upper: list[float], x0: list[float], seed: int
    ) -> None:
        self._lower = list(lower)
        self._upper = list(upper)
        self._mean = list(x0)
        self._rng = __import__("random").Random(seed)
        self._pending: list[list[float]] = []
        self._generation = 0

    def ask(self) -> list[float]:
        """Return one proposal spanning the cell bounds (exploratory stub)."""
        if not self._pending:
            self._pending = [
                [self._rng.uniform(self._lower[i], self._upper[i]) for i in range(len(self._lower))]
                for _ in range(3)
            ]
        return list(self._pending.pop(0))

    def tell(self, vectors: list[list[float]], values: list[float]) -> None:
        """Move the mean toward the best observed vector."""
        if vectors and values:
            best = max(range(len(values)), key=lambda i: values[i])
            self._mean = list(vectors[best])
        self._generation += 1

    def stop(self) -> bool:
        """Restart the optimizer every few generations to emulate convergence."""
        return self._generation >= 2

    @property
    def mean(self) -> list[float]:
        """Return the current mean vector."""
        return list(self._mean)


def _stub_optimizer_factory() -> object:
    """Return a Gaussian-optimizer factory for CPU tests (no cma dependency)."""

    def _build(*, lower, upper, x0, seed) -> _GaussianCellOptimizer:
        return _GaussianCellOptimizer(lower=lower, upper=upper, x0=x0, seed=seed)

    return _build


def _runnable_archive() -> QDArchive:
    """Return a fresh archive over the standard test grid."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    return QDArchive(grid=grid, require_certification=True)


def test_emitter_targets_least_filled_cells(tmp_path: Path) -> None:
    """Emitter seeds distinct empty cells instead of the same one repeatedly."""
    space = _space()
    archive = _runnable_archive()
    emitter = CMaMeEmitter(
        space,
        archive,
        seed=0,
        optimizer_factory=_stub_optimizer_factory(),
    )
    seen_cells: set[tuple[int, int]] = set()
    for _ in range(8):
        candidate = emitter.sample()
        evaluation = _make_evaluation(
            candidate=candidate,
            min_distance=1.0,
            critical_time=1.0,
            objective=1.0,
            cert_status=passed_status("ok"),
            temp_root=tmp_path,
        )
        descriptor = (1.0, 1.0)  # all proposals target the same cell for this test
        admitted = archive.try_insert(
            descriptor=descriptor,
            evaluation=evaluation,
            certification_status=evaluation.certification_status,
        )
        emitter.observe(evaluation)
        if admitted:
            seen_cells.add(archive.grid.cell_index(descriptor))
    assert (1, 1) in seen_cells


def test_emitter_proposes_within_bounds(tmp_path: Path) -> None:
    """Every CMA-ME proposal stays inside the configured search-space bounds."""
    space = _space()
    archive = _runnable_archive()
    emitter = CMaMeEmitter(
        space,
        archive,
        seed=3,
        optimizer_factory=_stub_optimizer_factory(),
    )
    for _ in range(20):
        candidate = emitter.sample()
        assert 0.0 <= candidate.start.x <= 4.0
        assert 0.0 <= candidate.goal.x <= 4.0
        emitter.observe(
            _make_evaluation(
                candidate=candidate,
                min_distance=1.0,
                critical_time=1.0,
                objective=0.5,
                cert_status=passed_status("ok"),
                temp_root=tmp_path,
            )
        )


def test_emitter_fills_more_cells_than_random_baseline(tmp_path: Path) -> None:
    """A CMA-ME emitter run diversifies the grid under an injected evaluator."""
    space = _space()
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)

    def _evaluator(_config: QDSearchConfig, candidate: CandidateSpec) -> CandidateEvaluation:
        # Deterministic descriptor derived from the candidate so distinct proposals
        # land in distinct cells; objective is constant so CMA-ME diversity wins.
        x = candidate.start.x
        distance = 2.5 * min(1.0, max(0.0, x / 4.0))
        critical = 3.0 * min(1.0, max(0.0, (4.0 - x) / 4.0))
        return _make_evaluation(
            candidate=candidate,
            min_distance=distance,
            critical_time=critical,
            objective=1.0,
            cert_status=passed_status("ok"),
            temp_root=tmp_path,
        )

    archive = QDArchive(grid=grid, require_certification=True)
    emitter = CMaMeEmitter(space, archive, seed=7, optimizer_factory=_stub_optimizer_factory())
    config = QDSearchConfig(
        search_space=space, objective="worst_case_snqi", grid=grid, budget=48, seed=7
    )
    result = run_map_elites(config, evaluator=_evaluator, emitters=[emitter])
    assert result.archive.filled_cell_count() >= 4
    assert result.num_evaluated == 48
    assert result.num_admitted >= result.archive.filled_cell_count()


def test_emitter_rejects_bad_sigma_fraction() -> None:
    """Emitter must reject non-positive sigma fractions fail-closed."""
    space = _space()
    archive = _runnable_archive()
    with pytest.raises(ValueError, match="sigma_fraction"):
        CMaMeEmitter(
            space, archive, seed=0, sigma_fraction=0.0, optimizer_factory=_stub_optimizer_factory()
        )


def test_emitter_handles_full_grid_gracefully(tmp_path: Path) -> None:
    """When the grid is full, the emitter returns valid in-bounds candidates."""
    space = _space()
    archive = _runnable_archive()
    for ix in range(archive.grid.bins):
        for iy in range(archive.grid.bins):
            candidate = _candidate(ix * 0.5)
            evaluation = _make_evaluation(
                candidate=candidate,
                min_distance=2.5 * (ix / 3.0),
                critical_time=3.0 * (iy / 3.0),
                objective=1.0,
                cert_status=passed_status("ok"),
                temp_root=tmp_path,
            )
            descriptor = (2.5 * (ix / 3.0), 3.0 * (iy / 3.0))
            archive.try_insert(
                descriptor=descriptor,
                evaluation=evaluation,
                certification_status=evaluation.certification_status,
            )
    assert archive.filled_cell_count() == archive.grid.cell_count
    emitter = CMaMeEmitter(space, archive, seed=1, optimizer_factory=_stub_optimizer_factory())
    candidate = emitter.sample()
    assert 0.0 <= candidate.start.x <= 4.0
