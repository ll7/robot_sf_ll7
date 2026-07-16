"""CMA-ME emitter for the MAP-Elites quality-diversity adversarial search (issue #5308).

Implements a CMA-ME (CMA-ES-Multimodal-Evolution) emitter that feeds the
``robot_sf/adversarial/qd.py`` MAP-Elites grid. Unlike the single-objective
``CmaEsCandidateSampler`` (which improves one global incumbent), a CMA-ME emitter
runs a separate CMA-ES instance **per target archive cell** so the search populates
DISTINCT behavior cells instead of only the globally best worst case. This is the
remaining "later CMA-ME work" named in issue #5308; it is a NEW capability, not a
duplicate of PR #5846 (grid archive) or PR #5852 (production wiring).

Target-cell selection is diversity-driven: the emitter repeatedly picks the
least-filled cell and seeds a local CMA-ES around that cell's incumbent candidate
(resampling inside the cell bounds). Every evaluated candidate is observed so the
active optimizer's generation is fed back once the population is fully observed, and
so the cell-selection frontier advances.

The heavyweight ``cma`` dependency is imported lazily. The emitter also accepts an
injected ``optimizer_factory`` so the CMA-ME scheduling/selection logic can be
validated on CPU without ``cma`` installed (the canonical QD test contract).

This module is capability plumbing only. It does not run benchmarks or claim
evidence; the populated archive artifact is an archive path, not a camera-ready
finding.
"""

from __future__ import annotations

import math
from random import Random
from typing import TYPE_CHECKING, Any, Protocol

from robot_sf.adversarial.config import CandidateEvaluation, CandidateSpec
from robot_sf.adversarial.qd import QDArchive

if TYPE_CHECKING:
    from robot_sf.adversarial.qd import BehaviorDescriptorFn

_CMA_ME_SCHEMA_VERSION = "adversarial_cma_me_emitter.v1"


class CellOptimizer(Protocol):
    """Protocol for a per-cell continuous optimizer backing a CMA-ME emitter."""

    def ask(self) -> list[float]:
        """Return one proposed active-dimension vector."""

    def tell(self, vectors: list[list[float]], values: list[float]) -> None:
        """Feed a finished population (vectors + objective values) to the optimizer."""

    def stop(self) -> bool:
        """Return True when the optimizer should be restarted around the incumbent."""

    @property
    def mean(self) -> list[float]:
        """Return the current distribution mean (active-dimension vector)."""


class OptimizerFactory(Protocol):
    """Protocol for building a per-cell optimizer on demand."""

    def __call__(
        self,
        *,
        lower: list[float],
        upper: list[float],
        x0: list[float],
        seed: int,
    ) -> CellOptimizer:
        """Build a fresh optimizer for one target cell."""


def default_optimizer_factory(
    *,
    sigma_fraction: float = 0.25,
    popsize: int | None = None,
) -> OptimizerFactory:
    """Return a CMA-ES-backed optimizer factory (imports ``cma`` lazily)."""

    def _build(
        *,
        lower: list[float],
        upper: list[float],
        x0: list[float],
        seed: int,
    ) -> CellOptimizer:
        return _CmaCellOptimizer(
            lower=lower,
            upper=upper,
            x0=x0,
            seed=seed,
            sigma_fraction=sigma_fraction,
            popsize=popsize,
        )

    return _build


class _CmaCellOptimizer:
    """Per-cell CMA-ES optimizer wrapping the ``cma`` package lazily."""

    def __init__(
        self,
        *,
        lower: list[float],
        upper: list[float],
        x0: list[float],
        seed: int,
        sigma_fraction: float = 0.25,
        popsize: int | None = None,
    ) -> None:
        """Initialize a bounded CMA-ES instance for one cell."""
        cma = _import_cma()
        self._cma = cma
        spans = [upper[i] - lower[i] for i in range(len(lower))]
        max_span = max(spans) if spans else 1.0
        sigma0 = max(1e-3, sigma_fraction * max_span)
        opts: dict[str, Any] = {
            "bounds": [lower, upper],
            "seed": int(seed),
            "verbose": -9,
        }
        if popsize is not None:
            if popsize < 1:
                raise ValueError("popsize must be >= 1")
            opts["popsize"] = int(popsize)
        self._es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)
        self._lower = list(lower)
        self._upper = list(upper)

    def ask(self) -> list[float]:
        """Return one CMA-ES proposal."""
        population = self._es.ask()
        self._pending = list(population)
        return list(population[0])

    def tell(self, vectors: list[list[float]], values: list[float]) -> None:
        """Feed a finished population to CMA-ES."""
        self._es.tell(vectors, values)

    def stop(self) -> bool:
        """Return True when CMA-ES converged and should be restarted."""
        return bool(self._es.stop())

    @property
    def mean(self) -> list[float]:
        """Return the current CMA-ES mean vector."""
        return list(self._es.mean)


def _import_cma() -> Any:
    """Import the cma package or raise an actionable optional-dependency error."""
    from robot_sf.common.optional_import import try_import  # noqa: PLC0415

    cma = try_import("cma")
    if cma is None:
        raise RuntimeError(
            "CMA-ME emitter requires cma. Install project dependencies with "
            "`uv sync --all-extras` before using the CMA-ES-backed quality-diversity emitter."
        )
    return cma


def _continuous_bounds(space: Any) -> dict[str, tuple[float, float]]:
    """Return the (min, max) bound for each continuous candidate dimension."""
    ranges: dict[str, Any] = {
        "start.x": space.start_x,
        "start.y": space.start_y,
        "goal.x": space.goal_x,
        "goal.y": space.goal_y,
        "spawn_time_s": space.spawn_time_s,
        "pedestrian_speed_mps": space.pedestrian_speed_mps,
        "pedestrian_delay_s": space.pedestrian_delay_s,
    }
    return {name: (float(r.min), float(r.max)) for name, r in ranges.items()}


def _clamp(bounds: tuple[float, float], value: float) -> float:
    """Clamp a scalar into its inclusive range."""
    return min(float(bounds[1]), max(float(bounds[0]), float(value)))


class CMaMeEmitter:
    """CMA-ME emitter that targets least-filled archive cells with per-cell CMA-ES.

    The emitter implements the ``QDEmitter`` protocol (``sample`` / ``observe``) so it
    can be mixed with the existing Random + CoordinateRefinement emitters in
    ``run_map_elites``. It reads the live ``QDArchive`` to pick the next least-filled
    cell, seeds a CMA-ES around that cell's incumbent candidate, and proposes vectors
    mapped back into the continuous search space. Discrete ``scenario_seed`` is drawn
    per proposal from the seeded RNG.

    When ``cma`` is unavailable, inject an ``optimizer_factory`` (e.g. a lightweight
    Gaussian stub) so the cell-selection and proposal-mapping logic still runs on CPU.
    """

    _CONTINUOUS_DIMENSIONS = (
        "start.x",
        "start.y",
        "goal.x",
        "goal.y",
        "spawn_time_s",
        "pedestrian_speed_mps",
        "pedestrian_delay_s",
    )

    def __init__(
        self,
        search_space: Any,
        archive: QDArchive,
        *,
        seed: int,
        behavior_descriptor: BehaviorDescriptorFn | None = None,
        optimizer_factory: OptimizerFactory | None = None,
        sigma_fraction: float = 0.25,
        popsize: int | None = None,
    ) -> None:
        """Initialize a diversity-driven CMA-ME emitter over the archive grid.

        Args:
            search_space: ``SearchSpaceConfig`` supplying continuous bounds + seeds.
            archive: Live MAP-Elites archive whose cells drive target selection.
            seed: Deterministic RNG seed for proposal / seed sampling.
            behavior_descriptor: Descriptor used to map candidates to cells; defaults
                to ``qd.default_behavior_descriptor``.
            optimizer_factory: Injected per-cell optimizer builder; defaults to CMA-ES
                (``default_optimizer_factory``), which imports ``cma`` lazily.
            sigma_fraction: Initial CMA-ES step fraction of the largest span.
            popsize: Optional explicit CMA-ES population size.
        """
        if not math.isfinite(sigma_fraction) or sigma_fraction <= 0.0:
            raise ValueError("sigma_fraction must be finite and positive")
        from robot_sf.adversarial.qd import default_behavior_descriptor  # noqa: PLC0415

        self._search_space = search_space
        self._archive = archive
        self._rng = Random(seed)
        self._seed_rng = Random(seed ^ 0x9E3779B9)
        self._behavior_descriptor = behavior_descriptor or default_behavior_descriptor
        self._optimizer_factory = optimizer_factory or default_optimizer_factory(
            sigma_fraction=sigma_fraction, popsize=popsize
        )
        self._bounds = _continuous_bounds(search_space)
        self._active_dims = [
            name
            for name in self._CONTINUOUS_DIMENSIONS
            if self._bounds[name][0] != self._bounds[name][1]
        ]
        self._fixed_values = {
            name: 0.5 * (self._bounds[name][0] + self._bounds[name][1])
            for name in self._CONTINUOUS_DIMENSIONS
            if self._bounds[name][0] == self._bounds[name][1]
        }
        self._current_cell: tuple[int, int] | None = None
        self._optimizer: CellOptimizer | None = None
        self._pending_vec: list[float] | None = None
        self._in_flight: list[tuple[CandidateSpec, list[float]]] = []
        self._observed: list[tuple[list[float], float]] = []

    def _select_target_cell(self) -> tuple[int, int] | None:
        """Pick the least-filled cell; return None when the grid is full."""
        counts = {cell: self._archive.cells.get(cell) is not None for cell in self._all_cells()}
        empty = [cell for cell, filled in counts.items() if not filled]
        if not empty:
            return None
        return empty[(self._rng.randint(0, len(empty) - 1))]

    def _all_cells(self) -> list[tuple[int, int]]:
        """Return every grid cell coordinate in row-major order."""
        grid = self._archive.grid
        return [(ix, iy) for ix in range(grid.bins) for iy in range(grid.bins)]

    def _incumbent_vec(self, cell: tuple[int, int]) -> list[float]:
        """Return the active-dimension vector of the cell incumbent, else the midpoint."""
        incumbent = self._archive.cells.get(cell)
        if incumbent is not None:
            candidate = incumbent.candidate
            return self._candidate_vec(candidate)
        lower = [self._bounds[name][0] for name in self._active_dims]
        upper = [self._bounds[name][1] for name in self._active_dims]
        return [0.5 * (lower[i] + upper[i]) for i in range(len(lower))]

    def _candidate_vec(self, candidate: CandidateSpec) -> list[float]:
        """Return the active-dimension vector of a candidate."""
        values = {
            "start.x": candidate.start.x,
            "start.y": candidate.start.y,
            "goal.x": candidate.goal.x,
            "goal.y": candidate.goal.y,
            "spawn_time_s": candidate.spawn_time_s,
            "pedestrian_speed_mps": candidate.pedestrian_speed_mps,
            "pedestrian_delay_s": candidate.pedestrian_delay_s,
        }
        return [float(values[name]) for name in self._active_dims]

    def _make_candidate(self, vec: list[float]) -> CandidateSpec:
        """Build a candidate from an active-dimension vector plus fixed dims."""
        values = dict(self._fixed_values)
        for idx, name in enumerate(self._active_dims):
            values[name] = _clamp(self._bounds[name], vec[idx])
        return CandidateSpec(
            start=_Pose2D(values["start.x"], values["start.y"]),
            goal=_Pose2D(values["goal.x"], values["goal.y"]),
            spawn_time_s=values["spawn_time_s"],
            pedestrian_speed_mps=values["pedestrian_speed_mps"],
            pedestrian_delay_s=values["pedestrian_delay_s"],
            scenario_seed=self._seed_rng.randint(
                int(self._search_space.scenario_seed.min),
                int(self._search_space.scenario_seed.max),
            ),
        )

    def _start_cell_optimizer(self, cell: tuple[int, int]) -> None:
        """Seed a fresh per-cell optimizer around the cell incumbent."""
        lower = [self._bounds[name][0] for name in self._active_dims]
        upper = [self._bounds[name][1] for name in self._active_dims]
        x0 = self._incumbent_vec(cell)
        self._current_cell = cell
        seed = self._rng.randint(0, 2**31 - 1)
        self._optimizer = self._optimizer_factory(lower=lower, upper=upper, x0=x0, seed=seed)

    def sample(self) -> CandidateSpec:
        """Return the next CMA-ME proposal, reseeding on a full grid or restart."""
        if self._current_cell is None or self._optimizer is None:
            cell = self._select_target_cell()
            if cell is None:
                return self._random_fallback()
            self._start_cell_optimizer(cell)
        if self._pending_vec is not None:
            vec = self._pending_vec
            self._pending_vec = None
        elif self._optimizer.stop():
            cell = self._select_target_cell()
            if cell is None:
                return self._random_fallback()
            self._start_cell_optimizer(cell)
            vec = self._optimizer.ask()
        else:
            vec = self._optimizer.ask()
        candidate = self._make_candidate(vec)
        self._in_flight.append((candidate, vec))
        return candidate

    def _random_fallback(self) -> CandidateSpec:
        """Proposal used only when the grid is full (emitter becomes a pass-through)."""
        return self._search_space.sample_candidate(self._seed_rng)

    def observe(self, evaluation: CandidateEvaluation) -> None:
        """Observe one evaluated candidate and feed the active optimizer generation.

        The optimizer is told only once its full population has been observed; scores
        of unevaluable candidates are treated as the worst value so the optimizer steps
        away from them.
        """
        if self._optimizer is None:
            return
        for index, (_candidate, vec) in enumerate(self._in_flight):
            if _candidate == evaluation.candidate:
                self._in_flight.pop(index)
                score = evaluation.objective_value
                value = float(score) if score is not None and math.isfinite(float(score)) else -1e9
                self._observed.append((list(vec), value))
                if not self._in_flight:
                    self._flush_generation()
                return

    def _flush_generation(self) -> None:
        """Feed the buffered generation to the optimizer once fully observed."""
        vectors = [vec for vec, _value in self._observed]
        values = [value for _vec, value in self._observed]
        if vectors:
            self._optimizer.tell(vectors, values)
        self._observed = []


def _Pose2D(x: float, y: float) -> Any:
    """Build a ``Pose2D`` from scalars (kept local to avoid a top-level import cycle)."""
    from robot_sf.adversarial.config import Pose2D  # noqa: PLC0415

    return Pose2D(float(x), float(y))


__all__ = [
    "_CMA_ME_SCHEMA_VERSION",
    "CMaMeEmitter",
    "CellOptimizer",
    "OptimizerFactory",
    "default_optimizer_factory",
]
