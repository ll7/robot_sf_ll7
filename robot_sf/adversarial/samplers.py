"""Candidate samplers for adversarial search."""

from __future__ import annotations

import math
from random import Random
from typing import TYPE_CHECKING, Any, Protocol

from robot_sf.adversarial.config import (
    CandidateSpec,
    Pose2D,
    RangeConfig,
    SearchSpaceConfig,
    WarmStartCandidate,
)
from robot_sf.common.optional_import import try_import

if TYPE_CHECKING:
    from robot_sf.adversarial.config import CandidateEvaluation


class CandidateSampler(Protocol):
    """Protocol for optimizer-backed candidate samplers."""

    def sample(self) -> CandidateSpec:
        """Return the next candidate."""


class FeedbackCandidateSampler(CandidateSampler, Protocol):
    """Candidate sampler that can update optimizer state from evaluations."""

    def observe(self, evaluation: CandidateEvaluation) -> None:
        """Observe one evaluated candidate."""


def build_sampler(
    name: str,
    search_space: SearchSpaceConfig,
    *,
    seed: int,
    warm_start: tuple[WarmStartCandidate, ...] = (),
) -> CandidateSampler:
    """Build a named adversarial sampler, optionally seeded with warm starts.

    Warm starts are knife-edge candidates from seed-sensitivity analysis
    (#5816/#5817) the optimizer should begin from. Each sampler family consumes
    them differently: random/coordinate samplers drain them first; Optuna
    enqueues them as fixed trials; CMA-ES centers initial exploration on them.
    """
    key = name.strip().lower()
    if key == "random":
        return RandomCandidateSampler(search_space, seed=seed, warm_start=warm_start)
    if key == "coordinate":
        return CoordinateRefinementSampler(search_space, seed=seed, warm_start=warm_start)
    if key == "optuna":
        return OptunaCandidateSampler(search_space, seed=seed, warm_start=warm_start)
    if key == "cmaes":
        return CmaEsCandidateSampler(search_space, seed=seed, warm_start=warm_start)
    raise ValueError("sampler must be one of: random, coordinate, optuna, cmaes")


class RandomCandidateSampler:
    """Dependency-light random-search sampler."""

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        *,
        seed: int,
        warm_start: tuple[WarmStartCandidate, ...] = (),
    ) -> None:
        """Initialize the sampler with a search space and deterministic seed."""
        self._search_space = search_space
        self._rng = Random(seed)
        self._warm_starts: list[CandidateSpec] = [warm.candidate for warm in warm_start]

    def sample(self) -> CandidateSpec:
        """Return the next warm-start candidate, then random candidates."""
        if self._warm_starts:
            return self._warm_starts.pop(0)
        return self._search_space.sample_candidate(self._rng)


class CoordinateRefinementSampler:
    """Deterministic dependency-light local-search sampler.

    The sampler starts at the search-space midpoint and then proposes coordinate
    perturbations around the best scored candidate observed so far. It is deliberately
    small: use it for synthetic or bounded stress-search pilots, not as a replacement
    for CMA-ES/Bayesian optimizers when those dependencies are justified.
    """

    _DIMENSIONS = (
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
        search_space: SearchSpaceConfig,
        *,
        seed: int,
        step_fraction: float = 0.5,
        warm_start: tuple[WarmStartCandidate, ...] = (),
    ) -> None:
        """Initialize a deterministic coordinate-refinement sampler."""
        if not math.isfinite(step_fraction) or step_fraction <= 0.0:
            raise ValueError("step_fraction must be finite and positive")
        self._search_space = search_space
        self._rng = Random(seed)
        self._step_fraction = float(step_fraction)
        self._best_candidate: CandidateSpec | None = None
        self._best_score: float | None = None
        self._iteration = 0
        self._warm_starts: list[CandidateSpec] = [warm.candidate for warm in warm_start]

    def sample(self) -> CandidateSpec:
        """Return the next warm-start, midpoint, or coordinate-refinement candidate."""
        if self._warm_starts:
            return self._warm_starts.pop(0)
        if self._best_candidate is None:
            self._iteration += 1
            return self._midpoint_candidate()

        dimension = self._DIMENSIONS[(self._iteration - 1) % len(self._DIMENSIONS)]
        direction = 1.0 if ((self._iteration - 1) // len(self._DIMENSIONS)) % 2 == 0 else -1.0
        self._iteration += 1
        return self._perturb(self._best_candidate, dimension=dimension, direction=direction)

    def observe(self, evaluation: CandidateEvaluation) -> None:
        """Update the local-search incumbent from a scored candidate."""
        score = evaluation.objective_value
        if score is None or not math.isfinite(float(score)):
            return
        if self._best_score is None or float(score) > self._best_score:
            self._best_score = float(score)
            self._best_candidate = evaluation.candidate

    def _midpoint_candidate(self) -> CandidateSpec:
        """Return the center point of the configured search space."""
        return CandidateSpec(
            start=Pose2D(
                _midpoint(self._search_space.start_x), _midpoint(self._search_space.start_y)
            ),
            goal=Pose2D(_midpoint(self._search_space.goal_x), _midpoint(self._search_space.goal_y)),
            spawn_time_s=_midpoint(self._search_space.spawn_time_s),
            pedestrian_speed_mps=_midpoint(self._search_space.pedestrian_speed_mps),
            pedestrian_delay_s=_midpoint(self._search_space.pedestrian_delay_s),
            scenario_seed=self._rng.randint(
                int(self._search_space.scenario_seed.min),
                int(self._search_space.scenario_seed.max),
            ),
        )

    def _perturb(
        self,
        candidate: CandidateSpec,
        *,
        dimension: str,
        direction: float,
    ) -> CandidateSpec:
        """Return ``candidate`` with one coordinate moved within configured bounds."""

        def move(value: float, bounds: RangeConfig) -> float:
            """Move a scalar by one normalized step while respecting bounds.

            Returns:
                float: Clamped coordinate value.
            """
            span = bounds.max - bounds.min
            if span <= 0.0:
                return float(value)
            return min(
                bounds.max, max(bounds.min, float(value) + direction * span * self._step_fraction)
            )

        if dimension == "start.x":
            return _replace_candidate(
                candidate, start_x=move(candidate.start.x, self._search_space.start_x)
            )
        if dimension == "start.y":
            return _replace_candidate(
                candidate, start_y=move(candidate.start.y, self._search_space.start_y)
            )
        if dimension == "goal.x":
            return _replace_candidate(
                candidate, goal_x=move(candidate.goal.x, self._search_space.goal_x)
            )
        if dimension == "goal.y":
            return _replace_candidate(
                candidate, goal_y=move(candidate.goal.y, self._search_space.goal_y)
            )
        if dimension == "spawn_time_s":
            return _replace_candidate(
                candidate,
                spawn_time_s=move(candidate.spawn_time_s, self._search_space.spawn_time_s),
            )
        if dimension == "pedestrian_speed_mps":
            return _replace_candidate(
                candidate,
                pedestrian_speed_mps=move(
                    candidate.pedestrian_speed_mps,
                    self._search_space.pedestrian_speed_mps,
                ),
            )
        if dimension == "pedestrian_delay_s":
            return _replace_candidate(
                candidate,
                pedestrian_delay_s=move(
                    candidate.pedestrian_delay_s, self._search_space.pedestrian_delay_s
                ),
            )
        raise AssertionError(f"unknown refinement dimension: {dimension}")


class OptunaCandidateSampler:
    """Optuna-backed feedback sampler for bounded adversarial search pilots.

    The sampler uses Optuna's ask/tell API so the existing sequential adversarial
    runner can provide objective feedback through ``observe`` without changing
    the runner contract.
    """

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        *,
        seed: int,
        warm_start: tuple[WarmStartCandidate, ...] = (),
    ) -> None:
        """Initialize an Optuna study with deterministic sampler seed."""
        optuna = _import_optuna()
        sampler = optuna.samplers.TPESampler(seed=int(seed))
        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        self._trial_state = optuna.trial.TrialState
        self._search_space = search_space
        self._pending_trials: list[tuple[CandidateSpec, Any]] = []
        # Enqueue warm starts as fixed trials so they are proposed first.
        for warm in warm_start:
            candidate = warm.candidate
            fixed: dict[str, Any] = {
                "start.x": float(candidate.start.x),
                "start.y": float(candidate.start.y),
                "goal.x": float(candidate.goal.x),
                "goal.y": float(candidate.goal.y),
                "spawn_time_s": float(candidate.spawn_time_s),
                "pedestrian_speed_mps": float(candidate.pedestrian_speed_mps),
                "pedestrian_delay_s": float(candidate.pedestrian_delay_s),
                "scenario_seed": int(candidate.scenario_seed),
            }
            self._study.enqueue_trial(fixed)

    def sample(self) -> CandidateSpec:
        """Return the next enqueued warm-start trial, then optimizer proposals."""
        trial = self._study.ask()
        candidate = CandidateSpec(
            start=Pose2D(
                _suggest_float(trial, "start.x", self._search_space.start_x),
                _suggest_float(trial, "start.y", self._search_space.start_y),
            ),
            goal=Pose2D(
                _suggest_float(trial, "goal.x", self._search_space.goal_x),
                _suggest_float(trial, "goal.y", self._search_space.goal_y),
            ),
            spawn_time_s=_suggest_float(trial, "spawn_time_s", self._search_space.spawn_time_s),
            pedestrian_speed_mps=_suggest_float(
                trial,
                "pedestrian_speed_mps",
                self._search_space.pedestrian_speed_mps,
            ),
            pedestrian_delay_s=_suggest_float(
                trial, "pedestrian_delay_s", self._search_space.pedestrian_delay_s
            ),
            scenario_seed=_suggest_int(trial, "scenario_seed", self._search_space.scenario_seed),
        )
        self._pending_trials.append((candidate, trial))
        return candidate

    def observe(self, evaluation: CandidateEvaluation) -> None:
        """Tell Optuna the objective value for a completed proposal."""
        for index, (candidate, trial) in enumerate(self._pending_trials):
            if candidate == evaluation.candidate:
                self._pending_trials.pop(index)
                score = evaluation.objective_value
                if score is None or not math.isfinite(float(score)):
                    self._study.tell(trial, state=self._trial_state.FAIL)
                    return
                self._study.tell(trial, float(score))
                return


def _import_optuna() -> Any:
    """Import Optuna or raise an actionable optional-dependency error."""
    optuna = try_import("optuna")
    if optuna is None:
        raise RuntimeError(
            "OptunaCandidateSampler requires optuna. Install project dependencies with "
            "`uv sync --all-extras` before using the optimizer-backed adversarial sampler."
        )
    return optuna


def _import_cma() -> Any:
    """Import the cma package or raise an actionable optional-dependency error."""
    cma = try_import("cma")
    if cma is None:
        raise RuntimeError(
            "CmaEsCandidateSampler requires cma. Install project dependencies with "
            "`uv sync --all-extras` before using the optimizer-backed adversarial sampler."
        )
    return cma


class CmaEsCandidateSampler:
    """CMA-ES-class feedback sampler for bounded adversarial search pilots.

    Covariance Matrix Adaptation Evolution Strategy (via the ``cma`` package)
    over the continuous candidate dimensions. The discrete ``scenario_seed`` is
    drawn per proposal from the seeded RNG. Implements the
    ``FeedbackCandidateSampler`` contract so the sequential adversarial runner
    can feed objective values back through ``observe`` without changing the
    runner contract. Used by issue #5326 to cover the CMA-ES-class search family
    required by the comparison scope.
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
        search_space: SearchSpaceConfig,
        *,
        seed: int,
        sigma_fraction: float = 0.25,
        popsize: int | None = None,
        warm_start: tuple[WarmStartCandidate, ...] = (),
    ) -> None:
        """Initialize a CMA-ES strategy over the bounded continuous dimensions.

        Degenerate (min == max) dimensions are held fixed at their midpoint and
        excluded from the CMA-ES vector, so the sampler also works on the
        fixed-range fixtures used by cheap-lane tests. The sequential adversarial
        runner calls ``sample``/``observe`` one candidate at a time, so the
        sampler buffers a full generation (CMA-ES population) internally and
        feeds it back to the optimizer only once the whole generation has been
        observed. The first warm start is used as the initial mean (x0) when it
        lies inside the active-dimension bounds; warm starts are otherwise
        proposed as the first candidates of the search.
        """
        if not math.isfinite(sigma_fraction) or sigma_fraction <= 0.0:
            raise ValueError("sigma_fraction must be finite and positive")
        cma = _import_cma()
        self._search_space = search_space
        self._cma = cma
        self._rng = Random(seed)
        self._seed_rng = Random(seed ^ 0x9E3779B9)
        self._pending: list[tuple[CandidateSpec, Any, list[float]]] = []
        self._in_flight: list[tuple[CandidateSpec, Any, list[float]]] = []
        self._observed: list[tuple[Any, list[float], float]] = []

        bounds = self._continuous_bounds()
        self._active_dims = [
            name for name in self._CONTINUOUS_DIMENSIONS if bounds[name][0] != bounds[name][1]
        ]
        self._fixed_values = {
            name: 0.5 * (bounds[name][0] + bounds[name][1])
            for name in self._CONTINUOUS_DIMENSIONS
            if bounds[name][0] == bounds[name][1]
        }
        lower = [bounds[name][0] for name in self._active_dims]
        upper = [bounds[name][1] for name in self._active_dims]
        spans = [upper[i] - lower[i] for i in range(len(lower))]
        max_span = max(spans) if spans else 1.0
        x0 = [0.5 * (lower[i] + upper[i]) for i in range(len(lower))]
        if warm_start:
            warm_vec = self._warm_start_vector(warm_start[0].candidate, bounds)
            if warm_vec is not None:
                # Seed the optimizer mean from the first warm start (knife-edge point).
                x0 = warm_vec
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
        self._popsize = int(opts.get("popsize", 4 * len(lower) + 3)) if self._active_dims else 1
        self._warm_starts: list[CandidateSpec] = [warm.candidate for warm in warm_start]
        self._es = cma.CMAEvolutionStrategy(x0, sigma0, opts) if self._active_dims else None

    def _warm_start_vector(
        self, candidate: CandidateSpec, bounds: dict[str, tuple[float, float]]
    ) -> list[float] | None:
        """Return the active-dimension vector for a warm start when fully in-bounds."""
        values = {
            "start.x": candidate.start.x,
            "start.y": candidate.start.y,
            "goal.x": candidate.goal.x,
            "goal.y": candidate.goal.y,
            "spawn_time_s": candidate.spawn_time_s,
            "pedestrian_speed_mps": candidate.pedestrian_speed_mps,
            "pedestrian_delay_s": candidate.pedestrian_delay_s,
        }
        vec: list[float] = []
        for name in self._active_dims:
            value = float(values[name])
            low, high = bounds[name]
            if value < low or value > high:
                return None
            vec.append(value)
        return vec

    def _continuous_bounds(self) -> dict[str, tuple[float, float]]:
        """Return the (min, max) bound for each continuous candidate dimension."""
        space = self._search_space
        ranges: dict[str, RangeConfig] = {
            "start.x": space.start_x,
            "start.y": space.start_y,
            "goal.x": space.goal_x,
            "goal.y": space.goal_y,
            "spawn_time_s": space.spawn_time_s,
            "pedestrian_speed_mps": space.pedestrian_speed_mps,
            "pedestrian_delay_s": space.pedestrian_delay_s,
        }
        return {name: (float(r.min), float(r.max)) for name, r in ranges.items()}

    def _range_for(self, name: str) -> RangeConfig:
        """Return the configured range for a continuous dimension name."""
        space = self._search_space
        ranges: dict[str, RangeConfig] = {
            "start.x": space.start_x,
            "start.y": space.start_y,
            "goal.x": space.goal_x,
            "goal.y": space.goal_y,
            "spawn_time_s": space.spawn_time_s,
            "pedestrian_speed_mps": space.pedestrian_speed_mps,
            "pedestrian_delay_s": space.pedestrian_delay_s,
        }
        return ranges[name]

    def _es_bounds(self) -> list[list[float]]:
        """Return the [lower, upper] CMA-ES bounds for the active dimensions."""
        bounds = self._continuous_bounds()
        return [
            [bounds[name][0] for name in self._active_dims],
            [bounds[name][1] for name in self._active_dims],
        ]

    def _make_candidate(self, vec: list[float]) -> CandidateSpec:
        """Build a candidate from an active-dimension vector plus fixed dims."""
        values = dict(self._fixed_values)
        for idx, name in enumerate(self._active_dims):
            values[name] = float(self._clamp_range(self._range_for(name), vec[idx]))
        return CandidateSpec(
            start=Pose2D(values["start.x"], values["start.y"]),
            goal=Pose2D(values["goal.x"], values["goal.y"]),
            spawn_time_s=values["spawn_time_s"],
            pedestrian_speed_mps=values["pedestrian_speed_mps"],
            pedestrian_delay_s=values["pedestrian_delay_s"],
            scenario_seed=self._seed_rng.randint(
                int(self._search_space.scenario_seed.min),
                int(self._search_space.scenario_seed.max),
            ),
        )

    def sample(self) -> CandidateSpec:
        """Return the next CMA-ES proposal within the configured bounds.

        CMA-ES optimizes a whole population per generation. The sequential
        runner asks for one candidate at a time, so the first ``sample`` of a
        generation draws a full population and queues the rest; subsequent
        ``sample`` calls drain the queue until the generation is exhausted.
        """
        if self._pending:
            candidate, es, vec = self._pending.pop(0)
            self._in_flight.append((candidate, es, vec))
            return candidate
        if self._warm_starts:
            return self._warm_starts.pop(0)
        if not self._active_dims:
            candidate = CandidateSpec(
                start=Pose2D(self._fixed_values["start.x"], self._fixed_values["start.y"]),
                goal=Pose2D(self._fixed_values["goal.x"], self._fixed_values["goal.y"]),
                spawn_time_s=self._fixed_values["spawn_time_s"],
                pedestrian_speed_mps=self._fixed_values["pedestrian_speed_mps"],
                pedestrian_delay_s=self._fixed_values["pedestrian_delay_s"],
                scenario_seed=self._seed_rng.randint(
                    int(self._search_space.scenario_seed.min),
                    int(self._search_space.scenario_seed.max),
                ),
            )
            self._pending.append((candidate, None, []))
            return candidate
        if self._es.stop():
            best = self._es.result.xbest if self._es.result.xbest is not None else self._es.mean
            self._es = self._cma.CMAEvolutionStrategy(
                best,
                max(1e-3, self._es.sigma * 0.5),
                {
                    "bounds": self._es_bounds(),
                    "seed": int(self._rng.randint(0, 2**31 - 1)),
                    "verbose": -9,
                },
            )
        population = self._es.ask(self._popsize)
        queued = [(self._make_candidate(list(vec)), self._es, list(vec)) for vec in population]
        self._pending.extend(queued)
        candidate, es, vec = self._pending.pop(0)
        self._in_flight.append((candidate, es, vec))
        return candidate

    @staticmethod
    def _clamp_range(bounds: RangeConfig, value: float) -> float:
        """Clamp a CMA-ES proposal into the inclusive range."""
        return min(float(bounds.max), max(float(bounds.min), float(value)))

    def observe(self, evaluation: CandidateEvaluation) -> None:
        """Return objective feedback for one observed candidate.

        Scores are buffered until the entire current generation has been
        observed, then fed to the optimizer in one ``tell`` call (CMA-ES
        requires a full population per generation). Degenerate fixed-range
        spaces skip optimization entirely.
        """
        if not self._active_dims:
            for index, (candidate, _es, _vec) in enumerate(self._pending):
                if candidate == evaluation.candidate:
                    self._pending.pop(index)
                    return
            return
        for index, (candidate, es, vec) in enumerate(self._in_flight):
            if candidate == evaluation.candidate:
                self._in_flight.pop(index)
                score = evaluation.objective_value
                value = float(score) if score is not None and math.isfinite(float(score)) else -1e9
                self._observed.append((es, list(vec), value))
                if not self._pending and not self._in_flight:
                    self._flush_generation()
                return

    def _flush_generation(self) -> None:
        """Feed the buffered generation to CMA-ES once it is fully observed."""
        generations = {}
        for es, vec, value in self._observed:
            generations.setdefault(es, ([], []))
            generations[es][0].append(vec)
            generations[es][1].append(value)
        for es, (vectors, values) in generations.items():
            es.tell(vectors, values)
        self._observed = []


def _suggest_float(trial: Any, name: str, bounds: RangeConfig) -> float:
    """Suggest a float while supporting degenerate fixed ranges."""
    if bounds.min == bounds.max:
        return float(bounds.min)
    return float(trial.suggest_float(name, float(bounds.min), float(bounds.max)))


def _suggest_int(trial: Any, name: str, bounds: RangeConfig) -> int:
    """Suggest an integer while supporting degenerate fixed ranges."""
    low = int(bounds.min)
    high = int(bounds.max)
    if low == high:
        return low
    return int(trial.suggest_int(name, low, high))


def _midpoint(bounds: RangeConfig) -> float:
    """Return the midpoint of a numeric range."""
    return float((bounds.min + bounds.max) / 2.0)


def _replace_candidate(
    candidate: CandidateSpec,
    *,
    start_x: float | None = None,
    start_y: float | None = None,
    goal_x: float | None = None,
    goal_y: float | None = None,
    spawn_time_s: float | None = None,
    pedestrian_speed_mps: float | None = None,
    pedestrian_delay_s: float | None = None,
) -> CandidateSpec:
    """Return a candidate with selected scalar fields replaced."""
    return CandidateSpec(
        start=Pose2D(
            candidate.start.x if start_x is None else start_x,
            candidate.start.y if start_y is None else start_y,
            candidate.start.theta,
        ),
        goal=Pose2D(
            candidate.goal.x if goal_x is None else goal_x,
            candidate.goal.y if goal_y is None else goal_y,
            candidate.goal.theta,
        ),
        spawn_time_s=candidate.spawn_time_s if spawn_time_s is None else spawn_time_s,
        pedestrian_speed_mps=candidate.pedestrian_speed_mps
        if pedestrian_speed_mps is None
        else pedestrian_speed_mps,
        pedestrian_delay_s=candidate.pedestrian_delay_s
        if pedestrian_delay_s is None
        else pedestrian_delay_s,
        scenario_seed=candidate.scenario_seed,
    )
