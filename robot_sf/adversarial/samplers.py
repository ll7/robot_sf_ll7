"""Candidate samplers for adversarial search."""

from __future__ import annotations

import math
from importlib import import_module
from random import Random
from typing import TYPE_CHECKING, Any, Protocol

from robot_sf.adversarial.config import CandidateSpec, Pose2D, RangeConfig, SearchSpaceConfig

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


class RandomCandidateSampler:
    """Dependency-light random-search sampler."""

    def __init__(self, search_space: SearchSpaceConfig, *, seed: int) -> None:
        """Initialize the sampler with a search space and deterministic seed."""
        self._search_space = search_space
        self._rng = Random(seed)

    def sample(self) -> CandidateSpec:
        """Return the next random candidate."""
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

    def __init__(self, search_space: SearchSpaceConfig, *, seed: int, step_fraction: float = 0.5):
        """Initialize a deterministic coordinate-refinement sampler."""
        if not math.isfinite(step_fraction) or step_fraction <= 0.0:
            raise ValueError("step_fraction must be finite and positive")
        self._search_space = search_space
        self._rng = Random(seed)
        self._step_fraction = float(step_fraction)
        self._best_candidate: CandidateSpec | None = None
        self._best_score: float | None = None
        self._iteration = 0

    def sample(self) -> CandidateSpec:
        """Return the next midpoint or coordinate-refinement candidate."""
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

    def __init__(self, search_space: SearchSpaceConfig, *, seed: int) -> None:
        """Initialize an Optuna study with deterministic sampler seed."""
        optuna = _import_optuna()
        sampler = optuna.samplers.TPESampler(seed=int(seed))
        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        self._trial_state = optuna.trial.TrialState
        self._search_space = search_space
        self._pending_trials: list[tuple[CandidateSpec, Any]] = []

    def sample(self) -> CandidateSpec:
        """Return the next optimizer proposal within the configured bounds."""
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
    try:
        return import_module("optuna")
    except ImportError as exc:
        raise RuntimeError(
            "OptunaCandidateSampler requires optuna. Install project dependencies with "
            "`uv sync --all-extras` before using the optimizer-backed adversarial sampler."
        ) from exc


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
