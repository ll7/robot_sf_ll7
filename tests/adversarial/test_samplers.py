"""Focused unit tests for ``robot_sf/adversarial/samplers.py``.

These tests cover the sampler module's public surface that is not exercised
elsewhere: the :func:`build_sampler` factory dispatch, constructor validation
for each sampler family, the dependency-light random and coordinate samplers,
and the optimizer-backed (Optuna / CMA-ES) samplers' feedback wiring.

Warm-start-first behaviour and end-to-end search integration are already
covered by ``test_adversarial_warm_start.py`` and ``test_adversarial_search.py``
and are intentionally not duplicated here.
"""

from __future__ import annotations

import sys

import pytest

from robot_sf.adversarial.certification import passed_status
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    Pose2D,
    SearchSpaceConfig,
    WarmStartCandidate,
)
from robot_sf.adversarial.samplers import (
    CmaEsCandidateSampler,
    CoordinateRefinementSampler,
    OptunaCandidateSampler,
    RandomCandidateSampler,
    build_sampler,
)


def _space() -> SearchSpaceConfig:
    """Build a search-space fixture with real continuous ranges.

    Continuous ranges are chosen so the midpoint is distinguishable from the
    bounds and so sampled candidates always satisfy the start/goal distance
    constraint.
    """
    return SearchSpaceConfig.from_mapping(
        {
            "variables": {
                "start_x": {"min": 0.0, "max": 2.0},
                "start_y": {"min": 2.0, "max": 4.0},
                "goal_x": {"min": 5.0, "max": 7.0},
                "goal_y": {"min": 2.0, "max": 4.0},
                "spawn_time_s": {"min": 0.0, "max": 0.5},
                "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                "scenario_seed": {"min": 7, "max": 9},
            },
            "constraints": {"min_start_goal_distance_m": 0.25},
        }
    )


def _fixed_space() -> SearchSpaceConfig:
    """Build a fully-degenerate search space (every continuous range pinned)."""
    return SearchSpaceConfig.from_mapping(
        {
            "variables": {
                "start_x": {"min": 1.0, "max": 1.0},
                "start_y": {"min": 2.0, "max": 2.0},
                "goal_x": {"min": 5.0, "max": 5.0},
                "goal_y": {"min": 2.0, "max": 2.0},
                "spawn_time_s": {"min": 0.0, "max": 0.0},
                "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                "scenario_seed": {"min": 7, "max": 9},
            },
            "constraints": {"min_start_goal_distance_m": 0.25},
        }
    )


def _candidate(
    *,
    start_x: float = 1.0,
    start_y: float = 2.0,
    goal_x: float = 5.0,
    goal_y: float = 2.0,
    seed: int = 7,
) -> CandidateSpec:
    """Build a candidate fixture with configurable pose coordinates."""
    return CandidateSpec(
        start=Pose2D(start_x, start_y),
        goal=Pose2D(goal_x, goal_y),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=seed,
    )


def _warm(seed: int = 7) -> WarmStartCandidate:
    """Build one knife-edge warm-start fixture inside the fixed space."""
    return WarmStartCandidate(
        candidate=_candidate(seed=seed),
        scenario="doorway",
        planner="goal",
        outcome_margin=0.1,
    )


def _evaluation(candidate: CandidateSpec, *, objective_value: float | None) -> CandidateEvaluation:
    """Build a minimal candidate evaluation with a configurable objective score."""
    return CandidateEvaluation(
        candidate=candidate,
        certification_status=passed_status(),
        objective_value=objective_value,
        failure_attribution=None,
        episode_record_path=None,
        trajectory_csv_path=None,
        scenario_yaml_path=None,
    )


# ---------------------------------------------------------------------------
# build_sampler factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["random", "Random", "RANDOM", " random "])
def test_build_sampler_returns_random_family_case_insensitive(name: str) -> None:
    """build_sampler should resolve the random family case-insensitively and trimmed."""
    sampler = build_sampler(name, _fixed_space(), seed=1)
    assert isinstance(sampler, RandomCandidateSampler)


@pytest.mark.parametrize("name", ["coordinate", "Coordinate", " COORDINATE "])
def test_build_sampler_returns_coordinate_family_case_insensitive(name: str) -> None:
    """build_sampler should resolve the coordinate family case-insensitively and trimmed."""
    sampler = build_sampler(name, _fixed_space(), seed=1)
    assert isinstance(sampler, CoordinateRefinementSampler)


def test_build_sampler_returns_optuna_family() -> None:
    """build_sampler should dispatch to the Optuna-backed sampler family."""
    pytest.importorskip("optuna")
    sampler = build_sampler("optuna", _fixed_space(), seed=1)
    assert isinstance(sampler, OptunaCandidateSampler)


def test_build_sampler_returns_cmaes_family() -> None:
    """build_sampler should dispatch to the CMA-ES-backed sampler family."""
    pytest.importorskip("cma")
    sampler = build_sampler("cmaes", _fixed_space(), seed=1)
    assert isinstance(sampler, CmaEsCandidateSampler)


def test_build_sampler_rejects_unknown_name() -> None:
    """build_sampler should fail closed with an actionable error for unknown names."""
    with pytest.raises(
        ValueError, match="sampler must be one of: random, coordinate, optuna, cmaes"
    ):
        build_sampler("bayesian", _fixed_space(), seed=1)


def test_build_sampler_propagates_warm_starts() -> None:
    """build_sampler should seed each family with the provided warm starts."""
    warm = _warm(seed=8)
    sampler = build_sampler("random", _fixed_space(), seed=1, warm_start=(warm,))
    assert isinstance(sampler, RandomCandidateSampler)
    assert sampler.sample() == warm.candidate


# ---------------------------------------------------------------------------
# RandomCandidateSampler
# ---------------------------------------------------------------------------


def test_random_sampler_is_deterministic_for_same_seed() -> None:
    """Two random samplers with the same seed should produce identical sequences."""
    space = _space()
    left = RandomCandidateSampler(space, seed=42)
    right = RandomCandidateSampler(space, seed=42)
    for _ in range(8):
        assert left.sample() == right.sample()


def test_random_sampler_produces_candidates_inside_the_space() -> None:
    """Random sampling (after warm starts) should stay within the search space."""
    space = _space()
    sampler = RandomCandidateSampler(space, seed=3)
    for _ in range(16):
        assert space.validate_candidate(sampler.sample()) == []


def test_random_sampler_different_seeds_diverge() -> None:
    """Random samplers with different seeds should not produce identical sequences."""
    space = _space()
    low = RandomCandidateSampler(space, seed=1)
    high = RandomCandidateSampler(space, seed=2)
    left_sequence = [low.sample() for _ in range(5)]
    right_sequence = [high.sample() for _ in range(5)]
    assert left_sequence != right_sequence


# ---------------------------------------------------------------------------
# CoordinateRefinementSampler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_step", [0.0, -0.1, float("nan"), float("inf")])
def test_coordinate_sampler_rejects_invalid_step_fraction(bad_step: float) -> None:
    """The coordinate sampler should reject non-positive or non-finite step fractions."""
    with pytest.raises(ValueError, match="step_fraction must be finite and positive"):
        CoordinateRefinementSampler(_space(), seed=1, step_fraction=bad_step)


def test_coordinate_sampler_first_candidate_is_search_space_midpoint() -> None:
    """The first coordinate candidate should be the configured search-space midpoint."""
    sampler = CoordinateRefinementSampler(_space(), seed=5)
    first = sampler.sample()
    assert first.start.x == pytest.approx(1.0)
    assert first.start.y == pytest.approx(3.0)
    assert first.goal.x == pytest.approx(6.0)
    assert first.goal.y == pytest.approx(3.0)
    assert first.spawn_time_s == pytest.approx(0.25)
    assert first.pedestrian_speed_mps == pytest.approx(1.0)
    assert first.pedestrian_delay_s == pytest.approx(0.0)
    assert 7 <= first.scenario_seed <= 9
    assert _space().validate_candidate(first) == []


def test_coordinate_sampler_ignores_none_objective_when_updating_incumbent() -> None:
    """A None objective value must not install a local-search incumbent."""
    sampler = CoordinateRefinementSampler(_space(), seed=5)
    sampler.sample()  # midpoint
    sampler.observe(_evaluation(_candidate(start_x=1.8), objective_value=None))
    assert sampler._best_candidate is None
    # No incumbent installed: the next sample is still the midpoint, not a perturbation.
    assert sampler.sample().start.x == pytest.approx(1.0)


def test_coordinate_sampler_ignores_non_finite_objective_when_updating_incumbent() -> None:
    """A non-finite objective value must not install a local-search incumbent."""
    sampler = CoordinateRefinementSampler(_space(), seed=5)
    sampler.sample()  # midpoint
    sampler.observe(_evaluation(_candidate(start_x=1.8), objective_value=float("nan")))
    assert sampler._best_candidate is None
    assert sampler.sample().start.x == pytest.approx(1.0)


def test_coordinate_sampler_keeps_best_incumbent_on_lower_scores() -> None:
    """observe should only replace the incumbent when the score strictly improves."""
    sampler = CoordinateRefinementSampler(_space(), seed=5)
    sampler.sample()  # midpoint
    high = _candidate(start_x=1.0)
    low = _candidate(start_x=1.2)
    sampler.observe(_evaluation(high, objective_value=0.9))
    sampler.observe(_evaluation(low, objective_value=0.1))
    assert sampler._best_score == pytest.approx(0.9)
    assert sampler._best_candidate == high


def test_coordinate_sampler_clamps_perturbations_within_bounds() -> None:
    """Coordinate perturbations should be clamped into the configured range."""
    space = _space()  # start_x range is [0.0, 2.0]
    sampler = CoordinateRefinementSampler(space, seed=5, step_fraction=0.5)
    sampler.sample()  # midpoint -> iteration advances to 1
    edge = _candidate(start_x=2.0, start_y=3.0, goal_x=6.0, goal_y=3.0, seed=7)
    sampler.observe(_evaluation(edge, objective_value=0.5))
    # First perturbation dimension is "start.x" with direction +1: 2.0 -> clamped to 2.0.
    proposed = sampler.sample()
    assert proposed.start.x == pytest.approx(2.0)
    assert space.validate_candidate(proposed) == []


# ---------------------------------------------------------------------------
# OptunaCandidateSampler
# ---------------------------------------------------------------------------


def test_optuna_sampler_marks_unscored_trial_as_failed() -> None:
    """A None objective should tell Optuna the trial failed instead of scoring it."""
    optuna = pytest.importorskip("optuna")
    sampler = OptunaCandidateSampler(_fixed_space(), seed=11)
    first = sampler.sample()
    sampler.observe(_evaluation(first, objective_value=None))
    trials = sampler._study.trials
    assert len(trials) == 1
    assert trials[0].state == optuna.trial.TrialState.FAIL
    assert trials[0].value is None


def test_optuna_sampler_marks_non_finite_trial_as_failed() -> None:
    """A non-finite objective should tell Optuna the trial failed."""
    optuna = pytest.importorskip("optuna")
    sampler = OptunaCandidateSampler(_fixed_space(), seed=11)
    first = sampler.sample()
    sampler.observe(_evaluation(first, objective_value=float("inf")))
    assert sampler._study.trials[0].state == optuna.trial.TrialState.FAIL


def test_optuna_sampler_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Optuna sampler should fail with an actionable dependency error."""
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(RuntimeError, match="OptunaCandidateSampler requires optuna"):
        OptunaCandidateSampler(_fixed_space(), seed=7)


# ---------------------------------------------------------------------------
# CmaEsCandidateSampler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_sigma", [0.0, -0.1, float("nan"), float("inf")])
def test_cmaes_sampler_rejects_invalid_sigma_fraction(bad_sigma: float) -> None:
    """The CMA-ES sampler should reject non-positive or non-finite sigma fractions."""
    pytest.importorskip("cma")
    with pytest.raises(ValueError, match="sigma_fraction must be finite and positive"):
        CmaEsCandidateSampler(_space(), seed=1, sigma_fraction=bad_sigma)


def test_cmaes_sampler_rejects_popsize_below_one() -> None:
    """The CMA-ES sampler should reject populations smaller than one."""
    pytest.importorskip("cma")
    with pytest.raises(ValueError, match="popsize must be >= 1"):
        CmaEsCandidateSampler(_space(), seed=1, popsize=0)


def test_cmaes_sampler_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CMA-ES sampler should fail with an actionable dependency error."""
    monkeypatch.setitem(sys.modules, "cma", None)
    with pytest.raises(RuntimeError, match="CmaEsCandidateSampler requires cma"):
        CmaEsCandidateSampler(_fixed_space(), seed=7)


def test_cmaes_sampler_runs_without_optimizer_on_degenerate_space() -> None:
    """A fully-degenerate space should sample without invoking the optimizer."""
    pytest.importorskip("cma")
    space = _fixed_space()  # every continuous dimension is pinned
    sampler = CmaEsCandidateSampler(space, seed=7)
    assert sampler._active_dims == []
    first = sampler.sample()
    assert space.validate_candidate(first) == []
    # A full sample/observe cycle should drain the degenerate buffer without error.
    sampler.observe(_evaluation(first, objective_value=0.3))
    assert sampler._pending == []


def test_cmaes_sampler_candidates_stay_within_bounds() -> None:
    """CMA-ES proposals should be clamped into the configured search space."""
    pytest.importorskip("cma")
    space = _space()
    sampler = CmaEsCandidateSampler(space, seed=7, popsize=6)
    for _ in range(sampler._popsize):
        candidate = sampler.sample()
        assert space.validate_candidate(candidate) == []
        sampler.observe(_evaluation(candidate, objective_value=0.0))


def test_cmaes_sampler_flushes_generation_after_full_population_observed() -> None:
    """CMA-ES should buffer scores until the whole generation is observed, then flush."""
    pytest.importorskip("cma")
    sampler = CmaEsCandidateSampler(_space(), seed=7, popsize=5)
    population = [sampler.sample() for _ in range(sampler._popsize)]
    assert len(population) == sampler._popsize
    # Observing all but the last candidate keeps the generation buffered.
    for candidate in population[:-1]:
        sampler.observe(_evaluation(candidate, objective_value=0.1))
    assert sampler._observed
    # The final observation flushes the buffered generation to the optimizer.
    sampler.observe(_evaluation(population[-1], objective_value=0.9))
    assert sampler._observed == []
    assert sampler._in_flight == []
